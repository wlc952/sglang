import os
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q, k, v, g, beta, o, h0, ht, cu_seqlens, scale, T,
    B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr, IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * H + i_h) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)
        else:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h *= tl.exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += H * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# ── Test framework ──────────────────────────────────────────────────────────

def _create_inputs(device, dtype, *, num_seqs=4, seqlen=1024, H=16, HV=32, K_dim=128, V_dim=128, use_initial_state=True):
    """Construct inputs for fused_recurrent_gated_delta_rule_fwd_kernel (extend scenario).

    cu_seqlens splits total_tokens = num_seqs * seqlen into num_seqs equal segments.
    """
    total_tokens = num_seqs * seqlen

    # CPU-first creation for cmodel compatibility
    q = torch.randn(1, total_tokens, H, K_dim, device="cpu", dtype=dtype)
    k = torch.randn(1, total_tokens, H, K_dim, device="cpu", dtype=dtype)
    v = torch.randn(1, total_tokens, HV, V_dim, device="cpu", dtype=dtype)
    g = torch.randn(1, total_tokens, HV, device="cpu", dtype=torch.float32)
    beta = torch.randn(1, total_tokens, HV, device="cpu", dtype=torch.float32)

    cu_seqlens = torch.tensor(
        [i * seqlen for i in range(num_seqs + 1)],
        dtype=torch.int32, device="cpu"
    )

    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(num_seqs, HV, V_dim, K_dim, device="cpu", dtype=torch.float32)

    # Move to device
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    g = g.to(device)
    beta = beta.to(device)
    cu_seqlens = cu_seqlens.to(device)
    if initial_state is not None:
        initial_state = initial_state.to(device)

    return {
        "q": q, "k": k, "v": v, "g": g, "beta": beta,
        "cu_seqlens": cu_seqlens, "initial_state": initial_state,
        "num_seqs": num_seqs, "seqlen": seqlen,
        "H": H, "HV": HV, "K_dim": K_dim, "V_dim": V_dim,
        "total_tokens": total_tokens,
    }


def _launch_kernel_impl(inputs):
    """Launch fused_recurrent_gated_delta_rule_fwd_kernel with env-var config."""
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    g = inputs["g"]
    beta = inputs["beta"]
    cu_seqlens = inputs["cu_seqlens"]
    initial_state = inputs["initial_state"]

    T = inputs["total_tokens"]
    H = inputs["H"]
    HV = inputs["HV"]
    K_dim = inputs["K_dim"]
    V_dim = inputs["V_dim"]
    N = inputs["num_seqs"]

    BK = triton.next_power_of_2(K_dim)
    BV = min(triton.next_power_of_2(V_dim), 32)
    NK = triton.cdiv(K_dim, BK)
    NV = triton.cdiv(V_dim, BV)

    num_warps = int(os.getenv("NUM_WARPS", "1"))
    num_stages = int(os.getenv("NUM_STAGES", "3"))

    o = q.new_empty(NK, *v.shape)
    final_state = None
    if initial_state is not None:
        final_state = q.new_empty(N, HV, V_dim, K_dim, dtype=torch.float32)

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q, k=k, v=v, g=g, beta=beta, o=o, h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, scale=K_dim ** -0.5, T=T,
        B=1, H=H, HV=HV, K=K_dim, V=V_dim, BK=BK, BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_BETA_HEADWISE=False,  # beta.ndim != v.ndim in extend
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_VARLEN=True,
        IS_KDA=False,
        num_warps=num_warps, num_stages=num_stages,
    )
    o = o.squeeze(0)

    inputs["output"] = o
    inputs["final_state"] = final_state


def _golden_compute(inputs):
    """PyTorch reference implementation for correctness check.

    Mirrors fused_recurrent_gated_delta_rule_fwd_kernel exactly (float32).
    h layout: (V, K)  — same as kernel b_h[BV, BK].
    """
    q  = inputs["q"].float()           # (1, T, H, K)
    k  = inputs["k"].float()           # (1, T, H, K)
    v  = inputs["v"].float()           # (1, T, HV, V)
    g  = inputs["g"].float()           # (1, T, HV)
    beta = inputs["beta"].float()      # (1, T, HV)  — IS_BETA_HEADWISE=False
    cu_seqlens  = inputs["cu_seqlens"]
    initial_state = inputs["initial_state"]
    H    = inputs["H"]
    HV   = inputs["HV"]
    K_dim = inputs["K_dim"]
    V_dim = inputs["V_dim"]
    N    = inputs["num_seqs"]
    scale = K_dim ** -0.5

    total_tokens = q.shape[1]
    o_ref = torch.zeros(1, total_tokens, HV, V_dim, dtype=torch.float32, device=q.device)

    for n in range(N):
        bos = int(cu_seqlens[n].item())
        eos = int(cu_seqlens[n + 1].item())

        for ihv in range(HV):
            ih = ihv // (HV // H)

            # h: (V, K) — mirrors kernel b_h[BV, BK]
            if initial_state is not None:
                h = initial_state[n, ihv].float().clone()  # (V, K)
            else:
                h = torch.zeros(V_dim, K_dim, dtype=torch.float32, device=q.device)

            for t in range(bos, eos):
                b_q = q[0, t, ih]        # (K,)
                b_k = k[0, t, ih]        # (K,)
                b_v = v[0, t, ihv]       # (V,)
                b_g = g[0, t, ihv]       # scalar
                b_beta = beta[0, t, ihv] # scalar

                # USE_QK_L2NORM_IN_KERNEL
                b_q = b_q / (b_q.dot(b_q) + 1e-6).sqrt()
                b_k = b_k / (b_k.dot(b_k) + 1e-6).sqrt()
                b_q = b_q * scale

                # gate: mirrors `b_h *= exp(b_g)`
                h = h * torch.exp(b_g)

                # delta correction: mirrors `b_v -= tl.sum(b_h * b_k[None,:], 1)`
                # h @ b_k  =>  (V,K) @ (K,) = (V,)
                b_v = b_v - h.mv(b_k)

                # beta scale: mirrors `b_v *= b_beta`
                b_v = b_v * b_beta

                # rank-1 update: mirrors `b_h += b_v[:,None] * b_k[None,:]`
                h = h + b_v.unsqueeze(1) * b_k.unsqueeze(0)   # (V,1)*(1,K) = (V,K)

                # output: mirrors `b_o = tl.sum(b_h * b_q[None,:], 1)`
                o_ref[0, t, ihv] = h.mv(b_q)                  # (V,K) @ (K,) = (V,)

    return o_ref


def run_accuracy(*, device=None, atol=0.01, rtol=0.01):
    torch.manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    # Read problem size from env vars
    num_seqs = int(os.getenv("ACC_NUM_SEQS", "4"))
    seqlen = int(os.getenv("ACC_SEQLEN", "1024"))
    H = int(os.getenv("ACC_H", "16"))
    HV = int(os.getenv("ACC_HV", "32"))
    K_dim = int(os.getenv("ACC_K", "128"))
    V_dim = int(os.getenv("ACC_V", "128"))

    inputs = _create_inputs(device, dtype, num_seqs=num_seqs, seqlen=seqlen,
                            H=H, HV=HV, K_dim=K_dim, V_dim=V_dim, use_initial_state=True)
    _launch_kernel_impl(inputs)
    if device.type in ("cuda", "supa"):
        torch.cuda.synchronize()

    kernel_output = inputs["output"]
    golden = _golden_compute(inputs)

    # Debug info
    diff = (kernel_output.float() - golden).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    has_nan = torch.isnan(kernel_output).any().item()

    match = torch.allclose(kernel_output.float(), golden, atol=atol, rtol=rtol)

    print(f"[accuracy] shape: kernel={kernel_output.shape}, golden={golden.shape}")
    print(f"[accuracy] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, nan={has_nan}")
    print(f"[accuracy] PASS={match}")


def run_perflog(*, device=None):
    """Single-run for cmodel perflog collection. No warmup/repeat."""
    torch.manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    # Read problem size from env vars
    num_seqs = int(os.getenv("PERF_NUM_SEQS", "4"))
    seqlen = int(os.getenv("PERF_SEQLEN", "1024"))
    H = int(os.getenv("PERF_H", "16"))
    HV = int(os.getenv("PERF_HV", "32"))
    K_dim = int(os.getenv("PERF_K", "128"))
    V_dim = int(os.getenv("PERF_V", "128"))

    inputs = _create_inputs(device, dtype, num_seqs=num_seqs, seqlen=seqlen,
                            H=H, HV=HV, K_dim=K_dim, V_dim=V_dim, use_initial_state=True)
    _launch_kernel_impl(inputs)
    if device.type in ("cuda", "supa"):
        torch.cuda.synchronize()
    print(f"[perflog] single-run done")


def _bench_with_events(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def run_bench(*, device=None):
    import json
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]
    results = []

    for seqlen in seq_lens:
        num_seqs = 4
        inputs = _create_inputs(device, dtype, num_seqs=num_seqs, seqlen=seqlen)

        def bench_fn():
            _launch_kernel_impl(inputs)
            if device.type in ("cuda", "supa"):
                torch.cuda.synchronize()

        ms = _bench_with_events(bench_fn, warmup=10, iters=100)
        results.append({"seqlen": seqlen, "num_seqs": num_seqs, "latency_ms": ms})
        print(f"seqlen={seqlen:>5}, num_seqs={num_seqs}, latency={ms:.3f} ms")

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    mode = os.getenv("MODE", "accuracy").lower()
    if mode == "perf":
        run_perflog()
    elif mode == "bench":
        run_bench()
    else:
        run_accuracy()