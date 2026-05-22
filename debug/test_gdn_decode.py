import os
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b, o,
    h0_source, h0_indices, cu_seqlens,
    intermediate_states_buffer, intermediate_state_indices, cache_steps,
    retrieve_parent_token_ptr, stride_retrieve_parent_token_seq: tl.constexpr, stride_retrieve_parent_token_token: tl.constexpr,
    scale, T, stride_q, stride_k, stride_v, stride_b,
    NP2_T: tl.constexpr, B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr, IS_VARLEN: tl.constexpr, IS_KDA: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr = False, CACHE_INTERMEDIATE_STATES: tl.constexpr = False, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
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

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = retrieve_parent_token_ptr + (i_n * stride_retrieve_parent_token_seq) + token_indices * stride_retrieve_parent_token_token
        parent_idx_tokens = tl.load(retrieve_parent_token_base, mask=mask_retrieve, other=0)

    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)

    step_idx = 0
    for _ in range(0, T):
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(tl.where(token_indices == step_idx, parent_idx_tokens, 0))
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = intermediate_states_buffer + cache_idx * cache_steps * HV * K * V + step_offset + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(beta_x <= softplus_threshold, (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)), x)
        b_g = -tl.exp(b_A_log) * softplus_x

        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = intermediate_states_buffer + cache_idx * cache_steps * HV * K * V + step_offset + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        if IS_KDA:
            p_a += HV * K
        else:
            p_a += HV

    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


# ── Test framework ──────────────────────────────────────────────────────────

def _create_inputs(device, dtype, *, num_seqs=128, seqlen=1, H=16, HV=32, K_dim=128, V_dim=128,
                   use_initial_state=True, softplus_beta=1.0, softplus_threshold=20.0):
    """Construct inputs for fused_sigmoid_gating_delta_rule_update_kernel (decode scenario).

    cu_seqlens splits total_tokens = num_seqs * seqlen into num_seqs equal segments.
    State layout: (K, V) — note transposed vs extend kernel's (V, K).
    """
    total_tokens = num_seqs * seqlen

    q = torch.randn(1, total_tokens, H, K_dim, device="cpu", dtype=dtype)
    k = torch.randn(1, total_tokens, H, K_dim, device="cpu", dtype=dtype)
    v = torch.randn(1, total_tokens, HV, V_dim, device="cpu", dtype=dtype)
    a = torch.randn(1, total_tokens, HV, device="cpu", dtype=torch.float32)
    b = torch.randn(1, total_tokens, HV, device="cpu", dtype=torch.float32)
    A_log = torch.randn(HV, device="cpu", dtype=torch.float32)
    dt_bias = torch.randn(HV, device="cpu", dtype=torch.float32)

    cu_seqlens = torch.tensor(
        [i * seqlen for i in range(num_seqs + 1)],
        dtype=torch.int32, device="cpu"
    )

    # h0_indices: direct mapping seq_idx -> state_idx
    h0_indices = torch.arange(num_seqs, dtype=torch.int32, device="cpu")

    initial_state = None
    if use_initial_state:
        # State layout for this kernel: (N_states, HV, K, V) — note (K, V) not (V, K)
        initial_state = torch.randn(num_seqs, HV, K_dim, V_dim, device="cpu", dtype=torch.float32)

    # Move to device
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    a = a.to(device)
    b = b.to(device)
    A_log = A_log.to(device)
    dt_bias = dt_bias.to(device)
    cu_seqlens = cu_seqlens.to(device)
    h0_indices = h0_indices.to(device)
    if initial_state is not None:
        initial_state = initial_state.to(device)

    return {
        "q": q, "k": k, "v": v, "a": a, "b": b,
        "A_log": A_log, "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens, "h0_indices": h0_indices,
        "initial_state": initial_state,
        "num_seqs": num_seqs, "seqlen": seqlen,
        "H": H, "HV": HV, "K_dim": K_dim, "V_dim": V_dim,
        "total_tokens": total_tokens,
        "softplus_beta": softplus_beta, "softplus_threshold": softplus_threshold,
    }


def _launch_kernel_impl(inputs):
    """Launch fused_sigmoid_gating_delta_rule_update_kernel with env-var config."""
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    a = inputs["a"]
    b = inputs["b"]
    A_log = inputs["A_log"]
    dt_bias = inputs["dt_bias"]
    cu_seqlens = inputs["cu_seqlens"]
    h0_indices = inputs["h0_indices"]
    initial_state = inputs["initial_state"]

    T = inputs["total_tokens"]
    H = inputs["H"]
    HV = inputs["HV"]
    K_dim = inputs["K_dim"]
    V_dim = inputs["V_dim"]
    N = inputs["num_seqs"]
    softplus_beta = inputs["softplus_beta"]
    softplus_threshold = inputs["softplus_threshold"]
    scale = K_dim ** -0.5

    BK = triton.next_power_of_2(K_dim)
    BV = min(triton.next_power_of_2(V_dim), 32)
    NK = triton.cdiv(K_dim, BK)
    NV = triton.cdiv(V_dim, BV)

    num_warps = int(os.getenv("NUM_WARPS", "1"))
    num_stages = int(os.getenv("NUM_STAGES", "3"))

    # Clone state so kernel in-place update doesn't corrupt input for golden
    h0_clone = initial_state.clone() if initial_state is not None else None

    o = q.new_empty(NK, *v.shape)
    NP2_T = triton.next_power_of_2(T)

    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        q=q, k=k, v=v, b=b, o=o,
        h0_source=h0_clone, h0_indices=h0_indices, cu_seqlens=cu_seqlens,
        intermediate_states_buffer=None, intermediate_state_indices=None,
        cache_steps=0, retrieve_parent_token_ptr=None,
        stride_retrieve_parent_token_seq=0, stride_retrieve_parent_token_token=0,
        scale=scale, T=T,
        stride_q=stride_q, stride_k=stride_k, stride_v=stride_v, stride_b=stride_b,
        NP2_T=NP2_T, B=1, H=H, HV=HV, K=K_dim, V=V_dim, BK=BK, BV=BV,
        USE_INITIAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_VARLEN=True,
        IS_KDA=False,
        DISABLE_STATE_UPDATE=False,
        CACHE_INTERMEDIATE_STATES=False,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False,
        num_warps=num_warps, num_stages=num_stages,
    )
    o = o.squeeze(0)

    inputs["output"] = o
    inputs["final_state"] = h0_clone


def _golden_compute(inputs):
    """PyTorch reference implementation for correctness check.

    Mirrors fused_sigmoid_gating_delta_rule_update_kernel exactly (float32).
    h layout: (K, V) — same as kernel b_h[BK, BV].
    """
    q = inputs["q"].float()
    k = inputs["k"].float()
    v = inputs["v"].float()
    a = inputs["a"].float()
    b = inputs["b"].float()
    A_log = inputs["A_log"].float()
    dt_bias = inputs["dt_bias"].float()
    cu_seqlens = inputs["cu_seqlens"]
    initial_state = inputs["initial_state"]
    h0_indices = inputs["h0_indices"]
    H = inputs["H"]
    HV = inputs["HV"]
    K_dim = inputs["K_dim"]
    V_dim = inputs["V_dim"]
    N = inputs["num_seqs"]
    scale = K_dim ** -0.5
    sp_beta = inputs["softplus_beta"]
    sp_threshold = inputs["softplus_threshold"]

    total_tokens = q.shape[1]
    o_ref = torch.zeros(1, total_tokens, HV, V_dim, dtype=torch.float32, device=q.device)
    final_state = initial_state.float().clone() if initial_state is not None else None

    for n in range(N):
        bos = int(cu_seqlens[n].item())
        eos = int(cu_seqlens[n + 1].item())
        state_idx = int(h0_indices[n].item())

        for ihv in range(HV):
            ih = ihv // (HV // H)

            # h: (K, V) — mirrors kernel b_h[BK, BV]
            if initial_state is not None and state_idx >= 0:
                h = initial_state[state_idx, ihv].float().clone()  # (K, V)
            else:
                h = torch.zeros(K_dim, V_dim, dtype=torch.float32, device=q.device)

            b_A_log = A_log[ihv]
            b_dt_bias = dt_bias[ihv]  # scalar (IS_KDA=False)

            for t in range(bos, eos):
                b_q = q[0, t, ih]       # (K,)
                b_k = k[0, t, ih]       # (K,)
                b_v = v[0, t, ihv]      # (V,)
                b_b = b[0, t, ihv]      # scalar
                b_a = a[0, t, ihv]      # scalar (IS_KDA=False)

                # Gate: softplus then exp decay
                x = b_a + b_dt_bias
                beta_x = sp_beta * x
                if beta_x <= sp_threshold:
                    softplus_x = (1.0 / sp_beta) * torch.log(1.0 + torch.exp(beta_x))
                else:
                    softplus_x = x
                b_g = -torch.exp(b_A_log) * softplus_x

                # Sigmoid beta
                b_beta = 1.0 / (1.0 + torch.exp(-b_b))

                # L2 norm
                b_q = b_q / (b_q.dot(b_q) + 1e-6).sqrt()
                b_k = b_k / (b_k.dot(b_k) + 1e-6).sqrt()
                b_q = b_q * scale

                # Gate state: IS_KDA=False => scalar gate
                h = h * torch.exp(b_g)

                # Delta correction: b_v -= sum(b_h * b_k[:,None], 0)
                # h is (K,V), b_k is (K,) => h.T @ b_k = (V,K) @ (K,) but kernel does sum(b_h * b_k[:,None], 0)
                # which is (K,V) with b_k[:,None] broadcast => sum along dim0 => (V,)
                b_v = b_v - (h * b_k.unsqueeze(1)).sum(0)  # (V,)

                # Beta scale
                b_v = b_v * b_beta

                # Rank-1 update: b_h += b_k[:,None] * b_v[None,:]
                h = h + b_k.unsqueeze(1) * b_v.unsqueeze(0)  # (K,1)*(1,V) = (K,V)

                # Output: b_o = sum(b_h * b_q[:,None], 0)
                o_ref[0, t, ihv] = (h * b_q.unsqueeze(1)).sum(0)  # (V,)

            # Write back final state
            if final_state is not None and state_idx >= 0:
                final_state[state_idx, ihv] = h

    return o_ref, final_state


def run_accuracy(*, device=None, atol=0.01, rtol=0.01):
    torch.manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    num_seqs = int(os.getenv("ACC_NUM_SEQS", "4"))
    seqlen = int(os.getenv("ACC_SEQLEN", "1"))
    H = int(os.getenv("ACC_H", "16"))
    HV = int(os.getenv("ACC_HV", "32"))
    K_dim = int(os.getenv("ACC_K", "128"))
    V_dim = int(os.getenv("ACC_V", "128"))

    inputs = _create_inputs(device, dtype, num_seqs=num_seqs, seqlen=seqlen,
                            H=H, HV=HV, K_dim=K_dim, V_dim=V_dim, use_initial_state=True)

    # Save original state for golden (kernel does in-place update)
    orig_state = inputs["initial_state"].clone()

    _launch_kernel_impl(inputs)
    if device.type in ("cuda", "supa"):
        torch.cuda.synchronize()

    kernel_output = inputs["output"]
    kernel_final_state = inputs["final_state"]

    # Restore state for golden compute
    inputs["initial_state"] = orig_state
    golden_output, golden_final_state = _golden_compute(inputs)

    # Output accuracy
    diff = (kernel_output.float() - golden_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    has_nan = torch.isnan(kernel_output).any().item()
    match_output = torch.allclose(kernel_output.float(), golden_output, atol=atol, rtol=rtol)

    print(f"[accuracy/output] shape: kernel={kernel_output.shape}, golden={golden_output.shape}")
    print(f"[accuracy/output] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, nan={has_nan}")
    print(f"[accuracy/output] PASS={match_output}")

    # State accuracy
    if kernel_final_state is not None and golden_final_state is not None:
        state_diff = (kernel_final_state.float() - golden_final_state).abs()
        state_max_diff = state_diff.max().item()
        state_mean_diff = state_diff.mean().item()
        match_state = torch.allclose(kernel_final_state.float(), golden_final_state, atol=atol, rtol=rtol)
        print(f"[accuracy/state] max_diff={state_max_diff:.6f}, mean_diff={state_mean_diff:.6f}")
        print(f"[accuracy/state] PASS={match_state}")
    else:
        match_state = True

    print(f"[accuracy] ALL_PASS={match_output and match_state}")


def run_perflog(*, device=None):
    """Single-run for cmodel perflog collection. No warmup/repeat."""
    torch.manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    num_seqs = int(os.getenv("PERF_NUM_SEQS", "128"))
    seqlen = int(os.getenv("PERF_SEQLEN", "1"))
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

    batch_sizes = [1, 4, 16, 32, 64, 128, 256, 512]
    results = []

    for num_seqs in batch_sizes:
        inputs = _create_inputs(device, dtype, num_seqs=num_seqs, seqlen=1)

        def bench_fn():
            _launch_kernel_impl(inputs)
            if device.type in ("cuda", "supa"):
                torch.cuda.synchronize()

        ms = _bench_with_events(bench_fn, warmup=10, iters=100)
        results.append({"num_seqs": num_seqs, "seqlen": 1, "latency_ms": ms})
        print(f"num_seqs={num_seqs:>4}, seqlen=1, latency={ms:.3f} ms")

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    mode = os.getenv("MODE", "perf").lower()
    if mode == "perf":
        run_perflog()
    elif mode == "bench":
        run_bench()
    else:
        run_accuracy()
