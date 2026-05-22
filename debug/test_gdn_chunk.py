"""
Self-contained test for chunk_gated_delta_rule pipeline (prefill/extend).

Kernels copied from sglang/srt/layers/attention/fla/:
  - cumsum.py        → chunk_local_cumsum_scalar_kernel
  - chunk_fwd.py     → chunk_gated_delta_rule_fwd_kkt_solve_kernel
  - wy_fast.py       → recompute_w_u_fwd_kernel
  - chunk_delta_h.py → chunk_gated_delta_rule_fwd_kernel_h_blockdim64
  - chunk_o.py       → chunk_fwd_kernel_o
  - l2norm.py        → l2norm_fwd_kernel / l2norm_fwd_kernel1

Golden reference: pure PyTorch token-by-token recurrent.
"""

import os
import torch
import triton
import triton.language as tl

_is_tf32_supported = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8
)
if _is_tf32_supported:
    _MERGE_DOT_PRECISION = tl.constexpr("tf32")
else:
    _MERGE_DOT_PRECISION = tl.constexpr("ieee")


# ════════════════════════════════════════════════════════════════════════════════
# Triton JIT helpers
# ════════════════════════════════════════════════════════════════════════════════

@triton.jit
def exp(x):
    return tl.exp(x)

@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


# ════════════════════════════════════════════════════════════════════════════════
# Python helpers (no framework dependency)
# ════════════════════════════════════════════════════════════════════════════════

def prepare_chunk_indices(cu_seqlens, chunk_size):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens, chunk_size):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(lens, chunk_size)]
    ).cumsum(-1)


# ════════════════════════════════════════════════════════════════════════════════
# Kernel 1: L2 Norm
# ════════════════════════════════════════════════════════════════════════════════

@triton.jit
def l2norm_fwd_kernel1(x, y, D, BD: tl.constexpr, eps):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit
def l2norm_fwd_kernel(x, y, eps, NB: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(x, eps=1e-6):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    y = torch.empty_like(x)
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D <= 512:
        NB = triton.cdiv(T, 2048)
        grid = (triton.cdiv(T, 16),)
        l2norm_fwd_kernel[grid](x, y, eps, NB=NB, T=T, D=D, BD=BD, BT=16, num_warps=8, num_stages=3)
    else:
        l2norm_fwd_kernel1[(T,)](x, y, eps=eps, D=D, BD=BD, num_warps=8, num_stages=3)
    return y.view(x_shape_og)


# ════════════════════════════════════════════════════════════════════════════════
# Kernel 2: chunk_local_cumsum (scalar gate)
# ════════════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s, o, scale, cu_seqlens, chunk_indices, T,
    B: tl.constexpr, H: tl.constexpr, BT: tl.constexpr,
    REVERSE: tl.constexpr, HAS_SCALE: tl.constexpr, IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(g, chunk_size, cu_seqlens=None):
    B, T, H = g.shape
    BT = chunk_size
    chunk_indices_t = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices_t)
    g_out = torch.empty_like(g, dtype=torch.float32)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g, o=g_out, scale=None, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices_t,
        T=T, B=B, H=H, BT=BT, HEAD_FIRST=False, REVERSE=False, HAS_SCALE=False,
        IS_VARLEN=cu_seqlens is not None, num_warps=8, num_stages=3,
    )
    return g_out


# ════════════════════════════════════════════════════════════════════════════════
# Kernel 3: fused kkt + solve_tril (intra-chunk)
# ════════════════════════════════════════════════════════════════════════════════

@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "Hg", "K", "BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k, g, beta, A, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * Hg + i_h // (H // Hg)) * K
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    p_b0 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    if USE_G:
        p_g0 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        p_g3 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)

    # Step 1: compute 10 lower-triangular blocks of K @ K^T
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))
        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))
            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))
                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    # Step 2: gate + beta scaling
    if USE_G:
        b_A00 *= safe_exp(b_g0[:, None] - b_g0[None, :])
        b_A11 *= safe_exp(b_g1[:, None] - b_g1[None, :])
        b_A22 *= safe_exp(b_g2[:, None] - b_g2[None, :])
        b_A33 *= safe_exp(b_g3[:, None] - b_g3[None, :])
        b_A10 *= safe_exp(b_g1[:, None] - b_g0[None, :])
        b_A20 *= safe_exp(b_g2[:, None] - b_g0[None, :])
        b_A21 *= safe_exp(b_g2[:, None] - b_g1[None, :])
        b_A30 *= safe_exp(b_g3[:, None] - b_g0[None, :])
        b_A31 *= safe_exp(b_g3[:, None] - b_g1[None, :])
        b_A32 *= safe_exp(b_g3[:, None] - b_g2[None, :])

    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_A00 = tl.where(m_d & (m_tc0[:, None] & m_tc0[None, :]), b_A00, 0.0) * b_b0[:, None]
    b_A11 = tl.where(m_d & (m_tc1[:, None] & m_tc1[None, :]), b_A11, 0.0) * b_b1[:, None]
    b_A22 = tl.where(m_d & (m_tc2[:, None] & m_tc2[None, :]), b_A22, 0.0) * b_b2[:, None]
    b_A33 = tl.where(m_d & (m_tc3[:, None] & m_tc3[None, :]), b_A33, 0.0) * b_b3[:, None]
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    # Step 3: forward substitution on diagonal blocks
    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    # Step 4: block merge
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=_MERGE_DOT_PRECISION),
        b_Ai00, input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=_MERGE_DOT_PRECISION),
        b_Ai11, input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=_MERGE_DOT_PRECISION),
        b_Ai22, input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A21, b_Ai10, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai21, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A31, b_Ai10, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai20, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )

    # Step 5: store
    p_A00 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_A10 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_A30 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    p_A33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), boundary_check=(0, 1))


# ════════════════════════════════════════════════════════════════════════════════
# Kernel 4: recompute_w_u
# ════════════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * Hg + i_h // (H // Hg)) * K, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# ════════════════════════════════════════════════════════════════════════════════
# Kernel 5: chunk_delta_h (inter-chunk state propagation)
# ════════════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k, v, w, v_new, g, gk, h, initial_state, initial_state_indices, cu_seqlens, chunk_offsets, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr, SAVE_NEW_VALUE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K), other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K), other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K), other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K), other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[None, :]

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if INPLACE_UPDATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ════════════════════════════════════════════════════════════════════════════════
# Kernel 6: chunk_fwd_o (output)
# ════════════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q, k, v, h, g, o, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, tl.trans(b_h))
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ════════════════════════════════════════════════════════════════════════════════
# Pipeline orchestration (Python launchers)
# ════════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 64


def _chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, initial_state_indices, cu_seqlens):
    """Full chunk pipeline: cumsum → intra → delta_h → chunk_o."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    Hg = k.shape[2]
    BT = CHUNK_SIZE

    chunk_indices_t = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices_t)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # 1. cumsum
    g_cumsum = chunk_local_cumsum(g, chunk_size=BT, cu_seqlens=cu_seqlens)

    # 2. fused kkt + solve_tril
    BC = 16
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k, g=g_cumsum, beta=beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices_t,
        T=T, H=H, Hg=Hg, K=K, BT=BT, BC=BC,
    )

    # 3. recompute_w_u
    BK_wu, BV_wu = 64, 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k, v=v, beta=beta, w=w, u=u, A=A, g=g_cumsum,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices_t,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=BK_wu, BV=BV_wu,
        IS_VARLEN=cu_seqlens is not None, num_warps=4, num_stages=3,
    )

    # 4. delta_h (inter-chunk state propagation + inplace state update)
    chunk_offsets_t = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(u)

    def grid_h(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid_h](
        k=k, v=u, w=w, v_new=v_new, g=g_cumsum, gk=None, h=h,
        initial_state=initial_state, initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets_t,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BV=32,
        USE_G=True, USE_GK=False,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=True, SAVE_NEW_VALUE=True,
        IS_VARLEN=cu_seqlens is not None, num_warps=4, num_stages=2,
    )

    # 5. chunk_o (output)
    o = torch.zeros_like(v)

    def grid_o(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid_o](
        q, k, v_new, h, g_cumsum, o, cu_seqlens, chunk_indices_t, scale,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=128, BV=64,
        USE_G=True, IS_VARLEN=cu_seqlens is not None,
        num_warps=4, num_stages=2,
    )
    return o


def chunk_gated_delta_rule(q, k, v, g, beta, initial_state, initial_state_indices, cu_seqlens, use_qk_l2norm_in_kernel=True):
    """Top-level API matching sglang's chunk_gated_delta_rule."""
    scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    o = _chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, initial_state_indices, cu_seqlens)
    return o


# ════════════════════════════════════════════════════════════════════════════════
# Golden reference: pure PyTorch token-by-token recurrent
# ════════════════════════════════════════════════════════════════════════════════

def _golden_recurrent(q, k, v, g, beta, initial_state, cache_indices, cu_seqlens, use_qk_l2norm=True):
    """Per-batch token-by-token recurrent reference (pure PyTorch, float32).

    Mirrors fused_recurrent_gated_delta_rule_fwd_kernel exactly.
    h layout: (V, K) — same as kernel b_h[BV, BK].
    """
    B, T_total, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1
    scale = K ** -0.5

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta_t = beta.float()

    if use_qk_l2norm:
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    pool = initial_state.clone()
    h_cur = pool[cache_indices.long()].contiguous().clone()  # (N, H, V, K)

    o_ref = torch.zeros(1, T_total, H, V, dtype=torch.float32, device=q.device)

    for n in range(N):
        bos = int(cu_seqlens[n].item())
        eos = int(cu_seqlens[n + 1].item())

        for ih in range(H):
            h = h_cur[n, ih].clone()  # (V, K)

            for t in range(bos, eos):
                b_q = q[0, t, ih] * scale      # (K,)
                b_k = k[0, t, ih]               # (K,)
                b_v = v[0, t, ih]               # (V,)
                b_g = g[0, t, ih]               # scalar
                b_beta = beta_t[0, t, ih]       # scalar

                h = h * torch.exp(b_g)
                b_v = b_v - h.mv(b_k)
                b_v = b_v * b_beta
                h = h + b_v.unsqueeze(1) * b_k.unsqueeze(0)  # (V,1)*(1,K)
                o_ref[0, t, ih] = h.mv(b_q)

            h_cur[n, ih] = h

    pool[cache_indices.long()] = h_cur
    return o_ref, pool


# ════════════════════════════════════════════════════════════════════════════════
# Test framework
# ════════════════════════════════════════════════════════════════════════════════

def _create_inputs(device, dtype, *, num_seqs=4, seqlen=128, H=16, K_dim=128, V_dim=128, state_pool_size=32):
    total_tokens = num_seqs * seqlen
    q = torch.randn(1, total_tokens, H, K_dim, device="cpu", dtype=dtype)
    k = torch.randn(1, total_tokens, H, K_dim, device="cpu", dtype=dtype)
    v = torch.randn(1, total_tokens, H, V_dim, device="cpu", dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(1, total_tokens, H, device="cpu", dtype=dtype))
    beta = torch.sigmoid(torch.randn(1, total_tokens, H, device="cpu", dtype=dtype))

    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.long, device="cpu")
    cu_seqlens[1:] = torch.arange(1, num_seqs + 1, dtype=torch.long) * seqlen

    perm = torch.randperm(state_pool_size)[:num_seqs]
    cache_indices = perm.to(torch.int32)
    state_pool = torch.randn(state_pool_size, H, V_dim, K_dim, device="cpu", dtype=torch.float32) * 0.1

    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    g = g.to(device)
    beta = beta.to(device)
    cu_seqlens = cu_seqlens.to(device)
    cache_indices = cache_indices.to(device)
    state_pool = state_pool.to(device)

    return {
        "q": q, "k": k, "v": v, "g": g, "beta": beta,
        "cu_seqlens": cu_seqlens, "state_pool": state_pool, "cache_indices": cache_indices,
        "num_seqs": num_seqs, "seqlen": seqlen, "H": H, "K_dim": K_dim, "V_dim": V_dim,
        "total_tokens": total_tokens, "state_pool_size": state_pool_size,
    }


def _launch_kernel_impl(inputs):
    pool = inputs["state_pool"].clone()
    o = chunk_gated_delta_rule(
        q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], beta=inputs["beta"],
        initial_state=pool, initial_state_indices=inputs["cache_indices"],
        cu_seqlens=inputs["cu_seqlens"], use_qk_l2norm_in_kernel=True,
    )
    inputs["output"] = o
    inputs["final_pool"] = pool


def _golden_compute(inputs):
    pool = inputs["state_pool"].clone()
    o_ref, pool_ref = _golden_recurrent(
        q=inputs["q"], k=inputs["k"], v=inputs["v"], g=inputs["g"], beta=inputs["beta"],
        initial_state=pool, cache_indices=inputs["cache_indices"],
        cu_seqlens=inputs["cu_seqlens"], use_qk_l2norm=True,
    )
    return o_ref, pool_ref


def run_accuracy(*, device=None, atol=0.02, rtol=0.01):
    torch.manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    num_seqs = int(os.getenv("ACC_NUM_SEQS", "4"))
    seqlen = int(os.getenv("ACC_SEQLEN", "128"))
    H = int(os.getenv("ACC_H", "16"))
    K_dim = int(os.getenv("ACC_K", "128"))
    V_dim = int(os.getenv("ACC_V", "128"))
    state_pool_size = int(os.getenv("ACC_STATE_POOL", "32"))

    inputs = _create_inputs(device, dtype, num_seqs=num_seqs, seqlen=seqlen,
                            H=H, K_dim=K_dim, V_dim=V_dim, state_pool_size=state_pool_size)
    _launch_kernel_impl(inputs)
    torch.cuda.synchronize()

    kernel_output = inputs["output"]
    kernel_pool = inputs["final_pool"]
    cache_indices = inputs["cache_indices"]

    golden_output, golden_pool = _golden_compute(inputs)
    torch.cuda.synchronize()

    # Output accuracy
    diff = (kernel_output.float() - golden_output.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    has_nan = torch.isnan(kernel_output).any().item()
    match_output = torch.allclose(kernel_output.float(), golden_output.float(), atol=atol, rtol=rtol)

    print(f"[accuracy/output] shape: kernel={kernel_output.shape}, golden={golden_output.shape}")
    print(f"[accuracy/output] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, nan={has_nan}")
    print(f"[accuracy/output] PASS={match_output}")

    # State accuracy
    ref_slots = golden_pool[cache_indices.long()].contiguous()
    new_slots = kernel_pool[cache_indices.long()].contiguous()
    state_diff = (ref_slots.float() - new_slots.float()).abs()
    state_max_diff = state_diff.max().item()
    state_mean_diff = state_diff.mean().item()
    match_state = torch.allclose(ref_slots.float(), new_slots.float(), atol=atol, rtol=rtol)

    print(f"[accuracy/state] max_diff={state_max_diff:.6f}, mean_diff={state_mean_diff:.6f}")
    print(f"[accuracy/state] PASS={match_state}")
    print(f"[accuracy] ALL_PASS={match_output and match_state}")


def run_perflog(*, device=None):
    torch.manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    num_seqs = int(os.getenv("PERF_NUM_SEQS", "4"))
    seqlen = int(os.getenv("PERF_SEQLEN", "1024"))
    H = int(os.getenv("PERF_H", "16"))
    K_dim = int(os.getenv("PERF_K", "128"))
    V_dim = int(os.getenv("PERF_V", "128"))

    inputs = _create_inputs(device, dtype, num_seqs=num_seqs, seqlen=seqlen, H=H, K_dim=K_dim, V_dim=V_dim)
    _launch_kernel_impl(inputs)
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

    configs = [
        {"num_seqs": 4, "seqlen": 64},
        {"num_seqs": 4, "seqlen": 128},
        {"num_seqs": 4, "seqlen": 256},
        {"num_seqs": 4, "seqlen": 512},
        {"num_seqs": 4, "seqlen": 1024},
        {"num_seqs": 8, "seqlen": 128},
        {"num_seqs": 16, "seqlen": 128},
        {"num_seqs": 32, "seqlen": 128},
    ]
    results = []
    for cfg in configs:
        inputs = _create_inputs(device, dtype, **cfg)

        def bench_fn():
            _launch_kernel_impl(inputs)
            torch.cuda.synchronize()

        ms = _bench_with_events(bench_fn, warmup=10, iters=100)
        results.append({**cfg, "latency_ms": ms})
        print(f"num_seqs={cfg['num_seqs']:>3}, seqlen={cfg['seqlen']:>5}, latency={ms:.3f} ms")

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    mode = os.getenv("MODE", "perf").lower()
    if mode == "perf":
        run_perflog()
    elif mode == "bench":
        run_bench()
    else:
        run_accuracy()
