"""
GDN performance test — imports from framework.
Used to verify correctness after algorithm team fixes, and benchmark perf.

Usage:
  MODE=extend python3 debug/test_gdn.py
  MODE=decode python3 debug/test_gdn.py
  MODE=chunk  python3 debug/test_gdn.py
  MODE=all    python3 debug/test_gdn.py       (default)
"""

import os
import torch

from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)


# ── Extend (fused_recurrent) ──────────────────────────────────────────────────

def test_extend():
    print("=== fused_recurrent_gated_delta_rule (extend) ===")
    num_seqs = 4
    seqlen = 1024
    H = 16
    HV = 32
    K_dim = 128
    V_dim = 128
    total_tokens = num_seqs * seqlen

    q = torch.empty(1, total_tokens, H, K_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, total_tokens, H, K_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(1, total_tokens, HV, V_dim, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(1, total_tokens, HV, device="cuda", dtype=torch.float32)
    beta = torch.ones(1, total_tokens, HV, device="cuda", dtype=torch.float32)
    initial_state = torch.zeros(num_seqs, HV, V_dim, K_dim, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.arange(0, (num_seqs + 1) * seqlen, seqlen, dtype=torch.long, device="cuda")

    out, final_state = fused_recurrent_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    print(f"  output shape: {out.shape}")
    print(f"  final_state shape: {final_state.shape}")
    print(f"  PASS")


# ── Chunk (extend, chunked pipeline) ─────────────────────────────────────────

def test_chunk():
    print("\n=== chunk_gated_delta_rule (chunk extend) ===")
    num_seqs = 4
    seqlen = 1024
    H = 16
    K_dim = 128
    V_dim = 128
    pool_size = 32
    total_tokens = num_seqs * seqlen

    q = torch.empty(1, total_tokens, H, K_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, total_tokens, H, K_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(1, total_tokens, H, V_dim, device="cuda", dtype=torch.bfloat16)
    g = torch.full((1, total_tokens, H), -0.5, device="cuda", dtype=torch.bfloat16)
    beta = torch.full((1, total_tokens, H), 0.5, device="cuda", dtype=torch.bfloat16)
    state_pool = torch.zeros(pool_size, H, V_dim, K_dim, device="cuda", dtype=torch.float32)
    cache_indices = torch.arange(num_seqs, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(0, (num_seqs + 1) * seqlen, seqlen, dtype=torch.long, device="cuda")

    out, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=state_pool,
        initial_state_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    print(f"  output shape: {out.shape}")
    print(f"  PASS")


# ── Decode (fused_sigmoid_gating) ─────────────────────────────────────────────

def test_decode():
    print("\n=== fused_sigmoid_gating_delta_rule_update (decode) ===")
    B = 128
    H = 16
    HV = 32
    K_dim = 128
    V_dim = 128

    q = torch.empty(1, B, H, K_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, B, H, K_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(1, B, HV, V_dim, device="cuda", dtype=torch.bfloat16)
    a = torch.zeros(1, B, HV, device="cuda", dtype=torch.float32)
    b = torch.zeros(1, B, HV, device="cuda", dtype=torch.float32)
    A_log = torch.zeros(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(HV, device="cuda", dtype=torch.float32)
    ssm_states = torch.zeros(8804, HV, K_dim, V_dim, device="cuda", dtype=torch.float32)
    cache_indices = torch.arange(B, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(B + 1, dtype=torch.int32, device="cuda")

    out = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    print(f"  output shape: {out.shape}")
    print(f"  PASS")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = os.getenv("MODE", "all").lower()

    if mode in ("extend", "all"):
        test_extend()
    if mode in ("chunk", "all"):
        test_chunk()
    if mode in ("decode", "all"):
        test_decode()
