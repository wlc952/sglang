# Benchmarks SGLang fused layernorm scale shift kernels with real production shapes (Wan 720p)
# 1. fused_norm_scale_shift
# 2. fused_scale_residual_norm_scale_shift (gate tensor, affine, scalar scale/shift)
# 3. fused_scale_residual_norm_scale_shift (no gate, no affine, vector scale/shift)
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
    fused_norm_scale_shift as cutedsl_fused_norm_scale_shift,
    fused_scale_residual_norm_scale_shift as cutedsl_fused_scale_residual_norm_scale_shift,
)
from sglang.jit_kernel.scale_residual_norm_scale_shift import (
    fused_norm_scale_shift as jit_fused_norm_scale_shift,
    fused_scale_residual_norm_scale_shift as jit_fused_scale_residual_norm_scale_shift,
)

B, S, D = 1, 75600, 5120
DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-6
LINE_VALS = ["CuteDSL", "JIT"]
LINE_NAMES = ["CuteDSL", "JIT"]
STYLES = [("red", "-"), ("blue", "--")]


# ============================================================================
# Benchmark 1: fused_norm_scale_shift
# x=[1,75600,5120], scale/shift=[1,1,5120], no affine, layer norm
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[S],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_norm_scale_shift",
        args={},
    )
)
def bench_fused_norm_scale_shift(S: int, provider: str) -> Tuple[float, float, float]:
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    if provider == "CuteDSL":
        fn = lambda: cutedsl_fused_norm_scale_shift(x, None, None, scale, shift, "layer", EPS)
    else:
        fn = lambda: jit_fused_norm_scale_shift(x, None, None, scale, shift, "layer", EPS)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


# ============================================================================
# Benchmark 2: fused_scale_residual_norm_scale_shift
# residual/x=[1,75600,5120], gate=[1,1,5120], affine=True, scale/shift=[1]
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[S],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_scale_residual_norm_scale_shift_gate_affine",
        args={},
    )
)
def bench_fused_scale_residual_gate_affine(
    S: int, provider: str
) -> Tuple[float, float, float]:
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(D, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
    scale = torch.ones(1, dtype=DTYPE, device=DEVICE)
    shift = torch.zeros(1, dtype=DTYPE, device=DEVICE)
    if provider == "CuteDSL":
        fn = lambda: cutedsl_fused_scale_residual_norm_scale_shift(
            residual, x, gate, weight, bias, scale, shift, "layer", EPS
        )
    else:
        fn = lambda: jit_fused_scale_residual_norm_scale_shift(
            residual, x, gate, weight, bias, scale, shift, "layer", EPS
        )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


# ============================================================================
# Benchmark 3: fused_scale_residual_norm_scale_shift
# residual/x=[1,75600,5120], gate=None, no affine, scale/shift=[1,1,5120]
# ============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[S],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused_scale_residual_norm_scale_shift_no_gate",
        args={},
    )
)
def bench_fused_scale_residual_no_gate(
    S: int, provider: str
) -> Tuple[float, float, float]:
    residual = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    scale = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    shift = torch.randn(B, 1, D, dtype=DTYPE, device=DEVICE)
    if provider == "CuteDSL":
        fn = lambda: cutedsl_fused_scale_residual_norm_scale_shift(
            residual, x, None, None, None, scale, shift, "layer", EPS
        )
    else:
        fn = lambda: jit_fused_scale_residual_norm_scale_shift(
            residual, x, None, None, None, scale, shift, "layer", EPS
        )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Benchmark 1: fused_norm_scale_shift")
    print(f"  x=[1,75600,5120], scale/shift=[1,1,5120], no affine")
    print(f"{'='*80}\n")
    bench_fused_norm_scale_shift.run(print_data=True)

    print(f"\n{'='*80}")
    print("Benchmark 2: fused_scale_residual_norm_scale_shift")
    print(f"  gate=[1,1,5120], affine=True, scale/shift=[1]")
    print(f"{'='*80}\n")
    bench_fused_scale_residual_gate_affine.run(print_data=True)

    print(f"\n{'='*80}")
    print("Benchmark 3: fused_scale_residual_norm_scale_shift")
    print(f"  gate=None, no affine, scale/shift=[1,1,5120]")
    print(f"{'='*80}\n")
    bench_fused_scale_residual_no_gate.run(print_data=True)
