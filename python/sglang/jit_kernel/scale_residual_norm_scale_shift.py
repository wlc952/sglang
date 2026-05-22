from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_scale_residual_norm_scale_shift_module(
    dtype: torch.dtype, norm_type: str
) -> Module:
    if norm_type not in ("layer", "rms"):
        raise ValueError('norm_type must be one of "layer" and "rms"')
    args = make_cpp_args(dtype, norm_type == "layer")
    jit_version = "v76"
    return load_jit(
        "scale_residual_norm_scale_shift",
        jit_version,
        *args,
        cuda_files=["diffusion/scale_residual_norm_scale_shift.cuh"],
        cuda_wrappers=[
            ("fused_norm_scale_shift", f"FusedNormScaleShiftKernel<{args}>::run"),
            (
                "fused_scale_residual_norm_scale_shift",
                f"FusedScaleResidualNormScaleShiftKernel<{args}>::run",
            ),
        ],
    )


_SCALAR_CACHE: dict[tuple[float, torch.dtype, torch.device], torch.Tensor] = {}
_CAST_CACHE: dict[
    tuple[int, torch.dtype],
    tuple[weakref.ReferenceType[torch.Tensor], int, torch.Tensor],
] = {}
_CAST_CACHE_MAX_ENTRIES = 2048


def _get_scalar_tensor(
    value: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (value, dtype, device)
    t = _SCALAR_CACHE.get(key)
    if t is None:
        t = torch.tensor([value], dtype=dtype, device=device)
        _SCALAR_CACHE[key] = t
    return t


def _cast_if_needed_cached(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype == dtype:
        return t

    key = (id(t), dtype)
    ver = int(getattr(t, "_version", -1))
    entry = _CAST_CACHE.get(key)
    if entry is not None:
        ref, cached_ver, cached_t = entry
        if ref() is t and cached_ver == ver and cached_t.device == t.device:
            return cached_t

    cast_t = t.to(dtype=dtype)
    _CAST_CACHE[key] = (weakref.ref(t), ver, cast_t)
    if len(_CAST_CACHE) > _CAST_CACHE_MAX_ENTRIES:
        _CAST_CACHE.pop(next(iter(_CAST_CACHE)))
    return cast_t


def _prepare_optional_affine(
    t: Optional[torch.Tensor],
    default_value: float,
    x_dtype: torch.dtype,
    x_device: torch.device,
) -> torch.Tensor:
    if t is None:
        return _get_scalar_tensor(default_value, x_dtype, x_device)

    out = _cast_if_needed_cached(t, x_dtype)
    if out.stride()[-1] != 1:
        out = out.contiguous()
    return out


def _broadcast_tensor_for_bsfd(
    tensor: Union[Optional[torch.Tensor], int],
    B: int,
    S: int,
    D: int,
) -> Union[Optional[torch.Tensor], int]:
    """
    Broadcast to (B, S, D) without mandatory materialization for
    [D], [1, D], [1, 1, D], [B, D], [B, 1, D], [B, S, D].
    Keep [1] scalar and [B, F, 1, D] as-is.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.ndim == 1:
        if tensor.numel() == 1:
            return tensor
        return tensor.view(1, 1, D).expand(B, S, D)
    if tensor.ndim == 2:
        return tensor.view(tensor.shape[0], 1, D).expand(B, S, D)
    if tensor.ndim == 3:
        return tensor.expand(B, S, D)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"BSFD broadcast: unsupported tensor ndim: {tensor.ndim}.")


def _validate_x(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (B, S, D):
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError("Validate failed: not contiguous on dim D.")


def _validate_weight_bias(t: Optional[torch.Tensor], B: int, S: int, D: int):
    if t is None:
        return
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (D,):
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError("Validate failed: not contiguous on dim D.")


def _validate_scale_shift(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    failed = False
    if t.ndim == 1 and (t.shape[0] not in (1, D)):
        failed = True
    elif t.ndim == 2 and ((t.shape[0] not in (1, B)) or t.shape[1] != D):
        failed = True
    elif t.ndim == 3 and (
        (t.shape[0] not in (1, B)) or (t.shape[1] not in (1, S) or t.shape[2] != D)
    ):
        failed = True
    elif t.ndim == 4 and (t.shape[0] != B or t.shape[2] != 1 or t.shape[3] != D):
        F = t.shape[1]
        if S % F != 0:
            raise ValueError(f"Validate failed: S({S}) must be divisible by F({F}).")
        failed = True
    if failed:
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError("Validate failed: not contiguous on dim D.")


def _validate_gate(t: Union[torch.Tensor, int, None], B: int, S: int, D: int):
    if not isinstance(t, torch.Tensor):
        return
    _validate_scale_shift(t, B, S, D)


def fused_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    SUPA JIT fused op:
      y = norm(x, weight, bias) * (1 + scale) + shift
    where norm is layernorm or rmsnorm.
    """
    if norm_type not in ("layer", "rms"):
        raise ValueError('norm_type must be one of "layer" and "rms"')

    B, S, D = x.shape
    _validate_x(x, B, S, D)
    _validate_weight_bias(weight, B, S, D)
    _validate_weight_bias(bias, B, S, D)
    _validate_scale_shift(scale, B, S, D)
    _validate_scale_shift(shift, B, S, D)

    if D % 256 != 0 or D > 8192:
        raise ValueError(f"D={D} not supported, must be multiple of 256 and <= 8192")

    x = x.contiguous()
    scale_b = _broadcast_tensor_for_bsfd(scale, B, S, D)
    shift_b = _broadcast_tensor_for_bsfd(shift, B, S, D)
    if not isinstance(scale_b, torch.Tensor) or not isinstance(shift_b, torch.Tensor):
        raise ValueError("scale and shift must be tensors")
    scale = scale_b
    shift = shift_b

    has_weight = weight is not None
    has_bias = bias is not None
    weight = _prepare_optional_affine(weight, 1.0, x.dtype, x.device)
    bias = _prepare_optional_affine(bias, 0.0, x.dtype, x.device)

    y = torch.empty_like(x)
    module = _jit_scale_residual_norm_scale_shift_module(x.dtype, norm_type)
    module.fused_norm_scale_shift(
        y,
        x,
        weight,
        bias,
        scale,
        shift,
        float(eps),
        int(has_weight),
        int(has_bias),
    )
    return y


def fused_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],  # Union[Optional[torch.Tensor], int] indeed
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SUPA JIT fused op:
      residual_out = residual + gate * x
      y = norm(residual_out, weight, bias) * (1 + scale) + shift
    where norm is layernorm or rmsnorm.
    """
    if norm_type not in ("layer", "rms"):
        raise ValueError('norm_type must be one of "layer" and "rms"')

    B, S, D = x.shape
    _validate_x(x, B, S, D)
    _validate_x(residual, B, S, D)
    _validate_gate(gate, B, S, D)
    _validate_weight_bias(weight, B, S, D)
    _validate_weight_bias(bias, B, S, D)
    _validate_scale_shift(scale, B, S, D)
    _validate_scale_shift(shift, B, S, D)

    if D % 256 != 0 or D > 8192:
        raise ValueError(f"D={D} not supported, must be multiple of 256 and <= 8192")

    residual = residual.contiguous()
    x = x.contiguous()

    gate_is_one_hint = gate is None
    gate_t = _get_scalar_tensor(1.0, x.dtype, x.device) if gate is None else gate
    if not isinstance(gate_t, torch.Tensor):
        raise ValueError("gate must be a tensor or None")

    gate_b = _broadcast_tensor_for_bsfd(gate_t, B, S, D)
    scale_b = _broadcast_tensor_for_bsfd(scale, B, S, D)
    shift_b = _broadcast_tensor_for_bsfd(shift, B, S, D)
    if (
        not isinstance(gate_b, torch.Tensor)
        or not isinstance(scale_b, torch.Tensor)
        or not isinstance(shift_b, torch.Tensor)
    ):
        raise ValueError("gate/scale/shift must be tensors")
    gate_t = gate_b
    scale = scale_b
    shift = shift_b

    has_weight = weight is not None
    has_bias = bias is not None
    weight = _prepare_optional_affine(weight, 1.0, x.dtype, x.device)
    bias = _prepare_optional_affine(bias, 0.0, x.dtype, x.device)

    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    module = _jit_scale_residual_norm_scale_shift_module(x.dtype, norm_type)
    module.fused_scale_residual_norm_scale_shift(
        y,
        residual_out,
        residual,
        x,
        gate_t,
        weight,
        bias,
        scale,
        shift,
        float(eps),
        int(has_weight),
        int(has_bias),
        int(gate_is_one_hint),
    )
    return y, residual_out
