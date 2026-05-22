#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

enum BSFDMode : int32_t {
  kScalarTensor = 0,
  kTensorBSD = 1,
  kTensorBF1D = 2,
};

enum BSFDDType : int32_t {
  kBSFDDTypeInvalid = 0,
  kBSFDDTypeFP16 = 1,
  kBSFDDTypeBF16 = 2,
  kBSFDDTypeFP32 = 3,
};

struct BSFDMeta {
  const void* ptr;
  int32_t mode;
  int32_t dtype;
  int32_t F;
  int64_t stride0;
  int64_t stride1;
  int64_t stride2;
  int64_t stride3;
};

template <typename T>
constexpr int32_t bsfd_dtype_code() {
  if constexpr (std::is_same_v<T, fp16_t>) {
    return kBSFDDTypeFP16;
  } else if constexpr (std::is_same_v<T, bf16_t>) {
    return kBSFDDTypeBF16;
  } else if constexpr (std::is_same_v<T, float>) {
    return kBSFDDTypeFP32;
  } else {
    return kBSFDDTypeInvalid;
  }
}

enum ResidualSpecialMode : int32_t {
  kResidualGeneric = 0,
  kResidualGateScalarScaleShiftBSD = 1,
  kResidualGateBSDScaleShiftScalar = 2,
  kResidualGateOneScaleShiftBSD = 3,
  kResidualGateOneScaleShiftBSDFP32 = 4,
  kResidualGateScalarScaleShiftBSDAffine = 5,
  kResidualGateOneScaleShiftBSDPacked = 6,
};

constexpr int kMaxCachedItersScalar = 8;
constexpr int kMaxCachedItersPacked = 4;

SGL_DEVICE float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

SGL_DEVICE float load_meta_value(const BSFDMeta& meta, int64_t idx) {
  if (meta.dtype == kBSFDDTypeFP16) {
    const fp16_t* ptr = static_cast<const fp16_t*>(meta.ptr);
    return device::cast<float>(ptr[idx]);
  }
  if (meta.dtype == kBSFDDTypeBF16) {
    const bf16_t* ptr = static_cast<const bf16_t*>(meta.ptr);
    return device::cast<float>(ptr[idx]);
  }
  if (meta.dtype == kBSFDDTypeFP32) {
    const float* ptr = static_cast<const float*>(meta.ptr);
    return ptr[idx];
  }
  return 0.0f;
}

SGL_DEVICE float load_meta_scalar(const BSFDMeta& meta) {
  return load_meta_value(meta, 0);
}

SGL_DEVICE float load_bsfd_value(const BSFDMeta& meta, int b, int s, int d, int S) {
  if (meta.mode == kScalarTensor) {
    return load_meta_scalar(meta);
  }
  if (meta.mode == kTensorBSD) {
    const int64_t idx = static_cast<int64_t>(b) * meta.stride0 + static_cast<int64_t>(s) * meta.stride1 +
                        static_cast<int64_t>(d) * meta.stride2;
    return load_meta_value(meta, idx);
  }
  const int frame_len = S / meta.F;
  const int f = s / frame_len;
  const int64_t idx = static_cast<int64_t>(b) * meta.stride0 + static_cast<int64_t>(f) * meta.stride1 +
                      static_cast<int64_t>(d) * meta.stride3;
  return load_meta_value(meta, idx);
}

SGL_DEVICE float load_bsd_value(const BSFDMeta& meta, int64_t token_base, int d) {
  const int64_t idx = token_base + static_cast<int64_t>(d) * meta.stride2;
  return load_meta_value(meta, idx);
}

SGL_DEVICE fp32x2_t load_bsfd_pair(const BSFDMeta& meta, int b, int s, int d2, int S) {
  const int d0 = d2 << 1;
  const int d1 = d0 + 1;
  if (meta.mode == kScalarTensor) {
    const float v = load_meta_scalar(meta);
    return {v, v};
  }
  if (meta.mode == kTensorBSD) {
    const int64_t token_base = static_cast<int64_t>(b) * meta.stride0 + static_cast<int64_t>(s) * meta.stride1;
    const int64_t idx0 = token_base + static_cast<int64_t>(d0) * meta.stride2;
    const int64_t idx1 = token_base + static_cast<int64_t>(d1) * meta.stride2;
    return {load_meta_value(meta, idx0), load_meta_value(meta, idx1)};
  }
  const int frame_len = S / meta.F;
  const int f = s / frame_len;
  const int64_t frame_base = static_cast<int64_t>(b) * meta.stride0 + static_cast<int64_t>(f) * meta.stride1;
  const int64_t idx0 = frame_base + static_cast<int64_t>(d0) * meta.stride3;
  const int64_t idx1 = frame_base + static_cast<int64_t>(d1) * meta.stride3;
  return {load_meta_value(meta, idx0), load_meta_value(meta, idx1)};
}

template <typename T, bool kIsLayerNorm, bool kFastBSFD, bool kHasWeight, bool kHasBias>
__global__ void fused_norm_scale_shift_kernel(
    const T* __restrict__ x,
    const int64_t x_stride0,
    const int64_t x_stride1,
    const int64_t x_stride2,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const BSFDMeta scale_meta,
    const BSFDMeta shift_meta,
    T* __restrict__ y,
    const int64_t y_stride0,
    const int64_t y_stride1,
    const int64_t y_stride2,
    const int B,
    const int S,
    const int D,
    const float eps) {
  __shared__ float smem_sum[32];
  __shared__ float smem_sumsq[32];
  __shared__ float smem_mean;
  __shared__ float smem_inv_std;
  extern __shared__ uint8_t smem_x_dynamic[];

  const int token = static_cast<int>(blockIdx.x);
  if (token >= B * S) {
    return;
  }
  const int b = token / S;
  const int s = token % S;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int num_warps = static_cast<int>(blockDim.x >> 5);
  constexpr bool kUseSmemX = kFastBSFD;

  const int64_t x_base = static_cast<int64_t>(b) * x_stride0 + static_cast<int64_t>(s) * x_stride1;

  float sum = 0.0f;
  float sum_sq = 0.0f;
  if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
    using PackT = packed_t<T>;
    using VecPackT = device::AlignedVector<PackT, 4>;
    const PackT* x_pack = reinterpret_cast<const PackT*>(x + x_base);
    PackT* x_smem_pack = reinterpret_cast<PackT*>(smem_x_dynamic);
    const int D2 = D >> 1;
    const int D4 = D2 >> 2;
    if (D4 <= static_cast<int>(blockDim.x)) {
      if (tid < D4) {
        const int d4 = tid;
        VecPackT x4;
        x4.load(x_pack, d4);
        if constexpr (kUseSmemX) {
          x4.store(x_smem_pack, d4);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const fp32x2_t v2 = device::cast<fp32x2_t>(x4[i]);
          if constexpr (kIsLayerNorm) {
            sum += v2.x + v2.y;
          }
          sum_sq += v2.x * v2.x + v2.y * v2.y;
        }
      }
    } else {
      for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
        VecPackT x4;
        x4.load(x_pack, d4);
        if constexpr (kUseSmemX) {
          x4.store(x_smem_pack, d4);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const fp32x2_t v2 = device::cast<fp32x2_t>(x4[i]);
          if constexpr (kIsLayerNorm) {
            sum += v2.x + v2.y;
          }
          sum_sq += v2.x * v2.x + v2.y * v2.y;
        }
      }
    }
  } else {
    const bool scale_is_bsd_pack =
        (scale_meta.mode == kTensorBSD) && (scale_meta.stride2 == 1) && (scale_meta.dtype == bsfd_dtype_code<T>());
    const bool shift_is_bsd_pack =
        (shift_meta.mode == kTensorBSD) && (shift_meta.stride2 == 1) && (shift_meta.dtype == bsfd_dtype_code<T>());
    const int D2 = D >> 1;
    const int D4 = D2 >> 2;
    const bool use_pack_path = !kHasWeight && !kHasBias && (scale_is_bsd_pack && shift_is_bsd_pack) &&
                               (x_stride2 == 1) && (y_stride2 == 1) &&
                               (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>);
    if (use_pack_path) { if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
      using PackT = packed_t<T>;
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int64_t scale_token_base = static_cast<int64_t>(b) * scale_meta.stride0 + static_cast<int64_t>(s) * scale_meta.stride1;
      const int64_t shift_token_base = static_cast<int64_t>(b) * shift_meta.stride0 + static_cast<int64_t>(s) * shift_meta.stride1;
      const PackT* x_pack = reinterpret_cast<const PackT*>(x + x_base);
      const PackT* scale_pack =
          reinterpret_cast<const PackT*>(static_cast<const T*>(scale_meta.ptr) + scale_token_base);
      const PackT* shift_pack =
          reinterpret_cast<const PackT*>(static_cast<const T*>(shift_meta.ptr) + shift_token_base);
      for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
        VecPackT x4;
        x4.load(x_pack, d4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const fp32x2_t v2 = device::cast<fp32x2_t>(x4[i]);
          if constexpr (kIsLayerNorm) {
            sum += v2.x + v2.y;
          }
          sum_sq += v2.x * v2.x + v2.y * v2.y;
        }
      }
    }} else {
      for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
        const float v = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
        if constexpr (kIsLayerNorm) {
          sum += v;
        }
        sum_sq += v * v;
      }
    }
  }

  if constexpr (kIsLayerNorm) {
    sum = warp_reduce_sum(sum);
  }
  sum_sq = warp_reduce_sum(sum_sq);

  if (lane == 0) {
    if constexpr (kIsLayerNorm) {
      smem_sum[warp_id] = sum;
    }
    smem_sumsq[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = 0.0f;
    float block_sumsq = 0.0f;
    if (lane < num_warps) {
      if constexpr (kIsLayerNorm) {
        block_sum = smem_sum[lane];
      }
      block_sumsq = smem_sumsq[lane];
    }
    if constexpr (kIsLayerNorm) {
      block_sum = warp_reduce_sum(block_sum);
    }
    block_sumsq = warp_reduce_sum(block_sumsq);
    if (lane == 0) {
      float mean = 0.0f;
      float inv_std = 0.0f;
      if constexpr (kIsLayerNorm) {
        mean = block_sum / static_cast<float>(D);
        const float var = fmaxf(block_sumsq / static_cast<float>(D) - mean * mean, 0.0f);
        inv_std = rsqrtf(var + eps);
      } else {
        inv_std = rsqrtf(block_sumsq / static_cast<float>(D) + eps);
      }
      smem_mean = mean;
      smem_inv_std = inv_std;
    }
  }
  __syncthreads();

  const int64_t y_base = static_cast<int64_t>(b) * y_stride0 + static_cast<int64_t>(s) * y_stride1;
  const float mean = smem_mean;
  const float inv_std = smem_inv_std;
  int64_t scale_token_base = 0;
  int64_t shift_token_base = 0;
  if constexpr (kFastBSFD) {
    scale_token_base = static_cast<int64_t>(b) * scale_meta.stride0 + static_cast<int64_t>(s) * scale_meta.stride1;
    shift_token_base = static_cast<int64_t>(b) * shift_meta.stride0 + static_cast<int64_t>(s) * shift_meta.stride1;
  } else {
    if (scale_meta.mode == kTensorBSD) {
      scale_token_base = static_cast<int64_t>(b) * scale_meta.stride0 + static_cast<int64_t>(s) * scale_meta.stride1;
    }
    if (shift_meta.mode == kTensorBSD) {
      shift_token_base = static_cast<int64_t>(b) * shift_meta.stride0 + static_cast<int64_t>(s) * shift_meta.stride1;
    }
  }

  if constexpr (kFastBSFD && (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>)) {
    using PackT = packed_t<T>;
    using VecPackT = device::AlignedVector<PackT, 4>;
    const PackT* x_pack = nullptr;
    if constexpr (kUseSmemX) {
      x_pack = reinterpret_cast<const PackT*>(smem_x_dynamic);
    } else {
      x_pack = reinterpret_cast<const PackT*>(x + x_base);
    }
    const PackT* weight_pack = reinterpret_cast<const PackT*>(weight);
    const PackT* bias_pack = reinterpret_cast<const PackT*>(bias);
    const PackT* scale_pack = reinterpret_cast<const PackT*>(static_cast<const T*>(scale_meta.ptr) + scale_token_base);
    const PackT* shift_pack = reinterpret_cast<const PackT*>(static_cast<const T*>(shift_meta.ptr) + shift_token_base);
    PackT* y_pack = reinterpret_cast<PackT*>(y + y_base);
    const int D2 = D >> 1;
    const int D4 = D2 >> 2;
    if constexpr (!kHasWeight && !kHasBias) {
      for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
        VecPackT x4;
        x4.load(x_pack, d4);
        VecPackT sc4;
        sc4.load(scale_pack, d4);
        VecPackT sh4;
        sh4.load(shift_pack, d4);
        VecPackT out4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const fp32x2_t v2 = device::cast<fp32x2_t>(x4[i]);
          const fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
          const fp32x2_t scale2 = device::cast<fp32x2_t>(sc4[i]);
          const fp32x2_t shift2 = device::cast<fp32x2_t>(sh4[i]);
          const fp32x2_t out2 = {
              normed2.x * (1.0f + scale2.x) + shift2.x,
              normed2.y * (1.0f + scale2.y) + shift2.y,
          };
          out4[i] = device::cast<PackT, fp32x2_t>(out2);
        }
        out4.store(y_pack, d4);
      }
    } else {
      for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
        VecPackT x4;
        x4.load(x_pack, d4);
        VecPackT sc4;
        sc4.load(scale_pack, d4);
        VecPackT sh4;
        sh4.load(shift_pack, d4);
        VecPackT out4;
        VecPackT w4;
        VecPackT b4;
        if constexpr (kHasWeight) {
          w4.load(weight_pack, d4);
        }
        if constexpr (kIsLayerNorm) {
          if constexpr (kHasBias) {
            b4.load(bias_pack, d4);
          }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const fp32x2_t v2 = device::cast<fp32x2_t>(x4[i]);
          fp32x2_t normed2;
          if constexpr (kIsLayerNorm) {
            normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
          } else {
            normed2 = {v2.x * inv_std, v2.y * inv_std};
          }
          if constexpr (kHasWeight) {
            const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
            normed2.x *= w2.x;
            normed2.y *= w2.y;
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
              normed2.x += b2.x;
              normed2.y += b2.y;
            }
          }

          const fp32x2_t scale2 = device::cast<fp32x2_t>(sc4[i]);
          const fp32x2_t shift2 = device::cast<fp32x2_t>(sh4[i]);
          const fp32x2_t out2 = {
              normed2.x * (1.0f + scale2.x) + shift2.x,
              normed2.y * (1.0f + scale2.y) + shift2.y,
          };
          out4[i] = device::cast<PackT, fp32x2_t>(out2);
        }
        out4.store(y_pack, d4);
      }
    }
  } else if constexpr (kHasWeight) {
    if constexpr (kIsLayerNorm) {
      if constexpr (kHasBias) {
        for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
          const float v = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
          const float w = device::cast<float>(weight[d]);
          const float base = (v - mean) * inv_std;
          const float normed = base * w + device::cast<float>(bias[d]);
          const float scale =
              kFastBSFD ? load_bsd_value(scale_meta, scale_token_base, d) : load_bsfd_value(scale_meta, b, s, d, S);
          const float shift =
              kFastBSFD ? load_bsd_value(shift_meta, shift_token_base, d) : load_bsfd_value(shift_meta, b, s, d, S);
          y[y_base + static_cast<int64_t>(d) * y_stride2] = device::cast<T>(normed * (1.0f + scale) + shift);
        }
      } else {
        for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
          const float v = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
          const float w = device::cast<float>(weight[d]);
          const float base = (v - mean) * inv_std;
          const float normed = base * w;
          const float scale =
              kFastBSFD ? load_bsd_value(scale_meta, scale_token_base, d) : load_bsfd_value(scale_meta, b, s, d, S);
          const float shift =
              kFastBSFD ? load_bsd_value(shift_meta, shift_token_base, d) : load_bsfd_value(shift_meta, b, s, d, S);
          y[y_base + static_cast<int64_t>(d) * y_stride2] = device::cast<T>(normed * (1.0f + scale) + shift);
        }
      }
    } else {
      for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
        const float v = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
        const float w = device::cast<float>(weight[d]);
        const float base = v * inv_std;
        const float normed = base * w;
        const float scale =
            kFastBSFD ? load_bsd_value(scale_meta, scale_token_base, d) : load_bsfd_value(scale_meta, b, s, d, S);
        const float shift =
            kFastBSFD ? load_bsd_value(shift_meta, shift_token_base, d) : load_bsfd_value(shift_meta, b, s, d, S);
        y[y_base + static_cast<int64_t>(d) * y_stride2] = device::cast<T>(normed * (1.0f + scale) + shift);
      }
    }
  } else {
    if constexpr (kIsLayerNorm) {
      if constexpr (kHasBias) {
        for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
          const float v = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
          const float normed = (v - mean) * inv_std + device::cast<float>(bias[d]);
          const float scale =
              kFastBSFD ? load_bsd_value(scale_meta, scale_token_base, d) : load_bsfd_value(scale_meta, b, s, d, S);
          const float shift =
              kFastBSFD ? load_bsd_value(shift_meta, shift_token_base, d) : load_bsfd_value(shift_meta, b, s, d, S);
          y[y_base + static_cast<int64_t>(d) * y_stride2] = device::cast<T>(normed * (1.0f + scale) + shift);
        }
      } else {
        for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
          const float v = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
          const float normed = (v - mean) * inv_std;
          const float scale =
              kFastBSFD ? load_bsd_value(scale_meta, scale_token_base, d) : load_bsfd_value(scale_meta, b, s, d, S);
          const float shift =
              kFastBSFD ? load_bsd_value(shift_meta, shift_token_base, d) : load_bsfd_value(shift_meta, b, s, d, S);
          y[y_base + static_cast<int64_t>(d) * y_stride2] = device::cast<T>(normed * (1.0f + scale) + shift);
        }
      }
    } else {
      for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
        const float v = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
        const float normed = v * inv_std;
        const float scale =
            kFastBSFD ? load_bsd_value(scale_meta, scale_token_base, d) : load_bsfd_value(scale_meta, b, s, d, S);
        const float shift =
            kFastBSFD ? load_bsd_value(shift_meta, shift_token_base, d) : load_bsfd_value(shift_meta, b, s, d, S);
        y[y_base + static_cast<int64_t>(d) * y_stride2] = device::cast<T>(normed * (1.0f + scale) + shift);
      }
    }
  }
}

template <bool kIsLayerNorm, bool kHasWeight, bool kHasBias>
SGL_DEVICE fp32x2_t apply_norm_scale_shift_packed(
    fp32x2_t v2, float mean, float inv_std,
    fp32x2_t w2, fp32x2_t b2, fp32x2_t scale2, fp32x2_t shift2) {
  fp32x2_t normed2;
  if constexpr (kIsLayerNorm) {
    normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
  } else {
    normed2 = {v2.x * inv_std, v2.y * inv_std};
  }
  if constexpr (kHasWeight) {
    normed2.x *= w2.x;
    normed2.y *= w2.y;
  }
  if constexpr (kIsLayerNorm && kHasBias) {
    normed2.x += b2.x;
    normed2.y += b2.y;
  }
  return {
    normed2.x * (1.0f + scale2.x) + shift2.x,
    normed2.y * (1.0f + scale2.y) + shift2.y,
  };
}

template <typename T, bool kIsLayerNorm, bool kFastBSFD, bool kHasWeight, bool kHasBias, int kSpecialMode, bool kSingleFragment = false>
__global__ void fused_scale_residual_norm_scale_shift_kernel(
    const T* __restrict__ residual,
    const int64_t residual_stride0,
    const int64_t residual_stride1,
    const int64_t residual_stride2,
    const T* __restrict__ x,
    const int64_t x_stride0,
    const int64_t x_stride1,
    const int64_t x_stride2,
    const BSFDMeta gate_meta,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const BSFDMeta scale_meta,
    const BSFDMeta shift_meta,
    T* __restrict__ y,
    const int64_t y_stride0,
    const int64_t y_stride1,
    const int64_t y_stride2,
    T* __restrict__ residual_out,
    const int64_t residual_out_stride0,
    const int64_t residual_out_stride1,
    const int64_t residual_out_stride2,
    const int B,
    const int S,
    const int D,
    const float eps) {
  __shared__ float smem_sum[32];
  __shared__ float smem_sumsq[32];
  __shared__ float smem_mean;
  __shared__ float smem_inv_std;

  const int token = static_cast<int>(blockIdx.x);
  if (token >= B * S) {
    return;
  }
  const int b = token / S;
  const int s = token % S;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int num_warps = static_cast<int>(blockDim.x >> 5);

  const int64_t residual_base = static_cast<int64_t>(b) * residual_stride0 + static_cast<int64_t>(s) * residual_stride1;
  const int64_t x_base = static_cast<int64_t>(b) * x_stride0 + static_cast<int64_t>(s) * x_stride1;
  const int64_t residual_out_base =
      static_cast<int64_t>(b) * residual_out_stride0 + static_cast<int64_t>(s) * residual_out_stride1;
  int64_t gate_token_base = 0;
  if constexpr (kFastBSFD) {
    gate_token_base = static_cast<int64_t>(b) * gate_meta.stride0 + static_cast<int64_t>(s) * gate_meta.stride1;
  } else if (gate_meta.mode == kTensorBSD) {
    gate_token_base = static_cast<int64_t>(b) * gate_meta.stride0 + static_cast<int64_t>(s) * gate_meta.stride1;
  }

  float sum = 0.0f;
  float sum_sq = 0.0f;
  constexpr int kMaxCachedIters =
      (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) ? kMaxCachedItersPacked : kMaxCachedItersScalar;
  constexpr bool kIsPacked = std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>;
  fp32x2_t cached_v2[kIsPacked ? kMaxCachedItersPacked : 1];
  float cached_v[kIsPacked ? 1 : kMaxCachedItersScalar];
  if constexpr (kFastBSFD && (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>)) {
    using PackT = packed_t<T>;
    const PackT* residual_pack = reinterpret_cast<const PackT*>(residual + residual_base);
    const PackT* x_pack = reinterpret_cast<const PackT*>(x + x_base);
    const PackT* gate_pack = reinterpret_cast<const PackT*>(static_cast<const T*>(gate_meta.ptr) + gate_token_base);
    PackT* residual_out_pack = reinterpret_cast<PackT*>(residual_out + residual_out_base);
    const int D2 = D >> 1;
    const int D4 = D2 >> 2;
    using VecPackT = device::AlignedVector<PackT, 4>;
    int cache_idx = 0;
    for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
      VecPackT rv4;
      rv4.load(residual_pack, d4);
      VecPackT xv4;
      xv4.load(x_pack, d4);
      VecPackT gv4;
      gv4.load(gate_pack, d4);
      VecPackT out4;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
        const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
        const fp32x2_t gv2 = device::cast<fp32x2_t>(gv4[i]);
        const fp32x2_t v2 = {rv2.x + gv2.x * xv2.x, rv2.y + gv2.y * xv2.y};
        out4[i] = device::cast<PackT, fp32x2_t>(v2);
        cached_v2[cache_idx++] = v2;
        if constexpr (kIsLayerNorm) {
          sum += v2.x + v2.y;
        }
        sum_sq += v2.x * v2.x + v2.y * v2.y;
      }
      out4.store(residual_out_pack, d4);
    }
  } else if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
    using PackT = packed_t<T>;
    const PackT* residual_pack = reinterpret_cast<const PackT*>(residual + residual_base);
    const PackT* x_pack = reinterpret_cast<const PackT*>(x + x_base);
    PackT* residual_out_pack = reinterpret_cast<PackT*>(residual_out + residual_out_base);
    const int D2 = D >> 1;
    int cache_idx = 0;

    if constexpr (
        kSpecialMode == kResidualGateOneScaleShiftBSD || kSpecialMode == kResidualGateOneScaleShiftBSDFP32 ||
        kSpecialMode == kResidualGateOneScaleShiftBSDPacked ||
        kSpecialMode == kResidualGateScalarScaleShiftBSD) {
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int D4 = D2 >> 2;
      if constexpr (
          kSpecialMode == kResidualGateOneScaleShiftBSD || kSpecialMode == kResidualGateOneScaleShiftBSDFP32 ||
          kSpecialMode == kResidualGateOneScaleShiftBSDPacked) {
        if (kSingleFragment || tid < D4) {
          const int d4 = tid;
          VecPackT rv4;
          rv4.load(residual_pack, d4);
          VecPackT xv4;
          xv4.load(x_pack, d4);
          VecPackT out4;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
            const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
            const fp32x2_t v2 = {rv2.x + xv2.x, rv2.y + xv2.y};
            out4[i] = device::cast<PackT, fp32x2_t>(v2);
            cached_v2[i] = v2;
            if constexpr (kIsLayerNorm) {
              sum += v2.x + v2.y;
            }
            sum_sq += v2.x * v2.x + v2.y * v2.y;
          }
          out4.store(residual_out_pack, d4);
        }
      } else {
        const float gate_scalar = load_meta_scalar(gate_meta);
        if (gate_scalar == 1.0f) {
          for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
            VecPackT rv4;
            rv4.load(residual_pack, d4);
            VecPackT xv4;
            xv4.load(x_pack, d4);
            VecPackT out4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
              const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
              const fp32x2_t v2 = {rv2.x + xv2.x, rv2.y + xv2.y};
              out4[i] = device::cast<PackT, fp32x2_t>(v2);
              cached_v2[cache_idx++] = v2;
              if constexpr (kIsLayerNorm) {
                sum += v2.x + v2.y;
              }
              sum_sq += v2.x * v2.x + v2.y * v2.y;
            }
            out4.store(residual_out_pack, d4);
          }
        } else {
          const fp32x2_t gate2 = {gate_scalar, gate_scalar};
          for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
            VecPackT rv4;
            rv4.load(residual_pack, d4);
            VecPackT xv4;
            xv4.load(x_pack, d4);
            VecPackT out4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
              const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
              const fp32x2_t v2 = {
                  rv2.x + gate2.x * xv2.x,
                  rv2.y + gate2.y * xv2.y,
              };
              out4[i] = device::cast<PackT, fp32x2_t>(v2);
              cached_v2[cache_idx++] = v2;
              if constexpr (kIsLayerNorm) {
                sum += v2.x + v2.y;
              }
              sum_sq += v2.x * v2.x + v2.y * v2.y;
            }
            out4.store(residual_out_pack, d4);
          }
        }
      }
    } else if constexpr (kSpecialMode == kResidualGateBSDScaleShiftScalar) {
      const PackT* gate_pack = reinterpret_cast<const PackT*>(static_cast<const T*>(gate_meta.ptr) + gate_token_base);
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int D4 = D2 >> 2;
      if (kSingleFragment || tid < D4) {
        const int d4 = tid;
        VecPackT rv4;
        rv4.load(residual_pack, d4);
        VecPackT xv4;
        xv4.load(x_pack, d4);
        VecPackT gv4;
        gv4.load(gate_pack, d4);
        VecPackT out4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
          const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
          const fp32x2_t gv2 = device::cast<fp32x2_t>(gv4[i]);
          const fp32x2_t v2 = {
              rv2.x + gv2.x * xv2.x,
              rv2.y + gv2.y * xv2.y,
          };
          out4[i] = device::cast<PackT, fp32x2_t>(v2);
          cached_v2[i] = v2;
          if constexpr (kIsLayerNorm) {
            sum += v2.x + v2.y;
          }
          sum_sq += v2.x * v2.x + v2.y * v2.y;
        }
        out4.store(residual_out_pack, d4);
      }
    } else {
      const bool gate_is_scalar = gate_meta.mode == kScalarTensor;
      const bool gate_is_bsd_pack =
          (gate_meta.mode == kTensorBSD) && (gate_meta.stride2 == 1) && (gate_meta.dtype == bsfd_dtype_code<T>());
      const bool gate_is_bsd_fp32 =
          (gate_meta.mode == kTensorBSD) && (gate_meta.stride2 == 1) && (gate_meta.dtype == kBSFDDTypeFP32);
      const float gate_scalar = gate_is_scalar ? load_meta_scalar(gate_meta) : 0.0f;
      const bool gate_scalar_is_one = gate_is_scalar && (gate_scalar == 1.0f);
      const PackT* gate_pack =
          gate_is_bsd_pack ? reinterpret_cast<const PackT*>(static_cast<const T*>(gate_meta.ptr) + gate_token_base)
                           : nullptr;
      const float* gate_fp32 = gate_is_bsd_fp32 ? (static_cast<const float*>(gate_meta.ptr) + gate_token_base) : nullptr;
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int D4 = D2 >> 2;

      if (gate_scalar_is_one) {
        if (kSingleFragment || tid < D4) {
          const int d4 = tid;
          VecPackT rv4;
          rv4.load(residual_pack, d4);
          VecPackT xv4;
          xv4.load(x_pack, d4);
          VecPackT out4;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
            const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
            const fp32x2_t v2 = {rv2.x + xv2.x, rv2.y + xv2.y};
            out4[i] = device::cast<PackT, fp32x2_t>(v2);
            cached_v2[i] = v2;
            if constexpr (kIsLayerNorm) {
              sum += v2.x + v2.y;
            }
            sum_sq += v2.x * v2.x + v2.y * v2.y;
          }
          out4.store(residual_out_pack, d4);
        }
      } else if (gate_is_scalar) {
        const fp32x2_t gate2 = {gate_scalar, gate_scalar};
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT rv4;
          rv4.load(residual_pack, d4);
          VecPackT xv4;
          xv4.load(x_pack, d4);
          VecPackT out4;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
            const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
            const fp32x2_t v2 = {
                rv2.x + gate2.x * xv2.x,
                rv2.y + gate2.y * xv2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(v2);
            cached_v2[cache_idx++] = v2;
            if constexpr (kIsLayerNorm) {
              sum += v2.x + v2.y;
            }
            sum_sq += v2.x * v2.x + v2.y * v2.y;
          }
          out4.store(residual_out_pack, d4);
        }
      } else if (gate_pack != nullptr) {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT rv4;
          rv4.load(residual_pack, d4);
          VecPackT xv4;
          xv4.load(x_pack, d4);
          VecPackT gv4;
          gv4.load(gate_pack, d4);
          VecPackT out4;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
            const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
            const fp32x2_t gv2 = device::cast<fp32x2_t>(gv4[i]);
            const fp32x2_t v2 = {
                rv2.x + gv2.x * xv2.x,
                rv2.y + gv2.y * xv2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(v2);
            cached_v2[cache_idx++] = v2;
            if constexpr (kIsLayerNorm) {
              sum += v2.x + v2.y;
            }
            sum_sq += v2.x * v2.x + v2.y * v2.y;
          }
          out4.store(residual_out_pack, d4);
        }
      } else if (gate_fp32 != nullptr) {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT rv4;
          rv4.load(residual_pack, d4);
          VecPackT xv4;
          xv4.load(x_pack, d4);
          VecPackT out4;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
            const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
            const int d2 = (d4 << 2) + i;
            const fp32x2_t gv2 = {gate_fp32[d2 << 1], gate_fp32[(d2 << 1) + 1]};
            const fp32x2_t v2 = {
                rv2.x + gv2.x * xv2.x,
                rv2.y + gv2.y * xv2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(v2);
            cached_v2[cache_idx++] = v2;
            if constexpr (kIsLayerNorm) {
              sum += v2.x + v2.y;
            }
            sum_sq += v2.x * v2.x + v2.y * v2.y;
          }
          out4.store(residual_out_pack, d4);
        }
      } else {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT rv4;
          rv4.load(residual_pack, d4);
          VecPackT xv4;
          xv4.load(x_pack, d4);
          VecPackT out4;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t rv2 = device::cast<fp32x2_t>(rv4[i]);
            const fp32x2_t xv2 = device::cast<fp32x2_t>(xv4[i]);
            const int d2 = (d4 << 2) + i;
            const fp32x2_t gv2 = load_bsfd_pair(gate_meta, b, s, d2, S);
            const fp32x2_t v2 = {
                rv2.x + gv2.x * xv2.x,
                rv2.y + gv2.y * xv2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(v2);
            cached_v2[cache_idx++] = v2;
            if constexpr (kIsLayerNorm) {
              sum += v2.x + v2.y;
            }
            sum_sq += v2.x * v2.x + v2.y * v2.y;
          }
          out4.store(residual_out_pack, d4);
        }
      }
    }
  } else {
    int cache_idx = 0;
    for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
      const float rv = device::cast<float>(residual[residual_base + static_cast<int64_t>(d) * residual_stride2]);
      const float xv = device::cast<float>(x[x_base + static_cast<int64_t>(d) * x_stride2]);
      const float gv =
          kFastBSFD ? load_bsd_value(gate_meta, gate_token_base, d) : load_bsfd_value(gate_meta, b, s, d, S);
      const float v = rv + gv * xv;
      const T v_t = device::cast<T>(v);
      cached_v[cache_idx++] = v;
      residual_out[residual_out_base + static_cast<int64_t>(d) * residual_out_stride2] = v_t;
      if constexpr (kIsLayerNorm) {
        sum += v;
      }
      sum_sq += v * v;
    }
  }

  if constexpr (kIsLayerNorm) {
    sum = warp_reduce_sum(sum);
  }
  sum_sq = warp_reduce_sum(sum_sq);

  if (lane == 0) {
    if constexpr (kIsLayerNorm) {
      smem_sum[warp_id] = sum;
    }
    smem_sumsq[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = 0.0f;
    float block_sumsq = 0.0f;
    if (lane < num_warps) {
      if constexpr (kIsLayerNorm) {
        block_sum = smem_sum[lane];
      }
      block_sumsq = smem_sumsq[lane];
    }
    if constexpr (kIsLayerNorm) {
      block_sum = warp_reduce_sum(block_sum);
    }
    block_sumsq = warp_reduce_sum(block_sumsq);
    if (lane == 0) {
      float mean = 0.0f;
      float inv_std = 0.0f;
      if constexpr (kIsLayerNorm) {
        mean = block_sum / static_cast<float>(D);
        const float var = fmaxf(block_sumsq / static_cast<float>(D) - mean * mean, 0.0f);
        inv_std = rsqrtf(var + eps);
      } else {
        inv_std = rsqrtf(block_sumsq / static_cast<float>(D) + eps);
      }
      smem_mean = mean;
      smem_inv_std = inv_std;
    }
  }
  __syncthreads();

  const int64_t y_base = static_cast<int64_t>(b) * y_stride0 + static_cast<int64_t>(s) * y_stride1;
  const float mean = smem_mean;
  const float inv_std = smem_inv_std;
  int64_t scale_token_base = 0;
  int64_t shift_token_base = 0;
  if constexpr (kFastBSFD) {
    scale_token_base = static_cast<int64_t>(b) * scale_meta.stride0 + static_cast<int64_t>(s) * scale_meta.stride1;
    shift_token_base = static_cast<int64_t>(b) * shift_meta.stride0 + static_cast<int64_t>(s) * shift_meta.stride1;
  } else {
    if (scale_meta.mode == kTensorBSD) {
      scale_token_base = static_cast<int64_t>(b) * scale_meta.stride0 + static_cast<int64_t>(s) * scale_meta.stride1;
    }
    if (shift_meta.mode == kTensorBSD) {
      shift_token_base = static_cast<int64_t>(b) * shift_meta.stride0 + static_cast<int64_t>(s) * shift_meta.stride1;
    }
  }

  if constexpr (kFastBSFD && (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>)) {
    using PackT = packed_t<T>;
    const PackT* weight_pack = reinterpret_cast<const PackT*>(weight);
    const PackT* bias_pack = reinterpret_cast<const PackT*>(bias);
    const PackT* scale_pack = reinterpret_cast<const PackT*>(static_cast<const T*>(scale_meta.ptr) + scale_token_base);
    const PackT* shift_pack = reinterpret_cast<const PackT*>(static_cast<const T*>(shift_meta.ptr) + shift_token_base);
    PackT* y_pack = reinterpret_cast<PackT*>(y + y_base);
    const int D2 = D >> 1;
    const int D4 = D2 >> 2;
    using VecPackT = device::AlignedVector<PackT, 4>;
    int cache_idx = 0;
    for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
      VecPackT sc4;
      sc4.load(scale_pack, d4);
      VecPackT sh4;
      sh4.load(shift_pack, d4);
      VecPackT out4;
      VecPackT w4;
      VecPackT b4;
      if constexpr (kHasWeight) {
        w4.load(weight_pack, d4);
      }
      if constexpr (kIsLayerNorm) {
        if constexpr (kHasBias) {
          b4.load(bias_pack, d4);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const fp32x2_t v2 = cached_v2[cache_idx + i];

        fp32x2_t normed2;
        if constexpr (kIsLayerNorm) {
          normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
        } else {
          normed2 = {v2.x * inv_std, v2.y * inv_std};
        }
        if constexpr (kHasWeight) {
          const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
          normed2.x *= w2.x;
          normed2.y *= w2.y;
        }
        if constexpr (kIsLayerNorm) {
          if constexpr (kHasBias) {
            const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
            normed2.x += b2.x;
            normed2.y += b2.y;
          }
        }

        const fp32x2_t scale2 = device::cast<fp32x2_t>(sc4[i]);
        const fp32x2_t shift2 = device::cast<fp32x2_t>(sh4[i]);
        const fp32x2_t out2 = {
            normed2.x * (1.0f + scale2.x) + shift2.x,
            normed2.y * (1.0f + scale2.y) + shift2.y,
        };
        out4[i] = device::cast<PackT, fp32x2_t>(out2);
      }
      cache_idx += 4;
      out4.store(y_pack, d4);
    }
  } else if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
    using PackT = packed_t<T>;
    const PackT* weight_pack = reinterpret_cast<const PackT*>(weight);
    const PackT* bias_pack = reinterpret_cast<const PackT*>(bias);
    PackT* y_pack = reinterpret_cast<PackT*>(y + y_base);
    const int D2 = D >> 1;
    int cache_idx = 0;

    if constexpr (
        kSpecialMode == kResidualGateOneScaleShiftBSD || kSpecialMode == kResidualGateOneScaleShiftBSDFP32 ||
        kSpecialMode == kResidualGateOneScaleShiftBSDPacked ||
        kSpecialMode == kResidualGateScalarScaleShiftBSD) {
      constexpr bool kGateOneFP32Mode = (kSpecialMode == kResidualGateOneScaleShiftBSDFP32);
      constexpr bool kGateOnePackedMode = (kSpecialMode == kResidualGateOneScaleShiftBSDPacked);
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int D4 = D2 >> 2;
      if constexpr (kGateOnePackedMode) {
        const PackT* sp = reinterpret_cast<const PackT*>(static_cast<const T*>(scale_meta.ptr) + scale_token_base);
        const PackT* hp = reinterpret_cast<const PackT*>(static_cast<const T*>(shift_meta.ptr) + shift_token_base);
        if (kSingleFragment || tid < D4) {
          const int d4 = tid;
          VecPackT sc4; sc4.load(sp, d4);
          VecPackT sh4; sh4.load(hp, d4);
          VecPackT out4;
          VecPackT w4, b4;
          if constexpr (kHasWeight) { w4.load(weight_pack, d4); }
          if constexpr (kIsLayerNorm && kHasBias) { b4.load(bias_pack, d4); }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[i];
            const fp32x2_t w2 = kHasWeight ? device::cast<fp32x2_t>(w4[i]) : fp32x2_t{};
            const fp32x2_t b2 = (kIsLayerNorm && kHasBias) ? device::cast<fp32x2_t>(b4[i]) : fp32x2_t{};
            const fp32x2_t scale2 = device::cast<fp32x2_t>(sc4[i]);
            const fp32x2_t shift2 = device::cast<fp32x2_t>(sh4[i]);
            const fp32x2_t out2 = apply_norm_scale_shift_packed<kIsLayerNorm, kHasWeight, kHasBias>(
                v2, mean, inv_std, w2, b2, scale2, shift2);
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
          out4.store(y_pack, d4);
        }
      } else {
      const bool scale_shift_pack = ((!kGateOneFP32Mode) && (scale_meta.dtype == bsfd_dtype_code<T>()) &&
                                    (shift_meta.dtype == bsfd_dtype_code<T>()));
      const bool scale_shift_fp32 =
          kGateOneFP32Mode || ((scale_meta.dtype == kBSFDDTypeFP32) && (shift_meta.dtype == kBSFDDTypeFP32));
      const PackT* scale_pack =
          scale_shift_pack ? reinterpret_cast<const PackT*>(static_cast<const T*>(scale_meta.ptr) + scale_token_base)
                           : nullptr;
      const PackT* shift_pack =
          scale_shift_pack ? reinterpret_cast<const PackT*>(static_cast<const T*>(shift_meta.ptr) + shift_token_base)
                           : nullptr;
      const float* scale_fp32 =
          scale_shift_fp32 ? (static_cast<const float*>(scale_meta.ptr) + scale_token_base) : nullptr;
      const float* shift_fp32 =
          scale_shift_fp32 ? (static_cast<const float*>(shift_meta.ptr) + shift_token_base) : nullptr;
      const float4* scale_fp32x4 = scale_shift_fp32 ? reinterpret_cast<const float4*>(scale_fp32) : nullptr;
      const float4* shift_fp32x4 = scale_shift_fp32 ? reinterpret_cast<const float4*>(shift_fp32) : nullptr;
      const float2* scale_fp32x2 = scale_shift_fp32 ? reinterpret_cast<const float2*>(scale_fp32) : nullptr;
      const float2* shift_fp32x2 = scale_shift_fp32 ? reinterpret_cast<const float2*>(shift_fp32) : nullptr;
      if constexpr (
          kSpecialMode == kResidualGateOneScaleShiftBSD || kSpecialMode == kResidualGateOneScaleShiftBSDFP32) {
        if (kSingleFragment || tid < D4) {
          const int d4 = tid;
          VecPackT out4;
          if constexpr (kGateOneFP32Mode && kHasWeight && kHasBias && kIsLayerNorm) {
            const int d4x2 = d4 << 1;
            const float4 sc4_lo = scale_fp32x4[d4x2];
            const float4 sc4_hi = scale_fp32x4[d4x2 + 1];
            const float4 sh4_lo = shift_fp32x4[d4x2];
            const float4 sh4_hi = shift_fp32x4[d4x2 + 1];
            VecPackT w4;
            VecPackT b4;
            w4.load(weight_pack, d4);
            b4.load(bias_pack, d4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const fp32x2_t v2 = cached_v2[i];
              fp32x2_t normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
              const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
              const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
              normed2.x = normed2.x * w2.x + b2.x;
              normed2.y = normed2.y * w2.y + b2.y;
              const fp32x2_t scale2 =
                  i == 0 ? fp32x2_t{sc4_lo.x, sc4_lo.y}
                  : i == 1 ? fp32x2_t{sc4_lo.z, sc4_lo.w}
                  : i == 2 ? fp32x2_t{sc4_hi.x, sc4_hi.y}
                           : fp32x2_t{sc4_hi.z, sc4_hi.w};
              const fp32x2_t shift2 =
                  i == 0 ? fp32x2_t{sh4_lo.x, sh4_lo.y}
                  : i == 1 ? fp32x2_t{sh4_lo.z, sh4_lo.w}
                  : i == 2 ? fp32x2_t{sh4_hi.x, sh4_hi.y}
                           : fp32x2_t{sh4_hi.z, sh4_hi.w};
              const fp32x2_t out2 = {
                  normed2.x * (1.0f + scale2.x) + shift2.x,
                  normed2.y * (1.0f + scale2.y) + shift2.y,
              };
              out4[i] = device::cast<PackT, fp32x2_t>(out2);
            }
          } else if constexpr (kGateOneFP32Mode) {
            const int d4x2 = d4 << 1;
            const float4 sc4_lo = scale_fp32x4[d4x2];
            const float4 sc4_hi = scale_fp32x4[d4x2 + 1];
            const float4 sh4_lo = shift_fp32x4[d4x2];
            const float4 sh4_hi = shift_fp32x4[d4x2 + 1];
            if constexpr (!kHasWeight && !kHasBias) {
              {
                const fp32x2_t v2 = cached_v2[0];
                const fp32x2_t n2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                const fp32x2_t sc2 = {sc4_lo.x, sc4_lo.y};
                const fp32x2_t sh2 = {sh4_lo.x, sh4_lo.y};
                const fp32x2_t out2 = {
                    n2.x * (1.0f + sc2.x) + sh2.x,
                    n2.y * (1.0f + sc2.y) + sh2.y,
                };
                out4[0] = device::cast<PackT, fp32x2_t>(out2);
              }
              {
                const fp32x2_t v2 = cached_v2[1];
                const fp32x2_t n2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                const fp32x2_t sc2 = {sc4_lo.z, sc4_lo.w};
                const fp32x2_t sh2 = {sh4_lo.z, sh4_lo.w};
                const fp32x2_t out2 = {
                    n2.x * (1.0f + sc2.x) + sh2.x,
                    n2.y * (1.0f + sc2.y) + sh2.y,
                };
                out4[1] = device::cast<PackT, fp32x2_t>(out2);
              }
              {
                const fp32x2_t v2 = cached_v2[2];
                const fp32x2_t n2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                const fp32x2_t sc2 = {sc4_hi.x, sc4_hi.y};
                const fp32x2_t sh2 = {sh4_hi.x, sh4_hi.y};
                const fp32x2_t out2 = {
                    n2.x * (1.0f + sc2.x) + sh2.x,
                    n2.y * (1.0f + sc2.y) + sh2.y,
                };
                out4[2] = device::cast<PackT, fp32x2_t>(out2);
              }
              {
                const fp32x2_t v2 = cached_v2[3];
                const fp32x2_t n2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                const fp32x2_t sc2 = {sc4_hi.z, sc4_hi.w};
                const fp32x2_t sh2 = {sh4_hi.z, sh4_hi.w};
                const fp32x2_t out2 = {
                    n2.x * (1.0f + sc2.x) + sh2.x,
                    n2.y * (1.0f + sc2.y) + sh2.y,
                };
                out4[3] = device::cast<PackT, fp32x2_t>(out2);
              }
            } else {
              VecPackT w4;
              VecPackT b4;
              if constexpr (kHasWeight) {
                w4.load(weight_pack, d4);
              }
              if constexpr (kIsLayerNorm) {
                if constexpr (kHasBias) {
                  b4.load(bias_pack, d4);
                }
              }
              {
                const fp32x2_t v2 = cached_v2[0];
                fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                if constexpr (kHasWeight) {
                  const fp32x2_t w2 = device::cast<fp32x2_t>(w4[0]);
                  normed2.x *= w2.x;
                  normed2.y *= w2.y;
                }
                if constexpr (kIsLayerNorm) {
                  if constexpr (kHasBias) {
                    const fp32x2_t b2 = device::cast<fp32x2_t>(b4[0]);
                    normed2.x += b2.x;
                    normed2.y += b2.y;
                  }
                }
                const fp32x2_t scale2 = {sc4_lo.x, sc4_lo.y};
                const fp32x2_t shift2 = {sh4_lo.x, sh4_lo.y};
                const fp32x2_t out2 = {
                    normed2.x * (1.0f + scale2.x) + shift2.x,
                    normed2.y * (1.0f + scale2.y) + shift2.y,
                };
                out4[0] = device::cast<PackT, fp32x2_t>(out2);
              }
              {
                const fp32x2_t v2 = cached_v2[1];
                fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                if constexpr (kHasWeight) {
                  const fp32x2_t w2 = device::cast<fp32x2_t>(w4[1]);
                  normed2.x *= w2.x;
                  normed2.y *= w2.y;
                }
                if constexpr (kIsLayerNorm) {
                  if constexpr (kHasBias) {
                    const fp32x2_t b2 = device::cast<fp32x2_t>(b4[1]);
                    normed2.x += b2.x;
                    normed2.y += b2.y;
                  }
                }
                const fp32x2_t scale2 = {sc4_lo.z, sc4_lo.w};
                const fp32x2_t shift2 = {sh4_lo.z, sh4_lo.w};
                const fp32x2_t out2 = {
                    normed2.x * (1.0f + scale2.x) + shift2.x,
                    normed2.y * (1.0f + scale2.y) + shift2.y,
                };
                out4[1] = device::cast<PackT, fp32x2_t>(out2);
              }
              {
                const fp32x2_t v2 = cached_v2[2];
                fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                if constexpr (kHasWeight) {
                  const fp32x2_t w2 = device::cast<fp32x2_t>(w4[2]);
                  normed2.x *= w2.x;
                  normed2.y *= w2.y;
                }
                if constexpr (kIsLayerNorm) {
                  if constexpr (kHasBias) {
                    const fp32x2_t b2 = device::cast<fp32x2_t>(b4[2]);
                    normed2.x += b2.x;
                    normed2.y += b2.y;
                  }
                }
                const fp32x2_t scale2 = {sc4_hi.x, sc4_hi.y};
                const fp32x2_t shift2 = {sh4_hi.x, sh4_hi.y};
                const fp32x2_t out2 = {
                    normed2.x * (1.0f + scale2.x) + shift2.x,
                    normed2.y * (1.0f + scale2.y) + shift2.y,
                };
                out4[2] = device::cast<PackT, fp32x2_t>(out2);
              }
              {
                const fp32x2_t v2 = cached_v2[3];
                fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                 : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                if constexpr (kHasWeight) {
                  const fp32x2_t w2 = device::cast<fp32x2_t>(w4[3]);
                  normed2.x *= w2.x;
                  normed2.y *= w2.y;
                }
                if constexpr (kIsLayerNorm) {
                  if constexpr (kHasBias) {
                    const fp32x2_t b2 = device::cast<fp32x2_t>(b4[3]);
                    normed2.x += b2.x;
                    normed2.y += b2.y;
                  }
                }
                const fp32x2_t scale2 = {sc4_hi.z, sc4_hi.w};
                const fp32x2_t shift2 = {sh4_hi.z, sh4_hi.w};
                const fp32x2_t out2 = {
                    normed2.x * (1.0f + scale2.x) + shift2.x,
                    normed2.y * (1.0f + scale2.y) + shift2.y,
                };
                out4[3] = device::cast<PackT, fp32x2_t>(out2);
              }
            }
          } else {
            if (scale_shift_pack) {
              VecPackT sc4;
              VecPackT sh4;
              sc4.load(scale_pack, d4);
              sh4.load(shift_pack, d4);
              if constexpr (!kHasWeight && !kHasBias) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  const fp32x2_t v2 = cached_v2[i];
                  const fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                        : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                  const fp32x2_t scale2 = device::cast<fp32x2_t>(sc4[i]);
                  const fp32x2_t shift2 = device::cast<fp32x2_t>(sh4[i]);
                  const fp32x2_t out2 = {
                      normed2.x * (1.0f + scale2.x) + shift2.x,
                      normed2.y * (1.0f + scale2.y) + shift2.y,
                  };
                  out4[i] = device::cast<PackT, fp32x2_t>(out2);
                }
              } else {
                VecPackT w4;
                VecPackT b4;
                if constexpr (kHasWeight) {
                  w4.load(weight_pack, d4);
                }
                if constexpr (kIsLayerNorm) {
                  if constexpr (kHasBias) {
                    b4.load(bias_pack, d4);
                  }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  const fp32x2_t v2 = cached_v2[i];
                  fp32x2_t normed2;
                  if constexpr (kIsLayerNorm) {
                    normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
                  } else {
                    normed2 = {v2.x * inv_std, v2.y * inv_std};
                  }
                  if constexpr (kHasWeight) {
                    const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
                    normed2.x *= w2.x;
                    normed2.y *= w2.y;
                  }
                  if constexpr (kIsLayerNorm) {
                    if constexpr (kHasBias) {
                      const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                      normed2.x += b2.x;
                      normed2.y += b2.y;
                    }
                  }
                  const fp32x2_t scale2 = device::cast<fp32x2_t>(sc4[i]);
                  const fp32x2_t shift2 = device::cast<fp32x2_t>(sh4[i]);
                  const fp32x2_t out2 = {
                      normed2.x * (1.0f + scale2.x) + shift2.x,
                      normed2.y * (1.0f + scale2.y) + shift2.y,
                  };
                  out4[i] = device::cast<PackT, fp32x2_t>(out2);
                }
              }
            } else {
              if constexpr (!kHasWeight && !kHasBias) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  const fp32x2_t v2 = cached_v2[i];
                  const fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                        : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
                  const int d2 = (d4 << 2) + i;
                  const int d0 = d2 << 1;
                  const int d1 = d0 + 1;
                  const fp32x2_t scale2 =
                      scale_shift_fp32 ? fp32x2_t{scale_fp32x2[d2].x, scale_fp32x2[d2].y}
                                       : fp32x2_t{
                                             load_bsd_value(scale_meta, scale_token_base, d0),
                                             load_bsd_value(scale_meta, scale_token_base, d1)};
                  const fp32x2_t shift2 =
                      scale_shift_fp32 ? fp32x2_t{shift_fp32x2[d2].x, shift_fp32x2[d2].y}
                                       : fp32x2_t{
                                             load_bsd_value(shift_meta, shift_token_base, d0),
                                             load_bsd_value(shift_meta, shift_token_base, d1)};
                  const fp32x2_t out2 = {
                      normed2.x * (1.0f + scale2.x) + shift2.x,
                      normed2.y * (1.0f + scale2.y) + shift2.y,
                  };
                  out4[i] = device::cast<PackT, fp32x2_t>(out2);
                }
              } else {
                VecPackT w4;
                VecPackT b4;
                if constexpr (kHasWeight) {
                  w4.load(weight_pack, d4);
                }
                if constexpr (kIsLayerNorm) {
                  if constexpr (kHasBias) {
                    b4.load(bias_pack, d4);
                  }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  const fp32x2_t v2 = cached_v2[i];
                  fp32x2_t normed2;
                  if constexpr (kIsLayerNorm) {
                    normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
                  } else {
                    normed2 = {v2.x * inv_std, v2.y * inv_std};
                  }
                  if constexpr (kHasWeight) {
                    const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
                    normed2.x *= w2.x;
                    normed2.y *= w2.y;
                  }
                  if constexpr (kIsLayerNorm) {
                    if constexpr (kHasBias) {
                      const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                      normed2.x += b2.x;
                      normed2.y += b2.y;
                    }
                  }
                  const int d2 = (d4 << 2) + i;
                  const int d0 = d2 << 1;
                  const int d1 = d0 + 1;
                  const fp32x2_t scale2 =
                      scale_shift_fp32 ? fp32x2_t{scale_fp32x2[d2].x, scale_fp32x2[d2].y}
                                       : fp32x2_t{
                                             load_bsd_value(scale_meta, scale_token_base, d0),
                                             load_bsd_value(scale_meta, scale_token_base, d1)};
                  const fp32x2_t shift2 =
                      scale_shift_fp32 ? fp32x2_t{shift_fp32x2[d2].x, shift_fp32x2[d2].y}
                                       : fp32x2_t{
                                             load_bsd_value(shift_meta, shift_token_base, d0),
                                             load_bsd_value(shift_meta, shift_token_base, d1)};
                  const fp32x2_t out2 = {
                      normed2.x * (1.0f + scale2.x) + shift2.x,
                      normed2.y * (1.0f + scale2.y) + shift2.y,
                  };
                  out4[i] = device::cast<PackT, fp32x2_t>(out2);
                }
              }
            }
          }
          out4.store(y_pack, d4);
        }
      } else {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT sc4;
          VecPackT sh4;
          if (scale_shift_pack) {
            sc4.load(scale_pack, d4);
            sh4.load(shift_pack, d4);
          }
          VecPackT out4;
          VecPackT w4;
          VecPackT b4;
          if constexpr (kHasWeight) {
            w4.load(weight_pack, d4);
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              b4.load(bias_pack, d4);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[cache_idx + i];

            fp32x2_t normed2;
            if constexpr (kIsLayerNorm) {
              normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
            } else {
              normed2 = {v2.x * inv_std, v2.y * inv_std};
            }
            if constexpr (kHasWeight) {
              const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
              normed2.x *= w2.x;
              normed2.y *= w2.y;
            }
            if constexpr (kIsLayerNorm) {
              if constexpr (kHasBias) {
                const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                normed2.x += b2.x;
                normed2.y += b2.y;
              }
            }

            const int d2 = (d4 << 2) + i;
            const int d0 = d2 << 1;
            const int d1 = d0 + 1;
            const fp32x2_t scale2 = scale_shift_pack
                                        ? device::cast<fp32x2_t>(sc4[i])
                                        : (scale_shift_fp32 ? fp32x2_t{scale_fp32[d0], scale_fp32[d1]}
                                                            : fp32x2_t{
                                                                  load_bsd_value(scale_meta, scale_token_base, d0),
                                                                  load_bsd_value(scale_meta, scale_token_base, d1)});
            const fp32x2_t shift2 = scale_shift_pack
                                        ? device::cast<fp32x2_t>(sh4[i])
                                        : (scale_shift_fp32 ? fp32x2_t{shift_fp32[d0], shift_fp32[d1]}
                                                            : fp32x2_t{
                                                                  load_bsd_value(shift_meta, shift_token_base, d0),
                                                                  load_bsd_value(shift_meta, shift_token_base, d1)});
            const fp32x2_t out2 = {
                normed2.x * (1.0f + scale2.x) + shift2.x,
                normed2.y * (1.0f + scale2.y) + shift2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
          cache_idx += 4;
          out4.store(y_pack, d4);
        }
      }
      } // kGateOnePackedMode else
    } else if constexpr (kSpecialMode == kResidualGateBSDScaleShiftScalar) {
      const float scale_scalar = load_meta_scalar(scale_meta);
      const float shift_scalar = load_meta_scalar(shift_meta);
      const fp32x2_t one_plus_scale2 = {1.0f + scale_scalar, 1.0f + scale_scalar};
      const fp32x2_t shift2 = {shift_scalar, shift_scalar};
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int D4 = D2 >> 2;
      if (kSingleFragment || tid < D4) {
        const int d4 = tid;
        VecPackT out4;
        if constexpr (!kHasWeight && !kHasBias) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[i];
            const fp32x2_t normed2 = kIsLayerNorm ? fp32x2_t{(v2.x - mean) * inv_std, (v2.y - mean) * inv_std}
                                                  : fp32x2_t{v2.x * inv_std, v2.y * inv_std};
            const fp32x2_t out2 = {
                normed2.x * one_plus_scale2.x + shift2.x,
                normed2.y * one_plus_scale2.y + shift2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
        } else {
          VecPackT w4;
          VecPackT b4;
          if constexpr (kHasWeight) {
            w4.load(weight_pack, d4);
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              b4.load(bias_pack, d4);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[i];

            fp32x2_t normed2;
            if constexpr (kIsLayerNorm) {
              normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
            } else {
              normed2 = {v2.x * inv_std, v2.y * inv_std};
            }
            if constexpr (kHasWeight) {
              const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
              normed2.x *= w2.x;
              normed2.y *= w2.y;
            }
            if constexpr (kIsLayerNorm) {
              if constexpr (kHasBias) {
                const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                normed2.x += b2.x;
                normed2.y += b2.y;
              }
            }

            const fp32x2_t out2 = {
                normed2.x * one_plus_scale2.x + shift2.x,
                normed2.y * one_plus_scale2.y + shift2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
        }
        out4.store(y_pack, d4);
      }
    } else if constexpr (kSpecialMode == kResidualGateScalarScaleShiftBSDAffine) {
      const bool scale_shift_pack = (scale_meta.dtype == bsfd_dtype_code<T>()) && (shift_meta.dtype == bsfd_dtype_code<T>()) &&
                                    (scale_meta.stride2 == 1) && (shift_meta.stride2 == 1);
      const PackT* scale_pack =
          scale_shift_pack ? reinterpret_cast<const PackT*>(static_cast<const T*>(scale_meta.ptr) + scale_token_base)
                           : nullptr;
      const PackT* shift_pack =
          scale_shift_pack ? reinterpret_cast<const PackT*>(static_cast<const T*>(shift_meta.ptr) + shift_token_base)
                           : nullptr;
      const bool scale_shift_fp32 = (scale_meta.dtype == kBSFDDTypeFP32) && (shift_meta.dtype == kBSFDDTypeFP32) &&
                                    (scale_meta.stride2 == 1) && (shift_meta.stride2 == 1);
      const float* scale_fp32 = scale_shift_fp32 ? (static_cast<const float*>(scale_meta.ptr) + scale_token_base) : nullptr;
      const float* shift_fp32 = scale_shift_fp32 ? (static_cast<const float*>(shift_meta.ptr) + shift_token_base) : nullptr;
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int D4 = D2 >> 2;
      for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
        VecPackT out4;
        VecPackT w4;
        VecPackT b4;
        if constexpr (kHasWeight) {
          w4.load(weight_pack, d4);
        }
        if constexpr (kIsLayerNorm) {
          if constexpr (kHasBias) {
            b4.load(bias_pack, d4);
          }
        }
        VecPackT sc4;
        VecPackT sh4;
        if (scale_shift_pack) {
          sc4.load(scale_pack, d4);
          sh4.load(shift_pack, d4);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const fp32x2_t v2 = cached_v2[cache_idx + i];
          fp32x2_t normed2;
          if constexpr (kIsLayerNorm) {
            normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
          } else {
            normed2 = {v2.x * inv_std, v2.y * inv_std};
          }
          if constexpr (kHasWeight) {
            const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
            normed2.x *= w2.x;
            normed2.y *= w2.y;
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
              normed2.x += b2.x;
              normed2.y += b2.y;
            }
          }
          const int d2 = (d4 << 2) + i;
          const int d0 = d2 << 1;
          const int d1 = d0 + 1;
          const fp32x2_t scale2 = scale_shift_pack
                                      ? device::cast<fp32x2_t>(sc4[i])
                                      : fp32x2_t{scale_fp32[d0], scale_fp32[d1]};
          const fp32x2_t shift2 = scale_shift_pack
                                      ? device::cast<fp32x2_t>(sh4[i])
                                      : fp32x2_t{shift_fp32[d0], shift_fp32[d1]};
          const fp32x2_t out2 = {
              normed2.x * (1.0f + scale2.x) + shift2.x,
              normed2.y * (1.0f + scale2.y) + shift2.y,
          };
          out4[i] = device::cast<PackT, fp32x2_t>(out2);
        }
        cache_idx += 4;
        out4.store(y_pack, d4);
      }
    } else {
      const bool scale_is_scalar = scale_meta.mode == kScalarTensor;
      const bool shift_is_scalar = shift_meta.mode == kScalarTensor;
      const bool scale_is_bsd_pack =
          (scale_meta.mode == kTensorBSD) && (scale_meta.stride2 == 1) && (scale_meta.dtype == bsfd_dtype_code<T>());
      const bool shift_is_bsd_pack =
          (shift_meta.mode == kTensorBSD) && (shift_meta.stride2 == 1) && (shift_meta.dtype == bsfd_dtype_code<T>());

      const float scale_scalar = scale_is_scalar ? load_meta_scalar(scale_meta) : 0.0f;
      const float shift_scalar = shift_is_scalar ? load_meta_scalar(shift_meta) : 0.0f;
      const fp32x2_t scale_scalar2 = {scale_scalar, scale_scalar};
      const fp32x2_t shift_scalar2 = {shift_scalar, shift_scalar};

      const PackT* scale_pack =
          scale_is_bsd_pack ? reinterpret_cast<const PackT*>(static_cast<const T*>(scale_meta.ptr) + scale_token_base)
                            : nullptr;
      const PackT* shift_pack =
          shift_is_bsd_pack ? reinterpret_cast<const PackT*>(static_cast<const T*>(shift_meta.ptr) + shift_token_base)
                            : nullptr;
      const bool scale_shift_fp32_bsd =
          (scale_meta.mode == kTensorBSD) && (shift_meta.mode == kTensorBSD) && (scale_meta.stride2 == 1) &&
          (shift_meta.stride2 == 1) && (scale_meta.dtype == kBSFDDTypeFP32) && (shift_meta.dtype == kBSFDDTypeFP32);
      const float* scale_fp32_bsd =
          scale_shift_fp32_bsd ? (static_cast<const float*>(scale_meta.ptr) + scale_token_base) : nullptr;
      const float* shift_fp32_bsd =
          scale_shift_fp32_bsd ? (static_cast<const float*>(shift_meta.ptr) + shift_token_base) : nullptr;
      using VecPackT = device::AlignedVector<PackT, 4>;
      const int D4 = D2 >> 2;

      if (scale_is_scalar && shift_is_scalar) {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT out4;
          VecPackT w4;
          VecPackT b4;
          if constexpr (kHasWeight) {
            w4.load(weight_pack, d4);
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              b4.load(bias_pack, d4);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[cache_idx + i];

            fp32x2_t normed2;
            if constexpr (kIsLayerNorm) {
              normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
            } else {
              normed2 = {v2.x * inv_std, v2.y * inv_std};
            }
            if constexpr (kHasWeight) {
              const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
              normed2.x *= w2.x;
              normed2.y *= w2.y;
            }
            if constexpr (kIsLayerNorm) {
              if constexpr (kHasBias) {
                const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                normed2.x += b2.x;
                normed2.y += b2.y;
              }
            }

            const fp32x2_t out2 = {
                normed2.x * (1.0f + scale_scalar2.x) + shift_scalar2.x,
                normed2.y * (1.0f + scale_scalar2.y) + shift_scalar2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
          cache_idx += 4;
          out4.store(y_pack, d4);
        }
      } else if ((scale_pack != nullptr) && (shift_pack != nullptr)) {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT sc4;
          sc4.load(scale_pack, d4);
          VecPackT sh4;
          sh4.load(shift_pack, d4);
          VecPackT out4;
          VecPackT w4;
          VecPackT b4;
          if constexpr (kHasWeight) {
            w4.load(weight_pack, d4);
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              b4.load(bias_pack, d4);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[cache_idx + i];

            fp32x2_t normed2;
            if constexpr (kIsLayerNorm) {
              normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
            } else {
              normed2 = {v2.x * inv_std, v2.y * inv_std};
            }
            if constexpr (kHasWeight) {
              const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
              normed2.x *= w2.x;
              normed2.y *= w2.y;
            }
            if constexpr (kIsLayerNorm) {
              if constexpr (kHasBias) {
                const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                normed2.x += b2.x;
                normed2.y += b2.y;
              }
            }

            const fp32x2_t scale2 = device::cast<fp32x2_t>(sc4[i]);
            const fp32x2_t shift2 = device::cast<fp32x2_t>(sh4[i]);
            const fp32x2_t out2 = {
                normed2.x * (1.0f + scale2.x) + shift2.x,
                normed2.y * (1.0f + scale2.y) + shift2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
          cache_idx += 4;
          out4.store(y_pack, d4);
        }
      } else if ((scale_fp32_bsd != nullptr) && (shift_fp32_bsd != nullptr)) {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT out4;
          VecPackT w4;
          VecPackT b4;
          if constexpr (kHasWeight) {
            w4.load(weight_pack, d4);
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              b4.load(bias_pack, d4);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[cache_idx + i];

            fp32x2_t normed2;
            if constexpr (kIsLayerNorm) {
              normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
            } else {
              normed2 = {v2.x * inv_std, v2.y * inv_std};
            }
            if constexpr (kHasWeight) {
              const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
              normed2.x *= w2.x;
              normed2.y *= w2.y;
            }
            if constexpr (kIsLayerNorm) {
              if constexpr (kHasBias) {
                const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                normed2.x += b2.x;
                normed2.y += b2.y;
              }
            }

            const int d2 = (d4 << 2) + i;
            const int d0 = d2 << 1;
            const int d1 = d0 + 1;
            const fp32x2_t scale2 = {scale_fp32_bsd[d0], scale_fp32_bsd[d1]};
            const fp32x2_t shift2 = {shift_fp32_bsd[d0], shift_fp32_bsd[d1]};
            const fp32x2_t out2 = {
                normed2.x * (1.0f + scale2.x) + shift2.x,
                normed2.y * (1.0f + scale2.y) + shift2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
          cache_idx += 4;
          out4.store(y_pack, d4);
        }
      } else {
        for (int d4 = tid; d4 < D4; d4 += static_cast<int>(blockDim.x)) {
          VecPackT sc4;
          if (scale_pack != nullptr) {
            sc4.load(scale_pack, d4);
          }
          VecPackT sh4;
          if (shift_pack != nullptr) {
            sh4.load(shift_pack, d4);
          }
          VecPackT out4;
          VecPackT w4;
          VecPackT b4;
          if constexpr (kHasWeight) {
            w4.load(weight_pack, d4);
          }
          if constexpr (kIsLayerNorm) {
            if constexpr (kHasBias) {
              b4.load(bias_pack, d4);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const fp32x2_t v2 = cached_v2[cache_idx + i];

            fp32x2_t normed2;
            if constexpr (kIsLayerNorm) {
              normed2 = {(v2.x - mean) * inv_std, (v2.y - mean) * inv_std};
            } else {
              normed2 = {v2.x * inv_std, v2.y * inv_std};
            }
            if constexpr (kHasWeight) {
              const fp32x2_t w2 = device::cast<fp32x2_t>(w4[i]);
              normed2.x *= w2.x;
              normed2.y *= w2.y;
            }
            if constexpr (kIsLayerNorm) {
              if constexpr (kHasBias) {
                const fp32x2_t b2 = device::cast<fp32x2_t>(b4[i]);
                normed2.x += b2.x;
                normed2.y += b2.y;
              }
            }

            const int d2 = (d4 << 2) + i;
            const fp32x2_t scale2 = scale_is_scalar
                                        ? scale_scalar2
                                        : (scale_pack != nullptr ? device::cast<fp32x2_t>(sc4[i])
                                                                 : load_bsfd_pair(scale_meta, b, s, d2, S));
            const fp32x2_t shift2 = shift_is_scalar
                                        ? shift_scalar2
                                        : (shift_pack != nullptr ? device::cast<fp32x2_t>(sh4[i])
                                                                 : load_bsfd_pair(shift_meta, b, s, d2, S));
            const fp32x2_t out2 = {
                normed2.x * (1.0f + scale2.x) + shift2.x,
                normed2.y * (1.0f + scale2.y) + shift2.y,
            };
            out4[i] = device::cast<PackT, fp32x2_t>(out2);
          }
          cache_idx += 4;
          out4.store(y_pack, d4);
        }
      }
    }
  } else {
    int cache_idx = 0;
    for (int d = tid; d < D; d += static_cast<int>(blockDim.x), ++cache_idx) {
      const float v = cached_v[cache_idx];

      float normed;
      if constexpr (kIsLayerNorm) {
        normed = (v - mean) * inv_std;
      } else {
        normed = v * inv_std;
      }
      if constexpr (kHasWeight) {
        normed *= device::cast<float>(weight[d]);
      }
      if constexpr (kIsLayerNorm) {
        if constexpr (kHasBias) {
          normed += device::cast<float>(bias[d]);
        }
      }

      const float scale =
          kFastBSFD ? load_bsd_value(scale_meta, scale_token_base, d) : load_bsfd_value(scale_meta, b, s, d, S);
      const float shift =
          kFastBSFD ? load_bsd_value(shift_meta, shift_token_base, d) : load_bsfd_value(shift_meta, b, s, d, S);
      y[y_base + static_cast<int64_t>(d) * y_stride2] = device::cast<T>(normed * (1.0f + scale) + shift);
    }
  }
}

template <typename DType>
inline BSFDMeta make_bsfd_meta(
    const tvm::ffi::TensorView t,
    const int64_t B,
    const int64_t S,
    const int64_t D,
    const DLDevice expected_device,
    const char* name) {
  using namespace host;
  const bool aux_dtype_supported =
      is_type<fp16_t>(t.dtype()) || is_type<bf16_t>(t.dtype()) || is_type<float>(t.dtype());
  RuntimeCheck(aux_dtype_supported, name, " dtype must be fp16/bf16/fp32");
  const DLDevice dev = t.device();
  RuntimeCheck(
      dev.device_type == expected_device.device_type && dev.device_id == expected_device.device_id,
      name,
      " must be on the same device as x");

  BSFDMeta meta{};
  meta.ptr = t.data_ptr();
  if (is_type<fp16_t>(t.dtype())) {
    meta.dtype = kBSFDDTypeFP16;
  } else if (is_type<bf16_t>(t.dtype())) {
    meta.dtype = kBSFDDTypeBF16;
  } else {
    meta.dtype = kBSFDDTypeFP32;
  }
  if (t.ndim() == 1) {
    RuntimeCheck(t.size(0) == 1, name, " 1D input must be scalar [1]");
    RuntimeCheck(t.stride(0) == 1, name, " must be contiguous");
    meta.mode = kScalarTensor;
    meta.F = 1;
    meta.stride0 = 0;
    meta.stride1 = 0;
    meta.stride2 = 0;
    meta.stride3 = 0;
    return meta;
  }

  if (t.ndim() == 3) {
    RuntimeCheck(t.size(0) == B && t.size(1) == S && t.size(2) == D, name, " shape mismatch");
    RuntimeCheck(t.stride(2) == 1, name, " must be contiguous on dim D");
    meta.mode = kTensorBSD;
    meta.F = 1;
    meta.stride0 = t.stride(0);
    meta.stride1 = t.stride(1);
    meta.stride2 = t.stride(2);
    meta.stride3 = 0;
    return meta;
  }

  if (t.ndim() == 4) {
    const int64_t F = t.size(1);
    RuntimeCheck(t.size(0) == B && t.size(2) == 1 && t.size(3) == D, name, " shape mismatch");
    RuntimeCheck(F > 0 && S % F == 0, name, " requires S % F == 0");
    RuntimeCheck(t.stride(3) == 1, name, " must be contiguous on dim D");
    meta.mode = kTensorBF1D;
    meta.F = static_cast<int32_t>(F);
    meta.stride0 = t.stride(0);
    meta.stride1 = t.stride(1);
    meta.stride2 = t.stride(2);
    meta.stride3 = t.stride(3);
    return meta;
  }

  Panic("Unsupported ndim for ", name, ": ", t.ndim());
}

inline uint32_t compute_num_threads(int64_t D, uint32_t num_tokens) {
  // Base policy: one warp per 256 hidden elements.
  uint32_t warps = static_cast<uint32_t>((D + 255) / 256);
  warps = std::max<uint32_t>(warps, 1);
  uint32_t threads = std::min<uint32_t>(warps * 32, 1024);

  // For small grids (few tokens), increase CTA size to improve occupancy.
  if (num_tokens < 1024) {
    const uint32_t dense_threads = std::min<uint32_t>(static_cast<uint32_t>(((D + 31) / 32) * 32), 1024);
    threads = std::max<uint32_t>(threads, dense_threads / 2);
  }

  return threads;
}

inline uint32_t compute_num_threads_norm(int64_t D, uint32_t num_tokens) {
  uint32_t threads = compute_num_threads(D, num_tokens);
  if (num_tokens >= 1024) {
    threads = std::min<uint32_t>(threads, 224);
  }
  return threads;
}

inline uint32_t compute_num_threads_residual(int64_t D, uint32_t num_tokens) {
  uint32_t threads = compute_num_threads(D, num_tokens);
  if (num_tokens >= 1024) {
    // d4_threads = D/8 is the minimum threads required by the cache iteration bound.
    const uint32_t d4_threads = static_cast<uint32_t>(D >> 3);
    // Cap CTA size for register pressure on H100/H200, but never below d4_threads
    // which is required to satisfy the cache iteration bound for large D.
    threads = std::min<uint32_t>(threads, std::max<uint32_t>(d4_threads, 256));
  }
  return threads;
}

template <typename DType, bool kIsLayerNorm>
struct FusedNormScaleShiftKernel {
  static void
  run(const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView scale,
      const tvm::ffi::TensorView shift,
      double eps,
      int64_t has_weight,
      int64_t has_bias) {
    using namespace host;
    RuntimeCheck(has_weight == 0 || has_weight == 1, "has_weight must be 0 or 1, got ", has_weight);
    RuntimeCheck(has_bias == 0 || has_bias == 1, "has_bias must be 0 or 1, got ", has_bias);
    const bool has_weight_bool = has_weight != 0;
    const bool has_bias_bool = has_bias != 0;

    RuntimeCheck(x.ndim() == 3, "x must be 3D [B, S, D]");
    RuntimeCheck(y.ndim() == 3, "y must be 3D [B, S, D]");
    RuntimeCheck(x.size(0) == y.size(0) && x.size(1) == y.size(1) && x.size(2) == y.size(2), "x/y shape mismatch");
    RuntimeCheck(is_type<DType>(x.dtype()) && is_type<DType>(y.dtype()), "x/y dtype mismatch");
    RuntimeCheck(x.stride(2) == 1 && y.stride(2) == 1, "x and y must be contiguous on dim D");
    RuntimeCheck(weight.ndim() == 1 && bias.ndim() == 1, "weight and bias must be 1D [D]");
    RuntimeCheck(is_type<DType>(weight.dtype()) && is_type<DType>(bias.dtype()), "weight/bias dtype mismatch");

    const int64_t B64 = x.size(0);
    const int64_t S64 = x.size(1);
    const int64_t D64 = x.size(2);
    RuntimeCheck(weight.stride(0) == 1 && bias.stride(0) == 1, "weight/bias must be contiguous");
    RuntimeCheck(
        has_weight_bool ? (weight.size(0) == D64) : (weight.size(0) == 1),
        has_weight_bool ? "weight size must be D" : "weight size must be 1 when has_weight=0");
    RuntimeCheck(
        has_bias_bool ? (bias.size(0) == D64) : (bias.size(0) == 1),
        has_bias_bool ? "bias size must be D" : "bias size must be 1 when has_bias=0");
    RuntimeCheck(D64 % 256 == 0 && D64 <= 8192, "D must be multiple of 256 and <= 8192, got ", D64);

    const DLDevice dev = x.device();
    RuntimeCheck(
        y.device().device_type == dev.device_type && y.device().device_id == dev.device_id,
        "x and y must be on same device");
    RuntimeCheck(
        weight.device().device_type == dev.device_type && weight.device().device_id == dev.device_id,
        "x and weight must be on same device");
    RuntimeCheck(
        bias.device().device_type == dev.device_type && bias.device().device_id == dev.device_id,
        "x and bias must be on same device");

    const BSFDMeta scale_meta = make_bsfd_meta<DType>(scale, B64, S64, D64, dev, "scale");
    const BSFDMeta shift_meta = make_bsfd_meta<DType>(shift, B64, S64, D64, dev, "shift");

    const int B = static_cast<int>(B64);
    const int S = static_cast<int>(S64);
    const int D = static_cast<int>(D64);
    const uint32_t num_tokens = static_cast<uint32_t>(B64 * S64);
    uint32_t threads = compute_num_threads_norm(D64, num_tokens);

    const bool scale_same_dtype = scale_meta.dtype == bsfd_dtype_code<DType>();
    const bool shift_same_dtype = shift_meta.dtype == bsfd_dtype_code<DType>();
    const bool use_fast_bsfd =
        (scale_meta.mode == kTensorBSD) && (shift_meta.mode == kTensorBSD) && scale_same_dtype && shift_same_dtype;
    size_t dynamic_smem = 0;
    if constexpr (std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>) {
      if (use_fast_bsfd) {
        dynamic_smem = static_cast<size_t>(D >> 1) * sizeof(packed_t<DType>);
      }
    }

#define LAUNCH_NORM_KERNEL(FAST, HW, HB)                                \
  LaunchKernel(num_tokens, threads, dev, dynamic_smem)(                 \
      fused_norm_scale_shift_kernel<DType, kIsLayerNorm, FAST, HW, HB>, \
      static_cast<const DType*>(x.data_ptr()),                          \
      x.stride(0),                                                      \
      x.stride(1),                                                      \
      x.stride(2),                                                      \
      static_cast<const DType*>(weight.data_ptr()),                     \
      static_cast<const DType*>(bias.data_ptr()),                       \
      scale_meta,                                                       \
      shift_meta,                                                       \
      static_cast<DType*>(y.data_ptr()),                                \
      y.stride(0),                                                      \
      y.stride(1),                                                      \
      y.stride(2),                                                      \
      B,                                                                \
      S,                                                                \
      D,                                                                \
      static_cast<float>(eps));

    if (use_fast_bsfd) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_NORM_KERNEL(true, true, true)
        } else {
          LAUNCH_NORM_KERNEL(true, true, false)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_NORM_KERNEL(true, false, true)
        } else {
          LAUNCH_NORM_KERNEL(true, false, false)
        }
      }
    } else {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_NORM_KERNEL(false, true, true)
        } else {
          LAUNCH_NORM_KERNEL(false, true, false)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_NORM_KERNEL(false, false, true)
        } else {
          LAUNCH_NORM_KERNEL(false, false, false)
        }
      }
    }

#undef LAUNCH_NORM_KERNEL
  }
};

template <typename DType, bool kIsLayerNorm>
struct FusedScaleResidualNormScaleShiftKernel {
  static void
  run(const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView residual_out,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView scale,
      const tvm::ffi::TensorView shift,
      double eps,
      int64_t has_weight,
      int64_t has_bias,
      int64_t gate_is_one) {
    using namespace host;
    RuntimeCheck(has_weight == 0 || has_weight == 1, "has_weight must be 0 or 1, got ", has_weight);
    RuntimeCheck(has_bias == 0 || has_bias == 1, "has_bias must be 0 or 1, got ", has_bias);
    RuntimeCheck(gate_is_one == 0 || gate_is_one == 1, "gate_is_one must be 0 or 1, got ", gate_is_one);
    const bool has_weight_bool = has_weight != 0;
    const bool has_bias_bool = has_bias != 0;
    const bool gate_is_one_bool = gate_is_one != 0;

    RuntimeCheck(x.ndim() == 3 && residual.ndim() == 3, "x/residual must be 3D [B, S, D]");
    RuntimeCheck(y.ndim() == 3 && residual_out.ndim() == 3, "y/residual_out must be 3D [B, S, D]");
    RuntimeCheck(
        x.size(0) == residual.size(0) && x.size(1) == residual.size(1) && x.size(2) == residual.size(2),
        "x/residual shape mismatch");
    RuntimeCheck(y.size(0) == x.size(0) && y.size(1) == x.size(1) && y.size(2) == x.size(2), "x/y shape mismatch");
    RuntimeCheck(
        residual_out.size(0) == x.size(0) && residual_out.size(1) == x.size(1) && residual_out.size(2) == x.size(2),
        "x/residual_out shape mismatch");
    RuntimeCheck(
        is_type<DType>(x.dtype()) && is_type<DType>(residual.dtype()) && is_type<DType>(y.dtype()) &&
            is_type<DType>(residual_out.dtype()),
        "x/residual/y/residual_out dtype mismatch");
    RuntimeCheck(
        x.stride(2) == 1 && residual.stride(2) == 1 && y.stride(2) == 1 && residual_out.stride(2) == 1,
        "x/residual/y/residual_out must be contiguous on dim D");
    RuntimeCheck(weight.ndim() == 1 && bias.ndim() == 1, "weight and bias must be 1D [D]");
    RuntimeCheck(is_type<DType>(weight.dtype()) && is_type<DType>(bias.dtype()), "weight/bias dtype mismatch");

    const int64_t B64 = x.size(0);
    const int64_t S64 = x.size(1);
    const int64_t D64 = x.size(2);
    RuntimeCheck(weight.stride(0) == 1 && bias.stride(0) == 1, "weight/bias must be contiguous");
    RuntimeCheck(
        has_weight_bool ? (weight.size(0) == D64) : (weight.size(0) == 1),
        has_weight_bool ? "weight size must be D" : "weight size must be 1 when has_weight=0");
    RuntimeCheck(
        has_bias_bool ? (bias.size(0) == D64) : (bias.size(0) == 1),
        has_bias_bool ? "bias size must be D" : "bias size must be 1 when has_bias=0");
    RuntimeCheck(D64 % 256 == 0 && D64 <= 8192, "D must be multiple of 256 and <= 8192, got ", D64);

    const DLDevice dev = x.device();
    RuntimeCheck(
        residual.device().device_type == dev.device_type && residual.device().device_id == dev.device_id,
        "x and residual must be on same device");
    RuntimeCheck(
        y.device().device_type == dev.device_type && y.device().device_id == dev.device_id,
        "x and y must be on same device");
    RuntimeCheck(
        residual_out.device().device_type == dev.device_type && residual_out.device().device_id == dev.device_id,
        "x and residual_out must be on same device");
    RuntimeCheck(
        weight.device().device_type == dev.device_type && weight.device().device_id == dev.device_id,
        "x and weight must be on same device");
    RuntimeCheck(
        bias.device().device_type == dev.device_type && bias.device().device_id == dev.device_id,
        "x and bias must be on same device");

    const BSFDMeta gate_meta = make_bsfd_meta<DType>(gate, B64, S64, D64, dev, "gate");
    const BSFDMeta scale_meta = make_bsfd_meta<DType>(scale, B64, S64, D64, dev, "scale");
    const BSFDMeta shift_meta = make_bsfd_meta<DType>(shift, B64, S64, D64, dev, "shift");

    const int B = static_cast<int>(B64);
    const int S = static_cast<int>(S64);
    const int D = static_cast<int>(D64);
    const uint32_t num_tokens = static_cast<uint32_t>(B64 * S64);
    uint32_t threads = compute_num_threads_residual(D64, num_tokens);
    constexpr bool kPackedCache = std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>;
    constexpr int kElemsPerCacheIter = kPackedCache ? 2 : 1;
    constexpr int kMaxCachedIters = kPackedCache ? kMaxCachedItersPacked : kMaxCachedItersScalar;
    RuntimeCheck(
        (((D / kElemsPerCacheIter) + static_cast<int>(threads) - 1) / static_cast<int>(threads)) <= kMaxCachedIters,
        "cache iteration bound exceeded");
    const bool can_single_fragment = (!kPackedCache) || (static_cast<int>(threads) >= (D >> 3));

    const bool gate_same_dtype = gate_meta.dtype == bsfd_dtype_code<DType>();
    const bool scale_same_dtype = scale_meta.dtype == bsfd_dtype_code<DType>();
    const bool shift_same_dtype = shift_meta.dtype == bsfd_dtype_code<DType>();
    const bool use_fast_bsfd = (gate_meta.mode == kTensorBSD) && (scale_meta.mode == kTensorBSD) &&
                               (shift_meta.mode == kTensorBSD) && gate_same_dtype && scale_same_dtype &&
                               shift_same_dtype;
    const bool use_gate_one_scale_shift_bsd_fp32 =
        can_single_fragment && gate_is_one_bool && (gate_meta.mode == kScalarTensor) &&
        (scale_meta.mode == kTensorBSD) && (shift_meta.mode == kTensorBSD) && (scale_meta.stride2 == 1) &&
        (shift_meta.stride2 == 1) && (scale_meta.dtype == kBSFDDTypeFP32) && (shift_meta.dtype == kBSFDDTypeFP32);
    const bool use_gate_one_scale_shift_bsd = can_single_fragment && gate_is_one_bool &&
                                              (gate_meta.mode == kScalarTensor) && (scale_meta.mode == kTensorBSD) &&
                                              (shift_meta.mode == kTensorBSD) && (scale_meta.stride2 == 1) &&
                                              (shift_meta.stride2 == 1) && !use_gate_one_scale_shift_bsd_fp32;
    const bool use_gate_one_scale_shift_bsd_packed =
        use_gate_one_scale_shift_bsd && scale_same_dtype && shift_same_dtype;
    const bool use_gate_scalar_scale_shift_bsd =
        (gate_meta.mode == kScalarTensor) && (scale_meta.mode == kTensorBSD) && (shift_meta.mode == kTensorBSD);
    const bool use_gate_scalar_scale_shift_bsd_affine =
        use_gate_scalar_scale_shift_bsd && !gate_is_one_bool &&
        (((scale_meta.dtype == bsfd_dtype_code<DType>()) && (shift_meta.dtype == bsfd_dtype_code<DType>()) &&
          (scale_meta.stride2 == 1) && (shift_meta.stride2 == 1)) ||
         ((scale_meta.dtype == kBSFDDTypeFP32) && (shift_meta.dtype == kBSFDDTypeFP32) && (scale_meta.stride2 == 1) &&
          (shift_meta.stride2 == 1)));
    const bool use_gate_bsd_scale_shift_scalar = can_single_fragment && (gate_meta.mode == kTensorBSD) &&
                                                 gate_same_dtype && (scale_meta.mode == kScalarTensor) &&
                                                 (shift_meta.mode == kScalarTensor);

#define LAUNCH_RESIDUAL_NORM_KERNEL(FAST, HW, HB, SM, SF)                                          \
  LaunchKernel(num_tokens, threads, dev)(                                                          \
      fused_scale_residual_norm_scale_shift_kernel<DType, kIsLayerNorm, FAST, HW, HB, SM, SF>,     \
      static_cast<const DType*>(residual.data_ptr()),                                      \
      residual.stride(0),                                                                  \
      residual.stride(1),                                                                  \
      residual.stride(2),                                                                  \
      static_cast<const DType*>(x.data_ptr()),                                             \
      x.stride(0),                                                                         \
      x.stride(1),                                                                         \
      x.stride(2),                                                                         \
      gate_meta,                                                                           \
      static_cast<const DType*>(weight.data_ptr()),                                        \
      static_cast<const DType*>(bias.data_ptr()),                                          \
      scale_meta,                                                                          \
      shift_meta,                                                                          \
      static_cast<DType*>(y.data_ptr()),                                                   \
      y.stride(0),                                                                         \
      y.stride(1),                                                                         \
      y.stride(2),                                                                         \
      static_cast<DType*>(residual_out.data_ptr()),                                        \
      residual_out.stride(0),                                                              \
      residual_out.stride(1),                                                              \
      residual_out.stride(2),                                                              \
      B,                                                                                   \
      S,                                                                                   \
      D,                                                                                   \
      static_cast<float>(eps));

    if (use_fast_bsfd) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(true, true, true, kResidualGeneric, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(true, true, false, kResidualGeneric, false)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(true, false, true, kResidualGeneric, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(true, false, false, kResidualGeneric, false)
        }
      }
    } else if (use_gate_one_scale_shift_bsd_fp32) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, true, kResidualGateOneScaleShiftBSDFP32, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, false, kResidualGateOneScaleShiftBSDFP32, true)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, true, kResidualGateOneScaleShiftBSDFP32, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, false, kResidualGateOneScaleShiftBSDFP32, true)
        }
      }
    } else if (use_gate_one_scale_shift_bsd_packed) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, true, kResidualGateOneScaleShiftBSDPacked, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, false, kResidualGateOneScaleShiftBSDPacked, true)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, true, kResidualGateOneScaleShiftBSDPacked, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, false, kResidualGateOneScaleShiftBSDPacked, true)
        }
      }
    } else if (use_gate_one_scale_shift_bsd) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, true, kResidualGateOneScaleShiftBSD, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, false, kResidualGateOneScaleShiftBSD, true)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, true, kResidualGateOneScaleShiftBSD, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, false, kResidualGateOneScaleShiftBSD, true)
        }
      }
    } else if (use_gate_scalar_scale_shift_bsd_affine) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, true, kResidualGateScalarScaleShiftBSDAffine, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, false, kResidualGateScalarScaleShiftBSDAffine, false)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, true, kResidualGateScalarScaleShiftBSDAffine, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, false, kResidualGateScalarScaleShiftBSDAffine, false)
        }
      }
    } else if (use_gate_scalar_scale_shift_bsd) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, true, kResidualGateScalarScaleShiftBSD, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, false, kResidualGateScalarScaleShiftBSD, false)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, true, kResidualGateScalarScaleShiftBSD, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, false, kResidualGateScalarScaleShiftBSD, false)
        }
      }
    } else if (use_gate_bsd_scale_shift_scalar) {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, true, kResidualGateBSDScaleShiftScalar, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, false, kResidualGateBSDScaleShiftScalar, true)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, true, kResidualGateBSDScaleShiftScalar, true)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, false, kResidualGateBSDScaleShiftScalar, true)
        }
      }
    } else {
      if (has_weight_bool) {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, true, kResidualGeneric, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, true, false, kResidualGeneric, false)
        }
      } else {
        if (has_bias_bool) {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, true, kResidualGeneric, false)
        } else {
          LAUNCH_RESIDUAL_NORM_KERNEL(false, false, false, kResidualGeneric, false)
        }
      }
    }

#undef LAUNCH_RESIDUAL_NORM_KERNEL
  }
};

}  // namespace
