// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>

namespace tilewright {
namespace detail {

enum class Isa { Scalar, Avx2, Avx512 };

// Columns of one interaction_batch call.
constexpr std::size_t kBatch = 16;

// fp32 inner-product kernels. Every variant returns identical bits: a row is
// accumulated in 16 interleaved lanes with fused multiply-adds over the first
// floor(k/16)*16 elements, the lanes are summed pairwise in the order of
// _mm512_reduce_add_ps, the bias is added, and the remaining elements are added
// one at a time with a separate multiply and add.
struct Kernels {
  Isa isa;
  // out[i] = b[i] + W[i, :] . x for a row-major m x k matrix W.
  void (*linear)(const float* W,
                 const float* b,
                 const float* x,
                 std::size_t m,
                 std::size_t k,
                 float* out);
  // As linear, then out[i] = out[i] > 0 ? out[i] : 0.
  void (*linear_relu)(const float* W,
                      const float* b,
                      const float* x,
                      std::size_t m,
                      std::size_t k,
                      float* out);
  // a . b without a bias.
  float (*dot)(const float* a, const float* b, std::size_t n);
  // For c < nc: out[c] = w2 . relu(W . X[:, c] + b), the bits of linear_relu
  // then dot on column c alone. X is k x kBatch, row j holding element j of
  // every column; hidden is m x kBatch scratch. Columns from nc on are ignored.
  void (*interaction_batch)(const float* W,
                            const float* b,
                            const float* w2,
                            const float* X,
                            std::size_t m,
                            std::size_t k,
                            std::size_t nc,
                            float* hidden,
                            float* out);
};

// The fastest variant this CPU supports, chosen once.
const Kernels& kernels() noexcept;

// A specific variant, or nullptr when this build or CPU lacks it.
const Kernels* kernels_for(Isa isa) noexcept;

}  // namespace detail
}  // namespace tilewright
