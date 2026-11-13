// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace tilewright {
namespace detail {

enum class Isa { Scalar, Avx2, Avx512 };

// Columns of one interaction_batch call.
constexpr std::size_t kBatch = 16;

// An m x k weight matrix in blocks of 16 rows: element (i, j) is entry
// ((i / 16) * k + j) * 16 + i % 16 of `data`, and the entries of rows m and
// on in the last block are 0. An entry is an fp32 value, the upper half of an
// fp32 value (bf16), or an int8 q standing for scale * float(q). The int8 form
// holds int8 weights, and int4 nibbles n as q = n - 8, with the model file's
// scale: these are the fp32 values Model::weights expands.
struct BlockWeights {
  enum class Type : std::uint8_t { F32, Bf16, Int8 };
  Type type        = Type::F32;
  const void* data = nullptr;
  float scale      = 1.0f;
};

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
  // linear and linear_relu of the values of W. linear_blocks with b null
  // gives dot of each row of W and x instead.
  void (*linear_blocks)(const BlockWeights& W,
                        const float* b,
                        const float* x,
                        std::size_t m,
                        std::size_t k,
                        float* out);
  void (*linear_blocks_relu)(const BlockWeights& W,
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
  // Row j of the k x kBatch X becomes (min(max(x, lo[j]), hi[j]) - mean[j]) /
  // scale[j] in place, the std::min and std::max of the scalar form; without
  // bounds (lo == nullptr), (x - mean[j]) / scale[j].
  void (*whiten_batch)(float* X,
                       const float* mean,
                       const float* scale,
                       const float* lo,
                       const float* hi,
                       std::size_t k);
};

// The fastest variant this CPU supports, chosen once.
const Kernels& kernels() noexcept;

// A specific variant, or nullptr when this build or CPU lacks it.
const Kernels* kernels_for(Isa isa) noexcept;

}  // namespace detail
}  // namespace tilewright
