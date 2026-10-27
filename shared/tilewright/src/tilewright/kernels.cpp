// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Built without fast-math and without floating-point contraction: the
// summation order written here is the result contract (see kernels.hpp).

#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <vector>

// __builtin_cpu_supports needs compiler-rt's CPU model, which MSVC-target links
// do not provide; Windows uses the scalar kernels.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(_WIN32)
#define TILEWRIGHT_X86_DISPATCH 1
#include <immintrin.h>
#else
#define TILEWRIGHT_X86_DISPATCH 0
#endif

namespace tilewright {
namespace detail {
namespace {

float reduce16(const float* a) {
  float t8[8];
  for (int i = 0; i < 8; ++i) t8[i] = a[i] + a[i + 8];
  float t4[4];
  for (int i = 0; i < 4; ++i) t4[i] = t8[i] + t8[i + 4];
  return (t4[0] + t4[2]) + (t4[1] + t4[3]);
}

template <bool Relu>
float finish(float v) {
  return Relu ? (v > 0.0f ? v : 0.0f) : v;
}

template <bool Relu>
void linear_scalar(const float* W,
                   const float* b,
                   const float* x,
                   std::size_t m,
                   std::size_t k,
                   float* out) {
  const std::size_t kv = k & ~static_cast<std::size_t>(15);
  for (std::size_t i = 0; i < m; ++i) {
    const float* w = W + i * k;
    float acc[16]  = {};
    for (std::size_t j = 0; j < kv; j += 16)
      for (std::size_t l = 0; l < 16; ++l) acc[l] = std::fma(w[j + l], x[j + l], acc[l]);
    float v = b[i] + reduce16(acc);
    for (std::size_t j = kv; j < k; ++j) v = v + w[j] * x[j];
    out[i] = finish<Relu>(v);
  }
}

float block_at(const BlockWeights& W, std::size_t e) {
  switch (W.type) {
    case BlockWeights::Type::F32: return static_cast<const float*>(W.data)[e];
    case BlockWeights::Type::Bf16: {
      const std::uint32_t bits =
          static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(W.data)[e]) << 16;
      float v;
      std::memcpy(&v, &bits, sizeof v);
      return v;
    }
    case BlockWeights::Type::Int8:
      return W.scale * static_cast<float>(static_cast<const std::int8_t*>(W.data)[e]);
  }
  return 0.0f;
}

// One row at a time through an fp32 kernel, whose rows do not depend on how
// many it is given.
template <
    void (*Linear)(const float*, const float*, const float*, std::size_t, std::size_t, float*),
    float (*Dot)(const float*, const float*, std::size_t)>
void linear_block_rows(const BlockWeights& W,
                       const float* b,
                       const float* x,
                       std::size_t m,
                       std::size_t k,
                       float* out) {
  thread_local std::vector<float> row;
  row.resize(k);
  for (std::size_t i = 0; i < m; ++i) {
    const std::size_t base = (i / 16) * k * 16 + i % 16;
    for (std::size_t j = 0; j < k; ++j) row[j] = block_at(W, base + j * 16);
    if (b == nullptr)
      out[i] = Dot(row.data(), x, k);
    else
      Linear(row.data(), b + i, x, 1, k, out + i);
  }
}

float dot_scalar(const float* a, const float* b, std::size_t n) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  float acc[16]        = {};
  for (std::size_t j = 0; j < nv; j += 16)
    for (std::size_t l = 0; l < 16; ++l) acc[l] = std::fma(a[j + l], b[j + l], acc[l]);
  float s = reduce16(acc);
  for (std::size_t j = nv; j < n; ++j) s = s + a[j] * b[j];
  return s;
}

void interaction_batch_scalar(const float* W,
                              const float* b,
                              const float* w2,
                              const float* X,
                              std::size_t m,
                              std::size_t k,
                              std::size_t nc,
                              float* hidden,
                              float* out) {
  const std::size_t kv = k & ~static_cast<std::size_t>(15);
  const std::size_t mv = m & ~static_cast<std::size_t>(15);
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t i = 0; i < m; ++i) {
      const float* w = W + i * k;
      float acc[16]  = {};
      for (std::size_t j = 0; j < kv; j += 16)
        for (std::size_t l = 0; l < 16; ++l)
          acc[l] = std::fma(w[j + l], X[(j + l) * kBatch + c], acc[l]);
      float v = b[i] + reduce16(acc);
      for (std::size_t j = kv; j < k; ++j) v = v + w[j] * X[j * kBatch + c];
      hidden[i * kBatch + c] = finish<true>(v);
    }
    float acc[16] = {};
    for (std::size_t i = 0; i < mv; i += 16)
      for (std::size_t l = 0; l < 16; ++l)
        acc[l] = std::fma(w2[i + l], hidden[(i + l) * kBatch + c], acc[l]);
    float s = reduce16(acc);
    for (std::size_t i = mv; i < m; ++i) s = s + w2[i] * hidden[i * kBatch + c];
    out[c] = s;
  }
}

void whiten_batch_scalar(float* X,
                         const float* mean,
                         const float* scale,
                         const float* lo,
                         const float* hi,
                         std::size_t k) {
  for (std::size_t j = 0; j < k; ++j) {
    float* row = X + j * kBatch;
    for (std::size_t c = 0; c < kBatch; ++c) {
      const float v = lo != nullptr ? std::min(std::max(row[c], lo[j]), hi[j]) : row[c];
      row[c]        = (v - mean[j]) / scale[j];
    }
  }
}

const Kernels kScalar = {Isa::Scalar,
                         &linear_scalar<false>,
                         &linear_scalar<true>,
                         &linear_block_rows<&linear_scalar<false>, &dot_scalar>,
                         &linear_block_rows<&linear_scalar<true>, &dot_scalar>,
                         &dot_scalar,
                         &interaction_batch_scalar,
                         &whiten_batch_scalar};

#if TILEWRIGHT_X86_DISPATCH

#define TILEWRIGHT_AVX2 __attribute__((target("avx2,fma")))
#define TILEWRIGHT_AVX512 __attribute__((target("avx512f")))

// Lanes 0-7 in `lo`, 8-15 in `hi`.
TILEWRIGHT_AVX2 inline float reduce_avx2(__m256 lo, __m256 hi) {
  const __m256 s8 = _mm256_add_ps(lo, hi);
  const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55)));
}

TILEWRIGHT_AVX512 inline float reduce_avx512(__m512 v) {
  const __m256 lo = _mm512_castps512_ps256(v);
  const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
  const __m256 s8 = _mm256_add_ps(lo, hi);
  const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55)));
}

// Rows are processed four at a time for instruction-level parallelism; each
// row's arithmetic is the same as in the one-row loop.
template <bool Relu>
TILEWRIGHT_AVX2 void linear_avx2(const float* W,
                                 const float* b,
                                 const float* x,
                                 std::size_t m,
                                 std::size_t k,
                                 float* out) {
  const std::size_t kv = k & ~static_cast<std::size_t>(15);
  std::size_t i        = 0;
  for (; i + 4 <= m; i += 4) {
    const float* w0 = W + i * k;
    const float* w1 = w0 + k;
    const float* w2 = w1 + k;
    const float* w3 = w2 + k;
    __m256 l0 = _mm256_setzero_ps(), h0 = l0, l1 = l0, h1 = l0, l2 = l0, h2 = l0, l3 = l0, h3 = l0;
    for (std::size_t j = 0; j < kv; j += 16) {
      const __m256 xl = _mm256_loadu_ps(x + j);
      const __m256 xh = _mm256_loadu_ps(x + j + 8);
      l0              = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j), xl, l0);
      h0              = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j + 8), xh, h0);
      l1              = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j), xl, l1);
      h1              = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j + 8), xh, h1);
      l2              = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j), xl, l2);
      h2              = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j + 8), xh, h2);
      l3              = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j), xl, l3);
      h3              = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j + 8), xh, h3);
    }
    float v0 = b[i] + reduce_avx2(l0, h0);
    float v1 = b[i + 1] + reduce_avx2(l1, h1);
    float v2 = b[i + 2] + reduce_avx2(l2, h2);
    float v3 = b[i + 3] + reduce_avx2(l3, h3);
    for (std::size_t j = kv; j < k; ++j) {
      const float xj = x[j];
      v0             = v0 + w0[j] * xj;
      v1             = v1 + w1[j] * xj;
      v2             = v2 + w2[j] * xj;
      v3             = v3 + w3[j] * xj;
    }
    out[i]     = finish<Relu>(v0);
    out[i + 1] = finish<Relu>(v1);
    out[i + 2] = finish<Relu>(v2);
    out[i + 3] = finish<Relu>(v3);
  }
  for (; i < m; ++i) {
    const float* w = W + i * k;
    __m256 lo = _mm256_setzero_ps(), hi = lo;
    for (std::size_t j = 0; j < kv; j += 16) {
      lo = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_loadu_ps(x + j), lo);
      hi = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 8), _mm256_loadu_ps(x + j + 8), hi);
    }
    float v = b[i] + reduce_avx2(lo, hi);
    for (std::size_t j = kv; j < k; ++j) v = v + w[j] * x[j];
    out[i] = finish<Relu>(v);
  }
}

TILEWRIGHT_AVX2 float dot_avx2(const float* a, const float* b, std::size_t n) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  __m256 lo = _mm256_setzero_ps(), hi = lo;
  for (std::size_t j = 0; j < nv; j += 16) {
    lo = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j), lo);
    hi = _mm256_fmadd_ps(_mm256_loadu_ps(a + j + 8), _mm256_loadu_ps(b + j + 8), hi);
  }
  float s = reduce_avx2(lo, hi);
  for (std::size_t j = nv; j < n; ++j) s = s + a[j] * b[j];
  return s;
}

// Rows are processed eight, then four at a time for instruction-level
// parallelism; each row's arithmetic is the same as in the one-row loop.
template <bool Relu>
TILEWRIGHT_AVX512 void linear_avx512(const float* W,
                                     const float* b,
                                     const float* x,
                                     std::size_t m,
                                     std::size_t k,
                                     float* out) {
  const std::size_t kv = k & ~static_cast<std::size_t>(15);
  std::size_t i        = 0;
  for (; i + 8 <= m; i += 8) {
    const float* w = W + i * k;
    __m512 a[8];
    for (int r = 0; r < 8; ++r) a[r] = _mm512_setzero_ps();
    for (std::size_t j = 0; j < kv; j += 16) {
      const __m512 xv = _mm512_loadu_ps(x + j);
      for (int r = 0; r < 8; ++r) a[r] = _mm512_fmadd_ps(_mm512_loadu_ps(w + r * k + j), xv, a[r]);
    }
    float v[8];
    for (int r = 0; r < 8; ++r) v[r] = b[i + r] + reduce_avx512(a[r]);
    for (std::size_t j = kv; j < k; ++j) {
      const float xj = x[j];
      for (int r = 0; r < 8; ++r) v[r] = v[r] + w[r * k + j] * xj;
    }
    for (int r = 0; r < 8; ++r) out[i + r] = finish<Relu>(v[r]);
  }
  for (; i + 4 <= m; i += 4) {
    const float* w0 = W + i * k;
    const float* w1 = w0 + k;
    const float* w2 = w1 + k;
    const float* w3 = w2 + k;
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (std::size_t j = 0; j < kv; j += 16) {
      const __m512 xv = _mm512_loadu_ps(x + j);
      a0              = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + j), xv, a0);
      a1              = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + j), xv, a1);
      a2              = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + j), xv, a2);
      a3              = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + j), xv, a3);
    }
    float v0 = b[i] + reduce_avx512(a0);
    float v1 = b[i + 1] + reduce_avx512(a1);
    float v2 = b[i + 2] + reduce_avx512(a2);
    float v3 = b[i + 3] + reduce_avx512(a3);
    for (std::size_t j = kv; j < k; ++j) {
      const float xj = x[j];
      v0             = v0 + w0[j] * xj;
      v1             = v1 + w1[j] * xj;
      v2             = v2 + w2[j] * xj;
      v3             = v3 + w3[j] * xj;
    }
    out[i]     = finish<Relu>(v0);
    out[i + 1] = finish<Relu>(v1);
    out[i + 2] = finish<Relu>(v2);
    out[i + 3] = finish<Relu>(v3);
  }
  for (; i < m; ++i) {
    const float* w = W + i * k;
    __m512 acc     = _mm512_setzero_ps();
    for (std::size_t j = 0; j < kv; j += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + j), _mm512_loadu_ps(x + j), acc);
    float v = b[i] + reduce_avx512(acc);
    for (std::size_t j = kv; j < k; ++j) v = v + w[j] * x[j];
    out[i] = finish<Relu>(v);
  }
}

// Entries e .. e + 15 of a BlockWeights matrix: one column of a block.
struct F32Blocks {
  const float* p;
  TILEWRIGHT_AVX512 __m512 v(std::size_t e) const { return _mm512_loadu_ps(p + e); }
};

struct Bf16Blocks {
  const std::uint16_t* p;
  TILEWRIGHT_AVX512 __m512 v(std::size_t e) const {
    const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + e));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
  }
};

struct Int8Blocks {
  const std::int8_t* p;
  float scale;
  TILEWRIGHT_AVX512 __m512 v(std::size_t e) const {
    const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + e));
    return _mm512_mul_ps(_mm512_set1_ps(scale), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)));
  }
};

// One block of 16 rows per vector, lane r holding row r: accumulator l takes
// the columns j = l mod 16 in order, the accumulators are summed in the
// reduce16 order, then the bias (if any) and the remaining columns are
// added, so each lane has the arithmetic of its row in the row kernels.
template <bool Relu, class Blocks>
TILEWRIGHT_AVX512 void linear_blocks_avx512_t(const Blocks& w,
                                              const float* b,
                                              const float* x,
                                              std::size_t m,
                                              std::size_t k,
                                              float* out) {
  const std::size_t kv = k & ~static_cast<std::size_t>(15);
  for (std::size_t i = 0; i < m; i += 16) {
    const std::size_t rows = m - i < 16 ? m - i : 16;
    const __mmask16 mask   = static_cast<__mmask16>((1u << rows) - 1u);
    const std::size_t base = i * k;
    __m512 a[16];
    for (int l = 0; l < 16; ++l) a[l] = _mm512_setzero_ps();
    for (std::size_t j = 0; j < kv; j += 16)
      for (int l = 0; l < 16; ++l)
        a[l] = _mm512_fmadd_ps(w.v(base + (j + l) * 16), _mm512_set1_ps(x[j + l]), a[l]);
    for (int l = 0; l < 8; ++l) a[l] = _mm512_add_ps(a[l], a[l + 8]);
    for (int l = 0; l < 4; ++l) a[l] = _mm512_add_ps(a[l], a[l + 4]);
    __m512 v = _mm512_add_ps(_mm512_add_ps(a[0], a[2]), _mm512_add_ps(a[1], a[3]));
    if (b != nullptr) v = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, b + i), v);
    for (std::size_t j = kv; j < k; ++j)
      v = _mm512_add_ps(v, _mm512_mul_ps(w.v(base + j * 16), _mm512_set1_ps(x[j])));
    // max(v, 0) is 0 for NaN and -0, as finish<true> is.
    if (Relu) v = _mm512_max_ps(v, _mm512_setzero_ps());
    _mm512_mask_storeu_ps(out + i, mask, v);
  }
}

template <bool Relu>
TILEWRIGHT_AVX512 void linear_blocks_avx512(const BlockWeights& W,
                                            const float* b,
                                            const float* x,
                                            std::size_t m,
                                            std::size_t k,
                                            float* out) {
  switch (W.type) {
    case BlockWeights::Type::F32:
      return linear_blocks_avx512_t<Relu>(
          F32Blocks{static_cast<const float*>(W.data)}, b, x, m, k, out);
    case BlockWeights::Type::Bf16:
      return linear_blocks_avx512_t<Relu>(
          Bf16Blocks{static_cast<const std::uint16_t*>(W.data)}, b, x, m, k, out);
    case BlockWeights::Type::Int8:
      return linear_blocks_avx512_t<Relu>(
          Int8Blocks{static_cast<const std::int8_t*>(W.data), W.scale}, b, x, m, k, out);
  }
}

TILEWRIGHT_AVX512 float dot_avx512(const float* a, const float* b, std::size_t n) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  __m512 acc           = _mm512_setzero_ps();
  for (std::size_t j = 0; j < nv; j += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j), acc);
  float s = reduce_avx512(acc);
  for (std::size_t j = nv; j < n; ++j) s = s + a[j] * b[j];
  return s;
}

// One column per vector lane: lane c of a row holds the row arithmetic of
// column c. Accumulators l and l + 8 are formed as a pair and summed at once,
// the first step of the reduce16 order.
TILEWRIGHT_AVX2 inline __m256 lane_dot_avx2(const float* w,
                                            const float* rows,
                                            std::size_t n,
                                            const float* bias) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  __m256 t8[8];
  for (std::size_t l = 0; l < 8; ++l) {
    __m256 lo = _mm256_setzero_ps(), hi = lo;
    for (std::size_t j = 0; j < nv; j += 16) {
      lo = _mm256_fmadd_ps(_mm256_set1_ps(w[j + l]), _mm256_loadu_ps(rows + (j + l) * kBatch), lo);
      hi = _mm256_fmadd_ps(
          _mm256_set1_ps(w[j + l + 8]), _mm256_loadu_ps(rows + (j + l + 8) * kBatch), hi);
    }
    t8[l] = _mm256_add_ps(lo, hi);
  }
  __m256 t4[4];
  for (std::size_t l = 0; l < 4; ++l) t4[l] = _mm256_add_ps(t8[l], t8[l + 4]);
  __m256 v = _mm256_add_ps(_mm256_add_ps(t4[0], t4[2]), _mm256_add_ps(t4[1], t4[3]));
  if (bias != nullptr) v = _mm256_add_ps(_mm256_set1_ps(*bias), v);
  for (std::size_t j = nv; j < n; ++j)
    v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_set1_ps(w[j]), _mm256_loadu_ps(rows + j * kBatch)));
  return v;
}

// std::max(v, lo) is lo > v ? lo : v, and std::min(v, hi) is hi < v ? hi : v,
// which _mm*_max_ps(lo, v) and _mm*_min_ps(hi, v) return.
TILEWRIGHT_AVX2 void whiten_batch_avx2(float* X,
                                       const float* mean,
                                       const float* scale,
                                       const float* lo,
                                       const float* hi,
                                       std::size_t k) {
  for (std::size_t j = 0; j < k; ++j) {
    float* row      = X + j * kBatch;
    const __m256 mu = _mm256_set1_ps(mean[j]), sd = _mm256_set1_ps(scale[j]);
    for (std::size_t c = 0; c < kBatch; c += 8) {
      __m256 v = _mm256_loadu_ps(row + c);
      if (lo != nullptr)
        v = _mm256_min_ps(_mm256_set1_ps(hi[j]), _mm256_max_ps(_mm256_set1_ps(lo[j]), v));
      _mm256_storeu_ps(row + c, _mm256_div_ps(_mm256_sub_ps(v, mu), sd));
    }
  }
}

TILEWRIGHT_AVX2 void interaction_batch_avx2(const float* W,
                                            const float* b,
                                            const float* w2,
                                            const float* X,
                                            std::size_t m,
                                            std::size_t k,
                                            std::size_t nc,
                                            float* hidden,
                                            float* out) {
  for (std::size_t c0 = 0; c0 < nc; c0 += 8) {
    for (std::size_t i = 0; i < m; ++i)
      _mm256_storeu_ps(
          hidden + i * kBatch + c0,
          _mm256_max_ps(lane_dot_avx2(W + i * k, X + c0, k, b + i), _mm256_setzero_ps()));
    float s[8];
    _mm256_storeu_ps(s, lane_dot_avx2(w2, hidden + c0, m, nullptr));
    for (std::size_t c = c0; c < nc && c < c0 + 8; ++c) out[c] = s[c - c0];
  }
}

TILEWRIGHT_AVX512 inline __m512 lane_dot_avx512(const float* w,
                                                const float* rows,
                                                std::size_t n,
                                                const float* bias) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  __m512 t8[8];
  for (std::size_t l = 0; l < 8; ++l) {
    __m512 lo = _mm512_setzero_ps(), hi = lo;
    for (std::size_t j = 0; j < nv; j += 16) {
      lo = _mm512_fmadd_ps(_mm512_set1_ps(w[j + l]), _mm512_loadu_ps(rows + (j + l) * kBatch), lo);
      hi = _mm512_fmadd_ps(
          _mm512_set1_ps(w[j + l + 8]), _mm512_loadu_ps(rows + (j + l + 8) * kBatch), hi);
    }
    t8[l] = _mm512_add_ps(lo, hi);
  }
  __m512 t4[4];
  for (std::size_t l = 0; l < 4; ++l) t4[l] = _mm512_add_ps(t8[l], t8[l + 4]);
  __m512 v = _mm512_add_ps(_mm512_add_ps(t4[0], t4[2]), _mm512_add_ps(t4[1], t4[3]));
  if (bias != nullptr) v = _mm512_add_ps(_mm512_set1_ps(*bias), v);
  for (std::size_t j = nv; j < n; ++j)
    v = _mm512_add_ps(v, _mm512_mul_ps(_mm512_set1_ps(w[j]), _mm512_loadu_ps(rows + j * kBatch)));
  return v;
}

// lane_dot_avx512 of rows w0 and w1 with shared loads of `rows`: the
// accumulators of lanes l, l + 8, l + 4 and l + 12 are formed together and
// reduced to t4[l] at once.
TILEWRIGHT_AVX512 inline void lane_dot2_avx512(const float* w0,
                                               const float* w1,
                                               const float* rows,
                                               std::size_t n,
                                               float b0,
                                               float b1,
                                               __m512* v0,
                                               __m512* v1) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  __m512 s0[4], s1[4];
  for (std::size_t l = 0; l < 4; ++l) {
    __m512 a0 = _mm512_setzero_ps(), a8 = a0, a4 = a0, a12 = a0;
    __m512 c0 = a0, c8 = a0, c4 = a0, c12 = a0;
    for (std::size_t j = 0; j < nv; j += 16) {
      const __m512 x0  = _mm512_loadu_ps(rows + (j + l) * kBatch);
      const __m512 x8  = _mm512_loadu_ps(rows + (j + l + 8) * kBatch);
      const __m512 x4  = _mm512_loadu_ps(rows + (j + l + 4) * kBatch);
      const __m512 x12 = _mm512_loadu_ps(rows + (j + l + 12) * kBatch);
      a0               = _mm512_fmadd_ps(_mm512_set1_ps(w0[j + l]), x0, a0);
      a8               = _mm512_fmadd_ps(_mm512_set1_ps(w0[j + l + 8]), x8, a8);
      a4               = _mm512_fmadd_ps(_mm512_set1_ps(w0[j + l + 4]), x4, a4);
      a12              = _mm512_fmadd_ps(_mm512_set1_ps(w0[j + l + 12]), x12, a12);
      c0               = _mm512_fmadd_ps(_mm512_set1_ps(w1[j + l]), x0, c0);
      c8               = _mm512_fmadd_ps(_mm512_set1_ps(w1[j + l + 8]), x8, c8);
      c4               = _mm512_fmadd_ps(_mm512_set1_ps(w1[j + l + 4]), x4, c4);
      c12              = _mm512_fmadd_ps(_mm512_set1_ps(w1[j + l + 12]), x12, c12);
    }
    s0[l] = _mm512_add_ps(_mm512_add_ps(a0, a8), _mm512_add_ps(a4, a12));
    s1[l] = _mm512_add_ps(_mm512_add_ps(c0, c8), _mm512_add_ps(c4, c12));
  }
  __m512 r0 = _mm512_add_ps(_mm512_add_ps(s0[0], s0[2]), _mm512_add_ps(s0[1], s0[3]));
  __m512 r1 = _mm512_add_ps(_mm512_add_ps(s1[0], s1[2]), _mm512_add_ps(s1[1], s1[3]));
  r0        = _mm512_add_ps(_mm512_set1_ps(b0), r0);
  r1        = _mm512_add_ps(_mm512_set1_ps(b1), r1);
  for (std::size_t j = nv; j < n; ++j) {
    const __m512 x = _mm512_loadu_ps(rows + j * kBatch);
    r0             = _mm512_add_ps(r0, _mm512_mul_ps(_mm512_set1_ps(w0[j]), x));
    r1             = _mm512_add_ps(r1, _mm512_mul_ps(_mm512_set1_ps(w1[j]), x));
  }
  *v0 = r0;
  *v1 = r1;
}

TILEWRIGHT_AVX512 void whiten_batch_avx512(float* X,
                                           const float* mean,
                                           const float* scale,
                                           const float* lo,
                                           const float* hi,
                                           std::size_t k) {
  for (std::size_t j = 0; j < k; ++j) {
    float* row = X + j * kBatch;
    __m512 v   = _mm512_loadu_ps(row);
    if (lo != nullptr)
      v = _mm512_min_ps(_mm512_set1_ps(hi[j]), _mm512_max_ps(_mm512_set1_ps(lo[j]), v));
    _mm512_storeu_ps(
        row, _mm512_div_ps(_mm512_sub_ps(v, _mm512_set1_ps(mean[j])), _mm512_set1_ps(scale[j])));
  }
}

// lane_dot2_avx512 of R rows, w + r * k for r < R, with the bias b[r] and
// shared loads of `rows`; the remaining columns are added to the rows side by
// side.
template <int R>
TILEWRIGHT_AVX512 inline void lane_dotn_avx512(const float* w,
                                               std::size_t k,
                                               const float* rows,
                                               const float* b,
                                               __m512* v) {
  const std::size_t nv = k & ~static_cast<std::size_t>(15);
  __m512 s[R][4];
  for (std::size_t l = 0; l < 4; ++l) {
    __m512 half[R];
    for (std::size_t p = 0; p < 2; ++p) {
      __m512 a0[R], a1[R];
      for (int r = 0; r < R; ++r) a0[r] = a1[r] = _mm512_setzero_ps();
      for (std::size_t j = 0; j < nv; j += 16) {
        const std::size_t c0 = j + l + 4 * p, c1 = c0 + 8;
        const __m512 x0 = _mm512_loadu_ps(rows + c0 * kBatch);
        const __m512 x1 = _mm512_loadu_ps(rows + c1 * kBatch);
        for (int r = 0; r < R; ++r) {
          a0[r] = _mm512_fmadd_ps(_mm512_set1_ps(w[r * k + c0]), x0, a0[r]);
          a1[r] = _mm512_fmadd_ps(_mm512_set1_ps(w[r * k + c1]), x1, a1[r]);
        }
      }
      for (int r = 0; r < R; ++r) {
        const __m512 t = _mm512_add_ps(a0[r], a1[r]);
        if (p == 0)
          half[r] = t;
        else
          s[r][l] = _mm512_add_ps(half[r], t);
      }
    }
  }
  for (int r = 0; r < R; ++r) {
    const __m512 t =
        _mm512_add_ps(_mm512_add_ps(s[r][0], s[r][2]), _mm512_add_ps(s[r][1], s[r][3]));
    v[r] = _mm512_add_ps(_mm512_set1_ps(b[r]), t);
  }
  for (std::size_t j = nv; j < k; ++j) {
    const __m512 x = _mm512_loadu_ps(rows + j * kBatch);
    for (int r = 0; r < R; ++r)
      v[r] = _mm512_add_ps(v[r], _mm512_mul_ps(_mm512_set1_ps(w[r * k + j]), x));
  }
}

TILEWRIGHT_AVX512 void interaction_batch_avx512(const float* W,
                                                const float* b,
                                                const float* w2,
                                                const float* X,
                                                std::size_t m,
                                                std::size_t k,
                                                std::size_t nc,
                                                float* hidden,
                                                float* out) {
  constexpr int kRows = 8;
  std::size_t i       = 0;
  for (; i + kRows <= m; i += kRows) {
    __m512 v[kRows];
    lane_dotn_avx512<kRows>(W + i * k, k, X, b + i, v);
    for (int r = 0; r < kRows; ++r)
      _mm512_storeu_ps(hidden + (i + r) * kBatch, _mm512_max_ps(v[r], _mm512_setzero_ps()));
  }
  for (; i + 2 <= m; i += 2) {
    __m512 v0, v1;
    lane_dot2_avx512(W + i * k, W + (i + 1) * k, X, k, b[i], b[i + 1], &v0, &v1);
    _mm512_storeu_ps(hidden + i * kBatch, _mm512_max_ps(v0, _mm512_setzero_ps()));
    _mm512_storeu_ps(hidden + (i + 1) * kBatch, _mm512_max_ps(v1, _mm512_setzero_ps()));
  }
  for (; i < m; ++i)
    _mm512_storeu_ps(hidden + i * kBatch,
                     _mm512_max_ps(lane_dot_avx512(W + i * k, X, k, b + i), _mm512_setzero_ps()));
  float s[16];
  _mm512_storeu_ps(s, lane_dot_avx512(w2, hidden, m, nullptr));
  for (std::size_t c = 0; c < nc; ++c) out[c] = s[c];
}

const Kernels kAvx2   = {Isa::Avx2,
                         &linear_avx2<false>,
                         &linear_avx2<true>,
                         &linear_block_rows<&linear_avx2<false>, &dot_avx2>,
                         &linear_block_rows<&linear_avx2<true>, &dot_avx2>,
                         &dot_avx2,
                         &interaction_batch_avx2,
                         &whiten_batch_avx2};
const Kernels kAvx512 = {Isa::Avx512,
                         &linear_avx512<false>,
                         &linear_avx512<true>,
                         &linear_blocks_avx512<false>,
                         &linear_blocks_avx512<true>,
                         &dot_avx512,
                         &interaction_batch_avx512,
                         &whiten_batch_avx512};

bool cpu_supports(Isa isa) {
  __builtin_cpu_init();
  switch (isa) {
    case Isa::Avx512: return __builtin_cpu_supports("avx512f") != 0;
    case Isa::Avx2:
      return __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0;
    case Isa::Scalar: return true;
  }
  return false;
}

#endif

}  // namespace

const Kernels* kernels_for(Isa isa) noexcept {
  if (isa == Isa::Scalar) return &kScalar;
#if TILEWRIGHT_X86_DISPATCH
  if (!cpu_supports(isa)) return nullptr;
  return isa == Isa::Avx512 ? &kAvx512 : &kAvx2;
#else
  return nullptr;
#endif
}

const Kernels& kernels() noexcept {
  static const Kernels* const best = [] {
    for (Isa isa : {Isa::Avx512, Isa::Avx2})
      if (const Kernels* k = kernels_for(isa)) return k;
    return &kScalar;
  }();
  return *best;
}

}  // namespace detail
}  // namespace tilewright
