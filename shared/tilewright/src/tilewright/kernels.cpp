// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Built without fast-math and without floating-point contraction: the
// summation order written here is the result contract (see kernels.hpp).

#include "kernels.hpp"

#include <cmath>
#include <initializer_list>

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

float dot_scalar(const float* a, const float* b, std::size_t n) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  float acc[16]        = {};
  for (std::size_t j = 0; j < nv; j += 16)
    for (std::size_t l = 0; l < 16; ++l) acc[l] = std::fma(a[j + l], b[j + l], acc[l]);
  float s = reduce16(acc);
  for (std::size_t j = nv; j < n; ++j) s = s + a[j] * b[j];
  return s;
}

const Kernels kScalar = {Isa::Scalar, &linear_scalar<false>, &linear_scalar<true>, &dot_scalar};

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

template <bool Relu>
TILEWRIGHT_AVX512 void linear_avx512(const float* W,
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

TILEWRIGHT_AVX512 float dot_avx512(const float* a, const float* b, std::size_t n) {
  const std::size_t nv = n & ~static_cast<std::size_t>(15);
  __m512 acc           = _mm512_setzero_ps();
  for (std::size_t j = 0; j < nv; j += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j), acc);
  float s = reduce_avx512(acc);
  for (std::size_t j = nv; j < n; ++j) s = s + a[j] * b[j];
  return s;
}

const Kernels kAvx2   = {Isa::Avx2, &linear_avx2<false>, &linear_avx2<true>, &dot_avx2};
const Kernels kAvx512 = {Isa::Avx512, &linear_avx512<false>, &linear_avx512<true>, &dot_avx512};

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
