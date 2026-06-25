// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// tilewright engine -- the MLREC_v1 runtime (framework-agnostic)
// =============================================================================
//
// The GRID-aware ("split-tree, per-cell two-tower MLP") GEMM kernel
// recommender: inference math, the on-disk MLREC_v1 `.bin` serialization, the
// feature catalog, the whitening, the smart-K signature filter, the argmax
// tie-break, and a self-contained LDS-capacity feasibility gate. The engine
// depends on no GEMM framework.
//
// Source-of-truth mapping (the Python training/deploy pipeline):
//   * .bin format            <- the deploy stage's v6 writer
//                               (_emit_v6_bin / _pack_weights /
//                                bf16 + int8 packing)
//   * model architecture     <- the trainer (GenericTwoTower:
//                               q_proj depth-3, i_proj depth-2,
//                               inter_mlp [Linear,ReLU,Linear], temperature)
//   * deployed inference     <- the trainer's deployed-pick routine: LDS gate ->
//                               feasibility filter -> smart_K signature filter
//                               (two-pass w/ fallback) -> MLP score ->
//                               argmax (first-max wins, matching torch.argmax)
//   * feature catalog        <- the training feature library
//                               (query|item|interaction feature groups)
//   * whitening              <- the trainer:
//                               sd<1e-6 -> 1.0, normed=(v-mean)/sd
//   * split-tree routing     <- the subcell/grid library (assign_subcell +
//                               cell_key / m_tier/n_tier/k_tier/b_tier)
//   * kernel signature       <- the config-signature helper (8-tuple
//                               mt_m,mt_n,mt_k,mi_m,mi_n,mi_k,cha,chb)
//   * feasibility filter      <- the analytical kernel-feasibility check
//
// CORRECTNESS BAR: built from this source, the C++ must produce the SAME
// kernel pick as the Python deployed-pick routine for every shape
// (parity == 100%). Hence feature math is done in DOUBLE then cast to
// float (mirroring Python: math in float64, then np.float32), whitening and
// the MLP run in float32, and the .bin is read byte-for-byte.
// =============================================================================

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __unix__
#include <dlfcn.h>
#endif

namespace tilewright {

namespace {

// ── float dot product (reused scaffolding from the prior runtime) ──────────
float scalar_dot(const float* a, const float* b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t j = 0; j < n; ++j) sum += a[j] * b[j];
  return sum;
}

#ifdef __AVX512F__
float avx512_dot(const float* a, const float* b, std::size_t n) {
  __m512 acc    = _mm512_setzero_ps();
  std::size_t j = 0;
  for (; j + 16 <= n; j += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j), acc);
  float sum = _mm512_reduce_add_ps(acc);
  for (; j < n; ++j) sum += a[j] * b[j];
  return sum;
}
#endif

using dot_fn_t = float (*)(const float*, const float*, std::size_t);

dot_fn_t select_dot_fn() {
#ifdef __AVX512F__
#ifdef __x86_64__
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1u << 16))) return avx512_dot;
#endif
#endif
  return scalar_dot;
}

static const dot_fn_t s_dot = select_dot_fn();

// Forward-pass inner products in bf16 (AVX512-BF16 _mm512_dpbf16_ps with
// PRE-CONVERTED weights built at load + once-per-layer activation packing;
// accumulation stays fp32).
//
// DEFAULT ON: bf16 inner products with near-exact pick parity vs the fp32 path
// (the rare differences are score tie-flips, no selection-quality impact).
// Set TILEWRIGHT_COMPUTE_BF16=0 to force the exact fp32 path (bitwise parity).
// Read once at first use. Compiles to fp32 automatically where __AVX512BF16__
// is unavailable (the bf16 weight mirrors are then never built -> fp32 path).
inline bool _tilewright_compute_bf16() {
  static const bool on = [] {
    const char* e = std::getenv("TILEWRIGHT_COMPUTE_BF16");
    if (e == nullptr) return true;  // default ON
    return (e[0] == '1' || e[0] == 't' || e[0] == 'y' || e[0] == 'T');
  }();
  return on;
}
#ifdef __AVX512BF16__
// Pack k fp32 -> k bf16 (round-to-nearest-even) into `dst`.
inline void _pack_bf16(const float* __restrict__ src,
                       std::uint16_t* __restrict__ dst,
                       std::size_t k) {
  std::size_t j = 0;
  for (; j + 16 <= k; j += 16)
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + j),
                        reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(_mm512_loadu_ps(src + j))));
  for (; j < k; ++j) {
    std::uint32_t bits;
    std::memcpy(&bits, src + j, 4);
    std::uint32_t r = bits + 0x7FFFu + ((bits >> 16) & 1u);
    dst[j]          = static_cast<std::uint16_t>(r >> 16);
  }
}
// Dot of a pre-packed bf16 weight row Wbf[k] with a pre-packed bf16 x[k].
inline float _bf16_dot_pre(const std::uint16_t* __restrict__ Wbf,
                           const std::uint16_t* __restrict__ xbf,
                           std::size_t k,
                           std::size_t* j_out) {
  __m512 acc    = _mm512_setzero_ps();
  std::size_t j = 0;
  for (; j + 32 <= k; j += 32) {
    __m512bh wb =
        reinterpret_cast<__m512bh>(_mm512_loadu_si512(reinterpret_cast<const void*>(Wbf + j)));
    __m512bh xb =
        reinterpret_cast<__m512bh>(_mm512_loadu_si512(reinterpret_cast<const void*>(xbf + j)));
    acc = _mm512_dpbf16_ps(acc, wb, xb);
  }
  *j_out = j;
  return _mm512_reduce_add_ps(acc);
}
// Thread-local bf16 scratch for the (single) activation vector x.
inline std::vector<std::uint16_t>& _xbf_scratch() {
  static thread_local std::vector<std::uint16_t> s;
  return s;
}
#endif
// ── int8 (VNNI) compute path ───────────────────────────────────────────────
inline bool _tilewright_compute_int8() {
  static const bool on = [] {
    const char* e = std::getenv("TILEWRIGHT_COMPUTE_INT8");
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'y' || e[0] == 'T');
  }();
  return on;
}
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
inline std::vector<std::uint8_t>& _xu8_scratch() {
  static thread_local std::vector<std::uint8_t> s;
  return s;
}
// VNNI dot: u8 activations (= s8 + 128) dotted with s8 weights.
// dpbusd accumulates sum(u8 * s8) in int32 lanes; we offset-correct with
// 128*rowsum(W_q) at the caller. Returns the int32 sum over the 64-wide body.
inline std::int32_t _i8_dot_vnni(const std::int8_t* __restrict__ Wq,
                                 const std::uint8_t* __restrict__ xu,
                                 std::size_t k,
                                 std::size_t* j_out) {
  __m512i acc   = _mm512_setzero_si512();
  std::size_t j = 0;
  for (; j + 64 <= k; j += 64) {
    __m512i a = _mm512_loadu_si512(reinterpret_cast<const void*>(xu + j));
    __m512i b = _mm512_loadu_si512(reinterpret_cast<const void*>(Wq + j));
    acc       = _mm512_dpbusd_epi32(acc, a, b);
  }
  *j_out = j;
  return _mm512_reduce_add_epi32(acc);
}
// Quantize activations to u8 = clamp(round(x/sx),-127,127)+128; returns sx.
inline float _quant_x_u8(const float* __restrict__ x,
                         std::size_t k,
                         std::uint8_t* __restrict__ xu) {
  float amax = 0.0f;
  for (std::size_t j = 0; j < k; ++j) {
    float a = std::fabs(x[j]);
    if (a > amax) amax = a;
  }
  float sx  = (amax > 0.0f) ? amax / 127.0f : 1.0f;
  float inv = 1.0f / sx;
  for (std::size_t j = 0; j < k; ++j) {
    int q = (int)std::lround(x[j] * inv);
    q     = q < -127 ? -127 : (q > 127 ? 127 : q);
    xu[j] = (std::uint8_t)(q + 128);
  }
  return sx;
}
// Generic int8 linear: out[i] = b[i] + sw*sx*(VNNI(xu,Wq_i) - 128*rowsum[i]).
inline bool _i8_linear(const float* __restrict__ b,
                       const float* __restrict__ x,
                       std::size_t m,
                       std::size_t k,
                       float* __restrict__ out,
                       const std::int8_t* Wq,
                       const std::int32_t* rowsum,
                       float wscale,
                       bool relu) {
  auto& xu = _xu8_scratch();
  xu.resize(k);
  float sx  = _quant_x_u8(x, k, xu.data());
  float deq = wscale * sx;
  for (std::size_t i = 0; i < m; ++i) {
    const std::int8_t* Wi = Wq + i * k;
    std::size_t j         = 0;
    std::int32_t s        = _i8_dot_vnni(Wi, xu.data(), k, &j);
    for (; j < k; ++j) s += (std::int32_t)xu[j] * (std::int32_t)Wi[j];
    float v = b[i] + deq * (float)(s - 128 * rowsum[i]);
    out[i]  = relu ? (v > 0.0f ? v : 0.0f) : v;
  }
  return true;
}
#endif
// `Wbf` (optional) is the pre-packed bf16 mirror of W; when non-null AND bf16
// compute is enabled, the forward uses pre-converted weights (x packed once).
inline void linear_relu(const float* __restrict__ W,
                        const float* __restrict__ b,
                        const float* __restrict__ x,
                        std::size_t m,
                        std::size_t k,
                        float* __restrict__ out,
                        const std::uint16_t* Wbf    = nullptr,
                        const std::int8_t* Wq       = nullptr,
                        const std::int32_t* Wrowsum = nullptr,
                        float wscale                = 1.0f) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
  if (Wq && _tilewright_compute_int8()) {
    _i8_linear(b, x, m, k, out, Wq, Wrowsum, wscale, /*relu=*/true);
    return;
  }
#endif
#ifdef __AVX512BF16__
  if (Wbf && _tilewright_compute_bf16()) {
    auto& xbf = _xbf_scratch();
    xbf.resize(k);
    _pack_bf16(x, xbf.data(), k);
    for (std::size_t i = 0; i < m; ++i) {
      std::size_t j   = 0;
      float v         = b[i] + _bf16_dot_pre(Wbf + i * k, xbf.data(), k, &j);
      const float* Wi = W + i * k;
      for (; j < k; ++j) v += Wi[j] * x[j];
      out[i] = v > 0.0f ? v : 0.0f;
    }
    return;
  }
#endif
#ifdef __AVX512F__
  for (std::size_t i = 0; i < m; ++i) {
    __m512 acc      = _mm512_setzero_ps();
    std::size_t j   = 0;
    const float* Wi = W + i * k;
    for (; j + 16 <= k; j += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(Wi + j), _mm512_loadu_ps(x + j), acc);
    float v = b[i] + _mm512_reduce_add_ps(acc);
    for (; j < k; ++j) v += Wi[j] * x[j];
    out[i] = v > 0.0f ? v : 0.0f;
  }
#else
  for (std::size_t i = 0; i < m; ++i) {
    float acc = b[i] + s_dot(W + i * k, x, k);
    out[i]    = acc > 0.0f ? acc : 0.0f;
  }
#endif
}
inline void linear(const float* __restrict__ W,
                   const float* __restrict__ b,
                   const float* __restrict__ x,
                   std::size_t m,
                   std::size_t k,
                   float* __restrict__ out,
                   const std::uint16_t* Wbf    = nullptr,
                   const std::int8_t* Wq       = nullptr,
                   const std::int32_t* Wrowsum = nullptr,
                   float wscale                = 1.0f) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
  if (Wq && _tilewright_compute_int8()) {
    _i8_linear(b, x, m, k, out, Wq, Wrowsum, wscale, /*relu=*/false);
    return;
  }
#endif
#ifdef __AVX512BF16__
  if (Wbf && _tilewright_compute_bf16()) {
    auto& xbf = _xbf_scratch();
    xbf.resize(k);
    _pack_bf16(x, xbf.data(), k);
    for (std::size_t i = 0; i < m; ++i) {
      std::size_t j   = 0;
      float v         = b[i] + _bf16_dot_pre(Wbf + i * k, xbf.data(), k, &j);
      const float* Wi = W + i * k;
      for (; j < k; ++j) v += Wi[j] * x[j];
      out[i] = v;
    }
    return;
  }
#endif
#ifdef __AVX512F__
  for (std::size_t i = 0; i < m; ++i) {
    __m512 acc      = _mm512_setzero_ps();
    std::size_t j   = 0;
    const float* Wi = W + i * k;
    for (; j + 16 <= k; j += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(Wi + j), _mm512_loadu_ps(x + j), acc);
    float v = b[i] + _mm512_reduce_add_ps(acc);
    for (; j < k; ++j) v += Wi[j] * x[j];
    out[i] = v;
  }
#else
  for (std::size_t i = 0; i < m; ++i) out[i] = b[i] + s_dot(W + i * k, x, k);
#endif
}
// (legacy scalar-only linear/linear_relu now folded into the unified
// definitions above, which handle AVX512 / scalar / bf16 internally.)

// ── feature-catalog dimensions (mirror the training feature library) ───────
//   query       = problem(21) + modular(18) + enum_tiles(16) = 55
//   item        = tile(12)                                   = 12
//   interaction = interaction(32) + hw_proxies(5)            = 37
// The former ml_kernel_params (11) item features and the
// ml_kernel_params_interaction (1) feature were removed 2026-06 (dead +
// train/serve-skew-prone; the base analytical scorer never read them).
constexpr std::size_t kQueryDim = 55;
constexpr std::size_t kItemDim  = 12;
constexpr std::size_t kInterDim = 37;

// ── dtype helpers (mirror the feature library's _BPE / _DTYPE_ID and the
//     analytical _DTYPE_BITS). Only the dtypes actually benched
//     (bf16/f16/f8/bf8/f32/xf32) are parity-critical. ──────────────────────
double bpe_for_dtype(DataType dt) {
  switch (dt) {
    case DataType::Float:
    case DataType::XFloat32:
    case DataType::Int32: return 4.0;
    case DataType::Double: return 8.0;
    case DataType::Half:
    case DataType::BFloat16: return 2.0;
    case DataType::Float8:
    case DataType::Float8_fnuz:
    case DataType::BFloat8:
    case DataType::BFloat8_fnuz:
    case DataType::Int8:
    case DataType::Float8BFloat8:
    case DataType::BFloat8Float8:
    case DataType::Float8BFloat8_fnuz:
    case DataType::BFloat8Float8_fnuz: return 1.0;
    case DataType::Float6:
    case DataType::BFloat6: return 0.75;
    case DataType::Float4:
    case DataType::Int4: return 0.5;
    default: return 2.0;
  }
}

int dtype_id(DataType dt) {
  switch (dt) {
    case DataType::Float: return 0;
    case DataType::XFloat32: return 1;
    case DataType::Half: return 2;
    case DataType::BFloat16: return 3;
    case DataType::Float8:
    case DataType::Float8_fnuz: return 4;
    case DataType::BFloat8:
    case DataType::BFloat8_fnuz: return 5;
    case DataType::Float6: return 6;
    case DataType::BFloat6: return 7;
    case DataType::Float4: return 8;
    case DataType::Int8: return 9;
    case DataType::Int32: return 10;
    case DataType::Float8BFloat8:
    case DataType::Float8BFloat8_fnuz: return 11;
    case DataType::BFloat8Float8:
    case DataType::BFloat8Float8_fnuz: return 12;
    case DataType::Double: return 13;
    case DataType::Int4: return 14;
    default: return 0;
  }
}

// analytical _dtype_to_bits (used by is_kernel_feasible).
int dtype_bits_feasible(DataType dt) {
  switch (dt) {
    case DataType::Float:
    case DataType::XFloat32: return 32;
    case DataType::Half:
    case DataType::BFloat16: return 16;
    case DataType::Float8:
    case DataType::Float8_fnuz:
    case DataType::BFloat8:
    case DataType::BFloat8_fnuz: return 8;
    default: return 16;  // bf16 fallback (matches Python)
  }
}

// Bits-per-element by dtype, used by the LDS-capacity gate below. Self-contained
// so the engine carries its own copy with no external dependency.
int datatype_to_bits(DataType type) {
  switch (type) {
    case DataType::Float: return 32;
    case DataType::Double: return 64;
    case DataType::ComplexFloat: return 64;
    case DataType::ComplexDouble: return 128;
    case DataType::Half: return 16;
    case DataType::Int8x4: return 32;
    case DataType::Int32: return 32;
    case DataType::BFloat16: return 16;
    case DataType::Int8: return 8;
    case DataType::Int4: return 4;
    case DataType::Int64: return 64;
    case DataType::XFloat32: return 32;
    case DataType::Float8_fnuz: return 8;
    case DataType::BFloat8_fnuz: return 8;
    case DataType::Float8BFloat8_fnuz: return 8;
    case DataType::BFloat8Float8_fnuz: return 8;
    case DataType::Float8: return 8;
    case DataType::BFloat8: return 8;
    case DataType::Float8BFloat8: return 8;
    case DataType::BFloat8Float8: return 8;
    case DataType::Float6: return 6;
    case DataType::BFloat6: return 6;
    case DataType::Float4: return 4;
    default: return -1;  // Invalid type
  }
}

inline double data_type_to_bytes(DataType type) {
  return static_cast<double>(datatype_to_bits(type)) / 8.0;
}

// LDS-capacity feasibility: does the macro tile's A+B LDS footprint fit in the
// hardware LDS budget? Self-contained so the engine has no external dependency.
bool check_lds_capacity(const Hardware& hardware,
                        const Dim3& mt,
                        const DataType& a_dtype,
                        const DataType& b_dtype) {
  // Unknown/None dtype (datatype_to_bits == -1) is not a valid kernel input;
  // treat as infeasible rather than letting a negative byte count pass the gate.
  if (datatype_to_bits(a_dtype) <= 0 || datatype_to_bits(b_dtype) <= 0) return false;
  const auto a_loads_in_bytes = mt.mk() * data_type_to_bytes(a_dtype);
  const auto b_loads_in_bytes = mt.nk() * data_type_to_bytes(b_dtype);
  const auto LDS_usage        = a_loads_in_bytes + b_loads_in_bytes;

  return LDS_usage <= static_cast<double>(hardware.lds_capacity);
}

// MI (MFMA) raw instruction cycles for gfx950 (same values origami's
// hardware INSTRUCTION_MAP exposes; the caller divides by parallel_mi_cu).
// Default 32 for any miss.
int mi_latency_raw(int mi_m, int mi_n, int mi_k, DataType dt) {
  switch (dt) {
    case DataType::Float:
      if (mi_m == 32 && mi_n == 32 && (mi_k == 2 || mi_k == 1)) return 64;
      if (mi_m == 16 && mi_n == 16 && (mi_k == 4 || mi_k == 1)) return 32;
      return 32;
    case DataType::BFloat16:
      if (mi_m == 32 && mi_n == 32 && mi_k == 16) return 32;
      if (mi_m == 16 && mi_n == 16 && mi_k == 32) return 16;
      return 32;
    case DataType::Half:
      if (mi_m == 32 && mi_n == 32 && (mi_k == 8 || mi_k == 16)) return 32;
      if (mi_m == 16 && mi_n == 16 && (mi_k == 16 || mi_k == 32)) return 16;
      return 32;
    case DataType::Float8:
    case DataType::Float8_fnuz:
      if (mi_m == 32 && mi_n == 32 && mi_k == 64) return 32;
      if (mi_m == 16 && mi_n == 16 && mi_k == 128) return 16;
      return 32;
    case DataType::BFloat8:
    case DataType::BFloat8_fnuz:
      if (mi_m == 32 && mi_n == 32 && mi_k == 64) return 32;
      if (mi_m == 16 && mi_n == 16 && mi_k == 128) return 16;
      return 32;
    case DataType::XFloat32:
      if (mi_m == 32 && mi_n == 32 && mi_k == 8) return 96;
      if (mi_m == 16 && mi_n == 16 && (mi_k == 16 || mi_k == 32)) return 48;
      return 32;
    default: return 32;
  }
}

// Hardware constants projected from the runtime Hardware -- queried from
// the SAME hardware the framework's feature path uses, so the two never drift.
struct HwView {
  double N_CU;
  double LDS;
  double L2;
  double parallel_mi_cu;
  double c0, c1, c2;  // mem_bw_per_wg_coefficients
};

HwView hw_view(const Hardware& h) {
  HwView v;
  v.N_CU           = static_cast<double>(h.N_CU);
  v.LDS            = static_cast<double>(h.lds_capacity);
  v.L2             = static_cast<double>(h.L2_capacity);
  v.parallel_mi_cu = static_cast<double>(h.parallel_mi_cu);
  v.c0             = std::get<0>(h.mem_bw_per_wg_coefficients);
  v.c1             = std::get<1>(h.mem_bw_per_wg_coefficients);
  v.c2             = std::get<2>(h.mem_bw_per_wg_coefficients);
  return v;
}

inline double dlog2(double x) { return std::log2(x); }
inline double is_pow2_d(double x) {
  long long xi = static_cast<long long>(x);
  return (xi > 0 && (xi & (xi - 1)) == 0) ? 1.0 : 0.0;
}

// ── feature builders -- exact mirror of the training feature library's
//     generic-feature builder, emitted in the same group/name order so the
//     output indices line up with query/item/interaction_feature_names().
//     All arithmetic is in double, then cast to float (matches Python: the
//     dict values are float64, np.asarray(..., float32) rounds once). ───────

void build_query_features(const Problem& p, const HwView& hw, float* out) {
  const double m         = static_cast<double>(p.size.m);
  const double n         = static_cast<double>(p.size.n);
  const double k         = static_cast<double>(p.size.k);
  const double b         = static_cast<double>(p.batch);
  const double bpe_a     = bpe_for_dtype(p.a_dtype);
  const double bpe_b     = bpe_for_dtype(p.b_dtype);
  const double bpe_c     = bpe_for_dtype(p.c_dtype);
  const double bpe_d     = bpe_for_dtype(p.d_dtype);
  const double flop_mult = (p.mi_dtype == DataType::XFloat32) ? 3.0 : 1.0;

  const double mn          = m * n;
  const double mk          = m * k;
  const double nk          = n * k;
  const double total_flops = flop_mult * 2.0 * m * n * k * b;
  const double total_bytes = mk * bpe_a * b + nk * bpe_b * b + mn * bpe_c * b + mn * bpe_d * b;
  const double ai_prob     = total_flops / std::max(total_bytes, 1.0);

  std::size_t i = 0;
  // problem group (21)
  out[i++] = static_cast<float>(dlog2(std::max(m, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(n, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(k, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(b, 1.0)));
  out[i++] = static_cast<float>(std::min(ai_prob, 1.0e6));
  out[i++] = static_cast<float>(dlog2(std::max(ai_prob, 0.001)));
  out[i++] = static_cast<float>(std::min(std::max(m / std::max(n, 1.0), 0.001), 10000.0));
  out[i++] = static_cast<float>(std::min(std::max(m / std::max(k, 1.0), 0.001), 10000.0));
  out[i++] = static_cast<float>(std::min(std::max(n / std::max(k, 1.0), 0.001), 10000.0));
  out[i++] = static_cast<float>(is_pow2_d(m));
  out[i++] = static_cast<float>(is_pow2_d(n));
  out[i++] = static_cast<float>(is_pow2_d(k));
  out[i++] = static_cast<float>(p.a_transpose == Transpose::T ? 1 : 0);
  out[i++] = static_cast<float>(p.b_transpose == Transpose::T ? 1 : 0);
  out[i++] = static_cast<float>(dtype_id(p.a_dtype));
  out[i++] = static_cast<float>(dtype_id(p.b_dtype));
  out[i++] = static_cast<float>(dtype_id(p.c_dtype));
  out[i++] = static_cast<float>(dtype_id(p.d_dtype));
  out[i++] = static_cast<float>(dtype_id(p.mi_dtype));
  out[i++] = static_cast<float>(bpe_a);
  out[i++] = static_cast<float>(bpe_b);

  // modular group (18) -- feature order: all m_mod, then n_mod, then
  // m_align (64,128,256), then n_align (64,128,256).
  const long long mi_ll = static_cast<long long>(p.size.m);
  const long long ni_ll = static_cast<long long>(p.size.n);
  for (int base : {256, 128, 64, 32, 16, 8}) out[i++] = static_cast<float>(mi_ll % base);
  for (int base : {256, 128, 64, 32, 16, 8}) out[i++] = static_cast<float>(ni_ll % base);
  for (int base : {64, 128, 256}) {
    long long r = mi_ll % base;
    out[i++]    = static_cast<float>(std::min(r, base - r) / static_cast<double>(base));
  }
  for (int base : {64, 128, 256}) {
    long long r = ni_ll % base;
    out[i++]    = static_cast<float>(std::min(r, base - r) / static_cast<double>(base));
  }

  // enum_tiles group (16)
  for (int st : {32, 64, 128, 256}) {
    double s_nt_m = std::ceil(m / st);
    double s_nt_n = std::ceil(n / st);
    out[i++]      = static_cast<float>(dlog2(std::max(s_nt_m * s_nt_n, 1.0)));
  }
  for (int st : {32, 64, 128, 256}) {
    double s_nt_m = std::ceil(m / st);
    double s_nt_n = std::ceil(n / st);
    out[i++]      = static_cast<float>(mn / std::max(s_nt_m * st * s_nt_n * st, 1.0));
  }
  for (int kd : {32, 64, 128, 256})
    out[i++] = static_cast<float>(dlog2(std::max(std::ceil(k / kd), 1.0)));
  for (int st : {128, 256}) {
    double s_nt = std::ceil(m / st) * std::ceil(n / st);
    double s_w  = std::ceil(s_nt / hw.N_CU);
    out[i++]    = static_cast<float>(dlog2(std::max(s_w, 1.0)));
  }
  for (int st : {128, 256}) {
    double s_nt = std::ceil(m / st) * std::ceil(n / st);
    double s_w  = std::ceil(s_nt / hw.N_CU);
    out[i++]    = static_cast<float>(s_nt / std::max(s_w * hw.N_CU, 1.0));
  }
  // i == 55
}

void build_item_features(const Config& c, float* out) {
  const double mt_m   = static_cast<double>(std::max<std::size_t>(c.mt.m, 1));
  const double mt_n   = static_cast<double>(std::max<std::size_t>(c.mt.n, 1));
  const double mt_k   = static_cast<double>(std::max<std::size_t>(c.mt.k, 1));
  const double mi_m   = static_cast<double>(std::max<std::size_t>(c.mi.m, 1));
  const double mi_n   = static_cast<double>(std::max<std::size_t>(c.mi.n, 1));
  const double mi_k   = static_cast<double>(std::max<std::size_t>(c.mi.k, 1));
  const double occ    = static_cast<double>(std::max<int>(c.occupancy, 1));
  const double grvw_a = static_cast<double>(std::max<std::size_t>(c.grvw_a, 1));
  const double grvw_b = static_cast<double>(std::max<std::size_t>(c.grvw_b, 1));
  const double gwvw_d = static_cast<double>(std::max<std::size_t>(c.gwvw_d, 1));

  std::size_t i = 0;
  // tile group (12)
  out[i++] = static_cast<float>(dlog2(mt_m));
  out[i++] = static_cast<float>(dlog2(mt_n));
  out[i++] = static_cast<float>(dlog2(mt_k));
  out[i++] = static_cast<float>(dlog2(mi_m));
  out[i++] = static_cast<float>(dlog2(mi_n));
  out[i++] = static_cast<float>(dlog2(mi_k));
  out[i++] = static_cast<float>(c.cache_hints_a / 7.0);
  out[i++] = static_cast<float>(c.cache_hints_b / 7.0);
  out[i++] = static_cast<float>(occ / 9.0);
  out[i++] = static_cast<float>(grvw_a / 8.0);
  out[i++] = static_cast<float>(grvw_b / 8.0);
  out[i++] = static_cast<float>(gwvw_d / 8.0);
  // i == 12
}

void build_inter_features(const Problem& p, const Config& c, const HwView& hw, float* out) {
  const double m      = static_cast<double>(p.size.m);
  const double n      = static_cast<double>(p.size.n);
  const double k      = static_cast<double>(p.size.k);
  const double b      = static_cast<double>(p.batch);
  const double mt_m   = static_cast<double>(std::max<std::size_t>(c.mt.m, 1));
  const double mt_n   = static_cast<double>(std::max<std::size_t>(c.mt.n, 1));
  const double mt_k   = static_cast<double>(std::max<std::size_t>(c.mt.k, 1));
  const double mi_m   = static_cast<double>(std::max<std::size_t>(c.mi.m, 1));
  const double mi_n   = static_cast<double>(std::max<std::size_t>(c.mi.n, 1));
  const double mi_k   = static_cast<double>(std::max<std::size_t>(c.mi.k, 1));
  const double grvw_a = static_cast<double>(std::max<std::size_t>(c.grvw_a, 1));
  const double grvw_b = static_cast<double>(std::max<std::size_t>(c.grvw_b, 1));

  const double bpe_a     = bpe_for_dtype(p.a_dtype);
  const double bpe_b     = bpe_for_dtype(p.b_dtype);
  const double bpe_c     = bpe_for_dtype(p.c_dtype);
  const double bpe_d     = bpe_for_dtype(p.d_dtype);
  const double flop_mult = (p.mi_dtype == DataType::XFloat32) ? 3.0 : 1.0;
  const double N_CU      = hw.N_CU;

  const double mn          = m * n;
  const double mk          = m * k;
  const double nk          = n * k;
  const double total_flops = flop_mult * 2.0 * m * n * k * b;
  const double total_bytes = mk * bpe_a * b + nk * bpe_b * b + mn * bpe_c * b + mn * bpe_d * b;

  const double nt_m            = std::ceil(m / mt_m);
  const double nt_n            = std::ceil(n / mt_n);
  const double num_tiles       = nt_m * nt_n;
  const double num_tiles_total = num_tiles * b;
  const double k_iters         = std::ceil(k / mt_k);

  const double waves             = std::ceil(num_tiles / N_CU);
  const double wave_eff          = waves > 0 ? num_tiles / (waves * N_CU) : 1.0;
  const double rho               = num_tiles_total / N_CU;
  const double batch_tiles_ratio = b * num_tiles / N_CU;

  const double launched_m = nt_m * mt_m;
  const double launched_n = nt_n * mt_n;
  const double launched_k = k_iters * mt_k;
  const double util_out   = mn / std::max(launched_m * launched_n, 1.0);
  const double util_3d    = (m * n * k) / std::max(launched_m * launched_n * launched_k, 1.0);

  const double lds_bytes      = mt_m * mt_k * bpe_a + mt_n * mt_k * bpe_b;
  const double lds_ratio      = lds_bytes / hw.LDS;
  const double l2_fit_ratio   = total_bytes / hw.L2;
  const double bw_per_cu      = total_bytes / N_CU;
  const double l2_working_set = (nt_m * mt_m * mt_k * bpe_a) + (nt_n * mt_n * mt_k * bpe_b);
  const double l2_fit_ws      = std::min(l2_working_set / hw.L2, 2.0) / 2.0;

  const double L_MI =
      mi_latency_raw(
          static_cast<int>(mi_m), static_cast<int>(mi_n), static_cast<int>(mi_k), p.mi_dtype) /
      std::max(hw.parallel_mi_cu, 1.0);
  const double n_mi = std::ceil(mt_m / mi_m) * std::ceil(mt_n / mi_n) * std::ceil(mt_k / mi_k);
  const double L_MT = n_mi * L_MI;
  const double ai_tile =
      (flop_mult * 2.0 * mt_m * mt_n * mt_k) / (mt_m * mt_k + mt_n * mt_k + mt_m * mt_n);
  const double active_cus = std::min(num_tiles_total, N_CU);
  const double bw_occ = std::min(1.0, hw.c0 * active_cus * active_cus + hw.c1 * active_cus + hw.c2);

  std::size_t i = 0;
  // interaction group (32)
  out[i++] = static_cast<float>(dlog2(std::max(num_tiles, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(num_tiles_total, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(k_iters, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(waves, 1.0)));
  out[i++] = static_cast<float>(wave_eff);
  out[i++] = static_cast<float>(dlog2(std::max(rho, 0.001)));
  out[i++] = static_cast<float>(dlog2(std::max(batch_tiles_ratio, 0.001)));
  out[i++] = static_cast<float>(util_out);
  out[i++] = static_cast<float>(util_3d);
  out[i++] = static_cast<float>(std::min((mt_m * mt_n) / std::max(mn, 1.0), 1.0));  // tile_coverage
  out[i++] = static_cast<float>(dlog2(std::max(lds_bytes, 1.0)));
  out[i++] = static_cast<float>(lds_ratio);
  out[i++] = static_cast<float>(std::min(l2_fit_ratio, 4.0) / 4.0);
  out[i++] = static_cast<float>(l2_fit_ws);
  out[i++] = static_cast<float>(dlog2(std::max(bw_per_cu, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(total_bytes, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(total_flops, 1.0)));
  out[i++] = static_cast<float>(std::min(1.0, mt_m / std::max(m, 1.0)));
  out[i++] = static_cast<float>(std::min(1.0, mt_n / std::max(n, 1.0)));
  out[i++] = static_cast<float>((k - (k_iters - 1.0) * mt_k) / std::max(k, 1.0));
  out[i++] = static_cast<float>(
      (std::fmod(k * bpe_a, 128.0) == 0.0 && std::fmod(mt_k * bpe_a, 128.0) == 0.0) ? 1.0 : 0.0);
  out[i++] = static_cast<float>((m <= 2.0 * mt_m) ? 1.0 : 0.0);
  out[i++] = static_cast<float>((n <= 2.0 * mt_n) ? 1.0 : 0.0);
  out[i++] = static_cast<float>((b > 1.0) ? 1.0 : 0.0);
  out[i++] = static_cast<float>(dlog2(std::max(mt_m * mt_k * bpe_a, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(mt_n * mt_k * bpe_b, 1.0)));
  out[i++] = static_cast<float>(
      dlog2(std::max((mt_m * mt_n * mt_k * 2.0 * flop_mult) /
                         std::max(mt_m * mt_k * bpe_a + mt_n * mt_k * bpe_b, 1.0),
                     0.001)));
  out[i++] = static_cast<float>(dlog2(std::max(k_iters * num_tiles * b, 1.0)));
  out[i++] = static_cast<float>(b * k_iters / std::max(num_tiles, 1.0));
  out[i++] = static_cast<float>(
      num_tiles_total > 0 ? num_tiles_total / (std::ceil(num_tiles_total / N_CU) * N_CU) : 1.0);
  out[i++] = static_cast<float>(dlog2(std::max(grvw_a * bpe_a, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(grvw_b * bpe_b, 1.0)));

  // hw_proxies group (5)
  out[i++] = static_cast<float>(ai_tile);
  out[i++] = static_cast<float>(L_MI);
  out[i++] = static_cast<float>(dlog2(std::max(L_MT, 1.0)));
  out[i++] = static_cast<float>(bw_occ);
  out[i++] = static_cast<float>(active_cus / N_CU);
  // i == 37
}

// ── per-cell trained model (one leaf of the split tree) ────────────────────
struct CellModel {
  std::string label;
  std::uint32_t embed_dim = 0, hidden_dim = 0, inter_hidden = 0;
  float temperature = 1.0f;
  std::vector<float> q_mean, q_std;  // kQueryDim
  std::vector<float> i_mean, i_std;  // kItemDim
  std::vector<float> x_mean, x_std;  // kInterDim
  // q_proj (depth 3): w0[h,q] b0[h]  w2[h,h] b2[h]  w4[e,h] b4[e]
  std::vector<float> q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  // i_proj (depth 2): w0[h,i] b0[h]  w2[e,h] b2[e]
  std::vector<float> i_w0, i_b0, i_w2, i_b2;
  // inter_mlp: w0[ih,x] b0[ih]  w2[1,ih] b2[1]
  std::vector<float> x_w0, x_b0, x_w2, x_b2;
  // P3b: bf16-packed mirrors of the weight matrices, built once at load when
  // TILEWRIGHT_COMPUTE_BF16=1, so the forward pass uses pre-converted weights
  // (half the weight bytes, no per-call fp32->bf16 conversion of W). Empty
  // when bf16 compute is off.
  std::vector<std::uint16_t> q_w0_bf, q_w2_bf, q_w4_bf, i_w0_bf, i_w2_bf, x_w0_bf, x_w2_bf;
  // P3b: int8 compute mirrors (per-tensor symmetric): int8 weights +
  // per-tensor scale + per-output-row sum(W_q) (for the VNNI u8*s8 offset
  // correction). Built once at load when TILEWRIGHT_COMPUTE_INT8=1.
  struct Int8W {
    std::vector<std::int8_t> q;        // row-major [m*k]
    std::vector<std::int32_t> rowsum;  // [m], sum of q over each row
    float scale = 1.0f;
  };
  Int8W q_w0_i8, q_w2_i8, q_w4_i8, i_w0_i8, i_w2_i8, x_w0_i8, x_w2_i8;
  // smart_K signature whitelist: 8-tuples (mt_m,mt_n,mt_k,mi_m,mi_n,mi_k,cha,chb)
  std::vector<std::array<int, 8>> smart_k_signatures;
};

// ── one node of the cumulative split tree (lib/subcells.SplitRule) ─────────
struct SplitRule {
  std::string cell;
  char axis     = 'M';  // 'M' / 'N' / 'K'
  int threshold = 0;    // inclusive on the lo side
  std::string lo_label;
  std::string hi_label;
};

struct LoadedModel {
  std::string feature_names_hash;
  std::string arch;
  std::uint32_t q_dim = 0, i_dim = 0, x_dim = 0;
  std::vector<CellModel> cells;
  std::unordered_map<std::string, std::size_t> cell_index;  // label -> cells[]
  std::unordered_map<std::string, SplitRule> splits;        // parent label -> rule
};

static LoadedModel g_model;
static bool g_loaded = false;
static std::once_flag g_load_once;

// ── multi-model handle registry ────────────────────────────────────────────
// Holds the models loaded via the handle API (load_model / load_model_by_index).
// Each model is heap-owned and CONST after load, so scoring against it needs no
// lock once the handle is in hand; the mutex only guards the registry
// containers (vector growth + the path->handle dedup map).
static std::mutex g_registry_mtx;
static std::vector<std::unique_ptr<LoadedModel>> g_models;
static std::unordered_map<std::string, int> g_path_to_handle;

constexpr std::uint8_t kDtypeFp32 = 0;
constexpr std::uint8_t kDtypeBf16 = 1;
constexpr std::uint8_t kDtypeInt8 = 2;
constexpr std::uint8_t kDtypeInt4 = 3;

// ── binary readers (MLREC_v1; mirror of the deploy stage's writer) ─────────
template <typename T>
bool read_pod(std::istream& f, T* out) {
  f.read(reinterpret_cast<char*>(out), sizeof(T));
  return f.gcount() == static_cast<std::streamsize>(sizeof(T));
}

bool read_bytes(std::istream& f, std::size_t n, std::string* out) {
  out->resize(n);
  if (n == 0) return true;
  f.read(&(*out)[0], static_cast<std::streamsize>(n));
  return f.gcount() == static_cast<std::streamsize>(n);
}

bool read_floats(std::istream& f, std::size_t n, std::vector<float>* out) {
  out->resize(n);
  if (n == 0) return true;
  f.read(reinterpret_cast<char*>(out->data()), sizeof(float) * n);
  return f.gcount() == static_cast<std::streamsize>(sizeof(float) * n);
}

// Read `n` weight values encoded as `dtype` and promote to float32:
//   0 fp32 : raw float32
//   1 bf16 : uint16, value = bits<<16 reinterpreted as float
//   2 int8 : a single float32 per-tensor scale, then n int8; value = q*scale
bool read_weights(std::istream& f, std::size_t n, std::vector<float>* out, std::uint8_t dtype) {
  if (dtype == kDtypeFp32) { return read_floats(f, n, out); }
  if (dtype == kDtypeBf16) {
    std::vector<std::uint16_t> tmp(n);
    if (n) {
      f.read(reinterpret_cast<char*>(tmp.data()), sizeof(std::uint16_t) * n);
      if (f.gcount() != static_cast<std::streamsize>(sizeof(std::uint16_t) * n)) return false;
    }
    out->resize(n);
    for (std::size_t j = 0; j < n; ++j) {
      std::uint32_t bits = static_cast<std::uint32_t>(tmp[j]) << 16;
      float v;
      std::memcpy(&v, &bits, sizeof(float));
      (*out)[j] = v;
    }
    return true;
  }
  if (dtype == kDtypeInt8) {
    float scale = 1.0f;
    if (!read_pod(f, &scale)) return false;
    std::vector<std::int8_t> q(n);
    if (n) {
      f.read(reinterpret_cast<char*>(q.data()), static_cast<std::streamsize>(n));
      if (f.gcount() != static_cast<std::streamsize>(n)) return false;
    }
    out->resize(n);
    for (std::size_t j = 0; j < n; ++j) (*out)[j] = scale * static_cast<float>(q[j]);
    return true;
  }
  if (dtype == kDtypeInt4) {
    // <float32 scale><ceil(n/2) bytes>: two signed 4-bit values per byte,
    // stored as (q+8) in [1,15]; low nibble = even idx, high = odd idx.
    float scale = 1.0f;
    if (!read_pod(f, &scale)) return false;
    const std::size_t nbytes = (n + 1) / 2;
    std::vector<std::uint8_t> packed(nbytes);
    if (nbytes) {
      f.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(nbytes));
      if (f.gcount() != static_cast<std::streamsize>(nbytes)) return false;
    }
    out->resize(n);
    for (std::size_t j = 0; j < n; ++j) {
      std::uint8_t byte = packed[j >> 1];
      int nib           = (j & 1) ? (byte >> 4) : (byte & 0x0F);
      (*out)[j]         = scale * static_cast<float>(nib - 8);
    }
    return true;
  }
  return false;
}

bool read_lenstr_u16(std::istream& f, std::string* out) {
  std::uint16_t len = 0;
  if (!read_pod(f, &len)) return false;
  return read_bytes(f, len, out);
}

bool read_lenstr_u32(std::istream& f, std::string* out) {
  std::uint32_t len = 0;
  if (!read_pod(f, &len)) return false;
  return read_bytes(f, len, out);
}

bool load_binary_stream(std::istream& f, LoadedModel* out) {
  if (!f) return false;

  char magic[8];
  f.read(magic, 8);
  if (f.gcount() != 8 || std::memcmp(magic, "MLREC_v1", 8) != 0) return false;

  std::uint32_t version = 0;
  if (!read_pod(f, &version)) return false;
  if (version != 1) return false;

  std::uint32_t endian = 0;
  if (!read_pod(f, &endian)) return false;
  if (endian != 0x01020304u) return false;  // little-endian sanity marker

  std::uint8_t wdt = kDtypeBf16;
  if (!read_pod(f, &wdt)) return false;
  if (wdt != kDtypeFp32 && wdt != kDtypeBf16 && wdt != kDtypeInt8 && wdt != kDtypeInt4)
    return false;

  if (!read_lenstr_u32(f, &out->feature_names_hash)) return false;
  if (!read_lenstr_u32(f, &out->arch)) return false;

  if (!read_pod(f, &out->q_dim)) return false;
  if (!read_pod(f, &out->i_dim)) return false;
  if (!read_pod(f, &out->x_dim)) return false;

  std::uint32_t n_cells = 0, n_splits = 0;
  if (!read_pod(f, &n_cells)) return false;
  if (!read_pod(f, &n_splits)) return false;

  // Splits.
  for (std::uint32_t s = 0; s < n_splits; ++s) {
    SplitRule r;
    if (!read_lenstr_u16(f, &r.cell)) return false;
    char axispair[2];
    f.read(axispair, 2);  // axis char + trailing NUL ('<cc')
    if (f.gcount() != 2) return false;
    r.axis           = axispair[0];
    std::int32_t thr = 0;
    if (!read_pod(f, &thr)) return false;
    r.threshold = static_cast<int>(thr);
    if (!read_lenstr_u16(f, &r.lo_label)) return false;
    if (!read_lenstr_u16(f, &r.hi_label)) return false;
    out->splits.emplace(r.cell, std::move(r));
  }

  // Cells.
  out->cells.resize(n_cells);
  for (std::uint32_t ci = 0; ci < n_cells; ++ci) {
    CellModel& cm = out->cells[ci];
    if (!read_lenstr_u16(f, &cm.label)) return false;
    if (!read_pod(f, &cm.embed_dim)) return false;
    if (!read_pod(f, &cm.hidden_dim)) return false;
    if (!read_pod(f, &cm.inter_hidden)) return false;
    if (!read_pod(f, &cm.temperature)) return false;

    if (!read_floats(f, out->q_dim, &cm.q_mean)) return false;
    if (!read_floats(f, out->q_dim, &cm.q_std)) return false;
    if (!read_floats(f, out->i_dim, &cm.i_mean)) return false;
    if (!read_floats(f, out->i_dim, &cm.i_std)) return false;
    if (!read_floats(f, out->x_dim, &cm.x_mean)) return false;
    if (!read_floats(f, out->x_dim, &cm.x_std)) return false;

    std::uint32_t n_sk = 0;
    if (!read_pod(f, &n_sk)) return false;
    cm.smart_k_signatures.resize(n_sk);
    for (std::uint32_t s = 0; s < n_sk; ++s) {
      std::int32_t sig[8];
      f.read(reinterpret_cast<char*>(sig), sizeof(sig));
      if (f.gcount() != static_cast<std::streamsize>(sizeof(sig))) return false;
      for (int t = 0; t < 8; ++t) cm.smart_k_signatures[s][t] = static_cast<int>(sig[t]);
    }

    const std::uint32_t qd = out->q_dim, id = out->i_dim, xd = out->x_dim;
    const std::uint32_t hd = cm.hidden_dim, ed = cm.embed_dim, ih = cm.inter_hidden;
    // Weight order from the deploy stage's writer (.weight uses the model's
    // weight dtype, .bias always fp32):
    if (!read_weights(f, std::size_t(hd) * qd, &cm.q_w0, wdt)) return false;
    if (!read_floats(f, hd, &cm.q_b0)) return false;
    if (!read_weights(f, std::size_t(hd) * hd, &cm.q_w2, wdt)) return false;
    if (!read_floats(f, hd, &cm.q_b2)) return false;
    if (!read_weights(f, std::size_t(ed) * hd, &cm.q_w4, wdt)) return false;
    if (!read_floats(f, ed, &cm.q_b4)) return false;
    if (!read_weights(f, std::size_t(hd) * id, &cm.i_w0, wdt)) return false;
    if (!read_floats(f, hd, &cm.i_b0)) return false;
    if (!read_weights(f, std::size_t(ed) * hd, &cm.i_w2, wdt)) return false;
    if (!read_floats(f, ed, &cm.i_b2)) return false;
    if (!read_weights(f, std::size_t(ih) * xd, &cm.x_w0, wdt)) return false;
    if (!read_floats(f, ih, &cm.x_b0)) return false;
    if (!read_weights(f, std::size_t(1) * ih, &cm.x_w2, wdt)) return false;
    if (!read_floats(f, 1, &cm.x_b2)) return false;

#ifdef __AVX512BF16__
    // P3b: build bf16 mirrors once so the bf16 forward uses pre-converted
    // weights (no per-call W conversion). Only when bf16 compute is enabled.
    if (_tilewright_compute_bf16()) {
      auto pack = [](const std::vector<float>& src, std::vector<std::uint16_t>& dst) {
        dst.resize(src.size());
        if (!src.empty()) _pack_bf16(src.data(), dst.data(), src.size());
      };
      pack(cm.q_w0, cm.q_w0_bf);
      pack(cm.q_w2, cm.q_w2_bf);
      pack(cm.q_w4, cm.q_w4_bf);
      pack(cm.i_w0, cm.i_w0_bf);
      pack(cm.i_w2, cm.i_w2_bf);
      pack(cm.x_w0, cm.x_w0_bf);
      pack(cm.x_w2, cm.x_w2_bf);
    }
#endif
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    if (_tilewright_compute_int8()) {
      // quantize each weight matrix to per-tensor symmetric int8 + per-row sum
      auto q8 =
          [](const std::vector<float>& src, std::size_t m, std::size_t k, CellModel::Int8W& dst) {
            float amax = 0.0f;
            for (float w : src) {
              float a = std::fabs(w);
              if (a > amax) amax = a;
            }
            dst.scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
            float inv = 1.0f / dst.scale;
            dst.q.resize(src.size());
            dst.rowsum.assign(m, 0);
            for (std::size_t i = 0; i < m; ++i) {
              std::int32_t rs = 0;
              for (std::size_t j = 0; j < k; ++j) {
                int q            = (int)std::lround(src[i * k + j] * inv);
                q                = q < -127 ? -127 : (q > 127 ? 127 : q);
                dst.q[i * k + j] = (std::int8_t)q;
                rs += q;
              }
              dst.rowsum[i] = rs;
            }
          };
      q8(cm.q_w0, hd, qd, cm.q_w0_i8);
      q8(cm.q_w2, hd, hd, cm.q_w2_i8);
      q8(cm.q_w4, ed, hd, cm.q_w4_i8);
      q8(cm.i_w0, hd, id, cm.i_w0_i8);
      q8(cm.i_w2, ed, hd, cm.i_w2_i8);
      q8(cm.x_w0, ih, xd, cm.x_w0_i8);
      q8(cm.x_w2, 1, ih, cm.x_w2_i8);
    }
#endif
    out->cell_index.emplace(cm.label, static_cast<std::size_t>(ci));
  }

  // Trailer (best-effort: validate if present).
  char trailer[8];
  f.read(trailer, 8);
  if (f.gcount() == 8 && std::memcmp(trailer, "MLRECEND", 8) != 0) return false;

  return true;
}

bool load_binary(const char* path, LoadedModel* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  return load_binary_stream(f, out);
}

// ── two-tower helpers (map to GenericTwoTower q_proj / i_proj / inter_mlp) ──

// Whiten then run q_proj (Linear,ReLU,Linear,ReLU,Linear) -> query embedding.
void compute_qe(const CellModel& cm,
                const float* q_feat,
                std::size_t q_dim,
                std::vector<float>& scratch,
                std::vector<float>& q_emb_out) {
  std::vector<float> q_norm(q_dim);
  for (std::size_t j = 0; j < q_dim; ++j) {
    float s   = (cm.q_std[j] < 1e-6f) ? 1.0f : cm.q_std[j];
    q_norm[j] = (q_feat[j] - cm.q_mean[j]) / s;
  }
  std::vector<float> h0(cm.hidden_dim), h2(cm.hidden_dim);
  scratch.assign(cm.hidden_dim, 0.0f);
  linear_relu(cm.q_w0.data(),
              cm.q_b0.data(),
              q_norm.data(),
              cm.hidden_dim,
              q_dim,
              h0.data(),
              cm.q_w0_bf.empty() ? nullptr : cm.q_w0_bf.data(),
              cm.q_w0_i8.q.empty() ? nullptr : cm.q_w0_i8.q.data(),
              cm.q_w0_i8.rowsum.data(),
              cm.q_w0_i8.scale);
  linear_relu(cm.q_w2.data(),
              cm.q_b2.data(),
              h0.data(),
              cm.hidden_dim,
              cm.hidden_dim,
              h2.data(),
              cm.q_w2_bf.empty() ? nullptr : cm.q_w2_bf.data(),
              cm.q_w2_i8.q.empty() ? nullptr : cm.q_w2_i8.q.data(),
              cm.q_w2_i8.rowsum.data(),
              cm.q_w2_i8.scale);
  q_emb_out.assign(cm.embed_dim, 0.0f);
  linear(cm.q_w4.data(),
         cm.q_b4.data(),
         h2.data(),
         cm.embed_dim,
         cm.hidden_dim,
         q_emb_out.data(),
         cm.q_w4_bf.empty() ? nullptr : cm.q_w4_bf.data(),
         cm.q_w4_i8.q.empty() ? nullptr : cm.q_w4_i8.q.data(),
         cm.q_w4_i8.rowsum.data(),
         cm.q_w4_i8.scale);
}

// Whiten then run i_proj (Linear,ReLU,Linear) -> item embedding.
void compute_ie(const CellModel& cm,
                const float* i_feat,
                std::size_t i_dim,
                std::vector<float>& scratch,
                std::vector<float>& i_emb_out) {
  std::vector<float> i_norm(i_dim);
  for (std::size_t j = 0; j < i_dim; ++j) {
    float s   = (cm.i_std[j] < 1e-6f) ? 1.0f : cm.i_std[j];
    i_norm[j] = (i_feat[j] - cm.i_mean[j]) / s;
  }
  scratch.assign(cm.hidden_dim, 0.0f);
  linear_relu(cm.i_w0.data(),
              cm.i_b0.data(),
              i_norm.data(),
              cm.hidden_dim,
              i_dim,
              scratch.data(),
              cm.i_w0_bf.empty() ? nullptr : cm.i_w0_bf.data(),
              cm.i_w0_i8.q.empty() ? nullptr : cm.i_w0_i8.q.data(),
              cm.i_w0_i8.rowsum.data(),
              cm.i_w0_i8.scale);
  i_emb_out.assign(cm.embed_dim, 0.0f);
  linear(cm.i_w2.data(),
         cm.i_b2.data(),
         scratch.data(),
         cm.embed_dim,
         cm.hidden_dim,
         i_emb_out.data(),
         cm.i_w2_bf.empty() ? nullptr : cm.i_w2_bf.data(),
         cm.i_w2_i8.q.empty() ? nullptr : cm.i_w2_i8.q.data(),
         cm.i_w2_i8.rowsum.data(),
         cm.i_w2_i8.scale);
}

// Item-embedding accessor. The 8-int signature does NOT capture every input
// to the item tower (grvw/gwvw/occupancy), so we recompute per candidate
// rather than cache by signature -- correctness over speed, since the parity
// bar is exact. Kept as a named entry point per the deployed API.
void get_or_compute_ie(const CellModel& cm,
                       const Config& cfg,
                       std::size_t i_dim,
                       std::vector<float>& scratch,
                       std::vector<float>& i_emb_out) {
  std::array<float, kItemDim> item_feat;
  build_item_features(cfg, item_feat.data());
  compute_ie(cm, item_feat.data(), i_dim, scratch, i_emb_out);
}

// score = dot(query_embed, item_embed) / temperature   (the embedding term;
// the interaction-MLP term is added by the caller). Mirrors
// GenericTwoTower.score_pairs: dot = (eg*et).sum(-1) / T,  T = |temp|.clamp(0.1).
float score_from_embeds(const CellModel& cm,
                        const float* q_emb,
                        const float* i_emb,
                        std::size_t embed_dim) {
  float temp = std::max(std::fabs(cm.temperature), 0.1f);
  float dotp = s_dot(q_emb, i_emb, embed_dim);
  return dotp / temp;
}

// Whiten interaction features then run inter_mlp (Linear,ReLU,Linear) -> scalar.
float compute_inter_score(const CellModel& cm,
                          const float* x_feat,
                          std::size_t x_dim,
                          std::vector<float>& scratch) {
  std::vector<float> x_norm(x_dim);
  for (std::size_t j = 0; j < x_dim; ++j) {
    float s   = (cm.x_std[j] < 1e-6f) ? 1.0f : cm.x_std[j];
    x_norm[j] = (x_feat[j] - cm.x_mean[j]) / s;
  }
  scratch.assign(cm.inter_hidden, 0.0f);
  linear_relu(cm.x_w0.data(),
              cm.x_b0.data(),
              x_norm.data(),
              cm.inter_hidden,
              x_dim,
              scratch.data(),
              cm.x_w0_bf.empty() ? nullptr : cm.x_w0_bf.data(),
              cm.x_w0_i8.q.empty() ? nullptr : cm.x_w0_i8.q.data(),
              cm.x_w0_i8.rowsum.data(),
              cm.x_w0_i8.scale);
  return cm.x_b2[0] + s_dot(cm.x_w2.data(), scratch.data(), cm.inter_hidden);
}

// ── split-tree routing (lib/grid.cell_key + lib/subcells.assign_subcell) ───
std::string m_tier(std::size_t v) {
  if (v <= 32) return "Tiny";
  if (v <= 128) return "Small";
  if (v <= 512) return "Mid";
  return "Large";
}
std::string k_tier(std::size_t v) {
  if (v <= 32) return "TinyK";
  if (v <= 512) return "MidK";
  return "LargeK";
}
std::string b_tier(std::size_t v) { return v == 1 ? "Bnone" : "Bany"; }

std::string base_cell_label(const Problem& p) {
  return m_tier(p.size.m) + "|" + m_tier(p.size.n) + "|" + k_tier(p.size.k) + "|" + b_tier(p.batch);
}

long long axis_value(char axis, const Problem& p) {
  switch (axis) {
    case 'M': return static_cast<long long>(p.size.m);
    case 'N': return static_cast<long long>(p.size.n);
    case 'K': return static_cast<long long>(p.size.k);
    case 'B': return static_cast<long long>(p.batch);
    default: return 0;
  }
}

// Walk the cumulative split tree from the base 96-grid label down to its leaf.
std::string assign_subcell(const LoadedModel& mdl, const Problem& p) {
  std::string label = base_cell_label(p);
  std::unordered_map<std::string, bool> seen;  // cycle guard
  for (;;) {
    auto it = mdl.splits.find(label);
    if (it == mdl.splits.end()) break;
    if (seen[label]) break;
    seen[label]        = true;
    const SplitRule& r = it->second;
    long long v        = axis_value(r.axis, p);
    label              = (v <= r.threshold) ? r.lo_label : r.hi_label;
  }
  return label;
}

// Resolve to the nearest ancestor cell (incl. the leaf) that has a trained
// model. Mirrors the deployed Python pick routine's cell resolution. Returns
// the cells[] index, or -1.
int resolve_model_cell_index(const LoadedModel& mdl, const Problem& p) {
  std::string cur = assign_subcell(mdl, p);
  for (;;) {
    auto it = mdl.cell_index.find(cur);
    if (it != mdl.cell_index.end()) return static_cast<int>(it->second);
    std::size_t j = cur.rfind('#');
    if (j == std::string::npos) return -1;
    cur = cur.substr(0, j);
  }
}

// ── feasibility filter (byte-for-byte mirror of the analytical
//     is_kernel_feasible check) ───────────────────────────────────────────
bool is_kernel_feasible(const Problem& p, const Config& c) {
  const long long M      = static_cast<long long>(p.size.m);
  const long long N      = static_cast<long long>(p.size.n);
  const long long K      = static_cast<long long>(p.size.k);
  const long long B      = static_cast<long long>(p.batch);
  const long long MT_M   = static_cast<long long>(c.mt.m);
  const long long MT_N   = static_cast<long long>(c.mt.n);
  const long long MT_K   = static_cast<long long>(c.mt.k);
  const long long MI_M   = static_cast<long long>(c.mi.m);
  const long long MI_N   = static_cast<long long>(c.mi.n);
  const long long MI_K   = static_cast<long long>(c.mi.k);
  const int cha          = c.cache_hints_a;
  const int chb          = c.cache_hints_b;
  const bool a_trans     = (p.a_transpose == Transpose::T);
  const bool b_trans     = (p.b_transpose == Transpose::T);
  const long long a_bits = dtype_bits_feasible(p.a_dtype);
  const long long b_bits = dtype_bits_feasible(p.b_dtype);

  // 1) Small problem with batch -- must fit in one tile.
  if (M <= 256 && N <= 256 && K < 1024 && B != 1 && (MT_M < M || MT_N < N)) return false;
  // 2) Dot2 is only correct for M < 3.
  if (MI_M == 1 && MI_N == 1 && MI_K == 64 && M > 2) return false;
  // 3) NTA/NTB rules.
  long long K_mod_128b    = (K * a_bits) % 1024;
  long long MT_K_mod_128b = (MT_K * a_bits) % 1024;
  if (K_mod_128b == 0 && MT_K_mod_128b == 0) {
    if (M <= MT_M * 2 && (!b_trans) && ((N * b_bits) / std::max<long long>(M * a_bits, 1) > 5)) {
      if (chb != 4) return false;
    } else if (N <= MT_N * 2 && a_trans &&
               ((M * a_bits) / std::max<long long>(N * b_bits, 1) > 5)) {
      if (cha != 4) return false;
    } else {
      if (cha || chb) return false;
    }
  } else if (cha || chb) {
    return false;
  }
  return true;
}

std::array<int, 8> sig_of(const Config& c) {
  return {static_cast<int>(c.mt.m),
          static_cast<int>(c.mt.n),
          static_cast<int>(c.mt.k),
          static_cast<int>(c.mi.m),
          static_cast<int>(c.mi.n),
          static_cast<int>(c.mi.k),
          c.cache_hints_a,
          c.cache_hints_b};
}

// ── model-agnostic scoring core ────────────────────────────────────────────
// These operate on ANY LoadedModel (const after load), so both the singleton
// and the handle API funnel through the SAME math. The singleton path is
// byte-for-byte unchanged: load_weights() populates g_model and route()/
// rank_configs() forward g_model to these helpers.

// Route a problem to its leaf model-cell index for `model`. Honors the
// TILEWRIGHT_FORCE_CELL override exactly like the legacy singleton route().
int route_impl(const LoadedModel& model, const Problem& problem) {
  const char* env = std::getenv("TILEWRIGHT_FORCE_CELL");
  if (env) {
    int forced = std::atoi(env);
    if (forced >= 0) return forced;
  }
  return resolve_model_cell_index(model, problem);
}

// All configs unscored (caller treats unscored as NaN latency). Shared by the
// impl below and by the singleton wrappers when no model is available.
std::vector<Result> unscored_all(const std::vector<Config>& configs) {
  std::vector<Result> result;
  result.reserve(configs.size());
  for (std::size_t j = 0; j < configs.size(); ++j) result.push_back(Result{j, 0.0, false});
  return result;
}

// The deployed ranker -- exact mirror of the deployed Python pick routine, run
// against `model`. (Body lifted verbatim from the legacy singleton
// rank_configs; only the model source changed from g_model to `model`.)
std::vector<Result> rank_configs_impl(const LoadedModel& model,
                                      const Problem& problem,
                                      const Hardware& hardware,
                                      const std::vector<Config>& configs,
                                      std::size_t min_scored) {
  std::vector<Result> result;

  auto fallback_all = [&]() {
    result.reserve(configs.size());
    for (std::size_t j = 0; j < configs.size(); ++j) result.push_back(Result{j, 0.0, false});
    return result;
  };

  if (configs.empty()) return fallback_all();

  int cell_idx = resolve_model_cell_index(model, problem);
  if (cell_idx < 0) return fallback_all();
  const CellModel& cm = model.cells[static_cast<std::size_t>(cell_idx)];

  const HwView hw         = hw_view(hardware);
  const std::size_t q_dim = model.q_dim;
  const std::size_t i_dim = model.i_dim;
  const std::size_t x_dim = model.x_dim;

  // smart_K signature whitelist for this cell (may be empty -> no filter).
  const bool have_sk = !cm.smart_k_signatures.empty();

  auto sig_in_set = [&](const std::array<int, 8>& s) -> bool {
    for (const auto& w : cm.smart_k_signatures)
      if (w == s) return true;
    return false;
  };

  // Two-pass survivor scan (LDS gate -> is_kernel_feasible -> optional
  // smart_K signature filter), with the C++/Python shared fallback: if the
  // signature filter rejects every config, re-scan without it.
  auto scan = [&](bool use_sig_filter, std::vector<std::uint32_t>& out_idx) {
    out_idx.clear();
    for (std::uint32_t ci = 0; ci < configs.size(); ++ci) {
      const Config& cc = configs[ci];
      // LDS budget gate (tilewright::check_lds_capacity).
      if (!check_lds_capacity(hardware, cc.mt, problem.a_dtype, problem.b_dtype)) continue;
      if (!is_kernel_feasible(problem, cc)) continue;
      if (use_sig_filter && have_sk) {
        if (!sig_in_set(sig_of(cc))) continue;
      }
      out_idx.push_back(ci);
    }
  };

  // Tier-1 candidates: smart_K whitelist survivors (the high-confidence set the
  // model was calibrated on for this cell). This is the default scored set.
  std::vector<std::uint32_t> cand;  // tier-1
  scan(/*use_sig_filter=*/true, cand);
  bool tier1_is_whitelist = have_sk;
  if (cand.empty() && have_sk) {
    // Legacy fallback: whitelist rejected everything -> score the full feasible
    // set (this already IS the full set, so no tier-2 extension applies).
    scan(/*use_sig_filter=*/false, cand);
    tier1_is_whitelist = false;
  }

  if (cand.empty()) return fallback_all();

  // Tier-2 (depth-on-demand): when the caller asks for more solutions than the
  // whitelist provides (min_scored > tier-1 size), also score the remaining
  // LDS+feasible configs. They are ranked strictly AFTER all tier-1 results, so
  // the top-|tier-1| picks -- and hence the default top-1 -- are byte-identical
  // to min_scored == 0. Cost (one item-tower pass per extra config) is paid only
  // when the deeper list is actually requested.
  std::vector<std::uint32_t> cand2;
  if (tier1_is_whitelist && min_scored > cand.size()) {
    std::vector<std::uint32_t> feasible;
    scan(/*use_sig_filter=*/false, feasible);
    std::vector<char> in_tier1(configs.size(), 0);
    for (std::uint32_t ci : cand) in_tier1[ci] = 1;
    for (std::uint32_t ci : feasible)
      if (!in_tier1[ci]) cand2.push_back(ci);
  }

  // Query tower (problem-only) computed once, shared by both tiers.
  std::array<float, kQueryDim> q_feat;
  build_query_features(problem, hw, q_feat.data());
  std::vector<float> scratch, q_emb;
  compute_qe(cm, q_feat.data(), q_dim, scratch, q_emb);

  // Score a candidate set: total = dot(qe, ie)/temp + inter_mlp(x). The score of
  // a config is independent of which tier it lands in (pure per-config function).
  std::vector<float> i_emb;
  std::array<float, kInterDim> x_feat;
  auto score_set = [&](const std::vector<std::uint32_t>& cset,
                       std::vector<std::pair<std::uint32_t, float>>& out) {
    out.reserve(cset.size());
    for (std::uint32_t ci : cset) {
      const Config& cc = configs[ci];
      get_or_compute_ie(cm, cc, i_dim, scratch, i_emb);
      float emb_score = score_from_embeds(cm, q_emb.data(), i_emb.data(), cm.embed_dim);
      build_inter_features(problem, cc, hw, x_feat.data());
      float inter_score = compute_inter_score(cm, x_feat.data(), x_dim, scratch);
      out.emplace_back(ci, emb_score + inter_score);
    }
    // Best-first. stable_sort so ties keep input order -> first-max wins,
    // matching numpy/torch argmax.
    std::stable_sort(out.begin(),
                     out.end(),
                     [](const std::pair<std::uint32_t, float>& a,
                        const std::pair<std::uint32_t, float>& b) { return a.second > b.second; });
  };

  std::vector<std::pair<std::uint32_t, float>> scored;
  score_set(cand, scored);
  if (!cand2.empty()) {
    std::vector<std::pair<std::uint32_t, float>> scored2;
    score_set(cand2, scored2);
    scored.insert(scored.end(), scored2.begin(), scored2.end());  // tier-2 after tier-1
  }

  // One [TILEWRIGHT_PICK] stderr line per ranked call (gated by TILEWRIGHT_PICK_LOG).
  // The deploy parity check parses this to compare the C++ top-1 pick against
  // the Python student's pick (the C++<->Python parity check).
  if (std::getenv("TILEWRIGHT_PICK_LOG") != nullptr && !scored.empty()) {
    const Config& top = configs[scored[0].first];
    std::fprintf(stderr,
                 "[TILEWRIGHT_PICK] m=%zu n=%zu k=%zu b=%zu tA=%c tB=%c leaf=%s "
                 "top1_sig=(mt_m=%zu,mt_n=%zu,mt_k=%zu,mi_m=%zu,mi_n=%zu,"
                 "mi_k=%zu,cha=%d,chb=%d) top1_score=%f n_configs=%zu\n",
                 problem.size.m,
                 problem.size.n,
                 problem.size.k,
                 problem.batch,
                 (problem.a_transpose == Transpose::T ? 'T' : 'N'),
                 (problem.b_transpose == Transpose::T ? 'T' : 'N'),
                 cm.label.c_str(),
                 top.mt.m,
                 top.mt.n,
                 top.mt.k,
                 top.mi.m,
                 top.mi.n,
                 top.mi.k,
                 top.cache_hints_a,
                 top.cache_hints_b,
                 static_cast<double>(scored[0].second),
                 configs.size());
    std::fflush(stderr);
  }

  result.reserve(configs.size());
  std::vector<char> used(configs.size(), 0);
  for (const auto& s : scored) {
    used[s.first] = 1;
    result.push_back(
        Result{static_cast<std::size_t>(s.first), static_cast<double>(s.second), true});
  }
  // Append the non-scored configs (filtered out) with scored == false so the
  // returned vector still covers every input config.
  for (std::size_t j = 0; j < configs.size(); ++j) {
    if (!used[j]) result.push_back(Result{j, 0.0, false});
  }
  return result;
}

}  // namespace

bool load_weights(const std::string& bin_path) {
  LoadedModel m;
  if (!load_binary(bin_path.c_str(), &m)) return false;
  g_model  = std::move(m);
  g_loaded = true;
  // One-shot weights-load banner (gated by TILEWRIGHT_DIAG). The deploy parity
  // check keys off this line to confirm the .bin was actually picked up
  // rather than silently falling back to the analytical scorer; its absence on
  // a tilewright-on bench is a HARD FAIL.
  if (std::getenv("TILEWRIGHT_DIAG") != nullptr) {
    std::fprintf(stderr,
                 "[TILEWRIGHT_DIAG FILE] arch=%s qhash=%s qdim=%u idim=%u xdim=%u "
                 "n_cells=%zu n_splits=%zu\n",
                 g_model.arch.c_str(),
                 g_model.feature_names_hash.c_str(),
                 g_model.q_dim,
                 g_model.i_dim,
                 g_model.x_dim,
                 g_model.cells.size(),
                 g_model.splits.size());
    std::fflush(stderr);
  }
  return true;
}

bool weights_loaded() { return g_loaded; }

namespace {

// Directory containing the loaded libtilewright (via dladdr). Empty on failure.
std::string self_library_dir() {
#ifdef __unix__
  Dl_info info;
  // Resolve a symbol that lives in THIS shared object.
  if (dladdr(reinterpret_cast<void*>(&load_binary_stream), &info) && info.dli_fname) {
    std::string path(info.dli_fname);
    std::size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) return path.substr(0, slash);
  }
#endif
  return std::string();
}

// Resolve a process-wide singleton weights .bin by searching relative to the
// loaded library -- the same place the kernel library ships. (The deployed
// hipBLASLt path uses the per-library handle API instead; see tilewright_data_dir.)
void ensure_weights() {
  if (g_loaded) return;
  std::call_once(g_load_once, []() {
    if (g_loaded) return;
    const std::string dir = self_library_dir();
    if (!dir.empty()) {
      const char* rels[] = {
          "/tilewright_weights.bin",
          "/hipblaslt/library/tilewright_weights.bin",
      };
      for (const char* rel : rels) {
        if (load_weights(dir + rel)) return;
      }
    }
    std::fprintf(stderr, "[tilewright] no singleton weights found relative to the library\n");
  });
}

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

// Find the directory holding the per-library tilewright weights + their
// "tilewright_index". This is the SAME directory the Tensile logic file (.dat)
// for this library was loaded from: the weights ship co-located with the kernel
// library, so they are discovered exactly the way the kernels are. `hint_dir` is
// that directory, passed by the loader (dirname of the logic file). Returns it
// when it contains a tilewright_index; empty otherwise (the caller then uses the
// base analytical path).
std::string tilewright_data_dir(const std::string& hint_dir = "") {
  if (!hint_dir.empty() && file_exists(hint_dir + "/tilewright_index")) return hint_dir;
  return std::string();
}

}  // namespace

int route(const Problem& problem) {
  ensure_weights();
  if (!g_loaded) return -1;
  return route_impl(g_model, problem);
}

// ── the deployed ranker -- exact mirror of the deployed Python pick routine ─
std::vector<Result> rank_configs(const Problem& problem,
                                 const Hardware& hardware,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored) {
  ensure_weights();
  if (!g_loaded) return unscored_all(configs);
  return rank_configs_impl(g_model, problem, hardware, configs, min_scored);
}

// ── multi-model handle API ─────────────────────────────────────────────────

int load_model(const std::string& bin_path) {
  std::lock_guard<std::mutex> lock(g_registry_mtx);
  // Dedup: same path -> same handle (don't reparse).
  auto it = g_path_to_handle.find(bin_path);
  if (it != g_path_to_handle.end()) return it->second;

  auto m = std::make_unique<LoadedModel>();
  if (!load_binary(bin_path.c_str(), m.get())) return -1;

  if (std::getenv("TILEWRIGHT_DIAG") != nullptr) {
    std::fprintf(stderr,
                 "[TILEWRIGHT_DIAG FILE] handle=%d path=%s arch=%s qhash=%s "
                 "qdim=%u idim=%u xdim=%u n_cells=%zu n_splits=%zu\n",
                 static_cast<int>(g_models.size()),
                 bin_path.c_str(),
                 m->arch.c_str(),
                 m->feature_names_hash.c_str(),
                 m->q_dim,
                 m->i_dim,
                 m->x_dim,
                 m->cells.size(),
                 m->splits.size());
    std::fflush(stderr);
  }

  const int handle = static_cast<int>(g_models.size());
  g_models.push_back(std::move(m));
  g_path_to_handle.emplace(bin_path, handle);
  return handle;
}

namespace {
// Fetch a const model pointer for a handle, or nullptr if out of range. Guards
// the registry vector read with the mutex (cheap; the LoadedModel itself is
// const after load, so the pointer stays valid without holding the lock).
const LoadedModel* model_for_handle(int handle) {
  if (handle < 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_registry_mtx);
  if (static_cast<std::size_t>(handle) >= g_models.size()) return nullptr;
  return g_models[static_cast<std::size_t>(handle)].get();
}
}  // namespace

std::vector<Result> rank_configs(int handle,
                                 const Problem& p,
                                 const Hardware& hw,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored) {
  const LoadedModel* m = model_for_handle(handle);
  if (m == nullptr) return unscored_all(configs);
  return rank_configs_impl(*m, p, hw, configs, min_scored);
}

int load_model_by_index(const std::string& logic_stem, const std::string& hint_dir) {
  const std::string data_dir = tilewright_data_dir(hint_dir);
  if (data_dir.empty()) return -1;

  std::ifstream idx(data_dir + "/tilewright_index");
  if (!idx) return -1;

  // Hand-parse the 2-column text index: "<logic_stem>\t<weights_filename>".
  // '#' comments and blank lines are ignored; any whitespace separates the
  // columns. No JSON dependency.
  std::string line;
  while (std::getline(idx, line)) {
    // Strip a trailing CR (CRLF files).
    if (!line.empty() && line.back() == '\r') line.pop_back();
    // Drop comments.
    std::size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);

    std::istringstream ls(line);
    std::string stem, weights_file;
    if (!(ls >> stem >> weights_file)) continue;  // blank / malformed -> skip
    if (stem == logic_stem) return load_model(data_dir + "/" + weights_file);
  }
  return -1;
}

// Note: there is intentionally no eager-init / global-weights warming. In the
// per-library model scheme each PredictionLibrary loads its own model lazily at
// deserialize (via tilewright_index, co-located with the kernel library). The
// singleton API (ensure_weights/load_weights) remains only for the standalone
// tests and Python bindings, which load a model explicitly.

}  // namespace tilewright
