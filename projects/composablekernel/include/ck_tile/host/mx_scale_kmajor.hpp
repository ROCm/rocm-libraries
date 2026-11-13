// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_spec.hpp"

/**
 * @file
 * Host side of MxGemmPipelineAgBgCrCompTDMScheduled (rocKE's mx_gemm): the K-major E8M0 scale
 * layout, bit-exact element codecs, the output rounding of its epilogue and a reference GEMM.
 *
 * Scales. One E8M0 byte per 32 K values of each A or B row. The natural layout is
 * [rows][ceil(K / 32)]; the kernel reads [ceil(K / 128)][rows][4]: K step s of row r is the int32
 * at (s * rows + r) * 4, its byte g the scale of K values s * 128 + g * 32 .. + 31. A K step past
 * ceil(K / 32) groups (a partial last K step) is zero-padded.
 *
 * Elements. A and B rows are K-contiguous bytes: FP4 two per byte (K even in the low nibble), FP8
 * E4M3 (OCP, no infinity) and BF8 E5M2 one per byte. C and D are row-major in the output type.
 */

namespace ck_tile {

/// Bytes of a K-major scale array of `rows` rows.
CK_TILE_HOST std::size_t mx_scale_kmajor_bytes(index_t rows, long_index_t K)
{
    return static_cast<std::size_t>(mx_tdm_scheduled::ceil_div(K, mx_tdm_scheduled::kAtomK)) *
           rows * 4;
}

/// [rows][ceil(K / 32)] (row r at natural + r * row_stride) to [ceil(K / 128)][rows][4].
CK_TILE_HOST std::vector<uint8_t>
mx_scale_to_kmajor(const uint8_t* natural, index_t rows, long_index_t K, long_index_t row_stride)
{
    const long_index_t groups = mx_tdm_scheduled::ceil_div(K, mx_tdm_scheduled::kBlockK);
    const long_index_t steps  = mx_tdm_scheduled::ceil_div(K, mx_tdm_scheduled::kAtomK);
    if(row_stride < groups)
        throw std::invalid_argument("mx_scale_to_kmajor: row_stride " + std::to_string(row_stride) +
                                    " < " + std::to_string(groups) + " scale groups");
    std::vector<uint8_t> out(mx_scale_kmajor_bytes(rows, K), 0);
    for(long_index_t s = 0; s < steps; ++s)
        for(index_t r = 0; r < rows; ++r)
            for(long_index_t g = 0; g < 4 && s * 4 + g < groups; ++g)
                out[(s * rows + r) * 4 + g] = natural[r * row_stride + s * 4 + g];
    return out;
}

/// True when Kernel declares `static constexpr bool KMajorScales = true`.
template <typename Kernel, typename = void>
struct uses_kmajor_mx_scales : std::false_type
{
};

template <typename Kernel>
struct uses_kmajor_mx_scales<Kernel, std::void_t<decltype(Kernel::KMajorScales)>>
    : std::bool_constant<Kernel::KMajorScales>
{
};

template <typename Kernel>
inline constexpr bool uses_kmajor_mx_scales_v = uses_kmajor_mx_scales<Kernel>::value;

/// mx_scale_to_kmajor of a [rows][K / 32] host tensor of one-byte scales, as a 1-D host tensor.
template <typename ScaleTensor>
CK_TILE_HOST ScaleTensor mx_scale_tensor_to_kmajor(const ScaleTensor& natural,
                                                   index_t rows,
                                                   long_index_t K)
{
    static_assert(sizeof(*natural.data()) == 1, "mx_scale_tensor_to_kmajor: one-byte scales");
    const auto bytes =
        mx_scale_to_kmajor(reinterpret_cast<const uint8_t*>(natural.data()),
                           rows,
                           K,
                           static_cast<long_index_t>(natural.mDesc.get_strides()[0]));
    ScaleTensor out({bytes.size()});
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

/// The inverse of mx_scale_to_kmajor: [rows][ceil(K / 32)], dense.
CK_TILE_HOST std::vector<uint8_t>
mx_scale_from_kmajor(const uint8_t* kmajor, index_t rows, long_index_t K)
{
    const long_index_t groups = mx_tdm_scheduled::ceil_div(K, mx_tdm_scheduled::kBlockK);
    std::vector<uint8_t> out(static_cast<std::size_t>(rows * groups));
    for(index_t r = 0; r < rows; ++r)
        for(long_index_t g = 0; g < groups; ++g)
            out[r * groups + g] = kmajor[((g / 4) * rows + r) * 4 + g % 4];
    return out;
}

namespace mx_tdm_scheduled {

CK_TILE_HOST float e8m0_to_float(uint8_t e)
{
    if(e == 0xFF)
        return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(e) - 127);
}

CK_TILE_HOST float fp4_e2m1_to_float(uint8_t nibble)
{
    static constexpr float kValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v                     = kValues[nibble & 7];
    return (nibble & 8) ? -v : v;
}

CK_TILE_HOST float fp8_e4m3_to_float(uint8_t b)
{
    const int exp  = (b >> 3) & 0xF;
    const int mant = b & 7;
    if(exp == 0xF && mant == 7)
        return std::numeric_limits<float>::quiet_NaN();
    const float v = exp == 0 ? std::ldexp(static_cast<float>(mant), -9)
                             : std::ldexp(1.0f + mant / 8.0f, exp - 7);
    return (b & 0x80) ? -v : v;
}

CK_TILE_HOST float bf8_e5m2_to_float(uint8_t b)
{
    const int exp  = (b >> 2) & 0x1F;
    const int mant = b & 3;
    float v        = 0.0f;
    if(exp == 0x1F)
        v = mant ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else
        v = exp == 0 ? std::ldexp(static_cast<float>(mant), -16)
                     : std::ldexp(1.0f + mant / 4.0f, exp - 15);
    return (b & 0x80) ? -v : v;
}

/// Element k of a K-contiguous A or B row.
CK_TILE_HOST float decode_element(MxTdmScheduledDtype t, const uint8_t* row, long_index_t k)
{
    switch(t)
    {
    case MxTdmScheduledDtype::Fp4: return fp4_e2m1_to_float((row[k / 2] >> (4 * (k % 2))) & 0xF);
    case MxTdmScheduledDtype::Fp8: return fp8_e4m3_to_float(row[k]);
    case MxTdmScheduledDtype::Bf8: return bf8_e5m2_to_float(row[k]);
    }
    throw std::invalid_argument("decode_element: unknown dtype");
}

/// Bytes of an A or B row of K elements.
CK_TILE_HOST long_index_t row_bytes(MxTdmScheduledDtype t, long_index_t K)
{
    return t == MxTdmScheduledDtype::Fp4 ? (K + 1) / 2 : K;
}

CK_TILE_HOST uint32_t float_bits(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}

CK_TILE_HOST float bits_float(uint32_t u)
{
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

/// The epilogue's FP8 saturation: +-inf to NaN, NaN kept, finite values clamped to +-448
/// (fma(v, 0, v) then llvm.maximum / llvm.minimum).
CK_TILE_HOST float saturate_fp8(float v)
{
    v = std::fma(v, 0.0f, v);
    if(std::isnan(v))
        return v;
    return v < -448.0f ? -448.0f : (v > 448.0f ? 448.0f : v);
}

/// f32 to FP8 E4M3 (OCP) bits, round to nearest even; |v| must not round past 448 (saturate
/// first). NaN becomes 0x7F with v's sign.
CK_TILE_HOST uint8_t float_to_fp8_e4m3_rne(float v)
{
    const uint8_t sign = std::signbit(v) ? 0x80 : 0;
    if(std::isnan(v))
        return sign | 0x7F;
    const double a = std::fabs(static_cast<double>(v));
    if(a < std::ldexp(1.0, -6))
        return sign | static_cast<uint8_t>(std::nearbyint(a / std::ldexp(1.0, -9)));
    int e    = std::ilogb(a);
    double q = std::nearbyint((a / std::ldexp(1.0, e) - 1.0) * 8.0);
    if(q == 8.0)
    {
        q = 0.0;
        ++e;
    }
    if(e > 8 || (e == 8 && q > 6.0))
        throw std::domain_error("float_to_fp8_e4m3_rne: " + std::to_string(v) + " past 448");
    return sign | static_cast<uint8_t>(((e + 7) << 3) | static_cast<int>(q));
}

/// f32 to f16 bits, round to nearest even (fptrunc): overflow to inf, NaN quieted.
CK_TILE_HOST uint16_t float_to_fp16_rne(float v)
{
    const uint16_t sign = std::signbit(v) ? 0x8000 : 0;
    if(std::isnan(v))
        return sign | 0x7E00 | static_cast<uint16_t>((float_bits(v) >> 13) & 0x1FF);
    const double a = std::fabs(static_cast<double>(v));
    if(std::isinf(v) || a >= 65520.0)
        return sign | 0x7C00;
    if(a < std::ldexp(1.0, -14))
        return sign | static_cast<uint16_t>(std::nearbyint(a / std::ldexp(1.0, -24)));
    int e    = std::ilogb(a);
    double q = std::nearbyint((a / std::ldexp(1.0, e) - 1.0) * 1024.0);
    if(q == 1024.0)
    {
        q = 0.0;
        ++e;
    }
    return sign | static_cast<uint16_t>(((e + 15) << 10) | static_cast<int>(q));
}

/// f32 to bf16 bits, round to nearest even (fptrunc): NaN quieted.
CK_TILE_HOST uint16_t float_to_bf16_rne(float v)
{
    const uint32_t u = float_bits(v);
    if(std::isnan(v))
        return static_cast<uint16_t>((u >> 16) | 0x40);
    return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

CK_TILE_HOST float fp16_bits_to_float(uint16_t h)
{
    const int exp  = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    float v        = 0.0f;
    if(exp == 0x1F)
        v = mant ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else
        v = exp == 0 ? std::ldexp(static_cast<float>(mant), -24)
                     : std::ldexp(1.0f + mant / 1024.0f, exp - 15);
    return (h & 0x8000) ? -v : v;
}

CK_TILE_HOST float bf16_bits_to_float(uint16_t h)
{
    return bits_float(static_cast<uint32_t>(h) << 16);
}

/// Bytes of one C or D element.
CK_TILE_HOST index_t out_bytes(MxTdmScheduledOutDtype t)
{
    return t == MxTdmScheduledOutDtype::Fp8E4M3 ? 1 : 2;
}

/// C or D element i of `data` as f32.
CK_TILE_HOST float load_out(MxTdmScheduledOutDtype t, const void* data, std::size_t i)
{
    if(t == MxTdmScheduledOutDtype::Fp8E4M3)
        return fp8_e4m3_to_float(static_cast<const uint8_t*>(data)[i]);
    const uint16_t h = static_cast<const uint16_t*>(data)[i];
    return t == MxTdmScheduledOutDtype::Fp16 ? fp16_bits_to_float(h) : bf16_bits_to_float(h);
}

/// The epilogue's conversion of a final f32 value (FP8 saturated first) into element i of `data`.
CK_TILE_HOST void store_out(MxTdmScheduledOutDtype t, void* data, std::size_t i, float v)
{
    if(t == MxTdmScheduledOutDtype::Fp8E4M3)
        static_cast<uint8_t*>(data)[i] = float_to_fp8_e4m3_rne(saturate_fp8(v));
    else if(t == MxTdmScheduledOutDtype::Fp16)
        static_cast<uint16_t*>(data)[i] = float_to_fp16_rne(v);
    else
        static_cast<uint16_t*>(data)[i] = float_to_bf16_rne(v);
}

} // namespace mx_tdm_scheduled

/**
 * @brief E = A @ B^T with K-major scales, as MxGemmPipelineAgBgCrCompTDMScheduled.
 *
 * The products are summed in double and rounded once to f32; with inputs whose f32 sums are
 * exact in any order (rocKE's check inputs) this is the kernel's accumulator bit for bit. Then
 * the output rounding (FP8 saturating).
 *
 * A and B are M (N) rows of row_bytes(dtype, K) bytes, lda (ldb) bytes apart; the scales are
 * K-major (mx_scale_to_kmajor); E is row-major M x N in the output type, lde elements apart.
 * With `rows` set only those num_rows rows of E are computed, stored compactly: e's row i is
 * row rows[i] of E.
 */
CK_TILE_HOST void reference_mx_gemm_tdm_scheduled(MxTdmScheduledDtype a_dtype,
                                                  MxTdmScheduledDtype b_dtype,
                                                  MxTdmScheduledOutDtype out_dtype,
                                                  const uint8_t* a,
                                                  long_index_t lda,
                                                  const uint8_t* b,
                                                  long_index_t ldb,
                                                  const uint8_t* a_scale,
                                                  const uint8_t* b_scale,
                                                  void* e,
                                                  long_index_t lde,
                                                  index_t M,
                                                  index_t N,
                                                  long_index_t K,
                                                  unsigned threads    = 0,
                                                  const index_t* rows = nullptr,
                                                  index_t num_rows    = 0)
{
    const index_t out_rows = rows ? num_rows : M;
    auto scale             = [](const uint8_t* s, index_t extent, index_t r, long_index_t k) {
        return mx_tdm_scheduled::e8m0_to_float(
            s[((k / mx_tdm_scheduled::kAtomK) * extent + r) * 4 +
              (k % mx_tdm_scheduled::kAtomK) / mx_tdm_scheduled::kBlockK]);
    };
    auto row = [&](index_t i) {
        const index_t m = rows ? rows[i] : i;
        for(index_t n = 0; n < N; ++n)
        {
            double sum = 0.0;
            for(long_index_t k = 0; k < K; ++k)
            {
                const double x =
                    static_cast<double>(mx_tdm_scheduled::decode_element(a_dtype, a + m * lda, k)) *
                    scale(a_scale, M, m, k);
                const double y =
                    static_cast<double>(mx_tdm_scheduled::decode_element(b_dtype, b + n * ldb, k)) *
                    scale(b_scale, N, n, k);
                sum += x * y;
            }
            mx_tdm_scheduled::store_out(
                out_dtype, e, static_cast<std::size_t>(i) * lde + n, static_cast<float>(sum));
        }
    };
    if(threads == 0)
        threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<unsigned>(threads, static_cast<unsigned>(std::max<index_t>(out_rows, 1)));
    std::vector<std::thread> pool;
    for(unsigned t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for(index_t i = static_cast<index_t>(t); i < out_rows;
                i += static_cast<index_t>(threads))
                row(i);
        });
    for(auto& th : pool)
        th.join();
}

} // namespace ck_tile
