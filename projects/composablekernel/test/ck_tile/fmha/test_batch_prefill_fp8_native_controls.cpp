// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The gfx1250 operations that form and consume the FP8 page tile's quantized P, on e4m3
// subnormals (accuracy_contract.md, "FP8 P": QuantizedP::keeps_subnormals):
//  - the P cast, fp32x8_to_fp8x8<fp8_rounding_mode::standard>(v, 1 / 256), which stores 256 * P
//    in e4m3 (the FP8 page tile's MakePForGemm1);
//  - the dequant of the quantized row sum, __builtin_amdgcn_cvt_scale_pk8_f32_fp8 with the literal
//    unit scale (the forward pipeline's Dequantize on gfx1250), and CK's fp8x8_to_fp32x8(q, 1),
//    which the other gfx125 targets take;
//  - gemm_1's FP8 WMMA (16x16x128, Default access order) on subnormal P operands.
// Compiled with the page-tile gtests' flags (-fgpu-flush-denormals-to-zero). The P cast must round
// to nearest even. A subnormal P that the cast keeps must also reach O (the WMMA) and the row sum
// (the dequant); otherwise the numerator and the denominator of O see different P and the
// contract's model does not apply. The FP8 gtests model kept subnormals (QuantizedPModel in
// test_batch_prefill_page_tile.cpp, the contract's keeps_subnormals), so the test requires the
// cast to keep them.

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ck_tile/host.hpp"
#include "fmha_test_common.hpp"

namespace {

using namespace ck_tile;

[[maybe_unused]] constexpr float kPScale = 256.0f; // the FP8 pipelines' P scale

// 256 * P of the cast inputs, and the e4m3 bits with subnormals kept and flushed (0xff: not
// checked in that mode).
struct CastCase
{
    float scaled;
    std::uint8_t kept;
    std::uint8_t flushed;
};
constexpr std::array<CastCase, 16> kCastCases{{
    {1.0625f, 0x38, 0x38},         // tie -> even mantissa (1.0)
    {1.1875f, 0x3a, 0x3a},         // tie -> even mantissa (1.25)
    {0x1p-7f, 0x04, 0x00},         // subnormal 4 * 2^-9
    {0x1p-9f, 0x01, 0x00},         // smallest subnormal
    {1.5f * 0x1p-9f, 0x02, 0x00},  // subnormal tie -> even (2 * 2^-9)
    {0x1p-6f, 0x08, 0x08},         // smallest normal
    {0.75f * 0x1p-6f, 0x06, 0x00}, // subnormal 6 * 2^-9
    {-0x1p-8f, 0x82, 0xff},        // negative subnormal (a flush may keep the sign)
    {0.75f * 0x1p-9f, 0x01, 0x00}, // rounds up to the smallest subnormal
    {2.5f * 0x1p-9f, 0x02, 0x00},  // subnormal tie -> even (2 * 2^-9)
    {7.5f * 0x1p-9f, 0x08, 0xff},  // tie between 7 * 2^-9 and 2^-6 -> 2^-6 (a flush may differ)
    {448.0f, 0x7e, 0x7e},          // largest finite
    {240.0f, 0x77, 0x77},          // 1.875 * 2^7
    {0.0f, 0x00, 0x00},            //
    {256.0f, 0x78, 0x78},          // P = 1
    {3.0f, 0x44, 0x44},            // 1.5 * 2^1
}};

// e4m3 bytes of the dequant inputs and their values with subnormals kept and flushed.
struct DequantCase
{
    std::uint8_t bits;
    float kept;
    float flushed;
};
constexpr std::array<DequantCase, 16> kDequantCases{{
    {0x01, 0x1p-9f, 0.0f},
    {0x02, 0x1p-8f, 0.0f},
    {0x03, 3 * 0x1p-9f, 0.0f},
    {0x04, 0x1p-7f, 0.0f},
    {0x06, 6 * 0x1p-9f, 0.0f},
    {0x07, 7 * 0x1p-9f, 0.0f},
    {0x08, 0x1p-6f, 0x1p-6f},
    {0x38, 1.0f, 1.0f},
    {0x81, -0x1p-9f, -0.0f},
    {0x87, -7 * 0x1p-9f, -0.0f},
    {0x00, 0.0f, 0.0f},
    {0x7e, 448.0f, 448.0f},
    {0x78, 256.0f, 256.0f},
    {0x40, 2.0f, 2.0f},
    {0x10, 0x1p-5f, 0x1p-5f},
    {0x09, 1.125f * 0x1p-6f, 1.125f * 0x1p-6f},
}};

struct Results
{
    std::uint8_t cast[16];
    float dequant_builtin[16];
    float dequant_ck[16];
    float wmma_subnormal[32]; // C[0] of each lane: 128 products of 2^-9 and 1
    float wmma_normal[32];    // 128 products of 2^-6 and 1
};

// gemm_1's warp GEMM of the FP8 page tile (BlockFmhaPipelineQRKSVSTdmDefaultPolicy::GetPVBlockGemm
// with the 16x16x128 FP8 Gemm1WarpTile).
using PvWarpGemm = WarpGemmDispatcher<fp8_t,
                                      fp8_t,
                                      float,
                                      16,
                                      16,
                                      128,
                                      true,
                                      false,
                                      false,
                                      WGAttrNumAccessEnum::Default>;

struct NativeProbe
{
    static constexpr index_t kBlockSize = 32;

    [[maybe_unused]] CK_TILE_DEVICE static float WmmaOf(std::uint8_t p_bits)
    {
        typename PvWarpGemm::AWarpTensor a;
        typename PvWarpGemm::BWarpTensor b;
        typename PvWarpGemm::CWarpTensor c;
        static_for<0, decltype(a)::get_thread_buffer_size(), 1>{}(
            [&](auto i) { a.get_thread_buffer()(i) = bit_cast<fp8_t>(p_bits); });
        static_for<0, decltype(b)::get_thread_buffer_size(), 1>{}(
            [&](auto i) { b.get_thread_buffer()(i) = bit_cast<fp8_t>(std::uint8_t{0x38}); });
        static_for<0, decltype(c)::get_thread_buffer_size(), 1>{}(
            [&](auto i) { c.get_thread_buffer()(i) = 0.0f; });
        PvWarpGemm{}(c, a, b);
        return c.get_thread_buffer()[number<0>{}];
    }

    [[maybe_unused]] CK_TILE_DEVICE void operator()(Results* results) const
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        const index_t lane = threadIdx.x;
        if(lane == 0)
        {
            static_for<0, 2, 1>{}([&](auto half) {
                fp32x8_t p;
                constexpr index_t kFirst = decltype(half)::value * 8;
                static_for<0, 8, 1>{}([&](auto e) {
                    constexpr float kP = kCastCases[kFirst + decltype(e)::value].scaled / kPScale;
                    p[e()]             = kP;
                });
                const fp8x8_t q = fp32x8_to_fp8x8<fp8_rounding_mode::standard>(p, 1.0f / kPScale);
                static_for<0, 8, 1>{}(
                    [&](auto e) { results->cast[kFirst + e()] = bit_cast<std::uint8_t>(q[e()]); });
            });
            static_for<0, 2, 1>{}([&](auto half) {
                constexpr index_t kFirst = decltype(half)::value * 8;
                fp8x8_t q;
                static_for<0, 8, 1>{}([&](auto e) {
                    constexpr std::uint8_t kBits = kDequantCases[kFirst + decltype(e)::value].bits;
                    q[e()]                       = bit_cast<fp8_t>(kBits);
                });
#if defined(__gfx1250__) && !defined(__gfx1250_strict__) && !defined(__gfx1251__)
                const fp32x8_t builtin =
                    __builtin_amdgcn_cvt_scale_pk8_f32_fp8(bit_cast<uint32x2_t>(q), 0x7f7f7f7fu, 0);
#else
                const fp32x8_t builtin = fp8x8_to_fp32x8(q, 1.0f);
#endif
                const fp32x8_t ck = fp8x8_to_fp32x8(q, 1.0f);
                static_for<0, 8, 1>{}([&](auto e) {
                    results->dequant_builtin[kFirst + e()] = builtin[e()];
                    results->dequant_ck[kFirst + e()]      = ck[e()];
                });
            });
        }
        // fp8x8_to_fp32x8 above runs a lane-dependent scale branch; the WMMAs run on the full wave.
        results->wmma_subnormal[lane] = WmmaOf(0x01);
        results->wmma_normal[lane]    = WmmaOf(0x08);
#else
        ignore = results;
#endif
    }
};

enum class Mode
{
    Kept,
    Flushed,
    Neither,
};

const char* Name(Mode mode)
{
    return mode == Mode::Kept ? "keeps subnormals"
                              : (mode == Mode::Flushed ? "flushes subnormals" : "neither");
}

bool SameBits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

} // namespace

using Fp8NativeControls = ck_tile::test::Gfx125FmhaTest;

TEST_F(Fp8NativeControls, QuantizedPSubnormalsAndRounding)
{
    DeviceMem device(sizeof(Results));
    launch_kernel(stream_config{nullptr, false, 0},
                  make_kernel<1, gfx125_t>(NativeProbe{},
                                           dim3(1),
                                           dim3(NativeProbe::kBlockSize),
                                           0,
                                           static_cast<Results*>(device.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());
    Results r{};
    device.FromDevice(&r);

    // P cast: the mode the subnormal cases follow; the normal cases and ties hold in both.
    bool cast_kept = true, cast_flushed = true, cast_normal = true;
    std::cout << "P cast (256 * P -> e4m3 bits):";
    for(std::size_t i = 0; i < kCastCases.size(); ++i)
    {
        const auto& c = kCastCases[i];
        std::cout << ' ' << std::hex << std::setw(2) << std::setfill('0') << int{r.cast[i]};
        if(c.kept == c.flushed)
            cast_normal &= r.cast[i] == c.kept;
        else
        {
            cast_kept &= r.cast[i] == c.kept;
            cast_flushed &= c.flushed == 0xff || r.cast[i] == c.flushed;
        }
    }
    std::cout << std::dec << '\n';
    const Mode cast = cast_kept ? Mode::Kept : (cast_flushed ? Mode::Flushed : Mode::Neither);

    auto dequant_mode = [](const float(&values)[16], const char* name) {
        bool kept = true, flushed = true, normal = true;
        std::cout << name << ':';
        for(std::size_t i = 0; i < kDequantCases.size(); ++i)
        {
            const auto& c = kDequantCases[i];
            std::cout << ' ' << values[i];
            if(SameBits(c.kept, c.flushed))
                normal &= SameBits(values[i], c.kept);
            else
            {
                kept &= SameBits(values[i], c.kept);
                flushed &= values[i] == c.flushed; // either sign of zero
            }
        }
        std::cout << '\n';
        EXPECT_TRUE(normal) << name << " converts normal e4m3 values wrongly";
        return kept ? Mode::Kept : (flushed ? Mode::Flushed : Mode::Neither);
    };
    const Mode dequant    = dequant_mode(r.dequant_builtin, "row-sum dequant (builtin)");
    const Mode dequant_ck = dequant_mode(r.dequant_ck, "fp8x8_to_fp32x8");

    bool wmma_kept = true, wmma_flushed = true, wmma_normal = true;
    for(int lane = 0; lane < 32; ++lane)
    {
        wmma_kept &= r.wmma_subnormal[lane] == 0.25f;
        wmma_flushed &= r.wmma_subnormal[lane] == 0.0f;
        wmma_normal &= r.wmma_normal[lane] == 2.0f;
    }
    const Mode wmma = wmma_kept ? Mode::Kept : (wmma_flushed ? Mode::Flushed : Mode::Neither);
    std::cout << "FP8 WMMA 16x16x128, 128 products: 2^-9 * 1 -> " << r.wmma_subnormal[0]
              << ", 2^-6 * 1 -> " << r.wmma_normal[0] << '\n';

    std::cout << "P cast " << Name(cast) << ", row-sum dequant " << Name(dequant)
              << ", fp8x8_to_fp32x8 " << Name(dequant_ck) << ", WMMA " << Name(wmma)
              << "; accuracy contract keeps_subnormals = "
              << (cast == Mode::Kept ? "true" : "false") << '\n';

    EXPECT_TRUE(cast_normal) << "the P cast does not round to nearest even on normal values";
    EXPECT_TRUE(wmma_normal) << "the FP8 WMMA does not sum 128 normal products exactly";
    EXPECT_EQ(cast, Mode::Kept)
        << "the FP8 gtests' QuantizedPModel (test_batch_prefill_page_tile.cpp) models kept "
           "subnormals";
    EXPECT_NE(dequant, Mode::Neither);
    EXPECT_NE(wmma, Mode::Neither);
    EXPECT_EQ(dequant_ck, dequant)
        << "the gfx1250 dequant and CK's convert treat subnormals differently";
    // A subnormal P the cast keeps must reach O and the row sum alike.
    if(cast == Mode::Kept)
    {
        EXPECT_EQ(wmma, Mode::Kept);
        EXPECT_EQ(dequant, Mode::Kept);
    }
}
