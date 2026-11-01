// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaPhiloxOracle.hpp"

/**
 * @file TestFlydslSdpaDropout.cpp
 * @brief Dropout as the FlyDSL SDPA engines draw it: the test-local Philox oracle pinned by
 *        known-answer vectors, and the host's keep threshold and scale.
 *
 * Every mask constant here was produced by a real AOTriton build on a GPU, so the oracle is
 * checked against the implementation it claims to reproduce rather than against itself:
 * - AOTriton b5e8cfb (0.15), `modules/flash/kernel/dropout.py` `fast_dropout_mask`, JIT-compiled
 *   by Triton 3.6 and run on gfx1151: one (batch, head) plane of Sq 16 x Skv 32, seed
 *   0x0123456789ABCDEF, offset 77, p 0.25, row stride cdiv(32, 4) = 8;
 * - PyTorch 2.10 (AOTriton 0.11.1) `aten::_scaled_dot_product_flash_attention(q, k, v, 0.25,
 *   false, return_debug_mask=true)` on gfx1151 after `torch.manual_seed(1234)`, q/k/v
 *   `randn` f16 [2, 3, 40|70|70, 64]: it returned seed 1234 and offset 12, and its debug mask.
 * The generation script is kept with the work's notes (scratch/parity/varlen-dropout/d1/gen_kat.py).
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing::philox_oracle;
using flydsl_sdpa::dropoutScale;
using flydsl_sdpa::dropoutThreshold;

// ---------------------------------------------------------------------------
// Philox4x32-10 itself: the Random123 known-answer vectors (kat_vectors, philox4x32 10 rounds)
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaPhiloxOracle, MatchesTheRandom123KnownAnswers)
{
    EXPECT_EQ(philox4x32({0U, 0U, 0U, 0U}, 0U, 0U),
              (Words{0x6627e8d5U, 0xe169c58dU, 0xbc57ac4cU, 0x9b00dbd8U}));
    EXPECT_EQ(
        philox4x32({0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU}, 0xffffffffU, 0xffffffffU),
        (Words{0x408f276dU, 0x41c83b0eU, 0xa20bc7c6U, 0x6d5451fdU}));
    EXPECT_EQ(
        philox4x32({0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U}, 0xa4093822U, 0x299f31d0U),
        (Words{0xd16cfe09U, 0x94fdccebU, 0x5001e420U, 0x24126ea1U}));
}

/// Bit j of a row is key j's keep bit.
template <size_t N>
uint64_t keepBits(uint64_t seed,
                  uint64_t offset,
                  MaskCoordinates at,
                  int64_t firstKey,
                  int32_t threshold,
                  const std::array<int, 4>& wordForColumn)
{
    uint64_t bits = 0;
    for(size_t j = 0; j < N; ++j)
    {
        at.key = firstKey + static_cast<int64_t>(j);
        if(keeps(seed, offset, at, threshold, wordForColumn))
        {
            bits |= uint64_t{1} << j;
        }
    }
    return bits;
}

TEST(TestFlydslSdpaPhiloxOracle, ReproducesAotriton015sMaskBitForBit)
{
    constexpr std::array<uint32_t, 16> ROWS = {0x3D97FBDDU,
                                               0xF1B477FFU,
                                               0x7FE7E541U,
                                               0xFFACFE63U,
                                               0xA3FFEFEAU,
                                               0x9BB76B7EU,
                                               0xEFF6DFC6U,
                                               0x9573B7FAU,
                                               0x7FB6FFEAU,
                                               0x5F57FFFEU,
                                               0xE7B7FF61U,
                                               0xFE5587FFU,
                                               0xFF2BB5D6U,
                                               0x5DB6D7BFU,
                                               0x7ECF6EF9U,
                                               0xBEE9FFDFU};
    const auto threshold = dropoutThreshold(0.25F);
    for(size_t row = 0; row < ROWS.size(); ++row)
    {
        MaskCoordinates at;
        at.row = static_cast<int64_t>(row);
        at.seqLenQ = 16;
        at.seqLenKv = 32;
        EXPECT_EQ(keepBits<32>(0x0123456789ABCDEFULL, 77U, at, 0, threshold, WORD_FOR_COLUMN),
                  ROWS[row])
            << "row " << row;
    }
}

TEST(TestFlydslSdpaPhiloxOracle, ReproducesPyTorch210sMaskWithAotriton011sWordOrder)
{
    struct Row
    {
        int64_t batch;
        int64_t head;
        int64_t row;
        uint64_t low; // keys 0..63
        uint64_t high; // keys 64..69
    };
    constexpr std::array<Row, 3> ROWS = {Row{0, 0, 0, 0xFFBFB6FFEF3F7EDFULL, 0x3EU},
                                         Row{1, 2, 39, 0xFEFDB4FEE65FB5DFULL, 0x15U},
                                         Row{1, 1, 17, 0xFB2FDFF6F859BF7EULL, 0x1AU}};
    const auto threshold = dropoutThreshold(0.25F);
    for(const auto& expected : ROWS)
    {
        const MaskCoordinates at{expected.batch, expected.head, expected.row, 0, 3, 40, 70};
        EXPECT_EQ(keepBits<64>(1234U, 12U, at, 0, threshold, WORD_FOR_COLUMN_AOTRITON_0_11),
                  expected.low);
        EXPECT_EQ(keepBits<6>(1234U, 12U, at, 64, threshold, WORD_FOR_COLUMN_AOTRITON_0_11),
                  expected.high);
        // The two orders differ, so a mask cannot match both: which AOTriton it reproduces is
        // a fact about the word order, not an accident of the constants.
        EXPECT_NE(keepBits<64>(1234U, 12U, at, 0, threshold, WORD_FOR_COLUMN), expected.low);
    }
}

TEST(TestFlydslSdpaPhiloxOracle, StepsOneCounterPerFourKeysAndOneRowStridePerRow)
{
    MaskCoordinates at{1, 2, 3, 9, 4, 5, 13}; // cdiv(13, 4) = 4 counters a row
    EXPECT_EQ(counterOf(100U, at), 100U + ((1U * 4U + 2U) * 5U + 3U) * 4U + 2U);
    // Counters past 2^32 carry into c1.
    at = MaskCoordinates{0, 0, 0, 0, 1, 1, 4};
    const uint64_t high = (uint64_t{1} << 32) + 5U;
    EXPECT_EQ(randomsAt(7U, counterOf(high, at)), philox4x32({5U, 1U, 0U, 0U}, 7U, 0U));
}

// ---------------------------------------------------------------------------
// The host's threshold and scale
// ---------------------------------------------------------------------------

/// AOTriton's f32 `(p - 0.5) * 0xFFFFFFFF` truncated, as its kernel and PyTorch-ROCm apply it,
/// with philox.py's double-precision host form beside it where the two differ.
TEST(TestFlydslSdpaDropoutThreshold, IsAotritonsF32Expression)
{
    struct Case
    {
        float p;
        int32_t f32;
    };
    constexpr std::array<Case, 11> CASES = {Case{0.0F, std::numeric_limits<int32_t>::min()},
                                            Case{0.05F, -1932735232},
                                            Case{0.1F, -1717986944},
                                            Case{0.25F, -1073741824},
                                            Case{0.5F, 0},
                                            Case{0.75F, 1073741824},
                                            Case{0.9F, 1717986816},
                                            Case{0.99F, 2104534016},
                                            // p values at which the f32 and double forms
                                            // disagree on a mask element; PyTorch kept
                                            // exactly what the f32 form keeps at each.
                                            Case{0.099648498F, -1719496576},
                                            Case{0.055121858F, -1910737024},
                                            Case{0.122958548F, -1619380736}};
    for(const auto& c : CASES)
    {
        EXPECT_EQ(dropoutThreshold(c.p), c.f32) << "p " << c.p;
    }
}

TEST(TestFlydslSdpaDropoutThreshold, DiffersFromTheDoubleFormWherePyTorchSaidItShould)
{
    // r is the random PyTorch's mask drew for one element (seed 1234, offset 12, Sq = Skv = 16,
    // element (0, 5)); the double form keeps it, the f32 form and PyTorch drop it.
    constexpr int32_t RANDOM = -1719496578;
    constexpr float P = 0.099648498F;
    const auto doubleForm = static_cast<int32_t>((static_cast<double>(P) - 0.5) * 4294967295.0);
    EXPECT_GT(RANDOM, doubleForm);
    EXPECT_FALSE(RANDOM > dropoutThreshold(P));
}

TEST(TestFlydslSdpaDropoutThreshold, KeepsEverythingAtZeroAndNothingNearOne)
{
    EXPECT_EQ(dropoutThreshold(0.0F), std::numeric_limits<int32_t>::min());
    // The largest f32 below 1: p - 0.5 is exact (0.5 - 2^-24), so 2^31 - 2^8, which keeps
    // only the 255 randoms above it.
    EXPECT_EQ(dropoutThreshold(std::nextafter(1.0F, 0.0F)), 2147483392);
    // p = 1 reaches 2^31, past int32: clamped, keeping nothing.
    EXPECT_EQ(dropoutThreshold(1.0F), std::numeric_limits<int32_t>::max());
}

TEST(TestFlydslSdpaDropoutThreshold, KeepsAboutOneMinusPOfTheStream)
{
    // 4096 x 64 = 262144 draws: a binomial standard deviation under 0.001 at any p.
    for(const float p : {0.1F, 0.25F, 0.5F, 0.9F})
    {
        const auto threshold = dropoutThreshold(p);
        int64_t kept = 0;
        int64_t total = 0;
        for(int64_t row = 0; row < 4096; ++row)
        {
            for(int64_t key = 0; key < 64; ++key)
            {
                kept += keeps(42U, 0U, MaskCoordinates{0, 0, row, key, 1, 4096, 64}, threshold) ? 1
                                                                                                : 0;
                ++total;
            }
        }
        const double fraction = static_cast<double>(kept) / static_cast<double>(total);
        EXPECT_NEAR(fraction, 1.0 - static_cast<double>(p), 0.005) << "p " << p;
    }
}

TEST(TestFlydslSdpaDropoutScale, IsOneOverOneMinusPInF32)
{
    EXPECT_EQ(dropoutScale(0.0F), 1.0F);
    EXPECT_EQ(dropoutScale(0.5F), 2.0F);
    EXPECT_EQ(dropoutScale(0.25F), 1.0F / 0.75F);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
