// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaPhiloxOracle.hpp"

/**
 * @file TestFlydslSdpaBwdPhiloxOracle.cpp
 * @brief The host side of the SDPA backward's dropout: the launch values the engines pass, the
 *        mask the device harness expects of the kernels, and the oracle's counter arithmetic at
 *        plane products the kernels' 64-bit counters reach and a 32-bit one does not.
 *
 * Host only; the device cases that run the kernels against this mask are
 * TestFlydslSdpaBwdObjects.cpp, TestFlydslSdpaBwdObjectsPerObject.cpp and
 * TestFlydslSdpaBwdPhiloxDevice.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;

// ---------------------------------------------------------------------------
// Launch values
// ---------------------------------------------------------------------------

/// The threshold and scale the backward engines launch with, recorded from the engines' own
/// pair (`native::dropoutThreshold` and `native::dropoutScale`, the f32 forms of AOTriton's
/// `((p - 0.5) * 0xFFFFFFFF).to(int32)` and `1 / (1 - p)`) before the engines took
/// FlydslSdpaDropout.hpp's. The scale is a hex float: the exact f32.
TEST(TestFlydslSdpaBwdDropoutLaunchValues, AreThoseTheBackwardEnginesLaunchedWith)
{
    struct Row
    {
        float p;
        int32_t threshold;
        float scale;
    };
    const std::array<Row, 7> ROWS = {
        Row{0.0F, std::numeric_limits<int32_t>::min(), 0x1.0p+0F},
        Row{0.1F, -1717986944, 0x1.1c71c8p+0F},
        Row{0.2F, -1288490240, 0x1.4p+0F},
        Row{0.22F, -1202590848, 0x1.483484p+0F},
        Row{0.5F, 0, 0x1.0p+1F},
        Row{0.9F, 1717986816, 0x1.3ffffcp+3F},
        Row{std::nextafter(1.0F, 0.0F), 2147483392, 0x1.0p+24F},
    };
    for(const auto& row : ROWS)
    {
        EXPECT_EQ(flydsl_sdpa::dropoutThreshold(row.p), row.threshold) << "p " << row.p;
        EXPECT_EQ(flydsl_sdpa::dropoutScale(row.p), row.scale) << "p " << row.p;
    }
}

// ---------------------------------------------------------------------------
// The harness's mask
// ---------------------------------------------------------------------------

/// The harness's keep decision for a graph with declared extents Sq = Skv = 100 and valid lengths
/// below them (padded) is the oracle's under those declared extents, whatever the layout: the
/// kernels count counters over the graph's extents, not over a batch's valid lengths.
TEST(TestFlydslSdpaBwdHarnessMask, UsesTheDeclaredExtentsOfAPaddedOrPackedGraph)
{
    constexpr int64_t SEQ_Q = 100;
    constexpr int64_t SEQ_KV = 100;
    constexpr float P = 0.25F;
    for(const auto layout : {BwdLayout::HEAD_MAJOR, BwdLayout::PACKED_QKV})
    {
        BwdCase c;
        c.batch = 2;
        c.heads = 4;
        c.kvHeads = 4;
        c.seqQ = SEQ_Q;
        c.seqKv = SEQ_KV;
        c.dropout = P;
        c.seed = 0x0123456789ABCDEFLL;
        c.offset = 77;
        c.layout = layout;
        c.lengthsQ = {70, 37};
        c.lengthsKv = {61, 100};
        const auto threshold = flydsl_sdpa::dropoutThreshold(P);
        int64_t kept = 0;
        int64_t total = 0;
        for(int64_t b = 0; b < c.batch; ++b)
        {
            for(int64_t h = 0; h < c.heads; ++h)
            {
                for(int64_t row = 0; row < SEQ_Q; ++row)
                {
                    for(int64_t column = 0; column < SEQ_KV; ++column)
                    {
                        const bool expected = testing::philox_oracle::keeps(
                            0x0123456789ABCDEFULL,
                            77U,
                            testing::philox_oracle::MaskCoordinates{
                                b, h, row, column, 4, SEQ_Q, SEQ_KV},
                            threshold);
                        ASSERT_EQ(keeps(c, b, h, row, column), expected)
                            << "b " << b << " h " << h << " row " << row << " column " << column;
                        kept += expected ? 1 : 0;
                        ++total;
                    }
                }
            }
        }
        // The mask is not degenerate: about 1 - P of the elements are kept.
        EXPECT_NEAR(static_cast<double>(kept) / static_cast<double>(total), 1.0 - static_cast<double>(P), 0.02);
    }
}

TEST(TestFlydslSdpaBwdHarnessMask, KeepsEverythingWithoutDropoutAndScalesByTheEnginesValue)
{
    BwdCase c;
    EXPECT_TRUE(keeps(c, 1, 2, 3, 4));
    EXPECT_EQ(dropoutScale(c), 1.0);
    c.dropout = 0.2F;
    EXPECT_EQ(dropoutScale(c), static_cast<double>(1.25F));
}

// ---------------------------------------------------------------------------
// Counters past 2^31 and 2^32
// ---------------------------------------------------------------------------

/// The counter as a 32-bit signed index computes it (AOTriton's bound is
/// `(B * Hq - 1) * Sq * cdiv(Skv, 4) < 2^31`), widened.
uint64_t counterOf32(uint64_t offset, const testing::philox_oracle::MaskCoordinates& at)
{
    const auto rowStride = static_cast<uint32_t>((at.seqLenKv + 3) / 4);
    const auto plane = static_cast<uint32_t>(at.batch * at.numHeads + at.head);
    const uint32_t counter = static_cast<uint32_t>(offset)
                             + (plane * static_cast<uint32_t>(at.seqLenQ)
                                + static_cast<uint32_t>(at.row))
                                   * rowStride
                             + static_cast<uint32_t>(at.key / 4);
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(counter)));
}

/// The oracle's counter is the 64-bit formula where the plane product is above 2^31 and above
/// 2^32, and a 32-bit computation is not: the kernels draw from the 64-bit one.
TEST(TestFlydslSdpaBwdPhiloxCounters, AreSixtyFourBitAtPlaneProductsPastTwoToThe31And32)
{
    using testing::philox_oracle::counterOf;
    using testing::philox_oracle::MaskCoordinates;
    using testing::philox_oracle::philox4x32;
    using testing::philox_oracle::randomOf;
    struct Row
    {
        MaskCoordinates at;
        uint64_t offset;
        uint64_t counter;
    };
    // The last element of plane 9999 (b 1249, h 7 of 8 heads) at Sq = Skv = 1024: counter 77 +
    // (9999 * 1024 + 1023) * 256 + 255 = 2621440076, between 2^31 and 2^32 (a 32-bit index is
    // negative); and of plane 191999 (b 2999, h 63 of 64) at Sq = Skv = 4096: 77 +
    // (191999 * 4096 + 4095) * 1024 + 1023 = 805306368076, above 2^32.
    const std::array<Row, 2> ROWS = {
        Row{MaskCoordinates{1249, 7, 1023, 1023, 8, 1024, 1024}, 77U, 0x9C40004CULL},
        Row{MaskCoordinates{2999, 63, 4095, 4095, 64, 4096, 4096}, 77U, 0xBB8000004CULL}};
    constexpr uint64_t SEED = 0x1234567U;
    for(const auto& row : ROWS)
    {
        const auto expected = row.counter;
        EXPECT_GT(expected - row.offset, uint64_t{1} << 31);
        EXPECT_EQ(counterOf(row.offset, row.at), expected);
        EXPECT_NE(counterOf32(row.offset, row.at), expected);
        // The 64-bit counter's high word reaches c1 and the draw comes from it.
        const auto words = philox4x32({static_cast<uint32_t>(expected),
                                       static_cast<uint32_t>(expected >> 32U),
                                       0U,
                                       0U},
                                      static_cast<uint32_t>(SEED),
                                      static_cast<uint32_t>(SEED >> 32U));
        EXPECT_EQ(randomOf(SEED, row.offset, row.at),
                  static_cast<int32_t>(words[static_cast<size_t>(row.at.key % 4)]));
    }
    EXPECT_GT(ROWS[1].counter, uint64_t{1} << 32);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
