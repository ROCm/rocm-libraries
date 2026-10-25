// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_datatype2string.hpp"
#include "mxDataGen.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
    TEST(ScaleBufferSize, smoke_NaturalGfx950ScalesUsePaddedRows)
    {
        // K=128 with 32-element blocks has four packed scales per row, while
        // the gfx950 host supplies a stride of eight. Include both padding boundaries.
        for(size_t freeDim : {0, 1, 31, 32, 33, 64})
            for(size_t kBlocks : {0, 1, 4, 7, 8, 9})
            {
                SCOPED_TRACE(::testing::Message()
                             << "freeDim=" << freeDim << ", kBlocks=" << kBlocks);
                size_t const         paddedRows = (freeDim + 31) / 32 * 32;
                size_t const         stride     = (kBlocks + 7) / 8 * 8;
                size_t const         size       = paddedRows * stride;
                size_t const         batches    = 3;
                std::vector<uint8_t> scales(size * batches + 16, 0xa5);
                auto* const          buffer = scales.data() + 8;
                for(size_t b = 0; b < batches; ++b)
                    for(size_t i = 0; i < freeDim * kBlocks; ++i)
                        buffer[b * size + i] = static_cast<uint8_t>((b * 47 + i) % 127 + 1);
                auto const packed = scales;

                for(size_t b = 0; b < batches; ++b)
                    padNaturalMXScaleBufferForGfx950(buffer + b * size, freeDim, kBlocks, 1);

                for(size_t b = 0; b < batches; ++b)
                    for(size_t row = 0; row < paddedRows; ++row)
                        for(size_t col = 0; col < stride; ++col)
                            EXPECT_EQ(buffer[b * size + row * stride + col],
                                      row < freeDim && col < kBlocks
                                          ? packed[8 + b * size + row * kBlocks + col]
                                          : 0);
                for(size_t i = 0; i < 8; ++i)
                {
                    EXPECT_EQ(scales[i], 0xa5);
                    EXPECT_EQ(scales[8 + size * batches + i], 0xa5);
                }
            }
    }
#endif

    TEST(ScaleBufferSize, smoke_PaddedLayoutsRequireKAlongRows)
    {
        // An NT operand has M or N rows and K columns. The padded helper expects
        // the TN orientation instead; debug builds must reject this misuse.
        for(bool rocroller : {false, true})
        {
            SCOPED_TRACE(rocroller);
            EXPECT_EQ(
                blockScaleBufferSize(
                    256, 1, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT, true, rocroller),
                256);
            EXPECT_DEBUG_DEATH(
                blockScaleBufferSize(
                    1, 256, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT, false, rocroller),
                "require K along the rows");
        }
        for(auto format : {hipblaslt_scaling_format::Block_32_UE8M0,
                           hipblaslt_scaling_format::Block_32_UE4M3,
                           hipblaslt_scaling_format::Block_32_UE5M3,
                           hipblaslt_scaling_format::Block_16_UE8M0,
                           hipblaslt_scaling_format::Block_16_UE4M3,
                           hipblaslt_scaling_format::Block_16_UE5M3})
        {
            SCOPED_TRACE(static_cast<int>(format));
            EXPECT_EQ(blockScaleBufferSize(256, 1, format, true, true),
                      blockSize(format) == 32 ? 256 : 512);
            // The alternate layout still accepts K along columns.
            EXPECT_EQ(blockScaleBufferSize(1, 256, format, false, false),
                      blockSize(format) == 32 ? 32 : 128);
            EXPECT_DEBUG_DEATH(blockScaleBufferSize(1, 256, format, false, true),
                               "require K along the rows");
        }
    }

    TEST(ScaleBufferSize, smoke_PreswizzledPaddingDoesNotDependOnRocroller)
    {
        struct Case
        {
            int64_t k;
            int64_t mn;
            size_t  expected;
        };
        // The raw TN operand has K rows and M or N columns. Check both sides
        // of the 32-column and eight-scale-row boundaries, including empty work.
        Case const cases[] = {{0, 1, 0},
                              {256, 0, 0},
                              {32, 1, 256},
                              {224, 1, 256},
                              {255, 1, 256},
                              {256, 1, 256},
                              {256, 31, 256},
                              {256, 32, 256},
                              {256, 33, 512},
                              {257, 1, 512},
                              {288, 1, 512},
                              {288, 33, 1024}};

        for(auto const& c : cases)
        {
            SCOPED_TRACE(::testing::Message() << "K=" << c.k << ", M/N=" << c.mn);
            for(bool rocroller : {false, true})
            {
                SCOPED_TRACE(rocroller);
                EXPECT_EQ(blockScaleBufferSize(c.k,
                                               c.mn,
                                               hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT,
                                               true,
                                               rocroller),
                          c.expected);
            }
        }
    }

    TEST(ScaleBufferSize, smoke_OtherFormatsKeepTheirExistingPadding)
    {
        for(auto format : {hipblaslt_scaling_format::Block_32_UE8M0,
                           hipblaslt_scaling_format::Block_32_UE4M3,
                           hipblaslt_scaling_format::Block_32_UE5M3})
        {
            SCOPED_TRACE(static_cast<int>(format));
            EXPECT_EQ(blockScaleBufferSize(128, 3, format, true, false), 12);
            EXPECT_EQ(blockScaleBufferSize(3, 128, format, false, false), 16);
            EXPECT_EQ(blockScaleBufferSize(128, 3, format, true, true), 256);
        }
        for(auto format : {hipblaslt_scaling_format::Block_16_UE8M0,
                           hipblaslt_scaling_format::Block_16_UE4M3,
                           hipblaslt_scaling_format::Block_16_UE5M3})
        {
            SCOPED_TRACE(static_cast<int>(format));
            EXPECT_EQ(blockScaleBufferSize(128, 3, format, true, false), 24);
            EXPECT_EQ(blockScaleBufferSize(3, 128, format, false, false), 64);
            EXPECT_EQ(blockScaleBufferSize(128, 3, format, true, true), 256);
        }
    }
}
