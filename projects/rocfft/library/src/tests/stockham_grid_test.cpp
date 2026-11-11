// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rtc_stockham_gen.h"
#include "tree_node.h"

#include <gtest/gtest.h>
#include <stdexcept>

// A launch of more than 2^32 - 1 work items in grid x fails, so stockham_grid_y
// splits it over grid y.  The split must divide the block count exactly, since
// Stockham kernels do not skip spare blocks, and launches that already fit must
// stay unsplit so their kernels are generated exactly as before.  The real
// launches are too large for most GPUs, so check the decision directly.

static constexpr size_t max_items = std::numeric_limits<uint32_t>::max();

static hipDeviceProp_t prop_with_max_y(int max_y)
{
    hipDeviceProp_t prop{};
    prop.maxGridSize[1] = max_y;
    return prop;
}

static void expect_valid_split(size_t blocks, unsigned int wgs)
{
    const auto   prop = prop_with_max_y(65536);
    const size_t y    = stockham_grid_y(blocks, wgs, prop);
    EXPECT_GT(y, 1u);
    EXPECT_LE(y, 65536u);
    EXPECT_EQ(blocks % y, 0u) << blocks << " blocks split into " << y << " rows";
    EXPECT_LE(blocks / y * wgs, max_items) << blocks << " blocks split into " << y << " rows";
}

TEST(rocfft_internal, stockham_grid_y_leaves_launches_that_fit_unsplit)
{
    // no grid y limit at all: a launch that fits must not even look at it
    const hipDeviceProp_t prop{};
    EXPECT_EQ(stockham_grid_y(1, 64, prop), 1u);
    // 2^32 - 1 = 255 * 16843009 work items: exactly at the limit
    EXPECT_EQ(stockham_grid_y(16843009, 255, prop), 1u);
}

TEST(rocfft_internal, stockham_grid_y_splits_oversized_launches_exactly)
{
    // the reported 512^3 batch 128 case: 8388608 blocks of 512 threads
    expect_valid_split(8388608, 512);
    // one work item past the limit
    expect_valid_split(1u << 24, 256);
    // 2^24 + 1 = 97 * 257 * 673: the fewest rows that fit (2) do not divide
    // the block count, so a larger divisor has to be found
    expect_valid_split((1u << 24) + 1, 256);
}

TEST(rocfft_internal, stockham_grid_y_throws_when_no_split_fits)
{
    // 16777259 is prime, so no row count up to the grid y limit divides it;
    // plan creation must fail instead of launching an oversized grid
    EXPECT_THROW(stockham_grid_y(16777259, 256, prop_with_max_y(65536)), std::runtime_error);
}

TEST(rocfft_internal, stockham_grid_y_kernel_has_its_own_name)
{
    // kernels are cached by name: if a split kernel shared a name with the
    // unsplit one, a split launch could load the unsplit kernel and silently
    // skip blocks
    StockhamGeneratorSpecs specs(KIntType::U32,
                                 {8, 8},
                                 {},
                                 rocfft_precision_single,
                                 "gfx942",
                                 64,
                                 PrintScheme(CS_KERNEL_STOCKHAM_BLOCK_CC));
    auto                   split_specs = specs;
    split_specs.grid_y                 = true;

    auto name = [](const StockhamGeneratorSpecs& s) {
        return stockham_rtc_kernel_name(s,
                                        s,
                                        CS_KERNEL_STOCKHAM_BLOCK_CC,
                                        -1,
                                        rocfft_precision_single,
                                        rocfft_placement_inplace,
                                        rocfft_array_type_complex_interleaved,
                                        rocfft_array_type_complex_interleaved,
                                        true,
                                        0,
                                        0,
                                        false,
                                        FORCE_OFF_OR_NOT_SUPPORT,
                                        DISABLE_BOTH,
                                        NONE,
                                        CallbackType::NONE,
                                        BFT_NONE,
                                        PPT_NONE,
                                        {},
                                        std::nullopt,
                                        std::nullopt);
    };
    EXPECT_NE(name(specs), name(split_specs));
}
