/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#pragma once
#include <Tensile/analytical/AnalyticalGemm.hpp>
#include <Tensile/analytical/Hardware.hpp>
#include <Tensile/analytical/Utils.hpp>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

struct InputWithExpected
{
    std::map<std::string, int> values;
    std::optional<int>         expected;
    std::optional<int>         expected_gt;
    std::optional<int>         expected_lt;
};

struct MyTestData
{
    std::string                    name;
    std::vector<InputWithExpected> inputs;
};

// Parameterized test class declaration
class AnalyticalGtest : public ::testing::TestWithParam<MyTestData>
{
};

void ComputeLoads(
    int MT_M, int MT_N, int MT_K, const std::optional<int> expected, bool debug = false)
{
    auto a_loads  = TensileLite::analytical::compute_A_loads(MT_M, MT_K, debug);
    auto b_loads  = TensileLite::analytical::compute_B_loads(MT_N, MT_K, debug);
    auto cu_loads = TensileLite::analytical::compute_CU_loads(MT_M, MT_N, MT_K, debug);
    EXPECT_EQ(a_loads, expected);
    EXPECT_EQ(b_loads, expected);
    EXPECT_EQ(cu_loads, a_loads + b_loads);
}

void EstimateL2Hit(const TensileLite::analytical::Hardware& hardware,
                   int                                      M,
                   int                                      N,
                   int                                      K,
                   int                                      batch,
                   int                                      MT_M,
                   int                                      MT_N,
                   int                                      MT_K,
                   size_t                                   element_size,
                   const std::optional<int>                 expected_gt,
                   const std::optional<int>                 expected_lt)
{
    double l2_hit;
    for(int i = 1; i < 1025; i++)
    {
        l2_hit = TensileLite::analytical::estimate_l2_hit(
            hardware, M, N, K, batch, MT_M, MT_N, MT_K, i, element_size);
        EXPECT_GT(l2_hit, expected_gt);
        EXPECT_LT(l2_hit, expected_lt);
    }
}

void ComputeNumMatrixInstructions(const TensileLite::analytical::Hardware& hardware,
                   int                                      MT_M,
                   int                                      MT_N,
                   int                                      MT_K,
                   int                                      MI_M,
                   int                                      MI_N,
                   int                                      MI_K,
                   const std::optional<int>                 expected)
{
    auto NumberMatrixInstructions = TensileLite::analytical::compute_number_matrix_instructions(
        hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, false);
    EXPECT_EQ(NumberMatrixInstructions, expected);
}

void ComputeMTComputeLatency(const TensileLite::analytical::Hardware& hardware,
                                size_t          M,
                                          size_t          N,
                                          size_t          K,
                                          bool            transA,
                                          bool                              transB,
                                          size_t                                   MT_M,
                                          size_t                                   MT_N,
                                          size_t                                   MT_K,
                                          size_t                                   MI_M,
                                          size_t                                   MI_N,
                                          size_t                                   MI_K,
                                          size_t                                   element_size_A,
                                          size_t                                   element_size_B,
                                          const std::optional<int>                 expected,
                                          const std::optional<int>                 expected_gt)
{
    auto latency = TensileLite::analytical::compute_mt_compute_latency(
        hardware, M, N, K, transA, transB, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, element_size_A, element_size_B, TensileLite::analytical::DataType::BFloat16, false);

    if (expected.has_value())
        EXPECT_EQ(latency, expected);
    else if(expected_gt.has_value())
        EXPECT_GT(latency, expected_gt);
}

// Computes the number of MT timesteps required to compute all MT. Last wave may be less occupied.
void ComputeNumberWaves(const TensileLite::analytical::Hardware& hardware,
                                    size_t          M,
                                    size_t          N,
                                    size_t          batch,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          split,
                                    const std::optional<int>                 expected)
{
    auto num_waves
        = TensileLite::analytical::compute_number_waves(hardware, M, N, batch, MT_M, MT_N, split, false);
    EXPECT_EQ(num_waves, expected);
}

void ComputeActiveCU(const TensileLite::analytical::Hardware& hardware, size_t M, size_t N, size_t batch, size_t MT_M, size_t MT_N,
                     const std::optional<int>                 expected)
{
    auto active_cu = TensileLite::analytical::compute_active_CU(hardware, M, N, batch, MT_M, MT_N);
    EXPECT_EQ(active_cu, expected);
}

void ComputeMemoryLatency(const TensileLite::analytical::Hardware& hardware,
                                      size_t          M,
                                      size_t          N,
                                      size_t          K,
                                      size_t          batch,
                                      bool            transA,
                                      bool            transB,
                                      size_t          MT_M,
                                      size_t          MT_N,
                                      size_t          MT_K,
                                      size_t          split,
                                      double          H_mem,
                                      size_t          element_size_A,
                                      size_t          element_size_B,
                                      size_t          mx_block_size)
{
    auto mem_latency_small = TensileLite::analytical::compute_memory_latency(
        hardware, M, N, K, batch, transA, transB, MT_M, MT_N, MT_K, split, H_mem, element_size_A, element_size_B, mx_block_size, false);

    auto mem_latency_large = TensileLite::analytical::compute_memory_latency(
        hardware, M, N, K, batch, transA, transB, MT_M * 2, MT_N * 2, MT_K * 2, split, H_mem, element_size_A, element_size_B, mx_block_size, false);

    EXPECT_LT(mem_latency_small, mem_latency_large);
}

void ComputeTileLatency(const TensileLite::analytical::Hardware& hardware,
                                    size_t          M,
                                    size_t          N,
                                    size_t          K,
                                    size_t          batch,
                                    bool            transA,
                                    bool            transB,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          MT_K,
                                    size_t          MI_M,
                                    size_t          MI_N,
                                    size_t          MI_K,
                                    size_t          split,
                                    double          H_mem,
                                    size_t          element_size_A, //In bits
                                    size_t          element_size_B, //In bits,
                                    size_t          element_size_out, //In bits
                                    size_t          mx_block_size)
{
    auto   tile_latency_small = TensileLite::analytical::compute_tile_latency(hardware,
                                                                            M,
                                                                            N,
                                                                            K,
                                                                            batch,
                                                                            transA,
                                                                            transB,
                                                                            MT_M,
                                                                            MT_N,
                                                                            MT_K,
                                                                            MI_M,
                                                                            MI_N,
                                                                            MI_K,
                                                                            split,
                                                                            H_mem,
                                                                            element_size_A,
                                                                            element_size_B,
                                                                            element_size_out,
                                                                            TensileLite::analytical::DataType::BFloat16,
                                                                            mx_block_size,
                                                                            false);

    auto   tile_latency_large = TensileLite::analytical::compute_tile_latency(hardware,
                                                                            M,
                                                                            N,
                                                                            K,
                                                                            batch,
                                                                            transA,
                                                                            transB,
                                                                            MT_M * 2,
                                                                            MT_N * 2,
                                                                            MT_K * 2,
                                                                            MI_M,
                                                                            MI_N,
                                                                            MI_K,
                                                                            split,
                                                                            H_mem,
                                                                            element_size_A,
                                                                            element_size_B,
                                                                            element_size_out,
                                                                            TensileLite::analytical::DataType::BFloat16,
                                                                            mx_block_size,
                                                                            false);

    EXPECT_GT(tile_latency_large, tile_latency_small);
}

void ComputeWaveLatency(const TensileLite::analytical::Hardware& hardware,
                                    size_t          M,
                                    size_t          N,
                                    size_t          K,
                                    size_t          batch,
                                    bool            transA,
                                    bool            transB,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          MT_K,
                                    size_t          MI_M,
                                    size_t          MI_N,
                                    size_t          MI_K,
                                    size_t          split,
                                    double          H_mem,
                                    size_t          element_size_A, //In bits
                                    size_t          element_size_B, //In bits,
                                    size_t          element_size_out, //In bits
                                    size_t          mx_block_size)
{
    auto   tile_latency = TensileLite::analytical::compute_tile_latency(hardware,
                                                                            M,
                                                                            N,
                                                                            K,
                                                                            batch,
                                                                            transA,
                                                                            transB,
                                                                            MT_M,
                                                                            MT_N,
                                                                            MT_K,
                                                                            MI_M,
                                                                            MI_N,
                                                                            MI_K,
                                                                            split,
                                                                            H_mem,
                                                                            element_size_A,
                                                                            element_size_B,
                                                                            element_size_out,
                                                                            TensileLite::analytical::DataType::BFloat16,
                                                                            mx_block_size,
                                                                            false);
    auto   wave_latency = TensileLite::analytical::compute_wave_latency(hardware,
                                                                            M,
                                                                            N,
                                                                            K,
                                                                            batch,
                                                                            transA,
                                                                            transB,
                                                                            MT_M,
                                                                            MT_N,
                                                                            MT_K,
                                                                            MI_M,
                                                                            MI_N,
                                                                            MI_K,
                                                                            split,
                                                                            H_mem,
                                                                            element_size_A,
                                                                            element_size_B,
                                                                            element_size_out,
                                                                            TensileLite::analytical::DataType::BFloat16,
                                                                            mx_block_size,
                                                                            false);
    EXPECT_DOUBLE_EQ(wave_latency, tile_latency);
}

void ComputeTotalLatency(const TensileLite::analytical::Hardware& hardware,
                                    size_t          M,
                                    size_t          N,
                                    size_t          K,
                                    size_t          batch,
                                    bool            transA,
                                    bool            transB,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          MT_K,
                                    size_t          MI_M,
                                    size_t          MI_N,
                                    size_t          MI_K,
                                    size_t          split,
                                    double          H_mem,
                                    size_t          element_size_A, //In bits
                                    size_t          element_size_B, //In bits,
                                    size_t          element_size_out, //In bits
                                    int             WGM,
                                    size_t          mx_block_size)
{
    double latency_cycles_small = TensileLite::analytical::compute_total_latency(hardware,
                                                                            M,
                                                                            N,
                                                                            K,
                                                                            batch,
                                                                            transA,
                                                                            transB,
                                                                            MT_M,
                                                                            MT_N,
                                                                            MT_K,
                                                                            MI_M,
                                                                            MI_N,
                                                                            MI_K,
                                                                            split,
                                                                            H_mem,
                                                                            element_size_A,
                                                                            element_size_B,
                                                                            element_size_out,
                                                                            TensileLite::analytical::DataType::BFloat16,
                                                                            WGM,
                                                                            mx_block_size,
                                                                            false);

    double latency_cycles_large = TensileLite::analytical::compute_total_latency(hardware,
                                                                            M,
                                                                            N,
                                                                            K,
                                                                            batch,
                                                                            transA,
                                                                            transB,
                                                                            MT_M * 2,
                                                                            MT_N * 2,
                                                                            MT_K * 2,
                                                                            MI_M,
                                                                            MI_N,
                                                                            MI_K,
                                                                            split,
                                                                            H_mem,
                                                                            element_size_A,
                                                                            element_size_B,
                                                                            element_size_out,
                                                                            TensileLite::analytical::DataType::BFloat16,
                                                                            WGM,
                                                                            mx_block_size,
                                                                            false);
    EXPECT_LT(latency_cycles_small, latency_cycles_large);
}

void ComputePerfGflops(size_t          M,
                                    size_t          N,
                                    size_t          K,
                                    size_t          batch,
                                    bool            transA,
                                    bool            transB,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          MT_K,
                                    size_t          MI_M,
                                    size_t          MI_N,
                                    size_t          MI_K,
                                    double          H_mem,
                                    size_t          element_size_A, //In bits
                                    size_t          element_size_B, //In bits,
                                    size_t          element_size_out, //In bits
                                    int             WGM)
{
    auto gfx942arch  = TensileLite::analytical::Hardware::archNameToEnum("gfx942");
    auto gfx942_slow = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.4, 1, 1.0);
    auto gfx942_fast = TensileLite::analytical::Hardware(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.8, 1, 1.0);
    double flops_slow = TensileLite::analytical::compute_perf_gflops(gfx942_slow,
                                                                     M,
                                                                     N,
                                                                     K,
                                                                     batch,
                                                                     transA,
                                                                     transB,
                                                                     MT_M,
                                                                     MT_N,
                                                                     MT_K,
                                                                     MI_M,
                                                                     MI_N,
                                                                     MI_K,
                                                                     H_mem,
                                                                     WGM,
                                                                     element_size_A,
                                                                     element_size_B,
                                                                     element_size_out,
                                                                     TensileLite::analytical::DataType::BFloat16,
                                                                     false);
    double flops_fast = TensileLite::analytical::compute_perf_gflops(gfx942_fast,
                                                                     M,
                                                                     N,
                                                                     K,
                                                                     batch,
                                                                     transA,
                                                                     transB,
                                                                     MT_M,
                                                                     MT_N,
                                                                     MT_K,
                                                                     MI_M,
                                                                     MI_N,
                                                                     MI_K,
                                                                     H_mem,
                                                                     WGM,
                                                                     element_size_A,
                                                                     element_size_B,
                                                                     element_size_out,
                                                                     TensileLite::analytical::DataType::BFloat16,
                                                                     false);
    EXPECT_GT(flops_fast, flops_slow); // faster clock = higher flops
}

void EstimateMallHit(const TensileLite::analytical::Hardware& hardware,
                                 int             M,
                                 int             N,
                                 int             K,
                                 int             batch,
                                 int             MT_M,
                                 int             MT_N,
                                 int             MT_K,
                                 const std::optional<int>                 expected_gt)
{
    double mall_hit;
    for(int i = 1; i < 1025; i++)
    {
        mall_hit = TensileLite::analytical::estimate_mall_hit(
            hardware, M, N, K, batch, MT_M, MT_N, MT_K, i);
        EXPECT_GT(mall_hit, expected_gt);
    }
}

void CheckLDSCapacity(const TensileLite::analytical::Hardware& hardware,
                                 int             MT_M,
                                 int             MT_N,
                                 int             MT_K,
                                 size_t          element_size)
{
    auto fit_lds_memory
        = TensileLite::analytical::check_LDS_capacity(hardware, MT_M, MT_N, MT_K, element_size, false);
    EXPECT_TRUE(fit_lds_memory);
}

// Hardware
void HardwareArchEnum(const std::string gpuArchNumber)
{
    auto gpuArchEnum = TensileLite::analytical::Hardware::archNameToEnum("gfx" + gpuArchNumber);
    EXPECT_EQ(gpuArchEnum, TensileLite::analytical::Hardware::Architecture::gfx942);
}

// Utils
void BestGridSize(const TensileLite::analytical::Hardware& hardware,
                                     size_t          M,
                                     size_t          N,
                                     size_t          K,
                                     size_t          batch,
                                     bool            transA,
                                     bool            transB,
                                     size_t          MT_M,
                                     size_t          MT_N,
                                     size_t          MT_K,
                                     size_t          MI_M,
                                     size_t          MI_N,
                                     size_t          MI_K,
                                     size_t          element_size_A,
                                     size_t          element_size_B,
                                     size_t          element_size_out,
                                     size_t          mx_block_size,
                                     double          H_L2,
                                     size_t          WGM,
                                     size_t          biggest_allowable_split,
                                     const std::optional<int>                 expected_gt)
{
    size_t grid_size = TensileLite::analytical::select_best_grid_size(M,
                                                                      N,
                                                                      K,
                                                                      batch,
                                                                      transA,
                                                                      transB,
                                                                      hardware,
                                                                      MT_M,
                                                                      MT_N,
                                                                      MT_K,
                                                                      MI_M,
                                                                      MI_N,
                                                                      MI_K,
                                                                      element_size_A,
                                                                      element_size_B,
                                                                      element_size_out,
                                                                      TensileLite::analytical::DataType::BFloat16,
                                                                      mx_block_size,
                                                                      H_L2,
                                                                      false,
                                                                      WGM,
                                                                      biggest_allowable_split);
    EXPECT_GT(grid_size, expected_gt);
}

void BestMacroTileSize(const TensileLite::analytical::Hardware& hardware,
                       size_t                        M,
                       size_t                        N,
                       size_t                        K,
                       size_t                        batch,
                       bool                          transA,
                                                             bool                          transB,
                                                             size_t element_size_A, //In bits
                                                             size_t element_size_B, //In bits
                                                             size_t element_size_out, //In bits
                                                             size_t mx_block_size,
                                                             double H_L2,
                                                             size_t WGM)
{
    std::vector<std::tuple<size_t, // MT_M
                           size_t, // MT_N
                           size_t, // MT_K
                           size_t, // MI_M
                           size_t, // MI_N
                           size_t, // MI_K
                           size_t // Occupancy
                           >>
        MT_list
        = {{256, 256, 32, 32, 32, 8, 1}, {128, 128, 64, 32, 32, 8, 1}, {64, 64, 64, 32, 32, 8, 1}};
    auto results = select_best_macro_tile_size(
        M, N, K, batch, transA, transB, hardware, MT_list, element_size_A, element_size_B, element_size_out, TensileLite::analytical::DataType::BFloat16, mx_block_size, H_L2, false, false, WGM);

    EXPECT_EQ(results.size(), MT_list.size());
    for(int i = 0; i < results.size() - 1; i++)
        EXPECT_LT(std::get<0>(results[i]), std::get<0>(results[i + 1]));
}

void BestWGM(const TensileLite::analytical::Hardware& hardware,
            size_t                     M,
            size_t                     N,
            size_t                     K,
            size_t                     batch,
            size_t                     MT_M,
            size_t                     MT_N,
            size_t                     MT_K,
            size_t                     MI_M,
            size_t                     MI_N,
            size_t                     MI_K,
            size_t                     element_size,
            double                     H_L2) // not needed for L2 hit rate but retained if your code expects it
{
    std::vector<size_t> WGM_list = {1, 2, 4, 6, 8, 12};

    auto                best_wgm_large_tile = select_best_wgm(
        M, N, K, batch, hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, WGM_list, element_size, H_L2, false, false);

    auto best_wgm_small_tile = select_best_wgm(
        M, N, K, batch, hardware, MT_M/2, MT_N/2, MT_K * 2, MI_M, MI_N, MI_K, WGM_list, element_size, H_L2, false, false);

    auto best_wgm_nonsquare = select_best_wgm(
        2048, 5120, K, batch, hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, WGM_list, element_size, H_L2, false, false);

    EXPECT_EQ(best_wgm_large_tile.second, best_wgm_small_tile.second);
    EXPECT_NE(best_wgm_large_tile.second, best_wgm_nonsquare.second);
}

void UtilsTFlopsFromLatency(size_t M, size_t N, size_t K, double latency_cycles, double clock_GHz)
{
    auto tflops = TensileLite::analytical::compute_TFLOPS_from_latency(
        latency_cycles, M, N, K, clock_GHz, false);
    double Expected = 1.99;
    EXPECT_LT(std::abs(tflops - Expected) / std::abs(Expected), 0.01);
}

