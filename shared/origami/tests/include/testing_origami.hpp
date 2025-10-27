// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/utils.hpp"
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

void ComputeLoads(int MT_M, int MT_N, int MT_K, const std::optional<int> expected)
{
    auto a_loads = origami::compute_A_loads(MT_M, MT_K);
    auto b_loads = origami::compute_B_loads(MT_N, MT_K);
    EXPECT_EQ(a_loads, expected);
    EXPECT_EQ(b_loads, expected);
}

void EstimateL2Hit(const origami::hardware_t& hardware,
                   int                        M,
                   int                        N,
                   int                        K,
                   int                        batch,
                   int                        MT_M,
                   int                        MT_N,
                   int                        MT_K,
                   size_t                     element_size,
                   int                        splittingFactor,
                   const std::optional<int>   expected_gt,
                   const std::optional<int>   expected_lt)
{
    double l2_hit;
    for(int i = 1; i < 1025; i++)
    {
        l2_hit = origami::estimate_l2_hit(
            hardware, M, N, K, batch, MT_M, MT_N, MT_K, element_size, i, splittingFactor);
        EXPECT_GT(l2_hit, expected_gt);
        EXPECT_LT(l2_hit, expected_lt);
    }
}

void ComputeNumMatrixInstructions(const origami::hardware_t& hardware,
                                  int                        MT_M,
                                  int                        MT_N,
                                  int                        MT_K,
                                  int                        MI_M,
                                  int                        MI_N,
                                  int                        MI_K,
                                  const std::optional<int>   expected)
{
    auto NumberMatrixInstructions
        = origami::compute_number_matrix_instructions(hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K);
    EXPECT_EQ(NumberMatrixInstructions, expected);
}

void ComputeMTComputeLatency(const origami::hardware_t& hardware,
                             size_t                     M,
                             size_t                     N,
                             size_t                     K,
                             bool                       transA,
                             bool                       transB,
                             size_t                     MT_M,
                             size_t                     MT_N,
                             size_t                     MT_K,
                             size_t                     MI_M,
                             size_t                     MI_N,
                             size_t                     MI_K,
                             size_t                     element_size_A,
                             size_t                     element_size_B,
                             const std::optional<int>   expected,
                             const std::optional<int>   expected_gt)
{
    auto latency = origami::compute_mt_compute_latency(hardware,
                                                       M,
                                                       N,
                                                       K,
                                                       transA,
                                                       transB,
                                                       MT_M,
                                                       MT_N,
                                                       MT_K,
                                                       MI_M,
                                                       MI_N,
                                                       MI_K,
                                                       element_size_A,
                                                       element_size_B,
                                                       origami::data_type_t::BFloat16);

    if(expected.has_value())
        EXPECT_EQ(latency, expected);
    else if(expected_gt.has_value())
        EXPECT_GT(latency, expected_gt);
}

void ComputeMemoryLatency(const origami::hardware_t& hardware,
                          size_t                     M,
                          size_t                     N,
                          size_t                     K,
                          size_t                     batch,
                          bool                       transA,
                          bool                       transB,
                          size_t                     MT_M,
                          size_t                     MT_N,
                          size_t                     MT_K,
                          size_t                     element_size_A,
                          size_t                     element_size_B,
                          size_t                     mx_block_size,
                          int                        wgm,
                          int                        numActiveCUs,
                          int                        splittingFactor)
{
    auto mem_latency_small = origami::compute_memory_latency(hardware,
                                                             M,
                                                             N,
                                                             K,
                                                             transA,
                                                             transB,
                                                             batch,
                                                             MT_M,
                                                             MT_N,
                                                             MT_K,
                                                             element_size_A,
                                                             element_size_B,
                                                             mx_block_size,
                                                             wgm,
                                                             numActiveCUs,
                                                             splittingFactor);

    auto mem_latency_large = origami::compute_memory_latency(hardware,
                                                             M,
                                                             N,
                                                             K,
                                                             transA,
                                                             transB,
                                                             batch,
                                                             MT_M * 2,
                                                             MT_N * 2,
                                                             MT_K * 2,
                                                             element_size_A,
                                                             element_size_B,
                                                             mx_block_size,
                                                             wgm,
                                                             numActiveCUs,
                                                             splittingFactor);

    EXPECT_LT(mem_latency_small, mem_latency_large);
}

void ComputeTileLatency(const origami::hardware_t& hardware,
                        size_t                     M,
                        size_t                     N,
                        size_t                     K,
                        size_t                     batch,
                        bool                       transA,
                        bool                       transB,
                        size_t                     MT_M,
                        size_t                     MT_N,
                        size_t                     MT_K,
                        size_t                     MI_M,
                        size_t                     MI_N,
                        size_t                     MI_K,
                        size_t                     element_size_A, //In bits
                        size_t                     element_size_B, //In bits,
                        size_t                     element_size_out, //In bits
                        size_t                     mx_block_size,
                        int                        WGM,
                        size_t                     numActiveCUs,
                        size_t                     splittingFactor)
{
    auto tile_latency_small = origami::compute_tile_latency(hardware,
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
                                                            element_size_A,
                                                            element_size_B,
                                                            element_size_out,
                                                            origami::data_type_t::BFloat16,
                                                            mx_block_size,
                                                            WGM,
                                                            1,
                                                            numActiveCUs,
                                                            splittingFactor);

    auto tile_latency_large = origami::compute_tile_latency(hardware,
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
                                                            element_size_A,
                                                            element_size_B,
                                                            element_size_out,
                                                            origami::data_type_t::BFloat16,
                                                            mx_block_size,
                                                            WGM,
                                                            1,
                                                            numActiveCUs,
                                                            splittingFactor);

    EXPECT_GT(tile_latency_large, tile_latency_small);
}

void ComputeWaveLatency(const origami::hardware_t& hardware,
                        size_t                     M,
                        size_t                     N,
                        size_t                     K,
                        size_t                     batch,
                        bool                       transA,
                        bool                       transB,
                        size_t                     MT_M,
                        size_t                     MT_N,
                        size_t                     MT_K,
                        size_t                     MI_M,
                        size_t                     MI_N,
                        size_t                     MI_K,
                        size_t                     element_size_A, //In bits
                        size_t                     element_size_B, //In bits,
                        size_t                     element_size_out, //In bits
                        size_t                     mx_block_size,
                        int                        WGM,
                        size_t                     numActiveCUs,
                        size_t                     splittingFactor)
{
    auto tile_latency = origami::compute_tile_latency(hardware,
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
                                                      element_size_A,
                                                      element_size_B,
                                                      element_size_out,
                                                      origami::data_type_t::BFloat16,
                                                      mx_block_size,
                                                      WGM,
                                                      1,
                                                      numActiveCUs,
                                                      splittingFactor);
    auto wave_latency = origami::compute_wave_latency(hardware,
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
                                                      element_size_A,
                                                      element_size_B,
                                                      element_size_out,
                                                      origami::data_type_t::BFloat16,
                                                      mx_block_size,
                                                      WGM,
                                                      1,
                                                      numActiveCUs,
                                                      splittingFactor);
    EXPECT_DOUBLE_EQ(wave_latency, tile_latency);
}

void ComputeTotalLatency(const origami::hardware_t& hardware,
                         size_t                     M,
                         size_t                     N,
                         size_t                     K,
                         size_t                     batch,
                         bool                       transA,
                         bool                       transB,
                         size_t                     MT_M,
                         size_t                     MT_N,
                         size_t                     MT_K,
                         size_t                     MI_M,
                         size_t                     MI_N,
                         size_t                     MI_K,
                         size_t                     element_size_A, //In bits
                         size_t                     element_size_B, //In bits,
                         size_t                     element_size_out, //In bits
                         size_t                     mx_block_size,
                         int                        WGM,
                         size_t                     splittingFactor,
                         size_t                     max_cus)
{
    double latency_cycles_small = origami::compute_total_latency(hardware,
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
                                                                 element_size_A,
                                                                 element_size_B,
                                                                 element_size_out,
                                                                 origami::data_type_t::BFloat16,
                                                                 mx_block_size,
                                                                 WGM,
                                                                 0,
                                                                 0,
                                                                 splittingFactor,
                                                                 max_cus);

    double latency_cycles_large = origami::compute_total_latency(hardware,
                                                                 M * 2,
                                                                 N * 2,
                                                                 K * 2,
                                                                 batch,
                                                                 transA,
                                                                 transB,
                                                                 MT_M,
                                                                 MT_N,
                                                                 MT_K,
                                                                 MI_M,
                                                                 MI_N,
                                                                 MI_K,
                                                                 element_size_A,
                                                                 element_size_B,
                                                                 element_size_out,
                                                                 origami::data_type_t::BFloat16,
                                                                 mx_block_size,
                                                                 WGM,
                                                                 0,
                                                                 0,
                                                                 splittingFactor,
                                                                 max_cus);
    EXPECT_LT(latency_cycles_small, latency_cycles_large);
}

void ComputePerfGflops(size_t M,
                       size_t N,
                       size_t K,
                       size_t batch,
                       bool   transA,
                       bool   transB,
                       size_t MT_M,
                       size_t MT_N,
                       size_t MT_K,
                       size_t MI_M,
                       size_t MI_N,
                       size_t MI_K,
                       size_t element_size_A, //In bits
                       size_t element_size_B, //In bits,
                       size_t element_size_out, //In bits
                       int    WGM,
                       size_t max_cus)
{
    auto gfx942arch  = origami::hardware_t::arch_name_to_enum("gfx942");
    auto gfx942_slow = origami::hardware_t(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.4, 1, std::make_tuple(0, 0.015, 0));
    auto gfx942_fast = origami::hardware_t(
        gfx942arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.8, 1, std::make_tuple(0, 0.015, 0));
    double flops_slow = origami::compute_perf_gflops(gfx942_slow,
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
                                                     element_size_A,
                                                     element_size_B,
                                                     element_size_out,
                                                     origami::data_type_t::BFloat16,
                                                     WGM,
                                                     max_cus);
    double flops_fast = origami::compute_perf_gflops(gfx942_fast,
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
                                                     element_size_A,
                                                     element_size_B,
                                                     element_size_out,
                                                     origami::data_type_t::BFloat16,
                                                     WGM,
                                                     max_cus);
    EXPECT_GT(flops_fast, flops_slow); // faster clock = higher flops
}

void EstimateMallHit(const origami::hardware_t& hardware,
                     int                        M,
                     int                        N,
                     int                        K,
                     int                        batch,
                     int                        MT_M,
                     int                        MT_N,
                     int                        MT_K,
                     size_t                     element_size,
                     size_t                     numActiveCUs,
                     size_t                     splittingFactor,
                     const std::optional<int>   expected_gt)
{
    double mall_hit;
    for(int i = 1; i < 1025; i++)
    {
        mall_hit = origami::estimate_mall_hit(hardware,
                                              M,
                                              N,
                                              K,
                                              batch,
                                              MT_M,
                                              MT_N,
                                              MT_K,
                                              element_size,
                                              i,
                                              numActiveCUs,
                                              splittingFactor);
        EXPECT_GT(mall_hit, expected_gt);
    }
}

void CheckLDSCapacity(
    const origami::hardware_t& hardware, int MT_M, int MT_N, int MT_K, size_t element_size)
{
    auto fit_lds_memory = origami::check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size);
    EXPECT_TRUE(fit_lds_memory);
}

// hardware_t
void HardwareArchEnum(const std::string gpuArchNumber)
{
    auto gpuArchEnum = origami::hardware_t::arch_name_to_enum("gfx" + gpuArchNumber);
    EXPECT_EQ(gpuArchEnum, origami::hardware_t::architecture_t::gfx942);
}

// Utils
void BestGridSize(const origami::hardware_t& hardware,
                  size_t                     M,
                  size_t                     N,
                  size_t                     K,
                  size_t                     batch,
                  bool                       transA,
                  bool                       transB,
                  size_t                     MT_M,
                  size_t                     MT_N,
                  size_t                     MT_K,
                  size_t                     MI_M,
                  size_t                     MI_N,
                  size_t                     MI_K,
                  size_t                     element_size_A,
                  size_t                     element_size_B,
                  size_t                     element_size_out,
                  size_t                     mx_block_size,
                  double                     H_L2,
                  size_t                     WGM,
                  size_t                     biggest_allowable_split,
                  size_t                     max_cus,
                  const std::optional<int>   expected_gt)
{
    size_t grid_size = origami::select_best_grid_size(M,
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
                                                      origami::data_type_t::BFloat16,
                                                      mx_block_size,
                                                      H_L2,
                                                      WGM,
                                                      biggest_allowable_split,
                                                      max_cus);
    EXPECT_GT(grid_size, expected_gt);
}

void BestMacroTileSize(const origami::hardware_t& hardware,
                       size_t                     M,
                       size_t                     N,
                       size_t                     K,
                       size_t                     batch,
                       bool                       transA,
                       bool                       transB,
                       size_t                     element_size_A, //In bits
                       size_t                     element_size_B, //In bits
                       size_t                     element_size_out, //In bits
                       size_t                     mx_block_size,
                       double                     H_L2,
                       size_t                     WGM,
                       size_t                     max_cus)
{
    const std::vector<std::tuple<size_t, // MT_M
                                 size_t, // MT_N
                                 size_t, // MT_K
                                 size_t, // MI_M
                                 size_t, // MI_N
                                 size_t, // MI_K
                                 size_t, // Occupancy
                                 int, // wgm
                                 size_t, // non_temporal_a
                                 size_t // non_temporal_b
                                 >>
         MT_list = {{256, 256, 32, 32, 32, 8, 1, 6, 0, 0},
                    {128, 128, 64, 32, 32, 8, 1, 6, 0, 0},
                    {64, 64, 64, 32, 32, 8, 1, 6, 0, 0}};
    auto results = select_best_macro_tile_size(M,
                                               N,
                                               K,
                                               batch,
                                               transA,
                                               transB,
                                               hardware,
                                               MT_list,
                                               element_size_A,
                                               element_size_B,
                                               element_size_out,
                                               origami::data_type_t::BFloat16,
                                               mx_block_size,
                                               H_L2,
                                               false,
                                               WGM,
                                               max_cus);

    EXPECT_EQ(results.size(), MT_list.size());
    for(int i = 0; i < results.size() - 1; i++)
        EXPECT_LT(std::get<0>(results[i]), std::get<0>(results[i + 1]));
}

void BestWGM(const origami::hardware_t& hardware,
             size_t                     M,
             size_t                     N,
             size_t                     K,
             size_t                     batch,
             size_t                     MT_M,
             size_t                     MT_N,
             size_t                     MT_K)
{
    std::vector<size_t> WGM_list = {1, 2, 4, 6, 8, 12};

    auto best_wgm_large_tile = select_best_wgm(hardware,
                                               M,
                                               N,
                                               K,
                                               batch,
                                               MT_M,
                                               MT_N,
                                               MT_K,
                                               false);

    auto best_wgm_small_tile = select_best_wgm(hardware,
                                               M / 4,
                                               N / 4,
                                               K,
                                               batch,
                                               MT_M,
                                               MT_N,
                                               MT_K * 2,
                                               false);

    auto best_wgm_nonsquare = select_best_wgm(hardware,
                                              1024,
                                              5120,
                                              K,
                                              batch,
                                              MT_M,
                                              MT_N,
                                              MT_K,
                                              false);

    EXPECT_GT(best_wgm_large_tile, best_wgm_small_tile);
    EXPECT_NE(best_wgm_large_tile, best_wgm_nonsquare);
}

void UtilsTFlopsFromLatency(size_t M, size_t N, size_t K, double latency_cycles, double clock_GHz)
{
    auto   tflops   = origami::compute_tflops_from_latency(latency_cycles, M, N, K, clock_GHz);
    double Expected = 1.99;
    EXPECT_LT(std::abs(tflops - Expected) / std::abs(Expected), 0.01);
}
