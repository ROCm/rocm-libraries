// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include <set>
#include <tuple>
#include <vector>
#include <functional> // For std::function

namespace origami
{
        using result_tuple = std::tuple<double, // latency
                                       size_t, // MT_M
                                       size_t, // MT_N
                                       size_t, // MT_K
                                       size_t, // MI_M
                                       size_t, // MI_N
                                       size_t, // MI_K
                                       size_t,  // Occupancy
                                       int,     // WGM
                                       size_t, // non_temporal_a
                                       size_t>; // non_temporal_b

        using tile_tuple = std::tuple<size_t, // MT_M
                                     size_t, // MT_N
                                     size_t, // MT_K
                                     size_t, // MI_M
                                     size_t, // MI_N
                                     size_t, // MI_K
                                     size_t,  // Occupancy
                                     int,     // WGM
                                     size_t, // non_temporal_a
                                     size_t>; // non_temporal_b

        size_t select_best_grid_size(size_t          M,
                                     size_t          N,
                                     size_t          K,
                                     size_t          batch,
                                     bool            transA,
                                     bool            transB,
                                     const hardware_t& hardware,
                                     size_t          MT_M,
                                     size_t          MT_N,
                                     size_t          MT_K,
                                     size_t          MI_M,
                                     size_t          MI_N,
                                     size_t          MI_K,
                                     size_t          element_size_A,
                                     size_t          element_size_B,
                                     size_t          element_size_out,
                                     data_type_t     mi_datatype,
                                     size_t          mx_block_size,
                                     double          H_L2,
                                     size_t          WGM,
                                     size_t          biggest_allowable_split = 8);

        std::vector<result_tuple> select_best_macro_tile_size(size_t                        M,
                                                             size_t                        N,
                                                             size_t                        K,
                                                             size_t                        batch,
                                                             bool                          transA,
                                                             bool                          transB,
                                                             const hardware_t&             hardware,
                                                             const std::vector<tile_tuple>& MT_list,
                                                             size_t element_size_A,
                                                             size_t element_size_B,
                                                             size_t element_size_out,
                                                             data_type_t mi_datatype,
                                                             size_t mx_block_size,
                                                             double H_L2,
                                                             bool   print,
                                                             size_t WGM);

        std::vector<result_tuple> sweep_macro_tile_sizes(size_t    M,
                                                        size_t    N,
                                                        size_t    K,
                                                        bool      transA,
                                                        bool      transB,
                                                        hardware_t& hardware,
                                                        size_t    element_size = 2,
                                                        size_t    max_MT_M     = 256,
                                                        size_t    max_MT_N     = 256,
                                                        size_t    max_MT_K     = 128,
                                                        size_t    step_MT_M    = 32,
                                                        size_t    step_MT_N    = 32,
                                                        size_t    step_MT_K    = 32,
                                                        double    H_L2         = 0.8,
                                                        const std::vector<tile_tuple>& tiles_to_add
                                                        = {},
                                                        bool print = false);

        std::pair<double, size_t> select_best_wgm(
            size_t                     M,
            size_t                     N,
            size_t                     K,
            size_t                     batch,
            const hardware_t&          hardware,
            size_t                     MT_M,
            size_t                     MT_N,
            size_t                     MT_K,
            size_t                     MI_M,
            size_t                     MI_N,
            size_t                     MI_K,
            const std::vector<size_t>& WGM_list,
            size_t                     element_size,
            double H_L2, // not needed for L2 hit rate but retained if your code expects it
            bool   print);

        double compute_tflops_from_latency(double latency_cycles,
                                           size_t M,
                                           size_t N,
                                           size_t K,
                                           double clock_GHz);

} // namespace origami
