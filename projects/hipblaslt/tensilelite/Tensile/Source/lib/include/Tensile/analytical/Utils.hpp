/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <Tensile/analytical/AnalyticalGemm.hpp>
#include <Tensile/analytical/Hardware.hpp>
#include <set>
#include <tuple>
#include <vector>
// #include "Hardware.hpp"
// #include "AnalyticalGemm.hpp"
#include <functional> // For std::function

namespace TensileLite
{
    namespace analytical
    {

        using ResultTuple = std::tuple<double, // latency
                                       size_t, // MT_M
                                       size_t, // MT_N
                                       size_t, // MT_K
                                       size_t, // MI_M
                                       size_t, // MI_N
                                       size_t, // MI_K
                                       size_t  // Occupancy
                                       >;

        using TileTuple = std::tuple<size_t, // MT_M
                                     size_t, // MT_N
                                     size_t, // MT_K
                                     size_t, // MI_M
                                     size_t, // MI_N
                                     size_t, // MI_K
                                     size_t  // Occupancy
                                     >;

        size_t select_best_grid_size(size_t          M,
                                     size_t          N,
                                     size_t          K,
                                     size_t          batch,
                                     bool            transA,
                                     bool            transB,
                                     const Hardware& hardware,
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
                                     bool            debug,
                                     size_t          WGM,
                                     size_t          biggest_allowable_split = 8);

        std::vector<ResultTuple> select_best_macro_tile_size(size_t                        M,
                                                             size_t                        N,
                                                             size_t                        K,
                                                             size_t                        batch,
                                                             bool                          transA,
                                                             bool                          transB,
                                                             const Hardware&               hardware,
                                                             const std::vector<TileTuple>& MT_list,
                                                             size_t element_size_A,
                                                             size_t element_size_B,
                                                             size_t element_size_out,
                                                             size_t mx_block_size,
                                                             double H_L2,
                                                             bool   debug,
                                                             bool   print,
                                                             size_t WGM);

        std::vector<ResultTuple> sweep_macro_tile_sizes(size_t    M,
                                                        size_t    N,
                                                        size_t    K,
                                                        bool      transA,
                                                        bool      transB,
                                                        Hardware& hardware,
                                                        size_t    element_size = 2,
                                                        size_t    max_MT_M     = 256,
                                                        size_t    max_MT_N     = 256,
                                                        size_t    max_MT_K     = 128,
                                                        size_t    step_MT_M    = 32,
                                                        size_t    step_MT_N    = 32,
                                                        size_t    step_MT_K    = 32,
                                                        double    H_L2         = 0.8,
                                                        bool      debug        = false,
                                                        const std::vector<TileTuple>& tiles_to_add
                                                        = {},
                                                        bool print = false);

        std::pair<double, size_t> select_best_wgm(
            size_t                     M,
            size_t                     N,
            size_t                     K,
            size_t                     batch,
            Hardware&                  hardware,
            size_t                     MT_M,
            size_t                     MT_N,
            size_t                     MT_K,
            size_t                     MI_M,
            size_t                     MI_N,
            size_t                     MI_K,
            const std::vector<size_t>& WGM_list,
            size_t                     element_size,
            double H_L2, // not needed for L2 hit rate but retained if your code expects it
            bool   debug,
            bool   print);

        double compute_TFLOPS_from_latency(double latency_cycles,
                                           size_t M,
                                           size_t N,
                                           size_t K,
                                           double clock_GHz,
                                           bool   debug = false);

    } // namespace analytical
} // namespace TensileLite
