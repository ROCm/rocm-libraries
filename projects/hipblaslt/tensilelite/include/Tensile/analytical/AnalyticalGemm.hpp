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

#include <Tensile/analytical/Hardware.hpp>
#include <vector>

namespace TensileLite
{
    namespace analytical
    {
        // Placeholder for compute_reuse_in_block_gemm function.
        // TODO move over L2 hit rate simulation for tie-breaking.
        double compute_reuse_in_block_gemm(size_t                  grid_m,
                                           size_t                  grid_n,
                                           size_t                  grid_k,
                                           size_t                  A_size,
                                           size_t                  B_size,
                                           size_t                  C_size,
                                           size_t                  nproc,
                                           size_t                  capacity,
                                           const std::vector<int>& radix,
                                           bool                    print_radix,
                                           bool                    print_output,
                                           size_t                  max_timesteps,
                                           size_t                  max_iters);

        // Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
        size_t compute_number_matrix_instructions(const Hardware& hardware,
                                                  size_t          MT_M,
                                                  size_t          MT_N,
                                                  size_t          MT_K,
                                                  size_t          MI_M,
                                                  size_t          MI_N,
                                                  size_t          MI_K,
                                                  bool            debug);

        // Determine the compute latency per MT_MxMT_NxMT_K Macro Tile (L_MT).
        size_t compute_mt_compute_latency(const Hardware& hardware,
                                          size_t          M,
                                          size_t          N,
                                          size_t          K,
                                          bool            transA,
                                          bool            transB,
                                          size_t          MT_M,
                                          size_t          MT_N,
                                          size_t          MT_K,
                                          size_t          MI_M,
                                          size_t          MI_N,
                                          size_t          MI_K,
                                          size_t          element_size_A, //In bits
                                          size_t          element_size_B, //In bits,
                                          bool            debug);

        // Computes the number of MT timesteps required to compute all MT. Last wave may be less occupied.
        size_t compute_number_waves(const Hardware& hardware,
                                    size_t          M,
                                    size_t          N,
                                    size_t          batch,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          split,
                                    bool            debug);

        // Compute the amount of data loaded from A to produce a MT_MxMT_NxMT_K tile.
        size_t compute_A_loads(size_t MT_M, size_t MT_K, bool debug);

        // Compute the amount of data loaded from B to produce a MT_MxMT_NxMT_K tile.
        size_t compute_B_loads(size_t MT_N, size_t MT_K, bool debug);

        // Computes total data loads per CU per MT from A and B
        // Reads happen every MT, Writes happen every K-complete tile.
        size_t compute_CU_loads(size_t MT_M, size_t MT_N, size_t MT_K, bool debug);

        // Computes the number of active compute units if there is only one wave and it is partial
        // Otherwise, returns hardware.N_CU
        size_t compute_active_CU(
            const Hardware& hardware, size_t M, size_t N, size_t MT_M, size_t MT_N);

        double compute_memory_latency(const Hardware& hardware,
                                      size_t          M,
                                      size_t          N,
                                      size_t          K,
                                      size_t          batch,
                                      size_t          MT_M,
                                      size_t          MT_N,
                                      size_t          MT_K,
                                      size_t          split,
                                      double          H_L2,
                                      size_t          element_size_A, //In bits
                                      size_t          element_size_B, //In bits,
                                      size_t          mx_block_size,
                                      bool            debug);

        // Computes the latency to compute a K-COMPLETE tile.
        double compute_tile_latency(const Hardware& hardware,
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
                                    double          H_L2,
                                    size_t          element_size_A, //In bits
                                    size_t          element_size_B, //In bits,
                                    size_t          element_size_out, //In bits
                                    size_t          mx_block_size,
                                    bool            debug);

        double estimate_l2_hit(const Hardware& hardware,
                               int             M,
                               int             N,
                               int             K,
                               int             batch,
                               int             MT_M,
                               int             MT_N,
                               int             MT_K,
                               int             WGM,
                               size_t          element_size);
        double estimate_mall_hit(const Hardware& hardware,
                                 int             M,
                                 int             N,
                                 int             K,
                                 int             batch,
                                 int             MT_M,
                                 int             MT_N,
                                 int             MT_K,
                                 int             WGM);

        // Computes the latency per K-complete MT wave.
        // A wave is defined as : The time it takes for one CU to complete one K-complete output tile
        double compute_wave_latency(const Hardware& hardware,
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
                                    double          H_L2,
                                    size_t          element_size_A, //In bits
                                    size_t          element_size_B, //In bits,
                                    size_t          element_size_out, //In bits
                                    size_t          mx_block_size,
                                    bool            debug);

        // Compute the total latency of a gemm based on the latency of one wave multiplied by the number of waves
        // A wave is defined as : The time it takes for one CU to complete one K-complete output tile
        double compute_total_latency(const Hardware& hardware,
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
                                     double          H_L2,
                                     size_t          element_size_A, //In bits
                                     size_t          element_size_B, //In bits,
                                     size_t          element_size_out, //In bits
                                     int             WGM,
                                     size_t          mx_block_size,
                                     bool            debug);

        // Compute the performance from the latency.
        // IMPORTANT : This program is NOT meant to be an analytical model for performance, but rather a way to rank different macro tile sizes.
        // These performance values could be wildly inaccurate in absolute terms, but will often result in the correct ranking of MTin relative terms.
        double compute_perf_gflops(const Hardware& hardware,
                                   size_t          M,
                                   size_t          N,
                                   size_t          K,
                                   size_t          batch,
                                   size_t          MT_M,
                                   size_t          MT_N,
                                   size_t          MT_K,
                                   size_t          MI_M,
                                   size_t          MI_N,
                                   size_t          MI_K,
                                   size_t          element_size_A,
                                   size_t          element_size_B,
                                   size_t          element_size_out,
                                   int             WGM,
                                   double          H_mem1,
                                   bool            debug);

        // Check if MT fits in LDS
        bool check_LDS_capacity(const Hardware& hardware,
                                size_t          MT_M,
                                size_t          MT_N,
                                size_t          MT_K,
                                size_t          element_size,
                                bool            debug);

    } // namespace analytical
} // namespace TensileLite
