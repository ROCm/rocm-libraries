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

#include <Tensile/analytical/AnalyticalGemm.hpp>
// #include "AnalyticalGemm.hpp"

#include <algorithm>
#include <chrono> // For timing
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <tuple>

namespace TensileLite
{
    namespace analytical
    {

        /**
         * Performs `(n + d - 1) / d`, but is robust against the case where
         * `(n + d - 1)` would overflow.
         */
        template <typename N, typename D>
        constexpr N safe_ceil_div(N n, D d)
        {
            // Static cast to undo integral promotion.
            return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
        }

        // Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
        size_t compute_number_matrix_instructions(const Hardware& hardware,
                                                  size_t          MT_M,
                                                  size_t          MT_N,
                                                  size_t          MT_K,
                                                  size_t          MI_M,
                                                  size_t          MI_N,
                                                  size_t          MI_K,
                                                  bool            debug)
        {
            // Compute the number of Matrix Instructions required in each dim
            size_t N_MI_M = safe_ceil_div(MT_M, MI_M);
            size_t N_MI_N = safe_ceil_div(MT_N, MI_N);
            size_t N_MI_K = safe_ceil_div(MT_K, MI_K);
            // Total number of matrix instructions for MT_MxMT_NxMT_K tile
            size_t N_MI = N_MI_M * N_MI_N * N_MI_K;

            return N_MI;
        }

        double arithmetic_intensity(double m, double n, double k, double bytes_per_element)
        {
            // Numerator: 2.0 * m * n * k
            // Denominator: (m*n + n*k + m*k) * bytes_per_element
            double numerator   = 2.0 * m * n * k;
            double denominator = (m * n + n * k + m * k) * bytes_per_element;
            return numerator / denominator;
        }

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
                                          size_t          element_size_A,
                                          size_t          element_size_B,
                                          bool            debug)
        {

            // Compute the number of matrix instructions
            size_t N_MI = compute_number_matrix_instructions(
                hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, debug);
            // Latency of a single MT_MxMT_NxMT_k tile is the latency of one MI multiplied by number of MI per MT_MxMT_NxMT_k.
            size_t L_MI = hardware.get_MI_latency(
                MI_M, MI_N, MI_K, std::max(element_size_A, element_size_B));

            // size_t mt_arith = arithmetic_intensity(MT_M, MT_N, MT_K, 2);
            // printf("MT_M:%d MT_N:%d MT_K:%d arith:%d\n", MT_M, MT_N, MT_K, mt_arith);
            // size_t arith = ((M * N * K * 2) / (M * K + N * K + M * N));
            size_t L_MT = L_MI * N_MI;

            //TN
            if(transA && !transB)
            {
                //We want to penalize tiles that can't be coalesced for T,N where K is contiguous dimension.
                //In this case, that's when the K dimension is indivisible by 128 bytes.
                if(MT_K * safe_ceil_div(element_size_A, 8) % 128 != 0)
                {
                    L_MT = L_MT * 1.5;
                }
                if(MT_K * safe_ceil_div(element_size_B, 8) % 128 != 0)
                {
                    L_MT = L_MT * 1.5;
                }
            }

            //NT : A is contiguous in M and B is contiguous in N
            if(!transA && transB)
            {

                //LDS Load Granularity is 128 Bytes -> If we load an amount indivisible by 128 bytes in either contiguous
                //dimesion from LDS then we will get poor LDS utilization. This actually happens as more like
                //a quantization effect where if either contiguous dimension of the tile is not evenly divisible by 128-bytes
                //We end up with inefficient loads.
                //Multiplication by a value is arbitrary, there is probably a better analytical method to quantify the true impact of this
                //Effect on the efficiency of computation.
                if((MT_M * safe_ceil_div(element_size_A, 8)) % (128) != 0)
                {
                    L_MT = L_MT * 2;
                }

                if((MT_N * safe_ceil_div(element_size_B, 8)) % 128 != 0)
                {
                    L_MT = L_MT * 2;
                }
            }

            //TT : A is contiguous in K and B is contiguous in N
            if(transA && transB)
            {
                if(MT_K * safe_ceil_div(element_size_A, 8) < 128)
                {
                    L_MT = L_MT * 2;
                }

                if(MT_N * safe_ceil_div(element_size_B, 8) < 128)
                {
                    L_MT = L_MT * 2;
                }
            }

            //NN : A is contiguous in M and B is contiguous in K
            if(!transA && !transB)
            {
                if(MT_M * safe_ceil_div(element_size_A, 8) < 128)
                {
                    L_MT = L_MT * 2;
                }

                if(MT_K * safe_ceil_div(element_size_B, 8) < 128)
                {
                    L_MT = L_MT * 2;
                }
            }

            return L_MT;
        }

        // Computes the number of MT timesteps required to compute all MT. Last wave may be less occupied.
        size_t compute_number_waves(const Hardware& hardware,
                                    size_t          M,
                                    size_t          N,
                                    size_t          batch,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          split,
                                    bool            debug)
        {
            // Compute number of MT_M in M
            size_t num_MT_M = safe_ceil_div(M, MT_M);
            // Number of MT_N in N
            size_t num_MT_N = safe_ceil_div(N, MT_N);
            // Total MT in the output space
            size_t total_MTs = num_MT_M * num_MT_N * batch;
            // Number of waves from hardware and output space (assumes data parallel)
            size_t N_waves = safe_ceil_div(total_MTs, hardware.N_CU);

            return N_waves;
        }

        // Compute the amount of data loaded from A to produce a MT_MxMT_NxMT_K tile.
        size_t compute_A_loads(size_t MT_M, size_t MT_K, bool debug)
        {
            // Compute the size of loads from A for a single MT_MxMT_NxMT_K tiles
            size_t Ld_A_value = MT_M * MT_K;

            return Ld_A_value;
        }
        // Compute the amount of data loaded from B to produce a MT_MxMT_NxMT_K tile.
        size_t compute_B_loads(size_t MT_N, size_t MT_K, bool debug)
        {
            // Compute the size of loads from B for a single MT_MxMT_NxMT_K tiles
            size_t Ld_B_value = MT_N * MT_K;

            return Ld_B_value;
        }

        // Computes total data loads per CU per MT from A and B
        // Reads happen every MT, Writes happen every K-complete tile.
        size_t compute_CU_loads(size_t MT_M, size_t MT_N, size_t MT_K, bool debug)
        {
            // Total loads are loads from A and loads from B
            size_t Ld_A_value = compute_A_loads(MT_M, MT_K, debug);
            size_t Ld_B_value = compute_B_loads(MT_N, MT_K, debug);
            size_t Ld_CU      = Ld_A_value + Ld_B_value;

            return Ld_CU;
        }

        // Computes the number of active compute units if there is only one wave and it is partial
        // Otherwise, returns hardware.N_CU
        size_t compute_active_CU(
            const Hardware& hardware, size_t M, size_t N, size_t batch, size_t MT_M, size_t MT_N)
        {
            size_t num_mt_m        = safe_ceil_div(M, MT_M);
            size_t num_mt_n        = safe_ceil_div(N, MT_N);
            size_t total_output_mt = num_mt_m * num_mt_n * batch;
            // size_t grid_size       = total_output_mt;

            if(total_output_mt > hardware.N_CU)
            {
                return hardware.N_CU; // If we have a full wave, just return the full wave value
            }
            else
            {
                return total_output_mt; // If we don't have a full wave, return the number of MT (Data parallel)
            }
        }

        // limite achievable memory bandwidth based on active CUs
        // Matches the Python logic: bw_limite = active_cu*0.008 for active_cu<100, capped at 1.0.
        double compute_bw_limite_from_occupancy(const Hardware& /*hardware*/, size_t active_cu)
        {
            double bw_limited = 1.0;
            if(active_cu < 100)
            {
                bw_limited = static_cast<double>(active_cu) * 0.008;
            }
            // cap at 1.0
            if(bw_limited > 1.0)
            {
                bw_limited = 1.0;
            }
            return bw_limited;
        }

        double compute_memory_latency(const Hardware& hardware,
                                      size_t          M,
                                      size_t          N,
                                      size_t          K,
                                      bool            transA,
                                      bool            transB,
                                      size_t          batch,
                                      size_t          MT_M,
                                      size_t          MT_N,
                                      size_t          MT_K,
                                      size_t          split,
                                      double          H_mem1,
                                      size_t          element_size_A,
                                      size_t          element_size_B,
                                      size_t          mx_block_size,
                                      bool            debug)
        {

            double H_mem2
                = estimate_mall_hit(hardware, M, N, K, batch, MT_M, MT_N, MT_K, /*WGM=*/1);

            // Total loads are loads from A and loads from B
            size_t Ld_A_value  = compute_A_loads(MT_M, MT_K, debug);
            size_t Ld_B_value  = compute_B_loads(MT_N, MT_K, debug);
            size_t Ld_CU_bytes = (Ld_A_value * safe_ceil_div(element_size_A, 8)) // A Bytes
                                 + (Ld_B_value * safe_ceil_div(element_size_B, 8)); //B Bytes

            /*Logic for block scaled datatypes (Assuming BS=32 and 8-bit scales)*/
            //TODO This is technically wrong, need separate flag to enable MX so we can differentiate FP8 and MX8

            if(element_size_A < 8 && mx_block_size != 0)
            {
                //Number of scales per tile
                size_t num_scales_A = safe_ceil_div(MT_M * MT_K, mx_block_size);
                Ld_CU_bytes += num_scales_A; //One Byte per scale
            }
            if(element_size_B < 8 && mx_block_size != 0)
            {
                //Number of scales per tile
                size_t num_scales_B = safe_ceil_div(MT_M * MT_K, mx_block_size);
                Ld_CU_bytes += num_scales_B; //One Byte per scale
            }
            // 3) occupancy
            size_t active_cu = compute_active_CU(hardware, M, N, batch, MT_M, MT_N) * split;
            active_cu        = std::min(active_cu, hardware.N_CU);
            // 4) total loads by all CUs
            double total_Ld = Ld_CU_bytes * static_cast<double>(active_cu);

            // 5) mem1‐limite factor (simple linear model)
            double mem1_bw_limited = 1;

            mem1_bw_limited = static_cast<double>(active_cu) / static_cast<double>(hardware.N_CU);

            double limited_mem1_bw = hardware.mem1_perf_ratio * mem1_bw_limited;

            // 6) mem1 latency
            double L_mem_mem1 = (limited_mem1_bw > 0) ? (total_Ld / (limited_mem1_bw)) : 0.0;

            // 7) mem2‐limited from occupancy (Can't Issue enough load/stores)
            double bw_limited = compute_bw_limite_from_occupancy(hardware, active_cu);

            // 8) loads that reach each level
            double Ld_mem2 = (1.0 - H_mem1) * total_Ld;
            double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

            // 9) enforce whole‐problem minimum loads
            if(active_cu < hardware.N_CU)
            {
                double min_load
                    = static_cast<double>(M * MT_K * safe_ceil_div(element_size_A, 8)
                                          + N * MT_K * safe_ceil_div(element_size_B, 8));
                Ld_MEM  = std::max(Ld_MEM, min_load) * batch;
                Ld_mem2 = std::max(Ld_mem2, min_load) * batch;
            }

            // 10) mem2 latency
            double L_mem_mem2 = ((hardware.mem2_perf_ratio * bw_limited) > 0)
                                    ? (Ld_mem2 / (hardware.mem2_perf_ratio * bw_limited))
                                    : 0.0;

            // 11) MEM latency
            double limited_mem_bw = hardware.mem3_perf_ratio * bw_limited;
            double L_mem_MEM      = (limited_mem_bw > 0) ? (Ld_MEM / limited_mem_bw) : 0.0;
            L_mem_MEM += 200; //Load Latency

            // 12) pick the worst‐case bound
            double L_mem = std::max({L_mem_mem1, L_mem_mem2, L_mem_MEM});

            //NT
            if(!transA && transB)
            {

                //LDS Load Granularity is 128 Bytes -> If we load an amount indivisible by 128 bytes in either contiguous
                //dimesion from LDS then we will get poor LDS utilization. This actually happens as more like
                //a quantization effect where if either contiguous dimension of the tile is not evenly divisible by 128-bytes
                //We end up with inefficient loads.
                //Multiplication by a value is arbitrary, there is probably a better analytical method to quantify the true impact of this
                //Effect on the efficiency of computation.
                if((MT_M * safe_ceil_div(element_size_A, 8)) % (128) != 0)
                {
                    L_mem = L_mem * 2;
                }

                if((MT_N * safe_ceil_div(element_size_B, 8)) % (128) != 0)
                {
                    L_mem = L_mem * 2;
                }
            }

            //TT : A is contiguous in K and B is contiguous in N
            if(transA && transB)
            {
                if(MT_K * safe_ceil_div(element_size_A, 8) < 128)
                {
                    L_mem = L_mem * 2;
                }

                if(MT_N * safe_ceil_div(element_size_B, 8) < 128)
                {
                    L_mem = L_mem * 2;
                }
            }

            //NN : A is contiguous in M and B is contiguous in K
            if(!transA && !transB)
            {
                if(MT_M * safe_ceil_div(element_size_A, 8) < 128)
                {
                    L_mem = L_mem * 2;
                }

                if(MT_K * safe_ceil_div(element_size_B, 8) < 128)
                {
                    L_mem = L_mem * 2;
                }
            }

            if(debug || Hardware::is_debug_enabled())
            {
                hardware.log_debug("Input M", M);
                hardware.log_debug("Input N", N);
                hardware.log_debug("Input K", K);
                hardware.log_debug("Macro_Tile",
                                   std::to_string(int(MT_M)) + "x" + std::to_string(int(MT_N)) + "x"
                                       + std::to_string(int(MT_K)));
                hardware.log_debug("Element Size A (bits)", element_size_A);
                hardware.log_debug("Element Size B (bits)", element_size_A);

                hardware.log_debug("H_mem1 (mem1 hit ratio)", H_mem1);
                hardware.log_debug("H_mem2 (mem2 hit ratio)", H_mem2);
                hardware.log_debug("Ld_mem1 (bytes)", Ld_mem2);
                hardware.log_debug("Active CUs", active_cu);
                hardware.log_debug("Total Load (bytes)", total_Ld);
                hardware.log_debug("L_mem_mem1 (cycles)", L_mem_mem1);
                hardware.log_debug("Ld_mem2 (bytes)", Ld_mem2);
                hardware.log_debug("L_mem_mem2 (cycles)", L_mem_mem2);
                hardware.log_debug("Ld_MEM (bytes)", Ld_MEM);
                hardware.log_debug("L_mem_MEM (cycles, incl. latency)", L_mem_MEM);
                hardware.log_debug("L_mem (final)", L_mem);
                hardware.log_debug("mem1_perf_ratio", hardware.mem1_perf_ratio);
                hardware.log_debug("mem2_perf_ratio", hardware.mem2_perf_ratio);
                hardware.log_debug("mem3_perf_ratio", hardware.mem3_perf_ratio);
                hardware.log_debug("percent_bw_per_wg", hardware.percent_bw_per_wg);
            }
            return L_mem;
        }

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
                                    double          H_mem1,
                                    size_t          element_size_A,
                                    size_t          element_size_B,
                                    size_t          element_size_out,
                                    size_t          mx_block_size,
                                    bool            debug)
        {
            // 1) Compute per-tile latencies
            double L_compute = compute_mt_compute_latency(hardware,
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
                                                          debug);

            double L_mem = compute_memory_latency(hardware,
                                                  M,
                                                  N,
                                                  K,
                                                  transA,
                                                  transB,
                                                  batch,
                                                  MT_M,
                                                  MT_N,
                                                  MT_K,
                                                  split,
                                                  H_mem1,
                                                  element_size_A,
                                                  element_size_B,
                                                  mx_block_size,
                                                  debug);

            // 2) Work-group setup & iteration latencies
            double L_WG_setup = 1; //WG_setup_Latency

            // 3) Prologue: 2.2× memory latency
            double L_prologue = 1.5 * L_mem;

            // 4) Epilogue: writes from all active CUs with limited bandwidth
            size_t active_cu       = compute_active_CU(hardware, M, N, batch, MT_M, MT_N);
            double epilogue_limite = 1;

            epilogue_limite = (static_cast<double>(active_cu) / static_cast<double>(hardware.N_CU));

            double limited_mem1 = hardware.mem1_perf_ratio * epilogue_limite;
            if(limited_mem1 < 1)
            {
                limited_mem1 = 10;
            }

            double L_epilogue = (static_cast<double>(active_cu) * MT_M * MT_N
                                 * safe_ceil_div(element_size_out, 8))
                                / limited_mem1;

            //K-Split reductions are globally coherent, we need to write and read split-1 MT_M*MT_N tiles to coherent memory
            if(split > 1)
            {
                size_t n_partials              = split - 1;
                double partial_readwrite_bytes = (2 * active_cu * safe_ceil_div(element_size_out, 8)
                                                  * MT_M * MT_N * n_partials);
                double L_reduce = partial_readwrite_bytes / (hardware.mem3_perf_ratio);
                L_epilogue += L_reduce * 1;
            }

            // 5) Single-tile latency (always additive)
            double L_tile_single = std::max(L_compute, L_mem);

            // 6) Number of K-iterations (excluding epilogue), at least 1
            long num_iter = static_cast<long>(((K + MT_K - 1) / MT_K)) - 1;
            num_iter      = std::ceil(num_iter / split);
            num_iter      = std::max(num_iter, 1L);

            // 7) Total tile latency
            double L_tile_total = (L_tile_single * num_iter) + L_prologue + L_epilogue + L_WG_setup
                                  + (28 * num_iter); //Iteration branch latency

            if(MT_K == 512)
            {
                L_tile_total *= 1.5;
            }

            if(debug || Hardware::is_debug_enabled())
            {
                hardware.log_debug("Problem_Size",
                                   std::to_string(int(M)) + "x" + std::to_string(int(N)) + "x"
                                       + std::to_string(int(K)));
                hardware.log_debug("Macro_Tile",
                                   std::to_string(int(MT_M)) + "x" + std::to_string(int(MT_N)) + "x"
                                       + std::to_string(int(MT_K)));
                hardware.log_debug("L_compute", L_compute);
                hardware.log_debug("L_mem", L_mem);
                hardware.log_debug("L_prologue", L_prologue);
                hardware.log_debug("L_epilogue", L_epilogue);
                hardware.log_debug("num_iter", num_iter);
            }

            return L_tile_total;
        }

        double estimate_l2_hit(const Hardware& hardware,
                               int             M,
                               int             N,
                               int             K,
                               int             batch,
                               int             MT_M,
                               int             MT_N,
                               int             MT_K,
                               int             WGM,
                               size_t          element_size)
        {
            // Compute grid dimensions
            int grid_m = static_cast<int>(safe_ceil_div(M, MT_M));
            int grid_n = static_cast<int>(safe_ceil_div(N, MT_N));
            int grid_k = static_cast<int>(safe_ceil_div(K, MT_K));

            WGM = std::max(WGM, 1); // WGM can't be less than one.

            // Get number of active CUs
            int num_cus = compute_active_CU(hardware, M, N, batch, MT_M, MT_N);

            // Distribute CUs per XCD. Ensure at least 1.
            int cu_per_xcd = std::max(safe_ceil_div(num_cus, hardware.NUM_XCD), 1);

            // N dimension of mem1 tile is divided by whichever is smaller between WGM and grid
            int l2_n = cu_per_xcd / std::min(WGM, grid_m);
            int l2_m = std::min(WGM, grid_m); // M dimension of mem1 tile

            // If a single mem1 tile is larger than the grid, extend M dimension
            if(l2_n > grid_n)
            {
                int wrap_amount = grid_n - l2_n;
                int num_wraps   = (l2_n / grid_n) - 1; // how many times we wrap
                l2_m += (num_wraps * WGM);
                l2_n = grid_n;
            }

            // Clamp mem1 tile dimensions to at least 1 and at most grid size
            l2_m = std::max(std::min(grid_m, l2_m), 1);
            l2_n = std::max(std::min(grid_n, l2_n), 1);

            // Compute "uncached" reads based on mem1 tile dimensions
            long long l2_A_uncached_reads = static_cast<long long>(l2_m) * MT_M * MT_K;
            long long l2_B_uncached_reads = static_cast<long long>(l2_n) * MT_N * MT_K;
            long long uncached_read       = l2_A_uncached_reads + l2_B_uncached_reads;

            // If bigger than cache capacity, reduce mem1 tile size and recompute uncached reads
            while(l2_A_uncached_reads + l2_B_uncached_reads
                  > hardware.L2_capacity / safe_ceil_div(element_size, 8))
            {
                // Reduce M dimension by 1
                l2_m -= 1;
                if(l2_m < 1)
                {
                    // We cannot shrink any more without going to zero or negative
                    l2_m = 1;
                    break;
                }
                l2_A_uncached_reads = static_cast<long long>(l2_m) * MT_M * MT_K;
                l2_B_uncached_reads = static_cast<long long>(l2_n) * MT_N * MT_K;
            }

            // Total reads considering repeated usage
            long long l2_A_reads = static_cast<long long>(l2_m) * l2_n * MT_M * MT_K;
            long long l2_B_reads = static_cast<long long>(l2_n) * l2_m * MT_N * MT_K;

            long long total_reads         = std::max(l2_A_reads + l2_B_reads, 1LL);
            long long total_uncached_read = l2_A_uncached_reads + l2_B_uncached_reads;
            long long cached_reads        = total_reads - total_uncached_read;

            double l2_hit = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

            // Guard against numeric anomalies
            if(l2_hit > 1.0)
            {
                std::cerr << "mem1 hit was greater than 1, which isn't possible.\n"
                          << "Problem Size: " << M << "x" << N << "x" << K << "\n"
                          << "Macro-Tile:  " << MT_M << "x" << MT_N << "x" << MT_K << "\n"
                          << "cu_per_xcd:  " << cu_per_xcd << "\n"
                          << "l2_m: " << l2_m << ", l2_n: " << l2_n << ", l2_hit: " << l2_hit
                          << "\n";
            }

            return l2_hit;
        }

        double estimate_mall_hit(const Hardware& hardware,
                                 int             M,
                                 int             N,
                                 int             K,
                                 int             batch,
                                 int             MT_M,
                                 int             MT_N,
                                 int             MT_K,
                                 int             WGM)
        {
            int grid_m = static_cast<int>(std::ceil(static_cast<double>(M) / MT_M));
            int grid_n = static_cast<int>(std::ceil(static_cast<double>(N) / MT_N));
            int grid_k = static_cast<int>(std::ceil(static_cast<double>(K) / MT_K));

            int num_cus = compute_active_CU(hardware, M, N, batch, MT_M, MT_N);

            // If the 2D grid is smaller than the number of CUs, not all will be active
            if((grid_m * grid_n * batch) < num_cus)
            {
                num_cus = (grid_m * grid_n * batch) / hardware.NUM_XCD;
            }

            // mem2 tile dimensions
            int mall_n = num_cus / WGM; // N dimension of mem2 tile
            int mall_m = std::min(WGM, grid_m); // M dimension of mem2 tile

            // If a single mem2 tile is larger than the grid, extend its M dimension
            if(mall_n > grid_n)
            {
                int wrap_amount = grid_n - mall_n;
                int num_wraps   = (mall_n / grid_n) - 1;
                mall_m += (num_wraps * WGM);
                mall_n = grid_n;
            }

            // Clamp the tile dimensions to valid ranges
            mall_m = std::max(std::min(grid_m, mall_m), 1);
            mall_n = std::max(std::min(grid_n, mall_n), 1);

            // Unique “uncached” entries of A/B for this XCD
            int mall_A_uncached_reads = mall_m * MT_M * MT_K;
            int mall_B_uncached_reads = mall_n * MT_N * MT_K;
            int total_uncached_read   = mall_A_uncached_reads + mall_B_uncached_reads;

            // Total A/B reads considering repeated usage
            long long mall_A_reads = static_cast<long long>(mall_m) * mall_n * MT_M * MT_K;
            long long mall_B_reads = static_cast<long long>(mall_n) * mall_m * MT_N * MT_K;

            // Avoid division by zero
            long long total_reads  = std::max(mall_A_reads + mall_B_reads, 1LL);
            long long cached_reads = total_reads - total_uncached_read;

            double mall_hit = static_cast<double>(cached_reads) / static_cast<double>(total_reads);
            return mall_hit;
        }

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
                                    double          H_mem1,
                                    size_t          element_size_A,
                                    size_t          element_size_B,
                                    size_t          element_size_out,
                                    size_t          mx_block_size,
                                    bool            debug)
        {
            // Assume latency of a wave is latency of a single k-complete output tile.

            double L_wave = compute_tile_latency(hardware,
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
                                                 H_mem1,
                                                 element_size_A,
                                                 element_size_B,
                                                 element_size_out,
                                                 mx_block_size,
                                                 debug);

            return L_wave;
        }

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
                                     double          H_mem1,
                                     size_t          element_size_A,
                                     size_t          element_size_B,
                                     size_t          element_size_out,
                                     int             WGM,
                                     size_t          mx_block_size,
                                     bool            debug)
        {

            //std::cout << "Split " << split << "\n";
            H_mem1
                = estimate_l2_hit(hardware, M, N, K, batch, MT_M, MT_N, MT_K, WGM, element_size_A);
            //Compute Prologue latency
            // double prologue_latency
            //     = compute_memory_latency(hardware, M, N, K, MT_M, MT_N, MT_K, H_mem1, debug);
            // Compute number of waves
            size_t N_waves = compute_number_waves(hardware, M, N, batch, MT_M, MT_N, split, debug);
            // Compute latency of a wave
            double L_wave = compute_wave_latency(hardware,
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
                                                 H_mem1,
                                                 element_size_A,
                                                 element_size_B,
                                                 element_size_out,
                                                 mx_block_size,
                                                 debug);
            // Compute latency for all waves and return it as the latency for the MT/problem
            double total_latency = L_wave * N_waves;
            //total_latency        = total_latency + prologue_latency;
            if(Hardware::is_debug_enabled())
            {
                hardware.print_debug_info();
            }
            return total_latency;
        }

        // Compute the performance from the latency.
        // IMPORTANT : This program is NOT meant to be an analytical model for performance, but rather a way to rank different macro tile sizes.
        // These performance values could be wildly inaccurate in absolute terms, but will often result in the correct ranking of MTin relative terms.
        double compute_perf_gflops(const Hardware& hardware,
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
                                   int             WGM,
                                   double          H_mem1,
                                   bool            debug)
        {
            // Compute total FLOPs
            double total_FLOPs = 2.0 * M * N * K; // For GEMM, each multiply-add is 2 FLOPs
            // Compute total time in seconds
            double cycles_per_second
                = hardware.compute_clock_ghz * 1e9; // 1 GHz = 1e9 cycles per second
            size_t mx_block_size      = 0;
            double latency_cycles     = compute_total_latency(hardware,
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
                                                          1,
                                                          H_mem1,
                                                          element_size_A,
                                                          element_size_B,
                                                          element_size_out,
                                                          WGM,
                                                          mx_block_size,
                                                          debug);
            double total_time_seconds = latency_cycles / cycles_per_second;
            // Compute performance in FLOPS
            double FLOPS = total_FLOPs / total_time_seconds;
            // Convert to TFLOPS
            double GFLOPS = FLOPS / 1e9; // 1 TFLOP = 1e9 FLOPs
            return GFLOPS;
        }

        // Check if MT fits in LDS
        bool check_LDS_capacity(const Hardware& hardware,
                                size_t          MT_M,
                                size_t          MT_N,
                                size_t          MT_K,
                                size_t          element_size,
                                bool            debug)
        {
            // A and B size
            size_t Ld_A_value = compute_A_loads(MT_M, MT_K, debug);
            size_t Ld_B_value = compute_B_loads(MT_N, MT_K, debug);
            // Size of those in bytes
            size_t LDS_usage = (Ld_A_value + Ld_B_value) * (element_size / 8);

            if(LDS_usage > hardware.LDS_capacity)
            {
                return false; // Exceeds LDS capacity
            }
            else
            {
                return true; // Within LDS capacity
            }
        }

    } // namespace analytical
} // namespace TensileLite
