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
// #include "Hardware.hpp"
// #include "Reuse.hpp"

namespace TensileLite
{
    namespace analytical
    {
        namespace streamk
        {
            namespace math
            {
                /**
                 * Performs `(n + d - 1) / d`, but is robust against the case where
                 * `(n + d - 1)` would overflow.
                 */
                template <typename N, typename D>
                __device__ __host__ inline constexpr N safe_ceil_div(N n, D d)
                {
                    // Static cast to undo integral promotion.
                    return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
                }
            } // namespace math

            constexpr size_t num_iters_total(size_t output_tiles, size_t iters_per_tile)
            {
                return output_tiles * iters_per_tile;
            }

            constexpr size_t num_iters_per_tile(size_t BLK_K, size_t k)
            {
                return math::safe_ceil_div(k, BLK_K);
            }

            constexpr size_t num_iters_per_cta(size_t iters_total, int g)
            {
                return math::safe_ceil_div(iters_total, g);
            }

            constexpr size_t
                number_of_output_tiles(size_t BLK_M, size_t BLK_N, size_t m, size_t n, size_t batch)
            {
                size_t m_tiles = math::safe_ceil_div(m, BLK_M);
                size_t n_tiles = math::safe_ceil_div(n, BLK_N);
                return m_tiles * n_tiles * batch;
            }

            constexpr size_t num_fixup_peers_v2(size_t g,
                                                size_t iters_total,
                                                size_t iters_per_tile,
                                                size_t iters_per_cta)
            {
                // If tiles don't evenly divide there are always at least 2 fixup peers, and more if iters_per_tile > iters_per_cta
                size_t hasFixup
                    = (iters_total % g == 0 && // Check if some WGs have more iters than others
                       iters_per_cta % iters_per_tile
                           == 0) // Check if WGs have an even number of full tiles
                          ? 0
                          : 1;
                return math::safe_ceil_div(iters_per_tile, iters_per_cta) + hasFixup;
            }

            constexpr size_t num_fixup_peers(size_t iters_per_tile, size_t iters_per_cta)
            {
                return math::safe_ceil_div(iters_per_tile, iters_per_cta);
            }

            std::tuple<double, size_t, size_t> predicted_runtime(size_t BLK_M,
                                                                 size_t BLK_N,
                                                                 size_t BLK_K,
                                                                 size_t m,
                                                                 size_t n,
                                                                 size_t k,
                                                                 size_t batch,
                                                                 int    g,
                                                                 double a,
                                                                 double b,
                                                                 double c,
                                                                 double d)
            {
                size_t output_tiles   = number_of_output_tiles(BLK_M, BLK_N, m, n, batch);
                size_t iters_per_tile = num_iters_per_tile(BLK_K, k);
                size_t iters_total    = num_iters_total(output_tiles, iters_per_tile);
                size_t iters_per_cta  = num_iters_per_cta(iters_total, g);
                size_t fixup_peers    = num_fixup_peers(iters_per_tile, iters_per_cta);

                double runtime
                    = a + (b * (fixup_peers > 1)) + (c * iters_per_cta) + (d * (fixup_peers - 1));

                return std::make_tuple(runtime, iters_per_cta, fixup_peers);
            }

            std::tuple<double, size_t, size_t, double> predicted_runtime_v2(size_t BLK_M,
                                                                            size_t BLK_N,
                                                                            size_t BLK_K,
                                                                            size_t m,
                                                                            size_t n,
                                                                            size_t k,
                                                                            size_t batch,
                                                                            int    g,
                                                                            double a,
                                                                            double b,
                                                                            double c,
                                                                            double d)
            {
                size_t output_tiles   = number_of_output_tiles(BLK_M, BLK_N, m, n, batch);
                size_t iters_per_tile = num_iters_per_tile(BLK_K, k);
                size_t iters_total    = num_iters_total(output_tiles, iters_per_tile);
                size_t iters_per_cta  = num_iters_per_cta(iters_total, g);
                size_t fixup_peers
                    = num_fixup_peers_v2(g, iters_total, iters_per_tile, iters_per_cta);

                size_t remainder_tiles = output_tiles % g;
                double k_split_ratio   = remainder_tiles / static_cast<double>(g);

                double cache_penalty = 0.0;
                if(fixup_peers >= 1)
                {
                    // Calculate the ideal equal split ratio
                    double ideal_split_ratio = 1.0 / fixup_peers;

                    // Measure deviation from the ideal equal split
                    double imbalance = 1 / std::abs(k_split_ratio - ideal_split_ratio);

                    // Scale the penalty by the imbalance and the per-collaborator cost (d)
                    cache_penalty = d * imbalance * fixup_peers;
                }

                // Include the cache penalty in the runtime prediction
                double runtime = a + (b * (fixup_peers > 1)) + (c * iters_per_cta)
                                 + (d * (fixup_peers - 1)) + cache_penalty;

                return std::make_tuple(runtime, iters_per_cta, fixup_peers, cache_penalty);
            }

            int best_predicted_grid_size(size_t BLK_M,
                                         size_t BLK_N,
                                         size_t BLK_K,
                                         size_t m,
                                         size_t n,
                                         size_t k,
                                         size_t batch,
                                         int    grid_start,
                                         int    grid_end,
                                         bool   verbose = false)
            {

                // Fixed overhead alpha (a), fixed-size cost incurred by
                // each work-group, e.g. the grid launch latency, the initial
                // compulsary cache misses, the cost of writing the final output tile
                // to C.
                // double a = 5544 + 9130;
                double a = 2.772 + 4.565; // 5.04 + 8.30;

                // Beta (b) incorporates conditional costs of outputting temporary partial
                // sums for scenarios where the number of output tiles does not quantize
                // perfectly across the number of processors.
                double b = 3.01; // 5.47; 6017;

                // c represents instruction and stall workload of each MAC-iteration.
                double c = 2.2935; // 4.17; 4587;

                // Delta (d) is the cost of reading and accumulating the partial sums from
                // other work-groups covering the same tile.
                double d = 10.22; // 18.59; 20449;

                std::pair<size_t, double> min_grid_runtime;
                std::pair<size_t, double> min_grid_runtime_v2;
                min_grid_runtime.second    = std::numeric_limits<double>::max();
                min_grid_runtime_v2.second = std::numeric_limits<double>::max();

                size_t g = grid_start;

                // Predict the number of CTAs to use between 1 and 304
                for(; g <= static_cast<size_t>(grid_end); ++g)
                {
                    auto [runtime, iters_per_cta, fixup_peers]
                        = predicted_runtime(BLK_M, BLK_N, BLK_K, m, n, k, batch, g, a, b, c, d);

                    auto [runtime_v2, iters_per_cta_v2, fixup_peers_v2, cache_penalty]
                        = predicted_runtime_v2(BLK_M, BLK_N, BLK_K, m, n, k, batch, g, a, b, c, d);

                    if(verbose)
                    {
                        std::cout << "[original] "
                                  << "grid size: " << g << ", runtime: " << runtime
                                  << ", iters_per_cta: " << iters_per_cta << ", fixup_peers: "
                                  << fixup_peers
                                  // << ", cache_penalty: " << cache_penalty
                                  << ", m: " << m << ", n: " << n << ", k: " << k << ", a: " << a
                                  << ", b: " << b << ", c: " << c << ", d: " << d << std::endl;

                        std::cout << "[cache-offset] "
                                  << "grid size: " << g << ", runtime: " << runtime_v2
                                  << ", iters_per_cta: " << iters_per_cta_v2
                                  << ", fixup_peers: " << fixup_peers_v2
                                  << ", cache_penalty: " << cache_penalty << ", m: " << m
                                  << ", n: " << n << ", k: " << k << ", a: " << a << ", b: " << b
                                  << ", c: " << c << ", d: " << d << std::endl;
                    }

                    if(min_grid_runtime.second > runtime)
                    {
                        min_grid_runtime.first  = g;
                        min_grid_runtime.second = runtime;
                    }

                    if(min_grid_runtime_v2.second > runtime_v2)
                    {
                        min_grid_runtime_v2.first  = g;
                        min_grid_runtime_v2.second = runtime_v2;
                    }
                }

                if(verbose)
                {
                    std::cout << "[original] Number of Output Tiles: "
                              << number_of_output_tiles(BLK_M, BLK_N, m, n, batch) << std::endl;
                    std::cout << "[original] Minimum runtime: " << min_grid_runtime.second
                              << " @ grid size: " << min_grid_runtime.first << std::endl;

                    std::cout << "[cache-offset] Number of Output Tiles: "
                              << number_of_output_tiles(BLK_M, BLK_N, m, n, batch) << std::endl;
                    std::cout << "[cache-offset] Minimum runtime: " << min_grid_runtime_v2.second
                              << " @ grid size: " << min_grid_runtime_v2.first << std::endl;
                }

                return min_grid_runtime_v2.first;
            }

        } // namespace streamk
    }
}
