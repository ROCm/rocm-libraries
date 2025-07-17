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

#include <Tensile/analytical/Utils.hpp>
// #include "Utils.hpp"

#include <algorithm>
#include <chrono> // For timing
#include <cmath>
#include <iomanip> // For output formatting
#include <iostream>

namespace TensileLite
{
    namespace analytical
    {

        //
        // Tiebreaker function.
        //
        void pick_best_tile_by_arithmetic_intensity(std::vector<ResultTuple>& top_results,
                                                    size_t                    num_to_sort)
        {
            if(top_results.empty())
            {
                throw std::runtime_error(
                    "pick_best_tile_by_arithmetic_intensity received empty list.");
            }

            // 1) Define a helper function to compute the arithmetic intensity of a tile.
            //    Here we assume:
            //    - Flops for tile (MT_M, MT_N, MT_K) is: 2 * MT_M * MT_N * MT_K
            //    - Memory traffic approximated as: MT_M*MT_K + MT_K*MT_N + MT_M*MT_N
            //    - Arithmetic intensity = flops / memory_traffic
            auto computeArithmeticIntensity = [](const ResultTuple& t) -> double {
                // The tuple is: (latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K)
                auto MT_M = std::get<1>(t);
                auto MT_N = std::get<2>(t);
                auto MT_K = std::get<3>(t);

                double flops = static_cast<double>(2ull * MT_M * MT_N * MT_K);
                double memory_traffic
                    = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

                // Avoid division by zero.
                if(memory_traffic == 0.0)
                    return 0.0;

                return flops / memory_traffic;
            };
            // 2) Sort the results in descending order of arithmetic intensity
            //    (highest arithmetic intensity first).
            std::sort(top_results.begin(),
                      top_results.begin() + num_to_sort,
                      [&](const ResultTuple& a, const ResultTuple& b) {
                          double ai_a = computeArithmeticIntensity(a);
                          double ai_b = computeArithmeticIntensity(b);
                          return ai_a > ai_b; // descending
                      });
            // 3) Return the tile with the highest arithmetic intensity
        }

        ResultTuple pick_best_tile_with_dimension_priority(
            const std::vector<ResultTuple>& top_results, size_t M, size_t N, size_t K)
        {
            if(top_results.empty())
            {
                throw std::runtime_error(
                    "pick_best_tile_with_dimension_priority received empty list.");
            }

            // 1) Determine whether M or N is more important
            //    (based on which is larger), and always place K last.
            //    This yields a priority order of either { 'M', 'N', 'K' }
            //    or { 'N', 'M', 'K' }.
            std::vector<char> dimPriority;
            if(M >= N)
                dimPriority = {'M', 'N', 'K'};
            else
                dimPriority = {'N', 'M', 'K'};

            // 2) Helper function to extract the tile dimension:
            //    (latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K)
            auto getTileSize = [](const ResultTuple& t, char dimChar) -> size_t {
                switch(dimChar)
                {
                case 'M':
                    return std::get<1>(t); // MT_M
                case 'N':
                    return std::get<2>(t); // MT_N
                case 'K':
                    return std::get<3>(t); // MT_K
                default:
                    return 0;
                }
            };

            // 3) Sort in descending order according to the dimension priority.
            //    - Compare dimensionPriority[0] first
            //    - If there's a tie, compare dimensionPriority[1]
            //    - If still a tie, compare dimensionPriority[2]
            //    - If they're all equal, consider them tied
            std::vector<ResultTuple> sorted = top_results; // copy
            std::sort(
                sorted.begin(), sorted.end(), [&](const ResultTuple& a, const ResultTuple& b) {
                    for(char d : dimPriority)
                    {
                        size_t ta = getTileSize(a, d);
                        size_t tb = getTileSize(b, d);
                        if(ta > tb)
                            return true;
                        if(ta < tb)
                            return false;
                    }
                    // If all relevant dimensions are the same, treat as a tie
                    return false;
                });

            // 4) Return the best tile (the first after sorting).
            return sorted.front();
        }

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
                                     size_t          biggest_allowable_split)
        {

            // compute how many 32×32 tiles are needed in each dim,
            // then multiply to get total grid size:
            size_t grid = ((M + MT_M - 1) / MT_M) * ((N + MT_N - 1) / MT_N) * batch;

            size_t max_hw_split = std::floor(hardware.N_CU / grid);
            size_t MAX_SPLIT    = std::min(biggest_allowable_split, max_hw_split);

            size_t best_split   = 1;
            double best_latency = std::numeric_limits<double>::infinity();

            for(size_t split = 1; split <= MAX_SPLIT; ++split)
            {
                double latency = compute_total_latency(hardware,
                                                       M,
                                                       N,
                                                       K, // problem dims
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
                                                       H_L2,
                                                       element_size_A, //ElementSizeA
                                                       element_size_B, //ElementSizeB
                                                       element_size_out, //ElementSizeout
                                                       WGM,
                                                       mx_block_size,
                                                       debug);

                if(latency < best_latency)
                {
                    best_latency = latency;
                    best_split   = split;
                }
            }
            size_t best_grid = best_split * grid;

            // you now have both `grid` and `best_split`—
            // return whichever is appropriate (here we stick with split):
            return best_grid;
        }

        std::vector<ResultTuple> select_best_macro_tile_size(size_t                        M,
                                                             size_t                        N,
                                                             size_t                        K,
                                                             size_t                        batch,
                                                             bool                          transA,
                                                             bool                          transB,
                                                             const Hardware&               hardware,
                                                             const std::vector<TileTuple>& MT_list,
                                                             size_t element_size_A, //In bits
                                                             size_t element_size_B, //In bits
                                                             size_t element_size_out, //In bits
                                                             size_t mx_block_size,
                                                             double H_L2,
                                                             bool   debug,
                                                             bool   print,
                                                             size_t WGM)
        {
            std::vector<ResultTuple> valid_results;
            valid_results.reserve(MT_list.size());

            for(const auto& mt : MT_list)
            {
                size_t MT_M = std::get<0>(mt);
                size_t MT_N = std::get<1>(mt);
                size_t MT_K = std::get<2>(mt);
                size_t MI_M = std::get<3>(mt);
                size_t MI_N = std::get<4>(mt);
                size_t MI_K = std::get<5>(mt);
                size_t occupancy = std::get<6>(mt);

                if(debug)
                {
                    std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N
                              << ", MT_K=" << MT_K << ", MI_M=" << MI_M << ", MI_N=" << MI_N
                              << ", MI_K=" << MI_K << "\n";
                }

                size_t split = 1;
                if(check_LDS_capacity(hardware, MT_M, MT_N, MT_K, element_size_A, debug))
                {
                    double Total_latency = compute_total_latency(hardware,
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
                                                                 H_L2,
                                                                 element_size_A,
                                                                 element_size_B,
                                                                 element_size_out,
                                                                 WGM,
                                                                 mx_block_size,
                                                                 debug);

                    valid_results.emplace_back(Total_latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, occupancy);
                }
                else if(debug)
                {
                    std::cout << "Skipping MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                              << " due to LDS capacity\n";
                }
            }

            if(valid_results.empty())
            {
                throw std::runtime_error("No valid macro-tile sizes found.");
            }

            // 1) Sort results by ascending latency.
            std::sort(valid_results.begin(), valid_results.end(), [](auto const& a, auto const& b) {
                return std::get<0>(a) < std::get<0>(b);
            });

            // 2) Collect results that tie for the absolute best latency.
            double best_latency = std::get<0>(valid_results.front());
            size_t num_the_same = 0;
            for(const auto& res : valid_results)
            {
                // If it's "essentially" the same as best_latency, include it
                if(std::fabs(std::get<0>(res) - best_latency) < 10)
                    num_the_same++;
                else
                    break; // Once we pass best_latency, we can stop.
            }
            // 3) If that tie group has at least 10 entries, we only use those.
            // 4) Otherwise, keep adding the next best latencies until we have 10 total or run out.
            // std::vector<ResultTuple> top_candidates = tie_results;
            // if(tie_results.size() < 10)
            // {
            //     size_t i = tie_results.size();
            //     while(top_candidates.size() < 10 && i < valid_results.size())
            //     {
            //         top_candidates.push_back(valid_results[i]);
            //         i++;
            //     }
            // }
            // Now ‘top_candidates’ is either all the tied best-latency results (if >=10),
            // or the top 10 latencies overall (including however many best-latency entries there were).

            // Finally, use your existing tie-breaker on top_candidates
            pick_best_tile_by_arithmetic_intensity(valid_results, num_the_same);
            if(print)
            {
                for(const auto& tile : valid_results)
                {
                    std::cout << M << "x" << N << "x" << K
                              << "Selected Macro-Tile: Latency=" << std::get<0>(tile)
                              << ", MT_M=" << std::get<1>(tile) << ", MT_N=" << std::get<2>(tile)
                              << ", MT_K=" << std::get<3>(tile) << ", MI_M=" << std::get<4>(tile)
                              << ", MI_N=" << std::get<5>(tile) << ", MI_K=" << std::get<6>(tile)
                              << "\n";
                }
            }

            return valid_results;
        }

        /*!
         * \brief Selects the best WGM (maximizing L2 hit rate) given fixed macro tile sizes.
         *
         * \param[in] M, N, K    - your overall problem sizes
         * \param[in] hardware   - a struct describing your hardware capabilities
         * \param[in] MT_M,MT_N,MT_K,MI_M,MI_N,MI_K - chosen macro/MI tile sizes
         * \param[in] WGM_list   - candidate WGM values to try
         * \param[in] element_size
         * \param[in] H_L2       - some hardware-related constant or factor (no longer used here,
         *                         but kept if your signature or other usage requires it)
         * \param[in] debug      - whether to print debug messages
         * \param[in] print      - whether to print the final best result
         *
         * \return A pair: (best_l2_hit_rate, best_WGM).
         */
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
            bool   print)
        {
            using WGMResult = std::pair<double, size_t>; // (l2_hit_rate, WGM)

            std::vector<WGMResult> valid_results;
            valid_results.reserve(WGM_list.size());

            // Iterate over all candidate WGM values
            for(const auto& candidate_wgm : WGM_list)
            {
                if(debug)
                {
                    std::cout << "Evaluating WGM=" << candidate_wgm << "\n";
                }

                // Optionally ensure we do not exceed LDS capacity
                // (If you want to factor in WGM, add it to your check_LDS_capacity signature.)
                // For now, let's just check the tile itself:
                if(!check_LDS_capacity(hardware, MT_M, MT_N, MT_K, element_size, debug))
                {
                    if(debug)
                    {
                        std::cout << "Skipping WGM=" << candidate_wgm << " due to LDS capacity.\n";
                    }
                    continue;
                }

                // Compute L2 hit rate for this WGM
                double current_hit = estimate_l2_hit(hardware,
                                                     static_cast<int>(M),
                                                     static_cast<int>(N),
                                                     static_cast<int>(K),
                                                     static_cast<int>(batch),
                                                     static_cast<int>(MT_M),
                                                     static_cast<int>(MT_N),
                                                     static_cast<int>(MT_K),
                                                     static_cast<int>(candidate_wgm),
                                                     element_size);

                valid_results.emplace_back(current_hit, candidate_wgm);
            }

            // If no valid WGM was found, throw an error
            if(valid_results.empty())
            {
                throw std::runtime_error("No valid WGM found.");
            }

            // Find the maximum L2 hit rate in valid_results
            // (Use max_element on the first value in the pair.)
            auto best_it = std::max_element(
                valid_results.begin(),
                valid_results.end(),
                [](const WGMResult& a, const WGMResult& b) {
                    return a.first < b.first; // "less" => a has smaller hit rate than b
                });

            double best_l2_hit = best_it->first;
            size_t best_wgm    = best_it->second;

            // Return (l2_hit_rate, WGM)
            return std::make_pair(best_l2_hit, best_wgm);
        }

        // Logic to decide between two MT that are "tied"
        std::vector<std::tuple<double, size_t, size_t, size_t>> tie_breaker_macro_tile_sizes(
            const std::vector<std::tuple<double, size_t, size_t, size_t>>& top_results,
            size_t                                                         M,
            size_t                                                         N,
            size_t                                                         K,
            Hardware&                                                      hardware,
            std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, Hardware&)>
                 tie_breaker_fn,
            bool debug)
        {
            std::vector<std::tuple<double, size_t, size_t, size_t>> tie_breaker_results;

            for(const auto& res : top_results)
            {
                size_t MT_M = std::get<1>(res);
                size_t MT_N = std::get<2>(res);
                size_t MT_K = std::get<3>(res);

                // Call user-provided tie-breaking function
                double precise_latency = tie_breaker_fn(M, N, K, MT_M, MT_N, MT_K, hardware);

                tie_breaker_results.emplace_back(precise_latency, MT_M, MT_N, MT_K);
            }

            // Sort results by precise_latency (ascending order)
            std::sort(tie_breaker_results.begin(), tie_breaker_results.end());

            return tie_breaker_results;
        }

        std::vector<std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t>>
            rank_macro_tile_sizes(
                size_t                        M,
                size_t                        N,
                size_t                        K,
                bool                          transA,
                bool                          transB,
                Hardware&                     hardware,
                const std::vector<TileTuple>& MT_list,
                size_t                        element_size,
                double                        H_L2,
                bool                          debug,
                bool                          print,
                size_t                        WGM,
                std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, Hardware&)>
                    tie_breaker_fn)
        {
            std::vector<std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t>> results;

            typedef std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t> ResultTuple;

            for(size_t i = 0; i < MT_list.size(); ++i)
            {
                size_t MT_M = std::get<0>(MT_list[i]);
                size_t MT_N = std::get<1>(MT_list[i]);
                size_t MT_K = std::get<2>(MT_list[i]);
                size_t MI_M = std::get<3>(MT_list[i]);
                size_t MI_N = std::get<4>(MT_list[i]);
                size_t MI_K = std::get<5>(MT_list[i]);

                if(debug)
                {
                    std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N
                              << ", MT_K=" << MT_K << ", MI_M=" << MI_M << ", MI_N=" << MI_N
                              << ", MI_K=" << MI_K << "\n";
                }

                if(check_LDS_capacity(hardware, MT_M, MT_N, MT_K, element_size, debug))
                {
                    size_t split         = 1;
                    size_t mx_block_size = 0;
                    double Total_latency
                        = compute_total_latency(hardware,
                                                M,
                                                N,
                                                K,
                                                1, //Batch
                                                transA,
                                                transB,
                                                MT_M,
                                                MT_N,
                                                MT_K,
                                                MI_M,
                                                MI_N,
                                                MI_K,
                                                split, //Split
                                                H_L2, //H_mem1
                                                element_size * 8, //Element Size A
                                                element_size * 8, //Element Size B
                                                element_size * 8, //Element Size out
                                                WGM, //WGM
                                                mx_block_size, //mx_block_size
                                                debug); //debug

                    results.push_back(
                        std::make_tuple(Total_latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K));
                }
                else if(debug)
                {
                    std::cout << "Skipping MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                              << " due to LDS capacity\n";
                }
            }

            // Sort results by Total_latency, from worst (largest latency) to best (smallest latency)
            std::sort(
                results.begin(), results.end(), [](const ResultTuple& a, const ResultTuple& b) {
                    return std::get<0>(a) > std::get<0>(b);
                });

            if(!results.empty())
            {
                double best_latency = std::get<0>(results.back());

                std::vector<ResultTuple> top_results;
                for(size_t i = 0; i < results.size(); ++i)
                {
                    if(std::abs(std::get<0>(results[i]) - best_latency) < 1e-6)
                    {
                        top_results.push_back(results[i]);
                    }
                }

                if(top_results.size() > 1)
                {
                    if(debug)
                    {
                        std::cout << "Tie detected among top-ranked tile sizes. Applying "
                                     "tie-breaker...\n";
                    }

                    // Compute tie-breaker scores and store them along with the result indices
                    std::vector<std::pair<double, size_t>>
                        tie_breaker_scores; // (score, index in top_results)

                    for(size_t i = 0; i < top_results.size(); ++i)
                    {
                        const ResultTuple& res  = top_results[i];
                        size_t             MT_M = std::get<1>(res);
                        size_t             MT_N = std::get<2>(res);
                        size_t             MT_K = std::get<3>(res);
                        size_t             MI_M = std::get<4>(res);
                        size_t             MI_N = std::get<5>(res);
                        size_t             MI_K = std::get<6>(res);
                        double score = tie_breaker_fn(MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, hardware);

                        tie_breaker_scores.push_back(std::make_pair(score, i));
                    }

                    // Now sort the tie_breaker_scores based on score
                    std::sort(tie_breaker_scores.begin(),
                              tie_breaker_scores.end(),
                              [](const std::pair<double, size_t>& a,
                                 const std::pair<double, size_t>& b) { return a.first > b.first; });

                    // Now re-order 'top_results' based on sorted indices
                    std::vector<ResultTuple> sorted_top_results;
                    for(size_t i = 0; i < tie_breaker_scores.size(); ++i)
                    {
                        size_t idx = tie_breaker_scores[i].second;
                        sorted_top_results.push_back(top_results[idx]);
                    }

                    // Remove the tied results from 'results' and insert the sorted 'sorted_top_results'
                    results.erase(
                        std::remove_if(results.begin(),
                                       results.end(),
                                       [best_latency](const ResultTuple& res) {
                                           return std::abs(std::get<0>(res) - best_latency) < 1e-6;
                                       }),
                        results.end());

                    results.insert(
                        results.end(), sorted_top_results.begin(), sorted_top_results.end());
                    // No need to re-sort results as total_latency remains same for tied results
                }
            }

            if(print)
            {
                std::cout << "Total Latency\tMT_M\tMT_N\tMT_K\tMI_M\tMI_N\tMI_K\n";
                for(size_t i = 0; i < results.size(); ++i)
                {
                    double latency = std::get<0>(results[i]);
                    size_t MT_M    = std::get<1>(results[i]);
                    size_t MT_N    = std::get<2>(results[i]);
                    size_t MT_K    = std::get<3>(results[i]);
                    size_t MI_M    = std::get<4>(results[i]);
                    size_t MI_N    = std::get<5>(results[i]);
                    size_t MI_K    = std::get<6>(results[i]);
                    std::cout << std::fixed << std::setprecision(2) << latency << "\t" << MT_M
                              << "\t" << MT_N << "\t" << MT_K << "\t" << MI_M << "\t" << MI_N
                              << "\t" << MI_K << "\n";
                }
            }

            return results;
        }

        double compute_TFLOPS_from_latency(
            double latency_cycles, size_t M, size_t N, size_t K, double clock_GHz, bool debug)
        {
            // Compute total FLOPs
            double total_FLOPs = 2.0 * M * N * K; // For GEMM, each multiply-add is 2 FLOPs
            // Compute total time in seconds
            double cycles_per_second  = clock_GHz * 1e9; // 1 GHz = 1e9 cycles per second
            double total_time_seconds = latency_cycles / cycles_per_second;
            // Compute performance in FLOPS
            double FLOPS = total_FLOPs / total_time_seconds;
            // Convert to TFLOPS
            double TFLOPS = FLOPS / 1e12; // 1 TFLOP = 1e12 FLOPs

            if(debug)
            {
                std::cout << "Total FLOPs: " << total_FLOPs << "\n";
                std::cout << "Total Time: " << total_time_seconds << " seconds\n";
                std::cout << "Performance: " << FLOPS << " FLOPS\n";
                std::cout << "Achieved Performance: " << TFLOPS << " TFLOPS\n";
            }

            return TFLOPS;
        }

    } // namespace analytical
} // namespace TensileLite
