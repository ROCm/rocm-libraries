// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/utils.hpp"

#include <algorithm>
#include <chrono> // For timing
#include <cmath>
#include <iomanip> // For output formatting
#include <iostream>

namespace origami
{
    //
    // Tiebreaker function.
    //
    void pick_best_tile_by_arithmetic_intensity(std::vector<result_tuple>& top_results,
                                                size_t                     num_to_sort)
    {
        if(top_results.empty())
        {
            throw std::runtime_error("pick_best_tile_by_arithmetic_intensity received empty list.");
        }

        // 1) Define a helper function to compute the arithmetic intensity of a tile.
        //    Here we assume:
        //    - Flops for tile (MT_M, MT_N, MT_K) is: 2 * MT_M * MT_N * MT_K
        //    - Memory traffic approximated as: MT_M*MT_K + MT_K*MT_N + MT_M*MT_N
        //    - Arithmetic intensity = flops / memory_traffic
        auto compute_arithmetic_intensity = [](const result_tuple& t) -> double {
            // The tuple is: (latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K)
            auto MT_M = std::get<1>(t);
            auto MT_N = std::get<2>(t);
            auto MT_K = std::get<3>(t);

            double flops          = static_cast<double>(2ull * MT_M * MT_N * MT_K);
            double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

            // Avoid division by zero.
            if(memory_traffic == 0.0)
                return 0.0;

            return flops / memory_traffic;
        };
        // 2) Sort the results in descending order of arithmetic intensity
        //    (highest arithmetic intensity first).
        std::stable_sort(top_results.begin(),
                         top_results.begin() + num_to_sort,
                         [&](const result_tuple& a, const result_tuple& b) {
                             double ai_a = compute_arithmetic_intensity(a);
                             double ai_b = compute_arithmetic_intensity(b);
                             return ai_a > ai_b; // descending
                         });
        // 3) Return the tile with the highest arithmetic intensity
    }

    result_tuple pick_best_tile_with_dimension_priority(
        const std::vector<result_tuple>& top_results, size_t M, size_t N, size_t K)
    {
        if(top_results.empty())
        {
            throw std::runtime_error("pick_best_tile_with_dimension_priority received empty list.");
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
        auto getTileSize = [](const result_tuple& t, char dimChar) -> size_t {
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
        std::vector<result_tuple> sorted = top_results; // copy
        std::stable_sort(
            sorted.begin(), sorted.end(), [&](const result_tuple& a, const result_tuple& b) {
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

    size_t select_best_grid_size(size_t            M,
                                 size_t            N,
                                 size_t            K,
                                 size_t            batch,
                                 bool              transA,
                                 bool              transB,
                                 const hardware_t& hardware,
                                 size_t            MT_M,
                                 size_t            MT_N,
                                 size_t            MT_K,
                                 size_t            MI_M,
                                 size_t            MI_N,
                                 size_t            MI_K,
                                 size_t            element_size_A,
                                 size_t            element_size_B,
                                 size_t            element_size_out,
                                 data_type_t       mi_datatype,
                                 size_t            mx_block_size,
                                 double            H_L2,
                                 size_t            WGM,
                                 size_t            biggest_allowable_split,
                                 size_t            max_cus)
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
                                                   element_size_A, //ElementSizeA
                                                   element_size_B, //ElementSizeB
                                                   element_size_out, //ElementSizeout
                                                   mi_datatype,
                                                   mx_block_size,
                                                   WGM,
                                                   0,
                                                   0,
                                                   0,
                                                   split,
                                                   max_cus);

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

    std::vector<result_tuple> select_best_macro_tile_size(size_t                         M,
                                                          size_t                         N,
                                                          size_t                         K,
                                                          size_t                         batch,
                                                          bool                           transA,
                                                          bool                           transB,
                                                          const hardware_t&              hardware,
                                                          const std::vector<tile_tuple>& MT_list,
                                                          size_t      element_size_A, //In bits
                                                          size_t      element_size_B, //In bits
                                                          size_t      element_size_out, //In bits
                                                          data_type_t mi_datatype,
                                                          size_t      mx_block_size,
                                                          double      H_L2,
                                                          bool        print,
                                                          size_t      defaultWGM,
                                                          size_t      max_cus)
    {
        std::vector<result_tuple> valid_results;
        valid_results.reserve(MT_list.size());

        // bool tf32_emu = ((mi_datatype == data_type_t::XFloat32)
        //                  && (hardware.arch == hardware_t::architecture_t::gfx950));

        for(const auto& mt : MT_list)
        {
            size_t MT_M           = std::get<0>(mt);
            size_t MT_N           = std::get<1>(mt);
            size_t MT_K           = std::get<2>(mt);
            size_t MI_M           = std::get<3>(mt);
            size_t MI_N           = std::get<4>(mt);
            size_t MI_K           = std::get<5>(mt);
            size_t occupancy      = std::get<6>(mt);
            size_t WGM            = std::get<7>(mt);
            size_t non_temporal_a = std::get<8>(mt);
            size_t non_temporal_b = std::get<9>(mt);

            if(hardware_t::is_debug_enabled())
            {
                std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                          << ", MI_M=" << MI_M << ", MI_N=" << MI_N << ", MI_K=" << MI_K << "\n";
            }

            if(check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size_A))
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
                                                             element_size_A,
                                                             element_size_B,
                                                             element_size_out,
                                                             mi_datatype,
                                                             mx_block_size,
                                                             WGM,
                                                             non_temporal_a,
                                                             non_temporal_b,
                                                             occupancy,
                                                             0,
                                                             max_cus);

                valid_results.emplace_back(Total_latency,
                                           MT_M,
                                           MT_N,
                                           MT_K,
                                           MI_M,
                                           MI_N,
                                           MI_K,
                                           occupancy,
                                           WGM,
                                           non_temporal_a,
                                           non_temporal_b);
            }
            else if(hardware_t::is_debug_enabled())
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
        std::stable_sort(
            valid_results.begin(), valid_results.end(), [](auto const& a, auto const& b) {
                return std::get<0>(a) < std::get<0>(b);
            });

        // 2) Collect results that tie for the absolute best latency.
        double best_latency = std::get<0>(valid_results.front());
        size_t num_the_same = 0;

        // Count the number of similar latencies
        for(const auto& res : valid_results)
        {
            double diff = std::fabs(std::get<0>(res) - best_latency);
            diff /= best_latency;
            // If it's within 1%, include it.
            if(diff < 0.01)
                num_the_same++;
            else
                break; // Once we pass best_latency, we can stop.
        }
        // 3) If that tie group has at least 10 entries, we only use those.
        // 4) Otherwise, keep adding the next best latencies until we have 10 total or run out.
        // std::vector<result_tuple> top_candidates = tie_results;
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
                          << ", MT_M=" << std::get<0>(tile) << ", MT_N=" << std::get<1>(tile)
                          << ", MT_K=" << std::get<2>(tile) << ", MI_M=" << std::get<3>(tile)
                          << ", MI_N=" << std::get<4>(tile) << ", MI_K=" << std::get<5>(tile)
                          << ", Occupancy=" << std::get<6>(tile) << ", WGM=" << std::get<7>(tile)
                          << ", NonTemporalA=" << std::get<8>(tile)
                          << ", NonTemporalB=" << std::get<9>(tile) << "\n";
            }
        }

        return valid_results;
    }

    /*!
         * \brief Selects the best WGMXCC (maximizing L2 hit rate) given fixed macro tile sizes.
         *
         * \param[in] hardware          - Hardware
         * \param[in] M, N, K, batch    - Problem
         * \param[in] MT_M, MT_N, MT_K  - Solution
         * \param[in] print             - whether to print the final best result
         *
         * \return best WGMXCC.
         */
    size_t select_best_wgmxcc(const hardware_t& hardware,
                              size_t            M,
                              size_t            N,
                              size_t            K,
                              size_t            batch,
                              size_t            MT_M,
                              size_t            MT_N,
                              size_t            MT_K,
                              bool              print)
    {
        // Return the number of XCDs
        return hardware.NUM_XCD;
    }

    /*!
         * \brief Selects the best WGM (maximizing L2 hit rate) given fixed macro tile sizes.
         *
         * \param[in] hardware          - Hardware
         * \param[in] M, N, K, batch    - Problem
         * \param[in] MT_M, MT_N, MT_K  - Solution
         * \param[in] print             - whether to print the final best result
         *
         * \return best WGM.
         */
    int32_t select_best_wgm(const hardware_t& hardware,
                            size_t            M,
                            size_t            N,
                            size_t            K,
                            size_t            batch,
                            size_t            MT_M,
                            size_t            MT_N,
                            size_t            MT_K,
                            bool              print)
    {
        constexpr size_t element_size = 32; // L2 scales linearly with element size so it doesn't matter
        
        std::vector<size_t> wgmList = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        
        using WGMResult = std::pair<double, size_t>; // (l2_hit_rate, WGM)
        std::vector<WGMResult> valid_results;
        valid_results.reserve(wgmList.size());

        // Iterate over all candidate WGM values
        for(const auto& candidate_wgm : wgmList)
        {
            if(hardware_t::is_debug_enabled())
            {
                std::cout << "Evaluating WGM=" << candidate_wgm << "\n";
            }

            // Compute L2 hit rate for this WGM
            double current_hit = estimate_l2_hit(hardware,
                                                 M,
                                                 N,
                                                 K,
                                                 batch,
                                                 MT_M,
                                                 MT_N,
                                                 MT_K,
                                                 element_size,
                                                 static_cast<int>(candidate_wgm),
                                                 1 /* splittingFactor */);

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
            valid_results.begin(), valid_results.end(), [](const WGMResult& a, const WGMResult& b) {
                return a.first < b.first; // "less" => a has smaller hit rate than b
            });
        
        // Return best WGM
        return best_it->second;
    }

    // Logic to decide between two MT that are "tied"
    std::vector<std::tuple<double, size_t, size_t, size_t>> tie_breaker_macro_tile_sizes(
        const std::vector<std::tuple<double, size_t, size_t, size_t>>& top_results,
        size_t                                                         M,
        size_t                                                         N,
        size_t                                                         K,
        hardware_t&                                                    hardware,
        std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, hardware_t&)>
            tie_breaker_fn)
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
        std::stable_sort(tie_breaker_results.begin(), tie_breaker_results.end());

        return tie_breaker_results;
    }

    std::vector<std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t>>
        rank_macro_tile_sizes(
            size_t                         M,
            size_t                         N,
            size_t                         K,
            bool                           transA,
            bool                           transB,
            hardware_t&                    hardware,
            const std::vector<tile_tuple>& MT_list,
            size_t                         element_size,
            data_type_t                    mi_datatype,
            double                         H_L2,
            bool                           print,
            size_t                         WGM,
            std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, hardware_t&)>
                tie_breaker_fn,
            size_t                         max_cus)
    {
        std::vector<std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t>> results;

        typedef std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t> result_tuple;

        for(size_t i = 0; i < MT_list.size(); ++i)
        {
            size_t MT_M = std::get<0>(MT_list[i]);
            size_t MT_N = std::get<1>(MT_list[i]);
            size_t MT_K = std::get<2>(MT_list[i]);
            size_t MI_M = std::get<3>(MT_list[i]);
            size_t MI_N = std::get<4>(MT_list[i]);
            size_t MI_K = std::get<5>(MT_list[i]);

            if(hardware_t::is_debug_enabled())
            {
                std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                          << ", MI_M=" << MI_M << ", MI_N=" << MI_N << ", MI_K=" << MI_K << "\n";
            }

            if(check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size))
            {
                size_t split         = 1;
                size_t mx_block_size = 0;
                double Total_latency = compute_total_latency(hardware,
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
                                                             element_size * 8, //Element Size A
                                                             element_size * 8, //Element Size B
                                                             element_size * 8, //Element Size out
                                                             mi_datatype,
                                                             mx_block_size,
                                                             WGM,
                                                             0,
                                                             0,
                                                             0,
                                                             split,
                                                             max_cus);

                results.push_back(
                    std::make_tuple(Total_latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K));
            }
            else if(hardware_t::is_debug_enabled())
            {
                std::cout << "Skipping MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                          << " due to LDS capacity\n";
            }
        }

        // Sort results by Total_latency, from worst (largest latency) to best (smallest latency)
        std::stable_sort(
            results.begin(), results.end(), [](const result_tuple& a, const result_tuple& b) {
                return std::get<0>(a) > std::get<0>(b);
            });

        if(!results.empty())
        {
            double best_latency = std::get<0>(results.back());

            std::vector<result_tuple> top_results;
            for(size_t i = 0; i < results.size(); ++i)
            {
                if(std::abs(std::get<0>(results[i]) - best_latency) < 1e-6)
                {
                    top_results.push_back(results[i]);
                }
            }

            if(top_results.size() > 1)
            {
                if(hardware_t::is_debug_enabled())
                {
                    std::cout << "Tie detected among top-ranked tile sizes. Applying "
                                 "tie-breaker...\n";
                }

                // Compute tie-breaker scores and store them along with the result indices
                std::vector<std::pair<double, size_t>>
                    tie_breaker_scores; // (score, index in top_results)

                for(size_t i = 0; i < top_results.size(); ++i)
                {
                    const result_tuple& res  = top_results[i];
                    size_t              MT_M = std::get<1>(res);
                    size_t              MT_N = std::get<2>(res);
                    size_t              MT_K = std::get<3>(res);
                    size_t              MI_M = std::get<4>(res);
                    size_t              MI_N = std::get<5>(res);
                    size_t              MI_K = std::get<6>(res);
                    double score = tie_breaker_fn(MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, hardware);

                    tie_breaker_scores.push_back(std::make_pair(score, i));
                }

                // Now sort the tie_breaker_scores based on score
                std::stable_sort(
                    tie_breaker_scores.begin(),
                    tie_breaker_scores.end(),
                    [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                        return a.first > b.first;
                    });

                // Now re-order 'top_results' based on sorted indices
                std::vector<result_tuple> sorted_top_results;
                for(size_t i = 0; i < tie_breaker_scores.size(); ++i)
                {
                    size_t idx = tie_breaker_scores[i].second;
                    sorted_top_results.push_back(top_results[idx]);
                }

                // Remove the tied results from 'results' and insert the sorted 'sorted_top_results'
                results.erase(std::remove_if(results.begin(),
                                             results.end(),
                                             [best_latency](const result_tuple& res) {
                                                 return std::abs(std::get<0>(res) - best_latency)
                                                        < 1e-6;
                                             }),
                              results.end());

                results.insert(results.end(), sorted_top_results.begin(), sorted_top_results.end());
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
                std::cout << std::fixed << std::setprecision(2) << latency << "\t" << MT_M << "\t"
                          << MT_N << "\t" << MT_K << "\t" << MI_M << "\t" << MI_N << "\t" << MI_K
                          << "\n";
            }
        }

        return results;
    }

    double compute_tflops_from_latency(
        double latency_cycles, size_t M, size_t N, size_t K, double clock_GHz)
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

        if(hardware_t::is_debug_enabled())
        {
            std::cout << "Total FLOPs: " << total_FLOPs << "\n";
            std::cout << "Total Time: " << total_time_seconds << " seconds\n";
            std::cout << "Performance: " << FLOPS << " FLOPS\n";
            std::cout << "Achieved Performance: " << TFLOPS << " TFLOPS\n";
        }

        return TFLOPS;
    }
} // namespace origami
