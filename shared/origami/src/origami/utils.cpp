// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/utils.hpp"

#include <algorithm>
#include <chrono>  // For timing
#include <string>
#include <cmath>
#include <iomanip> // For output formatting
#include <iostream>

namespace origami {

static double read_heuristics_variance_env_var() {
    constexpr double default_variance = 0.01; // 1%

    if (const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE")) {
        try {
            double val = std::stod(env);
            if (std::isfinite(val) && val > 0.0) {
                return val;
            }
        } catch (...) {
            // fall through to default
        }
    }
    return default_variance;
}

//
// Tiebreaker function.
//
void pick_best_tile_by_arithmetic_intensity(std::vector<result_tuple>& top_results,
                                            size_t num_to_sort) {
    if (top_results.empty()) {
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

        double flops = static_cast<double>(2ull * MT_M * MT_N * MT_K);
        double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

        // Avoid division by zero.
        if (memory_traffic == 0.0) return 0.0;

        return flops / memory_traffic;
    };
    // 2) Sort the results in descending order of arithmetic intensity
    //    (highest arithmetic intensity first).
    std::stable_sort(top_results.begin(), top_results.begin() + num_to_sort,
                     [&](const result_tuple& a, const result_tuple& b) {
                         double ai_a = compute_arithmetic_intensity(a);
                         double ai_b = compute_arithmetic_intensity(b);
                         return ai_a > ai_b;  // descending
                     });
    // 3) Return the tile with the highest arithmetic intensity
}

result_tuple pick_best_tile_with_dimension_priority(const std::vector<result_tuple>& top_results,
                                                    size_t M, size_t N, size_t K) {
    if (top_results.empty()) {
        throw std::runtime_error("pick_best_tile_with_dimension_priority received empty list.");
    }

    // 1) Determine whether M or N is more important
    //    (based on which is larger), and always place K last.
    //    This yields a priority order of either { 'M', 'N', 'K' }
    //    or { 'N', 'M', 'K' }.
    std::vector<char> dimPriority;
    if (M >= N)
        dimPriority = {'M', 'N', 'K'};
    else
        dimPriority = {'N', 'M', 'K'};

    // 2) Helper function to extract the tile dimension:
    //    (latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K)
    auto getTileSize = [](const result_tuple& t, char dimChar) -> size_t {
        switch (dimChar) {
            case 'M':
                return std::get<1>(t);  // MT_M
            case 'N':
                return std::get<2>(t);  // MT_N
            case 'K':
                return std::get<3>(t);  // MT_K
            default:
                return 0;
        }
    };

    // 3) Sort in descending order according to the dimension priority.
    //    - Compare dimensionPriority[0] first
    //    - If there's a tie, compare dimensionPriority[1]
    //    - If still a tie, compare dimensionPriority[2]
    //    - If they're all equal, consider them tied
    std::vector<result_tuple> sorted = top_results;  // copy
    std::stable_sort(sorted.begin(), sorted.end(),
                     [&](const result_tuple& a, const result_tuple& b) {
                         for (char d : dimPriority) {
                             size_t ta = getTileSize(a, d);
                             size_t tb = getTileSize(b, d);
                             if (ta > tb) return true;
                             if (ta < tb) return false;
                         }
                         // If all relevant dimensions are the same, treat as a tie
                         return false;
                     });

    // 4) Return the best tile (the first after sorting).
    return sorted.front();
}

size_t select_best_grid_size(size_t M, size_t N, size_t K, size_t batch, bool transA, bool transB,
                             const hardware_t& hardware, size_t MT_M, size_t MT_N, size_t MT_K,
                             size_t MI_M, size_t MI_N, size_t MI_K, size_t element_size_A,
                             size_t element_size_B, size_t element_size_out,
                             data_type_t mi_datatype, size_t mx_block_size, double H_L2, int WGM,
                             size_t biggest_allowable_split, size_t max_cus) {
    // compute how many 32×32 tiles are needed in each dim,
    // then multiply to get total grid size:
    size_t grid = ((M + MT_M - 1) / MT_M) * ((N + MT_N - 1) / MT_N) * batch;

    size_t max_hw_split = std::floor(hardware.N_CU / grid);
    size_t MAX_SPLIT = std::min(biggest_allowable_split, max_hw_split);

    size_t best_split = 1;
    double best_latency = std::numeric_limits<double>::infinity();

    for (size_t split = 1; split <= MAX_SPLIT; ++split) {
        double latency =
            compute_total_latency(hardware, M, N,
                                  K,  // problem dims
                                  batch, transA, transB, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K,
                                  element_size_A,    // ElementSizeA
                                  element_size_B,    // ElementSizeB
                                  element_size_out,  // ElementSizeout
                                  mi_datatype, mx_block_size, WGM, 0, 0, 0, split, max_cus);

        if (latency < best_latency) {
            best_latency = latency;
            best_split = split;
        }
        best_latency = latency;
        best_split = split;
    }

    size_t best_grid = best_split * grid;

    // you now have both `grid` and `best_split`—
    // return whichever is appropriate (here we stick with split):
    return best_grid;
}

std::vector<result_tuple> select_best_macro_tile_size(size_t M, size_t N, size_t K, size_t batch,
                                                      bool transA, bool transB,
                                                      const hardware_t& hardware,
                                                      const std::vector<tile_tuple>& MT_list,
                                                      size_t element_size_A,    // In bits
                                                      size_t element_size_B,    // In bits
                                                      size_t element_size_out,  // In bits
                                                      data_type_t mi_datatype, size_t mx_block_size,
                                                      double H_L2, bool print, int defaultWGM,
                                                      size_t max_cus) {
    std::vector<result_tuple> valid_results;
    valid_results.reserve(MT_list.size());

    // bool tf32_emu = ((mi_datatype == data_type_t::XFloat32)
    //                  && (hardware.arch == hardware_t::architecture_t::gfx950));

    for (const auto& mt : MT_list) {
        size_t MT_M = std::get<0>(mt);
        size_t MT_N = std::get<1>(mt);
        size_t MT_K = std::get<2>(mt);
        size_t MI_M = std::get<3>(mt);
        size_t MI_N = std::get<4>(mt);
        size_t MI_K = std::get<5>(mt);
        int occupancy = std::get<6>(mt);
        int WGM = std::get<7>(mt);
        int non_temporal_a = std::get<8>(mt);
        int non_temporal_b = std::get<9>(mt);

        if (hardware_t::is_debug_enabled()) {
            std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                      << ", MI_M=" << MI_M << ", MI_N=" << MI_N << ", MI_K=" << MI_K << "\n";
        }

        if (check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size_A)) {
            double Total_latency = compute_total_latency(
                hardware, M, N, K, batch, transA, transB, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K,
                element_size_A, element_size_B, element_size_out, mi_datatype, mx_block_size,
                defaultWGM, non_temporal_a, non_temporal_b, occupancy, 0, max_cus);

            valid_results.emplace_back(Total_latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, occupancy,
                                       WGM, non_temporal_a, non_temporal_b);
        } else if (hardware_t::is_debug_enabled()) {
            std::cout << "Skipping MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                      << " due to LDS capacity\n";
        }
    }

    if (valid_results.empty()) {
        throw std::runtime_error("No valid macro-tile sizes found.");
    }

    // 1) Sort results by ascending latency.
    std::stable_sort(valid_results.begin(), valid_results.end(),
                     [](auto const& a, auto const& b) { return std::get<0>(a) < std::get<0>(b); });

    // 2) Collect results that tie for the absolute best latency.
    double best_latency = std::get<0>(valid_results.front());
    size_t num_the_same = 0;

    // Count the number of similar latencies
    constexpr double epsilon = 1e-9;
    // variance is set through environment variable ANALYTICAL_GEMM_HEURISTICS_VARIANCE
    static const double top_N_heuristic = read_heuristics_variance_env_var();
    for (const auto& res : valid_results) {
        bool within_top;
        const double diff = std::abs(std::get<0>(res) - best_latency);

        if (top_N_heuristic <= epsilon) {
            // Absolute tolerance path
            within_top = diff < epsilon;
        } else {
            // Relative tolerance path (guard denom)
            const double denom = std::max(std::abs(best_latency), epsilon);
            // If it's within top_N_heuristic%, include it.
            within_top = (diff / denom) < top_N_heuristic;
        }

        if (within_top)
            ++num_the_same;
        else
            break;
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

    // After arithmetic intensity tie-breaking, check if we still have ties
    // among the top results (those with same latency and arithmetic intensity)
    if (num_the_same > 1) {
        // Helper to compute arithmetic intensity
        auto compute_arithmetic_intensity = [](const result_tuple& t) -> double {
            auto MT_M = std::get<1>(t);
            auto MT_N = std::get<2>(t);
            auto MT_K = std::get<3>(t);
            double flops = static_cast<double>(2ull * MT_M * MT_N * MT_K);
            double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);
            return (memory_traffic == 0.0) ? 0.0 : (flops / memory_traffic);
        };

        // Check if the top tiles still have the same arithmetic intensity
        double first_ai = compute_arithmetic_intensity(valid_results[0]);
        size_t num_same_ai = 1;
        for (size_t i = 1; i < num_the_same; ++i) {
            double current_ai = compute_arithmetic_intensity(valid_results[i]);
            if (std::abs(current_ai - first_ai) < 1e-6) {
                num_same_ai++;
            } else {
                break;
            }
        }

        // If we still have ties after arithmetic intensity, apply problem dimension tie-breaker
        if (num_same_ai > 1) {
            // Problem dimension-based tie breaker:
            // If M > N, prefer tiles with larger MT_M
            // If N > M, prefer tiles with larger MT_N
            // If M == N, this tie-breaker doesn't apply (will use final tie-breaker)

            if (M != N) {
                std::stable_sort(valid_results.begin(), valid_results.begin() + num_same_ai,
                                 [M, N](const result_tuple& a, const result_tuple& b) {
                                     size_t MT_M_a = std::get<1>(a);
                                     size_t MT_N_a = std::get<2>(a);
                                     size_t MT_M_b = std::get<1>(b);
                                     size_t MT_N_b = std::get<2>(b);

                                     if (M > N) {
                                         // M-dominant: prefer larger MT_M
                                         if (MT_M_a != MT_M_b) return MT_M_a > MT_M_b;
                                         // If MT_M is same, prefer larger MT_N as secondary
                                         return MT_N_a > MT_N_b;
                                     } else  // N > M
                                     {
                                         // N-dominant: prefer larger MT_N
                                         if (MT_N_a != MT_N_b) return MT_N_a > MT_N_b;
                                         // If MT_N is same, prefer larger MT_M as secondary
                                         return MT_M_a > MT_M_b;
                                     }
                                 });
            }

            // Final tie-breaker: when all else is equal (including square problems),
            // consistently prefer tiles with larger MT_M
            // This ensures deterministic selection regardless of input order
            std::stable_sort(valid_results.begin(), valid_results.begin() + num_same_ai,
                             [](const result_tuple& a, const result_tuple& b) {
                                 size_t MT_M_a = std::get<1>(a);
                                 size_t MT_N_a = std::get<2>(a);
                                 size_t MT_K_a = std::get<3>(a);
                                 size_t MT_M_b = std::get<1>(b);
                                 size_t MT_N_b = std::get<2>(b);
                                 size_t MT_K_b = std::get<3>(b);

                                 // Prefer larger MT_M first
                                 if (MT_M_a != MT_M_b) return MT_M_a > MT_M_b;
                                 // If MT_M is same, prefer larger MT_N
                                 if (MT_N_a != MT_N_b) return MT_N_a > MT_N_b;
                                 // If both MT_M and MT_N are same, prefer larger MT_K
                                 return MT_K_a > MT_K_b;
                             });
        }
    }

    if (print) {
        for (const auto& tile : valid_results) {
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

template <typename N, typename D>
constexpr N safe_ceil_div(N n, D d) {
    // Static cast to undo integral promotion.
    return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
}

/*!
 * \brief Selects the best WGM (maximizing L2 hit rate) given fixed macro tile sizes.
 *
 * \param[in] hardware          - Hardware
 * \param[in] M, N, K, batch    - Problem
 * \param[in] MT_M, MT_N, MT_K  - Solution
 * \param[in] print             - whether to print the final best result
 *
 * \return best WGMXCC, WGM.
 */
std::tuple<size_t, int32_t> select_best_wgm(const hardware_t& hardware, size_t M, size_t N,
                                            size_t K, size_t batch, size_t MT_M, size_t MT_N,
                                            size_t MT_K, int nta, int ntb, size_t skGrid,
                                            bool print) {
    // Default is the closest we can get to a square
    size_t max_CU_XCD = hardware.N_CU / hardware.NUM_XCD;
    size_t defaultWGM = ceil(std::sqrt(max_CU_XCD));

    // Number of output MTs per split and batch
    size_t numMT_M = safe_ceil_div(M, MT_M);
    size_t numMT_N = safe_ceil_div(N, MT_N);
    size_t numMTs = numMT_M * numMT_N;

    // What SK does -- we already have skGrid so just compute numWaves and splitFactor
    auto numWaves = skGrid > numMTs ? safe_ceil_div(skGrid, hardware.N_CU)
                                    : safe_ceil_div(numMTs, hardware.N_CU);
    auto splitFactor = safe_ceil_div(skGrid, numMTs);

    // -------------------
    // NonTemporal Cases
    // -------------------
    if (nta > 3 && ntb < 4)
        return std::make_tuple(hardware.NUM_XCD, 1);
    else if (nta < 4 && ntb > 3)
        return std::make_tuple(hardware.NUM_XCD,
                               std::min(max_CU_XCD, safe_ceil_div(numMTs, hardware.NUM_XCD)));
    else if (nta > 3 && ntb > 3)
        return std::make_tuple(hardware.NUM_XCD, 1);

    // -------------------
    // WGMXCC Prediction
    // -------------------
    // Default WGMXCC -- always number of XCD
    auto defaultWGMXCC = hardware.NUM_XCD;
    bool isWGMXCCset = false;
    size_t out_wgmxcc = defaultWGMXCC;

    // Batched GEMMs
    if (batch > 1 && !isWGMXCCset) {
        // Total tiles including batch count
        size_t numTotalTiles = numMTs * batch;

        // if only one MT per each GEMM -> no mapping
        // if less than hardware.NUM_XCD total tiles -> no mapping
        if (numMTs == 1 || numTotalTiles <= hardware.NUM_XCD) {
            out_wgmxcc = 1;
            isWGMXCCset = true;
        }
        // else use the default (num_xcd)
    }

    // If we are lucky that the splitFactor is a multiple of NUM_XCD -> no mapping
    if ((splitFactor % hardware.NUM_XCD == 0) && !isWGMXCCset) {
        out_wgmxcc = 1;
        isWGMXCCset = true;
    }

    // Small GEMMs
    if ((numMTs <= hardware.NUM_XCD) && !isWGMXCCset) {
        out_wgmxcc = 1;
        isWGMXCCset = true;
    }

    // For sizes that we have more than 2 waves of computations, we skip xcc mapping as MALL is
    // more important -- matrix should not be skinny
    // To avoid regressions, it's set to default, but it should actually be 1!
    bool MallIsImportant = (splitFactor == 1 && batch == 1 && numMTs > 2 * hardware.N_CU &&
                            numMT_M > 8 && numMT_N > 8);
    if (MallIsImportant && !isWGMXCCset) {
        out_wgmxcc = defaultWGMXCC;
        isWGMXCCset = true;
    }

    // -------------------
    // WGM Prediction
    // -------------------
    // Default WGM
    bool isWGMset = false;
    size_t out_wgm = defaultWGM;

    // shortcut:
    // 1. if we have decided to not remap xcc, there is no reason to use wgm
    // 2. GEMMs that only have one tile in one dimension don't need wgm
    // 3. Batched GEMMs don't need wgm (emprically -> batch count is often large!)
    if (((out_wgmxcc == 1 && !MallIsImportant) || numMT_M == 1 || numMT_N == 1 || batch > 1) &&
        !isWGMset) {
        out_wgm = 1;
        isWGMset = true;
    }

    // For tall cases (M >> N), if we have enough tiles to schedule, we use the number of tiles
    // in the smaller dimension as WGM value
    if (numMTs > hardware.N_CU && M > 10 * N && numMT_N <= 8) {
        out_wgm = numMT_N;
        isWGMset = true;
    }

    // Cases where we have multiple rounds of computation per each CU
    // To avoid regressions, it's set to defaultWGM. However, I think WGM=1 should be the winner
    if (MallIsImportant && !isWGMset) {
        out_wgm = defaultWGM;
        isWGMset = true;
    }

    if (!isWGMset) {
        size_t numWGs = numWaves * splitFactor * numMTs;
        size_t q = numWGs / hardware.NUM_XCD;
        size_t r = numWGs % hardware.NUM_XCD;

        std::vector<int> wgmList = {1, 2, 3, 4, 5, 6, 8, 16};
        int bestWGM = 1;
        int bestL2 = std::numeric_limits<int>::max();
        for (auto wgm : wgmList) {
            auto slabTiles = numMT_M * std::min(wgm, static_cast<int>(numMT_N));
            auto slabCount = safe_ceil_div(numMT_N, wgm);
            auto edgeSlabWidth = numMT_N - (slabCount - 1) * wgm;
            auto wgmL2Estimate = 0;
            auto numXCD = std::min(hardware.NUM_XCD, numWGs);

            // Compute unique loads per L2 tile
            for (uint32_t x = 0; x < numWaves * numXCD; ++x) {
                // Range of "output tiles" that this xcd takes.
                auto xccStart = q * x + (x < r ? x : r);
                auto xccEnd = xccStart + q - 1 + (x < r ? 1 : 0);
                // xccStart and xccEnd are supposed to be tile IDs
                // In case of splitting, they are WG IDs. Modify to get tile IDs
                xccStart /= splitFactor;
                xccEnd /= splitFactor;

                auto slabStart = xccStart / slabTiles;
                auto slabEnd = xccEnd / slabTiles;

                auto firstSlabWidth = (slabStart == slabCount - 1 ? edgeSlabWidth : wgm);
                auto firstSlabStartIndex = xccStart % slabTiles;
                auto firstSlabStartRow = firstSlabStartIndex / firstSlabWidth;
                auto firstSlabEndRow = std::min(
                    (firstSlabStartIndex + (xccEnd - xccStart)) / firstSlabWidth, numMT_M - 1);
                auto rowsInFirstSlab = firstSlabEndRow - firstSlabStartRow + 1;

                auto lastSlabWidth = (slabEnd == slabCount - 1 ? edgeSlabWidth : wgm);
                auto lastSlabEndIndex = xccEnd % slabTiles;
                auto lastSlabEndRow = lastSlabEndIndex / lastSlabWidth;
                auto colsInLastRow = (lastSlabEndIndex % lastSlabWidth) + 1;
                auto colsInLastSlab = (lastSlabEndRow > 0 ? lastSlabWidth : colsInLastRow);

                size_t uniqueRows = 0;
                size_t uniqueCols = 0;
                if (slabEnd == slabStart) {
                    uniqueRows = lastSlabEndRow - firstSlabStartRow + 1;
                    uniqueCols = firstSlabWidth;
                    if (rowsInFirstSlab <= 2)
                        uniqueCols = std::min(xccEnd - xccStart + 1, firstSlabWidth);
                } else {
                    auto colsInFirstRow = firstSlabWidth - (xccStart % firstSlabWidth);
                    auto colsInFirstSlab = (rowsInFirstSlab > 1 ? firstSlabWidth : colsInFirstRow);
                    auto fullSlabs = slabEnd - slabStart - 1;
                    uniqueRows =
                        (fullSlabs > 0 ? numMT_M
                                       : std::min(rowsInFirstSlab + lastSlabEndRow + 1, numMT_M));
                    uniqueCols = colsInFirstSlab + colsInLastSlab + fullSlabs * wgm;
                }

                // Sum up the L2 loads over all XCD
                // We should technically multiply by K (or splitted K), but it
                // has no effect on sorting
                auto xccL2Estimate = uniqueRows * MT_M + uniqueCols * MT_N;
                wgmL2Estimate += xccL2Estimate;
            }

            // If we have found a better WGM
            if (wgmL2Estimate < bestL2) {
                bestL2 = wgmL2Estimate;
                bestWGM = wgm;
            }

            if (print || hardware_t::is_debug_enabled())
                std::cout << "WGM (" << wgm << "), L2Estimate (" << wgmL2Estimate << ")"
                          << std::endl;
        }

        out_wgm = bestWGM;
    }

    return std::make_tuple(out_wgmxcc, out_wgm);
}

// Logic to decide between two MT that are "tied"
std::vector<std::tuple<double, size_t, size_t, size_t>> tie_breaker_macro_tile_sizes(
    const std::vector<std::tuple<double, size_t, size_t, size_t>>& top_results, size_t M, size_t N,
    size_t K, hardware_t& hardware,
    std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, hardware_t&)>
        tie_breaker_fn) {
    std::vector<std::tuple<double, size_t, size_t, size_t>> tie_breaker_results;

    for (const auto& res : top_results) {
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
    size_t M, size_t N, size_t K, bool transA, bool transB, hardware_t& hardware,
    const std::vector<tile_tuple>& MT_list, size_t element_size, data_type_t mi_datatype,
    double H_L2, bool print, size_t WGM,
    std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, hardware_t&)>
        tie_breaker_fn,
    size_t max_cus) {
    std::vector<std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t>> results;

    typedef std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t> result_tuple;

    for (size_t i = 0; i < MT_list.size(); ++i) {
        size_t MT_M = std::get<0>(MT_list[i]);
        size_t MT_N = std::get<1>(MT_list[i]);
        size_t MT_K = std::get<2>(MT_list[i]);
        size_t MI_M = std::get<3>(MT_list[i]);
        size_t MI_N = std::get<4>(MT_list[i]);
        size_t MI_K = std::get<5>(MT_list[i]);

        if (hardware_t::is_debug_enabled()) {
            std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                      << ", MI_M=" << MI_M << ", MI_N=" << MI_N << ", MI_K=" << MI_K << "\n";
        }

        if (check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size)) {
            size_t split = 1;
            size_t mx_block_size = 0;
            double Total_latency =
                compute_total_latency(hardware, M, N, K,
                                      1,  // Batch
                                      transA, transB, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K,
                                      element_size * 8,  // Element Size A
                                      element_size * 8,  // Element Size B
                                      element_size * 8,  // Element Size out
                                      mi_datatype, mx_block_size, WGM, 0, 0, 0, split, max_cus);

            results.push_back(std::make_tuple(Total_latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K));
        } else if (hardware_t::is_debug_enabled()) {
            std::cout << "Skipping MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                      << " due to LDS capacity\n";
        }
    }

    // Sort results by Total_latency, from worst (largest latency) to best (smallest latency)
    std::stable_sort(results.begin(), results.end(),
                     [](const result_tuple& a, const result_tuple& b) {
                         return std::get<0>(a) > std::get<0>(b);
                     });

    if (!results.empty()) {
        double best_latency = std::get<0>(results.back());

        std::vector<result_tuple> top_results;
        for (size_t i = 0; i < results.size(); ++i) {
            if (std::abs(std::get<0>(results[i]) - best_latency) < 1e-6) {
                top_results.push_back(results[i]);
            }
        }

        if (top_results.size() > 1) {
            if (hardware_t::is_debug_enabled()) {
                std::cout << "Tie detected among top-ranked tile sizes. Applying "
                             "tie-breaker...\n";
            }

            // Compute tie-breaker scores and store them along with the result indices
            std::vector<std::pair<double, size_t>>
                tie_breaker_scores;  // (score, index in top_results)

            for (size_t i = 0; i < top_results.size(); ++i) {
                const result_tuple& res = top_results[i];
                size_t MT_M = std::get<1>(res);
                size_t MT_N = std::get<2>(res);
                size_t MT_K = std::get<3>(res);
                size_t MI_M = std::get<4>(res);
                size_t MI_N = std::get<5>(res);
                size_t MI_K = std::get<6>(res);
                double score = tie_breaker_fn(MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, hardware);

                tie_breaker_scores.push_back(std::make_pair(score, i));
            }

            // Now sort the tie_breaker_scores based on score
            std::stable_sort(tie_breaker_scores.begin(), tie_breaker_scores.end(),
                             [](const std::pair<double, size_t>& a,
                                const std::pair<double, size_t>& b) { return a.first > b.first; });

            // Now re-order 'top_results' based on sorted indices
            std::vector<result_tuple> sorted_top_results;
            for (size_t i = 0; i < tie_breaker_scores.size(); ++i) {
                size_t idx = tie_breaker_scores[i].second;
                sorted_top_results.push_back(top_results[idx]);
            }

            // Remove the tied results from 'results' and insert the sorted 'sorted_top_results'
            results.erase(std::remove_if(results.begin(), results.end(),
                                         [best_latency](const result_tuple& res) {
                                             return std::abs(std::get<0>(res) - best_latency) <
                                                    1e-6;
                                         }),
                          results.end());

            results.insert(results.end(), sorted_top_results.begin(), sorted_top_results.end());
            // No need to re-sort results as total_latency remains same for tied results
        }
    }

    if (print) {
        std::cout << "Total Latency\tMT_M\tMT_N\tMT_K\tMI_M\tMI_N\tMI_K\n";
        for (size_t i = 0; i < results.size(); ++i) {
            double latency = std::get<0>(results[i]);
            size_t MT_M = std::get<1>(results[i]);
            size_t MT_N = std::get<2>(results[i]);
            size_t MT_K = std::get<3>(results[i]);
            size_t MI_M = std::get<4>(results[i]);
            size_t MI_N = std::get<5>(results[i]);
            size_t MI_K = std::get<6>(results[i]);
            std::cout << std::fixed << std::setprecision(2) << latency << "\t" << MT_M << "\t"
                      << MT_N << "\t" << MT_K << "\t" << MI_M << "\t" << MI_N << "\t" << MI_K
                      << "\n";
        }
    }

    return results;
}

double compute_tflops_from_latency(double latency_cycles, size_t M, size_t N, size_t K,
                                   double clock_GHz) {
    // Compute total FLOPs
    double total_FLOPs = 2.0 * M * N * K;  // For GEMM, each multiply-add is 2 FLOPs
    // Compute total time in seconds
    double cycles_per_second = clock_GHz * 1e9;  // 1 GHz = 1e9 cycles per second
    double total_time_seconds = latency_cycles / cycles_per_second;
    // Compute performance in FLOPS
    double FLOPS = total_FLOPs / total_time_seconds;
    // Convert to TFLOPS
    double TFLOPS = FLOPS / 1e12;  // 1 TFLOP = 1e12 FLOPs

    if (hardware_t::is_debug_enabled()) {
        std::cout << "Total FLOPs: " << total_FLOPs << "\n";
        std::cout << "Total Time: " << total_time_seconds << " seconds\n";
        std::cout << "Performance: " << FLOPS << " FLOPS\n";
        std::cout << "Achieved Performance: " << TFLOPS << " TFLOPS\n";
    }

    return TFLOPS;
}
}  // namespace origami
