// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/hardware.hpp"
#include "origami/streamk.hpp"
#include "origami/utils.hpp"
// #include <algorithm>

namespace origami
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

    const char* rtype_to_string(streamk::reduction_type r)
    {
        switch(r)
        {
        case streamk::reduction_type::Tree:
            return "Tree";
        case streamk::reduction_type::Parallel:
            return "Parallel";
        case streamk::reduction_type::None:
            return "None";
        default:
            return "Unknown";
        }
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

        size_t get_workspace(
            size_t x,
            size_t y,
            size_t mt_m,
            size_t mt_n,
            size_t bpe_c,
            size_t grid,
            size_t tiles,
            reduction_type reduction)
        {
            size_t size = 0;
            if(reduction == reduction_type::Tree)
            {
                if(tiles % grid == 0)
                {
                    size_t tileSize = mt_m * mt_n * bpe_c;
                    size += tileSize * grid;
                }
            }
            else if(reduction == reduction_type::Parallel)
            {
                size_t splitSize = x * y * bpe_c;
                size_t splitCount = grid / tiles;
                size += splitSize * splitCount;
            }
            return size;
        }

        reduction_type select_reduction(
            size_t x,
            size_t y,
            size_t z,
            size_t batch,
            size_t mt_m,
            size_t mt_n,
            size_t mt_k,
            const hardware_t& analytical_hardware,
            int dynamic_grid_version)
        {
            reduction_type reductionStrat = reduction_type::Tree;

            if(dynamic_grid_version == 6)
            {
                size_t tiles = number_of_output_tiles(mt_m, mt_n, x, y, batch);
                size_t cu_count = analytical_hardware.N_CU;
                size_t iters_per_tile = std::max(size_t(1), math::safe_ceil_div(z, mt_k));

                if (tiles < cu_count)
                {
                    // For problems with large k and low number of tiles, use parallel reduction
                    // TODO Benchmark to check if limits are correct
                    constexpr int MinItersForParallel = 64;
                    constexpr int MaxTilesForParallel = 64;
                    if (iters_per_tile >= MinItersForParallel && tiles <= MaxTilesForParallel)
                        reductionStrat = reduction_type::Parallel;
                }
            }

            return reductionStrat;
        }

        size_t select_grid(
            size_t x,
            size_t y,
            size_t z,
            size_t batch,
            bool            trans_a,
            bool            trans_b,
            size_t          element_size_A,
            size_t          element_size_B,
            size_t          element_size_out,
            data_type_t     mi_datatype,
            size_t          workspace_size,
            size_t          mt_m,
            size_t          mt_n,
            size_t          mt_k,
            size_t          mi_m,
            size_t          mi_n,
            size_t          mi_k,
            size_t          workgroup_mapping,
            size_t          workspace_size_per_elem_c,
            int             occupancy,
            const hardware_t& analytical_hardware,
            int dynamic_grid_version,
            reduction_type reduction_strategy,
            size_t max_cus)
        {
            size_t cu_count = analytical_hardware.N_CU;
            if(max_cus > 0)
                cu_count = std::min(cu_count, max_cus);
            size_t tiles = number_of_output_tiles(mt_m, mt_n, x, y, batch);

            // Dynamically pick the minimum between the cu_count or number of tiles.
            if(dynamic_grid_version == 1)
            {
                return std::min(cu_count, tiles);
            }

            // Dynamically pick the minimum between the cu_count or number of tiles,
            // and scale down really large sizes to use fewer CUs for power/energy savings.
            else if(dynamic_grid_version == 2)
            {
                size_t sk_grid = cu_count;
                if(tiles > sk_grid)
                {
                    for(size_t i = 1; i <= 32; i *= 2)
                    {
                        size_t tilesPerCU  = math::safe_ceil_div(i * tiles, cu_count);
                        size_t reducedGrid = math::safe_ceil_div(i * tiles, tilesPerCU);
                        float  utilization = ((float)reducedGrid) / ((float)cu_count);
                        if(utilization > 0.75f)
                        {
                            if(utilization < 1.0f)
                            sk_grid = reducedGrid;
                            break;
                        }
                    }
                }

                return std::min(sk_grid, tiles);
            }
            // Dynamically predict the best grid-size by weighing the cost of the fix-up
            // step and the cost of processing MAC-loop instructions. When the cost of fix-up
            // is the bottleneck, use smaller grid size.
            // Architecture dependent.
            else if(dynamic_grid_version == 3)
            {
                return origami::streamk::best_predicted_grid_size(mt_m,
                                        mt_n,
                                        mt_k,
                                        x,
                                        y,
                                        z,
                                        batch,
                                        1,
                                        cu_count);
            }
            // Fix Stream-K algorithm to function like a Data-parallel schedule
            // where grid size is equal to the number of output tiles.
            else if(dynamic_grid_version == 4)
            {
                return origami::streamk::number_of_output_tiles(
                mt_m, mt_n, x, y, batch);
            }
            else if(dynamic_grid_version == 5)
            {
                return origami::select_best_grid_size(x,
                            y,
                            z,
                            batch,
                            trans_a,
                            trans_b,
                            analytical_hardware,
                            mt_m,
                            mt_n,
                            mt_k,
                            mi_m,
                            mi_n,
                            mi_k,
                            element_size_A,
                            element_size_B,
                            element_size_out,
                            mi_datatype,
                            0,
                            0.0,
                            workgroup_mapping,
                            10,
                            max_cus);
            }
            else if(dynamic_grid_version == 6)
            {
                size_t iters_per_tile = std::max(size_t(1), math::safe_ceil_div(z, mt_k));
                size_t sk_grid = tiles; // Fallback if no good fractional tile is found
                if(max_cus > 0)
                    sk_grid = std::min(sk_grid, max_cus);
                size_t tile_size = mt_m * mt_n * workspace_size_per_elem_c;
                // More tiles than CUs
                // Distribute tiles evenly across maximum number of CUs
                // Split remaining tiles as evenly as possible for better caching
                if(tiles > cu_count)
                {
                    size_t virt_cu_count = cu_count;
                    if (occupancy > 1 && max_cus == 0)
                        virt_cu_count *= occupancy;

                    const std::vector<double> tile_fractions = {0.0, 1.0/2.0, 1.0/8.0, 1.0/5.0, 1.0/4.0, 1.0/3.0};
                    size_t min_even_tiles = tiles / virt_cu_count;
                    for(double frac: tile_fractions)
                    {
                        size_t frac_grid = (size_t)((tiles / (min_even_tiles + frac)) + 0.5);
                        // Check if higher occupancy would cause excessive workspace requirements (set current limit to 128MB)
                        if((tiles % frac_grid != 0) && (tile_size * frac_grid > 128*1024*1024))
                            continue;
                        if(frac_grid <= virt_cu_count)
                        {
                            sk_grid = frac_grid;
                            break;
                        }
                    }
                }
                // Fewer tiles than CUs
                // Split tiles evenly in k-dimension
                // Attempt to maximize CU utilization, up to a peak number of splits
                // Max splitting is currently constant, but should be dependant on K dimension
                else if (tiles < cu_count)
                {
                    // For problems with large k and low number of tiles, use parallel reduction
                    // TODO Benchmark to check if limits are correct
                    // constexpr int MinItersForParallel = 64;
                    // constexpr int MaxTilesForParallel = 16;
                    constexpr int MinItersPerCU = 8;
                    // if (iters_per_tile >= MinItersForParallel && tiles <= MaxTilesForParallel)
                    if(reduction_strategy == reduction_type::Parallel)
                    {
                        size_t virt_cu_count = cu_count;
                        // TODO check if using occupancy info makes workspace too large
                        // if (occupancy > 1 && max_cus == 0)
                        //     virt_cu_count *= occupancy;

                        // Find max splitting factor to use as much of GPU as possible
                        size_t maxSplitsForTiles = virt_cu_count / tiles;

                        // Find max splitting factor to ensure each CU has a minimum number of iterations to do
                        size_t maxSplitsForIters = iters_per_tile / MinItersPerCU;

                        size_t maxSplits = std::min(maxSplitsForTiles, maxSplitsForIters);
                        sk_grid = tiles * maxSplits;
                    }
                    else
                    {
                        const std::vector<size_t> tile_fractions = {16, 12, 8, 6, 4, 3, 2, 1};
                        for(size_t frac: tile_fractions)
                        {
                            size_t splitGrid = tiles * frac;
                            size_t itersPerCU = iters_per_tile / frac;
                            if(splitGrid <= cu_count && itersPerCU >= 8)
                            {
                                sk_grid = splitGrid;
                                break;
                            }
                        }
                    }
                }

                if (tiles % sk_grid != 0 && tile_size * sk_grid > workspace_size)
                    sk_grid = tiles;
                return sk_grid;
            }
            // If no option is specified, launch exactly cu_count worth of workgroups.
            else
            {
                return cu_count;
            }
        }
    } // namespace streamk
}
