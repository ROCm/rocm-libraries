// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/gemm.hpp"
#include "origami/streamk.hpp"

#include <algorithm>
#include <chrono> // For timing
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <tuple>

namespace origami
{
    /* ---------------------------------------------------------------------------------------- */
    /* Misc. functions                                                                          */
    /* ---------------------------------------------------------------------------------------- */
    // Performs `(n + d - 1) / d`, but is robust against the case where `(n + d - 1)` would
    // overflow.
    template <typename N, typename D>
    constexpr N safe_ceil_div(N n, D d)
    {
        // Static cast to undo integral promotion.
        return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
    }

    // Computes the number of active compute units if there is only one wave and it is partial
    // Otherwise, returns hardware.N_CU
    std::tuple<size_t, size_t, size_t> compute_CU_occupancy(const hardware_t& hardware,
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
                                                            data_type_t     mi_datatype,
                                                            int             WGM,
                                                            size_t          workspace_size,
                                                            size_t          workspace_size_per_elem_c,
                                                            int             occupancy,
                                                            int             dynamic_grid_version,
                                                            size_t          split)
    {
        // Number of output MTs
        size_t numMT_M  = safe_ceil_div(M, MT_M);
        size_t numMT_N  = safe_ceil_div(N, MT_N);
        size_t numMTs   = numMT_M * numMT_N * batch;

        size_t numWGs, numActiveCUs, numWaves, splitFactor;
        if(split) // if it is given
        {
            split        = split > 1 ? split : 1;
            numWGs       = numMTs * split;
            numActiveCUs = numWGs < hardware.N_CU ? numWGs : hardware.N_CU;
            numWaves     = safe_ceil_div(numWGs, hardware.N_CU);
            splitFactor  = split;
        }
        else // as what StreamK predicts
        {
            streamk::reduction_type rt = streamk::select_reduction(M, 
                                                                            N, 
                                                                            K, 
                                                                            batch, 
                                                                            MT_M, 
                                                                            MT_N, 
                                                                            MT_K, 
                                                                            hardware, 
                                                                            dynamic_grid_version);
            numWGs = streamk::select_grid(M,
                                                    N,
                                                    K,
                                                    batch,
                                                    transA,
                                                    transB,
                                                    element_size_A,
                                                    element_size_B,
                                                    element_size_out,
                                                    mi_datatype,
                                                    workspace_size,
                                                    MT_M,
                                                    MT_N,
                                                    MT_K,
                                                    MI_M,
                                                    MI_N,
                                                    MI_K,
                                                    WGM,
                                                    workspace_size_per_elem_c,
                                                    occupancy,
                                                    hardware,
                                                    dynamic_grid_version,
                                                    rt);

            // output variables
            numActiveCUs = numWGs < hardware.N_CU ? numWGs : hardware.N_CU;
            // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
            // That means, we artifically observe one full wave, but that is not what actually happens
            // under the hood. From a theoretical point of view, these distributions change all of the
            // computations in Origami. With current implementation, it is hard to capture that behaviour
            // analytically.
            // So for now, if the numWGs is less than the numMTs, we calculate numWaves based on the 
            // numMTs. Otherwise, we use numWGs to compute numWaves.
            numWaves     = numWGs > numMTs ? 
                            safe_ceil_div(numWGs, hardware.N_CU) : 
                            safe_ceil_div(numMTs, hardware.N_CU);
            splitFactor  = safe_ceil_div(numWGs, numMTs);
        }

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("numMTs", numMTs);
            hardware.log_debug("numWGs", numWGs);
            hardware.log_debug("numActiveCUs", numActiveCUs);
            hardware.log_debug("numWaves", numWaves);
            hardware.log_debug("splitFactor", splitFactor);
        }

        return std::make_tuple(numActiveCUs, numWaves, splitFactor);
    }

    /* ---------------------------------------------------------------------------------------- */
    /* Compute-related functions                                                                */
    /* ---------------------------------------------------------------------------------------- */
    // Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
    size_t compute_number_matrix_instructions(const hardware_t& hardware,
                                                size_t          MT_M,
                                                size_t          MT_N,
                                                size_t          MT_K,
                                                size_t          MI_M,
                                                size_t          MI_N,
                                                size_t          MI_K)
    {
        // Compute the number of Matrix Instructions required in each dim
        size_t N_MI_M = safe_ceil_div(MT_M, MI_M);
        size_t N_MI_N = safe_ceil_div(MT_N, MI_N);
        size_t N_MI_K = safe_ceil_div(MT_K, MI_K);
        // Total number of matrix instructions for MT_MxMT_NxMT_K tile
        size_t N_MI = N_MI_M * N_MI_N * N_MI_K;

        return N_MI;
    }

    // Compute arithmic intensity
    double arithmetic_intensity(double m, double n, double k, double bytes_per_element)
    {
        // Numerator: 2.0 * m * n * k
        // Denominator: (m*n + n*k + m*k) * bytes_per_element
        double numerator   = 2.0 * m * n * k;
        double denominator = (m * n + n * k + m * k) * bytes_per_element;

        return numerator / denominator;
    }

    /**
     * Computes Emulated arithmetic intensity for TF32 (assumes 3xBF16).
     * 
     */
    double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element)
    {
        // Numerator: 3.0 * 2.0 * m * n * k
        // Denominator: (m*n + n*k + m*k) * bytes_per_element
        double numerator   = 3.0 * 2.0 * m * n * k;
        double denominator = (m * n + n * k + m * k) * bytes_per_element;

        return numerator / denominator;
    }

    // Compute cvt overhead in tf32 emulation
    static inline double compute_cvt_overhead(const hardware_t& hardware,
                                                size_t          MT_M,
                                                size_t          MT_N,
                                                size_t          MT_K,
                                                size_t          MI_M,
                                                size_t          MI_N,
                                                size_t          MI_K,
                                                size_t          element_size_A,
                                                size_t          element_size_B)
    {
        // Wave tile sizes
        // TODO: Use kernel's actual wavetiles.
        const double wave_tile_m = MT_M / 2.0;
        const double wave_tile_n = MT_N / 2.0;
        const double wave_tile_k = MT_K / MI_K;

        // MFMA count and cycles
        const double N_MI = (wave_tile_m / MI_M) * (wave_tile_n / MI_N) * wave_tile_k;

        // TF32 emu: 3× BF16 MI issue slots
        const double num_mfma = 3.0 * static_cast<double>(N_MI);

        // Cycle scale per MI (use BF16 MI latency as the basic timing quantum)
        const double L_MI_bf16 = hardware.get_mi_latency(MI_M, MI_N, MI_K, data_type_t::BFloat16);
        const double mfma_cycles = num_mfma * L_MI_bf16;

        // 2) Bytes (per K-slice), using ceil-div to whole bytes
        const double bytesA
            = static_cast<double>(wave_tile_m) * MT_K * safe_ceil_div(element_size_A, 8);
        const double bytesB
            = static_cast<double>(wave_tile_n) * MT_K * safe_ceil_div(element_size_B, 8);

        const double mt_bytesA
            = static_cast<double>(MT_M) * MT_K * safe_ceil_div(element_size_A, 8);

        // 3) Modeled transfer quanta (128B lines)
        //      dsA = bytesA / (128 * MI_M)
        //      dsB = bytesB / (128 * MI_N)
        //      GR  = dsA  (global->LDS modeled equal to A-side DS)
        const double dsA = (bytesA / 128.0) / static_cast<double>(MI_M); // LDS->VGPR for A
        const double dsB = (bytesB / 128.0) / static_cast<double>(MI_N); // LDS->VGPR for B
        const double GR  = dsA; // Global->LDS reads
        const double LR  = dsA + dsB; // total DS->VGPR

        // 4) Heuristic cycle weights (scaled to MI latency).
        //    Preserves your A=104, B=8, C=4 when L_MI_bf16 == 16.
        // 24 vector instructions per 2 ds_reads (16x16x32)
        // 24 vector instructions per 2 ds_reads for A and for B.
        // 3 instructions per fp32 value read; number ds_read * size
        const double A = (104.0 / 16.0) * L_MI_bf16; // CVT per LR-sized chunk (DS->VGPR)
        const double B = (8.0 / 16.0) * L_MI_bf16; // hidden per spare MFMA slot
        // MI16: 16 - 4 (12 cycles), for those 4 cycles, VGPRs are locked. 8 cycles to do anything.
        const double C = (4.0 / 16.0) * L_MI_bf16; // hidden per (LR+GR) slot     // MI16
        // 32 cycles (mfma), 4 cycles, 28, 4 vgpr lock, 24 cycles left.
        // 24: 6 conv instructions, 3 ds_reads, ~6 grs

        // 5) Exposed vs hidden CVT
        const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
        const double cvt        = A * dsA; // only DS->VGPR contributes CVT
        const double H          = B * spare_mfma + C * (LR + GR); // hidden cycles
        const double overhead   = std::max(cvt - H, 0.0);

        // 6) Efficiency
        const double denom = mfma_cycles + overhead;
        const double eff   = (denom > 0.0) ? (mfma_cycles / denom) : 1;

        return overhead;
    }

    // Determine the compute latency per MT_MxMT_NxMT_K Macro Tile (L_MT).
    size_t compute_mt_compute_latency(const hardware_t& hardware,
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
                                        data_type_t     mi_datatype)
    {
        // Compute the number of matrix instructions
        size_t N_MI = compute_number_matrix_instructions(
            hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K);
        // Latency of a single MT_MxMT_NxMT_k tile is the latency of one MI multiplied by
        // number of MI per MT_MxMT_NxMT_k.
        size_t L_MI = hardware.get_mi_latency(MI_M, MI_N, MI_K, mi_datatype);

        // size_t mt_arith = arithmetic_intensity(MT_M, MT_N, MT_K, 2);
        // printf("MT_M:%d MT_N:%d MT_K:%d arith:%d\n", MT_M, MT_N, MT_K, mt_arith);
        // size_t arith = ((M * N * K * 2) / (M * K + N * K + M * N));
        size_t L_MT = L_MI * N_MI;

        // TN
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

        // NT: A is contiguous in M and B is contiguous in N
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
            //NT Transpose Overhead Scales in both.
        }

        // TT: A is contiguous in K and B is contiguous in N
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

        // NN: A is contiguous in M and B is contiguous in K
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

    /* ---------------------------------------------------------------------------------------- */
    /* Memory-related functions                                                                 */
    /* ---------------------------------------------------------------------------------------- */
    // Check if MT fits in LDS
    bool check_lds_capacity(const hardware_t& hardware,
                            size_t          MT_M,
                            size_t          MT_N,
                            size_t          MT_K,
                            size_t          element_size)
    {
        // A and B size
        size_t Ld_A_value = compute_A_loads(MT_M, MT_K);
        size_t Ld_B_value = compute_B_loads(MT_N, MT_K);
        // Size of those in bytes
        size_t LDS_usage = (Ld_A_value + Ld_B_value) * (element_size / 8);

        if(LDS_usage > hardware.lds_capacity)
        {
            return false; // Exceeds LDS capacity
        }
        else
        {
            return true; // Within LDS capacity
        }
    }

    // Compute the amount of data loaded from A to produce a MT_MxMT_NxMT_K tile.
    size_t compute_A_loads(size_t MT_M, size_t MT_K)
    {
        // Compute the size of loads from A for a single MT_MxMT_NxMT_K tiles
        size_t Ld_A_value = MT_M * MT_K;

        return Ld_A_value;
    }

    // Compute the amount of data loaded from B to produce a MT_MxMT_NxMT_K tile.
    size_t compute_B_loads(size_t MT_N, size_t MT_K)
    {
        // Compute the size of loads from B for a single MT_MxMT_NxMT_K tiles
        size_t Ld_B_value = MT_N * MT_K;

        return Ld_B_value;
    }

    // Compute limited achievable memory bandwidth based on active CUs
    double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t numActiveCUs)
    {
        const double CUs = static_cast<double>(numActiveCUs);

        if(numActiveCUs > hardware.N_CU)
            return 1.0;

        const double bw_limited = std::get<0>(hardware.mem_bw_per_wg_coefficients) * CUs * CUs
                                + std::get<1>(hardware.mem_bw_per_wg_coefficients) * CUs
                                + std::get<2>(hardware.mem_bw_per_wg_coefficients);

        return std::min(bw_limited, 1.0);
    }

    // Estimate L2 hit-rate
        double estimate_l2_hit(const hardware_t& hardware,
                            size_t          M,
                            size_t          N,
                            size_t          K,
                            size_t          batch,
                            size_t          MT_M,
                            size_t          MT_N,
                            size_t          MT_K,
                            size_t          element_size,
                            int             WGM,
                            size_t          splittingFactor)
        {
            // Compute grid dimensions
            int grid_m = static_cast<int>(safe_ceil_div(M, MT_M));
            int grid_n = static_cast<int>(safe_ceil_div(N, MT_N));

            // Distribute CUs per XCD
            // Modify cu_per_xcd to only take into account the CUs that might share same K-tiles
            // This is to factor in the effect of splitting on L2
            int cu_per_xcd = safe_ceil_div(grid_m * grid_n, hardware.NUM_XCD);
            cu_per_xcd /= splittingFactor;

            // N dimension of mem1 tile is divided by whichever is smaller between WGM and grid
            int l2_n = std::min(WGM, grid_n);
            int l2_m = cu_per_xcd / l2_n;

            // If a single mem1 tile is larger than the grid, extend M dimension
            if(l2_m > grid_m)
            {
                int num_wraps   = (l2_m / grid_m) - 1; // how many times we wrap
                l2_n += (num_wraps * WGM);
                l2_m = grid_m;
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

            if(hardware_t::is_debug_enabled())
            {
                hardware.log_debug("L2Tile_M", l2_m);
                hardware.log_debug("L2Tile_N", l2_n);
            }

            return l2_hit;
        }

    // Estimate MALL hit-rate
    double estimate_mall_hit(const hardware_t& hardware,
                                size_t          M,
                                size_t          N,
                                size_t          K,
                                size_t          batch,
                                size_t          MT_M,
                                size_t          MT_N,
                                size_t          MT_K,
                                int             WGM,
                                size_t          numActiveCUs,
                                size_t          splittingFactor)
    {
        int grid_m = static_cast<int>(safe_ceil_div(M, MT_M));
        int grid_n = static_cast<int>(safe_ceil_div(N, MT_N));

        // mem2 tile dimensions
        int mall_m = grid_m * grid_n / WGM;         // M dimension of mem2 tile
        int mall_n = std::min(WGM, grid_n); // N dimension of mem2 tile

        // If a single mem2 tile is larger than the grid, extend its M dimension
        if(mall_m > grid_m)
        {
            int num_wraps   = (mall_m / grid_m) - 1;
            mall_n += (num_wraps * WGM);
            mall_m = grid_m;
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

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("MallTile_M", mall_m);
            hardware.log_debug("MallTile_N", mall_n);
        }

        return mall_hit;
    }

    // Determine the memory latency
    double compute_memory_latency(const hardware_t& hardware,
                                    size_t          M,
                                    size_t          N,
                                    size_t          K,
                                    bool            transA,
                                    bool            transB,
                                    size_t          batch,
                                    size_t          MT_M,
                                    size_t          MT_N,
                                    size_t          MT_K,
                                    size_t          element_size_A,
                                    size_t          element_size_B,
                                    size_t          mx_block_size,
                                    int             WGM,
                                    size_t          numActiveCUs,
                                    size_t          splittingFactor)
    {
        // 1) Estimate L2 hit-rate
        double H_mem1
            = estimate_l2_hit(hardware, M, N, K, batch, MT_M, MT_N, MT_K, element_size_A, WGM, splittingFactor);

        // 2) Estimate mall hit-rate
        double H_mem2
            = estimate_mall_hit(hardware, M, N, K, batch, MT_M, MT_N, MT_K, WGM, numActiveCUs, splittingFactor);

        // 3) Total loads are loads from A and loads from B
        size_t Ld_A_value  = compute_A_loads(MT_M, MT_K);
        size_t Ld_B_value  = compute_B_loads(MT_N, MT_K);
        size_t Ld_CU_bytes = (Ld_A_value * safe_ceil_div(element_size_A, 8)) // A Bytes
                                + (Ld_B_value * safe_ceil_div(element_size_B, 8)); //B Bytes

        // Logic for block scaled datatypes (Assuming BS=32 and 8-bit scales)
        // TODO This is technically wrong, need separate flag to enable MX so we can differentiate FP8 and MX8
        if(element_size_A < 8 && mx_block_size != 0)
        {
            // Number of scales per tile
            size_t num_scales_A = safe_ceil_div(MT_M * MT_K, mx_block_size);
            Ld_CU_bytes += num_scales_A; //One Byte per scale
        }
        if(element_size_B < 8 && mx_block_size != 0)
        {
            // Number of scales per tile
            size_t num_scales_B = safe_ceil_div(MT_N * MT_K, mx_block_size);
            Ld_CU_bytes += num_scales_B; //One Byte per scale
        }

        // 4) total loads by all CUs
        double total_Ld = Ld_CU_bytes * static_cast<double>(numActiveCUs);

        // 5) mem1‐limited factor (simple linear model)
        double mem1_bw_limited = static_cast<double>(numActiveCUs) / static_cast<double>(hardware.N_CU);
        double limited_mem1_bw = hardware.mem1_perf_ratio * mem1_bw_limited;

        // 6) mem1 latency
        double L_mem_mem1 = (limited_mem1_bw > 0) ? (total_Ld / (limited_mem1_bw)) : 0.0;

        // 7) mem2‐limited from occupancy (Can't Issue enough load/stores)
        double bw_limited = compute_mem_bw_from_occupancy(hardware, numActiveCUs);

        // 8) loads that reach each level
        double Ld_mem2 = (1.0 - H_mem1) * total_Ld;
        double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

        // 9) enforce whole‐problem minimum loads
        if(numActiveCUs < hardware.N_CU)
        {
            double min_load
                = static_cast<double>(M * MT_K * splittingFactor * safe_ceil_div(element_size_A, 8)
                                        + N * MT_K * splittingFactor  * safe_ceil_div(element_size_B, 8));
            Ld_MEM  = std::max(Ld_MEM, min_load) * batch;
            Ld_mem2 = std::max(Ld_mem2, min_load) * batch;
        }

        // 10) mem2 latency
        double limited_mem2_bw = hardware.mem2_perf_ratio * bw_limited;
        double L_mem_mem2 = (limited_mem2_bw > 0) ? (Ld_mem2 / limited_mem2_bw) : 0.0;

        // 11) MEM latency
        double limited_mem_bw = hardware.mem3_perf_ratio * bw_limited;
        double L_mem_MEM      = (limited_mem_bw > 0) ? (Ld_MEM / limited_mem_bw) : 0.0;
        L_mem_MEM += 200; // Load Latency

        // 12) pick the worst‐case bound
        double L_mem = std::max({L_mem_mem1, L_mem_mem2, L_mem_MEM});

        // NT
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

        // TT : A is contiguous in K and B is contiguous in N
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

        // NN : A is contiguous in M and B is contiguous in K
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

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("mem1_perf_ratio", hardware.mem1_perf_ratio);
            hardware.log_debug("mem2_perf_ratio", hardware.mem2_perf_ratio);
            hardware.log_debug("mem3_perf_ratio", hardware.mem3_perf_ratio);
            hardware.log_debug("mem_bw_per_wg_coefficients(0)", std::get<0>(hardware.mem_bw_per_wg_coefficients));
            hardware.log_debug("mem_bw_per_wg_coefficients(1)", std::get<1>(hardware.mem_bw_per_wg_coefficients));
            hardware.log_debug("mem_bw_per_wg_coefficients(2)", std::get<2>(hardware.mem_bw_per_wg_coefficients));
            hardware.log_debug("H_mem1 (mem1 hit ratio)", H_mem1);
            hardware.log_debug("H_mem2 (mem2 hit ratio)", H_mem2);
            hardware.log_debug("Total Load (bytes)", total_Ld);
            hardware.log_debug("Ld_mem2 (bytes)", Ld_mem2);
            hardware.log_debug("Ld_MEM (bytes)", Ld_MEM);
            hardware.log_debug("L_mem_mem1 (cycles)", L_mem_mem1);
            hardware.log_debug("L_mem_mem2 (cycles)", L_mem_mem2);
            hardware.log_debug("L_mem_MEM (cycles)", L_mem_MEM);
        }

        return L_mem;
    }

    /* ---------------------------------------------------------------------------------------- */
    /* Tile-related functions                                                                   */
    /* ---------------------------------------------------------------------------------------- */
    double compute_tile_latency(const hardware_t& hardware,
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
                                data_type_t     mi_datatype,
                                size_t          mx_block_size,
                                int             WGM,
                                size_t          numActiveCUs,
                                size_t          splittingFactor)
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
                                                        mi_datatype);

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
                                                element_size_A,
                                                element_size_B,
                                                mx_block_size,
                                                WGM,
                                                numActiveCUs,
                                                splittingFactor);

        // 2) Work-group setup & iteration latencies
        double L_WG_setup = 1; // WG_setup_Latency

        // 3) Prologue: 2.2× memory latency
        double L_prologue = 1.5 * L_mem; // 1.5 chosen emprically

        // 4) Epilogue: writes from all active CUs with limited bandwidth
        double epilogue_limite = (static_cast<double>(numActiveCUs) / hardware.N_CU);
        double limited_mem1 = hardware.mem1_perf_ratio * epilogue_limite;
        if(limited_mem1 < 1)
        {
            limited_mem1 = 10;
        }
        double L_epilogue = (static_cast<double>(numActiveCUs)
                                * MT_M * MT_N * safe_ceil_div(element_size_out, 8))
                                / limited_mem1;

        // 4') K-split reductions are globally coherent, we need to write and read split-1 MT_M*MT_N tiles to coherent memory
        if(splittingFactor > 1)
        {
            size_t n_partials              = splittingFactor - 1;
            double partial_readwrite_bytes = (2 * numActiveCUs
                                                * MT_M * MT_N * safe_ceil_div(element_size_out, 8)
                                                * n_partials);
            double L_reduce = partial_readwrite_bytes / (hardware.mem3_perf_ratio);
            L_epilogue += L_reduce * 1;
        }

        // 4'') tf32 emu has some more overhead
        double L_cvt    = 0;
        bool   tf32_emu = ((mi_datatype == data_type_t::XFloat32)
                            && (hardware.arch == hardware_t::architecture_t::gfx950));
        if(tf32_emu)
        {
            L_cvt = compute_cvt_overhead(hardware,
                                            MT_M,
                                            MT_N,
                                            MT_K,
                                            MI_M,
                                            MI_N,
                                            MI_K,
                                            element_size_A,
                                            element_size_B);
        }

        // 5) Single-tile latency (always additive)
        double L_tile_single = std::max(L_compute, L_mem) + L_cvt;

        // 6) Number of K-iterations (excluding epilogue), at least 1
        // long num_iter = static_cast<long>(((K + MT_K - 1) / MT_K)) - 1;
        // num_iter      = std::ceil(num_iter / splittingFactor);
        // num_iter      = std::max(num_iter, 1L);
        long splittedK = static_cast<long>(safe_ceil_div(K, splittingFactor));
        long num_iter = static_cast<long>(safe_ceil_div(splittedK, MT_K)) - 1;

        // 7) Total tile latency
        double L_tile_total = (L_tile_single * num_iter)
                                + L_prologue
                                + L_epilogue
                                + L_WG_setup
                                + (28 * num_iter); // 7 instructions (each with 4 cycles) at the end of the loop

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("L_compute", L_compute);
            hardware.log_debug("L_mem", L_mem);
            hardware.log_debug("L_cvt", L_cvt);
            hardware.log_debug("L_tile_single", L_tile_single);
            hardware.log_debug("num_iter", num_iter);
            hardware.log_debug("L_prologue", L_prologue);
            hardware.log_debug("L_epilogue", L_epilogue);
            hardware.log_debug("L_tile_total", L_tile_total);
        }

        return L_tile_total;
    }

    // Computes the latency per K-complete MT wave
    // A wave is defined as : The time it takes for one CU to complete one K-complete output tile
    double compute_wave_latency(const hardware_t& hardware,
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
                                data_type_t     mi_datatype,
                                size_t          mx_block_size,
                                int             WGM,
                                size_t          numActiveCUs,
                                size_t          splittingFactor)
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
                                                element_size_A,
                                                element_size_B,
                                                element_size_out,
                                                mi_datatype,
                                                mx_block_size,
                                                WGM,
                                                numActiveCUs,
                                                splittingFactor);

        return L_wave;
    }

    // Compute the total latency of a gemm based on the latency of one wave multiplied by the number of waves
    // A wave is defined as : The time it takes for one CU to complete one K-complete output tile
    double compute_total_latency(const hardware_t& hardware,
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
                                    data_type_t     mi_datatype,
                                    size_t          mx_block_size,
                                    int             WGM,
                                    size_t          split)
    {
        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("Problem_Size",
                                std::to_string(int(M)) + "x" + std::to_string(int(N)) + "x"
                                    + std::to_string(int(K)));
            hardware.log_debug("Macro_Tile",
                                std::to_string(int(MT_M)) + "x" + std::to_string(int(MT_N)) + "x"
                                    + std::to_string(int(MT_K)));
            hardware.log_debug("Element Size A (bits)", element_size_A);
            hardware.log_debug("Element Size B (bits)", element_size_B);
        }

        // 0) Short-circuit
        // We don't need to compute latency for all MTs. With this, we can shortcut.
        bool shortCircuit = true;
        if(shortCircuit)
        {
            // When problem dimensions are small enough that we can fit them in one tile, we should do so.
            // This short circuit condition also decreases selection latency when problems are very small :)
            // TODO 256 and 256 here should be largest M and N tile dimensions in library
            if(M <= 256 && N <= 256 && K < 1024 && batch != 1 && (MT_M < M || MT_N < N))
                return std::numeric_limits<double>::max();

                // Override dot2 instruction with vector lane widths
            if(MI_N == 0 && MI_M == 0 && MI_K == 0)
            {
                // We only use Dot2 for NN layout where M < 3
                if(M > 2 || transA || transB)
                    return std::numeric_limits<double>::max();
                MI_M = 1;
                MI_N = 1;
                MI_K = 64;
            }
        }

        // 1-1) WGM
        WGM = std::max(WGM, 1); // WGM can't be less than one.

        // 1-2) Find CU occupancy
        auto [numActiveCUs, numWaves, splittingFactor] = compute_CU_occupancy(hardware,
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
                                                                                WGM,
                                                                                std::numeric_limits<size_t>::max(), // workspace
                                                                                std::numeric_limits<size_t>::max(), // workspace per c
                                                                                0, // occupancy
                                                                                6, // dynamic_grid
                                                                                split);

        // 2) Compute latency of a wave
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
                                                element_size_A,
                                                element_size_B,
                                                element_size_out,
                                                mi_datatype,
                                                mx_block_size,
                                                WGM,
                                                numActiveCUs,
                                                splittingFactor);
        // Compute latency for all waves and return it as the latency for the MT/problem
        double total_latency = L_wave * numWaves;

        // 3) Customized heuristics
        // TODO These are quantifying effects that don't work in the current math.
        // TODO THESE SHOULD BE TEMPORARY FIXES AND BE MORE SOLIDLY INTEGRATED LATER
        bool heuristics = hardware_t::is_heuristics_enabled();

        // Heuristics for TF32
        bool tf32_emu = ((mi_datatype == data_type_t::XFloat32)
                        && (hardware.arch == hardware_t::architecture_t::gfx950));
        if(tf32_emu && heuristics)
        {
            double bytes_per_element = static_cast<double>(element_size_A) / 8.0;
            double arith = emulated_tf32_arithmetic_intensity(M, N, K, bytes_per_element);
            double compute_threshold = 1000; // threshold empirically determined.

            // The kernel for this is more optimized (Custom kernel NT)
            if((!transA && transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
            {
                if (arith < compute_threshold)
                    total_latency = total_latency * 0.6;
                else
                    total_latency = total_latency * 0.4;
            }

            // The kernel for this is more optimized (Custom kernel NN)
            if((!transA && !transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
            {
                if (arith < compute_threshold)
                    total_latency = total_latency * 0.8;
                else
                    total_latency = total_latency * 0.4;
            }

            // The kernel for this is more optimized (Custom kernel TN)
            if((transA && !transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
            {
                if (arith < compute_threshold)
                    total_latency = total_latency * 0.8;
                else
                    total_latency = total_latency * 0.4;
            }

            // Bias large DU where K-dimension is large and M and N are small.
            if((K >= (M * 16) && K >= (N * 16)) && (MT_K >= 128))
            {
                total_latency = total_latency * 0.5;
            }
        }

        if(heuristics && !tf32_emu)
        {
            // Penalize tiles that lead to edge waste
            const size_t numMT_M = safe_ceil_div(M, MT_M);
            const size_t numMT_N = safe_ceil_div(N, MT_N);
            const double waste = static_cast<double>(numMT_M * MT_M * numMT_N * MT_N) / (M * N);
            double edge_penalty = std::pow(waste, 0.8);
            if(batch > 10)
                edge_penalty = std::pow(waste, 1.5) * std::log10((double)batch) * std::pow((double)numMT_M * numMT_N, 0.2);
            total_latency = total_latency * edge_penalty;

            // Penalize K iterations 
            size_t K_iters = safe_ceil_div(K, MT_K);
            if(K_iters <= 2)
            {
                total_latency = total_latency * 8;
            }
            else if(K_iters <= 4)
            {
                total_latency = total_latency * 4;
            }
            else if(K_iters <= 8)
            {
                total_latency = total_latency * 2.1;
            }

            // Bias toward not splitting for small K values
            // This should actually come from SK grid prediction
            if(splittingFactor > 1 && K < 2048)
            {
                total_latency = total_latency * splittingFactor;
            }

            // There is no case where a kernel with MT_K > K wins unless K < MI_K.
            // Unless it is Dot2.
            if(K < MT_K && MI_M != 1)
                total_latency = total_latency * (MT_K - K);

            // Bias Model towards at least one dim being power of 2
            bool MT_M_is_power_two = (MT_M > 0) && (MT_M & (MT_M - 1)) == 0;
            bool MT_N_is_power_two = (MT_N > 0) && (MT_N & (MT_N - 1)) == 0;
            if(!MT_M_is_power_two && !MT_N_is_power_two)
            {
                total_latency = total_latency * 1.1;
            }

            // Bias Model towards both dims being a power of 2
            if(MT_M_is_power_two && MT_N_is_power_two)
            {
                total_latency = total_latency * 0.9;
            }

            // Bias toward 512 tiles for sizes "very skinny" sizes
            // "very skinny" definition: either N or M less than 16 (1 tile) and the other one requires
            // more than 100 waves (100*numCUs tiles)
            if(M < 16 && N > 100 * hardware.N_CU * 512 && MT_N == 512)
            {
                total_latency = total_latency * 0.25;
            }
            if(N < 16 && M > 100 * hardware.N_CU * 512 && MT_M == 512)
            {
                total_latency = total_latency * 0.25;
            }

            // DOT2 Kernels
            if(MI_M == 1 && MI_N == 1 && MI_K == 64)
            {
                // Bias DOT2 kernels in which the tile dimensions in M and K are equal to the problem dimensions
                if(MT_M == M || MT_K == K)
                {
                    total_latency = total_latency * 0.8;
                }
            }

            // Heuristics for FP16
            if(element_size_A == 16)
            {
                // These kernels are more optimized (Custom kernels)
                // All layouts
                if(MT_M == 256 && MT_N == 256 && MT_K == 64)
                {
                    total_latency = total_latency * 0.85;
                    if((transA && !transB) && (M == MT_M && N > 256 * MT_N && K >= 4*MT_K))
                    {
                        total_latency = total_latency * 0.3;
                    }
                }

                // The kernel for this is less optimized, for some reason
                if(MT_M == 256 && MT_N == 16 && MT_K == 128)
                {
                    total_latency = total_latency * 2;
                }

                // The kernel for this is less optimized, for some reason
                if(MT_M == 16 && MT_N == 256 && MT_K == 128)
                {
                    total_latency = total_latency * 2;
                }
            }

            // Heuristics for FP8
            if(element_size_A == 8)
            {
                // The kernel for this is more optimized (Custom kernel)
                if(transA && !transB && MT_M == 256 && MT_N == 256 && MT_K == 128)
                {
                    total_latency = total_latency * 0.8;
                }

                // Bias towards dimensions divisible by 64 for 8-bit datatypes
                if((MT_M > 64) && (MT_M % 64 != 0))
                {
                    total_latency = total_latency * 1.2;
                }
                if((MT_N > 64) && (MT_N % 64 != 0))
                {
                    total_latency = total_latency * 1.2;
                }
            }
        }

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("Total_latency (with heuristics)", total_latency);

            hardware.print_debug_info();
        }

        return total_latency;
    }

    // Compute the performance from the latency.
    // IMPORTANT : This program is NOT meant to be an analytical model for performance, but rather a way to rank different macro tile sizes.
    // These performance values could be wildly inaccurate in absolute terms, but will often result in the correct ranking of MTin relative terms.
    double compute_perf_gflops(const hardware_t& hardware,
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
                                data_type_t     mi_datatype,
                                int             WGM)
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
                                                            element_size_A,
                                                            element_size_B,
                                                            element_size_out,
                                                            mi_datatype,
                                                            mx_block_size,
                                                            WGM);
        double total_time_seconds = latency_cycles / cycles_per_second;
        // Compute performance in FLOPS
        double FLOPS = total_FLOPs / total_time_seconds;
        // Convert to TFLOPS
        double GFLOPS = FLOPS / 1e9; // 1 TFLOP = 1e9 FLOPs
        return GFLOPS;
    }
} // namespace origami
