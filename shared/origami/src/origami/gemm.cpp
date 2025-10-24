// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/gemm.hpp"

#include "origami/streamk.hpp"
#include <algorithm>
#include <chrono> // For timing
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
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

    auto lds128_penalty(size_t dim_elems, size_t element_bits)
    {
        const size_t bytes = dim_elems * safe_ceil_div(element_bits, 8);
        const size_t mod   = bytes % 128;
        if(mod == 0)
            return 1.0;
        const double frac = double(mod) / 128.0; // 0..1
        const double base = (element_bits <= 16) ? 1.1 : 1.35; // BF16/FP16 < FP32
        return 1.0 + base * frac; // up to ~2.35x worst case
    };

    double calculate_work_utilization(
        size_t M, size_t N, size_t K, size_t MT_M, size_t MT_N, size_t MT_K)
    {
        if(M == 0 || N == 0 || K == 0 || MT_M == 0 || MT_N == 0 || MT_K == 0)
            return 1.0;

        // Calculate the full dimensions covered by the launched grid of tiles (spatial).
        const double launched_M = static_cast<double>(safe_ceil_div(M, MT_M)) * MT_M;
        const double launched_N = static_cast<double>(safe_ceil_div(N, MT_N)) * MT_N;

        // Calculate the full depth covered by the k-loop iterations (temporal).
        const double launched_K = static_cast<double>(safe_ceil_div(K, MT_K)) * MT_K;

        // The utilization is the ratio of the useful problem volume to the total scheduled volume.
        const double useful_volume   = static_cast<double>(M * N * K);
        const double launched_volume = launched_M * launched_N * launched_K;

        if(launched_volume < 1.0)
            return 1.0; // Avoid division by zero for tiny/empty problems

        const double utilization = useful_volume / launched_volume;

        return utilization;
    }

    double calculate_output_utilization(
        size_t M, size_t N, size_t MT_M, size_t MT_N, size_t vector_elems = 1)
    {
        if(M == 0 || N == 0 || MT_M == 0 || MT_N == 0)
            return 1.0;

        // Tiled coverage in M/N
        const double launched_M = static_cast<double>(safe_ceil_div(M, MT_M)) * MT_M;
        const double launched_N = static_cast<double>(safe_ceil_div(N, MT_N)) * MT_N;

        // Optional: model vectorization/alignment remainders (e.g., ld/st width)
        // This assumes vectors must be fully inside bounds; tail elements are scalarized.
        const size_t M_vec = (vector_elems > 1) ? (M / vector_elems) * vector_elems : M;
        const size_t N_vec = (vector_elems > 1) ? (N / vector_elems) * vector_elems : N;

        const double useful   = static_cast<double>(M_vec) * static_cast<double>(N_vec);
        const double launched = launched_M * launched_N;

        if(launched < 1.0)
            return 1.0;
        return useful / launched;
    }

    // Computes the number of active compute units if there is only one wave and it is partial
    // Otherwise, returns hardware.N_CU
    std::tuple<size_t, size_t, size_t> compute_CU_occupancy(const hardware_t& hardware,
                                                            size_t            M,
                                                            size_t            N,
                                                            size_t            K,
                                                            size_t            batch,
                                                            bool              transA,
                                                            bool              transB,
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
                                                            int               WGM,
                                                            size_t            workspace_size,
                                                            size_t workspace_size_per_elem_c,
                                                            int    occupancy,
                                                            int    dynamic_grid_version,
                                                            size_t split,
                                                            size_t max_cus = 0)
    {
        // Number of output MTs
        size_t numMT_M = safe_ceil_div(M, MT_M);
        size_t numMT_N = safe_ceil_div(N, MT_N);
        size_t numMTs  = numMT_M * numMT_N * batch;

        size_t numWGs, numActiveCUs, numWaves, splitFactor;

        if(split) // if it is given
        {
            split        = split > 1 ? split : 1;
            numWGs       = numMTs * split;
            numActiveCUs = numWGs < hardware.N_CU ? numWGs : hardware.N_CU;
            numWaves     = safe_ceil_div(numWGs, hardware.N_CU);
            splitFactor  = split;

            if(hardware_t::is_debug_enabled())
            {
                hardware.log_debug("reduction type", "Origami");
            }
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
                                          rt,
                                          max_cus);

            // output variables
            numActiveCUs = numWGs < hardware.N_CU ? numWGs : hardware.N_CU;
            // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
            // That means, we artifically observe one full wave, but that is not what actually happens
            // under the hood. From a theoretical point of view, these distributions change all of the
            // computations in Origami. With current implementation, it is hard to capture that
            // behaviour analytically. So for now, if the numWGs is less than the numMTs, we calculate
            // numWaves based on the numMTs. Otherwise, we use numWGs to compute numWaves.
            numWaves    = numWGs > numMTs ? safe_ceil_div(numWGs, hardware.N_CU)
                                          : safe_ceil_div(numMTs, hardware.N_CU);
            splitFactor = safe_ceil_div(numWGs, numMTs);

            if(hardware_t::is_debug_enabled())
            {
                hardware.log_debug("reduction type", streamk::rtype_to_string(rt));
            }
        }

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("numMTs", numMTs);
            hardware.log_debug("numWGs", numWGs);
            hardware.log_debug("numActiveCUs", numActiveCUs);
            hardware.log_debug("numWaves", numWaves);
            hardware.log_debug("splitFactor", splitFactor);
            hardware.log_debug("max_cus", max_cus);
        }

        return std::make_tuple(numActiveCUs, numWaves, splitFactor);
    }

    /* ---------------------------------------------------------------------------------------- */
    /* Compute-related functions                                                                */
    /* ---------------------------------------------------------------------------------------- */
    // Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
    size_t compute_number_matrix_instructions(const hardware_t& hardware,
                                              size_t            MT_M,
                                              size_t            MT_N,
                                              size_t            MT_K,
                                              size_t            MI_M,
                                              size_t            MI_N,
                                              size_t            MI_K)
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

    // Computes Emulated arithmetic intensity for TF32 (assumes 3xBF16).
    double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element)
    {
        // Numerator: 3.0 * 2.0 * m * n * k
        // Denominator: (m*n + n*k + m*k) * bytes_per_element
        double numerator   = 3.0 * 2.0 * m * n * k;
        double denominator = (m * n + n * k + m * k) * bytes_per_element;

        return numerator / denominator;
    }

    // Compute cvt overhead in x1 tf32 emulation
    // TODO: We can generalize the same routine to cover more GEMMs that perform conversion
    static inline double compute_cvt_overhead_x1(const hardware_t& hardware,
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
        // In X1 TF32 GEMMs, we do:
        // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
        // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
        // ds_write_b64
        // That is, the extra instructions that we need to account for are the two cvt_pk ops
        // per wave tile

        // However, these extra ops should not be added up to the overal tile latency becuase
        // they can be run in parallel to Matix and Memory operations (given they are not dependent).
        // So, We should ideally take L_tile = max{Mem, Comp, Vec (cvt latencies)}.
        // Since, Vec latency is not modeled yet, we somehow model that into the current logic
        // by scaling according to MFMA latencies and putting some heuristics to model the fact
        // that these vector operations can be hidden (read interleaved) with the other memory
        // or MFMA instructions.

        // TODO: Use kernel's actual wavetiles.
        const double wave_tile_m = MT_M / 2.0;
        const double wave_tile_n = MT_N / 2.0;
        const double wave_tile_k = MT_K / MI_K;

        // MFMA count
        const double N_MI = (wave_tile_m / MI_M) * (wave_tile_n / MI_N) * wave_tile_k;
        const double num_mfma = 1.0 * static_cast<double>(N_MI);
        // Cycle scale per MI
        const double L_MI = hardware.get_mi_latency(MI_M, MI_N, MI_K, mi_datatype);
        const double mfma_cycles = num_mfma * L_MI;

        // 2) Bytes (per K-slice), using ceil-div to whole bytes
        const double bytesA
            = static_cast<double>(wave_tile_m) * MT_K * safe_ceil_div(element_size_A, 8);
        const double bytesB
            = static_cast<double>(wave_tile_n) * MT_K * safe_ceil_div(element_size_B, 8);

        // 3) Modeled transfer quanta (128B lines)
        //      dsA = bytesA / (128 * MI_M)
        //      dsB = bytesB / (128 * MI_N)
        //      GR  = dsA  (global->LDS modeled equal to A-side DS)
        const double dsA = (bytesA / 128.0) / static_cast<double>(MI_M); // LDS->VGPR for A
        const double dsB = (bytesB / 128.0) / static_cast<double>(MI_N); // LDS->VGPR for B
        const double GR  = dsA; // Global->LDS reads
        const double LR  = dsA + dsB; // total DS->VGPR

        // 5) Exposed vs hidden CVT
        // spare MFMA
        const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
        // 2 cvt per each ds_write (this for SS_BSS -- should be revised for other datatypes)
        // Each cvt has a latency of four. It is scaled by the MI Latency
        // Note: change 16.0 based on mi_data_type if we want to generalize this for all
        // casting GEMMs.
        const double cvt        = (2.0 * 4.0 / 16.0 * L_MI) * LR; 
        // cvt ops are interleaved in main loop and don't stall matrix or memory units.
        // Heuristically, we set
        const double H          = (8.0 / 16.0 * L_MI) * spare_mfma + (4.0 / 16.0) * L_MI * (LR + GR);
        const double overhead   = std::max(cvt - H, 0.0);

        return overhead;
    }

    // Compute cvt overhead in tf32 emulation
    static inline double compute_cvt_overhead(const hardware_t& hardware,
                                              size_t            MT_M,
                                              size_t            MT_N,
                                              size_t            MT_K,
                                              size_t            MI_M,
                                              size_t            MI_N,
                                              size_t            MI_K,
                                              size_t            element_size_A,
                                              size_t            element_size_B)
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
        //const double mfma_cycles = num_mfma * L_MI_bf16;

        // 2) Bytes (per K-slice), using ceil-div to whole bytes
        const double bytesA
            = static_cast<double>(wave_tile_m) * MT_K * safe_ceil_div(element_size_A, 8);
        const double bytesB
            = static_cast<double>(wave_tile_n) * MT_K * safe_ceil_div(element_size_B, 8);

        // const double mt_bytesA
        //     = static_cast<double>(MT_M) * MT_K * safe_ceil_div(element_size_A, 8);

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
        //const double denom = mfma_cycles + overhead;
        //const double eff   = (denom > 0.0) ? (mfma_cycles / denom) : 1;

        return overhead;
    }

    // Determine the compute latency per MT_MxMT_NxMT_K Macro Tile (L_MT).
    size_t compute_mt_compute_latency(const hardware_t& hardware,
                                      size_t            M,
                                      size_t            N,
                                      size_t            K,
                                      bool              transA,
                                      bool              transB,
                                      size_t            MT_M,
                                      size_t            MT_N,
                                      size_t            MT_K,
                                      size_t            MI_M,
                                      size_t            MI_N,
                                      size_t            MI_K,
                                      size_t            element_size_A,
                                      size_t            element_size_B,
                                      data_type_t       mi_datatype)
    {
        // Compute the number of matrix instructions
        size_t N_MI
            = compute_number_matrix_instructions(hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K);
        // Latency of a single MT_MxMT_NxMT_k tile is the latency of one MI multiplied by
        // number of MI per MT_MxMT_NxMT_k.
        size_t L_MI = hardware.get_mi_latency(MI_M, MI_N, MI_K, mi_datatype);

        // size_t mt_arith = arithmetic_intensity(MT_M, MT_N, MT_K, 2);
        // printf("MT_M:%d MT_N:%d MT_K:%d arith:%d\n", MT_M, MT_N, MT_K, mt_arith);
        // size_t arith = ((M * N * K * 2) / (M * K + N * K + M * N));
        size_t L_MT = L_MI * N_MI;

        return L_MT;
    }

    /* ---------------------------------------------------------------------------------------- */
    /* Memory-related functions                                                                 */
    /* ---------------------------------------------------------------------------------------- */
    // Check if MT fits in LDS
    bool check_lds_capacity(
        const hardware_t& hardware, size_t MT_M, size_t MT_N, size_t MT_K, size_t element_size)
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

    /*
 * This heuristic models data reuse by defining a "tile of workgroups" that can fit
 * its working set (portions of matrices A and B) into the L2 cache. The hit rate
 * is the ratio of reused data reads to total data reads within this tile.
 *
 * @param M, N, K, batch Problem dimensions.
 * @param MT_M, MT_N, MT_K Macro-tile dimensions.
 * @param element_size Size of a single data element in bits.
 * @param WGM Workgroup mapping size (typically 64).
 * @param splittingFactor K-splitting factor, reduces L2 contention.
 * @return Estimated L2 hit rate (0.0 to 1.0).
 */
    double estimate_l2_hit(const hardware_t& hardware,
                           size_t            M,
                           size_t            N,
                           size_t            K,
                           size_t            batch,
                           size_t            MT_M,
                           size_t            MT_N,
                           size_t            MT_K,
                           size_t            element_size,
                           int               WGM,
                           size_t            splittingFactor)
    {
        // Use size_t for dimensions and counts to ensure type safety.
        const size_t workgroups_m     = safe_ceil_div(M, MT_M);
        const size_t workgroups_n     = safe_ceil_div(N, MT_N);
        const size_t total_workgroups = workgroups_m * workgroups_n;

        // Concurrently executing workgroups are limited by the number of CUs.a
        const size_t concurrent_workgroups = std::min(total_workgroups, hardware.N_CU);
        if(concurrent_workgroups == 0)
            throw std::runtime_error("#Workgroups is zero in estimate l2 hit");

        // Number of CUs that might share the same K-tiles, adjusted for K-splitting.
        // This affects contention on the L2 cache partitions (XCDs).
        const size_t effective_cus = safe_ceil_div(concurrent_workgroups, splittingFactor);
        const size_t cu_per_xcd    = std::max(safe_ceil_div(effective_cus, hardware.NUM_XCD), static_cast<size_t>(1));

        // Initial guess for the L2 tile dimensions (a tile of workgroups).
        size_t l2_tile_n = std::min(static_cast<size_t>(WGM), workgroups_n);
        size_t l2_tile_m = safe_ceil_div(cu_per_xcd, l2_tile_n);

        // Handle wrap-around case: if the tile is taller than the grid, wrap it to be wider.
        if(l2_tile_m > workgroups_m)
        {
            size_t num_wraps = (l2_tile_m / workgroups_m);
            l2_tile_n += (num_wraps * WGM);
            l2_tile_m = workgroups_m;
        }

        // Clamp initial tile dimensions to the actual grid size.
        l2_tile_m = std::max(std::min(workgroups_m, l2_tile_m), static_cast<size_t>(1));
        l2_tile_n = std::max(std::min(workgroups_n, l2_tile_n), static_cast<size_t>(1));

        // Calculate memory footprint in bytes.
        const size_t element_bytes       = safe_ceil_div(element_size, 8);
        auto         calculate_footprint = [&](size_t tile_m, size_t tile_n) {
            size_t a_footprint = tile_m * MT_M * MT_K * element_bytes;
            size_t b_footprint = tile_n * MT_N * MT_K * element_bytes;
            return a_footprint + b_footprint;
        };

        // Symmetrically shrink the L2 tile until it fits in the L2 cache capacity.
        // This is more robust than shrinking only one dimension.
        while(calculate_footprint(l2_tile_m, l2_tile_n) > hardware.L2_capacity)
        {
            if(l2_tile_m > 1 && l2_tile_m >= l2_tile_n)
            {
                l2_tile_m--;
            }
            else if(l2_tile_n > 1)
            {
                l2_tile_n--;
            }
            else
            {
                // Cannot shrink further.
                break;
            }
        }

        // Uncached reads are the first read of each unique element within the L2 tile.
        const long long uncached_A_reads     = static_cast<long long>(l2_tile_m) * MT_M * MT_K;
        const long long uncached_B_reads     = static_cast<long long>(l2_tile_n) * MT_N * MT_K;
        const long long total_uncached_reads = uncached_A_reads + uncached_B_reads;

        // Total reads are the sum of all reads performed by all workgroups in the L2 tile.
        // Matrix A is reused l2_tile_n times, Matrix B is reused l2_tile_m times.
        const long long total_A_reads = uncached_A_reads * l2_tile_n;
        const long long total_B_reads = uncached_B_reads * l2_tile_m;
        const long long total_reads   = std::max(total_A_reads + total_B_reads, 1LL);

        const long long cached_reads = total_reads - total_uncached_reads;

        double l2_hit_rate = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

        // Final clamping and logging.
        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("L2Tile_M", l2_tile_m);
            hardware.log_debug("L2Tile_N", l2_tile_n);
            hardware.log_debug("TotalWorkgroups", total_workgroups);
            hardware.log_debug("ConcurrentWorkgroups", concurrent_workgroups);
        }

        // Clamp the hit rate to be within a realistic [0, 1] range.
        return std::max(0.0, std::min(l2_hit_rate, 1.0));
    }

    // Estimate MALL hit-rate
    double estimate_mall_hit(const hardware_t& hardware,
                             size_t            M,
                             size_t            N,
                             size_t            K,
                             size_t            batch,
                             size_t            MT_M,
                             size_t            MT_N,
                             size_t            MT_K,
                             size_t            element_size,
                             int               WGM,
                             size_t            numActiveCUs,
                             size_t            splittingFactor)
    {
        const size_t workgroups_m = safe_ceil_div(M, MT_M);
        const size_t workgroups_n = safe_ceil_div(N, MT_N);

        if(numActiveCUs == 0)
            throw std::runtime_error("Number of Active CUs was 0");

        // --- Initial Tile Sizing based on Concurrency ---
        // Use ceiling division for a more accurate initial guess.
        size_t mall_tile_m = safe_ceil_div(numActiveCUs, static_cast<size_t>(WGM));
        size_t mall_tile_n = std::min(static_cast<size_t>(WGM), workgroups_n);

        // Handle wrap-around case if the tile is taller than the grid.
        if(mall_tile_m > workgroups_m)
        {
            size_t num_wraps = mall_tile_m / workgroups_m;
            mall_tile_n += (num_wraps * WGM);
            mall_tile_m = workgroups_m;
        }

        // Clamp initial tile dimensions to the actual grid size.
        mall_tile_m = std::max(std::min(workgroups_m, mall_tile_m), static_cast<size_t>(1));
        mall_tile_n = std::max(std::min(workgroups_n, mall_tile_n), static_cast<size_t>(1));

        // --- CRITICAL: Shrink tile to fit into MALL Capacity ---
        const size_t element_bytes       = safe_ceil_div(element_size, 8);
        auto         calculate_footprint = [&](size_t tile_m, size_t tile_n) {
            size_t a_footprint = tile_m * MT_M * MT_K * element_bytes;
            size_t b_footprint = tile_n * MT_N * MT_K * element_bytes;
            return a_footprint + b_footprint;
        };

        // --- Calculate Hit Rate based on the final, capacity-aware tile size ---
        const long long uncached_A_reads     = static_cast<long long>(mall_tile_m) * MT_M * MT_K;
        const long long uncached_B_reads     = static_cast<long long>(mall_tile_n) * MT_N * MT_K;
        const long long total_uncached_reads = uncached_A_reads + uncached_B_reads;

        const long long total_A_reads = uncached_A_reads * mall_tile_n;
        const long long total_B_reads = uncached_B_reads * mall_tile_m;
        const long long total_reads   = std::max(total_A_reads + total_B_reads, 1LL);

        const long long cached_reads = total_reads - total_uncached_reads;

        double mall_hit_rate = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("MallTile_M", mall_tile_m);
            hardware.log_debug("MallTile_N", mall_tile_n);
            hardware.log_debug("MallFootprint_Bytes",
                               calculate_footprint(mall_tile_m, mall_tile_n));
        }

        // Clamp the final result to the valid [0, 1] range.
        return std::max(0.0, std::min(mall_hit_rate, 1.0));
    }

    /**
    @brief Computes the L2 hit rate from a global,
    problem - wide perspective.
    **/
    double compute_l2_hit_rate_global(size_t M,
                                      size_t N,
                                      size_t MT_M,
                                      size_t MT_N,
                                      size_t MT_K,
                                      size_t element_size,
                                      size_t l2_capacity_bytes)
    {
        // --- Hardware Parameters (as requested, defined locally) ---
        // You would normally get l2_capacity_bytes from your hardware_t struct.
        if(l2_capacity_bytes == 0)
            throw std::runtime_error("L2 Capacity is zero");
        ;

        // 1. Calculate the grid dimensions in terms of macro-tiles
        const size_t grid_m = safe_ceil_div(M, MT_M);
        const size_t grid_n = safe_ceil_div(N, MT_N);

        if(grid_m == 0 || grid_n == 0)
            throw std::runtime_error("estimate_l2_hit grid dimensions can not be zero");
        ;

        // 2. Calculate the working set size for one full pass of global reuse
        // This is the data needed by one full column of CUs (for A) and one full row (for B).
        const double bytes_per_element = static_cast<double>(element_size) / 8.0;
        const double a_working_set = static_cast<double>(grid_m * MT_M * MT_K) * bytes_per_element;
        const double b_working_set = static_cast<double>(grid_n * MT_N * MT_K) * bytes_per_element;
        const double total_working_set_bytes = a_working_set + b_working_set;

        // 3. CRUCIAL: Check if the working set fits in the L2 cache.
        // If it doesn't, the global reuse pattern is broken by capacity misses,
        // and the hit rate will be very low.
        if(total_working_set_bytes > l2_capacity_bytes)
        {
            // Return a floor value for the hit rate. The exact value can be tuned,
            // but it should be low to indicate that the ideal reuse is not possible.
            return 0.1; // 10% hit rate
        }

        // 4. If it fits, calculate the idealized global hit rate
        // Total reads if nothing was cached
        const double total_A_reads = static_cast<double>(grid_m * grid_n * MT_M * MT_K);
        const double total_B_reads = static_cast<double>(grid_m * grid_n * MT_N * MT_K);

        // Uncached reads are the first-time fetches for each row/column
        const double uncached_A_reads
            = static_cast<double>(grid_m * MT_M * MT_K); // One full column fetches A
        const double uncached_B_reads
            = static_cast<double>(grid_n * MT_N * MT_K); // One full row fetches B

        const double total_reads = total_A_reads + total_B_reads;
        if(total_reads == 0)
            return 1.0; // No reads, perfect hit rate.

        const double cached_reads
            = (total_A_reads - uncached_A_reads) + (total_B_reads - uncached_B_reads);

        return cached_reads / total_reads;
    }

    inline size_t round_up_mul(size_t x, size_t m)
    {
        return (x + m - 1) / m * m;
    }

    size_t round_elements_to_128B(size_t elements, size_t element_size_bits)
    {
        const size_t transaction_bits = 128u * 8u; // 1024
        const size_t g                = std::gcd(element_size_bits, transaction_bits);
        const size_t E_block          = transaction_bits / g; // elements per 128B-aligned chunk
        return round_up_mul(elements, E_block);
    }

    // Determine the memory latency
    double compute_memory_latency(const hardware_t& hardware,
                                  size_t            M,
                                  size_t            N,
                                  size_t            K,
                                  bool              transA,
                                  bool              transB,
                                  size_t            batch,
                                  size_t            MT_M,
                                  size_t            MT_N,
                                  size_t            MT_K,
                                  size_t            element_size_A,
                                  size_t            element_size_B,
                                  size_t            mx_block_size,
                                  int               WGM,
                                  size_t            numActiveCUs,
                                  size_t            splittingFactor)
    {
        // 1) Estimate L2 hit-rate
        double H_mem1 = estimate_l2_hit(
            hardware, M, N, K, batch, MT_M, MT_N, MT_K, element_size_A, WGM, splittingFactor);

        double H_mem1_global = compute_l2_hit_rate_global(
            M, N, MT_M, MT_N, MT_K, element_size_A, hardware.L2_capacity * 1024);

        H_mem1 = std::min(H_mem1, H_mem1_global);

        if(H_mem1 == 0)
        {
            H_mem1 = 0.5;
        }

        // 2) Estimate mall hit-rate
        double H_mem2 = estimate_mall_hit(hardware,
                                          M,
                                          N,
                                          K,
                                          batch,
                                          MT_M,
                                          MT_N,
                                          MT_K,
                                          element_size_A,
                                          WGM,
                                          numActiveCUs,
                                          splittingFactor);

        // 3) Total loads are loads from A and loads from B
        size_t MT_M_rounded_128bytes = round_elements_to_128B(MT_M, element_size_A);
        size_t MT_N_rounded_128bytes = round_elements_to_128B(MT_N, element_size_A);
        size_t MT_K_rounded_128bytes = round_elements_to_128B(MT_K, element_size_A);
        if(!transA && !transB)
        {
            MT_N_rounded_128bytes = MT_N;
            MT_K_rounded_128bytes = MT_K;
        }
        else if(transA && !transB)
        {
            MT_M_rounded_128bytes = MT_M;
            MT_N_rounded_128bytes = MT_N;
        }
        else if(!transA && transB)
        {
            MT_K_rounded_128bytes = MT_K;
        }

        size_t Ld_A_value  = compute_A_loads(MT_M_rounded_128bytes, MT_K_rounded_128bytes);
        size_t Ld_B_value  = compute_B_loads(MT_N_rounded_128bytes, MT_K_rounded_128bytes);
        size_t Ld_CU_bytes = (Ld_A_value * safe_ceil_div(element_size_A, 8)) // A Bytes
                             + (Ld_B_value * safe_ceil_div(element_size_B, 8)); // B Bytes

        // Logic for block scaled datatypes (Assuming BS=32 and 8-bit scales)
        // TODO This is technically wrong, need separate flag to enable MX so we can differentiate FP8
        // and MX8
        if(element_size_A < 8 && mx_block_size != 0)
        {
            // Number of scales per tile
            size_t num_scales_A = safe_ceil_div(MT_M * MT_K, mx_block_size);
            Ld_CU_bytes += num_scales_A; // One Byte per scale
        }
        if(element_size_B < 8 && mx_block_size != 0)
        {
            // Number of scales per tile
            size_t num_scales_B = safe_ceil_div(MT_N * MT_K, mx_block_size);
            Ld_CU_bytes += num_scales_B; // One Byte per scale
        }

        // 4) total loads by all CUs
        double total_Ld = Ld_CU_bytes * static_cast<double>(numActiveCUs);

        // 5) mem1‐limited factor (simple linear model)
        double mem1_bw_limited
            = static_cast<double>(numActiveCUs) / static_cast<double>(hardware.N_CU);
        double limited_mem1_bw = (hardware.mem1_perf_ratio * mem1_bw_limited);

        // 6) mem1 latency
        double L_mem_mem1 = (limited_mem1_bw > 0) ? (total_Ld / (limited_mem1_bw)) : 0.0;

        // 7) mem2‐limited from occupancy (Can't Issue enough load/stores)
        double bw_limited = compute_mem_bw_from_occupancy(hardware, numActiveCUs);

        // 8) loads that reach each level
        double Ld_mem2 = (1.0 - H_mem1) * total_Ld;
        double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

        // 9) enforce whole‐problem minimum loads when we can fit M/N in the CUs.
        // Calculate the tile of workgroups that can run concurrently (logic from estimate_mall_hit).
        size_t grid_m = safe_ceil_div(M, MT_M);
        size_t grid_n = safe_ceil_div(N, MT_N);
        size_t mall_m = safe_ceil_div(numActiveCUs, static_cast<size_t>(WGM));
        size_t mall_n = std::min(static_cast<size_t>(WGM), grid_n);
        // Handle wrap-around case
        if(mall_m > grid_m)
        {
            size_t num_wraps = (mall_m / grid_m);
            mall_n += (num_wraps * WGM);
            mall_m = grid_m;
        }
        // Clamp tile dimensions
        mall_m = std::max(std::min(grid_m, mall_m), static_cast<size_t>(1));
        mall_n = std::max(std::min(grid_n, mall_n), static_cast<size_t>(1));
        // This is the minimum unique bytes needed from HBM to feed the concurrent workgroups.
        double min_load
            = static_cast<double>((mall_m * MT_M * MT_K * safe_ceil_div(element_size_A, 8))
                                  + (mall_n * MT_N * MT_K * safe_ceil_div(element_size_B, 8)))
              * batch; // Apply batching to the minimum load itself.
        // The actual loads cannot be less than this physical minimum.
        Ld_MEM  = std::max(Ld_MEM, min_load);
        Ld_mem2 = std::max(Ld_mem2, min_load);

        // 10) mem2 latency
        double limited_mem2_bw = (hardware.mem2_perf_ratio * bw_limited);
        double L_mem_mem2      = (limited_mem2_bw > 0) ? (Ld_mem2 / limited_mem2_bw) : 0.0;

        // 11) MEM latency
        double limited_mem_bw = (hardware.mem3_perf_ratio * bw_limited);
        double L_mem_MEM      = (limited_mem_bw > 0) ? (Ld_MEM / limited_mem_bw) : 0.0;
        L_mem_MEM += 200; // Load Latency

        // 12) pick the worst‐case bound
        double L_mem = std::max({L_mem_mem1, L_mem_mem2, L_mem_MEM});

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("mem1_perf_ratio", hardware.mem1_perf_ratio);
            hardware.log_debug("mem2_perf_ratio", hardware.mem2_perf_ratio);
            hardware.log_debug("mem3_perf_ratio", hardware.mem3_perf_ratio);
            hardware.log_debug("mem_bw_per_wg_coefficients(0)",
                               std::get<0>(hardware.mem_bw_per_wg_coefficients));
            hardware.log_debug("mem_bw_per_wg_coefficients(1)",
                               std::get<1>(hardware.mem_bw_per_wg_coefficients));
            hardware.log_debug("mem_bw_per_wg_coefficients(2)",
                               std::get<2>(hardware.mem_bw_per_wg_coefficients));
            hardware.log_debug("H_mem1 (mem1 hit ratio)", H_mem1);
            hardware.log_debug("H_mem2 (mem2 hit ratio)", H_mem2);
            hardware.log_debug("Total Load (bytes)", total_Ld);
            hardware.log_debug("Ld_mem2 (bytes)", Ld_mem2);
            hardware.log_debug("Ld_MEM (bytes)", Ld_MEM);
            hardware.log_debug("L_mem_mem1 (cycles)", L_mem_mem1);
            hardware.log_debug("L_mem_mem2 (cycles)", L_mem_mem2);
            hardware.log_debug("L_mem_MEM (cycles)", L_mem_MEM);
            hardware.log_debug("MT_K % 128 bytes", MT_K * safe_ceil_div(element_size_B, 8) % 128);
            hardware.log_debug("MT_M % 128 bytes", MT_M * safe_ceil_div(element_size_A, 8) % 128);
            hardware.log_debug("MT_N % 128 bytes", MT_N * safe_ceil_div(element_size_B, 8) % 128);
            hardware.log_debug("MT_N % 128 + MT_M % 128 bytes",
                               (MT_M * safe_ceil_div(element_size_A, 8) % 128)
                                   + MT_N * safe_ceil_div(element_size_B, 8) % 128);
            hardware.log_debug("MT_N % 64 + MT_M % 64 bytes",
                               (MT_M * safe_ceil_div(element_size_A, 8) % 64)
                                   + MT_N * safe_ceil_div(element_size_B, 8) % 64);
            hardware.log_debug("MT_K % 64 bytes", MT_K * safe_ceil_div(element_size_B, 8) % 64);
            hardware.log_debug("MT_M % 64 bytes", MT_M * safe_ceil_div(element_size_A, 8) % 64);
            hardware.log_debug("MT_N % 64 bytes", MT_N * safe_ceil_div(element_size_B, 8) % 64);
            hardware.log_debug("Tile Arithmetic Intensity",
                               MT_M * MT_N * MT_K / (MT_M * MT_K + MT_N * MT_K));
        }

        return L_mem;
    }

    /* ---------------------------------------------------------------------------------------- */
    /* Tile-related functions                                                                   */
    /* ---------------------------------------------------------------------------------------- */
    double compute_tile_latency(const hardware_t& hardware,
                                size_t            M,
                                size_t            N,
                                size_t            K,
                                size_t            batch,
                                bool              transA,
                                bool              transB,
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
                                int               WGM,
                                size_t            occupancy,
                                size_t            numActiveCUs,
                                size_t            splittingFactor)
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

        // TODO Does work utilization need to be 128-byte rounded for a cache line?
        double utilization        = calculate_work_utilization(M, N, K, MT_M, MT_N, MT_K);
        double output_utilization = calculate_output_utilization(M, N, MT_M, MT_N, 1);
        // The effective latency per useful operation increases as utilization drops.
        // This penalty affects BOTH compute and memory bounds for the tile's core work.
        double effective_tile_penalty = (utilization > 1e-9) ? (1.0 / (utilization)) : 1.0;
        double output_utilization_penalty
            = (output_utilization > 1e-9) ? (1.0 / (output_utilization)) : 1.0;
        // 2) Work-group setup & iteration latencies
        double L_WG_setup = 1; // WG_setup_Latency

        // 3) Prologue: 2.2× memory latency
        double L_prologue = 1.5 * L_mem; // 1.5 chosen emprically

        // L_compute *= std::max(L_compute, L_LDS);

        // 4) Epilogue: writes from all active CUs with limited bandwidth
        double mem_bw_occ            = compute_mem_bw_from_occupancy(hardware, numActiveCUs);
        double mem_bw_occ_limited    = hardware.mem3_perf_ratio * mem_bw_occ;
        size_t MT_M_rounded_128bytes = round_elements_to_128B(MT_M, element_size_A);

        double L_epilogue = (static_cast<double>(numActiveCUs / splittingFactor)
                             * MT_M_rounded_128bytes * MT_N * safe_ceil_div(element_size_out, 8))
                            / mem_bw_occ_limited;
        // One compute iteration happens in the prologue
        L_epilogue += L_compute * effective_tile_penalty;
        // Epilogue and Prologue overhead are reduced with higher occupancy kernels.
        int grid_m         = static_cast<int>(safe_ceil_div(M, MT_M));
        int grid_n         = static_cast<int>(safe_ceil_div(N, MT_N));
        int real_occupancy = std::min(occupancy,
                                      safe_ceil_div(grid_m * grid_n * batch * splittingFactor,
                                                    hardware.N_CU)); // Number of WGs per CU.
        L_prologue         = L_prologue * pow(0.95, real_occupancy); // Factor chosen empirically
        L_epilogue         = L_epilogue * pow(0.95, real_occupancy); // Factor chosen empirically
        // 4') K-split reductions are globally coherent, we need to write and read split-1 MT_M*MT_N
        // tiles to coherent memory
        if(splittingFactor > 1)
        {
            size_t n_partials = splittingFactor - 1;

            // Only the reduction CU reads from all splits.
            double partial_read_bytes = grid_m * grid_n * n_partials * MT_M_rounded_128bytes * MT_N
                                        * safe_ceil_div(element_size_out, 8);

            // All CUs write (once for each partial, and once by the reduction CU for the output.)
            double partial_write_bytes = grid_m * grid_n * MT_M_rounded_128bytes * MT_N
                                         * safe_ceil_div(element_size_out, 8);

            double partial_readwrite_bytes = partial_read_bytes + partial_write_bytes;

            // 64 Threads active in a SIMD. Exposed to at least latency of reducing splittingfactor
            // tiles.
            double partial_adds = ((MT_M * MT_N) * splittingFactor) / (64);
            // Things have to be written to memory
            double mem_bw_occ         = compute_mem_bw_from_occupancy(hardware, numActiveCUs);
            double mem_bw_occ_limited = hardware.mem3_perf_ratio * mem_bw_occ;

            double L_reduce = partial_readwrite_bytes / (mem_bw_occ_limited);
            L_epilogue += L_reduce + partial_adds + 10000;
        }
        // 4'') tf32 emu has some more overhead
        double L_cvt    = 0;
        if((mi_datatype == data_type_t::XFloat32)
            && (hardware.arch == hardware_t::architecture_t::gfx950))
        {
            L_cvt = compute_cvt_overhead(
                hardware, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, element_size_A, element_size_B);
        }
        else if((element_size_A == 32) && (element_size_B == 32)
                 && (mi_datatype == data_type_t::BFloat16)
                 && (hardware.arch == hardware_t::architecture_t::gfx950)) // SS_BSS on GFX950
        {
            L_cvt = compute_cvt_overhead_x1(hardware,
                                            MT_M,
                                            MT_N,
                                            MT_K,
                                            MI_M,
                                            MI_N,
                                            MI_K,
                                            element_size_A,
                                            element_size_B,
                                            mi_datatype);
        }

        // 5) Single-tile latency (always additive)
        // Calculate the fraction of the work that is useful (not padding).

        // 5) Single-tile latency (apply penalty after finding the bottleneck)
        double L_tile_single = (std::max(L_compute, L_mem) * effective_tile_penalty) + L_cvt;
        L_prologue *= effective_tile_penalty;
        // 6) Number of K-iterations (excluding epilogue), at least 1
        // long num_iter = static_cast<long>(((K + MT_K - 1) / MT_K)) - 1;
        // num_iter      = std::ceil(num_iter / splittingFactor);
        // num_iter      = std::max(num_iter, 1L);
        long splittedK = static_cast<long>(safe_ceil_div(K, splittingFactor));
        long num_iter
            = std::max(static_cast<long>(safe_ceil_div(splittedK, MT_K) - 1), static_cast<long>(1));
        // Zero Padding in the K dimension on last iteration
        if(K % MT_K != 0)
        {
            double problem_k_quant = ((K % MT_K) / (double)K);
            L_epilogue
                += problem_k_quant
                   * 50000; // Scale by remainder proportion of problem. 50k cycle penalty if have to zero pad all except 1.
            //(Scale Determined Empirically)
        }
        //L_epilogue *= output_utilization_penalty;

        // 7) Total tile latency
        double L_tile_total
            = (L_tile_single * num_iter) + L_prologue + L_epilogue * 2 + L_WG_setup
              + (500 * num_iter); // 7 instructions (each with 4 cycles) at the end of the loop

        if(MT_K == 1024)
        {
            L_prologue = L_prologue * 100;
        }

        if(hardware_t::is_debug_enabled())
        {
            double problem_k_quant = ((K % MT_K) / (double)K);
            hardware.log_debug("Iteration Compute Latency", L_compute);
            hardware.log_debug("L_mem", L_mem);
            hardware.log_debug("L_cvt", L_cvt);
            hardware.log_debug("L_tile_single", L_tile_single);
            hardware.log_debug("num_iter", num_iter);
            hardware.log_debug("L_prologue", L_prologue);
            hardware.log_debug("L_epilogue", L_epilogue);
            hardware.log_debug("L_tile_total", L_tile_total);
            hardware.log_debug("Effective Tile Peanlty", effective_tile_penalty);
            hardware.log_debug("Problem K quant", problem_k_quant);
            hardware.log_debug("K quant overhead", (problem_k_quant * 50000));
            hardware.log_debug("Problem Tiile Quant", utilization);
            hardware.log_debug("Real Occupancy", utilization);
            hardware.log_debug("Output Utilization Penalty", output_utilization_penalty);
            hardware.log_debug("Output Utilization", output_utilization);
            std::string bound_source;
            if(L_compute >= L_mem)
            {
                L_tile_single = L_compute + L_cvt;
                bound_source  = "Compute";
            }
            else
            {
                L_tile_single = L_mem + L_cvt;
                bound_source  = "Memory";
            }
            hardware.log_debug("Iteration Bound",
                               bound_source + " (" + std::to_string(L_tile_single) + ")");

            hardware.log_debug("K % MT_K", K % MT_K);
        }

        return L_tile_total;
    }

    // Computes the latency per K-complete MT wave
    // A wave is defined as : The time it takes for one CU to complete one K-complete output tile
    double compute_wave_latency(const hardware_t& hardware,
                                size_t            M,
                                size_t            N,
                                size_t            K,
                                size_t            batch,
                                bool              transA,
                                bool              transB,
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
                                int               WGM,
                                size_t            occupancy,
                                size_t            numActiveCUs,
                                size_t            splittingFactor)
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
                                             occupancy,
                                             numActiveCUs,
                                             splittingFactor);

        return L_wave;
    }

    // Compute the total latency of a gemm based on the latency of one wave multiplied by the number of
    // waves A wave is defined as : The time it takes for one CU to complete one K-complete output tile
    double compute_total_latency(const hardware_t& hardware,
                                 size_t            M,
                                 size_t            N,
                                 size_t            K,
                                 size_t            batch,
                                 bool              transA,
                                 bool              transB,
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
                                 int               WGM,
                                 size_t            non_temporal_a,
                                 size_t            non_temporal_b,
                                 size_t            occupancy,
                                 size_t            split,
                                 size_t            max_cus)
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
            // When problem dimensions are small enough that we can fit them in one tile, we should do
            // so. This short circuit condition also decreases selection latency when problems are very
            // small :)
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
        auto [numActiveCUs, numWaves, splittingFactor]
            = compute_CU_occupancy(hardware,
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
                                   0,
                                   max_cus);

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
                                             occupancy,
                                             numActiveCUs,
                                             splittingFactor);

        // Compute latency for all waves and return it as the latency for the MT/problem
        double total_latency = L_wave * numWaves;

        if(MT_M == 64 && MT_N == 32 && MT_K == 32 && !transB && element_size_A == 16)
        {
            total_latency = total_latency * 10;
        }

        // 3) Customized heuristics
        // TODO These are quantifying effects that don't work in the current math.
        // TODO THESE SHOULD BE TEMPORARY FIXES AND BE MORE SOLIDLY INTEGRATED LATER
        bool heuristics = hardware_t::is_heuristics_enabled();

        const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS");
        heuristics      = !(env && std::string(env) == "0");
        // heuristics = 0;
        //  Heuristics for TF32
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
                if(arith < compute_threshold)
                    total_latency = total_latency * 0.6;
                else
                    total_latency = total_latency * 0.4;
            }

            // The kernel for this is more optimized (Custom kernel NN)
            if((!transA && !transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
            {
                if(arith < compute_threshold)
                    total_latency = total_latency * 0.8;
                else
                    total_latency = total_latency * 0.4;
            }

            // The kernel for this is more optimized (Custom kernel TN)
            if((transA && !transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
            {
                if(arith < compute_threshold)
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

        if(hardware_t::is_debug_enabled())
        {
            hardware.log_debug("Total_latency (with heuristics)", total_latency);
            hardware.log_debug("non_temporal_a", non_temporal_a);
            hardware.log_debug("non_temporal_b", non_temporal_b);
            hardware.log_debug("kernel_occupancy", occupancy);
            hardware.log_debug("splitting_factor", splittingFactor);
            hardware.log_debug("Input Tile Size A", MT_M * MT_K);
            hardware.log_debug("Input Tile Size B", MT_N * MT_K);
            hardware.log_debug("Output Tile Size", MT_M * MT_N);
            hardware.log_debug("Tile M/N", MT_M / MT_N);
            hardware.log_debug("Tile N/M", MT_N / MT_M);
            hardware.log_debug("Problem M/N", MT_M / MT_N);
            hardware.log_debug("Problem N/M", MT_N / MT_M);
            size_t occupancy_percent = numActiveCUs / hardware.N_CU;
            hardware.log_debug("Peak theoretical GFLOPs based on occupancy",
                               1300 * occupancy_percent);
            if(hardware_t::is_debug_enabled())
            {
                hardware.print_debug_info();
            }
        }

        return total_latency;
    }

    // Compute the performance from the latency.
    // IMPORTANT : This program is NOT meant to be an analytical model for performance, but rather a way
    // to rank different macro tile sizes. These performance values could be wildly inaccurate in
    // absolute terms, but will often result in the correct ranking of MTin relative terms.
    double compute_perf_gflops(const hardware_t& hardware,
                               size_t            M,
                               size_t            N,
                               size_t            K,
                               size_t            batch,
                               bool              transA,
                               bool              transB,
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
                               int               WGM,
                               size_t            max_cus)
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
                                                          WGM,
                                                          0,
                                                          0,
                                                          1,
                                                          0,
                                                          max_cus);
        double total_time_seconds = latency_cycles / cycles_per_second;
        // Compute performance in FLOPS
        double FLOPS = total_FLOPs / total_time_seconds;
        // Convert to TFLOPS
        double GFLOPS = FLOPS / 1e9; // 1 TFLOP = 1e9 FLOPs
        return GFLOPS;
    }
} // namespace origami
