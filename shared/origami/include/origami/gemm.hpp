// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/hardware.hpp"
#include <vector>

namespace origami
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
                                              size_t            MI_K);

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
                                      size_t            element_size_A, //In bits
                                      size_t            element_size_B, //In bits,
                                      data_type_t       mi_datatype);

    /* ---------------------------------------------------------------------------------------- */
    /* Memory-related functions                                                                 */
    /* ---------------------------------------------------------------------------------------- */
    // Check if MT fits in LDS
    bool check_lds_capacity(
        const hardware_t& hardware, size_t MT_M, size_t MT_N, size_t MT_K, size_t element_size_out);

    // Compute the amount of data loaded from A to produce a MT_MxMT_NxMT_K tile.
    size_t compute_A_loads(size_t MT_M, size_t MT_K);

    // Compute the amount of data loaded from B to produce a MT_MxMT_NxMT_K tile.
    size_t compute_B_loads(size_t MT_N, size_t MT_K);

    // Computes total data loads per CU per MT from A and B
    // Reads happen every MT, Writes happen every K-complete tile.
    size_t compute_cu_loads(size_t MT_M, size_t MT_N, size_t MT_K);

    // Estimates the l2 hit-rate
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
                           size_t            splittingFactor);

    // Estimates the mall hit-rate
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
                             size_t            splittingFactor);

    // Determine the memory latency per MT_MxMT_NxMT_K Macro Tile (L_MT).
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
                                  size_t            element_size_A, //In bits
                                  size_t            element_size_B, //In bits,
                                  size_t            mx_block_size,
                                  int               WGM,
                                  size_t            numActiveCUs,
                                  size_t            splittingFactor);

    /* ---------------------------------------------------------------------------------------- */
    /* Tile-related functions                                                                   */
    /* ---------------------------------------------------------------------------------------- */
    // Computes the latency to compute a K-COMPLETE tile.
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
                                size_t            element_size_A, //In bits
                                size_t            element_size_B, //In bits,
                                size_t            element_size_out, //In bits
                                data_type_t       mi_datatype,
                                size_t            mx_block_size,
                                int               WGM,
                                size_t            occupancy,
                                size_t            numActiveCUs,
                                size_t            splittingFactor);

    // Computes the latency per K-complete MT wave.
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
                                size_t            element_size_A, //In bits
                                size_t            element_size_B, //In bits,
                                size_t            element_size_out, //In bits
                                data_type_t       mi_datatype,
                                size_t            mx_block_size,
                                int               WGM,
                                size_t            occupancy,
                                size_t            numActiveCUs,
                                size_t            splittingFactor);

    // Compute the total latency of a gemm based on the latency of one wave multiplied by the number of waves
    // A wave is defined as : The time it takes for one CU to complete one K-complete output tile
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
                                 size_t            element_size_A, //In bits
                                 size_t            element_size_B, //In bits,
                                 size_t            element_size_out, //In bits
                                 data_type_t       mi_datatype,
                                 size_t            mx_block_size,
                                 int               WGM,
                                 size_t            non_temporal_a = 0,
                                 size_t            non_temporal_b = 0,
                                 size_t            occupancy      = 1,
                                 size_t            split          = 0,
                                 size_t            max_cus        = 0);

    // Compute the performance from the latency.
    // IMPORTANT : This program is NOT meant to be an analytical model for performance, but rather a way to rank different macro tile sizes.
    // These performance values could be wildly inaccurate in absolute terms, but will often result in the correct ranking of MTin relative terms.
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
                               size_t            max_cus = 0);
} // namespace origami
