// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/cost_models/gemm/heuristics.hpp"
#include "origami/cost_models/gemm/memory.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/**
 * @brief Context for kernel execution.
 *
 * Holds derived/computed values for a GEMM kernel execution.
 * This struct bundles grid dimensions, occupancy info, and other
 * values computed from problem, config, and hardware.
 */
struct ORIGAMI_EXPORT context_t {
  /// Heuristic parameters (cached to avoid repeated lookups).
  heuristic_params_t heuristic{};

  /// Element sizes (fractional for sub-byte types like FP4).
  double a_bytes = 0.0;
  double b_bytes = 0.0;
  double d_bytes = 0.0;

  /// Element sizes of A and B in bits.
  int a_bits = 0;
  int b_bits = 0;

  /// Whether A and B are transposed.
  bool a_transposed = false;
  bool b_transposed = false;

  /// Threads per wavefront of the kernel, and the clocks one of its vector-ALU instructions
  /// occupies a SIMD.
  size_t wavefront_size = 0;
  size_t valu_cycles    = 0;

  /// Grid dimensions.
  size_t grid_m           = 0;
  size_t grid_n           = 0;
  size_t num_output_tiles = 0;

  /// Launch parameters.
  reduction_t reduction_strategy = reduction_t::none;
  size_t splitting_factor        = 0;
  size_t num_wgs                 = 0;
  size_t num_timesteps           = 0;

  /// Tile-scheduling mode.
  hybrid_mode_t tile_schedule = hybrid_mode_t::none;

  /// Hardware-derived values.
  /// Number of compute units usable for this problem. Equals hardware.N_CU
  /// unless problem.num_cus caps it lower (@see origami::problem_t::num_cus).
  /// This is the effective CU count the whole model schedules against.
  size_t n_cu                 = 0;
  size_t active_cus           = 0;
  double mem_bw_limited       = 0.0;
  double write_mem_bw_limited = 0.0;

  /// Tile-derived values.
  size_t k_per_split       = 0;
  size_t k_iters           = 0;
  size_t tile_elements     = 0;
  double output_tile_bytes = 0.0;

  /// Waves per workgroup along M and N, and LocalSplitU groups along K.  Each is at
  /// least 1: non-MFMA kernels report a 0x0 wave group.
  dim3_t wave_group{1, 1, 1};

  /// Per-operand DRAM traffic (cache/mem model input).
  operand_traffic_t traffic{};

  /// Workgroup mapping parameters.
  workgroup_mapping_t wgm{0, 8, 1};

  /// Debug flag (cached from runtime_options to avoid repeated singleton lookups).
  bool debug = false;

  /// Default constructor.
  context_t() = default;

  /**
   * @brief Constructor from config, problem, and hardware.
   *
   * @param problem Problem description (M, N, K, etc.)
   * @param hardware Hardware characteristics (@see origami::hardware_t)
   * @param config Kernel configuration.
   */
  context_t(const problem_t& problem, const hardware_t& hardware, const config_t& config);

  /**
   * @brief Constructor for a caller that has already looked the config's heuristics up.
   *
   * @param problem Problem description (M, N, K, etc.)
   * @param hardware Hardware characteristics (@see origami::hardware_t)
   * @param config Kernel configuration.
   * @param heuristic What get_heuristic_params returns for the three above.
   */
  context_t(const problem_t& problem,
            const hardware_t& hardware,
            const config_t& config,
            const heuristic_params_t& heuristic);

  /**
   * @brief Check if the context is valid.
   *
   * @return bool True if the context is valid, false otherwise.
   */
  bool is_valid() const;
};

// ---------------------------------------------------------------------------
// Utilization and capacity
// ---------------------------------------------------------------------------

/**
 * @brief calculate the work utilization which is the ratio of the useful problem volume to the
 * total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
ORIGAMI_EXPORT double calculate_work_utilization(const problem_t& problem, const config_t& config);

/**
 * @brief calculate the output utilization which is the ratio of the useful problem volume to the
 * total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @param vector_elems elements in the vector.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
ORIGAMI_EXPORT double calculate_output_utilization(const problem_t& problem,
                                                   const config_t& config,
                                                   size_t vector_elems);

/**
 * @brief This function rounds the number of elements up to the smallest value whose total size
 * (given the element bit-width) is an exact multiple of a `transaction_bytes` memory transaction.
 *
 * @param elements Macro tile dimension
 * @param element_size_bits size in bits
 * @param transaction_bytes memory transaction granularity in bytes (e.g. 128 for an L1 line,
 *                          64 for a DRAM sector)
 * @return size_t
 */
ORIGAMI_EXPORT size_t round_elements_to_NB(size_t elements,
                                           size_t element_size_bits,
                                           size_t transaction_bytes);

/**
 * @brief Check if MT fits in LDS
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro tile dimensions
 * @param a_dtype Data type of operand A
 * @param b_dtype Data type of operand B
 * @return bool True if MT fits in LDS, false otherwise
 */
ORIGAMI_EXPORT bool check_lds_capacity(const hardware_t& hardware,
                                       const dim3_t& mt,
                                       const data_type_t& a_dtype,
                                       const data_type_t& b_dtype);

// ---------------------------------------------------------------------------
// Tile model
// ---------------------------------------------------------------------------

/**
 * @brief The terms of the tile model, in the order the debug log prints them.
 *
 * FIELD(type, name, description) is a member of tile_terms_t that the log prints under its own
 * name.  FIELD_AS(type, name, log name, description) is a member printed under another name, or
 * not printed when the log name is empty.  LOGGED(log name, expression of t) is a log line that
 * is not a member.
 * The struct, the debug log (and with it the CSV columns) and the Python binding are all
 * generated from this list, so a term is added here once.
 */
// clang-format off
#define ORIGAMI_GEMM_TILE_TERMS(FIELD, FIELD_AS, LOGGED) \
  FIELD_AS(double, wgs_per_cu, "per_wave wgs_per_cu",                                        \
           "Workgroups resident on one CU")                                                  \
  FIELD_AS(double, simds_busy, "per_wave simds_busy",                                        \
           "SIMDs the resident waves occupy")                                                \
  FIELD_AS(double, wg_score, "per_wave wg_score",                                            \
           "Resident workgroups over the target, clamped to [0, 1]")                         \
  FIELD(double, per_wave_score, "Score the compute time is divided by")                      \
  FIELD(double, utilization, "Useful volume over launched volume")                           \
  FIELD(double, effective_tile_penalty, "1 / utilization")                                   \
  FIELD(double, compute_tile_penalty, "Penalty on compute terms (no sub-instruction waste)") \
  LOGGED("L_mem", t.memory.L_mem)                                                            \
  FIELD(double, L_lds_read, "LDS read time at the kernel's vector width")                    \
  FIELD(double, L_lds_floor, "LDS read time at full width")                                  \
  FIELD(double, L_local_read, "LDS read time exposed beyond the compute time")               \
  FIELD(double, L_latency_iter, "Load stall exposed per main iteration")                     \
  FIELD(double, L_cluster_sync, "Exposed cluster-barrier handshakes")                        \
  FIELD(double, L_compute, "Compute side")                                                   \
  FIELD(double, L_cvt, "Conversion of emulated data types")                                  \
  FIELD(size_t, k_per_split, "K one split works on")                                         \
  FIELD_AS(long, k_iters, "",                                                                \
           "Full DepthU iterations of one workgroup")                                        \
  LOGGED("k_iters", int(t.k_iters))                                                          \
  FIELD_AS(long, num_main_iters, "",                                                         \
           "Iterations that load and compute")                                               \
  LOGGED("num_main_iters", int(t.num_main_iters))                                            \
  FIELD_AS(long, num_ngll_iters, "",                                                         \
           "Iterations that compute on loads already in flight")                             \
  LOGGED("num_ngll_iters", int(t.num_ngll_iters))                                            \
  FIELD(double, L_mem_stream, "Memory time of the main iterations")                          \
  FIELD(double, L_compute_stream, "Compute time of the main iterations")                     \
  FIELD(double, L_prologue, "First-load stall")                                              \
  FIELD(double, L_main, "Main iterations, memory and compute overlapped")                    \
  FIELD(double, L_ngll, "Drain iterations")                                                  \
  FIELD(double, L_nll, "Final compute-only iteration")                                       \
  FIELD(size_t, tail_k, "K left over after the full iterations")                             \
  FIELD_AS(double, tail_fraction, "",                                                        \
           "tail_k / MT_K")                                                                  \
  FIELD(double, L_tail, "Partial iteration for K % MT_K")                                    \
  FIELD(double, pgr_unamortized_iters, "Prefetch fills a short K-loop does not amortize")    \
  FIELD(double, pgr_mem_exposure, "Memory share of one K-iteration")                         \
  FIELD(double, L_pgr_stall, "Prefetch fill and drain exposure")                             \
  FIELD(double, L_loop_overhead, "Per-iteration bookkeeping")                                \
  FIELD(double, L_single_lds_buffer, "Exposed LDS writes of a one-buffer kernel")            \
  FIELD(double, depth_waste_ratio, "Loaded but unused K over K")                             \
  FIELD(double, batched_fill_ratio, "Shortfall of a batched K-loop from its target length")  \
  FIELD(double, L_du_waste, "Charge for depth waste, M-edge waste and batched fill")         \
  FIELD(double, L_mainloop, "Main loop subtotal")                                            \
  FIELD(double, epi_cu_occupancy, "Workgroups per CU the epilogue overlaps with")            \
  FIELD(double, epi_occ_exposure, "Store exposure at that occupancy")                        \
  FIELD(double, store_exposure, "Epilogue share of main loop plus epilogue")                 \
  FIELD(double, scalar_store_fraction, "Share of the epilogue in scalar edge stores")        \
  FIELD(double, scalar_store_mult, "Amplification of exposed scalar stores")                 \
  FIELD(double, L_epilogue, "Tile store")                                                    \
  FIELD(double, L_lsu_reduce, "LocalSplitU reduction through LDS")                           \
  FIELD_AS(double, eff_scale, "eff_scale (main_loop_efficiency)",                            \
           "Main-loop efficiency of hand-optimized kernels")                                 \
  FIELD(double, weight_tile_total, "Discount of hand-optimized kernels")                     \
  FIELD(double, L_tile_total, "Latency of the tile")
// clang-format on

/**
 * @brief Every term of the tile model for one config.
 *
 * A kernel runs a tile as prologue, main loop and epilogue:
 *
 *   L_tile_total = (L_prologue + L_mainloop + L_epilogue + L_lsu_reduce) * weight_tile_total
 *   L_mainloop   = L_main + L_ngll + L_nll + L_tail + L_pgr_stall + L_loop_overhead
 *                  + L_single_lds_buffer + L_du_waste + L_cluster_sync
 *
 * Latencies are in cycles.  The members are the FIELD rows of ORIGAMI_GEMM_TILE_TERMS, plus the
 * memory side of one K-iteration.
 */
struct tile_terms_t {
  memory_terms_t memory{};  ///< Memory side of one K-iteration; L_mem is its time.

#define ORIGAMI_FIELD(type, name, description) type name{};
#define ORIGAMI_FIELD_AS(type, name, log_name, description) type name{};
#define ORIGAMI_LOGGED(log_name, expression)
  ORIGAMI_GEMM_TILE_TERMS(ORIGAMI_FIELD, ORIGAMI_FIELD_AS, ORIGAMI_LOGGED)
#undef ORIGAMI_FIELD
#undef ORIGAMI_FIELD_AS
#undef ORIGAMI_LOGGED
};

/**
 * @brief Evaluates the tile model and returns every term.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return tile_terms_t The terms; L_tile_total is the latency compute_tile_latency returns.
 */
ORIGAMI_EXPORT tile_terms_t compute_tile_terms(const problem_t& problem,
                                               const hardware_t& hardware,
                                               const config_t& config,
                                               const context_t& context);

/**
 * @brief Computes the latency to compute a K-COMPLETE tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_tile_latency(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           const context_t& context);

}  // namespace gemm
}  // namespace origami
