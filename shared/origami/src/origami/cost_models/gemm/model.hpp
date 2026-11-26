// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/hardware/family.hpp"
#include "origami/heuristics.hpp"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Helpers the source files share
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The kernel a config describes (tile.cpp)
// ---------------------------------------------------------------------------

const tensile_params_t& tparams(const config_t& config);
int load_temporal_hint(const config_t& config, bool b_operand);
int store_temporal_hint(const config_t& config);
bool l2_low_priority_load(int th);
size_t kernel_wavefront_size(const hardware_t& hardware, const config_t& config);
size_t kernel_valu_cycles(const hardware_t& hardware, const config_t& config);
dim3_t compute_wave_share(const config_t& config);
double compute_resident_wgs(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const context_t& context);
bool is_multicast_cluster(const config_t& config);
double operand_window_bytes(const config_t& config, const context_t& context);

// ---------------------------------------------------------------------------
// Launch (streamk.cpp)
// ---------------------------------------------------------------------------

size_t clusters_per_wave(const hardware_t& hardware, size_t usable_cus, size_t cluster);
size_t cluster_blocks(const problem_t& problem, const config_t& config);

// ---------------------------------------------------------------------------
// Utilization (tile.cpp)
// ---------------------------------------------------------------------------

double calculate_compute_utilization(const problem_t& problem, const config_t& config);

// ---------------------------------------------------------------------------
// Compute (compute.cpp)
// ---------------------------------------------------------------------------

double compute_w4a16_dequant_latency(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config);

// ---------------------------------------------------------------------------
// Memory (memory.cpp)
// ---------------------------------------------------------------------------

operand_traffic_t compute_operand_traffic(
    const problem_t& problem,
    const config_t& config,
    const context_t& context,
    size_t transaction_bytes = heuristic_defaults_t::DRAM_SECTOR_BYTES);
double compute_operand_window_bytes(const problem_t& problem,
                                    const config_t& config,
                                    const context_t& context,
                                    size_t k_window,
                                    size_t transaction_bytes);
double soft_min(double x, double y, double sharpness);

// ---------------------------------------------------------------------------
// Family models
// ---------------------------------------------------------------------------

/**
 * @brief What every stage of the tile model reads: the inputs of one config evaluation.
 */
struct tile_args_t {
  const problem_t& problem;
  const hardware_t& hardware;
  const config_t& config;
  const context_t& context;
  const tensile_params_t& tp;  ///< The config's TensileLite parameters, or the defaults.
  long pgr;                    ///< PrefetchGlobalRead depth.
};

/**
 * @brief The GEMM model of the families that share the original CDNA / RDNA behaviour.
 *
 * Each member is one decision or one stage of the model that an architecture family may do
 * differently.  The shared code (context, launch, tile skeleton, total latency) is written once
 * against these members and never asks which architecture it runs on.  A family that differs
 * derives from this struct and redeclares only the members it changes; see cdna2_model,
 * cdna4_model and cdna5_model.
 *
 * Each member is defined in the file of its stage: cache.cpp, compute.cpp, epilogue.cpp,
 * memory.cpp, streamk.cpp and tile.cpp.
 */
struct base_model {
  // -------------------------------------------------------------------------
  // Admission and fixed costs
  // -------------------------------------------------------------------------

  /// Whether the non-temporal A/B admission rules apply.
  static constexpr bool non_temporal_admission = true;

  /// Granule of a DRAM read, in bytes.
  static size_t dram_block_bytes(const hardware_t&) {
    return heuristic_defaults_t::DRAM_SECTOR_BYTES;
  }

  /// Cost of launching the kernel, in cycles.
  static double kernel_launch_overhead(const hardware_t&) {
    return heuristic_defaults_t::KERNEL_LAUNCH_OVERHEAD;
  }

  /// Cost of launching the parallel-reduction kernel, in cycles.
  static double reduction_launch_overhead(const hardware_t&) {
    return heuristic_defaults_t::POSTGSU_KERNEL_LAUNCH_OVERHEAD;
  }

  /// Cost a kernel pays once outside its tiles, beyond the launch, in cycles.
  static double kernel_fixed_cost(const hardware_t&, const config_t&, const context_t&) {
    return 0.0;
  }

  // -------------------------------------------------------------------------
  // Heuristics
  // -------------------------------------------------------------------------

  /// Number of hand-optimized kernels register_heuristics adds.
  static size_t hand_optimized_kernel_count() { return 0; }

  /// Adds the family's entries to the heuristics database.
  static void register_heuristics(heuristics_database_t&) {}

  /// Applies the family's rules that depend on the problem to looked-up heuristic parameters.
  static void adjust_heuristics(heuristic_params_t&,
                                const problem_t&,
                                const hardware_t&,
                                const config_t&) {}

  // -------------------------------------------------------------------------
  // Launch
  // -------------------------------------------------------------------------

  /// Workgroups per cluster; 1 where the hardware has no clusters.
  static size_t cluster_size(const config_t& config);

  /// Active CUs in the units of the hardware's bandwidth-versus-occupancy curve.
  static double occupancy_curve_cus(const hardware_t& hardware, size_t num_active_cus);

  /// Tile-scheduling mode a StreamK=5 hybrid kernel launches in, with @p sm_count_target CUs
  /// available to it (0 = all).
  static hybrid_mode_t tile_schedule(const problem_t&, const hardware_t&, const config_t&, size_t) {
    return hybrid_mode_t::static_;
  }

  /// Workgroup mapping the kernel launches with.
  static workgroup_mapping_t predict_workgroup_mapping(const problem_t& problem,
                                                       const hardware_t& hardware,
                                                       const config_t& config,
                                                       size_t grid_m,
                                                       size_t grid_n,
                                                       size_t splitting_factor);

  // -------------------------------------------------------------------------
  // Stages of one tile
  // -------------------------------------------------------------------------

  /// Memory side of one K-iteration.
  static memory_terms_t memory(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config,
                               const context_t& context);

  /// Memory side of one K-iteration as the tile model charges it.
  static memory_terms_t tile_memory(const tile_args_t& args);

  /// Whether the kernel pays for converting emulated data types (hardware without native TF32).
  static bool charges_conversion(const hardware_t& hardware) { return !hardware.has_native_TF32(); }

  /// Compute side of one K-iteration: sets per_wave_score and L_compute from the
  /// matrix-instruction time @p L_mi.
  static void compute_iteration(const tile_args_t& args, double L_mi, tile_terms_t& t);

  /// Whether launched volume beyond the problem's edges is charged (the tile penalties).
  static constexpr bool charges_edge_waste = true;

  /// Part of a load's transfer the global-read prefetch leaves uncovered: sets L_latency_iter.
  static void load_stall(const tile_args_t& args, tile_terms_t& t);

  /// Whether the kernel's first-load stall is charged (the prologue).
  static constexpr bool charges_prologue = true;

  /// Main iterations with memory and compute overlapped, before the efficiency scale.
  static double main_iterations(const tile_args_t& args, tile_terms_t& t);

  /// Partial iteration for K % MT_K; called only when there is one.
  static double tail_iteration(const tile_args_t& args, const tile_terms_t& t);

  /// Per-iteration bookkeeping of the K-loop.
  static double loop_overhead(const tile_args_t& args, const tile_terms_t& t);

  /// Exposed cluster-barrier handshakes.
  static double cluster_sync(const tile_args_t& args, const tile_terms_t& t);

  /// Charge for depth waste, M-edge waste and batched fill.
  static double depth_waste(const tile_args_t& args, const tile_terms_t& t);

  /// Store exposure of the epilogue at @p wgs_per_cu resident workgroups.
  static double epilogue_exposure(const hardware_t& hardware, double wgs_per_cu);

  /// Epilogue of one tile.
  static epilogue_terms_t epilogue(const problem_t& problem,
                                   const hardware_t& hardware,
                                   const config_t& config,
                                   const context_t& context);

  /// LocalSplitU reduction of one tile through LDS; called only when LocalSplitU > 1.
  static double lsu_reduction(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context);

  /// Tile latency after the costs the stage sum leaves out.
  static double finish_tile(const tile_args_t& args, const tile_terms_t& t);
};

/**
 * @brief The GEMM model of CDNA2 (gfx90a): the shared model without the conversion charge.
 */
struct cdna2_model : base_model {
  // TODO: gfx90a also lacks native TF32 and should get CVT overhead, but
  // enabling it changes rankings — address in a separate PR.
  static bool charges_conversion(const hardware_t&) { return false; }
};

/**
 * @brief The GEMM model of CDNA4 (gfx950): the shared model with this family's heuristics.
 *
 * The heuristics are defined in heuristics_cdna4.cpp and streamk_cdna4.cpp.  Their constants are
 * read through constants(), so each chip variant has its own values (cdna4_constants_t).
 */
struct cdna4_model : base_model {
  /// Constants of the hardware's chip variant.
  static const cdna4_constants_t& constants(const hardware_t& hardware) {
    return cdna4_constants(hardware.variant());
  }

  static size_t hand_optimized_kernel_count();
  static void register_heuristics(heuristics_database_t& db);
  static void adjust_heuristics(heuristic_params_t& params,
                                const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config);

  static hybrid_mode_t tile_schedule(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config,
                                     size_t sm_count_target);
};

/**
 * @brief The GEMM model of CDNA5 (gfx1250): tensor-DMA loads, one L2 per AID, no MALL, clusters.
 *
 * Members not redeclared here are the shared ones.  The stages are defined in the *_cdna5.cpp
 * files.  Every constant is read through constants() and fit(), so each chip variant has its
 * own values (cdna5_constants_t, cdna5_fit_t).
 */
struct cdna5_model : base_model {
  /// Measured constants of the hardware's chip variant.
  static const cdna5_constants_t& constants(const hardware_t& hardware) {
    return cdna5_constants(hardware.variant());
  }

  /// Fitted constants of the hardware's chip variant.
  static const cdna5_fit_t& fit(const hardware_t& hardware) {
    return cdna5_fit(hardware.variant());
  }

  static constexpr bool non_temporal_admission = false;
  static size_t dram_block_bytes(const hardware_t& hardware) {
    return constants(hardware).DRAM_BLOCK_BYTES;
  }
  static double kernel_launch_overhead(const hardware_t& hardware) {
    return constants(hardware).KERNEL_LAUNCH_CLK;
  }
  static double reduction_launch_overhead(const hardware_t& hardware) {
    return constants(hardware).KERNEL_LAUNCH_CLK;
  }
  static double kernel_fixed_cost(const hardware_t& hardware,
                                  const config_t& config,
                                  const context_t& context);

  static size_t cluster_size(const config_t& config);
  static double occupancy_curve_cus(const hardware_t& hardware, size_t num_active_cus);
  static workgroup_mapping_t predict_workgroup_mapping(const problem_t& problem,
                                                       const hardware_t& hardware,
                                                       const config_t& config,
                                                       size_t grid_m,
                                                       size_t grid_n,
                                                       size_t splitting_factor);

  static memory_terms_t memory(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config,
                               const context_t& context);
  static memory_terms_t tile_memory(const tile_args_t& args);
  static void compute_iteration(const tile_args_t& args, double L_mi, tile_terms_t& t);
  static constexpr bool charges_edge_waste = false;
  static void load_stall(const tile_args_t& args, tile_terms_t& t);
  static constexpr bool charges_prologue = false;
  static double main_iterations(const tile_args_t& args, tile_terms_t& t);
  static double tail_iteration(const tile_args_t& args, const tile_terms_t& t);
  static double loop_overhead(const tile_args_t& args, const tile_terms_t& t);
  static double cluster_sync(const tile_args_t& args, const tile_terms_t& t);
  static double depth_waste(const tile_args_t& args, const tile_terms_t& t);
  static double epilogue_exposure(const hardware_t& hardware, double wgs_per_cu);
  static epilogue_terms_t epilogue(const problem_t& problem,
                                   const hardware_t& hardware,
                                   const config_t& config,
                                   const context_t& context);
  static double lsu_reduction(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context);
  static double finish_tile(const tile_args_t& args, const tile_terms_t& t);
};

/**
 * @brief The GEMM model of an architecture family, selected at compile time by its tag.
 *
 * The primary template is the shared model; a family with its own model specializes it.
 */
template <class Arch>
struct model : base_model {};

template <>
struct model<arch::cdna2> : cdna2_model {};

template <>
struct model<arch::cdna4> : cdna4_model {};

template <>
struct model<arch::cdna5> : cdna5_model {};

/**
 * @brief Call @p fn with the model of the family of @p hardware.
 *
 * Non-template entry points use this to reach the family's model:
 *
 *   for_family(hardware, [&](auto family) { return family.memory(problem, ...); });
 */
template <class Fn>
decltype(auto) for_family(const hardware_t& hardware, Fn&& fn) {
  return arch::dispatch(hardware, [&](auto tag) { return fn(model<decltype(tag)>{}); });
}

/**
 * @brief Call @p fn with the model of every family.
 */
template <class Fn>
void for_each_family(Fn&& fn) {
  for (family_t family : {family_t::cdna2,
                          family_t::cdna3,
                          family_t::cdna4,
                          family_t::cdna5,
                          family_t::rdna3,
                          family_t::rdna4}) {
    arch::dispatch(family, [&](auto tag) { fn(model<decltype(tag)>{}); });
  }
}

}  // namespace gemm
}  // namespace origami
