// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "origami/origami_export.h"
#include "origami/types.hpp"

/**
 * @file types.hpp
 * @brief Origami Boost (backend boost_v1): a learned correction that re-orders Origami's top
 * candidates.
 *
 * A per-arch, per-dtype model file (ORIBOOST format, `.orr`) holds a gradient-boosted tree
 * ensemble that predicts yhat ~ log(measured_us) - log(origami_cycles) for one candidate kernel.
 * The booster computes each candidate's feature vector from Origami's own intermediates
 * (features/gemm_boost), evaluates the trees, and orders the candidates by
 * origami_cycles * exp(yhat), keeping Origami's order on ties. It never ranks by yhat alone, so
 * it runs after the analytical ranking (inference_mode_t::analytical_rerank).
 *
 * A model is tied to one Origami source state, one feature schema and its training kernel
 * library, by manifest diff: the library in use may drop kernels and add kernels of known
 * structure (library_stamp.hpp). booster_t::create refuses anything else. Every per-call failure
 * returns Origami's order unchanged with a reason.
 */
namespace origami::nn::boost {

/// Outcome of loading a model or ranking a call. Anything but ok means "use stock order".
enum class status_t : int {
  ok = 0,
  model_missing,       ///< no model file at the path
  bad_file,            ///< unreadable, bad magic, bad checksum, malformed section
  unsupported_format,  ///< format version, structural-field table or tree form this reader lacks
  incompatible,        ///< arch / dtype / Origami / schema / kernel-library mismatch
  out_of_range,        ///< problem outside the model's dtype class or training ranges
  missing_feature,     ///< a model input the extractor cannot compute for this candidate
  too_few_candidates,  ///< fewer than two usable candidates
};

/// The feature schema the extractor was written against: ordered column names and their hash.
struct feature_schema_t {
  std::string              id;
  std::string              hash;
  std::vector<std::string> columns;
};

/**
 * @brief The Stream-K launch the caller makes for one candidate on this problem.
 *
 * hipBLASLt: the fields of ContractionSolution::computeStreamKDecisions(problem, hardware), the
 * snapshot solve() takes its Stream-K launch from, for the problem hipBLASLt builds for the call
 * (CU budget, workspace and Stream-K parameters included), plus the kernel's sizeMapping values.
 * The streamk_launch columns (sk_*) are computed from these; a model that uses them falls back
 * with missing_feature when valid is false.
 */
struct streamk_launch_t {
  bool          valid      = false;  ///< false: not provided
  bool          is_streamk = false;  ///< sizeMapping.isStreamK(); false leaves the rest unused
  std::uint64_t final_grid = 0;      ///< d.finalGrid
  std::uint64_t tiles      = 0;      ///< d.tiles
  std::uint64_t sk_tiles   = 0;      ///< d.skTiles
  std::uint64_t sk_split   = 0;      ///< d.skSplit
  reduction_t   reduction  = reduction_t::none;  ///< d.reduction
  bool          is_dynamic = false;              ///< d.isDynamic
  bool          dp_only    = false;              ///< d.dpOnly
  std::uint64_t iters_per_tile = 0;  ///< max(1, problem.getItersPerTile(sizeMapping))
  std::uint64_t mt_x = 0, mt_y = 0;  ///< sizeMapping.macroTile.{x, y}
  std::uint64_t workspace_size_per_elem_c = 0;  ///< sizeMapping.workspaceSizePerElemC
  int           cu_occupancy = 0;               ///< sizeMapping.CUOccupancy
  std::uint64_t cu_count     = 0;  ///< the device's CU count (AMDGPU::computeUnitCount)
};

/// Fixup partial tiles of a Stream-K launch and the output tiles they cut.
struct streamk_partials_t {
  std::uint64_t partials    = 0;
  std::uint64_t split_tiles = 0;
};

/**
 * @brief Partial tiles a launch writes (feature schema v4, streamk_partials).
 *
 * Dynamic queue: none. Parallel reduction with skSplit >= 2: tiles * skSplit, every tile split.
 * Otherwise the static split: T = skTiles * itersPerTile iterations (0 if dpOnly) over G =
 * finalGrid workgroups under the kernel's global first-E mapping (workgroup g starts at
 * g * (T / G) + min(g, T % G)); each start inside a tile is one partial.
 */
ORIGAMI_EXPORT streamk_partials_t streamk_partials(const streamk_launch_t& launch);

/**
 * @brief Kernel fields the caller knows that config_t does not carry.
 *
 * Filled from the solution's library record (hipBLASLt: ContractionSolution::sizeMapping),
 * never from the kernel name. -1 means "not provided"; a model that needs a field the caller
 * did not provide falls back with missing_feature.
 */
struct kernel_fields_t {
  int num_loads_coalesced_a = -1;  ///< sizeMapping.NumLoadsCoalescedA
  int num_loads_coalesced_b = -1;  ///< sizeMapping.NumLoadsCoalescedB
  streamk_launch_t streamk{};      ///< per call: the launch on this problem
};

/// One candidate in Origami's ranked order.
struct candidate_t {
  /// The config rank_configs saw, with non_temporal_{a,b}_available set by
  /// memoize_non_temporal_availability over the full candidate list.
  const config_t* config          = nullptr;
  double          origami_latency = 0.0;  ///< rank_configs latency (cycles)
  kernel_fields_t fields{};
};

}  // namespace origami::nn::boost
