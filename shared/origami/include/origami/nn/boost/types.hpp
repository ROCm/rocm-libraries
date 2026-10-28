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
};

}  // namespace origami::nn::boost
