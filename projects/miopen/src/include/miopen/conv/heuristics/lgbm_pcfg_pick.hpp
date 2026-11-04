// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/config.hpp> // MIOPEN_INTERNALS_EXPORT

#include <string>
#include <vector>

namespace miopen {
struct Handle;
namespace conv {
struct ProblemDescription;
} // namespace conv

namespace ai {
namespace lgbm {
namespace pcfg {

// Rank the performance-config candidates for an already-chosen solver on the GPU
// described by `handle`. Scores every candidate config in the problem's
// (gfx_id, direction, data_type) bucket with the solver's lambdarank model and
// returns their verbatim descriptor strings ordered best->worst by predicted
// speed (stable, so ties keep catalog order). Each descriptor is ready for
// PerformanceConfig::Deserialize; a "" element denotes the solver default
// config. The caller walks this order and takes the first config that passes
// IsValidPerformanceConfig, falling back to GetDefaultPerformanceConfig if none
// is valid. Returns an empty vector (abstain, caller uses its default) when:
//   - metadata failed to load,
//   - the solver has no perf-config model,
//   - the bucket is unknown / empty.
// `solver_name` must be the solver's registry name (solver::Id::ToString()).
MIOPEN_INTERNALS_EXPORT std::vector<std::string> PickConfig(const std::string& solver_name,
                                                            const conv::ProblemDescription& problem,
                                                            const Handle& handle);

// Test seam: score a pre-built feature row (problem prefix already filled; the
// candidate arg tail is appended internally per candidate) for `solver_name`
// over an explicit candidate-arg list, returning the descriptors ranked
// best->worst (same order PickConfig produces). Lets gtest validate the scoring +
// ranking path against exported test vectors without a GPU or a
// ProblemDescription. Returns an empty vector if the solver model is unavailable.
MIOPEN_INTERNALS_EXPORT std::vector<std::string>
ScorePickForTest(const std::string& solver_name,
                 const std::vector<double>& prob_feature_prefix,
                 const std::vector<std::string>& cand_descs,
                 const std::vector<std::vector<double>>& cand_args);

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP
