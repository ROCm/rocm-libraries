// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/config.hpp> // MIOPEN_INTERNALS_EXPORT

#include <miopen/miopen.h> // miopenDataType_t

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace miopen {
struct Handle;
namespace conv {
struct ProblemDescription;
enum class Direction;
} // namespace conv

namespace ai {
namespace lgbm {
namespace pcfg {

struct SolverModel; // lgbm_pcfg_metadata.hpp

// Candidates ranked best->worst by predicted speed (higher score = faster): descs[i] is scored
// scores[i]. A "" desc denotes the solver default config.
struct RankedConfigs
{
    std::vector<std::string> descs;
    std::vector<double> scores;
};

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
// `default_desc` returns the descriptor of the solver's default config; it is called only when
// DefaultMargin applies to the problem.
MIOPEN_INTERNALS_EXPORT std::vector<std::string>
PickConfig(const std::string& solver_name,
           const conv::ProblemDescription& problem,
           const Handle& handle,
           const std::function<std::string()>& default_desc);

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

// Bucket key "<gfx_id>|<direction code>|<dtype>" that PickConfig looks up in a solver's
// buckets. `device_name` is the name Handle::GetDeviceName() returns; it is normalized
// (NormalizeArchName in lgbm_common.hpp), so "gfx1250-strict" gives the "gfx1250|..." key.
MIOPEN_INTERNALS_EXPORT std::string
BucketKey(const std::string& device_name, conv::Direction direction, miopenDataType_t data_type);

// gfx_code feature for solvers trained with one (SolverModel::has_gfx_code): the index of the
// normalized device name in the model's own gfx_vocab, -1 (missing category) if absent.
MIOPEN_INTERNALS_EXPORT int GfxCode(const SolverModel& model, const std::string& device_name);

// True when the picker must not use `solver_name`'s model for (gfx_id, direction), whether or not
// lgbm_pcfg.bin has buckets for it. PickConfig then abstains and the solver runs its default
// config.
MIOPEN_INTERNALS_EXPORT bool HeldOutOfPicker(const std::string& solver_name,
                                             const std::string& gfx_id,
                                             conv::Direction direction);

// The score margin a pick must exceed over the solver's default config, for (solver, gfx_id,
// direction), or nullopt when the pick is used as ranked. PickConfig then returns only the
// candidates that score above the default's score plus this margin, so the caller's first-valid
// walk falls through to the default config when none of them is valid.
MIOPEN_INTERNALS_EXPORT std::optional<double>
DefaultMargin(const std::string& solver_name, const std::string& gfx_id, conv::Direction direction);

// The candidates of `ranked` whose score exceeds the score of `default_desc` by more than
// `margin`, in ranked order. Empty (abstain, the solver runs its default config) when
// `default_desc` is not among the ranked candidates.
MIOPEN_INTERNALS_EXPORT std::vector<std::string>
KeepAboveDefaultMargin(const RankedConfigs& ranked, const std::string& default_desc, double margin);

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP
