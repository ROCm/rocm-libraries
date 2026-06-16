#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

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

// Pick the best performance-config for an already-chosen solver on the GPU
// described by `handle`. Scores every candidate config in the problem's
// (gfx_id, direction, data_type) bucket with the solver's lambdarank model and
// returns the argmax candidate's verbatim descriptor string (ready for
// PerformanceConfig::Deserialize). Returns "" (abstain) when:
//   - metadata failed to load,
//   - the solver has no perf-config model,
//   - the bucket is unknown / empty,
//   - the winning candidate is the default config ("" descriptor).
// `solver_name` must be the solver's registry name (solver::Id::ToString()).
std::string PickConfig(const std::string& solver_name,
                       const conv::ProblemDescription& problem,
                       const Handle& handle);

// Test seam: score a pre-built feature row (problem prefix already filled; the
// candidate arg tail is appended internally per candidate) for `solver_name`
// over an explicit candidate-arg list, returning the argmax descriptor. Lets
// gtest validate the scoring + argmax path against exported test vectors without
// a GPU or a ProblemDescription. Returns "" if the solver model is unavailable.
std::string ScorePickForTest(const std::string& solver_name,
                             const std::vector<double>& prob_feature_prefix,
                             const std::vector<std::string>& cand_descs,
                             const std::vector<std::vector<double>>& cand_args);

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_PICK_HPP
