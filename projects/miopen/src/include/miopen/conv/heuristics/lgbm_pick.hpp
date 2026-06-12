#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/solver_id.hpp>

#include <string>
#include <vector>

namespace miopen {
struct Handle;
namespace conv {
struct ProblemDescription;
} // namespace conv

namespace ai {
namespace lgbm {

// Score the v10 rank model over the full solver vocabulary for `problem` on the
// GPU described by `handle`, and return the argmax solver's Id. Returns an
// invalid Id (== abstain) when:
//   - the metadata failed to load,
//   - the GPU's gfx_id is not in the trained vocab / arch-constants table,
//   - the model recommends a solver name this MIOpen build doesn't recognize.
solver::Id PickSolver(const conv::ProblemDescription& problem, const Handle& handle);

// Test seam: score a pre-built 51-feature row (solver_name column is
// overwritten internally per candidate) over the full vocab and return the
// argmax solver name. Lets gtest fixtures validate the scoring + argmax path
// against the reference feature matrix without a GPU. Returns "" if metadata
// is unavailable.
std::string ScoreRowArgmaxForTest(const std::vector<double>& feature_row);

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP
