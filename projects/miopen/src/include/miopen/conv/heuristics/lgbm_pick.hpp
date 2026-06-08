#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/solver_id.hpp>

namespace miopen {
struct Handle;
namespace conv {
struct ProblemDescription;
} // namespace conv

namespace ai {
namespace lgbm {

// Score the LGBM rank + applicability models against `problem` for the GPU
// described by `handle`. Returns the chosen solver's Id on success, or an
// invalid Id (== abstain) when:
//   - the GPU's spec_id is not in the trained vocab,
//   - the rank model's top pick is too close to the runner-up (margin gate),
//   - the applicability model VETOes the top pick,
//   - the model recommends a solver name MIOpen doesn't recognize.
solver::Id PickSolver(const conv::ProblemDescription& problem, const Handle& handle);

// Test-only overload that bypasses GPU resolution and uses the supplied
// spec_id index directly (must be in [0, kNumSpecIds)). Lets gtest fixtures
// drive the picker without a real GPU.
solver::Id PickSolverForSpec(const conv::ProblemDescription& problem, int spec_id_code);

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP
