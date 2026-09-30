// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/config.hpp> // MIOPEN_INTERNALS_EXPORT

#include <cstdint>
#include <vector>

namespace miopen {
struct Handle;
namespace conv {
struct ProblemDescription;
} // namespace conv

namespace ai {
namespace lgbm {

// Score the rank model over the full solver vocabulary for `problem` on the
// GPU described by `handle`, and return the solver IDs ranked by predicted speed
// (best first). Returns an empty vector (abstain / fall through) when the
// metadata failed to load or the GPU's gfx_id is not in the trained vocab (and
// unseen-arch support is off).
//
// No candidate masking or applicability check is done here: the caller walks the
// ranked list and applies IsApplicable lazily, so the picker only needs the
// Handle, not an ExecutionContext.
MIOPEN_INTERNALS_EXPORT std::vector<uint64_t>
PickSolverRanked(const conv::ProblemDescription& problem, const Handle& handle);

// Test seam: score a pre-built candidate matrix (each row is a full encoded
// feature vector; categoricals as integer codes, missing as NaN) and return the
// index of the argmax row, or -1 on error. Lets gtest validate the
// encoding+scoring+argmax against the reference fixture without a GPU.
MIOPEN_INTERNALS_EXPORT int
ScoreCandidateMatrixForTest(const std::vector<std::vector<double>>& candidate_rows);

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PICK_HPP
