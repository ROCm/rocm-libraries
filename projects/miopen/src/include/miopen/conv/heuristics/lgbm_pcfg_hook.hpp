#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_HOOK_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_HOOK_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <string>

// Thin, header-light entry point for the perf-config picker, callable from the
// generic FindSolutionImpl template without dragging the conv-specific or
// nlohmann/json headers into every primitive's translation unit. The env gate
// (MIOPEN_DEBUG_LGBM_PCFG) lives in the .cpp so the env-var declaration has a
// single home.
namespace miopen {
struct Handle;
namespace conv {
struct ProblemDescription;
} // namespace conv

namespace ai {
namespace lgbm {
namespace pcfg {

// Returns a serializable perf-config string for `solver_db_id` on this problem +
// GPU, or "" to abstain (picker disabled, no model for the solver, unknown
// bucket, or the model chose the default config). Env-gated internally.
std::string MaybePickConfig(const std::string& solver_db_id,
                            const conv::ProblemDescription& problem,
                            const Handle& handle);

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_HOOK_HPP
