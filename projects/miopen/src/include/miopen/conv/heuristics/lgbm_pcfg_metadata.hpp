#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <string>
#include <unordered_map>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {
namespace pcfg {

// One candidate performance-config in a bucket: the verbatim descriptor string
// MIOpen's PerformanceConfig::Deserialize consumes, plus the pre-encoded arg
// feature values (in the solver's arg_cols order). The arg values were encoded
// at export time using the same vocab the model trained on, so the runtime does
// not need the vocab at all: it copies these straight into the feature row tail.
struct Candidate
{
    std::string desc;       // "" means "use the solver default config" (abstain)
    std::vector<double> args;
};

// Per-solver model metadata + candidate catalog. The compiled Treelite predict()
// for a solver indexes features by position over [prob_feat (27) | arg_cols (N)];
// prob_feat_count is the constant problem+GPU prefix shared by every solver, and
// feat_count == prob_feat_count + arg_count.
struct SolverModel
{
    int feat_count      = 0; // total columns the predict() consumes
    int prob_feat_count = 0; // constant problem+GPU prefix length (27)
    int arg_count       = 0; // per-solver candidate arg columns

    // bucket key "<gfx_id>|<direction>|<data_type>" -> candidate list
    std::unordered_map<std::string, std::vector<Candidate>> buckets;
};

// Number of problem+GPU prefix features, identical across all solvers. Asserted
// against the loaded metadata at construction.
inline constexpr int kNumProbFeatures = 27;

// Singleton bundling all per-solver perf-config models. Lazily constructed,
// thread-safe via the Meyers idiom. Loaded from GetSystemDbPath():
//   <SystemDbPath>/lgbm_pcfg_model_meta.json   (feat schema, per solver)
//   <SystemDbPath>/lgbm_pcfg_catalog.json      (per-bucket candidates)
// If either file is missing/invalid, IsReady() is false and the picker abstains.
class LgbmPcfgMetadata
{
public:
    static const LgbmPcfgMetadata& Get();

    bool IsReady() const { return ready; }

    // Look up the model for a solver (by solver_name). Returns nullptr when the
    // solver has no perf-config model.
    const SolverModel* Find(const std::string& solver_name) const;

private:
    LgbmPcfgMetadata();

    bool ready = false;
    std::unordered_map<std::string, SolverModel> models;
};

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP
