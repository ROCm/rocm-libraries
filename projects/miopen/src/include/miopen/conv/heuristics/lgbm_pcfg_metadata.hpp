// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/config.hpp> // MIOPEN_INTERNALS_EXPORT
#include <miopen/conv/heuristics/lgbm_forest.hpp>

#include <memory>
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
    std::string desc; // "" means "use the solver default config" (abstain)
    std::vector<double> args;
};

// Per-solver model metadata + candidate catalog. The model for a solver indexes
// features by position over [prob_feat | arg_cols (N)];
// prob_feat_count is the problem+GPU prefix length and feat_count ==
// prob_feat_count + arg_count. The base prefix is kNumBaseProbFeatures; some
// solvers append a trailing gfx_code categorical (has_gfx_code), making the
// prefix one longer.
struct SolverModel
{
    int feat_count      = 0;     // total columns the model consumes
    int prob_feat_count = 0;     // problem+GPU prefix length (base, or base+1)
    int arg_count       = 0;     // per-solver candidate arg columns
    bool has_gfx_code   = false; // prefix ends with the gfx_code categorical

    // The solver's LightGBM forest, walked at runtime (lgbm_forest.hpp). Held
    // by pointer so SolverModel stays cheap to move within the model map.
    std::shared_ptr<const LgbmForest> forest;

    // bucket key "<gfx_id>|<direction>|<data_type>" -> candidate list
    std::unordered_map<std::string, std::vector<Candidate>> buckets;
};

// Base problem+GPU prefix length (14 log-geom + 5 log-derived + 6 GPU numerics +
// direction + dtype_code). Solvers trained with PCFG_GFXID add a trailing
// gfx_code categorical, giving a prefix of kNumBaseProbFeatures + 1.
inline constexpr int kNumBaseProbFeatures = 27;

// Singleton bundling all per-solver perf-config models. Lazily constructed,
// thread-safe via the Meyers idiom. Loaded from GetSystemDbPath():
//   <SystemDbPath>/lgbm_pcfg_model_meta.json   (feat schema, per solver)
//   <SystemDbPath>/lgbm_pcfg_catalog.json      (per-bucket candidates)
// If either file is missing/invalid, IsReady() is false and the picker abstains.
class MIOPEN_INTERNALS_EXPORT LgbmPcfgMetadata
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
