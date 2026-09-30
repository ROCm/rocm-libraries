// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/config.hpp> // MIOPEN_INTERNALS_EXPORT
#include <miopen/conv/heuristics/lgbm_forest.hpp>
#include <miopen/filesystem.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
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

    // gfx_code categories in the order the model was trained with: a gfx_id's
    // code is its index here, an unknown gfx_id is -1 (the missing category).
    // Non-empty exactly when has_gfx_code.
    std::vector<std::string> gfx_vocab;

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
// thread-safe via the Meyers idiom. Loaded from
// <GetSystemDbPath()>/lgbm_pcfg.bin (see lgbm_binary.hpp). If the file is
// missing/invalid, IsReady() is false and the picker abstains.
//
// Only the per-solver directory (name -> byte range) is read up front; each
// solver's section (forest + candidate buckets) is read from disk and parsed on
// the first Find() for that solver, then cached. A conv find typically queries
// only one or two solvers, so this reads a few MB rather than the whole ~36 MB
// bundle, and holds only the parsed subset resident.
class MIOPEN_INTERNALS_EXPORT LgbmPcfgMetadata
{
public:
    static const LgbmPcfgMetadata& Get();

    bool IsReady() const { return ready; }

    // Look up the model for a solver (by solver_name). Reads + parses + caches
    // the solver's section on first call. Returns nullptr when the solver has no
    // perf-config model (or its section fails to load). The returned pointer is
    // stable for the process lifetime. Thread-safe.
    const SolverModel* Find(const std::string& solver_name) const;

    // Names of all per-solver models available in the bundle (unordered), whether
    // or not they have been parsed yet. For test enumeration and diagnostics; the
    // runtime path uses Find().
    std::vector<std::string> SolverNames() const;

private:
    LgbmPcfgMetadata();

    // Read [offset, offset+size) of the bundle from disk and parse it into `out`
    // (feature counts + FOREST + buckets). Returns false on a read/parse failure.
    bool LoadSection(std::uint64_t offset, std::uint64_t size, SolverModel& out) const;

    struct Section
    {
        std::uint64_t offset;
        std::uint64_t size;
    };

    bool ready = false;
    fs::path bin_path;                                  // path to lgbm_pcfg.bin
    std::unordered_map<std::string, Section> directory; // solver_name -> byte range

    // Lazily-parsed sections, guarded by mutex. mutable so Find() stays const.
    // unordered_map node pointers are stable across inserts, so a pointer handed
    // back from Find() remains valid as other solvers are cached later.
    mutable std::mutex mutex;
    mutable std::unordered_map<std::string, SolverModel> cache;
};

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_PCFG_METADATA_HPP
