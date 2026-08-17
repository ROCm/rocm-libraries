// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/config.hpp> // MIOPEN_INTERNALS_EXPORT

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

// Number of features the rank model consumes per row. Matches
// model_meta.json rank.feature_order length (41 base + 20 derived).
inline constexpr int kNumFeatures = 61;

// Singleton bundling the rank-model metadata (categorical vocabularies +
// solver list). Constructed lazily on first call to Get(); thread-safe via the
// Meyers idiom. Loaded from <GetSystemDbPath()>/lgbm_model_meta.json; if the
// file is missing or invalid, IsReady() returns false and the picker abstains.
class MIOPEN_INTERNALS_EXPORT LgbmMetadata
{
public:
    static const LgbmMetadata& Get();

    bool IsReady() const { return ready; }

    // Categorical vocabularies, keyed by column name (e.g. "solver_name").
    // The vector index doubles as the categorical code passed to the model.
    const std::unordered_map<std::string, std::vector<std::string>>& CategoricalVocab() const
    {
        return categorical_vocab;
    }

    // Master solver list (index doubles as the solver_name categorical code).
    const std::vector<std::string>& Solvers() const { return solvers; }

    // Helper: look up the integer code for a value in a categorical column.
    // Returns -1 (the missing sentinel) when the value is not in the vocab.
    int CategoricalCode(const std::string& column, const std::string& value) const;

    // Helper: look up the solver_name categorical code; -1 on miss.
    int SolverCode(const std::string& solver_name) const;

    // True if `solver_name` is one of the always-applicable naive fallback
    // solvers (rank.naive_fallback_solvers). These are demoted to the tail of
    // the ranked pick list so they are chosen only when nothing else applies.
    bool IsNaiveFallback(const std::string& solver_name) const;

    // Group-count threshold (rank.naive_guard_max_groups) below which the naive
    // demotion applies. At or above it, naive fallbacks keep their raw score
    // rank because naive is genuinely fastest on high-group convs. Defaults to
    // 64 when the key is absent.
    int NaiveGuardMaxGroups() const { return naive_guard_max_groups; }

    // True if the model was trained to handle architectures outside its gfx_id
    // vocab (rank.gfx_id_unseen_code present). Such a model was trained with
    // gfx_id feature-dropout, so routing an unknown arch through the missing
    // branch is meaningful: the continuous GPU-numeric features carry the arch
    // signal. When false, the picker abstains on an unknown arch instead.
    bool AllowUnseenArch() const { return allow_unseen_arch; }

    // Categorical code the model expects for an unseen gfx_id
    // (rank.gfx_id_unseen_code; -1 = the missing/NaN branch). Only meaningful
    // when AllowUnseenArch() is true.
    int UnseenArchCode() const { return unseen_arch_code; }

private:
    LgbmMetadata();

    bool ready = false;
    std::unordered_map<std::string, std::vector<std::string>> categorical_vocab;
    std::vector<std::string> solvers;
    std::unordered_map<std::string, int> solver_index;
    std::unordered_set<std::string> naive_fallback;
    int naive_guard_max_groups = 64;
    bool allow_unseen_arch     = false;
    int unseen_arch_code       = -1;
};

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP
