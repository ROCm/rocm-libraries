#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <string>
#include <unordered_map>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

// Number of features the v16 rank model consumes per row. Matches
// model_meta.json rank.feature_order length.
inline constexpr int kNumFeatures = 41;

// Singleton bundling the rank-model metadata (categorical vocabularies +
// solver list). Constructed lazily on first call to Get(); thread-safe via the
// Meyers idiom.
//
// Loaded via GetSystemDbPath() at runtime: <SystemDbPath>/lgbm_model_meta.json.
// If the file is missing or invalid, IsReady() returns false and the picker
// abstains (returning solver::Id{}).
//
// v10 is runtime-pure: no spec_id, no triple_vocab masking, no margin/appl
// thresholds. The picker scores the full solver vocabulary and takes the
// global argmax, so only the vocab + solver list are needed here.
class LgbmMetadata
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
    // Returns -1 (Treelite missing sentinel) when the value is not in the
    // vocab.
    int CategoricalCode(const std::string& column, const std::string& value) const;

    // Helper: look up the solver_name categorical code; -1 on miss.
    int SolverCode(const std::string& solver_name) const;

private:
    LgbmMetadata();

    bool ready = false;
    std::unordered_map<std::string, std::vector<std::string>> categorical_vocab;
    std::vector<std::string> solvers;
    std::unordered_map<std::string, int> solver_index;
};

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP
