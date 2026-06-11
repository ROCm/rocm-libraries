#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

// Number of features the rank + applicability models consume per row. Matches
// model_meta.json rank.feature_order length.
inline constexpr int kNumFeatures = 59;

// Singleton bundling the model metadata + per-spec thresholds. Constructed
// lazily on first call to Get(); thread-safe via the Meyers idiom.
//
// Files are located via GetSystemDbPath() at runtime:
//   <SystemDbPath>/lgbm_model_meta.json
//   <SystemDbPath>/lgbm_per_spec_thresh.json
//
// If either file is missing or invalid, IsReady() returns false and the
// picker should abstain (returning solver::Id{}).
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

    // Master solver list (52 entries) — index doubles as the solver_name
    // categorical code.
    const std::vector<std::string>& Solvers() const { return solvers; }

    // (spec_id, direction_int_as_string, data_type_string) -> candidate
    // solver names. Typical bucket has 5-15 of the 52 solvers.
    const std::unordered_map<std::string, std::vector<std::string>>& TripleVocab() const
    {
        return triple_vocab;
    }

    // Per-spec margin threshold (from per_spec_thresh.json). Falls back to
    // the global default when the spec_id has no entry.
    double MarginThresh(const std::string& spec_id) const;

    // Per-spec applicability threshold. Falls back to the global default.
    double ApplThresh(const std::string& spec_id) const;

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
    std::unordered_map<std::string, std::vector<std::string>> triple_vocab;
    std::unordered_map<std::string, double> per_spec_margin;
    std::unordered_map<std::string, double> per_spec_appl;
    double default_margin     = 1.7;
    double default_appl_prob  = 0.99;
};

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_METADATA_HPP
