// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_FOREST_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_FOREST_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/config.hpp>                       // MIOPEN_INTERNALS_EXPORT
#include <miopen/conv/heuristics/lgbm_predict.hpp> // LgbmEntry

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

// A gradient-boosted decision-tree forest parsed from a LightGBM text (`.txt`)
// model at runtime and walked here.
//
// Inference sums the reached leaf value across every tree. The training
// objective (lambdarank) has no output transform, so the raw sum is the score
// (higher = predicted faster). Feature rows use the LgbmEntry union (missing ==
// -1 marks NaN/absent; otherwise the numeric value, or a categorical code cast
// to double, lives in fvalue).
class MIOPEN_INTERNALS_EXPORT LgbmForest
{
public:
    // Parse a LightGBM text model file. On failure, IsReady() is false and
    // Score() returns 0 (caller treats an unusable model as "abstain").
    explicit LgbmForest(const std::string& model_path);

    bool IsReady() const { return ready_; }

    // Walk every tree and return the summed leaf value. `row` must point to at
    // least (max_feature_index + 1) entries; the caller owns feature layout.
    // Pointer+count so both std::array and std::vector rows work without a copy.
    double Score(const LgbmEntry* row, std::size_t n) const;

    // Lazily-loaded shared rank model (GetSystemDbPath()/lgbm_rank_model.txt),
    // mirroring LgbmMetadata::Get().
    static const LgbmForest& GetRank();

private:
    // A single node. For internal nodes, `left`/`right` index into nodes_ when
    // >= 0, or reference a leaf as ~child (LightGBM's convention) when < 0.
    struct Node
    {
        int split_feature; // feature index tested at this node
        double threshold;  // numeric split point (unused for categorical)
        int left;          // child encoding (see above)
        int right;
        int cat_index;     // >= 0: categorical split, index into cat blocks; -1: numeric
        bool default_left; // where a decided-missing feature goes
        int missing_type;  // 0=None, 1=Zero, 2=NaN (LightGBM decision_type bits 2-3)
    };

    struct Tree
    {
        std::vector<Node> nodes;         // internal nodes, node 0 is the root
        std::vector<double> leaf_values; // indexed by leaf id
        // Per-tree categorical bitsets: cat_bitset_[cat_offsets_[i] ..
        // cat_offsets_[i+1]) is the uint32 word run for categorical split i.
        std::vector<std::uint32_t> cat_bitset;
        std::vector<std::size_t> cat_offsets;
    };

    double ScoreTree(const Tree& tree, const LgbmEntry* row) const;

    std::vector<Tree> trees_;
    bool ready_ = false;
};

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_FOREST_HPP
