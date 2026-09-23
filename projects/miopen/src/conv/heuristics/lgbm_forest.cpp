// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_forest.hpp>
#include <miopen/db_path.hpp>
#include <miopen/load_file.hpp>
#include <miopen/logger.hpp>

#include <cmath>
#include <cstdint>
#include <exception>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

namespace {
// missing_type values (decoded from LightGBM's decision_type bits 2-3 by the
// exporter): 1=Zero, 2=NaN. Only NaN-type nodes treat a NaN input as missing
// and route it by default_left; Zero-type nodes send an exact zero left.
constexpr int kMissingTypeZero = 1;
constexpr int kMissingTypeNaN  = 2;
} // namespace

LgbmForest::LgbmForest(BinReader& reader)
{
    // FOREST block: u32 num_trees, then per tree the decoded node arrays,
    // leaf values, and categorical bitset/offset runs. See lgbm_binary.hpp and
    // script/convert_lgbm_binary.py; the exporter already decodes LightGBM's
    // decision_type into split/cat/default_left/missing_type, so nothing is
    // decoded here.
    const std::uint32_t num_trees = reader.ReadU32();
    for(std::uint32_t t = 0; t < num_trees; ++t)
    {
        const std::uint32_t num_nodes = reader.ReadU32();
        const auto split_feature      = reader.ReadArray<std::int32_t>(num_nodes);
        const auto threshold          = reader.ReadArray<double>(num_nodes);
        const auto left               = reader.ReadArray<std::int32_t>(num_nodes);
        const auto right              = reader.ReadArray<std::int32_t>(num_nodes);
        const auto cat_index          = reader.ReadArray<std::int32_t>(num_nodes);
        const auto default_left       = reader.ReadArray<std::uint8_t>(num_nodes);
        const auto missing_type       = reader.ReadArray<std::uint8_t>(num_nodes);

        const std::uint32_t num_leaves = reader.ReadU32();
        auto leaf_values               = reader.ReadArray<double>(num_leaves);

        const std::uint32_t cat_bitset_len = reader.ReadU32();
        auto cat_bitset                    = reader.ReadArray<std::uint32_t>(cat_bitset_len);

        const std::uint32_t cat_offsets_len = reader.ReadU32();
        const auto cat_offsets              = reader.ReadArray<std::int64_t>(cat_offsets_len);

        // Any short read latched reader.Ok() to false and left the arrays empty;
        // stop before indexing them so a truncated file abstains cleanly.
        if(!reader.Ok())
            break;

        Tree tree;
        tree.leaf_values = std::move(leaf_values);
        tree.cat_bitset  = std::move(cat_bitset);
        tree.cat_offsets.assign(cat_offsets.begin(), cat_offsets.end());
        tree.nodes.resize(num_nodes);
        for(std::uint32_t i = 0; i < num_nodes; ++i)
        {
            Node node{};
            node.split_feature = split_feature[i];
            node.threshold     = threshold[i];
            node.left          = left[i];
            node.right         = right[i];
            node.cat_index     = cat_index[i];
            node.default_left  = default_left[i] != 0;
            node.missing_type  = missing_type[i];
            tree.nodes[i]      = node;
        }
        trees_.push_back(std::move(tree));
    }

    ready_ = reader.Ok() && !trees_.empty();
    if(ready_)
        MIOPEN_LOG_I2("LGBM forest loaded: " << trees_.size() << " trees (binary)");
    else
        MIOPEN_LOG_W("LGBM forest: binary block truncated or empty; picker will abstain");
}

double LgbmForest::ScoreTree(const Tree& tree, const LgbmEntry* row) const
{
    if(tree.nodes.empty())
        return tree.leaf_values.empty() ? 0.0 : tree.leaf_values.front();

    int node = 0;
    for(;;)
    {
        const Node& n         = tree.nodes[static_cast<std::size_t>(node)];
        const LgbmEntry& e    = row[static_cast<std::size_t>(n.split_feature)];
        const bool is_missing = (e.missing == -1); // caller marks NaN/absent

        bool go_left;
        if(n.cat_index >= 0)
        {
            // Categorical: a missing value is never in the set, so it takes the
            // right (not-in-set) branch -- matching LightGBM's CategoricalDecision.
            // Otherwise test membership in this split's bitset: a little-endian
            // run of uint32 words covering categories [0, 32*nwords), with
            // word = cat_bitset[base + (c>>5)], bit = c & 31.
            go_left = false;
            if(!is_missing)
            {
                const int c = static_cast<int>(e.fvalue);
                if(c >= 0)
                {
                    const std::size_t base =
                        tree.cat_offsets[static_cast<std::size_t>(n.cat_index)];
                    const std::size_t end =
                        tree.cat_offsets[static_cast<std::size_t>(n.cat_index) + 1];
                    const std::size_t wi = static_cast<std::size_t>(c) >> 5;
                    if(base + wi < end)
                        go_left = ((tree.cat_bitset[base + wi] >> (c & 31)) & 1u) != 0u;
                }
            }
        }
        else
        {
            // Numeric split, following LightGBM's NumericalDecision. Only a
            // NaN-type node treats a NaN input as "missing" (routed by
            // default_left). A None/Zero-type node coerces the NaN to 0.0 first;
            // a Zero-type node then routes an exact zero by default_left, and
            // everything else is the ordinary threshold compare.
            const double fval          = is_missing ? 0.0 : e.fvalue;
            const bool decided_missing = (is_missing && n.missing_type == kMissingTypeNaN) ||
                                         (n.missing_type == kMissingTypeZero && fval == 0.0);
            go_left = decided_missing ? n.default_left : (fval <= n.threshold);
        }

        const int child = go_left ? n.left : n.right;
        if(child < 0)
        {
            // LightGBM encodes a leaf child as ~leaf_index, so leaf_index is
            // -child - 1. Compute in the signed domain, then index.
            return tree.leaf_values[static_cast<std::size_t>(-child - 1)];
        }
        node = child;
    }
}

double LgbmForest::Score(const LgbmEntry* row, std::size_t /*n*/) const
{
    double sum = 0.0;
    for(const auto& tree : trees_)
        sum += ScoreTree(tree, row);
    return sum;
}

const LgbmForest& LgbmForest::GetRank()
{
    static const std::vector<char> buffer = [] {
        try
        {
            return LoadFile(GetSystemDbPath() / "lgbm_rank.bin");
        }
        catch(const std::exception& e)
        {
            MIOPEN_LOG_W("LGBM forest: cannot load lgbm_rank.bin (" << e.what()
                                                                    << "); picker will abstain");
            return std::vector<char>{};
        }
    }();
    static const LgbmForest instance = [] {
        BinReader reader(buffer.data(), buffer.size());
        // "MIORANK1" + u32 version, then a bare FOREST block. A bad header
        // forces the reader to EOF so the forest read yields a not-ready model.
        if(!reader.ReadMagic("MIORANK1", 8) || reader.ReadU32() != kBinaryFormatVersion)
        {
            MIOPEN_LOG_W("LGBM forest: lgbm_rank.bin bad magic/version; picker will abstain");
            reader.SeekTo(buffer.size());
        }
        return LgbmForest(reader);
    }();
    return instance;
}

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
