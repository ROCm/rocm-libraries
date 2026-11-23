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

// Per-node flag bits of the compact FOREST block.
constexpr std::uint8_t kFlagDefaultLeft     = 0x01;
constexpr int kFlagMissingTypeShift         = 1;
constexpr std::uint8_t kFlagMissingTypeMask = 0x03;
constexpr std::uint8_t kFlagCategorical     = 0x08;

// Child indices are stored as signed integers of the block's child width
// (1, 2 or 4 bytes); widen them to the in-memory int.
std::vector<std::int32_t> ReadChildren(BinReader& reader, std::uint8_t width, std::size_t n)
{
    std::vector<std::int32_t> out;
    switch(width)
    {
    case 1: {
        const auto v = reader.ReadArray<std::int8_t>(n);
        out.assign(v.begin(), v.end());
        break;
    }
    case 2: {
        const auto v = reader.ReadArray<std::int16_t>(n);
        out.assign(v.begin(), v.end());
        break;
    }
    default: out = reader.ReadArray<std::int32_t>(n); break;
    }
    return out;
}

// Leaf values are stored as f32 (default) or f64 and always scored in double.
std::vector<double> ReadLeaves(BinReader& reader, std::uint8_t width, std::size_t n)
{
    if(width == 8)
        return reader.ReadArray<double>(n);
    const auto v = reader.ReadArray<float>(n);
    return {v.begin(), v.end()};
}
} // namespace

LgbmForest::LgbmForest(BinReader& reader)
{
    // Compact FOREST block (see lgbm_binary.hpp and script/convert_lgbm_binary.py):
    // u32 num_trees, u32 num_thresholds, f64[num_thresholds] threshold table,
    // u8 child_width (1|2|4), u8 leaf_width (4|8), then per tree
    // u16 n (internal nodes; leaves = n + 1), u8[n] split_feature,
    // u16[n] threshold-table index (cat_offsets index for categorical nodes),
    // child_width[n] left, child_width[n] right, u8[n] flags, leaf_width[n+1]
    // leaf values, u16 + u32[] cat_bitset, u16 + u16[] cat_offsets.
    // Thresholds stay exact f64, so every split decides exactly as the trained
    // model does; only leaf values may be narrowed to f32.
    const std::uint32_t num_trees      = reader.ReadU32();
    const std::uint32_t num_thresholds = reader.ReadU32();
    const auto thresholds              = reader.ReadArray<double>(num_thresholds);
    const std::uint8_t child_width     = reader.ReadU8();
    const std::uint8_t leaf_width      = reader.ReadU8();
    bool valid = (child_width == 1 || child_width == 2 || child_width == 4) &&
                 (leaf_width == 4 || leaf_width == 8);

    for(std::uint32_t t = 0; valid && t < num_trees; ++t)
    {
        const std::uint16_t num_nodes = reader.ReadU16();
        const auto split_feature      = reader.ReadArray<std::uint8_t>(num_nodes);
        const auto split_index        = reader.ReadArray<std::uint16_t>(num_nodes);
        const auto left               = ReadChildren(reader, child_width, num_nodes);
        const auto right              = ReadChildren(reader, child_width, num_nodes);
        const auto flags              = reader.ReadArray<std::uint8_t>(num_nodes);
        auto leaf_values              = ReadLeaves(reader, leaf_width, std::size_t{num_nodes} + 1);

        const std::uint16_t cat_bitset_len = reader.ReadU16();
        auto cat_bitset                    = reader.ReadArray<std::uint32_t>(cat_bitset_len);

        const std::uint16_t cat_offsets_len = reader.ReadU16();
        const auto cat_offsets              = reader.ReadArray<std::uint16_t>(cat_offsets_len);

        // Any short read latched reader.Ok() to false and left the arrays empty;
        // stop before indexing them so a truncated file abstains cleanly.
        if(!reader.Ok())
            break;

        Tree tree;
        tree.leaf_values = std::move(leaf_values);
        tree.cat_bitset  = std::move(cat_bitset);
        tree.cat_offsets.assign(cat_offsets.begin(), cat_offsets.end());
        tree.nodes.resize(num_nodes);
        for(std::uint16_t i = 0; i < num_nodes; ++i)
        {
            Node node{};
            const bool categorical = (flags[i] & kFlagCategorical) != 0;
            node.split_feature     = split_feature[i];
            node.left              = left[i];
            node.right             = right[i];
            node.default_left      = (flags[i] & kFlagDefaultLeft) != 0;
            node.missing_type      = (flags[i] >> kFlagMissingTypeShift) & kFlagMissingTypeMask;
            if(categorical)
            {
                node.cat_index = split_index[i];
                node.threshold = 0.0;
            }
            else
            {
                if(split_index[i] >= num_thresholds)
                {
                    valid = false;
                    break;
                }
                node.cat_index = -1;
                node.threshold = thresholds[split_index[i]];
            }
            tree.nodes[i] = node;
        }
        if(valid)
            trees_.push_back(std::move(tree));
    }

    ready_ = valid && reader.Ok() && !trees_.empty();
    if(ready_)
        MIOPEN_LOG_I2("LGBM forest loaded: " << trees_.size() << " trees (binary)");
    else
        MIOPEN_LOG_W("LGBM forest: binary block truncated, corrupt or empty; picker will abstain");
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
        if(!reader.ReadMagic("MIORANK1", 8) || reader.ReadU32() != kRankFormatVersion)
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
