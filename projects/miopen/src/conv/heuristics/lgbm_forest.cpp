// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_forest.hpp>
#include <miopen/db_path.hpp>
#include <miopen/logger.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

namespace {

// LightGBM decision_type is a small bitmask (see LightGBM's Tree/DecisionType):
//   bit 0    -> split is categorical (else numeric)
//   bit 1    -> "default left": a decided-missing feature follows the left child
//   bits 2-3 -> missing_type: 0=None, 1=Zero, 2=NaN
// The missing_type matters: only NaN-type nodes treat a NaN input as missing
// and route it by the default-left bit. None/Zero-type nodes coerce a NaN input
// to 0.0 first and then either send an exact zero via default-left (Zero type)
// or fall through to the ordinary threshold compare.
constexpr int kCategoricalMask  = 0x01;
constexpr int kDefaultLeftMask  = 0x02;
constexpr int kMissingTypeShift = 2;
constexpr int kMissingTypeMask  = 0x03;
constexpr int kMissingTypeZero  = 1;
constexpr int kMissingTypeNaN   = 2;

// Split a "a b c" value line (already past the '=') into doubles.
std::vector<double> ParseDoubles(const std::string& s)
{
    std::vector<double> out;
    std::istringstream iss(s);
    std::string tok;
    while(iss >> tok)
        out.push_back(std::strtod(tok.c_str(), nullptr));
    return out;
}

std::vector<int> ParseInts(const std::string& s)
{
    std::vector<int> out;
    std::istringstream iss(s);
    std::string tok;
    while(iss >> tok)
        out.push_back(static_cast<int>(std::strtol(tok.c_str(), nullptr, 10)));
    return out;
}

std::vector<std::uint32_t> ParseU32(const std::string& s)
{
    std::vector<std::uint32_t> out;
    std::istringstream iss(s);
    std::string tok;
    while(iss >> tok)
        out.push_back(static_cast<std::uint32_t>(std::strtoul(tok.c_str(), nullptr, 10)));
    return out;
}

} // namespace

LgbmForest::LgbmForest(const std::string& model_path)
{
    std::ifstream in(model_path);
    if(!in.is_open())
    {
        MIOPEN_LOG_W("LGBM forest: cannot open " << model_path << "; picker will abstain");
        return;
    }

    // The LightGBM text dump is a sequence of "key=values" lines grouped into
    // per-tree blocks separated by blank lines. A block that opens with a
    // "Tree=<n>" line is one boosting round; everything before the first such
    // block (feature_names, objective, ...) and after the last (feature
    // importances, parameters) is ignored -- we only need the tree topology.
    std::string line;
    bool in_tree = false;
    std::vector<int> split_feature, decision_type, left_child, right_child;
    std::vector<double> threshold, leaf_value;
    std::vector<int> cat_boundaries;
    std::vector<std::uint32_t> cat_threshold;

    auto flush_tree = [&]() {
        if(!in_tree)
            return;
        Tree tree;
        tree.leaf_values = std::move(leaf_value);
        tree.cat_offsets.assign(cat_boundaries.begin(), cat_boundaries.end());
        tree.cat_bitset = std::move(cat_threshold);

        const std::size_t n_internal = split_feature.size();
        tree.nodes.resize(n_internal);
        for(std::size_t i = 0; i < n_internal; ++i)
        {
            Node node{};
            const int dt       = i < decision_type.size() ? decision_type[i] : 0;
            node.split_feature = split_feature[i];
            node.default_left  = (dt & kDefaultLeftMask) != 0;
            node.missing_type  = (dt >> kMissingTypeShift) & kMissingTypeMask;
            node.left          = i < left_child.size() ? left_child[i] : -1;
            node.right         = i < right_child.size() ? right_child[i] : -1;
            if((dt & kCategoricalMask) != 0)
            {
                // Categorical split: `threshold` holds the categorical-split
                // index, i.e. which cat_offsets[k]..[k+1] bitset run to test.
                node.cat_index = static_cast<int>(threshold[i]);
                node.threshold = 0.0;
            }
            else
            {
                node.cat_index = -1;
                node.threshold = i < threshold.size() ? threshold[i] : 0.0;
            }
            tree.nodes[i] = node;
        }
        trees_.push_back(std::move(tree));

        split_feature.clear();
        decision_type.clear();
        left_child.clear();
        right_child.clear();
        threshold.clear();
        leaf_value.clear();
        cat_boundaries.clear();
        cat_threshold.clear();
        in_tree = false;
    };

    while(std::getline(in, line))
    {
        if(line.empty())
        {
            flush_tree();
            continue;
        }
        const auto eq = line.find('=');
        if(eq == std::string::npos)
            continue;
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);

        if(key == "Tree")
        {
            flush_tree(); // in case blocks are not blank-separated
            in_tree = true;
        }
        else if(key == "split_feature")
            split_feature = ParseInts(val);
        else if(key == "decision_type")
            decision_type = ParseInts(val);
        else if(key == "left_child")
            left_child = ParseInts(val);
        else if(key == "right_child")
            right_child = ParseInts(val);
        else if(key == "threshold")
            threshold = ParseDoubles(val);
        else if(key == "leaf_value")
            leaf_value = ParseDoubles(val);
        else if(key == "cat_boundaries")
            cat_boundaries = ParseInts(val);
        else if(key == "cat_threshold")
            cat_threshold = ParseU32(val);
    }
    flush_tree(); // last tree if file didn't end on a blank line

    ready_ = !trees_.empty();
    if(ready_)
        MIOPEN_LOG_I2("LGBM forest loaded: " << trees_.size() << " trees from " << model_path);
    else
        MIOPEN_LOG_W("LGBM forest: no trees parsed from " << model_path << "; picker will abstain");
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
    static const LgbmForest instance((GetSystemDbPath() / "lgbm_rank_model.txt").string());
    return instance;
}

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
