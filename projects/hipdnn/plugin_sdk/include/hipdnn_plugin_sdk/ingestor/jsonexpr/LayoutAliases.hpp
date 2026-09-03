// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// LayoutAliases.hpp - the `stride_order` layout-name pre-pass.
//
// A json -> json rewrite that runs before compilation, so the node tree and
// evaluation only ever see integer arrays. This is the one part of the
// language that knows anything about tensors.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Error.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Syntax.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail
{
// ---- layout aliases -------------------------------------------------------
// A `stride_order` is an array of integers: for each logical dimension, that
// dimension's stride rank, where 0 is the fastest-varying. The common layouts
// have names, and a name is expanded to its array here, at compile time, so
// the array stays the single canonical form.

/// The longest layout the table names. Rows are fixed-width so the table can
/// be a constexpr value instead of pointers to separate objects.
inline constexpr std::size_t MAX_LAYOUT_RANK = 5;

struct LayoutAlias
{
    std::string_view name;
    /// The stride order. Only the first `rank` entries are meaningful; the
    /// rest is padding and is never read, since `order()` is the only accessor.
    std::array<std::int64_t, MAX_LAYOUT_RANK> dims;
    std::size_t rank;

    /// The first `rank` entries of `dims`, which is the layout itself.
    [[nodiscard]] std::vector<std::int64_t> order() const
    {
        return {dims.begin(), dims.begin() + static_cast<std::ptrdiff_t>(rank)};
    }
};

/// Every layout name the language knows, and the array each expands to.
inline constexpr std::array<LayoutAlias, 5> LAYOUT_ALIAS_TABLE = {{{"nchw", {3, 2, 1, 0}, 4},
                                                                   {"nhwc", {3, 0, 2, 1}, 4},
                                                                   {"ncdhw", {4, 3, 2, 1, 0}, 5},
                                                                   {"ndhwc", {4, 0, 3, 2, 1}, 5},
                                                                   {"bhsd", {3, 2, 1, 0}, 4}}};

inline const LayoutAlias* lookupLayoutAlias(const std::string& name)
{
    for(const auto& e : LAYOUT_ALIAS_TABLE)
    {
        if(name == e.name)
        {
            return &e;
        }
    }
    return nullptr;
}

/// The accepted names, for use in the error message that lists them. Built
/// from the table so a new alias always appears in the diagnostic.
inline std::string knownLayoutAliases()
{
    std::string s;
    for(const auto& e : LAYOUT_ALIAS_TABLE)
    {
        if(!s.empty())
        {
            s += ", ";
        }
        s += e.name;
    }
    return s;
}

/// The variable path in a sigil-prefixed string, or nullptr if `j` is not one.
inline const std::string* variablePath(const nlohmann::json& j)
{
    if(!j.is_string())
    {
        return nullptr;
    }
    const auto& s = j.get_ref<const nlohmann::json::string_t&>();
    // "$$x" is an escaped literal, and compileNode rejects a bare "$".
    if(s.size() < 2 || s[0] != VARIABLE_SIGIL || s[1] == VARIABLE_SIGIL)
    {
        return nullptr;
    }
    return &s;
}

/// True for a string that may be a layout alias. A sigil-prefixed string is
/// either a variable reference ("$k.stride_order") or an escaped literal
/// ("$$nhwc"), and neither names a layout. Without this check, comparing one
/// tensor's layout against another's would be read as a misspelled alias.
inline bool isLayoutAliasCandidate(const nlohmann::json& j)
{
    if(!j.is_string())
    {
        return false;
    }
    const auto& s = j.get_ref<const nlohmann::json::string_t&>();
    return s.empty() || s[0] != VARIABLE_SIGIL;
}

/// True for a path whose last segment is `segment`, such as ".stride_order" or
/// ".rank". A path needs more than the segment itself, so "$.rank" names no
/// tensor and does not match.
inline bool pathEndsWithSegment(const std::string& path, std::string_view segment)
{
    return path.size() > segment.size() + 1
           && path.compare(path.size() - segment.size(), segment.size(), segment) == 0;
}

/// True for a reference whose last path segment is `stride_order`.
inline bool isStrideOrderRef(const nlohmann::json& j)
{
    const std::string* s = variablePath(j);
    return s != nullptr && pathEndsWithSegment(*s, ".stride_order");
}

/// The tensor a `.rank` / `.stride_order` reference is about: the whole path
/// ahead of that final segment, sigil dropped. "$q.stride_order" -> "q",
/// "$inputs[1].rank" -> "inputs[1]", "$a.b.c.rank" -> "a.b.c".
///
/// The whole prefix is used rather than the path's first segment, because two
/// elements of one array share a root but are separate tensors with their own
/// ranks. Keying on "inputs" would let a rank pin on $inputs[0] veto a layout
/// alias on $inputs[1], which it does not constrain.
///
/// Both callers only get here after matching their suffix, so the last '.' is
/// the separator before that suffix and is always present.
inline std::string tensorKey(const std::string& sigilPath)
{
    const std::string path = sigilPath.substr(1);
    return path.substr(0, path.rfind('.'));
}

/// True when `j` is a numeric rank literal that fits an int64_t pin exactly.
/// A floating-point input is accepted only when it is finite, integral, and
/// exactly representable.
inline bool rankPinLiteral(const nlohmann::json& j, std::int64_t& value)
{
    if(j.is_number_unsigned())
    {
        const auto raw = j.get<nlohmann::json::number_unsigned_t>();
        if(raw > static_cast<nlohmann::json::number_unsigned_t>(
               std::numeric_limits<std::int64_t>::max()))
        {
            return false;
        }
        value = static_cast<std::int64_t>(raw);
        return true;
    }
    if(j.is_number_integer())
    {
        value = j.get<std::int64_t>();
        return true;
    }
    if(j.is_number_float())
    {
        // Within +/-2^53 every integral double is exactly an int64_t, so the
        // range check plus the integrality check make the conversion below
        // lossless.
        const double raw = j.get<double>();
        constexpr double maxExactInteger = 9007199254740992.0; // 2^53
        if(!std::isfinite(raw) || raw < -maxExactInteger || raw > maxExactInteger)
        {
            return false;
        }
        if(raw != std::trunc(raw))
        {
            return false;
        }
        value = static_cast<std::int64_t>(raw);
        return true;
    }
    return false;
}

/// Collect `{"==": ["$x.rank", N]}` rank pins that always hold: those at the
/// root, and those reachable from it through `and` alone. A pin inside an
/// `or`, `if`, or `!` arm is conditional and cannot contradict an alias, so it
/// is deliberately skipped.
inline void collectRankPins(const nlohmann::json& j,
                            std::map<std::string, std::int64_t>& pins,
                            std::size_t depth = 0)
{
    checkExpressionDepth(depth);
    if(!j.is_object() || j.size() != 1)
    {
        return;
    }
    const auto it = j.begin();
    const std::string& key = it.key();
    const nlohmann::json& val = it.value();
    if(key == "and")
    {
        if(val.is_array())
        {
            for(const auto& e : val)
            {
                collectRankPins(e, pins, depth + 1);
            }
        }
        else
        {
            collectRankPins(val, pins, depth + 1);
        }
        return;
    }
    if(key != "==" || !val.is_array() || val.size() != 2)
    {
        return;
    }
    for(std::size_t i = 0; i < 2; ++i)
    {
        const std::string* s = variablePath(val.at(i));
        std::int64_t rank = 0;
        if(s == nullptr || !rankPinLiteral(val.at(1 - i), rank))
        {
            continue;
        }
        if(pathEndsWithSegment(*s, ".rank"))
        {
            // First pin wins. A second, contradictory pin makes the criteria
            // unsatisfiable on their own, which is not this pass's problem.
            pins.emplace(tensorKey(*s), rank);
        }
    }
}

/// Resolve one alias string against a `stride_order` reference, or throw.
inline nlohmann::json resolveLayoutAlias(const nlohmann::json& aliasNode,
                                         const std::string& refPath,
                                         const std::map<std::string, std::int64_t>& rankPins)
{
    // A stride_order is an array of integers, so a string in this position can
    // only be an alias. An unknown one is a typo, which would otherwise
    // compare unequal forever and decline silently at match time.
    const auto& name = aliasNode.get_ref<const nlohmann::json::string_t&>();
    const LayoutAlias* alias = lookupLayoutAlias(name);
    if(alias == nullptr)
    {
        throw JsonExpressionCompileError("unknown layout alias '" + name + "' compared against "
                                         + refPath + "; expected an integer array or one of: "
                                         + knownLayoutAliases());
    }
    // Every alias has a fixed rank, so an alias compared against a tensor the
    // criteria pin to a different rank can never hold. Reject it here instead
    // of declining silently on every graph. The pin must name this same
    // tensor: $inputs[0] and $inputs[1] are two different ones.
    const auto pin = rankPins.find(tensorKey(refPath));
    if(pin != rankPins.end() && pin->second != static_cast<std::int64_t>(alias->rank))
    {
        throw JsonExpressionCompileError(
            "layout alias '" + name + "' is rank " + std::to_string(alias->rank)
            + ", but the expression pins " + refPath + " to rank " + std::to_string(pin->second));
    }
    return alias->order();
}

/// Rewrite every layout alias into its array form. An alias is only recognized
/// where a `stride_order` reference gives it that meaning: opposite one in an
/// `==` or `!=`, or as an element of the array an `in` searches. Everywhere
/// else "nhwc" stays an ordinary string literal.
///
/// This pass counts depth exactly as Compiler.hpp does, because the two share
/// one MAX_EXPRESSION_DEPTH. An operator's argument array is not a level of its
/// own, so `{"!": [X]}` puts X one level deeper, not two. Counting the array as
/// well would halve the effective limit, and because compile() runs this pass
/// first, rules would be rejected at that halved depth while the error still
/// named the documented limit.
inline nlohmann::json expandLayoutAliases(const nlohmann::json& j,
                                          const std::map<std::string, std::int64_t>& rankPins,
                                          std::size_t depth = 0);

/// Expand an operator's value the way compileObject descends into one. Both
/// the elements of an argument array and a bare non-array value sit one level
/// below the operator.
inline nlohmann::json expandOperatorValue(const nlohmann::json& val,
                                          const std::map<std::string, std::int64_t>& rankPins,
                                          std::size_t depth)
{
    if(!val.is_array())
    {
        return expandLayoutAliases(val, rankPins, depth + 1);
    }
    nlohmann::json out = nlohmann::json::array();
    for(const auto& e : val)
    {
        out.push_back(expandLayoutAliases(e, rankPins, depth + 1));
    }
    return out;
}

inline nlohmann::json expandLayoutAliases(const nlohmann::json& j,
                                          const std::map<std::string, std::int64_t>& rankPins,
                                          std::size_t depth)
{
    checkExpressionDepth(depth);
    if(j.is_array())
    {
        // A bare array literal is a level of its own, matching compileNode.
        nlohmann::json out = nlohmann::json::array();
        for(const auto& e : j)
        {
            out.push_back(expandLayoutAliases(e, rankPins, depth + 1));
        }
        return out;
    }
    if(!j.is_object())
    {
        return j;
    }

    nlohmann::json out = nlohmann::json::object();
    for(auto it = j.begin(); it != j.end(); ++it)
    {
        const std::string& key = it.key();
        const nlohmann::json& val = it.value();
        const bool binary = val.is_array() && val.size() == 2;

        // {"==" / "!=": [$x.stride_order, <alias>]}, either operand order.
        if(binary && (key == "==" || key == "!="))
        {
            nlohmann::json args = nlohmann::json::array();
            for(std::size_t i = 0; i < 2; ++i)
            {
                const nlohmann::json& side = val.at(i);
                const nlohmann::json& ref = val.at(1 - i);
                if(isStrideOrderRef(ref) && isLayoutAliasCandidate(side))
                {
                    args.push_back(resolveLayoutAlias(side, ref.get<std::string>(), rankPins));
                }
                else
                {
                    // An argument-array operand: one level below the operator.
                    args.push_back(expandLayoutAliases(side, rankPins, depth + 1));
                }
            }
            out[key] = std::move(args);
            continue;
        }

        // {"in": [$x.stride_order, [<alias-or-array>, ...]]} is the documented
        // way to accept a set of layouts. Only the haystack's own elements can
        // be aliases; a nested expression there is left alone.
        if(binary && key == "in" && isStrideOrderRef(val.at(0)) && val.at(1).is_array())
        {
            const std::string refPath = val.at(0).get<std::string>();
            nlohmann::json hay = nlohmann::json::array();
            for(const auto& e : val.at(1))
            {
                // The haystack is an operand (depth + 1) and is itself an
                // array, so its elements are one level below that.
                hay.push_back(isLayoutAliasCandidate(e)
                                  ? resolveLayoutAlias(e, refPath, rankPins)
                                  : expandLayoutAliases(e, rankPins, depth + 2));
            }
            out[key] = nlohmann::json::array({val.at(0), std::move(hay)});
            continue;
        }

        out[key] = expandOperatorValue(val, rankPins, depth);
    }
    return out;
}
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
