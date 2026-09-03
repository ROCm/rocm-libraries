// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// LayoutAliases.hpp - the `stride_order` layout-name pre-pass.
//
// A pure json -> json rewrite run before compilation, so the node tree and
// evaluation only ever see integer arrays. This is the one place in the
// language that knows anything about tensors.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Error.hpp>

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
// A `stride_order` is an IntArray: for each logical dimension d, that
// dimension's stride rank, 0 being the fastest-varying. The common layouts get
// names, and a name expands to its array here, at compile time, so the array
// stays the single canonical form and evaluation never sees an alias.

/// The longest layout the table names. Rows are fixed-width so the table can
/// be a constexpr value rather than pointers into separate objects.
inline constexpr std::size_t MAX_LAYOUT_RANK = 5;

struct LayoutAlias
{
    std::string_view name;
    /// The stride order, `rank` entries wide; anything past that is padding
    /// and is never read -- `order()` is the only accessor.
    std::array<std::int64_t, MAX_LAYOUT_RANK> dims;
    std::size_t rank;

    /// The meaningful prefix of `dims`, i.e. the layout itself.
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

/// The accepted names, for the diagnostic naming them. Built from the table so
/// a new alias cannot be added without the error message following it.
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
    // "$$x" is an escaped literal, and a bare "$" is rejected in compileNode.
    if(s.size() < 2 || s[0] != VARIABLE_SIGIL || s[1] == VARIABLE_SIGIL)
    {
        return nullptr;
    }
    return &s;
}

/// True for a string that can be read as a layout alias. A sigil-prefixed
/// string is a variable reference ("$k.stride_order") or an escaped literal
/// ("$$nhwc") -- both are strings, and neither names a layout, so comparing
/// one tensor's layout against another's must not be read as a typo'd alias.
inline bool isLayoutAliasCandidate(const nlohmann::json& j)
{
    if(!j.is_string())
    {
        return false;
    }
    const auto& s = j.get_ref<const nlohmann::json::string_t&>();
    return s.empty() || s[0] != VARIABLE_SIGIL;
}

/// True for a path whose last segment is `segment` -- ".stride_order" or
/// ".rank". A path is more than its final segment, so the segment alone
/// ("$.rank") does not name a tensor and does not match.
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
/// It has to be the whole prefix rather than the path's first segment: two
/// elements of one array share a root but are separate tensors of their own
/// ranks, so keying on "inputs" would let a rank pin on $inputs[0] veto a
/// layout alias on $inputs[1], which it does not constrain.
///
/// Both callers reach here only after matching their suffix, so the last '.'
/// is the separator before it and is always present.
inline std::string tensorKey(const std::string& sigilPath)
{
    const std::string path = sigilPath.substr(1);
    return path.substr(0, path.rfind('.'));
}

/// True when `j` is a numeric rank literal that can be carried exactly as an
/// int64_t pin. Floating-point inputs are accepted only when they are finite,
/// integral, and exactly representable.
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
        // bound plus the integrality check together guarantee the conversion
        // below is lossless.
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

/// Collect `{"==": ["$x.rank", N]}` rank pins that hold unconditionally: the
/// root, and anything reachable from it through `and` only. A pin inside an
/// `or` / `if` / `!` arm is conditional and cannot contradict an alias, so it
/// is deliberately not collected.
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
            // First pin wins; a second, contradictory one makes the criteria
            // unsatisfiable on its own terms, which is not the alias's problem.
            pins.emplace(tensorKey(*s), rank);
        }
    }
}

/// Resolve one alias string against a `stride_order` reference, or throw.
inline nlohmann::json resolveLayoutAlias(const nlohmann::json& aliasNode,
                                         const std::string& refPath,
                                         const std::map<std::string, std::int64_t>& rankPins)
{
    // A stride_order is an IntArray, so a string in this position can only be
    // an alias; an unknown one is a typo that would otherwise compare unequal
    // forever and decline silently at match time.
    const auto& name = aliasNode.get_ref<const nlohmann::json::string_t&>();
    const LayoutAlias* alias = lookupLayoutAlias(name);
    if(alias == nullptr)
    {
        throw JsonExpressionCompileError("unknown layout alias '" + name + "' compared against "
                                         + refPath + "; expected an integer array or one of: "
                                         + knownLayoutAliases());
    }
    // Every alias is fixed-rank, so an alias compared against a tensor the
    // criteria pin to a different rank can never hold. Refuse it here rather
    // than let it decline silently on every graph. The pin has to name this
    // same tensor: $inputs[0] and $inputs[1] are two of them.
    const auto pin = rankPins.find(tensorKey(refPath));
    if(pin != rankPins.end() && pin->second != static_cast<std::int64_t>(alias->rank))
    {
        throw JsonExpressionCompileError(
            "layout alias '" + name + "' is rank " + std::to_string(alias->rank)
            + ", but the expression pins " + refPath + " to rank " + std::to_string(pin->second));
    }
    return alias->order();
}

/// Rewrite every layout alias into its canonical array. An alias is recognized
/// only where a `stride_order` reference gives it that meaning -- opposite one
/// in an `==` / `!=`, or as an element of the array an `in` searches -- so
/// "nhwc" stays an ordinary string literal everywhere else.
///
/// Depth is charged exactly as Compiler.hpp charges it, because the two share
/// one MAX_EXPRESSION_DEPTH: an operator's argument array is not a level of
/// its own, so `{"!": [X]}` puts X one deeper, not two. Charging the array as
/// well would halve the effective limit for every rule -- and, since compile()
/// runs this pass first, would reject at that halved depth while reporting the
/// full documented limit.
inline nlohmann::json expandLayoutAliases(const nlohmann::json& j,
                                          const std::map<std::string, std::int64_t>& rankPins,
                                          std::size_t depth = 0);

/// Expand an operator's value the way compileObject descends into one: the
/// elements of an argument array sit one level below the operator, and a bare
/// non-array value sits one level below it too.
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
        // A bare array literal IS a level of its own, matching compileNode.
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
                    // An operand of the argument array: one level below the operator.
                    args.push_back(expandLayoutAliases(side, rankPins, depth + 1));
                }
            }
            out[key] = std::move(args);
            continue;
        }

        // {"in": [$x.stride_order, [<alias-or-array>, ...]]} -- the documented
        // way to accept a set of layouts. Only the haystack's own elements are
        // aliases; a nested expression there is left alone.
        if(binary && key == "in" && isStrideOrderRef(val.at(0)) && val.at(1).is_array())
        {
            const std::string refPath = val.at(0).get<std::string>();
            nlohmann::json hay = nlohmann::json::array();
            for(const auto& e : val.at(1))
            {
                // The haystack is an operand (depth + 1) and is itself an
                // array, so its elements are a further level down.
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
