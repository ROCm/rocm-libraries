// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/node_operands_generated.h>

namespace hipdnn_plugin_sdk::heuristics
{

/// Why a published feature can be absent on a valid graph.
enum class OptionalFeatureKind
{
    ATTRIBUTE, ///< an optional scalar attribute the graph leaves unset
    TENSOR, ///< every field of an optional tensor operand the graph leaves out
    TENSORS, ///< every field of an element of a vector of tensor operands
    ELEMENTS, ///< an element of a vector attribute, whose length the schema leaves open
    WORK, ///< a work feature, unbound when the graph's work is unknown (RFC 0019 section 6.8)
};

inline std::string_view toString(OptionalFeatureKind kind)
{
    switch(kind)
    {
    case OptionalFeatureKind::ATTRIBUTE:
        return "attribute";
    case OptionalFeatureKind::TENSOR:
        return "tensor";
    case OptionalFeatureKind::TENSORS:
        return "tensors";
    case OptionalFeatureKind::ELEMENTS:
        return "elements";
    case OptionalFeatureKind::WORK:
        return "work";
    default:
        return "unknown";
    }
}

/// A family of published features (EngineFeatures.hpp) a valid graph may leave unbound, so a
/// UHD must read it through `value_or_default` or behind `present` (RFC 0019 section 6.2).
struct OptionalFeature
{
    /// The published name without `$`: `[*]` stands for any index, and a trailing `.*` for
    /// any name below the prefix (`graph.nodes[*].attn_mask.*`).
    std::string pattern;
    OptionalFeatureKind kind;
    /// The node types whose schema makes it optional; empty for a work feature. A name may
    /// also be required in other node types: a feature's name does not say its node's type.
    std::vector<std::string> nodeTypes;
};

/// The work features problemFeatures() leaves unbound when a graph's work is unknown
/// (EngineFeatures.hpp); keep in step with it.
inline constexpr std::array<std::string_view, 5> WORK_FEATURE_PATTERNS{{
    "graph.flops",
    "graph.flops_by_type.*",
    "graph.logical_bytes",
    "graph.arithmetic_intensity",
    "graph.nodes[*].flops",
}};

/// Whether ranking metric @p metric is computed from the graph's logical work. Where the
/// work is unknown, so is such a metric, so a model of it rightly reports UNAVAILABLE there
/// and may read the work features bare.
inline bool metricIsDerivedFromWork(std::string_view metric)
{
    return metric == "tflops";
}

namespace detail
{

inline OptionalFeatureKind
    optionalKindOf(hipdnn_flatbuffers_sdk::data_objects::node_operands::OperandKind kind)
{
    using hipdnn_flatbuffers_sdk::data_objects::node_operands::OperandKind;
    switch(kind)
    {
    case OperandKind::TENSOR:
        return OptionalFeatureKind::TENSOR;
    case OperandKind::TENSORS:
        return OptionalFeatureKind::TENSORS;
    case OperandKind::ELEMENTS:
        return OptionalFeatureKind::ELEMENTS;
    case OperandKind::SCALAR:
    default:
        return OptionalFeatureKind::ATTRIBUTE;
    }
}

/// The node index and operand @p reference reads (`$graph.nodes[3].attn_mask.dims[2]` reads
/// operand `attn_mask` of node `3`), or nullopt for a reference to no node's operand.
inline std::optional<std::pair<std::string, std::string>> nodeOperandOf(std::string_view reference)
{
    constexpr std::string_view NODES = "$graph.nodes[";
    if(reference.substr(0, NODES.size()) != NODES)
    {
        return std::nullopt;
    }
    const auto close = reference.find(']', NODES.size());
    if(close == std::string_view::npos || close + 1 >= reference.size()
       || reference[close + 1] != '.')
    {
        return std::nullopt;
    }
    const auto rest = reference.substr(close + 2);
    return std::make_pair(std::string(reference.substr(NODES.size(), close - NODES.size())),
                          std::string(rest.substr(0, rest.find_first_of(".["))));
}

/// The node types having every operand in @p names. A name no node type has as an operand
/// (`type`, `flops`, `data_dependent`) constrains nothing.
inline std::set<std::string_view> candidateNodeTypes(const std::set<std::string>& names)
{
    using hipdnn_flatbuffers_sdk::data_objects::node_operands::OPERANDS;
    std::set<std::string_view> types;
    for(const auto& operand : OPERANDS)
    {
        types.insert(operand.nodeType);
    }
    const auto has = [](std::string_view type, std::string_view name) {
        return std::any_of(OPERANDS.begin(), OPERANDS.end(), [&](const auto& operand) {
            return (type.empty() || operand.nodeType == type) && operand.name == name;
        });
    };
    std::set<std::string_view> candidates;
    for(const auto type : types)
    {
        if(std::all_of(names.begin(), names.end(), [&](const auto& name) {
               return !has({}, name) || has(type, name);
           }))
        {
            candidates.insert(type);
        }
    }
    return candidates;
}

/// How NodeFeatureBinder publishes an operand of @p kind named @p name.
inline std::string operandPattern(std::string_view name, OptionalFeatureKind kind)
{
    std::string pattern = "graph.nodes[*].";
    pattern += name;
    switch(kind)
    {
    case OptionalFeatureKind::TENSOR:
        pattern += ".*";
        break;
    case OptionalFeatureKind::TENSORS:
        pattern += "[*].*";
        break;
    case OptionalFeatureKind::ELEMENTS:
        pattern += "[*]";
        break;
    default:
        break;
    }
    return pattern;
}

/// Whether published @p name (no `$`) is one @p pattern stands for.
inline bool matchesPattern(std::string_view pattern, std::string_view name)
{
    size_t p = 0;
    size_t n = 0;
    while(p < pattern.size())
    {
        if(pattern.compare(p, 3, "[*]") == 0)
        {
            if(n >= name.size() || name[n] != '[')
            {
                return false;
            }
            size_t digit = n + 1;
            while(digit < name.size() && std::isdigit(static_cast<unsigned char>(name[digit])) != 0)
            {
                ++digit;
            }
            if(digit == n + 1 || digit >= name.size() || name[digit] != ']')
            {
                return false;
            }
            n = digit + 1;
            p += 3;
            continue;
        }
        if(p + 2 == pattern.size() && pattern.compare(p, 2, ".*") == 0)
        {
            return n + 1 < name.size() && name[n] == '.';
        }
        if(n >= name.size() || pattern[p] != name[n])
        {
            return false;
        }
        ++p;
        ++n;
    }
    return n == name.size();
}

using ReferenceSet = std::set<std::string>;

/// The operand @p reference reads below, as `$graph.nodes[i].<role>.`, or "" when it is not
/// a field of a node operand: proving one field of an operand bound proves the operand present.
inline std::string operandPrefix(std::string_view reference)
{
    constexpr std::string_view NODES = "$graph.nodes[";
    if(reference.substr(0, NODES.size()) != NODES)
    {
        return {};
    }
    size_t dots = 0;
    for(size_t index = 0; index < reference.size(); ++index)
    {
        if(reference[index] == '.' && ++dots == 3)
        {
            return std::string(reference.substr(0, index + 1));
        }
    }
    return {};
}

/// Adds @p bound, and the operands it proves present, to @p scope.
inline void addProven(ReferenceSet& scope, const ReferenceSet& bound)
{
    for(const auto& reference : bound)
    {
        scope.insert(reference);
        if(auto prefix = operandPrefix(reference); !prefix.empty())
        {
            scope.insert(std::move(prefix));
        }
    }
}

inline bool isGuarded(const std::string& reference, const ReferenceSet& scope)
{
    if(scope.count(reference) != 0)
    {
        return true;
    }
    const auto prefix = operandPrefix(reference);
    return !prefix.empty() && scope.count(prefix) != 0;
}

/// The arguments of operator object @p node: an array, or a single argument.
inline std::vector<nlohmann::json> operatorArguments(const nlohmann::json& node)
{
    const auto& arguments = node.begin().value();
    if(arguments.is_array())
    {
        return arguments.get<std::vector<nlohmann::json>>();
    }
    return {arguments};
}

inline void collectReferences(const nlohmann::json& node, ReferenceSet& references)
{
    if(node.is_string())
    {
        const auto& text = node.get_ref<const std::string&>();
        if(text.size() > 1 && text.front() == '$')
        {
            references.insert(text);
        }
    }
    else if(node.is_array() || node.is_object())
    {
        for(const auto& child : node)
        {
            collectReferences(child, references);
        }
    }
}

/// The references @p condition proves bound when it is true (`present`, `and` of them), or,
/// with @p negated, when it is false (`not_present`, `or` of them).
inline ReferenceSet provenBound(const nlohmann::json& condition, bool negated)
{
    ReferenceSet bound;
    if(!condition.is_object() || condition.size() != 1)
    {
        return bound;
    }
    const auto& op = condition.begin().key();
    if(op == (negated ? "not_present" : "present"))
    {
        collectReferences(condition.begin().value(), bound);
    }
    else if(op == (negated ? "or" : "and"))
    {
        for(const auto& argument : operatorArguments(condition))
        {
            const auto inner = provenBound(argument, negated);
            bound.insert(inner.begin(), inner.end());
        }
    }
    return bound;
}

/// Adds to @p unguarded every reference @p node reads where an unbound value would leave it
/// unresolved: not the first argument of `value_or_default`, not an argument of `present` /
/// `not_present`, and not under a condition that has proven it bound (`if`, `and`, `or`).
inline void collectUnguardedReads(const nlohmann::json& node,
                                  const ReferenceSet& guarded,
                                  ReferenceSet& unguarded)
{
    if(node.is_string())
    {
        const auto& text = node.get_ref<const std::string&>();
        if(text.size() > 1 && text.front() == '$' && !isGuarded(text, guarded))
        {
            unguarded.insert(text);
        }
        return;
    }
    if(node.is_array())
    {
        for(const auto& child : node)
        {
            collectUnguardedReads(child, guarded, unguarded);
        }
        return;
    }
    if(!node.is_object() || node.size() != 1)
    {
        return;
    }
    const auto& op = node.begin().key();
    const auto arguments = operatorArguments(node);
    if(op == "present" || op == "not_present")
    {
        return;
    }
    if(op == "value_or_default")
    {
        for(size_t index = 1; index < arguments.size(); ++index)
        {
            collectUnguardedReads(arguments[index], guarded, unguarded);
        }
        return;
    }
    if(op == "if" || op == "?:")
    {
        // [condition, then, condition, then, ..., else]: an arm runs only when its condition
        // holds and every earlier one failed.
        auto scope = guarded;
        size_t index = 0;
        for(; index + 1 < arguments.size(); index += 2)
        {
            collectUnguardedReads(arguments[index], scope, unguarded);
            auto taken = scope;
            addProven(taken, provenBound(arguments[index], false));
            collectUnguardedReads(arguments[index + 1], taken, unguarded);
            addProven(scope, provenBound(arguments[index], true));
        }
        if(index < arguments.size())
        {
            collectUnguardedReads(arguments[index], scope, unguarded);
        }
        return;
    }
    if(op == "and" || op == "or")
    {
        // Three-valued: a later argument matters only once the earlier ones held (`and`) or
        // failed (`or`).
        auto scope = guarded;
        for(const auto& argument : arguments)
        {
            collectUnguardedReads(argument, scope, unguarded);
            addProven(scope, provenBound(argument, op == "or"));
        }
        return;
    }
    for(const auto& argument : arguments)
    {
        collectUnguardedReads(argument, guarded, unguarded);
    }
}

} // namespace detail

/// Every optional feature family, from the schemas' optional operands (generated with the
/// operand visitor) and the work features.
inline const std::vector<OptionalFeature>& optionalFeatures()
{
    static const std::vector<OptionalFeature> features = [] {
        std::vector<OptionalFeature> result;
        for(const auto& operand : hipdnn_flatbuffers_sdk::data_objects::node_operands::OPERANDS)
        {
            if(!operand.optional)
            {
                continue;
            }
            const auto kind = detail::optionalKindOf(operand.kind);
            auto pattern = detail::operandPattern(operand.name, kind);
            auto found = std::find_if(result.begin(), result.end(), [&](const auto& feature) {
                return feature.pattern == pattern;
            });
            if(found == result.end())
            {
                result.push_back({std::move(pattern), kind, {}});
                found = std::prev(result.end());
            }
            found->nodeTypes.emplace_back(operand.nodeType);
        }
        for(const auto pattern : WORK_FEATURE_PATTERNS)
        {
            result.push_back({std::string(pattern), OptionalFeatureKind::WORK, {}});
        }
        return result;
    }();
    return features;
}

/// The optional feature family published @p name (with or without `$`) belongs to, or
/// nullptr when every valid graph that has the feature's node binds it.
inline const OptionalFeature* findOptionalFeature(std::string_view name)
{
    if(!name.empty() && name.front() == '$')
    {
        name.remove_prefix(1);
    }
    for(const auto& feature : optionalFeatures())
    {
        if(detail::matchesPattern(feature.pattern, name))
        {
            return &feature;
        }
    }
    return nullptr;
}

/// A reference a UHD reads unguarded although a valid graph may leave it unbound.
struct UnguardedReference
{
    std::string reference;
    const OptionalFeature* feature;
};

/// The references of @p signature that make a valid graph unscorable: optional features read
/// outside `value_or_default` / `present`, sorted. A feature's name does not say its node's
/// type, so each node may be any type having every operand the signature reads at it; an
/// operand required in all of those is not optional there. Work features are exempt in a model
/// of a metric derived from the work (@p metric, see metricIsDerivedFromWork()).
inline std::vector<UnguardedReference>
    unguardedOptionalReferences(const std::vector<nlohmann::json>& signature,
                                std::string_view metric)
{
    detail::ReferenceSet unguarded;
    detail::ReferenceSet read;
    for(const auto& entry : signature)
    {
        detail::collectUnguardedReads(entry, {}, unguarded);
        detail::collectReferences(entry, read);
    }
    std::map<std::string, std::set<std::string>> operandsByNode;
    for(const auto& reference : read)
    {
        if(auto operand = detail::nodeOperandOf(reference))
        {
            operandsByNode[operand->first].insert(std::move(operand->second));
        }
    }
    std::vector<UnguardedReference> result;
    for(const auto& reference : unguarded)
    {
        const auto* feature = findOptionalFeature(reference);
        if(feature == nullptr
           || (feature->kind == OptionalFeatureKind::WORK && metricIsDerivedFromWork(metric)))
        {
            continue;
        }
        // Work features are no operand: their absence does not depend on the node's type.
        const auto operand = feature->kind == OptionalFeatureKind::WORK
                                 ? std::nullopt
                                 : detail::nodeOperandOf(reference);
        if(operand.has_value())
        {
            const auto candidates = detail::candidateNodeTypes(operandsByNode[operand->first]);
            if(!candidates.empty()
               && std::none_of(feature->nodeTypes.begin(),
                               feature->nodeTypes.end(),
                               [&](const auto& type) { return candidates.count(type) != 0; }))
            {
                continue;
            }
        }
        result.push_back({reference, feature});
    }
    return result;
}

/// optionalFeatures() as JSON, as `hipdnn_uhd_features --optional-features` prints it.
inline nlohmann::json optionalFeaturesJson()
{
    auto features = nlohmann::json::array();
    for(const auto& feature : optionalFeatures())
    {
        features.push_back({{"pattern", feature.pattern},
                            {"kind", toString(feature.kind)},
                            {"node_types", feature.nodeTypes}});
    }
    return features;
}

} // namespace hipdnn_plugin_sdk::heuristics
