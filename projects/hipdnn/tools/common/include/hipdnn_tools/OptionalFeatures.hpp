// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_tools/node_operand_table_generated.h>

/// @file OptionalFeatures.hpp
/// @brief Which published features a valid graph may leave unbound, and which references of a
///        UHD signature read one unguarded. Tools-internal (hipdnn_tools_common): the descriptor
///        validator and hipdnn_uhd_features use it; nothing at selection time does.
namespace hipdnn_tools
{

/// Why a published feature can be absent on a valid graph.
enum class OptionalFeatureKind
{
    ATTRIBUTE, ///< an optional scalar attribute the graph leaves unset
    TENSOR, ///< every field of an optional tensor operand the graph leaves out
    TENSORS, ///< every field of an element of a vector of tensor operands
    ELEMENTS, ///< an element of a vector attribute, whose length the schema leaves open
    WORK, ///< a work feature, unbound when the graph's work is unknown (RFC 0019 section 6.8)
    CONSTRAINT, ///< a `constraint.*` feature of the engine config, bound only when it pins it
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
    case OptionalFeatureKind::CONSTRAINT:
        return "constraint";
    default:
        return "unknown";
    }
}

/// A family of published features (EngineFeatures.hpp) a valid graph may leave unbound, so a
/// UHD must read it through `value_or_default` or behind `present` (RFC 0019 section 6.7,
/// absent is not zero, and section 6.8 rule 9).
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

/// The engine-level (L1) features engineFeatures() binds only when the engine config carries
/// them: the workspace limit, and the knobs it pins (EngineFeatures.hpp).
inline constexpr std::array<std::string_view, 2> CONSTRAINT_FEATURE_PATTERNS{{
    "constraint.workspace_limit",
    "constraint.knobs.*",
}};

/// Whether ranking metric @p metric is computed from the graph's logical work, so that a model
/// of it may read the work features bare. Where the work is unknown, so is such a metric and
/// the model has nothing to rank by, whatever its role: an engine-level (L1) model reports
/// UNAVAILABLE there, and a catalog ranker (L2) leaves the catalog in its declared order.
inline bool metricIsDerivedFromWork(std::string_view metric)
{
    return metric == "tflops";
}

namespace detail
{

inline OptionalFeatureKind optionalKindOf(node_operands::OperandKind kind)
{
    using node_operands::OperandKind;
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
    using node_operands::OPERANDS;
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

/// Whether @p field of an operand is published whenever the operand is in the graph. Its
/// `dims[i]` and `strides[i]` exist only up to the operand's rank, and `numel` only while every
/// dim is non-negative, so proving the operand present proves none of them.
inline bool boundWheneverOperandIsPresent(std::string_view field)
{
    return field == "data_type" || field == "virtual" || field == "rank";
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

/// Whether @p reference is proven bound by @p scope: itself, or a field published whenever its
/// operand is, of an operand some proven field shows present.
inline bool isGuarded(const std::string& reference, const ReferenceSet& scope)
{
    if(scope.count(reference) != 0)
    {
        return true;
    }
    const auto prefix = operandPrefix(reference);
    return !prefix.empty() && scope.count(prefix) != 0
           && boundWheneverOperandIsPresent(std::string_view(reference).substr(prefix.size()));
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

/// Whether @p node is a `$` reference to a published feature.
inline bool isReferenceText(const nlohmann::json& node)
{
    return node.is_string() && node.get_ref<const std::string&>().size() > 1
           && node.get_ref<const std::string&>().front() == '$';
}

/// @p node with the `shape` and `rank` short-hands of RFC 0019 section 6.2 read as the
/// features they stand for, as ExpressionSet lowers them before it evaluates (`{"shape":
/// ["$t", k]}` is `$t.dims[k]`, `{"rank": "$t"}` is `$t.rank`): a reference to the tensor
/// alone names no published feature. Keep in step with ExpressionSet::lower
/// (heuristics/uhd/Expressions.hpp), except that a malformed short-hand, which ExpressionSet
/// refuses, is left as it is.
inline nlohmann::json lowerShorthands(const nlohmann::json& node)
{
    if(node.is_array())
    {
        auto result = nlohmann::json::array();
        for(const auto& child : node)
        {
            result.push_back(lowerShorthands(child));
        }
        return result;
    }
    if(!node.is_object())
    {
        return node;
    }
    const auto isTensor = [](const nlohmann::json& argument) {
        return isReferenceText(argument) && argument.get_ref<const std::string&>()[1] != '$';
    };
    if(node.size() == 1)
    {
        const auto& name = node.begin().key();
        const auto& body = node.begin().value();
        if(name == "shape" && body.is_array() && body.size() == 2 && isTensor(body[0])
           && body[1].is_number_integer() && body[1].get<int64_t>() >= 0
           && body[1].get<int64_t>() <= 1024)
        {
            return body[0].get<std::string>() + ".dims[" + std::to_string(body[1].get<int64_t>())
                   + "]";
        }
        const auto& tensor = body.is_array() && body.size() == 1 ? body[0] : body;
        if(name == "rank" && isTensor(tensor))
        {
            return tensor.get<std::string>() + ".rank";
        }
    }
    auto result = nlohmann::json::object();
    for(const auto& [key, value] : node.items())
    {
        result[key] = lowerShorthands(value);
    }
    return result;
}

/// Whether @p node is an operator whose result, once resolved, is a boolean.
inline bool yieldsBoolean(const nlohmann::json& node)
{
    if(!node.is_object() || node.size() != 1)
    {
        return false;
    }
    const auto& op = node.begin().key();
    return op == "present" || op == "not_present" || op == "!" || op == "!!";
}

/// The references @p condition proves bound when it is true, or, with @p negated, when it is
/// false. Only these forms prove anything, each by what the language answers:
///  - `present` of bare references (true: every one resolved); an expression among its
///    arguments says nothing of which of its references are bound;
///  - `not_present` of one bare reference (false: that one is not wholly absent); with several
///    it is false as soon as any is bound, which proves none of them;
///  - `and` (true: every argument held) and `or` (false: every argument failed);
///  - `!`, `!!`, and `==` / `!=` against a boolean literal of one of the above that yield a
///    boolean, which read the same proof either way round.
/// Anything else, a comparison for one, proves nothing.
inline ReferenceSet provenBound(const nlohmann::json& condition, bool negated)
{
    ReferenceSet bound;
    if(!condition.is_object() || condition.size() != 1)
    {
        return bound;
    }
    const auto& op = condition.begin().key();
    const auto arguments = operatorArguments(condition);
    if(op == "present" ? !negated : (op == "not_present" && negated && arguments.size() == 1))
    {
        for(const auto& argument : arguments)
        {
            if(isReferenceText(argument))
            {
                bound.insert(argument.get<std::string>());
            }
        }
    }
    else if(op == (negated ? "or" : "and"))
    {
        for(const auto& argument : arguments)
        {
            const auto inner = provenBound(argument, negated);
            bound.insert(inner.begin(), inner.end());
        }
    }
    else if((op == "!" || op == "!!") && arguments.size() == 1)
    {
        return provenBound(arguments[0], op == "!" ? !negated : negated);
    }
    else if((op == "==" || op == "!=") && arguments.size() == 2)
    {
        // Equal or unequal to a boolean literal: the condition holds exactly when the other
        // argument has one definite boolean value, and fails when it has the other.
        const size_t literal = arguments[0].is_boolean() ? 0 : 1;
        if(arguments[literal].is_boolean() && yieldsBoolean(arguments[1 - literal]))
        {
            const bool valueWhenTrue = (op == "==") == arguments[literal].get<bool>();
            return provenBound(arguments[1 - literal], negated == valueWhenTrue);
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

/// Every optional feature family: the schemas' optional operands (generated with the operand
/// visitor), the work features, and the engine config's `constraint.*` features.
///
/// A family is listed when a valid graph may leave it unbound whatever its shape. What
/// depends on the graph's shape is not listed, and is a gap training sees as a column absent in
/// some rows, for the corpus to cover: a tensor's `dims[i]` and `strides[i]` past its rank, its
/// `numel` when a dim is negative, and `graph.nodes[i]` / `graph.tensors[j]` past the graph's
/// node and tensor counts. A bare read of one passes the lint, and a graph of another rank or
/// size is one the engine's model cannot score.
inline const std::vector<OptionalFeature>& optionalFeatures()
{
    static const std::vector<OptionalFeature> s_features = [] {
        std::vector<OptionalFeature> result;
        for(const auto& operand : node_operands::OPERANDS)
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
        for(const auto pattern : CONSTRAINT_FEATURE_PATTERNS)
        {
            result.push_back({std::string(pattern), OptionalFeatureKind::CONSTRAINT, {}});
        }
        return result;
    }();
    return s_features;
}

/// The optional feature family published @p name (with or without `$`) belongs to, or
/// nullptr when it is not listed: every valid graph that has the feature's node binds it, or
/// it depends on the graph's shape (see optionalFeatures()).
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
/// outside `value_or_default` / `present`, sorted. The `shape` and `rank` short-hands are
/// read as the `dims[k]` and `rank` they stand for. A feature's name does not say its node's
/// type, so each node may be any type having every operand the signature reads bare at it
/// (a guarded read narrows nothing); an operand required in all of those is not optional
/// there. Work features are exempt in a model of a metric derived from the work (@p metric,
/// see metricIsDerivedFromWork()).
inline std::vector<UnguardedReference>
    unguardedOptionalReferences(const std::vector<nlohmann::json>& signature,
                                std::string_view metric)
{
    detail::ReferenceSet unguarded;
    for(const auto& entry : signature)
    {
        detail::collectUnguardedReads(detail::lowerShorthands(entry), {}, unguarded);
    }
    std::map<std::string, std::set<std::string>> operandsByNode;
    for(const auto& reference : unguarded)
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

} // namespace hipdnn_tools
