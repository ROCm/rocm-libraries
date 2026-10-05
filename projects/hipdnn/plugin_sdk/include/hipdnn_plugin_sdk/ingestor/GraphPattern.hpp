// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/ingestor/DescriptorJsonRules.hpp>

/// @file GraphPattern.hpp
/// @brief The declarative `graph_match.nodes` pattern of RFC 0020 §4.3, as parsed data.
///
/// parseGraphPattern() checks the §4.3.1 grammar and the §4.3.2 well-formedness rules,
/// which need nothing but the block itself. Resolving opcodes and edge names against the
/// op-schema registry (§4.3.3) is compileGraphPattern()'s job (CompiledGraphPattern.hpp).

namespace hipdnn_plugin_sdk::ingestor
{

/// Bounds a pattern so a hostile or broken UED cannot make compile or match unbounded.
inline constexpr size_t MAX_PATTERN_NODES = 32;
/// Operand plus result bindings on one node.
inline constexpr size_t MAX_PATTERN_EDGES_PER_NODE = 64;
/// Members of one `one_of` opcode set.
inline constexpr size_t MAX_PATTERN_OPCODE_SET = 32;
/// Candidate assignments the matcher tries for one graph before it declines (RFC 0020
/// §4.3). A step is one pattern node tried on one free graph node: at most
/// MAX_PATTERN_OPCODE_SET opcode compares, then one field read and one hash lookup per
/// edge the op declares, a few hundred operations at worst. The budget therefore caps one
/// search near twenty million operations, paid once per graph and device because the
/// catalog cache keeps a decline like any other result. A search that never backtracks
/// needs at most N(N+1)/2 steps for N pattern nodes, 528 at MAX_PATTERN_NODES, so only a
/// graph forcing a hundredfold backtracking reaches the budget: one with many
/// interchangeable nodes, whose search grows factorially with their number and which no
/// feasible budget would finish.
inline constexpr size_t MAX_PATTERN_MATCH_STEPS = 65536;

/// Keys a pattern node object may carry, besides extension keys.
inline constexpr std::array<std::string_view, 5> PATTERN_NODE_KEYS{
    "kind", "id", "op", "operands", "results"};
/// Keys every pattern node object must carry.
inline constexpr std::array<std::string_view, 3> PATTERN_NODE_REQUIRED_KEYS{"kind", "id", "op"};
/// Keys an opcode-set object (`"op": {"one_of": [...]}`) may carry, besides extension keys.
inline constexpr std::array<std::string_view, 1> PATTERN_OPCODE_SET_KEYS{"one_of"};
/// Token roots the binding owns (RFC 0020 §6.1); neither a node id nor a variable may use one.
inline constexpr std::array<std::string_view, 3> PATTERN_RESERVED_ROOTS{
    "graph", "kernel", "device"};

/// One edge's binding: `"$x?"` is `{"x", true}`.
struct PatternBinding
{
    std::string variable;
    /// The pattern still matches a graph omitting this operand. Operands only.
    bool optional = false;
};

/// One named edge of a pattern node.
struct PatternEdge
{
    /// The operand or result name, as the op-schema registry spells it.
    std::string name;
    PatternBinding binding;
};

/// One `{"kind": "op", ...}` object.
struct PatternNode
{
    /// Unique within the pattern; the root this node's attributes publish under.
    std::string id;
    /// Authored order; one entry for a bare-string `op`.
    std::vector<std::string> opcodes;
    /// Authored as `{"one_of": [...]}`.
    bool opcodeSet = false;
    /// Sorted by edge name.
    std::vector<PatternEdge> operands;
    /// Sorted by edge name.
    std::vector<PatternEdge> results;
};

class GraphPattern;

/// Parses @p nodes, the value of `graph_match.nodes`, and checks the RFC 0020 §4.3.1
/// grammar and §4.3.2 well-formedness. @p where names the UED in every error.
/// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) naming the UED, the
///         node, and the offending name or value.
inline GraphPattern parseGraphPattern(const nlohmann::json& nodes, const std::string& where);

/// A well-formed pattern, immutable once parsed.
class GraphPattern
{
public:
    const std::vector<PatternNode>& nodes() const noexcept
    {
        return _nodes;
    }

private:
    explicit GraphPattern(std::vector<PatternNode> nodes)
        : _nodes(std::move(nodes))
    {
    }

    friend GraphPattern parseGraphPattern(const nlohmann::json& nodes, const std::string& where);

    std::vector<PatternNode> _nodes;
};

namespace detail
{

inline bool isPatternIdentifier(std::string_view text)
{
    const auto isLeading
        = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
    if(text.empty() || !isLeading(text.front()))
    {
        return false;
    }
    return std::all_of(text.begin() + 1, text.end(), [&isLeading](char c) {
        return isLeading(c) || (c >= '0' && c <= '9');
    });
}

inline bool isReservedPatternRoot(std::string_view name)
{
    return std::find(PATTERN_RESERVED_ROOTS.begin(), PATTERN_RESERVED_ROOTS.end(), name)
           != PATTERN_RESERVED_ROOTS.end();
}

inline std::string patternNodeLocator(const std::string& id, const std::string& where)
{
    return "node '" + id + "' in " + where;
}

/// `"$name"` or, when @p allowOptional, `"$name?"`.
inline PatternBinding parsePatternBinding(const nlohmann::json& value,
                                          bool allowOptional,
                                          const std::string& edgeLocator)
{
    if(!value.is_string())
    {
        fail(edgeLocator + " must be a string binding '$<identifier>'");
    }
    const auto& text = value.get_ref<const std::string&>();
    std::string_view body{text};
    PatternBinding binding;
    if(!body.empty() && body.back() == '?')
    {
        if(!allowOptional)
        {
            fail(edgeLocator + " has binding '" + text
                 + "'; '?' is only allowed on an operand binding");
        }
        binding.optional = true;
        body.remove_suffix(1);
    }
    if(body.empty() || body.front() != '$' || !isPatternIdentifier(body.substr(1)))
    {
        fail(edgeLocator + " has malformed binding '" + text + "'; expected '$<identifier>'"
             + (allowOptional ? " or '$<identifier>?'" : ""));
    }
    binding.variable = std::string(body.substr(1));
    return binding;
}

/// An `operands` or `results` object. nlohmann::json keeps object members in a std::map,
/// so the edges come out sorted by name.
inline std::vector<PatternEdge> parsePatternEdges(const nlohmann::json& node,
                                                  std::string_view key,
                                                  bool operands,
                                                  const std::string& locator)
{
    std::vector<PatternEdge> edges;
    const auto it = node.find(std::string(key));
    if(it == node.end())
    {
        return edges;
    }
    if(!it->is_object())
    {
        fail("key '" + std::string(key) + "' of " + locator
             + " must be an object mapping edge names to bindings");
    }
    for(const auto& item : it->items())
    {
        const std::string edgeLocator
            = std::string(operands ? "operand '" : "result '") + item.key() + "' of " + locator;
        edges.push_back(
            PatternEdge{item.key(), parsePatternBinding(item.value(), operands, edgeLocator)});
    }
    return edges;
}

inline void
    parsePatternOpcodes(const nlohmann::json& op, PatternNode& node, const std::string& locator)
{
    if(op.is_string())
    {
        const auto& opcode = op.get_ref<const std::string&>();
        if(opcode.empty())
        {
            fail("key 'op' of " + locator + " must not be empty");
        }
        node.opcodes.push_back(opcode);
        return;
    }
    if(!op.is_object())
    {
        fail("key 'op' of " + locator
             + " must be an opcode string or an object {\"one_of\": [opcodes]}");
    }
    const std::string setLocator = "key 'op' of " + locator;
    requireKnownKeys(op, PATTERN_OPCODE_SET_KEYS, setLocator);
    const auto members = op.find("one_of");
    if(members == op.end())
    {
        fail("missing required key 'one_of' in " + setLocator);
    }
    if(!members->is_array())
    {
        fail("key 'one_of' in " + setLocator + " must be an array of opcode strings");
    }
    if(members->size() < 2)
    {
        fail("key 'one_of' in " + setLocator + " lists " + std::to_string(members->size())
             + " opcode(s); an opcode set needs at least 2");
    }
    if(members->size() > MAX_PATTERN_OPCODE_SET)
    {
        fail("key 'one_of' in " + setLocator + " lists " + std::to_string(members->size())
             + " opcodes; the limit is " + std::to_string(MAX_PATTERN_OPCODE_SET));
    }
    node.opcodeSet = true;
    for(const auto& member : *members)
    {
        if(!member.is_string() || member.get_ref<const std::string&>().empty())
        {
            fail("key 'one_of' in " + setLocator + " must list non-empty opcode strings");
        }
        const auto& opcode = member.get_ref<const std::string&>();
        if(std::find(node.opcodes.begin(), node.opcodes.end(), opcode) != node.opcodes.end())
        {
            std::string message = "opcode '";
            message += opcode;
            message += "' is listed twice in key 'one_of' in ";
            message += setLocator;
            fail(message);
        }
        node.opcodes.push_back(opcode);
    }
}

inline PatternNode
    parsePatternNode(const nlohmann::json& object, size_t index, const std::string& where)
{
    const std::string position = "graph_match.nodes[" + std::to_string(index) + "] in " + where;
    if(!object.is_object())
    {
        fail(position + " must be a JSON object");
    }
    requireKnownKeys(object, PATTERN_NODE_KEYS, position);
    for(const auto key : PATTERN_NODE_REQUIRED_KEYS)
    {
        if(!object.contains(std::string(key)))
        {
            fail("missing required key '" + std::string(key) + "' in " + position);
        }
    }

    const auto& id = object.at("id");
    if(!id.is_string() || !isPatternIdentifier(id.get_ref<const std::string&>()))
    {
        fail("key 'id' in " + position + " must be an identifier ([A-Za-z_][A-Za-z0-9_]*), got "
             + id.dump());
    }
    PatternNode node;
    node.id = id.get<std::string>();
    const auto locator = patternNodeLocator(node.id, where);
    if(isReservedPatternRoot(node.id))
    {
        fail(locator + ": node id '" + node.id + "' is a reserved root");
    }

    const auto& kind = object.at("kind");
    if(!kind.is_string() || kind.get_ref<const std::string&>() != "op")
    {
        fail("key 'kind' of " + locator + " must be \"op\", got " + kind.dump());
    }

    parsePatternOpcodes(object.at("op"), node, locator);
    node.operands = parsePatternEdges(object, "operands", true, locator);
    node.results = parsePatternEdges(object, "results", false, locator);
    const auto edgeCount = node.operands.size() + node.results.size();
    if(edgeCount > MAX_PATTERN_EDGES_PER_NODE)
    {
        fail(locator + " binds " + std::to_string(edgeCount) + " edges; the limit is "
             + std::to_string(MAX_PATTERN_EDGES_PER_NODE));
    }
    return node;
}

/// The cross-node §4.3.2 rules: unique ids, variable names, and bound-once.
inline void checkPatternVariables(const std::vector<PatternNode>& nodes, const std::string& where)
{
    std::set<std::string_view> ids;
    for(const auto& node : nodes)
    {
        if(!ids.insert(node.id).second)
        {
            fail("node id '" + node.id + "' is used by two nodes in " + where);
        }
    }

    // The single edge each variable is bound by: its producing result, or else the one
    // operand reading it.
    struct EdgeSite
    {
        const PatternNode* node;
        const PatternEdge* edge;
    };
    const auto describe = [](const EdgeSite& site, std::string_view direction) {
        return "node '" + site.node->id + "' " + std::string(direction) + " '" + site.edge->name
               + "'";
    };
    const auto checkName = [&](const PatternNode& node, const PatternEdge& edge) {
        const auto& variable = edge.binding.variable;
        if(isReservedPatternRoot(variable))
        {
            fail(patternNodeLocator(node.id, where) + ": variable '$" + variable
                 + "' is a reserved root");
        }
        if(ids.count(variable) != 0)
        {
            fail(patternNodeLocator(node.id, where) + ": variable '$" + variable
                 + "' collides with node id '" + variable + "'");
        }
    };

    std::map<std::string_view, EdgeSite> producers;
    for(const auto& node : nodes)
    {
        for(const auto& edge : node.results)
        {
            checkName(node, edge);
            const auto [it, inserted]
                = producers.emplace(edge.binding.variable, EdgeSite{&node, &edge});
            if(!inserted)
            {
                fail("variable '$" + edge.binding.variable + "' is bound by two results in " + where
                     + ": " + describe(it->second, "result") + " and "
                     + describe(EdgeSite{&node, &edge}, "result"));
            }
        }
    }

    std::map<std::string_view, EdgeSite> graphInputs;
    for(const auto& node : nodes)
    {
        for(const auto& edge : node.operands)
        {
            checkName(node, edge);
            const auto& variable = edge.binding.variable;
            const auto producer = producers.find(variable);
            if(producer != producers.end())
            {
                if(edge.binding.optional)
                {
                    fail(patternNodeLocator(node.id, where) + ": optional binding '$" + variable
                         + "?' on operand '" + edge.name + "' names a variable "
                         + describe(producer->second, "result")
                         + " produces; only a graph input may be optional");
                }
                continue;
            }
            const auto [it, inserted] = graphInputs.emplace(variable, EdgeSite{&node, &edge});
            if(!inserted)
            {
                std::string message = "graph input variable '$";
                message += variable;
                message += "' is bound by two operands in ";
                message += where;
                message += ": ";
                message += describe(it->second, "operand");
                message += " and ";
                message += describe(EdgeSite{&node, &edge}, "operand");
                message += "; a variable no node produces is bound by exactly one operand";
                fail(message);
            }
        }
    }
}

} // namespace detail

inline GraphPattern parseGraphPattern(const nlohmann::json& nodes, const std::string& where)
{
    const std::string block = "graph_match.nodes in " + where;
    if(!nodes.is_array())
    {
        detail::fail(block + " must be an array of node objects");
    }
    if(nodes.empty())
    {
        detail::fail(block + " must not be empty");
    }
    if(nodes.size() > MAX_PATTERN_NODES)
    {
        detail::fail(block + " has " + std::to_string(nodes.size()) + " nodes; the limit is "
                     + std::to_string(MAX_PATTERN_NODES));
    }
    std::vector<PatternNode> parsed;
    parsed.reserve(nodes.size());
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        parsed.push_back(detail::parsePatternNode(nodes[i], i, where));
    }
    detail::checkPatternVariables(parsed, where);
    return GraphPattern(std::move(parsed));
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
