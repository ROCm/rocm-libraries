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
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/ingestor/DescriptorJsonRules.hpp>

/// @file GraphPattern.hpp
/// @brief Parsed form of the declarative `graph_match.nodes` pattern (RFC 0020 §4.3).
///
/// detail::parsePatternNodes() checks the grammar and well-formedness rules, which need
/// only the block itself. compileGraphPattern() in CompiledGraphPattern.hpp runs it, then
/// resolves opcodes and edge names against the op-schema registry.

namespace hipdnn_plugin_sdk::ingestor
{

/// Maximum nodes in one pattern. This and MAX_PATTERN_MATCH_STEPS keep a broken or
/// hostile UED from making compile or match unbounded.
inline constexpr size_t MAX_PATTERN_NODES = 32;
/// Maximum candidate assignments the matcher tries for one graph before it declines.
/// One step tries one pattern node on one free graph node and costs a few hundred
/// operations at most, so the budget caps a search near twenty million operations. The
/// catalog cache keeps a decline, so this cost is paid once per graph and device.
/// A search without backtracking needs at most N(N+1)/2 steps for N pattern nodes, which
/// is 528 at MAX_PATTERN_NODES. Only graphs with many interchangeable nodes reach the
/// budget. Their search grows factorially, so no practical budget would finish it.
inline constexpr size_t MAX_PATTERN_MATCH_STEPS = 65536;

namespace detail
{

/// Keys a pattern node object may carry, besides extension keys.
inline constexpr std::array<std::string_view, 5> PATTERN_NODE_KEYS{
    "kind", "id", "op", "operands", "results"};
/// Keys every pattern node object must carry.
inline constexpr std::array<std::string_view, 3> PATTERN_NODE_REQUIRED_KEYS{"kind", "id", "op"};
/// Keys an opcode-set object (`"op": {"one_of": [...]}`) may carry, besides extension keys.
inline constexpr std::array<std::string_view, 1> PATTERN_OPCODE_SET_KEYS{"one_of"};
/// Token roots the binding owns (RFC 0020 §6.1). Node ids and variables cannot use them.
inline constexpr std::array<std::string_view, 3> PATTERN_RESERVED_ROOTS{
    "graph", "kernel", "device"};

/// One edge's binding. `"$x?"` parses to `{"x", true}`.
struct PatternBinding
{
    std::string variable;
    /// The pattern also matches a graph that omits this operand. Operands only.
    bool optional = false;
};

/// One named edge of a pattern node.
struct PatternEdge
{
    /// Operand or result name, spelled as in the op-schema registry.
    std::string name;
    PatternBinding binding;
};

/// One `{"kind": "op", ...}` object.
struct PatternNode
{
    /// Unique within the pattern. Also the root this node's attributes publish under.
    std::string id;
    /// Opcodes in authored order. A bare-string `op` gives one entry.
    std::vector<std::string> opcodes;
    /// Sorted by edge name.
    std::vector<PatternEdge> operands;
    /// Sorted by edge name.
    std::vector<PatternEdge> results;
};

inline bool isReservedPatternRoot(std::string_view name)
{
    return std::find(PATTERN_RESERVED_ROOTS.begin(), PATTERN_RESERVED_ROOTS.end(), name)
           != PATTERN_RESERVED_ROOTS.end();
}

inline std::string patternNodeLocator(const std::string& id, const std::string& where)
{
    return "node '" + id + "' in " + where;
}

/// Parses `"$name"`, or `"$name?"` when @p allowOptional is true.
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
    if(body.empty() || body.front() != '$' || !isIdentifier(body.substr(1)))
    {
        fail(edgeLocator + " has malformed binding '" + text + "'; expected '$<identifier>'"
             + (allowOptional ? " or '$<identifier>?'" : ""));
    }
    binding.variable = std::string(body.substr(1));
    return binding;
}

/// Parses an `operands` or `results` object. nlohmann::json stores object members in a
/// std::map, so the edges come out sorted by name.
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
    std::set<std::string_view> seen;
    for(const auto& member : *members)
    {
        if(!member.is_string() || member.get_ref<const std::string&>().empty())
        {
            fail("key 'one_of' in " + setLocator + " must list non-empty opcode strings");
        }
        const auto& opcode = member.get_ref<const std::string&>();
        if(!seen.insert(opcode).second)
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
        static_cast<void>(requireKey(object, key, position));
    }

    const auto& id = object.at("id");
    if(!id.is_string() || !isIdentifier(id.get_ref<const std::string&>()))
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
    return node;
}

/// Checks the cross-node rules: unique node ids, legal variable names, and each variable
/// bound once.
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

    // The edge that binds a variable. This is its producing result, or for a graph input,
    // the single operand that reads it.
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

/// Fails when nodes depend on each other in a cycle: each reads a result of the one before
/// it, so none can come first. hipDNN graphs are acyclic, so such a pattern matches no
/// graph. A node that reads its own result is a cycle of one. Run after
/// checkPatternVariables(), which guarantees each variable has at most one producer.
inline void checkPatternAcyclic(const std::vector<PatternNode>& nodes, const std::string& where)
{
    const size_t count = nodes.size();
    std::map<std::string_view, size_t> producerOf;
    for(size_t node = 0; node < count; ++node)
    {
        for(const auto& edge : nodes[node].results)
        {
            producerOf.emplace(edge.binding.variable, node);
        }
    }
    std::vector<std::vector<size_t>> producers(count);
    for(size_t node = 0; node < count; ++node)
    {
        for(const auto& edge : nodes[node].operands)
        {
            const auto producer = producerOf.find(edge.binding.variable);
            if(producer != producerOf.end())
            {
                producers[node].push_back(producer->second);
            }
        }
    }

    // Place every node whose producers are all placed. What remains is the cycles and the
    // nodes downstream of them.
    std::vector<bool> placed(count, false);
    size_t placedCount = 0;
    for(bool progress = true; progress && placedCount < count;)
    {
        progress = false;
        for(size_t node = 0; node < count; ++node)
        {
            if(!placed[node]
               && std::all_of(producers[node].begin(),
                              producers[node].end(),
                              [&placed](size_t producer) { return placed[producer]; }))
            {
                placed[node] = true;
                ++placedCount;
                progress = true;
            }
        }
    }
    if(placedCount == count)
    {
        return;
    }

    // Every unplaced node has an unplaced producer, so following producers from any one of
    // them must revisit a node. The walk from that node on is a cycle.
    std::vector<size_t> walk;
    std::vector<bool> visited(count, false);
    auto current
        = static_cast<size_t>(std::find(placed.begin(), placed.end(), false) - placed.begin());
    while(!visited[current])
    {
        visited[current] = true;
        walk.push_back(current);
        current = *std::find_if(producers[current].begin(),
                                producers[current].end(),
                                [&placed](size_t producer) { return !placed[producer]; });
    }

    // The walk runs from reader to producer. Reverse it so each node reads a result of the
    // one before it, and close the loop on the first node.
    std::string cycle;
    const auto first = std::find(walk.begin(), walk.end(), current);
    for(auto it = walk.end(); it != first;)
    {
        --it;
        cycle += "'" + nodes[*it].id + "' -> ";
    }
    cycle += "'" + nodes[walk.back()].id + "'";
    fail("pattern nodes " + cycle + " in " + where
         + " depend on each other in a cycle; no hipDNN graph is cyclic, so the pattern could"
           " never match");
}

/// Parses @p nodes, the value of `graph_match.nodes`, and checks its grammar and
/// well-formedness. @p where names the UED in every error.
inline std::vector<PatternNode> parsePatternNodes(const nlohmann::json& nodes,
                                                  const std::string& where)
{
    const std::string block = "graph_match.nodes in " + where;
    if(!nodes.is_array())
    {
        fail(block + " must be an array of node objects");
    }
    if(nodes.empty())
    {
        fail(block + " must not be empty");
    }
    if(nodes.size() > MAX_PATTERN_NODES)
    {
        fail(block + " has " + std::to_string(nodes.size()) + " nodes; the limit is "
             + std::to_string(MAX_PATTERN_NODES));
    }
    std::vector<PatternNode> parsed;
    parsed.reserve(nodes.size());
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        parsed.push_back(parsePatternNode(nodes[i], i, where));
    }
    checkPatternVariables(parsed, where);
    checkPatternAcyclic(parsed, where);
    return parsed;
}

} // namespace detail

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
