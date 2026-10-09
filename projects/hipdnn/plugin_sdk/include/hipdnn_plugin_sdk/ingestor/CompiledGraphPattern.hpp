// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/op_schema_registry_generated.h>
#include <hipdnn_plugin_sdk/ingestor/DescriptorJsonRules.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPattern.hpp>

/// @file CompiledGraphPattern.hpp
/// @brief A `graph_match.nodes` pattern resolved against the op-schema registry (RFC 0020
///        §4.3.3), plus the symbol table it publishes.
///
/// Patterns compile once at load. Each pattern edge then points at the registry accessor
/// that reads its tensor uid, so matching a graph needs no string lookups.

namespace hipdnn_plugin_sdk::ingestor
{

/// Kind of value a published symbol carries. A TENSOR symbol is a tensor root, with the
/// fields publishTensor() in BindingPublication.hpp publishes.
enum class SymbolKind
{
    TENSOR,
    INT,
    FLOAT,
    BOOL,
    ENUM_NAME
};

/// A root the pattern publishes. It is a tensor variable (`x`), a graph field
/// (`graph.node_count`), or a node attribute (`<node id>.<attribute>`).
struct PublishedSymbol
{
    std::string name;
    SymbolKind kind;
    /// A match may leave this symbol unbound.
    bool optional;
};

enum class EdgeDirection
{
    OPERAND,
    RESULT
};

/// One named edge of one pattern node.
struct PatternEdgeRef
{
    std::string nodeId;
    std::string edge;
    EdgeDirection direction;
};

/// A pattern variable, with the edge that binds it and the edges that only read it.
struct PatternVariable
{
    std::string name;
    /// A node's result, or the single operand reading a graph input.
    PatternEdgeRef binding;
    /// Bound with `?`. If the graph omits it, the match succeeds and publishes nothing for it.
    bool optional;
    /// Operand edges reading a variable a result binds, in authored order.
    std::vector<PatternEdgeRef> readers;
};

/// One opcode a pattern node may match, with its edges resolved.
struct CompiledPatternCandidate
{
    const hipdnn_flatbuffers_sdk::data_objects::op_schema::OpSchema* schema;
    /// Index-aligned with CompiledPatternNode::operandVariables.
    std::vector<const hipdnn_flatbuffers_sdk::data_objects::op_schema::EdgeSchema*> operands;
    /// Index-aligned with CompiledPatternNode::resultVariables.
    std::vector<const hipdnn_flatbuffers_sdk::data_objects::op_schema::EdgeSchema*> results;
    /// Optional edges the pattern does not name. A graph node that supplies one does not
    /// match.
    std::vector<const hipdnn_flatbuffers_sdk::data_objects::op_schema::EdgeSchema*>
        unnamedOptionalEdges;
    /// Index-aligned with CompiledPatternNode::attributes.
    std::vector<const hipdnn_flatbuffers_sdk::data_objects::op_schema::AttributeSchema*> attributes;
};

/// An attribute a pattern node publishes.
struct CompiledPatternAttribute
{
    /// `<node id>.<attribute>`.
    std::string token;
    SymbolKind kind;
    bool optional;
};

struct CompiledPatternNode
{
    std::string id;
    /// Indices into CompiledGraphPattern::variables(), one per named operand.
    std::vector<size_t> operandVariables;
    /// Indices into CompiledGraphPattern::variables(), one per named result.
    std::vector<size_t> resultVariables;
    /// Attributes every candidate declares with the same kind.
    std::vector<CompiledPatternAttribute> attributes;
    /// In authored opcode order.
    std::vector<CompiledPatternCandidate> candidates;
};

class CompiledGraphPattern;

/// Resolves @p pattern against the op-schema registry. @p where names the UED in every
/// error.
/// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) naming the UED, the
///         node, and the unresolved name.
inline std::shared_ptr<const CompiledGraphPattern> compileGraphPattern(const GraphPattern& pattern,
                                                                       const std::string& where);

/// A resolved pattern. It is immutable, so threads can share one instance.
class CompiledGraphPattern
{
    struct ConstructionKey
    {
        explicit ConstructionKey() = default;
    };

public:
    CompiledGraphPattern(ConstructionKey /*key*/,
                         std::string where,
                         std::vector<CompiledPatternNode> nodes,
                         std::vector<size_t> matchOrder,
                         std::vector<PatternVariable> variables,
                         std::vector<std::string> rootOpcodes,
                         std::vector<PublishedSymbol> publishedSymbols)
        : _where(std::move(where))
        , _nodes(std::move(nodes))
        , _matchOrder(std::move(matchOrder))
        , _variables(std::move(variables))
        , _rootOpcodes(std::move(rootOpcodes))
        , _publishedSymbols(std::move(publishedSymbols))
    {
    }

    /// The UED this pattern came from, for diagnostics.
    const std::string& where() const noexcept
    {
        return _where;
    }

    size_t nodeCount() const noexcept
    {
        return _nodes.size();
    }

    /// Nodes in authored order.
    const std::vector<CompiledPatternNode>& nodes() const noexcept
    {
        return _nodes;
    }

    /// Indices into nodes(), with each producer before the nodes that read its results.
    /// Ties keep authored order. Nodes in a dependency cycle come last, in authored order.
    const std::vector<size_t>& matchOrder() const noexcept
    {
        return _matchOrder;
    }

    /// Sorted by name.
    const std::vector<PatternVariable>& variables() const noexcept
    {
        return _variables;
    }

    const PatternVariable* findVariable(std::string_view name) const noexcept
    {
        const auto it = std::lower_bound(_variables.begin(),
                                         _variables.end(),
                                         name,
                                         [](const PatternVariable& variable, std::string_view key) {
                                             return variable.name < key;
                                         });
        return it != _variables.end() && it->name == name ? &*it : nullptr;
    }

    /// Opcodes a graph's root node may carry. This is the union of the opcode sets of the
    /// nodes that read no variable another node produces. Sorted and distinct. The matcher
    /// declines a graph with a root outside this set before it searches.
    const std::vector<std::string>& rootOpcodes() const noexcept
    {
        return _rootOpcodes;
    }

    /// Sorted by name.
    const std::vector<PublishedSymbol>& publishedSymbols() const noexcept
    {
        return _publishedSymbols;
    }

private:
    friend std::shared_ptr<const CompiledGraphPattern>
        compileGraphPattern(const GraphPattern& pattern, const std::string& where);

    std::string _where;
    std::vector<CompiledPatternNode> _nodes;
    std::vector<size_t> _matchOrder;
    std::vector<PatternVariable> _variables;
    std::vector<std::string> _rootOpcodes;
    std::vector<PublishedSymbol> _publishedSymbols;
};

namespace detail
{

inline SymbolKind symbolKindOf(hipdnn_flatbuffers_sdk::data_objects::op_schema::ValueKind kind)
{
    using hipdnn_flatbuffers_sdk::data_objects::op_schema::ValueKind;
    switch(kind)
    {
    case ValueKind::INT:
        return SymbolKind::INT;
    case ValueKind::FLOAT:
        return SymbolKind::FLOAT;
    case ValueKind::BOOL:
        return SymbolKind::BOOL;
    case ValueKind::ENUM_NAME:
        return SymbolKind::ENUM_NAME;
    default:
        fail("op-schema registry attribute has an unknown value kind");
    }
}

inline const hipdnn_flatbuffers_sdk::data_objects::op_schema::EdgeSchema*
    findEdgeSchema(hipdnn_flatbuffers_sdk::data_objects::op_schema::SchemaSpan<
                       hipdnn_flatbuffers_sdk::data_objects::op_schema::EdgeSchema> edges,
                   std::string_view name)
{
    const auto it = std::find_if(
        edges.begin(), edges.end(), [name](const auto& edge) { return edge.name == name; });
    return it == edges.end() ? nullptr : it;
}

/// Resolves one named edge on every candidate. Checks the edge direction, the `?` marker,
/// and that all `one_of` members agree.
inline void resolvePatternEdge(const PatternNode& node,
                               const PatternEdge& edge,
                               EdgeDirection direction,
                               std::vector<CompiledPatternCandidate>& candidates,
                               const std::string& where)
{
    const bool operand = direction == EdgeDirection::OPERAND;
    const std::string_view kind = operand ? "operand" : "result";
    const std::string_view otherKind = operand ? "a result" : "an operand";
    const std::string_view otherSection = operand ? "results" : "operands";
    const std::string locator = patternNodeLocator(node.id, where);
    const hipdnn_flatbuffers_sdk::data_objects::op_schema::EdgeSchema* first = nullptr;
    for(auto& candidate : candidates)
    {
        const auto& schema = *candidate.schema;
        const auto* resolved
            = findEdgeSchema(operand ? schema.operands : schema.results, edge.name);
        if(resolved == nullptr)
        {
            std::string message = locator;
            message += ": opcode '";
            message += schema.opcode;
            message += "' declares no ";
            message += kind;
            message += " '";
            message += edge.name;
            message += "'";
            if(findEdgeSchema(operand ? schema.results : schema.operands, edge.name) != nullptr)
            {
                message += "; it is ";
                message += otherKind;
                message += ", so it belongs under '";
                message += otherSection;
                message += "'";
            }
            fail(message);
        }
        if(edge.binding.optional && !resolved->optional)
        {
            fail(locator + ": operand '" + edge.name + "' of opcode '" + std::string(schema.opcode)
                 + "' is required, so binding '$" + edge.binding.variable
                 + "?' cannot be optional");
        }
        if(first == nullptr)
        {
            first = resolved;
        }
        else if(first->optional != resolved->optional)
        {
            std::string message = locator;
            message += ": one_of members disagree on ";
            message += kind;
            message += " '";
            message += edge.name;
            message += "': opcode '";
            message += candidates.front().schema->opcode;
            message += "' declares it ";
            message += first->optional ? "optional" : "required";
            message += ", opcode '";
            message += schema.opcode;
            message += "' declares it ";
            message += resolved->optional ? "optional" : "required";
            fail(message);
        }
        (operand ? candidate.operands : candidate.results).push_back(resolved);
    }
}

/// Collects the attributes every candidate declares with the same kind. An attribute is
/// optional if any candidate declares it optional. Order follows the first candidate.
inline void resolvePatternAttributes(const PatternNode& node, CompiledPatternNode& compiled)
{
    const auto& firstSchema = *compiled.candidates.front().schema;
    for(const auto& attribute : firstSchema.attributes)
    {
        bool optional = attribute.optional;
        std::vector<const hipdnn_flatbuffers_sdk::data_objects::op_schema::AttributeSchema*>
            perCandidate;
        for(const auto& candidate : compiled.candidates)
        {
            const auto& attributes = candidate.schema->attributes;
            const auto it = std::find_if(
                attributes.begin(), attributes.end(), [&attribute](const auto& other) {
                    return other.name == attribute.name && other.kind == attribute.kind;
                });
            if(it == attributes.end())
            {
                break;
            }
            optional = optional || it->optional;
            perCandidate.push_back(it);
        }
        if(perCandidate.size() != compiled.candidates.size())
        {
            continue;
        }
        compiled.attributes.push_back(CompiledPatternAttribute{
            node.id + "." + std::string(attribute.name), symbolKindOf(attribute.kind), optional});
        for(size_t i = 0; i < perCandidate.size(); ++i)
        {
            compiled.candidates[i].attributes.push_back(perCandidate[i]);
        }
    }
}

inline void collectUnnamedOptionalEdges(const PatternNode& node,
                                        CompiledPatternCandidate& candidate)
{
    const auto named = [](const std::vector<PatternEdge>& edges, std::string_view name) {
        return std::any_of(edges.begin(), edges.end(), [name](const PatternEdge& edge) {
            return edge.name == name;
        });
    };
    for(const auto& edge : candidate.schema->operands)
    {
        if(edge.optional && !named(node.operands, edge.name))
        {
            candidate.unnamedOptionalEdges.push_back(&edge);
        }
    }
    for(const auto& edge : candidate.schema->results)
    {
        if(edge.optional && !named(node.results, edge.name))
        {
            candidate.unnamedOptionalEdges.push_back(&edge);
        }
    }
}

inline std::vector<PatternVariable> collectPatternVariables(const GraphPattern& pattern)
{
    std::map<std::string, PatternVariable> variables;
    for(const auto& node : pattern.nodes())
    {
        for(const auto& edge : node.results)
        {
            variables.emplace(
                edge.binding.variable,
                PatternVariable{edge.binding.variable,
                                PatternEdgeRef{node.id, edge.name, EdgeDirection::RESULT},
                                false,
                                {}});
        }
    }
    for(const auto& node : pattern.nodes())
    {
        for(const auto& edge : node.operands)
        {
            PatternEdgeRef reference{node.id, edge.name, EdgeDirection::OPERAND};
            const auto it = variables.find(edge.binding.variable);
            if(it == variables.end())
            {
                // parseGraphPattern() guarantees a graph input has exactly one operand.
                variables.emplace(
                    edge.binding.variable,
                    PatternVariable{
                        edge.binding.variable, std::move(reference), edge.binding.optional, {}});
            }
            else
            {
                it->second.readers.push_back(std::move(reference));
            }
        }
    }
    std::vector<PatternVariable> sorted;
    sorted.reserve(variables.size());
    for(auto& entry : variables)
    {
        sorted.push_back(std::move(entry.second));
    }
    return sorted;
}

/// Returns, for each node, the other nodes that produce a variable it reads.
inline std::vector<std::vector<size_t>>
    patternProducers(const std::vector<CompiledPatternNode>& nodes,
                     const std::vector<PatternVariable>& variables)
{
    std::vector<size_t> producerOf(variables.size(), nodes.size());
    for(size_t node = 0; node < nodes.size(); ++node)
    {
        for(const auto variable : nodes[node].resultVariables)
        {
            producerOf[variable] = node;
        }
    }
    std::vector<std::vector<size_t>> producers(nodes.size());
    for(size_t node = 0; node < nodes.size(); ++node)
    {
        for(const auto variable : nodes[node].operandVariables)
        {
            const auto producer = producerOf[variable];
            if(producer != nodes.size() && producer != node)
            {
                producers[node].push_back(producer);
            }
        }
    }
    return producers;
}

inline std::vector<size_t> patternMatchOrder(const std::vector<std::vector<size_t>>& producers)
{
    const size_t count = producers.size();
    std::vector<bool> placed(count, false);
    std::vector<size_t> order;
    order.reserve(count);
    while(order.size() < count)
    {
        size_t next = count;
        for(size_t node = 0; node < count && next == count; ++node)
        {
            if(!placed[node]
               && std::all_of(producers[node].begin(),
                              producers[node].end(),
                              [&placed](size_t producer) { return placed[producer]; }))
            {
                next = node;
            }
        }
        if(next == count)
        {
            // Dependency cycle. No node is ready, so take the first unplaced one.
            next = static_cast<size_t>(std::find(placed.begin(), placed.end(), false)
                                       - placed.begin());
        }
        placed[next] = true;
        order.push_back(next);
    }
    return order;
}

} // namespace detail

inline std::shared_ptr<const CompiledGraphPattern> compileGraphPattern(const GraphPattern& pattern,
                                                                       const std::string& where)
{
    namespace op_schema = hipdnn_flatbuffers_sdk::data_objects::op_schema;

    auto variables = detail::collectPatternVariables(pattern);
    const auto variableIndex = [&variables](const std::string& name) {
        return static_cast<size_t>(
            std::lower_bound(variables.begin(),
                             variables.end(),
                             name,
                             [](const PatternVariable& variable, const std::string& key) {
                                 return variable.name < key;
                             })
            - variables.begin());
    };

    std::vector<CompiledPatternNode> nodes;
    nodes.reserve(pattern.nodes().size());
    for(const auto& node : pattern.nodes())
    {
        CompiledPatternNode compiled;
        compiled.id = node.id;
        for(const auto& opcode : node.opcodes)
        {
            const auto* schema = op_schema::findOpSchema(opcode);
            if(schema == nullptr)
            {
                detail::fail(detail::patternNodeLocator(node.id, where) + ": opcode '" + opcode
                             + "' is not in the op-schema registry");
            }
            compiled.candidates.push_back(CompiledPatternCandidate{schema, {}, {}, {}, {}});
        }
        for(const auto& edge : node.operands)
        {
            detail::resolvePatternEdge(
                node, edge, EdgeDirection::OPERAND, compiled.candidates, where);
            compiled.operandVariables.push_back(variableIndex(edge.binding.variable));
        }
        for(const auto& edge : node.results)
        {
            detail::resolvePatternEdge(
                node, edge, EdgeDirection::RESULT, compiled.candidates, where);
            compiled.resultVariables.push_back(variableIndex(edge.binding.variable));
        }
        for(auto& candidate : compiled.candidates)
        {
            detail::collectUnnamedOptionalEdges(node, candidate);
        }
        detail::resolvePatternAttributes(node, compiled);
        nodes.push_back(std::move(compiled));
    }

    const auto producers = detail::patternProducers(nodes, variables);
    auto matchOrder = detail::patternMatchOrder(producers);

    std::vector<std::string> rootOpcodes;
    for(size_t node = 0; node < nodes.size(); ++node)
    {
        if(producers[node].empty())
        {
            const auto& opcodes = pattern.nodes()[node].opcodes;
            rootOpcodes.insert(rootOpcodes.end(), opcodes.begin(), opcodes.end());
        }
    }
    std::sort(rootOpcodes.begin(), rootOpcodes.end());
    rootOpcodes.erase(std::unique(rootOpcodes.begin(), rootOpcodes.end()), rootOpcodes.end());

    std::vector<PublishedSymbol> symbols;
    symbols.push_back(PublishedSymbol{"graph.node_count", SymbolKind::INT, false});
    symbols.push_back(PublishedSymbol{"graph.is_override_shape_enabled", SymbolKind::BOOL, false});
    for(const auto& variable : variables)
    {
        symbols.push_back(PublishedSymbol{variable.name, SymbolKind::TENSOR, variable.optional});
    }
    for(const auto& node : nodes)
    {
        for(const auto& attribute : node.attributes)
        {
            symbols.push_back(PublishedSymbol{attribute.token, attribute.kind, attribute.optional});
        }
    }
    std::sort(symbols.begin(), symbols.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });

    return std::make_shared<CompiledGraphPattern>(CompiledGraphPattern::ConstructionKey{},
                                                  where,
                                                  std::move(nodes),
                                                  std::move(matchOrder),
                                                  std::move(variables),
                                                  std::move(rootOpcodes),
                                                  std::move(symbols));
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
