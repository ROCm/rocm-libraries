// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/op_schema_registry_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/// @file GraphPatternMatcher.hpp
/// @brief Matches a compiled `graph_match.nodes` pattern against a graph and publishes
///        the binding (RFC 0020 §4.3, §6.1).
///
/// A graph matches when its node count equals the pattern's and the pattern's nodes map
/// one-to-one onto the graph's nodes such that:
///  - each graph node carries one of its pattern node's opcodes;
///  - every named edge is present (a `?` operand may be absent), names a graph tensor,
///    and one variable always names one tensor uid;
///  - no optional edge the pattern leaves unnamed is present;
///  - a variable no pattern node produces has no producer in the graph.
/// Distinct variables may name the same tensor. A tensor-uid list field (OpSchema's
/// resultLists) is never bound, but its uids count as that node's results. A tensor a list
/// produces is therefore not a graph input and cannot have a second producer.
///
/// The search is deterministic. It visits pattern nodes in CompiledGraphPattern::matchOrder()
/// and graph nodes in graph order, and the first complete assignment wins. It reads the raw
/// flatbuffer graph, not IGraph's lazily built wrappers, because those are not safe to build
/// from two threads at once.

namespace hipdnn_plugin_sdk::ingestor
{

namespace detail
{

/// One graph, indexed once per match: tensors by uid, plus each uid's producer and
/// consumers.
///
/// Producers include result-list fields, so the graph-input rule sees every producer.
/// Consumers only narrow the search from a bound variable to the graph nodes that read it.
/// A pattern edge is always a scalar field, so readers through operand lists are left out.
struct PatternGraphIndex
{
    std::vector<const hipdnn_flatbuffers_sdk::data_objects::Node*> nodes;
    std::vector<const hipdnn_flatbuffers_sdk::data_objects::op_schema::OpSchema*> schemas;
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        tensors;
    std::unordered_map<int64_t, size_t> producers;
    /// Graph node indices that read the uid through a scalar operand field. Ascending, no
    /// duplicates.
    std::unordered_map<int64_t, std::vector<size_t>> consumers;
};

/// Returns nullopt when the graph cannot be matched: missing node or tensor vectors, a node
/// with no op-schema or no attribute table, a repeated tensor uid, or a uid produced twice.
inline std::optional<PatternGraphIndex>
    indexPatternGraph(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    namespace op_schema = hipdnn_flatbuffers_sdk::data_objects::op_schema;
    const auto* nodes = graph.nodes();
    const auto* tensors = graph.tensors();
    if(nodes == nullptr || tensors == nullptr)
    {
        return std::nullopt;
    }

    PatternGraphIndex index;
    index.tensors.reserve(tensors->size());
    for(const auto* tensor : *tensors)
    {
        if(tensor == nullptr || !index.tensors.emplace(tensor->uid(), tensor).second)
        {
            return std::nullopt;
        }
    }

    index.nodes.reserve(nodes->size());
    index.schemas.reserve(nodes->size());
    for(const auto* node : *nodes)
    {
        if(node == nullptr || node->attributes() == nullptr)
        {
            return std::nullopt;
        }
        const auto* schema = op_schema::findOpSchema(node->attributes_type());
        if(schema == nullptr)
        {
            return std::nullopt;
        }
        const auto nodeIndex = index.nodes.size();
        for(const auto& operand : schema->operands)
        {
            const auto uid = operand.read(*node);
            if(!uid.has_value())
            {
                continue;
            }
            auto& readers = index.consumers[*uid];
            // A node reading one uid through two operands is one consumer.
            if(readers.empty() || readers.back() != nodeIndex)
            {
                readers.push_back(nodeIndex);
            }
        }
        for(const auto& result : schema->results)
        {
            const auto uid = result.read(*node);
            if(uid.has_value() && !index.producers.emplace(*uid, nodeIndex).second)
            {
                return std::nullopt;
            }
        }
        for(const auto& resultList : schema->resultLists)
        {
            if(const auto* uids = resultList.read(*node); uids != nullptr)
            {
                for(const auto uid : *uids)
                {
                    if(!index.producers.emplace(uid, nodeIndex).second)
                    {
                        return std::nullopt;
                    }
                }
            }
        }
        index.nodes.push_back(node);
        index.schemas.push_back(schema);
    }
    return index;
}

/// Backtracking search for an injective pattern-node to graph-node assignment.
class PatternSearch
{
public:
    enum class Outcome
    {
        FOUND,
        NOT_FOUND,
        BUDGET_EXHAUSTED
    };

    enum class VariableState : uint8_t
    {
        UNSET,
        ABSENT,
        BOUND
    };

    PatternSearch(const CompiledGraphPattern& pattern, const PatternGraphIndex& graph)
        : _pattern(pattern)
        , _graph(graph)
        , _graphNodeUsed(graph.nodes.size(), false)
        , _assignedGraphNode(pattern.nodeCount(), 0)
        , _assignedCandidate(pattern.nodeCount(), 0)
        , _state(pattern.variables().size(), VariableState::UNSET)
        , _uid(pattern.variables().size(), 0)
    {
    }

    Outcome run()
    {
        return search(0);
    }

    size_t assignedGraphNode(size_t patternNode) const
    {
        return _assignedGraphNode[patternNode];
    }

    size_t assignedCandidate(size_t patternNode) const
    {
        return _assignedCandidate[patternNode];
    }

    VariableState state(size_t variable) const
    {
        return _state[variable];
    }

    int64_t uid(size_t variable) const
    {
        return _uid[variable];
    }

private:
    Outcome search(size_t depth)
    {
        if(depth == _pattern.matchOrder().size())
        {
            return Outcome::FOUND;
        }
        const auto patternNode = _pattern.matchOrder()[depth];
        const auto& compiled = _pattern.nodes()[patternNode];

        // An operand whose variable is already bound limits the candidates to that uid's
        // consumers. Consumer lists are in graph order, so the search order is unchanged.
        const std::vector<size_t>* consumers = nullptr;
        for(const auto variable : compiled.operandVariables)
        {
            if(_state[variable] != VariableState::BOUND)
            {
                continue;
            }
            const auto it = _graph.consumers.find(_uid[variable]);
            if(it == _graph.consumers.end())
            {
                return Outcome::NOT_FOUND;
            }
            if(consumers == nullptr || it->second.size() < consumers->size())
            {
                consumers = &it->second;
            }
        }

        if(consumers != nullptr)
        {
            for(const auto graphNode : *consumers)
            {
                const auto outcome = tryAssign(depth, patternNode, graphNode);
                if(outcome != Outcome::NOT_FOUND)
                {
                    return outcome;
                }
            }
            return Outcome::NOT_FOUND;
        }
        for(size_t graphNode = 0; graphNode < _graph.nodes.size(); ++graphNode)
        {
            const auto outcome = tryAssign(depth, patternNode, graphNode);
            if(outcome != Outcome::NOT_FOUND)
            {
                return outcome;
            }
        }
        return Outcome::NOT_FOUND;
    }

    /// Tries @p graphNode as the image of @p patternNode, then the rest of the search.
    Outcome tryAssign(size_t depth, size_t patternNode, size_t graphNode)
    {
        if(_graphNodeUsed[graphNode])
        {
            return Outcome::NOT_FOUND;
        }
        if(++_steps > MAX_PATTERN_MATCH_STEPS)
        {
            return Outcome::BUDGET_EXHAUSTED;
        }
        const auto& compiled = _pattern.nodes()[patternNode];
        const auto type = _graph.schemas[graphNode]->attributesType;
        for(size_t candidate = 0; candidate < compiled.candidates.size(); ++candidate)
        {
            if(compiled.candidates[candidate].schema->attributesType != type)
            {
                continue;
            }
            const auto mark = _trail.size();
            if(bindNode(compiled, compiled.candidates[candidate], *_graph.nodes[graphNode]))
            {
                _graphNodeUsed[graphNode] = true;
                _assignedGraphNode[patternNode] = graphNode;
                _assignedCandidate[patternNode] = candidate;
                const auto outcome = search(depth + 1);
                if(outcome != Outcome::NOT_FOUND)
                {
                    return outcome;
                }
                _graphNodeUsed[graphNode] = false;
            }
            undo(mark);
            // Opcodes in one set are distinct ops, so at most one candidate fits.
            break;
        }
        return Outcome::NOT_FOUND;
    }

    bool bindNode(const CompiledPatternNode& compiled,
                  const CompiledPatternCandidate& candidate,
                  const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        for(const auto* edge : candidate.unnamedOptionalEdges)
        {
            if(edge->read(node).has_value())
            {
                return false;
            }
        }
        for(size_t i = 0; i < compiled.operandVariables.size(); ++i)
        {
            if(!bindEdge(compiled.operandVariables[i], candidate.operands[i]->read(node)))
            {
                return false;
            }
        }
        for(size_t i = 0; i < compiled.resultVariables.size(); ++i)
        {
            if(!bindEdge(compiled.resultVariables[i], candidate.results[i]->read(node)))
            {
                return false;
            }
        }
        return true;
    }

    bool bindEdge(size_t variable, std::optional<int64_t> uid)
    {
        const auto& declared = _pattern.variables()[variable];
        if(!uid.has_value())
        {
            // Only a `?` graph input may be absent. parseGraphPattern() gives it exactly one
            // binding edge, so it is still unset here.
            if(!declared.optional)
            {
                return false;
            }
            setVariable(variable, VariableState::ABSENT, 0);
            return true;
        }
        if(_graph.tensors.count(*uid) == 0)
        {
            return false;
        }
        if(_state[variable] == VariableState::BOUND)
        {
            return _uid[variable] == *uid;
        }
        // A graph input must not be produced inside the graph. A produced variable needs no
        // check: its producer's result field supplies the uid, and every other edge must
        // agree with it.
        if(declared.binding.direction == EdgeDirection::OPERAND
           && _graph.producers.count(*uid) != 0)
        {
            return false;
        }
        setVariable(variable, VariableState::BOUND, *uid);
        return true;
    }

    void setVariable(size_t variable, VariableState state, int64_t uid)
    {
        _state[variable] = state;
        _uid[variable] = uid;
        _trail.push_back(variable);
    }

    void undo(size_t mark)
    {
        while(_trail.size() > mark)
        {
            _state[_trail.back()] = VariableState::UNSET;
            _trail.pop_back();
        }
    }

    const CompiledGraphPattern& _pattern;
    const PatternGraphIndex& _graph;
    std::vector<bool> _graphNodeUsed;
    std::vector<size_t> _assignedGraphNode;
    std::vector<size_t> _assignedCandidate;
    std::vector<VariableState> _state;
    std::vector<int64_t> _uid;
    std::vector<size_t> _trail;
    size_t _steps = 0;
};

/// Adds one attribute's value to @p bound; false when the value is unpublishable.
inline bool publishPatternAttribute(
    BoundTokens& bound,
    const CompiledPatternAttribute& attribute,
    const hipdnn_flatbuffers_sdk::data_objects::op_schema::AttributeSchema& schema,
    const hipdnn_flatbuffers_sdk::data_objects::Node& node)
{
    const auto value = schema.read(node);
    if(!value.has_value())
    {
        return attribute.optional;
    }
    switch(attribute.kind)
    {
    case SymbolKind::INT:
        if(const auto* number = std::get_if<int64_t>(&*value))
        {
            bound.emplace(attribute.token, *number);
            return true;
        }
        return false;
    case SymbolKind::FLOAT:
        if(const auto* number = std::get_if<double>(&*value);
           number != nullptr && std::isfinite(*number))
        {
            bound.emplace(attribute.token, *number);
            return true;
        }
        return false;
    case SymbolKind::BOOL:
        if(const auto* flag = std::get_if<bool>(&*value))
        {
            bound.emplace(attribute.token, *flag);
            return true;
        }
        return false;
    case SymbolKind::ENUM_NAME:
        if(const auto* name = std::get_if<std::string_view>(&*value);
           name != nullptr && !name->empty())
        {
            bound.emplace(attribute.token, std::string(*name));
            return true;
        }
        return false;
    case SymbolKind::TENSOR:
    default:
        return false;
    }
}

inline std::optional<BoundTokens> matchGraphPatternOrThrow(const CompiledGraphPattern& pattern,
                                                           const MatchContext& context)
{
    if(context.graph.nodeCount() == 0 || context.graph.nodeCount() != pattern.nodeCount())
    {
        return std::nullopt;
    }
    const auto index = indexPatternGraph(context.graph.getGraph());
    if(!index.has_value() || index->nodes.size() != pattern.nodeCount())
    {
        return std::nullopt;
    }

    PatternSearch search(pattern, *index);
    const auto outcome = search.run();
    if(outcome == PatternSearch::Outcome::BUDGET_EXHAUSTED)
    {
        HIPDNN_PLUGIN_LOG_WARN("graph pattern matcher: " << pattern.where() << " exceeded "
                                                         << MAX_PATTERN_MATCH_STEPS
                                                         << " search steps; declining the graph");
        return std::nullopt;
    }
    if(outcome != PatternSearch::Outcome::FOUND)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    if(!publishGraph(bound, context.graph))
    {
        return std::nullopt;
    }
    const auto& variables = pattern.variables();
    for(size_t variable = 0; variable < variables.size(); ++variable)
    {
        if(search.state(variable) == PatternSearch::VariableState::BOUND
           && !publishTensor(
               bound, variables[variable].name, index->tensors.at(search.uid(variable))))
        {
            return std::nullopt;
        }
    }
    for(size_t patternNode = 0; patternNode < pattern.nodeCount(); ++patternNode)
    {
        const auto& compiled = pattern.nodes()[patternNode];
        const auto& candidate = compiled.candidates[search.assignedCandidate(patternNode)];
        const auto& node = *index->nodes[search.assignedGraphNode(patternNode)];
        for(size_t i = 0; i < compiled.attributes.size(); ++i)
        {
            if(!publishPatternAttribute(
                   bound, compiled.attributes[i], *candidate.attributes[i], node))
            {
                return std::nullopt;
            }
        }
    }
    return bound;
}

} // namespace detail

/// Matches @p context's graph against @p pattern and publishes the binding: graph fields,
/// every present variable's tensor fields, and every compiled attribute the matched node
/// carries. nullopt declines the graph. Never throws. An unexpected failure is logged and
/// declines the graph.
inline std::optional<BoundTokens> matchGraphPattern(const CompiledGraphPattern& pattern,
                                                    const MatchContext& context)
{
    try
    {
        return detail::matchGraphPatternOrThrow(pattern, context);
    }
    catch(const std::exception& error)
    {
        HIPDNN_PLUGIN_LOG_ERROR("graph pattern matcher: "
                                << pattern.where() << " failed while matching: " << error.what()
                                << "; declining");
    }
    catch(...)
    {
        HIPDNN_PLUGIN_LOG_ERROR("graph pattern matcher: " << pattern.where()
                                                          << " failed while matching; declining");
    }
    return std::nullopt;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
