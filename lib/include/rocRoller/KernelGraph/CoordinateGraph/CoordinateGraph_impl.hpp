#pragma once

#include <vector>

#include "CoordinateGraph.hpp"
#include <rocRoller/Graph/Hypergraph.hpp>

namespace rocRoller
{

    namespace KernelGraph::CoordinateGraph
    {

        template <Graph::Direction Dir, typename Visitor>
        inline std::vector<Expression::ExpressionPtr>
            CoordinateGraph::traverse(std::vector<Expression::ExpressionPtr> sdims,
                                      std::vector<int> const&                srcs,
                                      std::vector<int> const&                dsts,
                                      Visitor&                               visitor,
                                      Expression::ExpressionTransducer       transducer)
        {
            bool constexpr forward     = Dir == Graph::Direction::Downstream;
            auto constexpr OppositeDir = opposite(Dir);

            auto const& starts = forward ? srcs : dsts;
            auto const& ends   = forward ? dsts : srcs;

            std::map<int, Expression::ExpressionPtr> exprMap;
            std::vector<Expression::ExpressionPtr>   visitedExprs;

            auto edgeSelector = [this](int element) {
                return getEdgeType(element) == EdgeType::CoordinateTransform;
            };

            for(size_t i = 0; i < sdims.size(); i++)
            {
                int key = starts[i];
                exprMap.emplace(key, sdims[i]);
            }

            // traverse through the edges in the path from `starts` to `ends`
            // and generate expressions (populates exprMap) for the target
            // dimension(s) in `ends` via the given visitor.
            for(auto const elemId : path<Dir>(starts, ends, edgeSelector))
            {
                Element const& element = getElement(elemId);
                if(std::holds_alternative<Edge>(element))
                {
                    Edge const& edge = std::get<CoordinateTransformEdge>(std::get<Edge>(element));

                    std::vector<Expression::ExpressionPtr> einds;
                    std::vector<int>                       keys, localSrcTags, localDstTags;
                    std::vector<Dimension>                 localSrcs, localDsts;

                    for(auto const& tag : getNeighbours<Graph::Direction::Upstream>(elemId))
                    {
                        if(forward)
                        {
                            einds.push_back(exprMap[tag]);
                        }
                        else
                        {
                            keys.push_back(tag);
                        }
                        localSrcs.emplace_back(getNode(tag));
                        localSrcTags.emplace_back(tag);
                    }
                    for(auto const& tag : getNeighbours<Graph::Direction::Downstream>(elemId))
                    {
                        if(!forward)
                        {
                            einds.push_back(exprMap[tag]);
                        }
                        else
                        {
                            keys.push_back(tag);
                        }
                        localDsts.emplace_back(getNode(tag));
                        localDstTags.emplace_back(tag);
                    }

                    visitor.setLocation(einds, localSrcs, localDsts, localSrcTags, localDstTags);
                    visitedExprs = visitor.call(edge);

                    AssertFatal(visitedExprs.size() == keys.size(), ShowValue(visitedExprs));
                    for(size_t i = 0; i < visitedExprs.size(); i++)
                    {
                        exprMap[keys[i]] = std::move(visitedExprs[i]);
                    }
                }
            }

            std::vector<Expression::ExpressionPtr> results;

            for(int const key : ends)
            {
                if(!exprMap.contains(key))
                {
                    auto keys = [&exprMap]() -> Generator<int> {
                        for(auto const& pair : exprMap)
                            co_yield pair.first;
                    }()
                                                    .template to<std::vector>();
                    std::ostringstream msg;
                    streamJoin(msg, keys, ", ");
                    AssertFatal(exprMap.contains(key),
                                "Path not found for ",
                                Graph::variantToString(getElement(key)),
                                ShowValue(key),
                                msg.str());
                }
                results.push_back(transducer ? transducer(exprMap.at(key)) : exprMap.at(key));
            }

            return results;
        }

        template <Graph::Direction Dir>
        inline bool CoordinateGraph::hasPath(std::vector<int> const& srcs,
                                             std::vector<int> const& dsts)
        {
            bool constexpr forward = Dir == Graph::Direction::Downstream;

            auto const& starts = forward ? srcs : dsts;
            auto const& ends   = forward ? dsts : srcs;

            auto edgeSelector = [this](int element) {
                return getEdgeType(element) == EdgeType::CoordinateTransform;
            };

            auto partial = path<Dir>(starts, ends, edgeSelector).template to<std::unordered_set>();

            for(auto end : ends)
            {
                if(!partial.contains(end))
                    return false;
            }

            return true;
        }

        inline EdgeType CoordinateGraph::getEdgeType(int index)
        {
            Element const& elem = getElement(index);
            if(std::holds_alternative<Edge>(elem))
            {
                Edge const& edge = std::get<Edge>(elem);
                if(std::holds_alternative<DataFlowEdge>(edge))
                {
                    return EdgeType::DataFlow;
                }
                else if(std::holds_alternative<CoordinateTransformEdge>(edge))
                {
                    return EdgeType::CoordinateTransform;
                }
            }
            return EdgeType::None;
        }

        template <typename T>
        requires(std::constructible_from<CoordinateGraph::Element, T>)
            std::optional<T> CoordinateGraph::get(int tag)
        const
        {
            auto x = getElement(tag);
            if constexpr(std::constructible_from<Edge, T>)
            {
                if(std::holds_alternative<Edge>(x))
                {
                    auto y = std::get<Edge>(x);
                    if constexpr(std::constructible_from<DataFlowEdge, T>)
                    {
                        if(std::holds_alternative<DataFlowEdge>(y))
                        {
                            if(std::holds_alternative<T>(std::get<DataFlowEdge>(y)))
                            {
                                return std::get<T>(std::get<DataFlowEdge>(y));
                            }
                        }
                    }
                    else if constexpr(std::constructible_from<CoordinateTransformEdge, T>)
                    {
                        if(std::holds_alternative<CoordinateTransformEdge>(y))
                        {
                            if(std::holds_alternative<T>(std::get<CoordinateTransformEdge>(y)))
                            {
                                return std::get<T>(std::get<CoordinateTransformEdge>(y));
                            }
                        }
                    }
                }
            }
            if constexpr(std::constructible_from<Dimension, T>)
            {
                if(std::holds_alternative<Dimension>(x))
                {
                    if(std::holds_alternative<T>(std::get<Dimension>(x)))
                    {
                        return std::get<T>(std::get<Dimension>(x));
                    }
                }
            }
            return {};
        }

        inline std::string name(CoordinateGraph::Element const& el)
        {
            return CoordinateGraph::ElementName(el);
        }
    }
}
