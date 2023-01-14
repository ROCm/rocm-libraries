#pragma once

#include <vector>

#include "CoordinateEdgeVisitor.hpp"
#include "CoordinateGraph.hpp"
#include "Graph/Hypergraph.hpp"

namespace rocRoller
{

    namespace KernelGraph::CoordinateGraph
    {

        inline std::vector<Expression::ExpressionPtr>
            CoordinateGraph::forward(std::vector<Expression::ExpressionPtr> sdims,
                                     std::vector<int> const&                srcs,
                                     std::vector<int> const&                dsts,
                                     Expression::ExpressionTransducer       transducer)
        {
            AssertFatal(sdims.size() == srcs.size(), ShowValue(sdims));
            auto visitor = ForwardEdgeVisitor();
            return traverse<Graph::Direction::Downstream>(sdims, srcs, dsts, visitor, transducer);
        }

        inline std::vector<Expression::ExpressionPtr>
            CoordinateGraph::reverse(std::vector<Expression::ExpressionPtr> sdims,
                                     std::vector<int> const&                srcs,
                                     std::vector<int> const&                dsts,
                                     Expression::ExpressionTransducer       transducer)
        {
            AssertFatal(sdims.size() == dsts.size(), ShowValue(sdims));
            auto visitor = ReverseEdgeVisitor();
            return traverse<Graph::Direction::Upstream>(sdims, srcs, dsts, visitor, transducer);
        }

        template <Graph::Direction Dir, typename Visitor>
        inline std::vector<Expression::ExpressionPtr>
            CoordinateGraph::traverse(std::vector<Expression::ExpressionPtr> sdims,
                                      std::vector<int> const&                srcs,
                                      std::vector<int> const&                dsts,
                                      Visitor&                               visitor,
                                      Expression::ExpressionTransducer       transducer)
        {
            bool const forward = Dir == Graph::Direction::Downstream;

            std::map<int, Expression::ExpressionPtr> exprMap;
            std::map<int, bool>                      visited;
            std::vector<Expression::ExpressionPtr>   visitedExprs;

            auto edgeSelector = [this](int element) {
                return getEdgeType(element) == EdgeType::CoordinateTransform;
            };

            for(size_t i = 0; i < sdims.size(); i++)
            {
                int key = forward ? srcs[i] : dsts[i];
                exprMap.emplace(key, sdims[i]);
            }

            for(auto const elemId :
                path<Dir>(forward ? srcs : dsts, forward ? dsts : srcs, visited, edgeSelector))
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
                        localSrcs.emplace_back(std::get<Dimension>(getElement(tag)));
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
                        localDsts.emplace_back(std::get<Dimension>(getElement(tag)));
                        localDstTags.emplace_back(tag);
                    }

                    if(forward)
                    {
                        visitor.setLocation(
                            einds, localSrcs, localDsts, localSrcTags, localDstTags);
                        visitedExprs = visitor.call(edge);
                    }
                    else
                    {
                        visitor.setLocation(
                            einds, localSrcs, localDsts, localSrcTags, localDstTags);
                        visitedExprs = visitor.call(edge);
                    }

                    AssertFatal(visitedExprs.size() == keys.size(), ShowValue(visitedExprs));
                    for(size_t i = 0; i < visitedExprs.size(); i++)
                    {
                        exprMap.emplace(keys[i], visitedExprs[i]);
                    }
                }
            }

            std::vector<Expression::ExpressionPtr> results;

            for(int const key : (forward ? dsts : srcs))
            {
                AssertFatal(exprMap.contains(key), "Path not found");
                results.push_back(transducer ? transducer(exprMap.at(key)) : exprMap.at(key));
            }

            return results;
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
    }
}
