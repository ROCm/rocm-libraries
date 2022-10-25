#pragma once

#include "CoordinateHypergraph.hpp"
#include "EdgeVisitor.hpp"
#include "Graph/Hypergraph.hpp"
#include <vector>

namespace rocRoller
{

    namespace KernelGraph::CoordGraph
    {

        inline std::vector<Expression::ExpressionPtr>
            CoordinateHypergraph::forward(std::vector<Expression::ExpressionPtr> sdims,
                                          std::vector<int> const&                srcs,
                                          std::vector<int> const&                dsts,
                                          Expression::ExpressionTransducer       transducer)
        {
            AssertFatal(sdims.size() == srcs.size(), ShowValue(sdims));
            auto visitor = ForwardEdgeVisitor();
            return traverse<Graph::Direction::Downstream>(sdims, srcs, dsts, visitor, transducer);
        }

        inline std::vector<Expression::ExpressionPtr>
            CoordinateHypergraph::reverse(std::vector<Expression::ExpressionPtr> sdims,
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
            CoordinateHypergraph::traverse(std::vector<Expression::ExpressionPtr> sdims,
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
                        visitedExprs = visitor(edge);
                    }
                    else
                    {
                        visitor.setLocation(
                            einds, localSrcs, localDsts, localSrcTags, localDstTags);
                        visitedExprs = visitor(edge);
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
                AssertRecoverable(exprMap.contains(key), "Path not found");
                results.push_back(transducer ? transducer(exprMap.at(key)) : exprMap.at(key));
            }

            return results;
        }

        inline EdgeType CoordinateHypergraph::getEdgeType(int index)
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
    }
}
