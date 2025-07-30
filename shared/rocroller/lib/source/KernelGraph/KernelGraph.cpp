/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        static bool isParentSetCoordinate(ControlGraph::ControlGraph const& graph,
                                          int const                         edge,
                                          int const                         node)
        {
            return graph.get<ControlGraph::SetCoordinate>(node).has_value()
                   && (graph.get<ControlGraph::Initialize>(edge).has_value()
                       || graph.get<ControlGraph::Body>(edge).has_value());
        }

        static bool isParentForLoopOp(ControlGraph::ControlGraph const& graph,
                                      int const                         edge,
                                      int const                         node)
        {
            return graph.get<ControlGraph::ForLoopOp>(node).has_value()
                   && (graph.get<ControlGraph::ForLoopIncrement>(edge).has_value()
                       || graph.get<ControlGraph::Body>(edge).has_value());
        }

        static void buildControlStack(int                               tag,
                                      std::unordered_map<int, int>&     controlStack,
                                      ControlGraph::ControlGraph const& graph)
        {
            using GD = rocRoller::Graph::Direction;

            int parent = -1;

            auto const edge = graph.getNeighbours<Graph::Direction::Upstream>(tag).take(1).only();
            if(edge.has_value())
            {
                auto const node
                    = graph.getNeighbours<Graph::Direction::Upstream>(*edge).take(1).only();
                AssertFatal(node.has_value(), "Node does not exist!");

                if(isParentSetCoordinate(graph, *edge, *node)
                   or isParentForLoopOp(graph, *edge, *node))
                    parent = *node;
                else
                {
                    if(not controlStack.contains(*node))
                        buildControlStack(*node, controlStack, graph);
                    parent = controlStack.at(*node);
                }
            }

            controlStack[tag] = parent;
        }

        static std::unordered_map<int, int> buildControlStack(KernelGraph const& kg)
        {
            std::unordered_map<int, int> controlStack;

            for(auto const node : kg.control.getNodes())
            {
                if(not controlStack.contains(node))
                    buildControlStack(node, controlStack, kg.control);
            }

            return controlStack;
        }

        static void setTransformerByForLoopOp(CoordinateGraph::Transformer& transformer,
                                              KernelGraph&                  kg,
                                              int                           forLoopOp)
        {
            auto loopIncrTag = kg.mapper.get(forLoopOp, NaryArgument::DEST);
            auto expr = std::make_shared<Expression::Expression>(rocRoller::Expression::DataFlowTag{
                loopIncrTag, Register::Type::Scalar, rocRoller::DataType::Int32});
            auto loopDims
                = kg.coordinates.getOutputNodeIndices<CoordinateGraph::DataFlowEdge>(loopIncrTag);
            for(auto const& dim : loopDims | std::views::filter([&](int dim) {
                                      return !transformer.hasCoordinate(dim);
                                  }))
            {
                transformer.setCoordinate(dim, expr);
            }
        }

        static void setTransformerBySetCoordinate(CoordinateGraph::Transformer& transformer,
                                                  KernelGraph&                  kg,
                                                  int                           setCoordinateOp)
        {
            auto connections = kg.mapper.getConnections(setCoordinateOp);
            if(not transformer.hasCoordinate(connections[0].coordinate))
            {
                auto setCoordinate
                    = kg.control.get<ControlGraph::SetCoordinate>(setCoordinateOp).value();

                transformer.setCoordinate(connections[0].coordinate, setCoordinate.value);
            }
        }

        std::string KernelGraph::toDOT(bool drawMappings, std::string title) const
        {
            std::stringstream ss;
            ss << "digraph {\n";
            if(!title.empty())
            {
                ss << "labelloc=\"t\";" << std::endl;
                ss << "label=\"" << title << "\";" << std::endl;
            }
            ss << coordinates.toDOT("coord", false);
            ss << "subgraph clusterCF {";
            ss << "label = \"Control Graph\";" << std::endl;
            ss << control.toDOT("cntrl", false);
            ss << "}" << std::endl;
            if(drawMappings)
            {
                ss << mapper.toDOT("coord", "cntrl");
            }
            ss << "}" << std::endl;
            return ss.str();
        }

        ConstraintStatus
            KernelGraph::checkConstraints(const std::vector<GraphConstraint>& constraints) const
        {
            ConstraintStatus retval;
            for(int i = 0; i < constraints.size(); i++)
            {
                auto check = constraints[i](*this);
                if(!check.satisfied)
                {
                    Log::warn("Constraint failed: {}", check.explanation);
                }
                retval.combine(check);
            }
            return retval;
        }

        void KernelGraph::initializeTransformersForCodeGen(
            Expression::ExpressionTransducer transducer)
        {
            for(auto& p : m_transformers)
            {
                p.second.setCoordinateGraph(&coordinates);
                p.second.setTransducer(transducer);
            }
        }

        void KernelGraph::updateTransformer(int op, int coord, Expression::ExpressionPtr expr)
        {
            AssertFatal(m_transformers.contains(op), "Transformer does not exist");
            m_transformers.at(op).setCoordinate(coord, expr);
        }

        void KernelGraph::buildAllTransformers()
        {
            m_transformers.clear();

            auto cs = buildControlStack(*this);
            for(auto const& [node, parent] : cs)
            {
                auto [iter, _] = m_transformers.emplace(node, &coordinates);

                auto tag = parent;
                while(tag != -1)
                {
                    if(std::holds_alternative<ControlGraph::SetCoordinate>(control.getNode(tag)))
                        setTransformerBySetCoordinate(iter->second, *this, tag);
                    else
                    {
                        AssertFatal(
                            control.isElemType<ControlGraph::ForLoopOp>()(tag),
                            "A node in control stack is not a ForLoopOp nor a SetCoordinate");
                        setTransformerByForLoopOp(iter->second, *this, tag);
                    }

                    tag = cs.at(tag);
                }
            }
        }

        rocRoller::KernelGraph::CoordinateGraph::Transformer KernelGraph::getTransformer(int op)
        {
            if(not m_transformers.contains(op))
            {
                using GD = rocRoller::Graph::Direction;

                auto [iter, _] = m_transformers.emplace(op, &coordinates);

                auto node = op;
                while(true)
                {
                    auto const edge
                        = control.getNeighbours<Graph::Direction::Upstream>(node).take(1).only();
                    if(not edge.has_value())
                        break;

                    auto const parent
                        = control.getNeighbours<Graph::Direction::Upstream>(*edge).take(1).only();

                    if(isParentSetCoordinate(control, edge.value(), parent.value()))
                    {
                        setTransformerBySetCoordinate(iter->second, *this, parent.value());
                    }
                    else if(isParentForLoopOp(control, edge.value(), parent.value()))
                    {
                        setTransformerByForLoopOp(iter->second, *this, parent.value());
                    }

                    node = parent.value();
                }
            }
            return m_transformers.at(op);
        }

        ConstraintStatus KernelGraph::checkConstraints() const
        {
            return checkConstraints(m_constraints);
        }

        void KernelGraph::addConstraints(const std::vector<GraphConstraint>& constraints)
        {
            m_constraints.insert(m_constraints.end(), constraints.begin(), constraints.end());
        }

        std::vector<GraphConstraint> KernelGraph::getConstraints() const
        {
            return m_constraints;
        }

        KernelGraph KernelGraph::transform(std::shared_ptr<GraphTransform> const& transformation)
        {
            auto transformString  = concatenate("KernelGraph::transform ", transformation->name());
            auto checkConstraints = Settings::getInstance()->get(Settings::EnforceGraphConstraints);

            if(checkConstraints)
            {
                auto check = (*this).checkConstraints(transformation->preConstraints());
                AssertFatal(check.satisfied,
                            concatenate(transformString, " PreCheck: \n", check.explanation));
            }

            KernelGraph newGraph = transformation->apply(*this);

            bool drawMappings = Settings::getInstance()->get(Settings::LogGraphMapperConnections);

            if(Settings::getInstance()->get(Settings::LogGraphs))
                Log::debug("KernelGraph::transform: {}, post: {}",
                           transformation->name(),
                           newGraph.toDOT(drawMappings, transformString));

            if(checkConstraints)
            {
                newGraph.addConstraints(transformation->postConstraints());
                auto check = newGraph.checkConstraints();
                AssertFatal(check.satisfied,
                            concatenate(transformString, " PostCheck: \n", check.explanation));
            }

            newGraph.m_transforms.push_back(transformation->name());

            return newGraph;
        }

        std::vector<std::string> const& KernelGraph::appliedTransforms() const
        {
            return m_transforms;
        }

        void KernelGraph::addAppliedTransforms(std::vector<std::string> const& transforms)
        {
            m_transforms.insert(m_transforms.end(), transforms.begin(), transforms.end());
        }
    }
}
