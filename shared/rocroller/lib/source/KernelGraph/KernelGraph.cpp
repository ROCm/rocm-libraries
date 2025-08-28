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
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        void KernelGraph::setTransformerByForLoopOp(CoordinateGraph::Transformer& transformer,
                                                    int                           forLoopOp)
        {
            auto loopIncrTag = mapper.get(forLoopOp, NaryArgument::DEST);
            auto expr = std::make_shared<Expression::Expression>(rocRoller::Expression::DataFlowTag{
                loopIncrTag, Register::Type::Scalar, rocRoller::DataType::Int32});
            auto loopDims
                = coordinates.getOutputNodeIndices<CoordinateGraph::DataFlowEdge>(loopIncrTag);
            for(auto const& dim : loopDims | std::views::filter([&](int dim) {
                                      return !transformer.hasCoordinate(dim);
                                  }))
            {
                transformer.setCoordinate(dim, expr);
            }
        }

        void KernelGraph::setTransformerBySetCoordinate(CoordinateGraph::Transformer& transformer,
                                                        int setCoordinateOp)
        {
            auto connections = mapper.getConnections(setCoordinateOp);
            if(not transformer.hasCoordinate(connections[0].coordinate))
            {
                auto setCoordinate
                    = control.get<ControlGraph::SetCoordinate>(setCoordinateOp).value();

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
            for(auto const& node : control.getNodes())
                buildTransformer(node, IgnoreCache);
        }

        rocRoller::KernelGraph::CoordinateGraph::Transformer KernelGraph::buildTransformer(int op)
        {
            if(not m_transformers.contains(op))
                return buildTransformer(op, IgnoreCache);
            return m_transformers.at(op);
        }

        rocRoller::KernelGraph::CoordinateGraph::Transformer
            KernelGraph::buildTransformer(int op, IgnoreCachePolicy const)
        {
            auto [iter, _] = m_transformers.insert_or_assign(op, &coordinates);

            auto const stk = controlStack(op, control);

            for(int index : stk | std::views::reverse)
            {
                std::visit(
                    [&](auto&& node) {
                        using OpType = std::decay_t<decltype(node)>;
                        if constexpr(std::is_same_v<OpType, ControlGraph::SetCoordinate>)
                        {
                            setTransformerBySetCoordinate(iter->second, index);
                        }
                        else if constexpr(std::is_same_v<OpType, ControlGraph::ForLoopOp>)
                        {
                            setTransformerByForLoopOp(iter->second, index);
                        }
                    },
                    control.getNode(index));
            }
            return iter->second;
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

            KernelGraph newGraph = [&]() {
                TIMER(t, "KernelGraph::Transformation::" + transformation->name());
                return transformation->apply(*this);
            }();

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
