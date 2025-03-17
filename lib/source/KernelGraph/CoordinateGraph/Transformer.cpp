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

#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdgeVisitor.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>

// TODO Remove this when Workgroup removed from RegisterTagManager
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        Transformer::Transformer(CoordinateGraph const* graph)
            : Transformer::Transformer(graph, Expression::simplify)
        {
        }

        Transformer::Transformer(CoordinateGraph const*           graph,
                                 Expression::ExpressionTransducer transducer)
            : m_graph(graph)
            , m_transducer(transducer)
        {
            AssertFatal(graph != nullptr);
        }

        void Transformer::fillExecutionCoordinates(ContextPtr context)
        {
            auto const& kernelWorkgroupIndexes = context->kernel()->workgroupIndex();
            auto const& kernelWorkitemIndexes  = context->kernel()->workitemIndex();
            for(auto const& tag : m_graph->getNodes())
            {
                auto dimension = m_graph->getNode(tag);
                if(std::holds_alternative<Workgroup>(dimension))
                {
                    auto dimensionWorkgroup = std::get<Workgroup>(dimension);
                    AssertFatal(dimensionWorkgroup.dim >= 0
                                    && kernelWorkgroupIndexes.size()
                                           > (size_t)dimensionWorkgroup.dim,
                                "Unable to get workgroup size (kernel dimension mismatch).",
                                ShowValue(toString(dimension)),
                                ShowValue(dimensionWorkgroup.dim),
                                ShowValue(kernelWorkgroupIndexes.size()));
                    auto expr = kernelWorkgroupIndexes.at(dimensionWorkgroup.dim)->expression();
                    // TODO Remove this when Workgroup removed from RegisterTagManager
                    context->registerTagManager()->addRegister(
                        tag, kernelWorkgroupIndexes.at(dimensionWorkgroup.dim));
                    setCoordinate(tag, expr);
                }
                if(std::holds_alternative<Workitem>(dimension))
                {
                    auto dimensionWorkitem = std::get<Workitem>(dimension);
                    AssertFatal(dimensionWorkitem.dim >= 0
                                    && kernelWorkitemIndexes.size() > (size_t)dimensionWorkitem.dim,
                                "Unable to get workitem size (kernel dimension mismatch).",
                                ShowValue(toString(dimension)),
                                ShowValue(dimensionWorkitem.dim),
                                ShowValue(kernelWorkitemIndexes.size()));
                    auto expr = kernelWorkitemIndexes.at(dimensionWorkitem.dim)->expression();
                    setCoordinate(tag, expr);
                }
            }
        }

        void Transformer::setTransducer(Expression::ExpressionTransducer transducer)
        {
            m_transducer = transducer;
        }

        void Transformer::setCoordinate(int tag, Expression::ExpressionPtr index)
        {
            if(Log::getLogger()->should_log(LogLevel::Debug))
            {
                auto elemName = std::visit([](auto const& el) { return toString(el); },
                                           m_graph->getElement(tag));

                rocRoller::Log::getLogger()->debug(
                    "Transformer::setCoordinate: setting {} ({}) to {}",
                    tag,
                    elemName,
                    toString(index));
            }
            m_indexes.insert_or_assign(tag, index);
        }

        Expression::ExpressionPtr Transformer::getCoordinate(int tag) const
        {
            AssertFatal(m_indexes.contains(tag), "Coordinate not set.", ShowValue(tag));
            return m_indexes.at(tag);
        }

        bool Transformer::hasCoordinate(int tag) const
        {
            return m_indexes.contains(tag);
        }

        void Transformer::removeCoordinate(int tag)
        {
            m_indexes.erase(tag);
        }

        bool Transformer::hasPath(std::vector<int> const& dsts, bool forward) const
        {
            std::vector<int> srcs;
            for(auto const& kv : m_indexes)
            {
                srcs.push_back(kv.first);
            }
            if(forward)
                return m_graph->hasPath<Graph::Direction::Downstream>(srcs, dsts);
            return m_graph->hasPath<Graph::Direction::Upstream>(dsts, srcs);
        }

        std::vector<Expression::ExpressionPtr>
            Transformer::forward(std::vector<int> const& dsts) const
        {
            std::vector<Expression::ExpressionPtr> indexes;
            std::vector<int>                       srcs;
            for(auto const& kv : m_indexes)
            {
                srcs.push_back(kv.first);
                indexes.push_back(kv.second);
            }
            return transduce(m_graph->forward(indexes, srcs, dsts));
        }

        std::vector<Expression::ExpressionPtr>
            Transformer::reverse(std::vector<int> const& dsts) const
        {
            std::vector<Expression::ExpressionPtr> indexes;
            std::vector<int>                       srcs;
            for(auto const& kv : m_indexes)
            {
                srcs.push_back(kv.first);
                indexes.push_back(kv.second);
            }
            return transduce(m_graph->reverse(indexes, dsts, srcs));
        }

        template <typename Visitor>
        std::vector<Expression::ExpressionPtr>
            Transformer::stride(std::vector<int> const& dsts, bool forward, Visitor& visitor) const

        {
            std::vector<Expression::ExpressionPtr> indexes;
            std::vector<int>                       srcs;
            for(auto const& kv : m_indexes)
            {
                srcs.push_back(kv.first);
                indexes.push_back(kv.second);
            }

            // this call to traverse with EdgeDiffVisitor populates the deltas associated with dsts.
            if(forward)
                m_graph->traverse<Graph::Direction::Downstream>(indexes, srcs, dsts, visitor);
            else
                m_graph->traverse<Graph::Direction::Upstream>(indexes, dsts, srcs, visitor);

            std::vector<Expression::ExpressionPtr> deltas;
            for(auto const& dst : dsts)
            {
                auto delta = visitor.deltas.at(dst);
                deltas.push_back(delta);
            }
            return transduce(deltas);
        }

        std::vector<Expression::ExpressionPtr> Transformer::forwardStride(
            int x, Expression::ExpressionPtr dx, std::vector<int> const& dsts) const
        {
            AssertFatal(dx);
            auto visitor = ForwardEdgeDiffVisitor(x, dx);
            return stride(dsts, true, visitor);
        }

        std::vector<Expression::ExpressionPtr> Transformer::reverseStride(
            int x, Expression::ExpressionPtr dx, std::vector<int> const& dsts) const
        {
            AssertFatal(dx);
            auto visitor = ReverseEdgeDiffVisitor(x, dx);
            return stride(dsts, false, visitor);
        }

        Expression::ExpressionPtr Transformer::transduce(Expression::ExpressionPtr exp) const
        {
            if(!m_transducer)
                return exp;

            return m_transducer(exp);
        }
        std::vector<Expression::ExpressionPtr>
            Transformer::transduce(std::vector<Expression::ExpressionPtr> exps) const
        {
            if(!m_transducer)
                return exps;

            for(auto& exp : exps)
                exp = m_transducer(exp);

            return exps;
        }
    }
}
