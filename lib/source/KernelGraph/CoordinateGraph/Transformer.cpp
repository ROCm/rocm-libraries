
#include "Graph/Hypergraph.hpp"
#include "KernelGraph/CoordGraph/CoordinateHypergraph.hpp"
#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/CoordGraph/EdgeVisitor.hpp>
#include <rocRoller/KernelGraph/CoordGraph/Transformer.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordGraph
    {
        using namespace Expression;

        /*
         * Diff edge visitors.
         */
        struct BaseEdgeDiffVisitor : public BaseEdgeVisitor
        {
            ExpressionPtr zero;

            std::map<int, ExpressionPtr> deltas;

            BaseEdgeDiffVisitor() = delete;
            BaseEdgeDiffVisitor(int x, ExpressionPtr dx)
            {
                deltas.emplace(x, dx);
                zero = literal(0);
            }

            //
            // Get delta associated with Dimension thus far.
            //
            ExpressionPtr getDelta(int tag) const
            {
                if(deltas.count(tag) > 0)
                {
                    return literal(evaluate(deltas.at(tag)));
                }
                return zero;
            }
        };

        struct ForwardEdgeDiffVisitor : public BaseEdgeDiffVisitor
        {
            using BaseEdgeDiffVisitor::BaseEdgeDiffVisitor;

            std::vector<ExpressionPtr> operator()(Flatten const& e)
            {
                AssertFatal(srcs.size() > 0 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                std::vector<ExpressionPtr> strides(srcs.size());
                strides[srcs.size() - 1] = literal(1);
                for(size_t i = srcs.size() - 1; i > 0; --i)
                {
                    strides[i - 1] = strides[i] * getSize(srcs[i]);
                }

                auto index = indexes[0] * strides[0];
                auto delta = getDelta(srcTags[0]) * strides[0];
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    index = index + indexes[d] * strides[d];
                    delta = delta + getDelta(srcTags[d]) * strides[d];
                }
                deltas.emplace(dstTags[0], delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(Join const& e)
            {
                AssertFatal(srcs.size() > 0 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                auto index = indexes[0] * getStride(srcs[0]);
                auto delta = getDelta(srcTags[0]) * getStride(srcs[0]);
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    index = index + indexes[d] * getStride(srcs[d]);
                    delta = delta + getDelta(srcTags[d]) * getStride(srcs[d]);
                }
                deltas.emplace(dstTags[0], delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(Tile const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                auto delta = getDelta(srcTags[0]);
                auto input = indexes[0];

                std::vector<ExpressionPtr> rv(dsts.size());
                for(int i = dsts.size() - 1; i > 0; i--)
                {
                    auto size = getSize(dsts[i]);
                    rv[i]     = input % size;
                    input     = input / size;
                    deltas.emplace(dstTags[i], delta % size);
                    delta = delta / size;
                }
                deltas.emplace(dstTags[0], delta);
                rv[0] = input;
                return rv;
            }

            std::vector<ExpressionPtr> operator()(PassThrough const& e)
            {
                AssertFatal(srcs.size() == 1 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                auto delta = getDelta(srcTags[0]);
                deltas.emplace(dstTags[0], delta);
                return {indexes};
            }

            template <CTUndefinedEdge T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <CTEdgePassthrough T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                return std::visit(*this, e);
            }

            template <typename T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Forward derivative not implemented yet for: ",
                                  ShowValue(e.toString()));
            }
        };

        struct ReverseEdgeDiffVisitor : public BaseEdgeDiffVisitor
        {
            using BaseEdgeDiffVisitor::BaseEdgeDiffVisitor;

            std::vector<ExpressionPtr> operator()(Split const& e)
            {
                AssertFatal(dsts.size() > 0 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto index = indexes[0] * getStride(dsts[0]);
                auto delta = getDelta(dstTags[0]) * getStride(dsts[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    index = index + indexes[d] * getStride(dsts[d]);
                    delta = delta + getDelta(dstTags[d]) * getStride(dsts[d]);
                }
                deltas.emplace(srcTags[0], delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(Tile const& e)
            {
                AssertFatal(dsts.size() > 0 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto index = indexes[0];
                auto delta = getDelta(dstTags[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    index = index * getSize(dsts[d]) + indexes[d];
                    delta = delta * getSize(dsts[d]) + getDelta(dstTags[d]);
                }
                deltas.emplace(srcTags[0], delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(Flatten const& e)
            {
                AssertFatal(dsts.size() == 1 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() > 1, ShowValue(srcs.size()));

                auto delta = getDelta(dstTags[0]);
                auto input = indexes[0];

                std::vector<ExpressionPtr> rv(srcs.size());
                for(int i = srcs.size() - 1; i > 0; i--)
                {
                    auto size = getSize(srcs[i]);
                    rv[i]     = input % size;
                    input     = input / size;
                    deltas.emplace(srcTags[i], delta % size);
                    delta = delta / size;
                }
                deltas.emplace(srcTags[0], delta);
                rv[0] = input;
                return rv;
            }

            std::vector<ExpressionPtr> operator()(PassThrough const& e)
            {
                AssertFatal(dsts.size() == 1 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto delta = getDelta(dstTags[0]);
                deltas.emplace(srcTags[0], delta);
                return {indexes};
            }

            template <CTUndefinedEdge T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <CTEdgePassthrough T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                return std::visit(*this, e);
            }

            template <typename T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Reverse derivative not implemented yet for: ",
                                  ShowValue(e.toString()));
            }
        };

        Transformer::Transformer(std::shared_ptr<CoordinateHypergraph> graph,
                                 std::shared_ptr<Context>              context,
                                 ExpressionTransducer                  transducer)
            : m_graph(graph)
            , m_context(context)
            , m_transducer(transducer)
        {
            if(m_context)
                fillExecutionCoordinates();
        }

        void Transformer::fillExecutionCoordinates()
        {
            auto const& kernelWorkgroupIndexes = m_context->kernel()->workgroupIndex();
            auto const& kernelWorkitemIndexes  = m_context->kernel()->workitemIndex();
            for(auto const& tag : m_graph->getNodes())
            {
                auto dimension = std::get<Dimension>(m_graph->getElement(tag));
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

        void Transformer::setTransducer(ExpressionTransducer transducer)
        {
            m_transducer = transducer;
        }

        ExpressionTransducer Transformer::getTransducer() const
        {
            return m_transducer;
        }

        void Transformer::setCoordinate(int tag, ExpressionPtr index)
        {
            rocRoller::Log::getLogger()->debug(
                "Transformer::setCoordinate: setting {} to {}", tag, toString(index));
            m_indexes.insert_or_assign(tag, index);
        }

        void Transformer::removeCoordinate(int tag)
        {
            m_indexes.erase(tag);
        }

        std::vector<ExpressionPtr> Transformer::forward(std::vector<int> const& dsts) const
        {
            std::vector<ExpressionPtr> indexes;
            std::vector<int>           srcs;
            for(auto const& kv : m_indexes)
            {
                srcs.push_back(kv.first);
                indexes.push_back(kv.second);
            }
            return m_graph->forward(indexes, srcs, dsts, m_transducer);
        }

        std::vector<ExpressionPtr> Transformer::reverse(std::vector<int> const& dsts) const
        {
            std::vector<ExpressionPtr> indexes;
            std::vector<int>           srcs;
            for(auto const& kv : m_indexes)
            {
                srcs.push_back(kv.first);
                indexes.push_back(kv.second);
            }
            return m_graph->reverse(indexes, dsts, srcs, m_transducer);
        }

        template <typename Visitor>
        std::vector<ExpressionPtr>
            Transformer::stride(std::vector<int> const& dsts, bool forward, Visitor& visitor) const

        {
            std::vector<ExpressionPtr> indexes;
            std::vector<int>           srcs;
            for(auto const& kv : m_indexes)
            {
                srcs.push_back(kv.first);
                indexes.push_back(kv.second);
            }

            if(forward)
                m_graph->traverse<Graph::Direction::Downstream>(
                    indexes, srcs, dsts, visitor, m_transducer);
            else
                m_graph->traverse<Graph::Direction::Upstream>(
                    indexes, dsts, srcs, visitor, m_transducer);

            std::vector<ExpressionPtr> deltas;
            for(auto const& dst : dsts)
            {
                auto delta = visitor.deltas.at(dst);
                deltas.push_back(m_transducer ? m_transducer(delta) : delta);
            }
            return deltas;
        }

        std::vector<ExpressionPtr>
            Transformer::forwardStride(int x, ExpressionPtr dx, std::vector<int> const& dsts) const
        {
            AssertFatal(dx);
            auto visitor = ForwardEdgeDiffVisitor(x, dx);
            return stride(dsts, true, visitor);
        }

        std::vector<ExpressionPtr>
            Transformer::reverseStride(int x, ExpressionPtr dx, std::vector<int> const& dsts) const
        {
            AssertFatal(dx);
            auto visitor = ReverseEdgeDiffVisitor(x, dx);
            return stride(dsts, false, visitor);
        }
    }
}
