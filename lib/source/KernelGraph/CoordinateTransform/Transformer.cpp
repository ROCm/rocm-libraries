
#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/EdgeVisitor.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/Transformer.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateTransform
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /*
         * Diff edge visitors.
         */
        struct BaseEdgeDiffVisitor : public BaseEdgeVisitor
        {
            ExpressionPtr zero;

            std::map<TagType, ExpressionPtr> deltas;

            BaseEdgeDiffVisitor() = delete;
            BaseEdgeDiffVisitor(Dimension x, ExpressionPtr dx)
            {
                deltas.emplace(getTag(x), dx);
                zero = literal(0);
            }

            //
            // Get delta associated with Dimension thus far.
            //
            ExpressionPtr getDelta(Dimension const& dim) const
            {
                auto tag = getTag(dim);
                if(deltas.count(tag) > 0)
                    return deltas.at(tag);
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

                auto index = indexes[0];
                for(uint d = 1; d < srcs.size(); ++d)
                    index = index * getSize(srcs[d]) + indexes[d];

                std::vector<ExpressionPtr> coeffs(srcs.size());
                for(size_t d = 0; d < srcs.size(); ++d)
                {
                    coeffs[d] = literal(1);
                    for(size_t j = 0; j < srcs.size() - d - 1; ++j)
                    {
                        coeffs[d] = coeffs[d] * getSize(srcs[j]);
                    }
                }

                auto delta = getDelta(srcs[0]) * coeffs[0];
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    delta = delta + getDelta(srcs[d]) * coeffs[d];
                }
                deltas.emplace(getTag(dsts[0]), delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(Join const& e)
            {
                AssertFatal(srcs.size() > 0 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                auto index = indexes[0] * getStride(srcs[0]);
                auto delta = getDelta(srcs[0]) * getStride(srcs[0]);
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    index = index + indexes[d] * getStride(srcs[d]);
                    delta = delta + getDelta(srcs[d]) * getStride(srcs[d]);
                }
                deltas.emplace(getTag(dsts[0]), delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(PassThrough const& e)
            {
                AssertFatal(srcs.size() == 1 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                auto delta = getDelta(srcs[0]);
                deltas.emplace(getTag(dsts[0]), delta);
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
                auto delta = getDelta(dsts[0]) * getStride(dsts[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    index = index + indexes[d] * getStride(dsts[d]);
                    delta = delta + getDelta(dsts[d]) * getStride(dsts[d]);
                }
                deltas.emplace(getTag(srcs[0]), delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(Tile const& e)
            {
                AssertFatal(dsts.size() > 0 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto index = indexes[0];
                auto delta = getDelta(dsts[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    index = index * getSize(dsts[d]) + indexes[d];
                    delta = delta * getSize(dsts[d]) + getDelta(dsts[d]);
                }
                deltas.emplace(getTag(srcs[0]), delta);
                return {index};
            }

            std::vector<ExpressionPtr> operator()(PassThrough const& e)
            {
                AssertFatal(dsts.size() == 1 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto delta = getDelta(dsts[0]);
                deltas.emplace(getTag(srcs[0]), delta);
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

        Transformer::Transformer(std::shared_ptr<HyperGraph> graph,
                                 std::shared_ptr<Context>    context,
                                 ExpressionTransducer        transducer)
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
            for(auto const& dimension : m_graph->getDimensions())
            {
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
                    auto& val = kernelWorkgroupIndexes.at(dimensionWorkgroup.dim);
                    AssertFatal(val,
                                "Null workgroup index",
                                ShowValue(dimensionWorkgroup.dim),
                                ShowValue(kernelWorkgroupIndexes.size()));
                    auto expr = val->expression();
                    setCoordinate(dimension, expr);
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
                    setCoordinate(dimension, expr);
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

        void Transformer::setCoordinate(Dimension const& dim, ExpressionPtr index)
        {
            rocRoller::Log::getLogger()->debug(
                "Transformer::setCoorindate: setting {} to {}", toString(dim), toString(index));
            m_dimensions.insert_or_assign(getTag(dim), dim);
            m_indexes.insert_or_assign(getTag(dim), index);
        }

        void Transformer::removeCoordinate(Dimension const& dim)
        {
            m_dimensions.erase(getTag(dim));
            m_indexes.erase(getTag(dim));
        }

        std::vector<ExpressionPtr> Transformer::forward(std::vector<Dimension> const& dsts) const
        {
            std::vector<ExpressionPtr> indexes;
            std::vector<Dimension>     srcs;
            for(auto const& kv : m_dimensions)
            {
                indexes.push_back(m_indexes.at(kv.first));
                srcs.push_back(kv.second);
            }
            return m_graph->forward(indexes, srcs, dsts, m_transducer);
        }

        std::vector<ExpressionPtr> Transformer::reverse(std::vector<Dimension> const& dsts) const
        {
            std::vector<ExpressionPtr> indexes;
            std::vector<Dimension>     srcs;
            for(auto const& kv : m_dimensions)
            {
                indexes.push_back(m_indexes.at(kv.first));
                srcs.push_back(kv.second);
            }
            return m_graph->reverse(indexes, dsts, srcs, m_transducer);
        }

        template <typename Visitor>
        std::vector<ExpressionPtr> Transformer::stride(std::vector<Dimension> const& dsts,
                                                       bool                          forward,
                                                       Visitor&                      visitor) const

        {
            std::vector<ExpressionPtr> indexes;
            std::vector<Dimension>     srcs;
            for(auto const& kv : m_dimensions)
            {
                indexes.push_back(m_indexes.at(kv.first));
                srcs.push_back(kv.second);
            }

            if(forward)
                m_graph->traverse(indexes, srcs, dsts, forward, visitor, m_transducer);
            else
                m_graph->traverse(indexes, dsts, srcs, forward, visitor, m_transducer);

            std::vector<ExpressionPtr> deltas;
            for(auto const& dst : dsts)
            {
                auto delta = visitor.deltas.at(getTag(dst));
                deltas.push_back(m_transducer ? m_transducer(delta) : delta);
            }
            return deltas;
        }

        std::vector<ExpressionPtr> Transformer::forwardStride(
            Dimension const& x, ExpressionPtr dx, std::vector<Dimension> const& dsts) const

        {
            AssertFatal(dx);
            auto visitor = ForwardEdgeDiffVisitor(x, dx);
            return stride(dsts, true, visitor);
        }

        std::vector<ExpressionPtr> Transformer::reverseStride(
            Dimension const& x, ExpressionPtr dx, std::vector<Dimension> const& dsts) const

        {
            AssertFatal(dx);
            auto visitor = ReverseEdgeDiffVisitor(x, dx);
            return stride(dsts, false, visitor);
        }

    }
}
