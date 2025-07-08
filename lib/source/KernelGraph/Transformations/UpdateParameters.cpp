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

/*
 * Update parameters and propagate tile information.
 */

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Transforms/UpdateParameters.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;
        using namespace Expression;

        using Size = std::vector<int>;

        struct SizeVisitor
        {
            KernelGraph& graph;

            SizeVisitor(KernelGraph& graph)
                : graph(graph)
            {
            }

            template <CBinary BinaryExp>
            Size operator()(BinaryExp const& expr)
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);
                if(lhs != rhs)
                {
                    if(lhs.size() == 1 && lhs.at(0) == 1)
                        return rhs;
                    if(rhs.size() == 1 && rhs.at(0) == 1)
                        return lhs;
                    if(m_subTiles)
                        return {};
                    Throw<FatalError>("Expression size mismatch: ", ShowValue(expr));
                }
                return lhs;
            }

            template <CUnary UnaryExp>
            Size operator()(UnaryExp const& expr)
            {
                return call(expr.arg);
            }

            Size operator()(DataFlowTag const& expr)
            {
                auto maybeTile = graph.coordinates.get<MacroTile>(expr.tag);
                if(maybeTile)
                    return m_subTiles ? maybeTile->subTileSizes : maybeTile->sizes;
                auto maybeVGPR = graph.coordinates.get<VGPR>(expr.tag);
                if(maybeVGPR)
                    // XXX Scalar
                    return {1};
                return {};
            }

            Size operator()(CommandArgumentValue const& expr)
            {
                return {1};
            }

            Size operator()(auto const& expr)
            {
                Throw<FatalError>("SizeVisitor not implemented yet.");
            }

            Size call(rocRoller::Expression::Expression const& expr)
            {
                return std::visit(*this, expr);
            }

            Size call(ExpressionPtr expr)
            {
                return call(*expr);
            }

            Size call(ExpressionPtr expr, bool subTiles)
            {
                m_subTiles = subTiles;
                return call(*expr);
            }

        private:
            bool m_subTiles = false;
        };

        struct MemoryTypeVisitor
        {
            KernelGraph& graph;

            MemoryTypeVisitor(KernelGraph& graph)
                : graph(graph)
            {
            }

            MemoryType dropLDS(MemoryType const& type) const
            {
                if(type == MemoryType::WAVE_LDS)
                    return MemoryType::WAVE;
                if(type == MemoryType::LDS)
                    return MemoryType::VGPR;
                return type;
            }

            template <CBinary BinaryExp>
            MemoryType operator()(BinaryExp const& expr)
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);
                if(lhs == rhs)
                    return dropLDS(lhs);
                if(lhs == MemoryType::WAVE || rhs == MemoryType::WAVE)
                    return MemoryType::WAVE;
                if(lhs == MemoryType::VGPR || lhs == MemoryType::AGPR || lhs == MemoryType::Literal)
                    return rhs;
                if(rhs == MemoryType::VGPR || rhs == MemoryType::AGPR || rhs == MemoryType::Literal)
                    return lhs;
                Throw<FatalError>(
                    "Unhandled MemoryType combination: ", ShowValue(lhs), ShowValue(rhs));
            }

            template <CUnary UnaryExp>
            MemoryType operator()(UnaryExp const& expr)
            {
                return dropLDS(call(expr.arg));
            }

            MemoryType operator()(DataFlowTag const& expr)
            {
                auto dim = graph.coordinates.getNode(expr.tag);
                return std::visit(rocRoller::overloaded{
                                      [](MacroTile const& x) -> MemoryType { return x.memoryType; },
                                      [](VGPR const&) -> MemoryType { return MemoryType::VGPR; },
                                      [](auto const&) -> MemoryType { return MemoryType::None; }},
                                  dim);
            }

            MemoryType operator()(CommandArgumentValue const& expr)
            {
                return MemoryType::Literal;
            }

            MemoryType operator()(auto const& expr)
            {
                Throw<FatalError>("MemoryTypeVisitor not implemented yet.");
            }

            MemoryType call(rocRoller::Expression::Expression const& expr)
            {
                return std::visit(*this, expr);
            }

            MemoryType call(ExpressionPtr expr)
            {
                return call(*expr);
            }
        };

        struct LayoutTypeVisitor
        {
            KernelGraph& graph;

            LayoutTypeVisitor(KernelGraph& graph)
                : graph(graph)
            {
            }

            template <CBinary BinaryExp>
            LayoutType operator()(BinaryExp const& expr)
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);
                if(lhs == rhs)
                    return lhs;
                if(lhs == LayoutType::MATRIX_A && rhs == LayoutType::MATRIX_B)
                    return LayoutType::MATRIX_ACCUMULATOR;
                if(lhs == LayoutType::MATRIX_ACCUMULATOR || rhs == LayoutType::MATRIX_ACCUMULATOR)
                    return LayoutType::MATRIX_ACCUMULATOR;
                Throw<FatalError>(
                    "Unhandled LayoutType combination: ", ShowValue(lhs), ShowValue(rhs));
            }

            template <CUnary UnaryExp>
            LayoutType operator()(UnaryExp const& expr)
            {
                return call(expr.arg);
            }

            LayoutType operator()(DataFlowTag const& expr)
            {
                auto dim = graph.coordinates.getNode(expr.tag);
                return std::visit(rocRoller::overloaded{
                                      [](MacroTile const& x) -> LayoutType { return x.layoutType; },
                                      [](auto const&) -> LayoutType { return LayoutType::None; }},
                                  dim);
            }

            LayoutType operator()(CommandArgumentValue const& expr)
            {
                return LayoutType::None;
            }

            LayoutType operator()(auto const& expr)
            {
                Throw<FatalError>("LayoutTypeVisitor not implemented yet.");
            }

            LayoutType call(rocRoller::Expression::Expression const& expr)
            {
                return std::visit(*this, expr);
            }

            LayoutType call(ExpressionPtr expr)
            {
                return call(*expr);
            }
        };

        class PropagateTileInfoVisitor
        {
            KernelGraph&      m_graph;
            SizeVisitor       m_sizeVisitor;
            MemoryTypeVisitor m_memoryTypeVisitor;
            LayoutTypeVisitor m_layoutTypeVisitor;

        public:
            PropagateTileInfoVisitor() = delete;
            PropagateTileInfoVisitor(KernelGraph& graph)
                : m_graph(graph)
                , m_sizeVisitor(graph)
                , m_memoryTypeVisitor(graph)
                , m_layoutTypeVisitor(graph)
            {
            }

            template <typename T>
            Dimension visitDimension(int tag, T const& dim)
            {
                return dim;
            }

            template <typename T>
            Operation visitOperation(int tag, T const& op)
            {
                return op;
            }

            Operation visitOperation(int tag, TensorContraction const& op)
            {
                auto lhsTag = m_graph.mapper.get(tag, NaryArgument::LHS);
                auto rhsTag = m_graph.mapper.get(tag, NaryArgument::RHS);
                auto dstTag = m_graph.mapper.get(tag, NaryArgument::DEST);

                auto lhs = *m_graph.coordinates.get<MacroTile>(lhsTag);
                auto rhs = *m_graph.coordinates.get<MacroTile>(rhsTag);
                auto dst = *m_graph.coordinates.get<MacroTile>(dstTag);

                if(dst.sizes.empty())
                {
                    auto ntile         = dst;
                    ntile.rank         = 2;
                    ntile.sizes        = {lhs.sizes[0], rhs.sizes[1]};
                    ntile.subTileSizes = lhs.subTileSizes;
                    ntile.memoryType   = MemoryType::WAVE;
                    ntile.layoutType   = LayoutType::MATRIX_ACCUMULATOR;
                    m_graph.coordinates.setElement(dstTag, ntile);
                }

                return op;
            }

            Operation visitOperation(int tag, Assign const& op)
            {
                auto dst       = m_graph.mapper.get(tag, NaryArgument::DEST);
                auto maybeTile = m_graph.coordinates.get<MacroTile>(dst);

                if(maybeTile)
                {
                    auto tile = *maybeTile;

                    if(tile.sizes.empty())
                    {
                        auto ntile         = tile;
                        ntile.sizes        = m_sizeVisitor.call(op.expression, false);
                        ntile.subTileSizes = m_sizeVisitor.call(op.expression, true);
                        ntile.memoryType   = m_memoryTypeVisitor.call(op.expression);
                        ntile.layoutType   = m_layoutTypeVisitor.call(op.expression);
                        ntile.rank         = ntile.sizes.size();
                        m_graph.coordinates.setElement(dst, ntile);
                    }
                }

                return op;
            }
        };

        struct UpdateParametersVisitor
        {
            UpdateParametersVisitor(CommandParametersPtr params)
            {
                m_newDimensions = params->getDimensionInfo();
            }

            template <typename T>
            Dimension visitDimension(int tag, T const& dim)
            {
                if(m_newDimensions.count(dim.commandTag) > 0)
                {
                    if(name(m_newDimensions.at(dim.commandTag)) == name(dim))
                    {
                        auto newDim = m_newDimensions.at(dim.commandTag);
                        setCommandTag(newDim, dim.commandTag);
                        return newDim;
                    }
                }
                return dim;
            }

            template <typename T>
            Operation visitOperation(int tag, T const& op)
            {
                return op;
            }

        private:
            std::map<Operations::OperationTag, Dimension> m_newDimensions;
        };

        KernelGraph UpdateParameters::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::UpdateParameters");

            if(!m_params)
                return original;

            auto updateVisitor = UpdateParametersVisitor(m_params);
            auto kgraph        = rewriteDimensions(original, updateVisitor);

            // This visitor modifies the coordinate graph while
            // rewriteDimensions walks the control graph.
            auto infoVisitor = PropagateTileInfoVisitor(kgraph);
            rewriteDimensions(kgraph, infoVisitor);

            return kgraph;
        }

        KernelGraph UpdateWavefrontParameters::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::UpdateWavefrontParameters");

            if(!m_params)
                return original;

            auto kgraph = original;
            auto counts = m_params->getManualWavefrontCounts();
            if(counts)
            {
                auto wfx = std::get<0>(*counts);
                auto wfy = std::get<1>(*counts);
                auto WF  = Wavefront(-1, literal(wfx * wfy), nullptr);
                auto WFX = Wavefront(0, literal(wfx), nullptr);
                auto WFY = Wavefront(1, literal(wfy), nullptr);
                for(auto tag : kgraph.coordinates.getNodes<Wavefront>())
                {
                    auto wavefront = *kgraph.coordinates.get<Wavefront>(tag);
                    if(wavefront.dim == -1)
                        kgraph.coordinates.setElement(tag, WF);
                    if(wavefront.dim == 0)
                        kgraph.coordinates.setElement(tag, WFX);
                    if(wavefront.dim == 1)
                        kgraph.coordinates.setElement(tag, WFY);
                }
            }

            return kgraph;
        }

        KernelGraph SetWorkitemCount::apply(KernelGraph const& original)
        {
            auto workgroupSize = m_context->kernel()->workgroupSize();
            auto workitemCount = std::array<ExpressionPtr, 3>{nullptr, nullptr, nullptr};
            auto workgroupTags = original.coordinates.getNodes<Workgroup>().to<std::vector>();
            for(auto const& workgroupTag : workgroupTags)
            {
                auto workgroup = *original.coordinates.get<Workgroup>(workgroupTag);
                if(workgroup.size)
                {
                    // TODO: For linear things this isn't quite right;
                    // we need to set according to the size of the
                    // incoming tensor.
                    workitemCount[workgroup.dim]
                        = workgroup.size * literal(workgroupSize[workgroup.dim]);
                    Log::debug("Setting Workitem count to {} based on size of Workgroup {}",
                               toString(workgroup.size),
                               workgroupTag);
                }
            }
            m_context->kernel()->setWorkitemCount(workitemCount);
            return original;
        }
    }
}
