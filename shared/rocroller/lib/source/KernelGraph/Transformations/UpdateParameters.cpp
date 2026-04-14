// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

                AssertFatal(lhs != LayoutType::Count && rhs != LayoutType::Count,
                            "Invalid LayoutType::Count in expression");

                if(lhs == rhs)
                    return lhs;
                if(lhs == LayoutType::MATRIX_A && rhs == LayoutType::MATRIX_B)
                    return LayoutType::MATRIX_ACCUMULATOR;
                if(lhs == LayoutType::MATRIX_ACCUMULATOR || rhs == LayoutType::MATRIX_ACCUMULATOR)
                    return LayoutType::MATRIX_ACCUMULATOR;
                if(lhs == LayoutType::None && rhs != LayoutType::None)
                    return rhs;
                if(lhs != LayoutType::None && rhs == LayoutType::None)
                    return lhs;
                Throw<FatalError>("Unhandled LayoutType combination: ",
                                  ShowValue(lhs),
                                  ShowValue(rhs),
                                  ShowValue(expr));
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

        struct SetUserSizeVisitor
        {
            SetUserSizeVisitor(KernelGraph& graph, CommandParametersPtr params)
                : m_graph(graph)
                , m_params(params)
            {
            }

            template <typename T>
            Dimension visitDimension(int tag, T const& dim)
            {
                return dim;
            }

            Dimension visitDimension(int tag, MacroTile const& dim)
            {
                Log::debug("SetUserSizeVisitor: Processing MacroTile {}", tag);

                if(dim.layoutType == LayoutType::Count)
                {
                    Log::debug("SetUserSizeVisitor: Skipping MacroTile {} with invalid layout type",
                               tag);
                    return dim;
                }

                // Find User by traversing the graph structure:
                // For loads: MacroTile <- ConstructMacroTile <- SubDims <- Split <- User
                // For stores: MacroTile -> DestructMacroTile -> SubDims -> Join -> User

                // Pair of User tag and the list of subdimension tags along the path
                std::vector<std::pair<std::optional<int>, std::vector<int>>> userPaths;

                // Try load path
                auto loadSubDims
                    = m_graph.coordinates.getInputNodeIndices(tag, isEdge<ConstructMacroTile>)
                          .to<std::vector>();
                if(!loadSubDims.empty())
                {
                    auto loadUserTag = only(
                        m_graph.coordinates.getInputNodeIndices(loadSubDims[0], isEdge<Split>));
                    userPaths.push_back({loadUserTag, loadSubDims});
                    Log::debug("SetUserSizeVisitor: Found load path for MacroTile {}", tag);
                }

                // Try store path
                auto storeSubDims
                    = m_graph.coordinates.getOutputNodeIndices(tag, isEdge<DestructMacroTile>)
                          .to<std::vector>();
                if(!storeSubDims.empty())
                {
                    auto storeUserTag = only(
                        m_graph.coordinates.getOutputNodeIndices(storeSubDims[0], isEdge<Join>));
                    userPaths.push_back({storeUserTag, storeSubDims});
                    Log::debug("SetUserSizeVisitor: Found store path for MacroTile {}", tag);
                }

                if(userPaths.empty())
                {
                    Log::debug("SetUserSizeVisitor: No User found via ConstructMacroTile/Split or "
                               "DestructMacroTile/Join for MacroTile {}",
                               tag);
                    return dim;
                }

                // Process each user path
                for(const auto& [maybeUserTag, subDims] : userPaths)
                {
                    if(!maybeUserTag.has_value())
                    {
                        Log::debug("SetUserSizeVisitor: No User tag in path for MacroTile {}", tag);
                        continue;
                    }

                    auto userTag   = maybeUserTag.value();
                    auto maybeUser = m_graph.coordinates.get<User>(userTag);
                    if(!maybeUser)
                    {
                        Log::debug(
                            "SetUserSizeVisitor: Tag {} is not a User for MacroTile {} - skipping",
                            userTag,
                            tag);
                        continue;
                    }

                    // Skip if this user already has a size set
                    if(maybeUser->size)
                    {
                        Log::debug("SetUserSizeVisitor: User {} already has size, skipping",
                                   userTag);
                        continue;
                    }

                    auto hasDynamicSize = [&](int subDimTag) -> bool {
                        auto size = getSize(m_graph.coordinates.getNode(subDimTag));
                        return !rocRoller::Expression::evaluationTimes(
                            size)[rocRoller::Expression::EvaluationTime::Translate];
                    };

                    // Filter subdimensions to keep only those with dynamic (non-literal) sizes
                    std::vector<int> dynamicSubDims;
                    for(auto subDimTag : subDims)
                    {
                        if(hasDynamicSize(subDimTag))
                            dynamicSubDims.push_back(subDimTag);
                    }

                    // Current implementation assumes 2D Users (e.g., GEMM M×N, K×N, M×K)
                    // or 4D Users with two fixed size dimensions and two dynamic subdimensions (pre-tiling)
                    AssertFatal(
                        dynamicSubDims.size() == 2,
                        "SetUserSizeVisitor: Expected 2 dynamic subdimensions for MacroTile "
                        "{}, got {}",
                        tag,
                        dynamicSubDims.size());

                    // Determine which dimension has the largest stride based on memory layout
                    // Column-major (rightmost fastest): leftmost dim has largest stride
                    // Row-major (leftmost fastest): rightmost dim has largest stride
                    bool rightmostFastest  = m_params->transposeMemoryAccess[dim.layoutType];
                    int  maxStrideDimIndex = rightmostFastest ? 0 : 1;

                    auto subDim
                        = m_graph.coordinates.get<SubDimension>(dynamicSubDims[maxStrideDimIndex]);
                    AssertFatal(
                        subDim && subDim->size && subDim->stride,
                        "SubDimension must have size and stride defined for User.size calculation");

                    // User.size = maximum extent = (stride × size) of the slowest-changing dimension
                    auto user = maybeUser.value();
                    user.size = subDim->stride * subDim->size;
                    m_graph.coordinates.setElement(userTag, user);

                    Log::debug(
                        "SetUserSizeVisitor: Set User {}.size to {} based on SubDimension {} "
                        "for MacroTile {}",
                        userTag,
                        toString(user.size),
                        dynamicSubDims[maxStrideDimIndex],
                        tag);
                }

                return dim;
            }

            Dimension visitDimension(int tag, Linear const& dim)
            {
                Log::debug("SetUserSizeVisitor: Processing Linear {}", tag);

                // Find User by traversing the graph structure:
                // For loads: Linear <- Flatten <- SubDims <- Split <- User
                // For stores: Linear -> Split -> SubDims -> Join -> User

                // Pair of User tag and the list of subdimension tags along the path
                std::vector<std::pair<std::optional<int>, std::vector<int>>> userPaths;

                // Try load path
                auto loadSubDims = m_graph.coordinates.getInputNodeIndices(tag, isEdge<Flatten>)
                                       .to<std::vector>();
                if(!loadSubDims.empty())
                {
                    auto loadUserTag = only(
                        m_graph.coordinates.getInputNodeIndices(loadSubDims[0], isEdge<Split>));
                    userPaths.push_back({loadUserTag, loadSubDims});
                    Log::debug("SetUserSizeVisitor: Found load path for Linear {}", tag);
                }

                // Try store path
                auto storeSubDims = m_graph.coordinates.getOutputNodeIndices(tag, isEdge<Split>)
                                        .to<std::vector>();
                if(!storeSubDims.empty())
                {
                    auto storeUserTag = only(
                        m_graph.coordinates.getOutputNodeIndices(storeSubDims[0], isEdge<Join>));
                    userPaths.push_back({storeUserTag, storeSubDims});
                    Log::debug("SetUserSizeVisitor: Found store path for Linear {}", tag);
                }

                if(userPaths.empty())
                {
                    Log::debug("SetUserSizeVisitor: No User found via Flatten/Split or Split/Join "
                               "for Linear {}",
                               tag);
                    return dim;
                }

                // Process each user path
                for(const auto& [maybeUserTag, subDims] : userPaths)
                {
                    // This could be relaxed if needed, but the user size
                    // computation will need to be updated
                    AssertFatal(subDims.size() == 1,
                                "Expected one SubDimension in path for Linear {}",
                                tag);

                    if(!maybeUserTag.has_value())
                    {
                        Log::debug("SetUserSizeVisitor: No User tag in path for Linear {}", tag);
                        continue;
                    }

                    auto userTag   = maybeUserTag.value();
                    auto maybeUser = m_graph.coordinates.get<User>(userTag);
                    if(!maybeUser)
                    {
                        Log::debug(
                            "SetUserSizeVisitor: Tag {} is not a User for Linear {} - skipping",
                            userTag,
                            tag);
                        continue;
                    }

                    // Skip if this user already has a size set
                    if(maybeUser->size)
                    {
                        Log::debug("SetUserSizeVisitor: User {} already has size, skipping",
                                   userTag);
                        continue;
                    }

                    auto subDim = m_graph.coordinates.get<SubDimension>(subDims[0]);
                    AssertFatal(subDim && subDim->size && subDim->stride,
                                "SubDimension must have size and stride defined for User.size");

                    auto user = maybeUser.value();
                    user.size = subDim->stride * subDim->size;
                    ;
                    m_graph.coordinates.setElement(userTag, user);

                    Log::debug("SetUserSizeVisitor: Set User {}.size to {} for Linear {}",
                               userTag,
                               toString(user.size),
                               tag);
                }

                return dim;
            }

            template <typename T>
            Operation visitOperation(int tag, T const& op)
            {
                return op;
            }

        private:
            KernelGraph&         m_graph;
            CommandParametersPtr m_params;
        };

        KernelGraph UpdateParameters::apply(KernelGraph const& original)
        {
            if(!m_params)
                return original;

            auto updateVisitor = UpdateParametersVisitor(m_params);
            auto kgraph        = rewriteDimensions(original, updateVisitor);

            // This visitor modifies the coordinate graph while
            // rewriteDimensions walks the control graph.
            auto infoVisitor = PropagateTileInfoVisitor(kgraph);
            rewriteDimensions(kgraph, infoVisitor);

            // Set User.size for tensor contraction inputs
            auto userSizeVisitor = SetUserSizeVisitor(kgraph, m_params);
            rewriteDimensions(kgraph, userSizeVisitor);

            return kgraph;
        }

        KernelGraph UpdateWavefrontParameters::apply(KernelGraph const& original)
        {
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
