
#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;
    namespace Expression = rocRoller::Expression;
    using namespace Expression;

    using GD = Graph::Direction;

    struct ComputeIndexChain
    {
        int top, bottom;
        int update = -1;
    };

    /*
     * Helpers
     */

    /**
     * @brief Return existing Buffer edge between src and dst, or
     * create a new one.
     */
    int getBuffer(KernelGraph& graph, int src, int dst)
    {
        for(auto neighbour : graph.coordinates.getNeighbours<GD::Upstream>(dst).to<std::vector>())
        {
            if(graph.coordinates.get<Buffer>(neighbour))
            {
                return neighbour;
            }
        }
        return graph.coordinates.addElement(Buffer(), {src}, {dst});
    }

    /**
     * @brief Add a ComputeIndex node and add mapper connections.
     */
    int makeComputeIndex(KernelGraph&     graph,
                         int              target,
                         int              increment,
                         int              base,
                         int              offset,
                         int              stride,
                         int              buffer,
                         bool             forward,
                         DataType         valueType,
                         std::vector<int> zero       = {},
                         DataType         offsetType = DataType::UInt64,
                         DataType         strideType = DataType::UInt64)
    {
        using CCI = Connections::ComputeIndex;
        using CCA = Connections::ComputeIndexArgument;

        auto ci
            = graph.control.addElement(ComputeIndex(forward, valueType, offsetType, strideType));

        if(base > 0)
            graph.mapper.connect(ci, base, CCI{CCA::BASE});
        if(buffer > 0)
            graph.mapper.connect(ci, buffer, CCI{CCA::BUFFER});
        if(increment > 0)
            graph.mapper.connect(ci, increment, CCI{CCA::INCREMENT});
        if(offset > 0)
            graph.mapper.connect(ci, offset, CCI{CCA::OFFSET});
        if(stride > 0)
            graph.mapper.connect(ci, stride, CCI{CCA::STRIDE});
        if(target > 0)
            graph.mapper.connect(ci, target, CCI{CCA::TARGET});
        for(int i = 0; i < zero.size(); ++i)
            graph.mapper.connect(ci, zero[i], CCI{CCA::ZERO, i});

        rocRoller::Log::getLogger()->debug(
            "KernelGraph::makeComputeIndex: ci {} {}/{} {}; {}/{}/{}",
            ci,
            target,
            increment,
            forward,
            base,
            offset,
            stride);

        return ci;
    }

    /*
     * Helpers for building ComputeIndex chains for specific layouts.
     */

    /**
     * @brief Add ComputeIndexes for VGPR MATRIX_A/B from global.
     */
    ComputeIndexChain
        computeIndexVGPRMATRIXAB(KernelGraph& graph, int load, int sdim, ExpressionPtr step)
    {
        AssertFatal(isOperation<LoadTiled>(graph.control.getElement(load)));

        auto user   = graph.mapper.get<User>(load);
        auto mac    = graph.mapper.get<MacroTileNumber>(load, sdim);
        auto elem_x = graph.mapper.get<ElementNumber>(load, 0);
        auto elem_y = graph.mapper.get<ElementNumber>(load, 1);

        auto dtype = graph.control.get<LoadTiled>(load)->vtype.dataType;

        auto offset_mac = graph.coordinates.addElement(Offset(), {user}, {mac});
        auto stride_mac = graph.coordinates.addElement(Stride(), {user}, {mac});
        auto row_offset = graph.coordinates.addElement(Offset(), {user}, {elem_x});
        auto row_stride = graph.coordinates.addElement(Stride(), {user}, {elem_x});
        auto col_offset = graph.coordinates.addElement(Offset(), {user}, {elem_y});
        auto col_stride = graph.coordinates.addElement(Stride(), {user}, {elem_y});
        auto buffer     = getBuffer(graph, user, mac);

        graph.mapper.connect<Offset>(load, offset_mac, -1);
        graph.mapper.connect<Offset>(load, row_offset, 0);
        graph.mapper.connect<Offset>(load, col_offset, 1);
        graph.mapper.connect<Stride>(load, stride_mac, -1);
        graph.mapper.connect<Stride>(load, row_stride, 0);
        graph.mapper.connect<Stride>(load, col_stride, 1);
        graph.mapper.connect<Buffer>(load, buffer);

        auto ci_mac = makeComputeIndex(
            graph, user, mac, -1, offset_mac, stride_mac, buffer, false, dtype, {elem_x, elem_y});
        auto ci_row = makeComputeIndex(graph,
                                       user,
                                       elem_x,
                                       offset_mac,
                                       row_offset,
                                       row_stride,
                                       buffer,
                                       false,
                                       dtype,
                                       {mac, elem_y});
        auto ci_col = makeComputeIndex(graph,
                                       user,
                                       elem_y,
                                       row_offset,
                                       col_offset,
                                       col_stride,
                                       buffer,
                                       false,
                                       dtype,
                                       {mac, elem_x});

        graph.control.addElement(Sequence(), {ci_mac}, {ci_row});
        graph.control.addElement(Sequence(), {ci_row}, {ci_col});

        auto offset_mac_expr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{offset_mac, Register::Type::Vector, DataType::UInt64});
        auto stride_mac_expr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{stride_mac, Register::Type::Scalar, DataType::UInt64});

        auto offsetUpdate = graph.control.addElement(
            Assign{Register::Type::Vector, offset_mac_expr + step * stride_mac_expr});
        graph.mapper.connect(offsetUpdate, offset_mac, NaryArgument::DEST);

        return {ci_mac, ci_col, offsetUpdate};
    }

    /**
     * @brief Add ComputeIndexes for generic VGPR MATRIX from global.
     */
    ComputeIndexChain computeIndexVGPR(KernelGraph& graph, int loadstore, int source, bool forward)
    {
        auto elem_x = graph.mapper.get<ElementNumber>(loadstore, 0);
        auto elem_y = graph.mapper.get<ElementNumber>(loadstore, 1);

        DataType dtype, offsettype = DataType::UInt64;
        {
            auto l  = graph.control.get<LoadTiled>(loadstore);
            auto ll = graph.control.get<LoadLDSTile>(loadstore);
            auto s  = graph.control.get<StoreTiled>(loadstore);
            auto sl = graph.control.get<StoreLDSTile>(loadstore);
            if(l)
                dtype = l->vtype.dataType;
            if(ll)
            {
                dtype      = ll->vtype.dataType;
                offsettype = DataType::UInt32;
            }
            if(s)
                dtype = s->dataType;
            if(sl)
            {
                dtype      = sl->dataType;
                offsettype = DataType::UInt32;
            }
        }

        int row_offset, row_stride, col_offset, col_stride, buffer;
        if(forward)
        {
            row_offset = graph.coordinates.addElement(Offset(), {elem_x}, {source});
            row_stride = graph.coordinates.addElement(Stride(), {elem_x}, {source});
            col_offset = graph.coordinates.addElement(Offset(), {elem_y}, {source});
            col_stride = graph.coordinates.addElement(Stride(), {elem_y}, {source});
            buffer     = getBuffer(graph, elem_x, source);
        }
        else
        {
            row_offset = graph.coordinates.addElement(Offset(), {source}, {elem_x});
            row_stride = graph.coordinates.addElement(Stride(), {source}, {elem_x});
            col_offset = graph.coordinates.addElement(Offset(), {source}, {elem_y});
            col_stride = graph.coordinates.addElement(Stride(), {source}, {elem_y});
            buffer     = getBuffer(graph, source, elem_x);
        }

        graph.mapper.connect<Offset>(loadstore, row_offset, 0);
        graph.mapper.connect<Offset>(loadstore, col_offset, 1);
        graph.mapper.connect<Stride>(loadstore, row_stride, 0);
        graph.mapper.connect<Stride>(loadstore, col_stride, 1);
        graph.mapper.connect<Buffer>(loadstore, buffer);

        auto ci_row = makeComputeIndex(graph,
                                       source,
                                       elem_x,
                                       -1,
                                       row_offset,
                                       row_stride,
                                       buffer,
                                       forward,
                                       dtype,
                                       {elem_y},
                                       offsettype,
                                       offsettype);
        auto ci_col = makeComputeIndex(graph,
                                       source,
                                       elem_y,
                                       row_offset,
                                       col_offset,
                                       col_stride,
                                       buffer,
                                       forward,
                                       dtype,
                                       {elem_x},
                                       offsettype,
                                       offsettype);

        graph.control.addElement(Sequence(), {ci_row}, {ci_col});

        return {ci_row, ci_col};
    }

    /**
     * @brief Add ComputeIndexes for WAVE MATRIX_A/B from LDS.
     */
    ComputeIndexChain computeIndexLDSWaveAB(KernelGraph& graph, int load, int sdim)
    {
        AssertFatal(isOperation<LoadLDSTile>(graph.control.getElement(load)));
        auto lds  = graph.mapper.get<LDS>(load);
        auto wave = graph.mapper.get<WaveTileNumber>(load, sdim);
        auto vgpr = graph.mapper.get<VGPR>(load);

        auto dtype = graph.control.get<LoadLDSTile>(load)->vtype.dataType;

        auto offset_wave = graph.coordinates.addElement(Offset(), {lds}, {wave});
        auto stride_wave = graph.coordinates.addElement(Stride(), {lds}, {wave});
        auto offset_vgpr = graph.coordinates.addElement(Offset(), {lds}, {vgpr});
        auto stride_vgpr = graph.coordinates.addElement(Stride(), {lds}, {vgpr});

        graph.mapper.connect<Offset>(load, offset_wave, 0);
        graph.mapper.connect<Offset>(load, offset_vgpr, 1);
        graph.mapper.connect<Stride>(load, stride_wave, 0);
        graph.mapper.connect<Stride>(load, stride_vgpr, 1);

        auto ci_wave = makeComputeIndex(graph,
                                        lds,
                                        wave,
                                        -1,
                                        offset_wave,
                                        stride_wave,
                                        -1,
                                        false,
                                        dtype,
                                        {vgpr},
                                        DataType::UInt32,
                                        DataType::UInt32);
        auto ci_vgpr = makeComputeIndex(graph,
                                        lds,
                                        vgpr,
                                        offset_wave,
                                        offset_vgpr,
                                        stride_vgpr,
                                        -1,
                                        false,
                                        dtype,
                                        {wave},
                                        DataType::UInt32,
                                        DataType::UInt32);

        graph.control.addElement(Sequence(), {ci_wave}, {ci_vgpr});

        return {ci_wave, ci_vgpr};
    }

    /**
     * @brief Add ComputeIndexes for WAVE MATRIX_A/B from global.
     */
    ComputeIndexChain
        computeIndexWAVEMATRIXAB(KernelGraph& graph, int load, int sdim, ExpressionPtr step)
    {
        auto user = graph.mapper.get<User>(load);
        auto mac  = graph.mapper.get<MacroTileNumber>(load, sdim);
        auto wave = graph.mapper.get<WaveTileNumber>(load, sdim);
        auto vgpr = graph.mapper.get<VGPR>(load);

        auto dtype = graph.control.get<LoadTiled>(load)->vtype.dataType;

        auto offset_mac  = graph.coordinates.addElement(Offset(), {user}, {mac});
        auto stride_mac  = graph.coordinates.addElement(Stride(), {user}, {mac});
        auto offset_wave = graph.coordinates.addElement(Offset(), {user}, {wave});
        auto stride_wave = graph.coordinates.addElement(Stride(), {user}, {wave});
        auto offset_vgpr = graph.coordinates.addElement(Offset(), {user}, {vgpr});
        auto stride_vgpr = graph.coordinates.addElement(Stride(), {user}, {vgpr});
        auto buffer      = getBuffer(graph, user, mac);

        graph.mapper.connect<Offset>(load, offset_mac, -1);
        graph.mapper.connect<Offset>(load, offset_wave, 0);
        graph.mapper.connect<Offset>(load, offset_vgpr, 1);
        graph.mapper.connect<Stride>(load, stride_mac, -1);
        graph.mapper.connect<Stride>(load, stride_wave, 0);
        graph.mapper.connect<Stride>(load, stride_vgpr, 1);
        graph.mapper.connect<Buffer>(load, buffer);

        auto ci_mac = makeComputeIndex(
            graph, user, mac, -1, offset_mac, stride_mac, buffer, false, dtype, {wave, vgpr});
        auto ci_wave = makeComputeIndex(graph,
                                        user,
                                        wave,
                                        offset_mac,
                                        offset_wave,
                                        stride_wave,
                                        buffer,
                                        false,
                                        dtype,
                                        {mac, vgpr});
        auto ci_vgpr = makeComputeIndex(graph,
                                        user,
                                        vgpr,
                                        offset_wave,
                                        offset_vgpr,
                                        stride_vgpr,
                                        buffer,
                                        false,
                                        dtype,
                                        {mac, wave});

        graph.control.addElement(Sequence(), {ci_mac}, {ci_wave});
        graph.control.addElement(Sequence(), {ci_wave}, {ci_vgpr});

        auto offset_mac_expr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{offset_mac, Register::Type::Vector, DataType::UInt64});
        auto stride_mac_expr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{stride_mac, Register::Type::Scalar, DataType::UInt64});

        auto offsetUpdate = graph.control.addElement(
            Assign{Register::Type::Vector, offset_mac_expr + step * stride_mac_expr});
        graph.mapper.connect(offsetUpdate, offset_mac, NaryArgument::DEST);

        return {ci_mac, ci_vgpr, offsetUpdate};
    }

    /**
     * @brief Add ComputeIndexes for VGPR MATRIX_ACCUMULATOR from global or LDS.
     */
    ComputeIndexChain addComputeIndexC(KernelGraph& graph, int op, bool forward)
    {
        rocRoller::Log::getLogger()->debug("KernelGraph::addComputeIndexC({}, {})", op, forward);

        auto [source, _d] = getOperationTarget(op, graph);
        AssertFatal(source > 0, "User or LDS dimension not found");

        auto vgpr_block = graph.mapper.get<VGPRBlockNumber>(op);
        auto vgpr_index = graph.mapper.get<VGPRBlockIndex>(op);

        DataType dtype, offsettype = DataType::UInt64;
        {
            auto l  = graph.control.get<LoadTiled>(op);
            auto ll = graph.control.get<LoadLDSTile>(op);
            auto s  = graph.control.get<StoreTiled>(op);
            auto sl = graph.control.get<StoreLDSTile>(op);
            if(l)
                dtype = l->vtype.dataType;
            if(ll)
            {
                dtype      = ll->vtype.dataType;
                offsettype = DataType::UInt32;
            }
            if(s)
                dtype = s->dataType;
            if(sl)
            {
                dtype      = sl->dataType;
                offsettype = DataType::UInt32;
            }
        }

        int offset_vgpr_block, stride_vgpr_block, offset_vgpr_index, stride_vgpr_index, buffer;
        if(forward)
        {
            offset_vgpr_block = graph.coordinates.addElement(Offset(), {vgpr_block}, {source});
            stride_vgpr_block = graph.coordinates.addElement(Stride(), {vgpr_block}, {source});
            offset_vgpr_index = graph.coordinates.addElement(Offset(), {vgpr_index}, {source});
            stride_vgpr_index = graph.coordinates.addElement(Stride(), {vgpr_index}, {source});
            buffer            = getBuffer(graph, vgpr_index, source);
        }
        else
        {
            offset_vgpr_block = graph.coordinates.addElement(Offset(), {source}, {vgpr_block});
            stride_vgpr_block = graph.coordinates.addElement(Stride(), {source}, {vgpr_block});
            offset_vgpr_index = graph.coordinates.addElement(Offset(), {source}, {vgpr_index});
            stride_vgpr_index = graph.coordinates.addElement(Stride(), {source}, {vgpr_index});
            buffer            = getBuffer(graph, source, vgpr_index);
        }

        graph.mapper.connect<Offset>(op, offset_vgpr_block, 0);
        graph.mapper.connect<Offset>(op, offset_vgpr_index, 1);
        graph.mapper.connect<Stride>(op, stride_vgpr_block, 0);
        graph.mapper.connect<Stride>(op, stride_vgpr_index, 1);
        graph.mapper.connect<Buffer>(op, buffer);

        auto ci_vgpr_block = makeComputeIndex(graph,
                                              source,
                                              vgpr_block,
                                              -1,
                                              offset_vgpr_block,
                                              stride_vgpr_block,
                                              buffer,
                                              forward,
                                              dtype,
                                              {vgpr_index},
                                              offsettype,
                                              offsettype);
        auto ci_vgpr_index = makeComputeIndex(graph,
                                              source,
                                              vgpr_index,
                                              offset_vgpr_block,
                                              offset_vgpr_index,
                                              stride_vgpr_index,
                                              buffer,
                                              forward,
                                              dtype,
                                              {vgpr_block},
                                              offsettype,
                                              offsettype);

        graph.control.addElement(Sequence(), {ci_vgpr_block}, {ci_vgpr_index});

        return {ci_vgpr_block, ci_vgpr_index};
    }

    bool needsComputeIndex(Operation const& op)
    {
        if(std::holds_alternative<StoreTiled>(op) || std::holds_alternative<StoreLDSTile>(op)
           || std::holds_alternative<LoadTiled>(op) || std::holds_alternative<LoadLDSTile>(op))
            return true;
        return false;
    }

    /**
     * @brief Find load/store operations that need their indexes
     * precomputed by ComputeIndex.
     */
    std::vector<int> findComputeIndexCandidates(KernelGraph const& kgraph)
    {
        std::vector<int> rv;

        auto kernel = *kgraph.control.roots().begin();
        return kgraph.control
            .findNodes(
                kernel,
                [&](int tag) -> bool {
                    auto elem = kgraph.control.getElement(tag);
                    if(!std::holds_alternative<Operation>(elem))
                        return false;
                    auto op = std::get<Operation>(elem);
                    return needsComputeIndex(op);
                },
                GD::Downstream)
            .to<std::vector>();
    }

    /**
     * @brief Get required Unroll coordinate values.
     *
     * For each Unroll coordinate required by the candidate operation,
     * query the Control Flow graph (upstream) for SetCoordinate
     * operations and record the Unroll coordinate values.
     *
     * @param candidate Candidate operation in the Control Flow graph:
     * look upstream from here.
     * @param unrollCoordinates Required Unroll coordinates.
     * @param kgraph Kernel Graph.
     * @return Map from Unroll coordinate to set value (ExpressionPtr).
     */
    std::map<int, ExpressionPtr> getUnrollCoordinateValues(
        int candidate, std::unordered_set<int> unrollCoordinates, KernelGraph const& kgraph)
    {
        auto burnDown = unrollCoordinates;

        std::map<int, ExpressionPtr> rv;
        for(auto tag : kgraph.control.depthFirstVisit(candidate, GD::Upstream))
        {
            auto maybeSetCoordinate = kgraph.control.get<SetCoordinate>(tag);
            if(!maybeSetCoordinate)
                continue;

            auto coordinate = kgraph.mapper.get<Unroll>(tag);

            if(burnDown.contains(coordinate))
            {
                rv[coordinate] = maybeSetCoordinate->value;
                burnDown.erase(coordinate);
            }

            if(burnDown.empty())
                return rv;
        }

        return rv;
    }

    /**
     * @brief Find earliest possible set of SetCoordinate operations
     * that set Unroll coordinates to specific values.
     *
     * From the root (Kernel) of the Control Flow graph, follow the
     * path to the candidate operation.  When a SetCoordinate node is
     * visited, check to see if it is setting one of the Unroll
     * dimensions of interest to the exact value required.  When the
     * list of required Unroll values is exhausted, return the current
     * SetCoordinate operation.
     *
     * @param candidate Candidate load/store operation.
     * @param values Map of Unroll dimension tag to required Unroll value.
     * @param kgraph Kernel Graph
     */
    std::optional<int> findEarliestMatchingSetCoordinate(int                           candidate,
                                                         std::map<int, ExpressionPtr>& values,
                                                         KernelGraph const&            kgraph)
    {

        std::unordered_set<int> burnDown;
        for(auto const& kv : values)
            burnDown.insert(kv.first);

        auto kernel = *kgraph.control.roots().begin();
        auto path   = kgraph.control
                        .path<GD::Downstream>(std::vector<int>{kernel}, std::vector<int>{candidate})
                        .to<std::vector>();

        for(auto tag : path)
        {
            auto maybeSetCoordinate = kgraph.control.get<SetCoordinate>(tag);
            if(!maybeSetCoordinate)
                continue;

            auto coordinate = kgraph.mapper.get<Unroll>(tag);

            if(burnDown.contains(coordinate))
            {
                if(Expression::identical(maybeSetCoordinate->value, values[coordinate]))
                {
                    burnDown.erase(coordinate);
                }
            }

            if(burnDown.empty())
                return tag;
        }

        return {};
    }

    /**
     * @brief Generic routine to create a ComputeIndex chain for a
     * load/store operation.
     *
     * @param kgraph Kernel graph to add ComputeIndex operations to.
     * @param tag Load/store operation that needs ComputeIndex operations.
     */
    ComputeIndexChain addComputeIndex(KernelGraph& kgraph, int tag, ExpressionPtr step)
    {
        auto log = rocRoller::Log::getLogger();

        auto store    = kgraph.control.get<StoreTiled>(tag);
        auto storeLDS = kgraph.control.get<StoreLDSTile>(tag);
        if(store || storeLDS)
        {
            auto [tile_tag, tile] = kgraph.getDimension<MacroTile>(tag);
            if(tile.memoryType == MemoryType::VGPR || tile.memoryType == MemoryType::LDS)
            {
                auto [source, _d] = getOperationTarget(tag, kgraph);
                return computeIndexVGPR(kgraph, tag, source, true);
            }
            if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR
               && tile.memoryType == MemoryType::WAVE)
            {
                return addComputeIndexC(kgraph, tag, true);
            }
        }

        auto load = kgraph.control.get<LoadTiled>(tag);
        if(load)
        {
            auto [tile_tag, tile] = kgraph.getDimension<MacroTile>(tag);
            if(tile.memoryType == MemoryType::VGPR
               && (tile.layoutType == LayoutType::MATRIX_A
                   || tile.layoutType == LayoutType::MATRIX_B))
            {
                int sdim = tile.layoutType == LayoutType::MATRIX_A ? 1 : 0;
                return computeIndexVGPRMATRIXAB(kgraph, tag, sdim, step);
            }
            if(tile.memoryType == MemoryType::VGPR)
            {
                auto [source, _d] = getOperationTarget(tag, kgraph);
                return computeIndexVGPR(kgraph, tag, source, false);
            }
            if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
            {
                return addComputeIndexC(kgraph, tag, false);
            }
            if(tile.layoutType == LayoutType::MATRIX_A || tile.layoutType == LayoutType::MATRIX_B)
            {
                int sdim = tile.layoutType == LayoutType::MATRIX_A ? 1 : 0;
                return computeIndexWAVEMATRIXAB(kgraph, tag, sdim, step);
            }
        }

        auto loadLDS = kgraph.control.get<LoadLDSTile>(tag);
        if(loadLDS)
        {
            auto [tile_tag, tile] = kgraph.getDimension<MacroTile>(tag);
            if(tile.memoryType == MemoryType::VGPR || tile.memoryType == MemoryType::LDS)
            {
                auto [source, _d] = getOperationTarget(tag, kgraph);
                return computeIndexVGPR(kgraph, tag, source, false);
            }

            if(tile.layoutType == LayoutType::MATRIX_A || tile.layoutType == LayoutType::MATRIX_B)
            {
                int sdim = tile.layoutType == LayoutType::MATRIX_A ? 1 : 0;
                return computeIndexLDSWaveAB(kgraph, tag, sdim);
            }
        }

        Throw<FatalError>("Not implemented yet.");
    }

    /**
     * @brief Add ComputeIndex nodes in the right places.
     *
     * Called on all load/store operations in the graph.
     *
     * For each candidate load/store operation:
     *
     * 1. ComputeIndex operations (chain) are created.
     *
     * 2. Find all required coordinates by querying the Coordinate
     *    Transform graph.
     *
     * 3. If one-or-more Unroll dimension(s) are required:
     *
     *    a. Find SetCoordinate operations above the candidate and
     *       record the values of required Unroll dimensions.
     *
     *    b. Find the earliest matching set of SetCoordinate
     *       operations that are identical (ie, Unroll dimension and
     *       value) to the required Unroll dimensions.
     *
     *    c. The chain is added below the SetCoordinate operation from
     *       (b).
     *
     * 4. If a ForLoop dimension is required, find the containing
     *    ForLoop operation.  The chain is added above the ForLoop
     *    operation.
     *
     * 5. If both ForLoop and Unroll dimensions are required, the
     *    chain is added above the containing ForLoop.
     */
    struct AddComputeIndex
    {
        void operator()(KernelGraph& kgraph, int candidate)
        {
            auto log = rocRoller::Log::getLogger();

            log->debug("KernelGraph::addComputeIndex({}): ", candidate);

            auto addChainAbove = [&](int target, ComputeIndexChain& chain) {
                if(loopScopes.count(target) == 0)
                {
                    loopScopes[target] = replaceWithScope(kgraph, target, false);
                }
                auto scope = loopScopes[target];
                kgraph.control.addElement(Body(), {scope}, {chain.top});
                kgraph.control.addElement(Sequence(), {chain.bottom}, {target});
            };

            auto addChainBelow = [&](int target, ComputeIndexChain& chain) {
                kgraph.control.addElement(Initialize(), {target}, {chain.top});
            };

            auto [target, direction] = getOperationTarget(candidate, kgraph);
            auto required            = findRequiredCoordinates(target, direction, kgraph);
            auto forLoopCoordinates  = filterCoordinates<ForLoop>(required, kgraph);
            auto unrollCoordinates   = filterCoordinates<Unroll>(required, kgraph);

            auto maybeForLoop = findContainingOperation<ForLoopOp>(candidate, kgraph);

            auto hasForLoop = !forLoopCoordinates.empty();
            auto hasUnroll  = !unrollCoordinates.empty();

            ExpressionPtr step = Expression::literal(1u);
            if(maybeForLoop)
            {
                auto [lhs, rhs] = getForLoopIncrement(kgraph, *maybeForLoop);
                step            = simplify(rhs);
            }

            auto chain = addComputeIndex(kgraph, candidate, step);

            // TODO: Handle ACCUMULATOR inside loops properly
            {
                auto [tile_tag, tile] = kgraph.getDimension<MacroTile>(candidate);
                if(hasForLoop && tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
                {
                    log->debug("KernelGraph::addComputeIndex({}): immediate", candidate);
                    auto scope = replaceWithScope(kgraph, candidate);
                    kgraph.control.addElement(Body(), {scope}, {chain.top});
                    kgraph.control.addElement(Sequence(), {chain.bottom}, {candidate});
                    return;
                }
            }

            if(hasUnroll)
            {
                log->debug("KernelGraph::addComputeIndex({}): hasUnroll", candidate);

                auto unrollCoordinateValues
                    = getUnrollCoordinateValues(candidate, unrollCoordinates, kgraph);
                auto maybeSetCoordinate
                    = findEarliestMatchingSetCoordinate(candidate, unrollCoordinateValues, kgraph);

                AssertFatal(maybeSetCoordinate, "Missing SetCoordinate operation.");

                auto setCoordinate = *maybeSetCoordinate;

                if(chain.update > 0)
                {
                    AssertFatal(hasForLoop, "Update with no ForLoop?");
                    AssertFatal(maybeForLoop, "Missing ForLoop operation.");

                    auto forLoop = *maybeForLoop;
                    addChainAbove(forLoop, chain);
                    kgraph.control.addElement(ForLoopIncrement(), {forLoop}, {chain.update});
                }
                else
                {
                    addChainBelow(setCoordinate, chain);
                }

                return;
            }

            if(hasForLoop)
            {
                log->debug("KernelGraph::addComputeIndex({}): forLoop", candidate);
                AssertFatal(maybeForLoop, "Missing ForLoop operation.");

                auto forLoop = *maybeForLoop;
                addChainAbove(forLoop, chain);
                if(chain.update > 0)
                    kgraph.control.addElement(ForLoopIncrement(), {forLoop}, {chain.update});

                return;
            }

            if(maybeForLoop)
            {
                log->debug("KernelGraph::addComputeIndex({}): containing forLoop", candidate);

                auto forLoop = *maybeForLoop;
                addChainAbove(forLoop, chain);
                if(chain.update > 0)
                    kgraph.control.addElement(ForLoopIncrement(), {forLoop}, {chain.update});

                return;
            }

            log->debug("KernelGraph::addComputeIndex({}): immediate", candidate);
            auto scope = replaceWithScope(kgraph, candidate);
            kgraph.control.addElement(Body(), {scope}, {chain.top});
            kgraph.control.addElement(Sequence(), {chain.bottom}, {candidate});
        }

    private:
        std::map<int, int> loopScopes;
    };

    KernelGraph addComputeIndexOperations(KernelGraph const& original)
    {
        auto            kgraph = original;
        AddComputeIndex precompute;
        for(auto candidate : findComputeIndexCandidates(kgraph))
            precompute(kgraph, candidate);
        return kgraph;
    }
}
