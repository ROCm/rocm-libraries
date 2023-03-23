
#include <algorithm>
#include <typeindex>
#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;
    namespace Expression = rocRoller::Expression;
    using namespace Expression;

    using GD = Graph::Direction;

    enum ComputeIndexChainType
    {
        STORE_ELEM,
        STORE_WAVE_MATRIX_ACCUMULATOR,

        LOAD_ELEM,
        LOAD_ELEM_MATRIX_A,
        LOAD_ELEM_MATRIX_B,

        LOAD_WAVE_MATRIX_A,
        LOAD_WAVE_MATRIX_B,
        LOAD_WAVE_MATRIX_ACCUMULATOR,

        LOAD_LDS_MATRIX_A,
        LOAD_LDS_MATRIX_B,
    };

    struct ComputeIndexChainSpecification
    {
        int                     target;
        ComputeIndexChainType   type;
        int                     location;
        Graph::Direction        direction;
        int                     forLoop = -1;
        std::unordered_set<int> zeros;
    };

    bool operator<(const ComputeIndexChainSpecification& a, const ComputeIndexChainSpecification& b)
    {
        return std::tie(a.target, a.type, a.location, a.direction)
               < std::tie(b.target, b.type, b.location, b.direction);
    }

    struct DeferredConnection
    {
        ConnectionSpec connectionSpec;
        int            coordinate;
    };

    struct ComputeIndexChain
    {
        int top, bottom;

        std::vector<DeferredConnection> connections;

        int update = -1;
    };

    /*
     * Helpers
     */

    /**
     * @brief Get ForLoop dimension assciated with ForLoopOp.
     */
    int getForLoop(int forLoopOp, KernelGraph const& kgraph)
    {
        namespace CG = rocRoller::KernelGraph::CoordinateGraph;

        auto range = kgraph.mapper.getConnections(forLoopOp)[0].coordinate;
        auto forLoop
            = only(kgraph.coordinates.getOutputNodeIndices(range, CG::isEdge<CG::DataFlow>));
        return *forLoop;
    }

    /**
     * @brief Append elements of b onto a.
     */
    std::vector<int> addZeros(std::vector<int> const& a, std::unordered_set<int> const& b)
    {
        std::vector<int> rv = a;
        std::copy(b.cbegin(), b.cend(), std::back_inserter(rv));
        return rv;
    }

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

    template <typename T>
    DeferredConnection DC(int coordinate, int sdim = 0)
    {
        DeferredConnection rv;
        rv.connectionSpec = Connections::TypeAndSubDimension{typeid(T), sdim};
        rv.coordinate     = coordinate;
        return rv;
    }

    /**
     * @brief Add ComputeIndexes for MATRIX_A/B from global.
     */
    ComputeIndexChain
        computeIndexElementMatrixAB(KernelGraph& graph, int load, int sdim, ExpressionPtr step)
    {
        AssertFatal(isOperation<LoadTiled>(graph.control.getElement(load)));

        auto user  = graph.mapper.get<User>(load);
        auto mac   = graph.mapper.get<MacroTileNumber>(load, sdim);
        auto elemX = graph.mapper.get<ElementNumber>(load, 0);
        auto elemY = graph.mapper.get<ElementNumber>(load, 1);

        auto dtype = graph.control.get<LoadTiled>(load)->vtype.dataType;

        auto offsetMac = graph.coordinates.addElement(Offset(), {user}, {mac});
        auto strideMac = graph.coordinates.addElement(Stride(), {user}, {mac});
        auto rowOffset = graph.coordinates.addElement(Offset(), {user}, {elemX});
        auto rowStride = graph.coordinates.addElement(Stride(), {user}, {elemX});
        auto colOffset = graph.coordinates.addElement(Offset(), {user}, {elemY});
        auto colStride = graph.coordinates.addElement(Stride(), {user}, {elemY});
        auto buffer    = getBuffer(graph, user, mac);

        std::vector<DeferredConnection> connections;
        connections.push_back(DC<Offset>(offsetMac, -1));
        connections.push_back(DC<Offset>(rowOffset, 0));
        connections.push_back(DC<Offset>(colOffset, 1));
        connections.push_back(DC<Stride>(strideMac, -1));
        connections.push_back(DC<Stride>(rowStride, 0));
        connections.push_back(DC<Stride>(colStride, 1));
        connections.push_back(DC<Buffer>(buffer));

        auto ciMac = makeComputeIndex(
            graph, user, mac, -1, offsetMac, strideMac, buffer, false, dtype, {elemX, elemY});
        auto ciRow = makeComputeIndex(graph,
                                      user,
                                      elemX,
                                      offsetMac,
                                      rowOffset,
                                      rowStride,
                                      buffer,
                                      false,
                                      dtype,
                                      {mac, elemY});
        auto ciCol = makeComputeIndex(graph,
                                      user,
                                      elemY,
                                      rowOffset,
                                      colOffset,
                                      colStride,
                                      buffer,
                                      false,
                                      dtype,
                                      {mac, elemX});

        graph.control.addElement(Sequence(), {ciMac}, {ciRow});
        graph.control.addElement(Sequence(), {ciRow}, {ciCol});

        auto offsetMacExpr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{offsetMac, Register::Type::Vector, DataType::UInt64});
        auto strideMacExpr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{strideMac, Register::Type::Scalar, DataType::UInt64});

        auto offsetUpdate = graph.control.addElement(
            Assign{Register::Type::Vector, offsetMacExpr + step * strideMacExpr});
        graph.mapper.connect(offsetUpdate, offsetMac, NaryArgument::DEST);

        return {ciMac, ciCol, connections, offsetUpdate};
    }

    /**
     * @brief Add ComputeIndexes for generic MATRIX from global.
     */
    ComputeIndexChain
        computeIndexElementMatrix(KernelGraph& graph, int loadstore, int source, bool forward)
    {
        auto elemX = graph.mapper.get<ElementNumber>(loadstore, 0);
        auto elemY = graph.mapper.get<ElementNumber>(loadstore, 1);

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

        int rowOffset, rowStride, colOffset, colStride, buffer;
        if(forward)
        {
            rowOffset = graph.coordinates.addElement(Offset(), {elemX}, {source});
            rowStride = graph.coordinates.addElement(Stride(), {elemX}, {source});
            colOffset = graph.coordinates.addElement(Offset(), {elemY}, {source});
            colStride = graph.coordinates.addElement(Stride(), {elemY}, {source});
            buffer    = getBuffer(graph, elemX, source);
        }
        else
        {
            rowOffset = graph.coordinates.addElement(Offset(), {source}, {elemX});
            rowStride = graph.coordinates.addElement(Stride(), {source}, {elemX});
            colOffset = graph.coordinates.addElement(Offset(), {source}, {elemY});
            colStride = graph.coordinates.addElement(Stride(), {source}, {elemY});
            buffer    = getBuffer(graph, source, elemX);
        }

        std::vector<DeferredConnection> connections;
        connections.push_back(DC<Offset>(rowOffset, 0));
        connections.push_back(DC<Offset>(colOffset, 1));
        connections.push_back(DC<Stride>(rowStride, 0));
        connections.push_back(DC<Stride>(colStride, 1));
        connections.push_back(DC<Buffer>(buffer));

        auto ciRow = makeComputeIndex(graph,
                                      source,
                                      elemX,
                                      -1,
                                      rowOffset,
                                      rowStride,
                                      buffer,
                                      forward,
                                      dtype,
                                      {elemY},
                                      offsettype,
                                      offsettype);
        auto ciCol = makeComputeIndex(graph,
                                      source,
                                      elemY,
                                      rowOffset,
                                      colOffset,
                                      colStride,
                                      buffer,
                                      forward,
                                      dtype,
                                      {elemX},
                                      offsettype,
                                      offsettype);

        graph.control.addElement(Sequence(), {ciRow}, {ciCol});

        return {ciRow, ciCol, connections};
    }

    /**
     * @brief Add ComputeIndexes for WAVE MATRIX_A/B from LDS.
     */
    ComputeIndexChain computeIndexWaveMatrixABLDS(KernelGraph&            graph,
                                                  int                     load,
                                                  int                     sdim,
                                                  std::unordered_set<int> zeros)
    {
        AssertFatal(isOperation<LoadLDSTile>(graph.control.getElement(load)));
        auto lds  = graph.mapper.get<LDS>(load);
        auto wave = graph.mapper.get<WaveTileNumber>(load, sdim);
        auto vgpr = graph.mapper.get<VGPR>(load);

        auto dtype = graph.control.get<LoadLDSTile>(load)->vtype.dataType;

        auto offsetWave = graph.coordinates.addElement(Offset(), {lds}, {wave});
        auto strideWave = graph.coordinates.addElement(Stride(), {lds}, {wave});
        auto offsetVgpr = graph.coordinates.addElement(Offset(), {lds}, {vgpr});
        auto strideVgpr = graph.coordinates.addElement(Stride(), {lds}, {vgpr});

        std::vector<DeferredConnection> connections;
        connections.push_back(DC<Offset>(offsetWave, 0));
        connections.push_back(DC<Offset>(offsetVgpr, 1));
        connections.push_back(DC<Stride>(strideWave, 0));
        connections.push_back(DC<Stride>(strideVgpr, 1));

        auto waveZeros = addZeros({vgpr}, zeros);
        auto vgprZeros = addZeros({wave}, zeros);

        std::vector<int> ciOperations;

        auto [required, path] = findRequiredCoordinates(lds, Graph::Direction::Downstream, graph);
        auto unrolls          = filterCoordinates<Unroll>(required, graph);
        for(auto unroll : unrolls)
        {
            auto unrollZeros = addZeros({wave, vgpr}, zeros);
            auto parents     = graph.coordinates.parentNodes(unroll);
            for(auto parent : parents)
            {
                if(path.contains(parent))
                {
                    auto strideUnroll = graph.coordinates.addElement(Stride(), {lds}, {parent});

                    ciOperations.push_back(makeComputeIndex(graph,
                                                            lds,
                                                            parent,
                                                            -1,
                                                            -1,
                                                            strideUnroll,
                                                            -1,
                                                            false,
                                                            dtype,
                                                            unrollZeros,
                                                            DataType::Int64,
                                                            DataType::Int64));
                }
            }
        }

        ciOperations.push_back(makeComputeIndex(graph,
                                                lds,
                                                wave,
                                                -1,
                                                offsetWave,
                                                strideWave,
                                                -1,
                                                false,
                                                dtype,
                                                waveZeros,
                                                DataType::UInt32,
                                                DataType::UInt32));
        ciOperations.push_back(makeComputeIndex(graph,
                                                lds,
                                                vgpr,
                                                offsetWave,
                                                offsetVgpr,
                                                strideVgpr,
                                                -1,
                                                false,
                                                dtype,
                                                vgprZeros,
                                                DataType::UInt32,
                                                DataType::UInt32));

        for(int i = 1; i < ciOperations.size(); ++i)
            graph.control.addElement(Sequence(), {ciOperations[i - 1]}, {ciOperations[i]});

        return {ciOperations.front(), ciOperations.back(), connections};
    }

    /**
     * @brief Add ComputeIndexes for WAVE MATRIX_A/B from global.
     */
    ComputeIndexChain computeIndexWaveMatrixAB(
        KernelGraph& graph, int load, int sdim, ExpressionPtr step, std::unordered_set<int> zeros)
    {
        auto user = graph.mapper.get<User>(load);
        auto mac  = graph.mapper.get<MacroTileNumber>(load, sdim);
        auto wave = graph.mapper.get<WaveTileNumber>(load, sdim);
        auto vgpr = graph.mapper.get<VGPR>(load);

        auto dtype = graph.control.get<LoadTiled>(load)->vtype.dataType;

        auto offsetMac  = graph.coordinates.addElement(Offset(), {user}, {mac});
        auto strideMac  = graph.coordinates.addElement(Stride(), {user}, {mac});
        auto offsetWave = graph.coordinates.addElement(Offset(), {user}, {wave});
        auto strideWave = graph.coordinates.addElement(Stride(), {user}, {wave});
        auto offsetVgpr = graph.coordinates.addElement(Offset(), {user}, {vgpr});
        auto strideVgpr = graph.coordinates.addElement(Stride(), {user}, {vgpr});
        auto buffer     = getBuffer(graph, user, mac);

        std::vector<DeferredConnection> connections;
        connections.push_back(DC<Offset>(offsetMac, -1));
        connections.push_back(DC<Offset>(offsetWave, 0));
        connections.push_back(DC<Offset>(offsetVgpr, 1));
        connections.push_back(DC<Stride>(strideMac, -1));
        connections.push_back(DC<Stride>(strideWave, 0));
        connections.push_back(DC<Stride>(strideVgpr, 1));
        connections.push_back(DC<Buffer>(buffer));

        auto macZeros    = addZeros({wave, vgpr}, zeros);
        auto waveZeros   = addZeros({mac, vgpr}, zeros);
        auto vgprZeros   = addZeros({mac, wave}, zeros);
        auto unrollZeros = addZeros({mac, wave, vgpr}, zeros);

        std::vector<int> ciOperations;

        auto [required, path] = findRequiredCoordinates(user, Graph::Direction::Downstream, graph);
        auto unrolls          = filterCoordinates<Unroll>(required, graph);
        for(auto unroll : unrolls)
        {
            auto parents = graph.coordinates.parentNodes(unroll);
            for(auto parent : parents)
            {
                if(path.contains(parent))
                {
                    auto strideUnroll = graph.coordinates.addElement(Stride(), {user}, {parent});

                    ciOperations.push_back(makeComputeIndex(graph,
                                                            user,
                                                            parent,
                                                            -1,
                                                            -1,
                                                            strideUnroll,
                                                            -1,
                                                            false,
                                                            dtype,
                                                            unrollZeros,
                                                            DataType::Int64,
                                                            DataType::Int64));
                }
            }
        }

        ciOperations.push_back(makeComputeIndex(
            graph, user, mac, -1, offsetMac, strideMac, buffer, false, dtype, macZeros));
        ciOperations.push_back(makeComputeIndex(
            graph, user, wave, offsetMac, offsetWave, strideWave, buffer, false, dtype, waveZeros));
        ciOperations.push_back(makeComputeIndex(graph,
                                                user,
                                                vgpr,
                                                offsetWave,
                                                offsetVgpr,
                                                strideVgpr,
                                                buffer,
                                                false,
                                                dtype,
                                                vgprZeros));

        auto offsetMacExpr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{offsetMac, Register::Type::Vector, DataType::UInt64});
        auto strideMacExpr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{strideMac, Register::Type::Scalar, DataType::UInt64});

        auto offsetUpdate = graph.control.addElement(
            Assign{Register::Type::Vector, offsetMacExpr + step * strideMacExpr});
        graph.mapper.connect(offsetUpdate, offsetMac, NaryArgument::DEST);

        for(int i = 1; i < ciOperations.size(); ++i)
            graph.control.addElement(Sequence(), {ciOperations[i - 1]}, {ciOperations[i]});

        return {ciOperations.front(), ciOperations.back(), connections, offsetUpdate};
    }

    /**
     * @brief Add ComputeIndexes for VGPR MATRIX_ACCUMULATOR from global or LDS.
     */
    ComputeIndexChain computeIndexMatrixAccumulator(KernelGraph&                   graph,
                                                    int                            op,
                                                    bool                           forward,
                                                    std::unordered_set<int> const& zeros)
    {
        rocRoller::Log::getLogger()->debug(
            "KernelGraph::addComputeIndexMatrixAccumulator({}, {})", op, forward);

        auto [source, _d] = getOperationTarget(op, graph);
        AssertFatal(source > 0, "User or LDS dimension not found");

        auto vgprBlock = graph.mapper.get<VGPRBlockNumber>(op);
        auto vgprIndex = graph.mapper.get<VGPRBlockIndex>(op);

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

        int offsetVgprBlock, strideVgprBlock, offsetVgprIndex, strideVgprIndex, buffer;
        if(forward)
        {
            offsetVgprBlock = graph.coordinates.addElement(Offset(), {vgprBlock}, {source});
            strideVgprBlock = graph.coordinates.addElement(Stride(), {vgprBlock}, {source});
            offsetVgprIndex = graph.coordinates.addElement(Offset(), {vgprIndex}, {source});
            strideVgprIndex = graph.coordinates.addElement(Stride(), {vgprIndex}, {source});
            buffer          = getBuffer(graph, vgprIndex, source);
        }
        else
        {
            offsetVgprBlock = graph.coordinates.addElement(Offset(), {source}, {vgprBlock});
            strideVgprBlock = graph.coordinates.addElement(Stride(), {source}, {vgprBlock});
            offsetVgprIndex = graph.coordinates.addElement(Offset(), {source}, {vgprIndex});
            strideVgprIndex = graph.coordinates.addElement(Stride(), {source}, {vgprIndex});
            buffer          = getBuffer(graph, source, vgprIndex);
        }

        std::vector<DeferredConnection> connections;
        connections.push_back(DC<Offset>(offsetVgprBlock, 0));
        connections.push_back(DC<Offset>(offsetVgprIndex, 1));
        connections.push_back(DC<Stride>(strideVgprBlock, 0));
        connections.push_back(DC<Stride>(strideVgprIndex, 1));
        connections.push_back(DC<Buffer>(buffer));

        std::vector<int> ciOperations;

        auto vgprBlockZeros = addZeros({vgprIndex}, zeros);
        auto vgprIndexZeros = addZeros({vgprBlock}, zeros);
        auto unrollZeros    = addZeros({vgprIndex, vgprBlock}, zeros);

        auto [required, path]
            = findRequiredCoordinates(source, forward ? GD::Upstream : GD::Downstream, graph);

        int baseFor    = -1;
        int baseUpdate = -1;

        auto maybeForLoop = findContainingOperation<ForLoopOp>(op, graph);
        if(maybeForLoop)
        {
            auto forLoop = getForLoop(*maybeForLoop, graph);

            int offset, stride;
            if(forward)
            {
                offset = graph.coordinates.addElement(Offset(), {forLoop}, {source});
                stride = graph.coordinates.addElement(Stride(), {forLoop}, {source});
            }
            else
            {
                offset = graph.coordinates.addElement(Offset(), {source}, {forLoop});
                stride = graph.coordinates.addElement(Stride(), {source}, {forLoop});
            }
            connections.push_back(DC<Offset>(offset, -1));
            connections.push_back(DC<Stride>(stride, -1));

            auto offsetExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{offset, Register::Type::Vector, DataType::UInt64});
            auto strideExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{stride, Register::Type::Scalar, DataType::UInt64});

            baseUpdate
                = graph.control.addElement(Assign{Register::Type::Vector, offsetExpr + strideExpr});
            graph.mapper.connect(baseUpdate, offset, NaryArgument::DEST);

            baseFor = offset;

            ciOperations.push_back(makeComputeIndex(graph,
                                                    source,
                                                    forLoop,
                                                    -1,
                                                    offset,
                                                    stride,
                                                    -1,
                                                    forward,
                                                    dtype,
                                                    unrollZeros,
                                                    DataType::UInt64,
                                                    DataType::UInt64));

            vgprBlockZeros.push_back(forLoop);
            vgprIndexZeros.push_back(forLoop);
            unrollZeros.push_back(forLoop);
        }

        auto unrolls = filterCoordinates<Unroll>(required, graph);
        for(auto unroll : unrolls)
        {
            std::vector<int> neighbourNodes;
            if(forward)
                neighbourNodes = graph.coordinates.childNodes(unroll).to<std::vector>();
            else
                neighbourNodes = graph.coordinates.parentNodes(unroll).to<std::vector>();
            for(auto neighbourNode : neighbourNodes)
            {
                if(path.contains(neighbourNode))
                {
                    int strideUnroll;
                    if(forward)
                        strideUnroll
                            = graph.coordinates.addElement(Stride(), {neighbourNode}, {source});
                    else
                        strideUnroll
                            = graph.coordinates.addElement(Stride(), {source}, {neighbourNode});

                    ciOperations.push_back(makeComputeIndex(graph,
                                                            source,
                                                            neighbourNode,
                                                            baseFor,
                                                            -1,
                                                            strideUnroll,
                                                            -1,
                                                            forward,
                                                            dtype,
                                                            unrollZeros,
                                                            DataType::Int64,
                                                            DataType::Int64));
                }
            }
        }

        ciOperations.push_back(makeComputeIndex(graph,
                                                source,
                                                vgprBlock,
                                                baseFor,
                                                offsetVgprBlock,
                                                strideVgprBlock,
                                                buffer,
                                                forward,
                                                dtype,
                                                vgprBlockZeros,
                                                offsettype,
                                                offsettype));
        ciOperations.push_back(makeComputeIndex(graph,
                                                source,
                                                vgprIndex,
                                                offsetVgprBlock,
                                                offsetVgprIndex,
                                                strideVgprIndex,
                                                buffer,
                                                forward,
                                                dtype,
                                                vgprIndexZeros,
                                                offsettype,
                                                offsettype));

        for(int i = 1; i < ciOperations.size(); ++i)
            graph.control.addElement(Sequence(), {ciOperations[i - 1]}, {ciOperations[i]});

        return {ciOperations.front(), ciOperations.back(), connections, baseUpdate};
    }

    /**
     * @brief Generic routine to create a ComputeIndex chain for a
     * load/store operation.
     *
     * @param kgraph Kernel graph to add ComputeIndex operations to.
     * @param tag Load/store operation that needs ComputeIndex operations.
     */
    ComputeIndexChainType computeIndexChainType(KernelGraph const& kgraph, int tag)
    {
        auto store    = kgraph.control.get<StoreTiled>(tag);
        auto storeLDS = kgraph.control.get<StoreLDSTile>(tag);
        if(store || storeLDS)
        {
            auto [tileTag, tile] = kgraph.getDimension<MacroTile>(tag);
            if(tile.memoryType == MemoryType::VGPR || tile.memoryType == MemoryType::LDS)
            {
                return STORE_ELEM;
            }
            if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR
               && tile.memoryType == MemoryType::WAVE)
            {
                return STORE_WAVE_MATRIX_ACCUMULATOR;
            }
        }

        auto load = kgraph.control.get<LoadTiled>(tag);
        if(load)
        {
            auto [tileTag, tile] = kgraph.getDimension<MacroTile>(tag);
            if(tile.memoryType == MemoryType::VGPR && tile.layoutType == LayoutType::MATRIX_A)
            {
                return LOAD_ELEM_MATRIX_A;
            }
            if(tile.memoryType == MemoryType::VGPR && tile.layoutType == LayoutType::MATRIX_B)
            {
                return LOAD_ELEM_MATRIX_B;
            }
            if(tile.memoryType == MemoryType::VGPR)
            {
                return LOAD_ELEM;
            }
            if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
            {
                return LOAD_WAVE_MATRIX_ACCUMULATOR;
            }
            if(tile.layoutType == LayoutType::MATRIX_A)
            {
                return LOAD_WAVE_MATRIX_A;
            }
            if(tile.layoutType == LayoutType::MATRIX_B)
            {
                return LOAD_WAVE_MATRIX_B;
            }
        }

        auto loadLDS = kgraph.control.get<LoadLDSTile>(tag);
        if(loadLDS)
        {
            auto [tileTag, tile] = kgraph.getDimension<MacroTile>(tag);
            if(tile.memoryType == MemoryType::VGPR || tile.memoryType == MemoryType::LDS)
            {
                return LOAD_ELEM;
            }
            if(tile.layoutType == LayoutType::MATRIX_A)
            {
                return LOAD_LDS_MATRIX_A;
            }
            if(tile.layoutType == LayoutType::MATRIX_B)
            {
                return LOAD_LDS_MATRIX_B;
            }
        }

        Throw<FatalError>("Not implemented yet.");
    }

    /**
     * @brief Generic routine to create a ComputeIndex chain for a
     * load/store operation.
     *
     * @param kgraph Kernel graph to add ComputeIndex operations to.
     * @param tag Load/store operation that needs ComputeIndex operations.
     */
    ComputeIndexChain addComputeIndex(KernelGraph&            kgraph,
                                      int                     tag,
                                      ComputeIndexChainType   chainType,
                                      ExpressionPtr           step,
                                      std::unordered_set<int> zeros)
    {
        auto [source, _d] = getOperationTarget(tag, kgraph);

        switch(chainType)
        {
        case STORE_ELEM:
            return computeIndexElementMatrix(kgraph, tag, source, true);
        case STORE_WAVE_MATRIX_ACCUMULATOR:
            return computeIndexMatrixAccumulator(kgraph, tag, true, zeros);
        case LOAD_ELEM_MATRIX_A:
            return computeIndexElementMatrixAB(kgraph, tag, 1, step);
        case LOAD_ELEM_MATRIX_B:
            return computeIndexElementMatrixAB(kgraph, tag, 0, step);
        case LOAD_ELEM:
            return computeIndexElementMatrix(kgraph, tag, source, false);
        case LOAD_WAVE_MATRIX_ACCUMULATOR:
            return computeIndexMatrixAccumulator(kgraph, tag, false, zeros);
        case LOAD_WAVE_MATRIX_A:
            return computeIndexWaveMatrixAB(kgraph, tag, 1, step, zeros);
        case LOAD_WAVE_MATRIX_B:
            return computeIndexWaveMatrixAB(kgraph, tag, 0, step, zeros);
        case LOAD_LDS_MATRIX_A:
            return computeIndexWaveMatrixABLDS(kgraph, tag, 1, zeros);
        case LOAD_LDS_MATRIX_B:
            return computeIndexWaveMatrixABLDS(kgraph, tag, 0, zeros);
        default:
            Throw<FatalError>("Not implemented yet.");
        }

        Throw<FatalError>("Not implemented yet.");
    }

    /**
     * @brief Add ComputeIndex operations.
     *
     * Adding ComputeIndex operations to the control graph is done in
     * two phases: staging and committing.
     *
     * During the staging phase, we look at all load/store operations
     * in the control graph and "stage" the addition of ComputeIndex
     * operations.  During the staging phase, we are able to detect
     * when two or more load/store operations would result in the same
     * chain of ComputeIndex operations, and eliminate any
     * redundancies.
     *
     * Usually ComputeIndex operations come in sequential groups of
     * two or more operations, and hence we call them "compute index
     * chains".
     *
     * During the commit stage, we add ComputeIndex operations to the
     * graphs, and add connections for load/store operations to the
     * newly Base, Offset, and Stride elements of the coordinate
     * graph.
     *
     * For each candidate load/store operation:
     *
     * 1. The type of ComputeIndex chain is determined.
     *
     * 2. The required location of the ComputeIndex chain is
     *    determined.
     *
     * 3. The chain is staged.
     *
     * To determined where the chain should be placed:
     *
     * 1. Find all required coordinates by querying the Coordinate
     *    Transform graph.
     *
     * 2. If one-or-more Unroll dimension(s) are required:
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
     * 3. If a ForLoop dimension is required, find the containing
     *    ForLoop operation.  The chain is added above the ForLoop
     *    operation.
     *
     * 4. If both ForLoop and Unroll dimensions are required, the
     *    chain is added above the containing ForLoop.
     */
    struct AddComputeIndex
    {
        void stageChain(int                     target,
                        int                     candidate,
                        int                     location,
                        ComputeIndexChainType   type,
                        Graph::Direction        direction,
                        int                     forLoop = -1,
                        std::unordered_set<int> zeros   = {})
        {
            ComputeIndexChainSpecification spec{target, type, location, direction, forLoop, zeros};
            m_chains[spec].push_back(candidate);
        }

        void stage(KernelGraph const& kgraph, int candidate)
        {
            auto log = rocRoller::Log::getLogger();

            log->debug("KernelGraph::addComputeIndex({}): ", candidate);

            auto [target, direction] = getOperationTarget(candidate, kgraph);
            auto [required, path]    = findRequiredCoordinates(target, direction, kgraph);
            auto forLoopCoordinates  = filterCoordinates<ForLoop>(required, kgraph);
            auto unrollCoordinates   = filterCoordinates<Unroll>(required, kgraph);

            auto maybeForLoop = findContainingOperation<ForLoopOp>(candidate, kgraph);

            auto hasForLoop = !forLoopCoordinates.empty();
            auto hasUnroll  = !unrollCoordinates.empty();

            auto type = computeIndexChainType(kgraph, candidate);

            if(hasForLoop)
            {
                log->debug("KernelGraph::addComputeIndex({}): forLoop", candidate);
                AssertFatal(maybeForLoop, "Missing ForLoop operation.");

                auto forLoop = *maybeForLoop;
                stageChain(
                    target, candidate, forLoop, type, GD::Upstream, forLoop, unrollCoordinates);
                return;
            }

            if(hasUnroll)
            {
                log->debug("KernelGraph::addComputeIndex({}): hasUnroll", candidate);

                auto kernel = *kgraph.control.roots().begin();
                stageChain(target, candidate, kernel, type, GD::Downstream, -1, unrollCoordinates);
                return;
            }

            if(maybeForLoop)
            {
                log->debug("KernelGraph::addComputeIndex({}): containing forLoop", candidate);

                auto forLoop = *maybeForLoop;
                stageChain(target, candidate, forLoop, type, GD::Upstream, forLoop);
                return;
            }

            log->debug("KernelGraph::addComputeIndex({}): immediate", candidate);
            stageChain(target, candidate, candidate, type, GD::Upstream);
        }

        KernelGraph commit(KernelGraph const& original) const
        {
            auto               kgraph = original;
            std::map<int, int> scopes;

            for(auto const& [spec, candidates] : m_chains)
            {
                ExpressionPtr step = Expression::literal(1u);
                if(spec.forLoop > 0)
                {
                    auto [lhs, rhs] = getForLoopIncrement(kgraph, spec.forLoop);
                    step            = simplify(rhs);
                }

                // Use first candidate to compute indexes
                rocRoller::Log::getLogger()->debug("KernelGraph::addComputeIndex()::commit(): {}",
                                                   candidates[0]);
                auto chain = addComputeIndex(kgraph, candidates[0], spec.type, step, spec.zeros);

                if(spec.direction == GD::Downstream)
                {
                    // Add ComputeIndexes to an Initialize block below target
                    kgraph.control.addElement(Initialize(), {spec.location}, {chain.top});
                }
                else
                {
                    // Add ComputeIndexes in a Scope above target
                    if(!scopes.contains(spec.location))
                    {
                        scopes[spec.location] = replaceWith(
                            kgraph, spec.location, kgraph.control.addElement(Scope()), false);
                    }
                    auto scope = scopes[spec.location];
                    kgraph.control.addElement(Body(), {scope}, {chain.top});
                    kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                }

                // If the chain has a increment
                if(chain.update > 0)
                    kgraph.control.addElement(ForLoopIncrement(), {spec.forLoop}, {chain.update});

                // Add deferred connections
                for(auto candidate : candidates)
                {
                    for(auto const& dc : chain.connections)
                    {
                        kgraph.mapper.connect(candidate, dc.coordinate, dc.connectionSpec);
                    }
                }
            }

            return kgraph;
        }

    private:
        std::map<ComputeIndexChainSpecification, std::vector<int>> m_chains;
    };

    KernelGraph addComputeIndexOperations(KernelGraph const& original)
    {
        AddComputeIndex indexer;
        for(auto candidate :
            findComputeIndexCandidates(original, *original.control.roots().begin()))
            indexer.stage(original, candidate);
        return indexer.commit(original);
    }
}
