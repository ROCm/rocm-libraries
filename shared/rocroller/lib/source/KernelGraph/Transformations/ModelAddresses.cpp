// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ModelAddresses.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    ModelAddresses::ModelAddresses(ContextPtr context)
        : m_context(context)
    {
        setup();
    }

    void ModelAddresses::setup()
    {
        namespace CT         = rocRoller::KernelGraph::CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        for(int i = 0; i < 3; ++i)
        {
            workgroupOffset[i] = arguments.size();
            auto wg_name       = concatenate("WG", i);
            auto wg_carg       = CommandArgument(
                nullptr, DataType::UInt32, workgroupOffset[i], DataDirection::ReadOnly, wg_name);
            auto wg = std::make_shared<CommandArgument>(wg_carg);
            arguments.appendUnbound<uint>(wg_name);

            workitemOffset[i] = arguments.size();
            auto wi_name      = concatenate("WI", i);
            auto wi_carg      = CommandArgument(
                nullptr, DataType::UInt32, workitemOffset[i], DataDirection::ReadOnly, wi_name);
            auto wi = std::make_shared<CommandArgument>(wi_carg);
            arguments.appendUnbound<uint>(wi_name);

            kernelWorkgroupIndexes[i] = std::make_shared<Expression::Expression>(wg);
            kernelWorkitemIndexes[i]  = std::make_shared<Expression::Expression>(wi);
        }

        rawArguments     = arguments.dataVector();
        runtimeArguments = RuntimeArguments(rawArguments.data(), rawArguments.size());
    }

    void ModelAddresses::setWorkgroup(uint dim, uint value)
    {
        std::memcpy(rawArguments.data() + workgroupOffset[dim], &value, sizeof(value));
    }

    void ModelAddresses::setWorkitem(uint dim, uint value)
    {
        std::memcpy(rawArguments.data() + workitemOffset[dim], &value, sizeof(value));
    }

    std::vector<size_t>
        ModelAddresses::getLDSAddressesImpl(KernelGraph&                                     graph,
                                            int                                              tag,
                                            LoadStoreTileGenerator::LoadStoreTileInfo const& info,
                                            LDSDirection direction)
    {
        namespace CT         = rocRoller::KernelGraph::CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace CoordinateGraph;
        using namespace Expression;

        auto [ldsTag, lds]   = graph.getDimension<LDS>(tag);
        auto [tileTag, tile] = graph.getDimension<MacroTile>(tag);

        auto maybeParentLDS
            = only(graph.coordinates.getOutputNodeIndices(ldsTag, CT::isEdge<Duplicate>));
        if(maybeParentLDS)
            ldsTag = *maybeParentLDS;

        const auto     varInfo     = DataTypeInfo::Get(info.varType);
        const auto     segInfo     = DataTypeInfo::Get(varInfo.segmentVariableType);
        const auto     packedCount = std::max<uint32_t>(1u, varInfo.packing);
        const uint64_t numElements = info.m * info.n * packedCount;
        const uint64_t segmentBits = static_cast<uint64_t>(segInfo.elementBits);

        AssertFatal(numElements > 0, "Invalid LDS tile element count.", ShowValue(numElements));

        auto coords = graph.buildTransformer(tag);

        // Follows code in loadStoreTileGenerator
        if(tile.memoryType == MemoryType::WAVE || tile.memoryType == MemoryType::WAVE_SWIZZLE
           || tile.memoryType == MemoryType::WAVE_FROM_GLOBAL)
        {
            auto [vgprTag, vgpr] = graph.getDimension<VGPR>(tag);
            coords.setCoordinate(vgprTag, Expression::literal(0));
        }
        else if(tile.memoryType == MemoryType::VGPR || tile.memoryType == MemoryType::WAVE_SPLIT
                || tile.memoryType == MemoryType::LDS)
        {
            auto [elemXTag, elemX] = graph.getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = graph.getDimension<ElementNumber>(tag, 1);
            coords.setCoordinate(elemXTag, Expression::literal(0));
            coords.setCoordinate(elemYTag, Expression::literal(0));
        }

        coords.fillExecutionCoordinates(kernelWorkgroupIndexes, kernelWorkitemIndexes);

        auto index = (direction == LDSDirection::Load) ? coords.reverse({ldsTag})[0]
                                                       : coords.forward({ldsTag})[0];

        Log::debug("{}: tag {}, segmentBits {}, numElements {}",
                   (direction == LDSDirection::Load) ? "LoadLDSTile" : "StoreLDSTile",
                   tag,
                   segmentBits,
                   numElements);

        const auto byteIndex = index * Expression::literal(segmentBits) / Expression::literal(8u);

        Log::debug("Offset expression: {}", toString(byteIndex));

        setWorkgroup(0, 0);

        std::vector<size_t> addresses;
        for(uint wi = 0; wi < product(m_context->kernel()->workgroupSize()); ++wi)
        {
            setWorkitem(0, wi);

            const auto offsetValue = Expression::evaluate(byteIndex, runtimeArguments);

            const auto offset = std::visit(
                [](auto&& x) {
                    using T = std::decay_t<decltype(x)>;
                    if constexpr(std::is_same_v<T, rocRoller::Buffer>)
                    {
                        Throw<FatalError>("Cannot extract LDS address from "
                                          "rocRoller::Buffer");
                        return size_t{0};
                    }
                    else
                    {
                        return (size_t)x;
                    }
                },
                offsetValue);

            addresses.push_back(offset);
        }
        return addresses;
    }

    template <typename Op>
    std::vector<size_t> ModelAddresses::getLDSAddresses(KernelGraph& graph, int tag, Op const& op)
    {
        constexpr bool isLoad    = std::is_same_v<Op, LoadLDSTile>;
        constexpr auto direction = isLoad ? LDSDirection::Load : LDSDirection::Store;

        // Use nullptr context to avoid modifying the real tag manager during modeling.
        auto                   graphPtr = std::make_shared<KernelGraph>(graph);
        LoadStoreTileGenerator tileGenerator(
            graphPtr, nullptr, m_context->kernel()->max_flat_workgroup_size());

        LoadStoreTileGenerator::LoadStoreTileInfo info;
        if constexpr(isLoad)
        {
            info = tileGenerator.getLoadLDSTileInfo(tag, op);
        }
        else
        {
            info = tileGenerator.getStoreLDSTileInfo(tag, op);
        }

        return getLDSAddressesImpl(graph, tag, info, direction);
    }

    template std::vector<size_t>
        ModelAddresses::getLDSAddresses(KernelGraph&, int, LoadLDSTile const&);
    template std::vector<size_t>
        ModelAddresses::getLDSAddresses(KernelGraph&, int, StoreLDSTile const&);

    KernelGraph ModelAddresses::apply(KernelGraph const& original)
    {
        KernelGraph graph = original;

        auto root = graph.control.roots().only();
        AssertFatal(root.has_value());

        auto allNodes
            = graph.control.depthFirstVisit(*root).filter(graph.control.isElemType<Operation>());

        for(const auto node : allNodes)
        {
            auto modelAddresses = [&](std::vector<size_t> addresses) {
                if(addresses.empty())
                    return;

                std::vector<size_t> normalizedAddresses;
                auto minAddress = *std::min_element(addresses.begin(), addresses.end());
                for(auto addr : addresses)
                {
                    normalizedAddresses.push_back(addr - minAddress);
                }

                graph.modelledAddresses[node] = std::move(normalizedAddresses);
            };

            auto visitor
                = rocRoller::overloaded{[&](CIsAnyOf<LoadLDSTile, StoreLDSTile> auto op) {
                                            modelAddresses(getLDSAddresses(graph, node, op));
                                        },
                                        [&](auto op) {}};

            std::visit(visitor, graph.control.getNode(node));
        }
        return graph;
    }

    std::string ModelAddresses::name() const
    {
        return "ModelAddresses";
    }
}
