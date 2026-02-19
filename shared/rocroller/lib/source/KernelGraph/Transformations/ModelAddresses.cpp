/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2026 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ModelAddresses.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <algorithm>
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

    void ModelAddresses::setWorkgroup(uint offset, uint value)
    {
        *((uint*)(rawArguments.data() + workgroupOffset[offset])) = value;
    }

    void ModelAddresses::setWorkitem(uint offset, uint value)
    {
        *((uint*)(rawArguments.data() + workitemOffset[offset])) = value;
    }

    Generator<size_t>
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
        const auto     packedCount = std::max<uint32_t>(1u, varInfo.packing);
        const uint64_t numElements = info.m * info.n * packedCount;
        const uint64_t numBits     = static_cast<uint64_t>(varInfo.elementBits);

        AssertFatal(numElements > 0, "Invalid LDS tile element count.", ShowValue(numElements));

        auto numBytes = (numBits * numElements + 7u) / 8u;

        auto coords = graph.buildTransformer(tag);

        // Set the appropriate coordinate to 0 based on memory type
        if(tile.memoryType == MemoryType::WAVE)
        {
            auto [vgprTag, vgpr] = graph.getDimension<VGPR>(tag);
            coords.setCoordinate(vgprTag, Expression::literal(0));
        }
        else if(tile.memoryType == MemoryType::VGPR)
        {
            auto [elemXTag, elemX] = graph.getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = graph.getDimension<ElementNumber>(tag, 1);
            coords.setCoordinate(elemXTag, Expression::literal(0));
            coords.setCoordinate(elemYTag, Expression::literal(0));
        }

        coords.fillExecutionCoordinates(nullptr, kernelWorkgroupIndexes, kernelWorkitemIndexes);

        // Use reverse for loads (downstream: LDS -> registers), forward for stores (upstream: registers -> LDS)
        auto index = (direction == LDSDirection::Load) ? coords.reverse({ldsTag})[0]
                                                       : coords.forward({ldsTag})[0];

        Log::debug("{}: tag {}, numBits {}, numElements {}, numBytes {}",
                   (direction == LDSDirection::Load) ? "LoadLDSTile" : "StoreLDSTile",
                   tag,
                   numBits,
                   numElements,
                   numBytes);

        const auto byteIndex
            = index * Expression::literal(numBytes) / Expression::literal(numElements);

        Log::debug("Offset expression: {}", toString(byteIndex));

        for(uint wg = 0; wg < 1; ++wg)
        {
            setWorkgroup(0, wg);
            for(uint wi = 0; wi < 64; ++wi)
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

                co_yield offset;
            }
        }
    }

    template <typename Op>
    Generator<size_t> ModelAddresses::getLDSAddresses(KernelGraph& graph, int tag, Op const& op)
    {
        auto [tileTag, tile] = graph.getDimension<MacroTile>(tag);

        constexpr bool isLoad          = std::is_same_v<Op, LoadLDSTile>;
        const auto     expectedMemType = isLoad ? MemoryType::WAVE : MemoryType::VGPR;
        constexpr auto direction       = isLoad ? LDSDirection::Load : LDSDirection::Store;

        if(tile.memoryType == expectedMemType)
        {
            auto                   graphPtr = std::make_shared<KernelGraph>(graph);
            LoadStoreTileGenerator tileGenerator(
                graphPtr, m_context, m_context->kernel()->max_flat_workgroup_size());

            LoadStoreTileGenerator::LoadStoreTileInfo info;
            if constexpr(isLoad)
            {
                info = tileGenerator.getLoadLDSTileInfo(tag, op);
            }
            else
            {
                std::vector<std::string> comments;
                info = tileGenerator.getStoreLDSTileInfo(tag, op, comments);
            }

            co_yield getLDSAddressesImpl(graph, tag, info, direction);
        }
        else
        {
            Log::debug("Skipping LDS address annotations due to {}", toString(tile.memoryType));
        }
    }

    template Generator<size_t>
        ModelAddresses::getLDSAddresses(KernelGraph&, int, LoadLDSTile const&);
    template Generator<size_t>
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
            auto modelAddresses = [&](auto&& generator) {
                const auto addresses
                    = std::forward<decltype(generator)>(generator).template to<std::vector>();

                // TODO: add assert
                // AssertFatal(!addresses.empty());

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
