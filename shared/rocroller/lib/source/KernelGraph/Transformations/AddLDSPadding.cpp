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

/**
@class AddLDSPadding
@brief Add element padding to LDS buffers.

Padding LDS buffers is used to reduce the number of LDS bank
conflicts.

Padding is added to flat LDS buffers.  For each `LDS` node, upstream
`Flatten` edges are transformed to `Join` edges, and downstream
`Tile` edges are transformed to `Split` edges.

Recall that `Join` and `Split` edges honour the `stride` attribute of
upstream/downstream nodes.  Padding is accomplished by updating the
`stride` attributes of the upstream/downsteam nodes.

In particular, the slow strides are set to the fast size plus a
padding value, and the fast strides are set to 1.

For example:

    MacroTileIndex                  MacroTileIndex
       size=256                       size=128
                 \                  /
                  ------------------
                          |
                       Flatten
                          |
                         LDS
                     size=32,768
                          |
                        Tile
                          |
                  ------------------
                 /                  \
    MacroTileIndex                 MacroTileIndex
       size=256                       size=128

will be transformed to:

    MacroTileIndex                  MacroTileIndex
       size=256                       size=128
   stride=128+padding                 stride=1
                 \                  /
                  ------------------
                          |
                        Join
                          |
                         LDS
              size=32,768 + 256*padding
                          |
                        Split
                          |
                  ------------------
                 /                  \
    MacroTileIndex                 MacroTileIndex
       size=256                       size=128
    stride=128+padding                stride=1


Note that the size of the LDS allocation is computed in `getNumLDSElements()`.

For direct-to-LDS loads... TODO

*/

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using GD = Graph::Direction;

        using namespace Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        namespace AddLDSPaddingDetail
        {
            uint computeDefaultLDSPaddingElements(KernelGraph const&    graph,
                                                  LDSPaddingInfo const& info)
            {
                constexpr uint ldsBankWidthInBits = 32u;

                auto dataTypeInfo = DataTypeInfo::Get(info.dataType);

                if(dataTypeInfo.elementBits % 4 != 0)
                    return 0u;

                return ldsBankWidthInBits / dataTypeInfo.elementBits;
            }
        }

        using namespace AddLDSPaddingDetail;

        /**
         * @brief Get the upstream Flatten edge coming in to an LDS node.
         */
        std::optional<int> getFlattenEdgeTag(KernelGraph const& graph, int tag)
        {
            for(auto elem : graph.coordinates.getNeighbours<GD::Upstream>(tag))
            {
                auto maybeFlatten = graph.coordinates.get<Flatten>(elem);
                if(maybeFlatten)
                    return elem;
            }
            return {};
        }

        /**
         * @brief Get the downstream Tile edge coming out of an LDS node.
         */
        std::optional<int> getTileEdgeTag(KernelGraph const& graph, int tag)
        {
            for(auto elem : graph.coordinates.getNeighbours<GD::Downstream>(tag))
            {
                auto maybeTile = graph.coordinates.get<Tile>(elem);
                if(maybeTile)
                    return elem;
            }
            return {};
        }

        /**
         * @brief Propagate the size and stride of the slow dimension
         * to upstream/downstream node.
         *
         * For example, after padding the slow dimension's stride, the
         * joined dimension size needs to be updated.
         *
         *     Slow Dimension     Fast Dimension
         *        size=256           size=128
         *    stride=128+padding     stride=1
         *           \              /
         *            --------------
         *                    |
         *                  Join
         *                    |
         *                Dimension
         *               size=256*128   <--- should be updated to 256*(128+padding)
         */
        void propagateSize(KernelGraph& graph, int slowTag, Graph::Direction propagateDirection)
        {
            auto truePredicate = [](auto x) { return true; };

            auto slow       = graph.coordinates.getNode(slowTag);
            auto slowSize   = getUnsignedInt(evaluate(getSize(slow)));
            auto slowStride = getUnsignedInt(evaluate(getStride(slow)));

            int propagateToTag;
            if(propagateDirection == GD::Upstream)
            {
                propagateToTag = *only(graph.coordinates.getConnectedNodeIndices<GD::Upstream>(
                    slowTag, truePredicate));
            }
            else
            {
                propagateToTag = *only(graph.coordinates.getConnectedNodeIndices<GD::Downstream>(
                    slowTag, truePredicate));
            }

            auto propagateTo = graph.coordinates.getNode(propagateToTag);
            setSize(propagateTo, literal(slowSize * slowStride));
            graph.coordinates.setElement(propagateToTag, propagateTo);

            Log::info("KernelGraph::AddLDSPadding::propagateSize: updating tag {}, new size {}",
                      propagateToTag,
                      toString(getSize(propagateTo)));
        }

        /**
         * @brief Update the strides of the slow and fast dimensions.
         *
         * The slow dimension's stride is set to the size of the fast
         * dimension plus padding.
         *
         * The fast dimension's stride is set to 1.
         */
        void updateStrides(KernelGraph&              graph,
                           std::array<int, 2> const& tags,
                           uint                      numPaddingElements)
        {
            AssertFatal(tags.size() == 2, ShowValue(tags));

            auto getCoordinateSize
                = [&](auto tag) { return getSize(graph.coordinates.getNode(tag)); };

            auto slowerTag = tags[0];
            auto fasterTag = tags[1];

            auto slowerDim = graph.coordinates.getNode(slowerTag);
            auto fasterDim = graph.coordinates.getNode(fasterTag);

            auto slowStride = simplify(getSize(fasterDim) + literal(numPaddingElements));
            setStride(slowerDim, slowStride);
            setStride(fasterDim, literal(1u));

            graph.coordinates.setElement(slowerTag, slowerDim);
            graph.coordinates.setElement(fasterTag, fasterDim);

            Log::info("KernelGraph::AddLDSPadding::updateStrides: slow {}, fast {}, "
                      "numPaddingElements {}, new slow stride {}",
                      slowerTag,
                      fasterTag,
                      numPaddingElements,
                      toString(slowStride));
        }

        /**
         * @brief Match the transform coming out of an LDS node to the
         * same pattern as the transform coming into the LDS node.
         *
         * This is used when Direct2LDS loads are padded.
         *
         * For example, consider the following transforms in and out
         * of LDS:
         *
         *                               TileNumber     TileIndex   <---- sister tags
         *                                 size=64       size=4
         *                                       \       /
         *                                        Flatten
         *                                           |
         *     MacroTileIndex                  MacroTileIndex
         *        size=256                       size=128
         *               \                      /
         *                ----------------------
         *                           |
         *                        Flatten
         *                           |
         *                          LDS
         *                           |
         *                         Tile
         *                           |
         *                ----------------------
         *               /                      \
         *     MacroTileIndex                  MacroTileIndex
         *        size=256                       size=128
         *           |
         *         Tile     <---- edge
         *        /    \
         *  TileNumber  TileIndex    <---- tags
         *   size=32       size=8
         *
         * Note that above the LDS node, the small tile is 64x4; but
         * below the LDS the small tile is 32x8.  These are compatible
         * when there is no padding, and all edges are either Flatten or
         * Tile edges.
         *
         * If we pad one of the early dimensions, the compatibility
         * may be broken.  We can fix it by matching the downstream
         * transform to the upstream "sister" transform:
         *
         *                               TileNumber     TileIndex    <---- sister tags
         *                                 size=64       size=4
         *                            stride=64+padding   |
         *                                       \       /
         *                                         Join
         *                                           |
         *     MacroTileIndex                  MacroTileIndex
         *        size=128                   size=256+4*padding
         *               \                      /
         *                ----------------------
         *                           |
         *                        Flatten
         *                           |
         *                          LDS
         *                           |
         *                         Tile
         *                           |
         *                ----------------------
         *               /                      \
         *     MacroTileIndex                  MacroTileIndex
         *   size=256+4*padding                   size=128
         *           |
         *         Split   <---- new edge (overwrites edge)
         *           |
         *         -------------
         *        /             \
         *  TileNumber           TileIndex    <---- new tags (overwrites tags)
         *   size=64              size=4
         *  stride=64+padding    stride=1
         *       \              /
         *        --------------
         *            |
         *         Flatten
         *            |
         *          Linear
         *         size=256
         *            |
         *          Tile
         *         /    \
         *  TileNumber  TileIndex
         *   size=32      size=8
         *
         */
        bool matchTransforms(KernelGraph&              graph,
                             int&                      edge,
                             std::array<int, 2>&       tags,
                             std::array<int, 2> const& sisterTags)
        {
            auto getCoordinateSize
                = [&](int tag) { return getSize(graph.coordinates.getNode(tag)); };

            auto slowSize       = getCoordinateSize(tags[0]);
            auto slowSisterSize = getCoordinateSize(sisterTags[0]);

            if(identical(slowSize, slowSisterSize))
                return false;

            auto one = literal(1u);

            auto fastSize       = getCoordinateSize(tags[1]);
            auto fastSisterSize = getCoordinateSize(sisterTags[1]);

            auto newSlowStride = fastSisterSize;
            auto linearSlow = graph.coordinates.addElement(Linear(slowSisterSize, newSlowStride));
            auto linearFast = graph.coordinates.addElement(Linear(fastSisterSize, one));

            auto linearFlat
                = graph.coordinates.addElement(Linear(simplify(slowSize * fastSize), one));

            auto topTag = *only(graph.coordinates.getNeighbours<GD::Upstream>(edge));
            graph.coordinates.deleteElement(edge);

            edge = graph.coordinates.addElement(Split(), {topTag}, {linearSlow, linearFast});

            graph.coordinates.addElement(Flatten(), {linearSlow, linearFast}, {linearFlat});
            graph.coordinates.addElement(
                Tile(), std::vector<int>{linearFlat}, std::vector<int>{tags[0], tags[1]});

            tags = {linearSlow, linearFast};

            return true;
        }

        /**
         * @brief Get the layout type and data type of the LDS node.
         *
         * This will traverse upstream from the LDS node until it finds
         * a MacroTile node, and returns the layout type and data type
         * of that MacroTile.
         *
         * If no MacroTile is found, an empty optional is returned.
         */
        std::optional<std::pair<LayoutType, DataType>>
            getLayoutTypeAndDataType(KernelGraph const& graph, int ldsTag)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            std::optional<LayoutType> layoutType;
            std::optional<DataType>   dataType;

            auto isDataFlow = [&](int tag) -> bool {
                return graph.coordinates.get<CT::DataFlow>(tag).has_value();
            };

            auto target = ldsTag;
            while(true)
            {
                if(!dataType)
                {
                    for(auto conn : graph.mapper.getCoordinateConnections(target))
                        dataType = getVariableType(graph, conn.control).dataType;
                }

                auto edge = only(
                    filter(isDataFlow, graph.coordinates.getNeighbours<GD::Upstream>(target)));
                if(!edge)
                    break;
                target = only(graph.coordinates.getNeighbours<GD::Upstream>(*edge)).value();

                if(!layoutType)
                {
                    auto maybeMacroTile = graph.coordinates.get<MacroTile>(target);
                    if(maybeMacroTile)
                        layoutType = maybeMacroTile->layoutType;
                }
            }

            if(layoutType && dataType)
                return std::pair<LayoutType, DataType>{layoutType.value(), dataType.value()};

            return {};
        }

        /**
         * @brief Add LDS padding transformer.
         */
        struct AddLDSPaddingVisitor
        {
            AddLDSPaddingVisitor(CommandParametersPtr params)
                : m_params(std::move(params))
            {
            }

            void stage(KernelGraph const&, int);
            void commit(KernelGraph&);

        private:
            uint getLDSPaddingElements(KernelGraph const&, LDSPaddingInfo const&) const;

            CommandParametersPtr          m_params;
            std::map<int, LDSPaddingInfo> m_ldsTags;
        };

        /**
         * @brief Get the number of padding elements for a given LDS
         * node.
         *
         * If the padding is set to -1, this will compute a default
         * padding value.
         */
        uint AddLDSPaddingVisitor::getLDSPaddingElements(KernelGraph const&    graph,
                                                         LDSPaddingInfo const& info) const
        {
            if(m_params->padLDS.contains(info.layoutType))
            {
                auto padding = m_params->padLDS.at(info.layoutType);
                if(padding < 0)
                    return computeDefaultLDSPaddingElements(graph, info);
                return static_cast<uint>(padding);
            }
            return 0u;
        }

        /**
         * @brief Stage LDS nodes that are flattened/tiled.
         */
        void AddLDSPaddingVisitor::stage(KernelGraph const& graph, int ldsTag)
        {
            bool isDirect2LDS = graph.coordinates.get<LDS>(ldsTag)->isDirect2LDS;

            auto flattenEdgeTag = getFlattenEdgeTag(graph, ldsTag);
            auto tileEdgeTag    = getTileEdgeTag(graph, ldsTag);

            if(isDirect2LDS)
            {
                // Look for Flatten/Tile above/below the slower moving
                // dimension; add padding there.  This means padding
                // is applied to M0.
                auto slowFlattenTag = *only(
                    take(1, graph.coordinates.getNeighbours<GD::Upstream>(*flattenEdgeTag)));
                flattenEdgeTag = getFlattenEdgeTag(graph, slowFlattenTag);
                auto slowTileTag
                    = *only(take(1, graph.coordinates.getNeighbours<GD::Downstream>(*tileEdgeTag)));
                tileEdgeTag = getTileEdgeTag(graph, slowTileTag);
            }

            if((not flattenEdgeTag) || (not tileEdgeTag))
                return;

            auto maybeLayoutTypeAndDataType = getLayoutTypeAndDataType(graph, ldsTag);
            if(!maybeLayoutTypeAndDataType)
            {
                Log::debug("KernelGraph::AddLDSPadding: "
                           "Could not determine layout type and data type for LDS tag {}",
                           ldsTag);
                return;
            }

            auto upstreamTags
                = graph.coordinates.getNeighbours<GD::Upstream>(*flattenEdgeTag).to<std::vector>();
            auto downstreamTags
                = graph.coordinates.getNeighbours<GD::Downstream>(*tileEdgeTag).to<std::vector>();

            m_ldsTags[ldsTag] = LDSPaddingInfo{ldsTag,
                                               *flattenEdgeTag,
                                               *tileEdgeTag,
                                               {upstreamTags[0], upstreamTags[1]},
                                               {downstreamTags[0], downstreamTags[1]},
                                               maybeLayoutTypeAndDataType->second,
                                               maybeLayoutTypeAndDataType->first};
        }

        /**
         * @brief Commit LDS padding changes to the graph.
         *
         * This will change the upstream Flatten edge to a Join edge,
         * and the downstream Tile edge to a Split edge.
         */
        void AddLDSPaddingVisitor::commit(KernelGraph& graph)
        {
            for(auto& [ldsTag, info] : m_ldsTags)
            {
                uint paddingElements = getLDSPaddingElements(graph, info);

                Log::info("KernelGraph::AddLDSPadding: ldsTag {}, upstreamEdge {}, "
                          "downstreamEdge {}, paddingElements {}",
                          info.ldsTag,
                          info.upstreamEdge,
                          info.downstreamEdge,
                          paddingElements);

                if(paddingElements == 0)
                    continue;

                bool needsPropagateSize = matchTransforms(
                    graph, info.downstreamEdge, info.downstreamTags, info.upstreamTags);

                // Change upstream edge to Join
                graph.coordinates.setElement(info.upstreamEdge, Join());

                // Change downstream edge to Split
                graph.coordinates.setElement(info.downstreamEdge, Split());

                updateStrides(graph, info.upstreamTags, paddingElements);
                updateStrides(graph, info.downstreamTags, paddingElements);

                if(needsPropagateSize)
                {
                    propagateSize(graph, info.upstreamTags[0], GD::Downstream);
                    propagateSize(graph, info.downstreamTags[0], GD::Upstream);
                }
            }
        }

        KernelGraph AddLDSPadding::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddLDSPadding");
            auto graph   = original;
            auto visitor = AddLDSPaddingVisitor(m_params);
            for(auto ldsTag : graph.coordinates.getNodes<LDS>())
                visitor.stage(graph, ldsTag);
            visitor.commit(graph);
            return graph;
        }
    }
}
