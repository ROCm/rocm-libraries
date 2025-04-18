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
@class AddLDS
@brief Add load/store through LDS to the graph; and prefetching.

# Overview

Load/store operations inside loops, and tagged with MemoryType LDS
or WAVE_LDS, are transformed.

An entire tile is loaded once per loop iteration (which may be
unrolled) into LDS.  Subsequent loads in the loop read from LDS.

*/

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDS.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CF = rocRoller::KernelGraph::ControlGraph;
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        using GD = rocRoller::Graph::Direction;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;
        using namespace Register;

        template <Graph::Direction Dir>
        void insertInstead(rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph& coordinates,
                           int                                                       newTag,
                           int                                                       oldTag)
        {
            auto edgeTags = coordinates.getNeighbours<Dir>(oldTag).template to<std::vector>();
            for(auto edgeTag : edgeTags)
            {
                auto edge = coordinates.getElement(edgeTag);

                auto upDirTags = coordinates.getNeighbours<Graph::opposite(Dir)>(edgeTag)
                                     .template to<std::vector>();
                auto downDirTags
                    = coordinates.getNeighbours<Dir>(edgeTag).template to<std::vector>();

                std::replace(upDirTags.begin(), upDirTags.end(), oldTag, newTag);

                coordinates.deleteElement(edgeTag);
                if constexpr(Dir == Graph::Direction::Upstream)
                {
                    coordinates.addElement(edge, downDirTags, upDirTags);
                }
                else
                {
                    coordinates.addElement(edge, upDirTags, downDirTags);
                }
            }
        }

        /**
         * Add LDS transformer.
         *
         * Splits LoadTiled operations into:
         * - LoadTiled
         * - StoreLDSTile
         * - LoadLDSTile
         *
         * Similarly for StoreTiled operations.
         */
        struct AddLDSVisitor
        {
            AddLDSVisitor(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            void stage(KernelGraph const&, int);
            void commit(KernelGraph&);

        private:
            std::set<int> m_operations;

            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };

        KernelGraph AddLDS::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddLDS");

            auto graph = original;

            auto visitor = AddLDSVisitor(m_params, m_context);
            for(auto const& loadTag : graph.control.getNodes<LoadTiled>())
                visitor.stage(graph, loadTag);
            for(auto const& storeTag : graph.control.getNodes<StoreTiled>())
                visitor.stage(graph, storeTag);

            visitor.commit(graph);

            return graph;
        }

        ConstraintStatus NoLDSTiles(const KernelGraph& graph)
        {
            ConstraintStatus retval;
            for(auto tag : graph.coordinates.getNodes<MacroTile>())
            {
                auto tile = *graph.coordinates.get<MacroTile>(tag);
                if(tile.memoryType == MemoryType::LDS || tile.memoryType == MemoryType::WAVE_LDS
                   || tile.memoryType == MemoryType::WAVE_Direct2LDS)
                {
                    retval.combine(false, concatenate("Tile has LDS memory type: ", tag));
                }
            }
            return retval;
        }

        std::vector<GraphConstraint> AddLDS::postConstraints() const
        {
            return {NoLDSTiles};
        }

        void AddLDSVisitor::stage(KernelGraph const& k, int opTag)
        {
            auto [userTag, user] = k.getDimension<User>(opTag);
            auto [tileTag, tile] = k.getDimension<MacroTile>(opTag);

            if(!(tile.memoryType == MemoryType::WAVE_LDS || tile.memoryType == MemoryType::LDS
                 || tile.memoryType == MemoryType::WAVE_Direct2LDS))
                return;

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDS()::stage({}): User {}, MacroTile {}", opTag, userTag, tileTag);

            m_operations.insert(opTag);
        }

        void AddLDSVisitor::commit(KernelGraph& k)
        {
            //
            // Commit: Create operations and DataFlow
            //
            for(auto opTag : m_operations)
            {
                Log::debug("KernelGraph::AddLDS()::commit({})", opTag);

                auto isLoad       = k.control.get<LoadTiled>(opTag).has_value();
                auto tileTag      = k.mapper.get<MacroTile>(opTag);
                auto userTag      = k.mapper.get<User>(opTag);
                auto varType      = getVariableType(k, opTag);
                auto tile         = k.coordinates.getNode<MacroTile>(tileTag);
                auto isDirect2LDS = tile.memoryType == MemoryType::WAVE_Direct2LDS;

                // TODO: enable SwizzleScale when store D via LDS
                auto isStoreD = tile.layoutType == LayoutType::MATRIX_ACCUMULATOR;
                if(!isLoad && isStoreD)
                    AssertFatal(!m_params->swizzleScale,
                                "Store D via LDS is not supported by SwizzleScale");

                // Create new coordinates
                auto              ldsTag      = k.coordinates.addElement(LDS(isDirect2LDS));
                std::vector<uint> jammedTiles = {1, 1};
                bool              splitStore  = false;

                if(!isLoad)
                {
                    // For StoreTiled operations, the waves are
                    // "serialized" in OrderEpilogueBlocks, so we can
                    // use a smaller internal tile.
                    jammedTiles = m_params->getWaveTilesPerWavefront();
                    splitStore  = m_params->getSplitStoreTileIntoWaveBlocks();
                }

                auto internalTag = createInternalTile(
                    k, varType, tileTag, jammedTiles, splitStore, m_params, m_context);

                // Connect coordinates with DataFlow edges
                insertInstead<Graph::Direction::Downstream>(k.coordinates, internalTag, tileTag);
                k.coordinates.addElement(DataFlow(), {internalTag}, {ldsTag});
                k.coordinates.addElement(DataFlow(), {ldsTag}, {tileTag});

                // Create new operations and update old operation
                bool isTransposedTile = false;
                if(isLoad)
                {
                    auto loadTile    = k.control.get<LoadTiled>(opTag).value();
                    isTransposedTile = loadTile.isTransposedTile;
                }
                auto loadLDSOp  = k.control.addElement(LoadLDSTile(varType, isTransposedTile));
                auto storeLDSOp = k.control.addElement(StoreLDSTile(varType.dataType));

                // Update tile
                if(isDirect2LDS)
                    tile.memoryType = MemoryType::WAVE;
                if(tile.memoryType == MemoryType::WAVE_LDS)
                    tile.memoryType = MemoryType::WAVE;
                if(tile.memoryType == MemoryType::LDS)
                    tile.memoryType = MemoryType::VGPR;
                k.coordinates.setElement(tileTag, tile);

                if(!isLoad)
                {
                    // For StoreTiled operations, the waves are
                    // "serialized" in OrderEpilogueBlocks, so we can
                    // use a smaller tile size.
                    tile.sizes[0] /= m_params->getWaveTilesPerWavefront()[0];
                    tile.sizes[1] /= m_params->getWaveTilesPerWavefront()[1];
                    k.coordinates.setElement(tileTag, tile);
                }

                // Update connections and connect operations
                k.control.addElement(Sequence(), {storeLDSOp}, {loadLDSOp});

                k.mapper.purge(opTag);
                k.mapper.connect<User>(opTag, userTag);
                k.mapper.connect<LDS>(storeLDSOp, ldsTag);
                k.mapper.connect<User>(loadLDSOp, userTag); // For F6 Padding
                k.mapper.connect<LDS>(loadLDSOp, ldsTag);

                if(isDirect2LDS)
                    k.mapper.connect<LDS>(opTag, ldsTag);

                if(isLoad)
                {
                    insertAfter(k, opTag, storeLDSOp, loadLDSOp);

                    k.mapper.connect<MacroTile>(opTag, internalTag);
                    k.mapper.connect<MacroTile>(storeLDSOp, internalTag);
                    k.mapper.connect<MacroTile>(loadLDSOp, tileTag);
                }
                else
                {
                    insertBefore(k, opTag, storeLDSOp, loadLDSOp);

                    k.mapper.connect<MacroTile>(storeLDSOp, tileTag);
                    k.mapper.connect<MacroTile>(loadLDSOp, internalTag);
                    k.mapper.connect<MacroTile>(opTag, internalTag);
                }
            }
        }

    }
}
