#pragma once

#include <vector>

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Connect unbound (leaf) MacroTileNumber coordinates
         * to a linearised ForLoop+Workgroup coordinate pair.
         *
         * This transform searches for MacroTileNumber coordinates
         * that are leafs (don't have outgoing/incoming edges), and
         * transforms them into a Workgroup+ForLoop pair.  That is,
         * tiles are iterated over using a combination of an implicit
         * workgroup and an explicit for-loop.
	 *
	 * For example, suppose we have a graph with dangling
	 * MacroTileNumbers with sub-dimensions 0 and 1; and we want
	 * to iterate over those using a combined Workgroup+ForLoop
	 * coordinate pair.
	 *
	 * We create:
	 *
	 *    WG = Workgroup()
	 *    FL = ForLoop()
	 *    LinearTileIterationCoord = Flatten(WG, FL)
	 *
	 *    WG0 = MacroTileNumber(SubDimension=0)
	 *    WG1 = MacroTileNumber(SubDimension=1)
	 *
	 *    WG0, WG1 = Tile(LinearTileIterationCoord)
	 *
	 * Then we connect all pre-exisitng leaf-MacroTileNumbers to
	 * either WG0 or WG1 (matching on the SubDimension).
	 *
	 * The transformation is parameterised by:
	 *
	 * @param dims The sub-dimensions that should be iterated over
	 * using the new Workgroup+ForLoop pair.
	 *
	 * @param tileNumberCoordSizes How many tiles are in each
	 * sud-dimension.
	 *
	 * @param numIteratedTiles How many tiles are iterated over
	 * using a ForLoop.
	 *
	 * @param topLoop Where to insert the new ForLoopOp in the
	 * control graph.
	 *
	 * Note that the kernel should be launched so that the number
	 * of Workgroups matches:
	 *
	 *    Launched workgroups = product(tileNumberCoordSizes) * numIteratedTiles
         */
        class LoopOverTileNumbers : public GraphTransform
        {
        public:
            LoopOverTileNumbers() = delete;
            LoopOverTileNumbers(std::vector<int> const&  dims,
                                std::vector<uint> const& tileNumberCoordSizes,
                                uint                     numIteratedTiles,
                                std::string const&       topLoop,
                                ContextPtr               context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;

        private:
            void stage(KernelGraph const& graph);
            void commit(KernelGraph& graph);

            ContextPtr m_context;

            // Location
            std::vector<int> m_dimensions;
            std::string      m_topLoop;

            int m_topLoopOp;

            // Parameters
            std::vector<Expression::ExpressionPtr> m_tileNumberCoordSizes;
            Expression::ExpressionPtr              m_numIteratedTileNumbers;

            // Staged MacroTileNumber coodinates
            //
            // Mapping: dimension -> set of MacroTileNumber coordinates
            std::map<int, std::unordered_set<int>> m_tileNumberCoords;
        };
    }
}
