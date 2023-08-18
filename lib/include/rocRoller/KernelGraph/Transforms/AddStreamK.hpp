#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Flatten tile space and stream accumulation tiles.
         *
 	 * See `StreamKCoordinatetransformDesign`.
         *
         * The AddStreamK transformation is typically in matrix-matrix
         * multiply problems of the form D = A B where A and B have
         * been tiled with A: M x K tiles, and B: K x N tiles.  Here
         * the K tiles are the accumulation tiles.
         *
         * The `dims` parameter selects the M and N dimensions.  The
         * `topLoop` parameter selects the K dimension.
         *
         * The AddStreamK transform creates a flattened "global tile
         * space" from all of the M/N/K tiles.  The flattened M/N/K
         * global tile-space is distributed evenly among the CUs.
         * Each CU iterates over its portion of the flattened global
         * tile-space; with the K tiles iterated over in the
         * inner-most "streaming" loop.
         *
	 * The transformation is parameterised by:
	 *
	 * @param dims The sub-dimensions of dangling
	 * `MacroTileNumber`s that should be included in the streaming
	 * construct.
	 *
	 * @param tileNumberCoordSizes Sizes of `MacroTileNumber`s
	 * matched by `dims`.
	 *
	 * @param topLoop Which accumulation loop to stream.
         *
         * @param numCUs How many CUs/workgroups will be launched.
         */
        class AddStreamK : public GraphTransform
        {
        public:
            AddStreamK()                  = delete;
            AddStreamK(AddStreamK const&) = delete;

            AddStreamK(std::vector<int> const&   dims,
                       std::string const&        topLoop,
                       std::string const&        accumulatorLoop,
                       Expression::ExpressionPtr numCUs,
                       ContextPtr                context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;

        private:
            int addTileSpaceCT(KernelGraph&              graph,
                               bool                      forward,
                               Expression::ExpressionPtr numTotalTiles,
                               Expression::ExpressionPtr numTilesPerCU);

            void stage(KernelGraph const& graph);
            void setupArguments();
            void commit(KernelGraph& graph);

            ContextPtr m_context;

            // Location
            std::vector<int> m_dimensions;
            std::string      m_topLoop;
            std::string      m_accumulatorLoop;

            int m_topLoopOp;
            int m_accumulatorLoopOp;
            int m_accumulatorCoord;

            // Kernel arguments
            std::vector<Expression::ExpressionPtr> m_numTiles, m_numTileArgExprs;
            Expression::ExpressionPtr              m_numCUs, m_numTilesPerCU;

            // Staged MacroTileNumber coordinates
            //
            // Mapping: dimension -> set of MacroTileNumber coordinates
            std::map<int, std::unordered_set<int>> m_tileNumberCoords;
        };
    }
}
