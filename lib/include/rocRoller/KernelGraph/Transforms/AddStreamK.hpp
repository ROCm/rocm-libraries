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
         * The AddStreamK transformation is typically applied in
         * matrix-matrix multiply problems of the form D = A B where A
         * and B have been tiled with A: M x K tiles, and B: K x N
         * tiles.  Here the K tiles are the accumulation tiles.
         *
         * The `dims` parameter selects the free (M and N) dimensions.
         * The `topLoop` parameter selects the accumulation loop
         * (which was most likely created during the
         * LowerTensorContraction transformation).
         *
         * The AddStreamK transform creates a flattened "global tile
         * space" from all of the M/N/K tiles.  The flattened M/N/K
         * global tile-space is distributed evenly among the WGs.
         * Each WG iterates over its portion of the flattened global
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
         * @param topLoop Which loop to insert the local tile-loop
         * above.
         *
         * @param accmulatorLoop Which accumulation loop to stream.
         *
         * @param numWGs How many workgroups will be launched.
         */
        class AddStreamK : public GraphTransform
        {
        public:
            AddStreamK()                  = delete;
            AddStreamK(AddStreamK const&) = delete;

            AddStreamK(std::vector<int> const&   dims,
                       std::string const&        topLoop,
                       std::string const&        accumulatorLoop,
                       Expression::ExpressionPtr numWGs,
                       ContextPtr                context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;

        private:
            int addTileSpaceCT(KernelGraph&              graph,
                               bool                      forward,
                               Expression::ExpressionPtr numTotalTiles,
                               Expression::ExpressionPtr numTilesPerWG);

            void stage(KernelGraph const& graph);
            void setupArguments();
            void commit(KernelGraph& graph);

            ContextPtr m_context;

            /**
             * The sub-dimensions of dangling `MacroTileNumber`s that
             * should be included in the streaming construct.
             */
            std::vector<int> m_dimensionIndices;

            /**
             * Name of the loop to insert the local tile-loop above.
             */
            std::string m_topLoop;

            /**
             * Name of the accumulator (K) loop.
             */
            std::string m_accumulatorLoop;

            // Kernel arguments
            std::vector<Expression::ExpressionPtr> m_numTiles, m_numTileArgExprs;
            Expression::ExpressionPtr              m_numWGs, m_numTilesPerWG;

            // Staged MacroTileNumber coordinates
            //
            // Mapping: dimension -> set of MacroTileNumber coordinates
            std::map<int, std::unordered_set<int>> m_tileNumberCoords;

            // Internal
            int m_accumulatorCoord; /// Coordinate dimension of the K loop.
            int m_accumulatorLoopOp; /// Control node of the accumulator loop.
            int m_accumulatorTile; /// Coordinate dimension of tile into which the K loop accumulated
            int m_topLoopOp; /// Control node of the top loop.

            std::unordered_set<int> m_usesAccumulatorTile; /// Set of control nodes, after
                /// the accumulator loop, that used the accumulator
                /// tile.
        };
    }
}
