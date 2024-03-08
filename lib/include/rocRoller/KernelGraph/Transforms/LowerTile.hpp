#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief After LowerTile is applied, there should be no ConstructMacroTile or
         *        DestructMacroTile edges in the coordinate graph.
         */
        ConstraintStatus NoConstructDestructMT(const KernelGraph& k);

        /**
         * @brief Rewrite KernelGraph to distribute tiled packets onto
         * GPU.
         *
         * When loading tiles, the tile size, storage location (eg,
         * VGPRs vs LDS), and affinity (eg, owned by a thread vs
         * workgroup) of each tile is specified by the destination
         * tile.  These attributes do not need to be known at
         * translation time.  To specify these attributes, call
         * `setDimension`.
         */
        class LowerTile : public GraphTransform
        {
        public:
            LowerTile(std::shared_ptr<CommandParameters> params, ContextPtr context)

                : m_params(params)
                , m_context(context)
                , m_splitStoreTileIntoWaveBlocks(true)
            {
            }

            LowerTile(std::shared_ptr<CommandParameters> params,
                      ContextPtr                         context,
                      bool                               splitStoreTileIntoWaveBlocks)
                : m_params(params)
                , m_context(context)
                , m_splitStoreTileIntoWaveBlocks(splitStoreTileIntoWaveBlocks)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "LowerTile";
            }

            std::vector<GraphConstraint> postConstraints() const override
            {
                return {&NoConstructDestructMT};
            }

        private:
            std::shared_ptr<CommandParameters> m_params;
            ContextPtr                         m_context;
            bool                               m_splitStoreTileIntoWaveBlocks;
        };
    }
}
