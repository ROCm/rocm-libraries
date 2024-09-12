#pragma once
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Only SetCoordinate, LoadTiled, and Multiply nodes and Body and Sequence edges are allowed in the ForLoop.
         *
         * If there are ever any other nodes/edges, they will need to be handled.
         */
        ConstraintStatus AcceptablePrefetchNodes(const KernelGraph& k);

        /**
         * @brief Rewrite KernelGraph to add LDS operations for
         * loading/storing data.
         *
         * Modifies the coordinate and control graphs to add LDS
         * information.
	 *
	 * @ingroup Transformations
         */
        class AddLDS : public GraphTransform
        {
        public:
            AddLDS(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddLDS";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {&AcceptablePrefetchNodes};
            }

        private:
            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };
    }
}
