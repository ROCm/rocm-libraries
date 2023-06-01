#pragma once
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Rewrite KernelGraph to add LDS operations for
         * loading/storing data.
         *
         * Modifies the coordinate and control graphs to add LDS
         * information.
         */
        class AddLDS : public GraphTransform
        {
        public:
            AddLDS(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddLDS";
            }

        private:
            ContextPtr m_context;
        };
    }
}
