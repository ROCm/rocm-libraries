#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Merge the LoadTiled and StoreLDSTile operation to enable Direct2LDS
         *
     * @ingroup Transformations
    */
        class AddDirect2LDS : public GraphTransform
        {
        public:
            AddDirect2LDS(ContextPtr context, CommandParametersPtr params)
                : m_context(context)
                , m_params(params)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;

            std::string name() const override
            {
                return "AddDirect2LDS";
            }

        private:
            ContextPtr           m_context;
            CommandParametersPtr m_params;
        };
    }
}
