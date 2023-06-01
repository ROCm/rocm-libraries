#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Updates dimension parameters within the coordinate
         * graph based on the command parameters.
         */
        class UpdateParameters : public GraphTransform
        {
        public:
            UpdateParameters(std::shared_ptr<CommandParameters> params)
                : m_params(params)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "UpdateParameters";
            }

        private:
            std::shared_ptr<CommandParameters> m_params;
        };
    }
}
