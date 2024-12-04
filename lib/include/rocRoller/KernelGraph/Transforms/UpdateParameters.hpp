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
            UpdateParameters(CommandParametersPtr params)
                : m_params(params)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "UpdateParameters";
            }

        private:
            CommandParametersPtr m_params;
        };

        class UpdateWavefrontParameters : public GraphTransform
        {
        public:
            UpdateWavefrontParameters(CommandParametersPtr params)
                : m_params(params)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "UpdateWavefrontParameters";
            }

        private:
            CommandParametersPtr m_params;
        };

        class SetWorkitemCount : public GraphTransform
        {
        public:
            SetWorkitemCount(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "SetWorkitemCount";
            }

        private:
            ContextPtr m_context;
        };

    }
}
