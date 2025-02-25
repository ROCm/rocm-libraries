#pragma once
#include <rocRoller/CommandSolution_fwd.hpp>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Performs the Loop Unrolling transformation.
         *
         * Unrolls every loop that does not have a previous iteration
         * dependency by a value of 2.
         */
        class UnrollLoops : public GraphTransform
        {
        public:
            UnrollLoops(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "UnrollLoops";
            }

        private:
            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };
    }
}
