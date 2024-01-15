#pragma once
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Removes all CommandArgruments found within an
         * expression with the appropriate AssemblyKernel Argument.
         */
        class CleanArguments : public GraphTransform
        {
        public:
            CleanArguments(ContextPtr context, CommandPtr command)
                : m_context(context)
                , m_command(command)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "CleanArguments";
            }

        private:
            ContextPtr m_context;
            CommandPtr m_command;
        };
    }
}
