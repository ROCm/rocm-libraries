#pragma once
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

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
            CleanArguments(std::shared_ptr<AssemblyKernel> kernel)
                : m_kernel(kernel)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "CleanArguments";
            }

        private:
            std::shared_ptr<AssemblyKernel> m_kernel;
        };
    }
}
