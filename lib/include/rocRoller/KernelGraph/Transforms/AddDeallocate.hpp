#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Rewrite KernelGraph to add Deallocate operations.
         *
         * The control graph is analyzed to determine register
         * lifetimes.  Deallocate operations are added when registers
         * are no longer needed.
         */
        class AddDeallocate : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddDeallocate";
            }
        };
    }
}
