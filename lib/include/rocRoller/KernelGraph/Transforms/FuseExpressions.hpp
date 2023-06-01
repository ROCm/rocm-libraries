#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Performs the expression fusion transformation.
         *
         * Fuses neighbouring expressions where possible.
         */
        class FuseExpressions : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "FuseExpressions";
            }
        };
    }
}
