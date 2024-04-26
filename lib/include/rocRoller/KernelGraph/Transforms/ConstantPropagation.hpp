#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Propagate constant to prune unneeded Load and Assign operations. Only support for beta==0 for now.
         */
        class ConstantPropagation : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;
        };
    }
}
