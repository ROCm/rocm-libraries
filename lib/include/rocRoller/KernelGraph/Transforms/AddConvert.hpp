#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Add required convert nodes to the control graph.
         *
         * Finds any multiply node that might require converting the
         * datatype of the arguments to the multiply and adds control
         * nodes that will perform the appropriate type conversion.
         */
        class AddConvert : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddConvert";
            }
        };
    }
}
