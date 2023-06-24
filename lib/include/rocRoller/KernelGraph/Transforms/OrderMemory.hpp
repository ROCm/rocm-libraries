#pragma once
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief After OrderMemory, there should be no ambiguously ordered memory nodes of the same type.
         */
        ConstraintStatus NoAmbiguousNodes(const KernelGraph& k);

        /**
         * @brief Order ambiguous memory operations in the control graph.
         */
        class OrderMemory : public GraphTransform
        {
        public:
            OrderMemory() {}

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "OrderMemory";
            }

            std::vector<GraphConstraint> postConstraints() const
            {
                return {&NoAmbiguousNodes};
            }
        };
    }
}
