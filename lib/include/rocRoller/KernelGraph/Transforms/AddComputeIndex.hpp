#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Rewrite KernelGraph to add ComputeIndex operations.
         *
         * The ComputeIndex operations are used so that indices do not
         * need to be completely recalculated everytime when iterating
         * through a tile of data.
         *
         * ComputeIndex operations are added before For Loops and
         * calculate the first index in the loop.
         *
         * A new ForLoopIncrement is added to the loop as well to
         * increment the index by the stride amount.
         *
         * Offset, Stride and Buffer edges are added to the DataFlow
         * portion of the Coordinate graph to keep track of the data
         * needed to perform the operations.
         */
        class AddComputeIndex : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddComputeIndex";
            }
        };
    }
}
