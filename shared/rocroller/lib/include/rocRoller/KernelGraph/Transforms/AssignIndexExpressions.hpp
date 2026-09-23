// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Rewrite KernelGraph to add index computation operations.
         *
         * This transform adds operations to compute memory offsets and
         * strides for load/store operations, so that indices do not
         * need to be completely recalculated every time when iterating
         * through a tile of data.
         *
         * Index computation operations are added before For Loops and
         * calculate the first index in the loop.
         *
         * A new ForLoopIncrement is added to the loop as well to
         * increment the index by the stride amount.
         *
         * Offset, Stride and Buffer edges are added to the DataFlow
         * portion of the Coordinate graph to keep track of the data
         * needed to perform the operations.
         */
        class AssignIndexExpressions : public GraphTransform
        {
        public:
            AssignIndexExpressions(ContextPtr context, CommandPtr command)
                : m_context(context)
                , m_command(command)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AssignIndexExpressions";
            }

        private:
            ContextPtr m_context;
            CommandPtr m_command;
        };

    }
}
