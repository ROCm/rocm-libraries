// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>
#include <set>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Hoist loop-invariant Assign operations outside of loops.
         * 
         * This transformation identifies Assign operations within loop bodies
         * that do not depend on loop variables and moves them to execute
         * before the loop, improving performance by avoiding redundant
         * computations.
         */
        class HoistLoopInvariant : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;
        };
    }
}
