// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <set>

#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Generates EXEC-mask-based conditional code for ConditionalOp nodes.
         *
         * Handles OpMode::Exec and OpMode::BranchAndExec cases,
         * including saving/restoring the EXEC mask and branching on EXECZ.
         */
        class ExecuteMaskGenerator
        {
        public:
            ExecuteMaskGenerator(KernelGraphPtr graph, ContextPtr context);

            /**
             * @brief Generate instructions for OpMode::Exec conditional.
             *
             * Saves the EXEC mask, AND-masks with the condition VCC,
             * generates the true body, optionally generates the else body
             * with the complementary mask, then restores the EXEC mask.
             *
             * @param tag       Control graph node tag for the ConditionalOp.
             * @param op        The ConditionalOp node.
             * @param generateFn Callback to generate instructions for a set of body nodes.
             */
            Generator<Instruction>
                genExec(int                                                  tag,
                        ControlGraph::ConditionalOp const&                   op,
                        std::function<Generator<Instruction>(std::set<int>)> generateFn);

            /**
             * @brief Generate instructions for OpMode::BranchAndExec conditional.
             *
             * Like genExec but additionally branches over the true body when EXECZ
             * is set (i.e. the entire EXEC mask is zero) and branches over the else
             * body similarly.
             *
             * @param tag       Control graph node tag for the ConditionalOp.
             * @param op        The ConditionalOp node.
             * @param generateFn Callback to generate instructions for a set of body nodes.
             */
            Generator<Instruction>
                genBranchAndExec(int                                                  tag,
                                 ControlGraph::ConditionalOp const&                   op,
                                 std::function<Generator<Instruction>(std::set<int>)> generateFn);

        private:
            ContextPtr     m_context;
            KernelGraphPtr m_graph;

            Expression::ExpressionTransducer m_fastArith;
        };
    }
}
