// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <string>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Generates conditional code for ConditionalOp nodes.
         *
         * Handles ConditionalMode::Branch, ConditionalMode::Exec, and ConditionalMode::BranchAndExec cases.
         */
        class ConditionalGenerator
        {
        public:
            ConditionalGenerator(ContextPtr context);

            /**
             * @brief Generate instructions for a ConditionalOp node.
             *
             * Dispatches to the appropriate implementation based on @p mode:
             * - ConditionalMode::Branch: evaluates the condition, branches over the true body
             *   if false, generates the true body, branches to the bottom label, then
             *   optionally generates the else body.
             * - ConditionalMode::Exec or ConditionalMode::BranchAndExec: saves the EXEC mask,
             *   AND-masks with the condition VCC, generates the true body, optionally generates
             *   the else body with the complementary mask, then restores the EXEC mask.
             *   When mode is BranchAndExec, additionally branches over each body when EXECZ is
             *   set (i.e. the entire EXEC mask is zero).
             *
             * @param condition  The condition expression (already fast-arithmetic transformed).
             * @param labelBase  Base string for generated labels; should encode both the
             *                   conditionName and the control graph node tag to ensure uniqueness.
             * @param trueBodyFn Callback to generate instructions for the true (Body) nodes.
             * @param elseBodyFn Callback to generate instructions for the else (Else) nodes,
             *                   or empty if there is no else body.
             * @param mode       The ConditionalMode to use.
             */
            Generator<Instruction>
                genConditional(Expression::ExpressionPtr               condition,
                               std::string const&                      labelBase,
                               std::function<Generator<Instruction>()> trueBodyFn,
                               std::function<Generator<Instruction>()> elseBodyFn,
                               ControlGraph::ConditionalMode           mode);

        private:
            ContextPtr m_context;

            Generator<Instruction> genBranch(Expression::ExpressionPtr               condition,
                                             std::string const&                      labelBase,
                                             std::function<Generator<Instruction>()> trueBodyFn,
                                             std::function<Generator<Instruction>()> elseBodyFn);

            Generator<Instruction> genExec(Expression::ExpressionPtr               condition,
                                           std::string const&                      labelBase,
                                           std::function<Generator<Instruction>()> trueBodyFn,
                                           std::function<Generator<Instruction>()> elseBodyFn,
                                           ControlGraph::ConditionalMode           mode);
        };
    }
}
