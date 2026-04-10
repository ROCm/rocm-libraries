// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <string>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Generates conditional code for ConditionalOp nodes.
         *
         * Handles OpMode::Branch, OpMode::Exec, and OpMode::BranchAndExec cases.
         */
        class ConditionalGenerator
        {
        public:
            ConditionalGenerator(ContextPtr context);

            /**
             * @brief Generate instructions for OpMode::Branch conditional.
             *
             * Evaluates the condition, branches over the true body if false,
             * generates the true body, branches to the bottom label, then
             * optionally generates the else body.
             *
             * @param condition     The condition expression (already fast-arithmetic transformed).
             * @param conditionName Name used in generated labels.
             * @param trueBodyFn    Callback to generate instructions for the true (Body) nodes.
             * @param elseBodyFn    Callback to generate instructions for the else (Else) nodes,
             *                      or empty if there is no else body.
             */
            Generator<Instruction> genBranch(Expression::ExpressionPtr               condition,
                                             std::string const&                      conditionName,
                                             std::function<Generator<Instruction>()> trueBodyFn,
                                             std::function<Generator<Instruction>()> elseBodyFn);

            /**
             * @brief Generate instructions for OpMode::Exec conditional.
             *
             * Saves the EXEC mask, AND-masks with the condition VCC,
             * generates the true body, optionally generates the else body
             * with the complementary mask, then restores the EXEC mask.
             *
             * @param condition     The condition expression.
             * @param conditionName Name used in generated labels.
             * @param trueBodyFn    Callback to generate instructions for the true (Body) nodes.
             * @param elseBodyFn    Callback to generate instructions for the else (Else) nodes,
             *                      or empty if there is no else body.
             */
            Generator<Instruction> genExec(Expression::ExpressionPtr               condition,
                                           std::string const&                      conditionName,
                                           std::function<Generator<Instruction>()> trueBodyFn,
                                           std::function<Generator<Instruction>()> elseBodyFn);

            /**
             * @brief Generate instructions for OpMode::BranchAndExec conditional.
             *
             * Like genExec but additionally branches over the true body when EXECZ
             * is set (i.e. the entire EXEC mask is zero) and branches over the else
             * body similarly.
             *
             * @param condition     The condition expression.
             * @param conditionName Name used in generated branch labels.
             * @param trueBodyFn    Callback to generate instructions for the true (Body) nodes.
             * @param elseBodyFn    Callback to generate instructions for the else (Else) nodes,
             *                      or empty if there is no else body.
             */
            Generator<Instruction>
                genBranchAndExec(Expression::ExpressionPtr               condition,
                                 std::string const&                      conditionName,
                                 std::function<Generator<Instruction>()> trueBodyFn,
                                 std::function<Generator<Instruction>()> elseBodyFn);

        private:
            ContextPtr m_context;
        };
    }
}
