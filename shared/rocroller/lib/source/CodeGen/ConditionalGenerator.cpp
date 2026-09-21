// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/ConditionalGenerator.hpp>

#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;

        static Register::ValuePtr MakeScalarBool(ContextPtr context, uint32_t wavefrontSize)
        {
            auto dataType = wavefrontSize == 64 ? DataType::Bool64 : DataType::Bool32;
            return std::make_shared<Register::Value>(
                context,
                Register::Type::Scalar,
                dataType,
                1,
                Register::AllocationOptions::FullyContiguous());
        }

        ConditionalGenerator::ConditionalGenerator(ContextPtr context)
            : m_context(context)
        {
        }

        Generator<Instruction>
            ConditionalGenerator::genConditional(Expression::ExpressionPtr               condition,
                                                 std::string const&                      labelBase,
                                                 std::function<Generator<Instruction>()> trueBodyFn,
                                                 std::function<Generator<Instruction>()> elseBodyFn,
                                                 ControlGraph::ConditionalMode           mode)
        {
            switch(mode)
            {
            case ControlGraph::ConditionalMode::Branch:
                co_yield genBranch(condition, labelBase, trueBodyFn, elseBodyFn);
                break;
            case ControlGraph::ConditionalMode::Exec:
            case ControlGraph::ConditionalMode::BranchAndExec:
                co_yield genExec(condition, labelBase, trueBodyFn, elseBodyFn, mode);
                break;
            default:
                Throw<FatalError>("Unhandled ConditionalMode: ", ShowValue(mode));
            }
        }

        Generator<Instruction>
            ConditionalGenerator::genBranch(Expression::ExpressionPtr               condition,
                                            std::string const&                      labelBase,
                                            std::function<Generator<Instruction>()> trueBodyFn,
                                            std::function<Generator<Instruction>()> elseBodyFn)
        {
            Log::debug("ConditionalGenerator::genBranch({})", labelBase);

            auto falseLabel
                = m_context->labelAllocator()->label(fmt::format("ConditionalFalse_{}", labelBase));
            auto botLabel = m_context->labelAllocator()->label(
                fmt::format("ConditionalBottom_{}", labelBase));

            co_yield Instruction::Lock(Scheduling::Dependency::Branch,
                                       "Lock for Conditional Branch");

            auto conditionResult = m_context->brancher()->resultRegister(condition);
            co_yield Expression::generate(conditionResult, condition, m_context);
            // -------------------------------------------------------------------------------
            // TODO: remove this once we better handle data-flow across branches
            {
                co_yield Instruction::Wait(
                    WaitCount::Zero(m_context->targetArchitecture(),
                                    "REMOVEME: Wait before branching into conditional label!"));
            }
            // -------------------------------------------------------------------------------

            co_yield m_context->brancher()->branchIfZero(
                falseLabel,
                conditionResult,
                concatenate("Condition: False, jump to ", falseLabel->toString()));
            co_yield trueBodyFn();
            // -------------------------------------------------------------------------------
            // TODO: remove this once we better handle data-flow across branches
            {
                co_yield Instruction::Wait(WaitCount::Zero(
                    m_context->targetArchitecture(), "REMOVEME: Wait before conditional label!"));
            }
            // -------------------------------------------------------------------------------
            co_yield m_context->brancher()->branch(
                botLabel, concatenate("Condition: Done, jump to ", botLabel->toString()));

            co_yield Instruction::Label(falseLabel);
            if(elseBodyFn)
            {
                co_yield elseBodyFn();
                // -------------------------------------------------------------------------------
                // TODO: remove this once we better handle data-flow across branches
                {
                    co_yield Instruction::Wait(
                        WaitCount::Zero(m_context->targetArchitecture(),
                                        "REMOVEME: Wait before conditional label!"));
                }
                // -------------------------------------------------------------------------------
            }
            co_yield Instruction::Label(botLabel);
            co_yield Instruction::Unlock("Unlock Conditional");
        }

        Generator<Instruction>
            ConditionalGenerator::genExec(Expression::ExpressionPtr               condition,
                                          std::string const&                      labelBase,
                                          std::function<Generator<Instruction>()> trueBodyFn,
                                          std::function<Generator<Instruction>()> elseBodyFn,
                                          ControlGraph::ConditionalMode           mode)
        {
            Log::debug("ConditionalGenerator::genExec({}, {})", labelBase, toString(mode));
            auto const wavefrontSize = m_context->kernel()->wavefront_size();
            AssertFatal(wavefrontSize == 32 || wavefrontSize == 64, ShowValue(wavefrontSize));

            auto vcc = m_context->brancher()->resultRegister(condition);

            auto regType = vcc->regType();
            auto varType = vcc->variableType();
            AssertFatal((regType == Register::Type::VCC && varType == DataType::Bool64
                         && wavefrontSize == 64)
                            || (regType == Register::Type::VCC_LO && varType == DataType::Bool32
                                && wavefrontSize == 32),
                        ShowValue(regType),
                        ShowValue(varType),
                        ShowValue(wavefrontSize));

            bool const branchAndExec = (mode == ControlGraph::ConditionalMode::BranchAndExec);

            Register::ValuePtr elseLabel, exitLabel;
            if(branchAndExec)
            {
                elseLabel = m_context->labelAllocator()->label(
                    fmt::format("ELSE_Conditional_EXECZ_{}", labelBase));
                exitLabel = m_context->labelAllocator()->label(
                    fmt::format("EXIT_Conditional_EXECZ_{}", labelBase));
            }

            co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock for Conditional EXEC");
            // code-gen the if-condition
            co_yield Expression::generate(vcc, condition, m_context);

            // s_and_saveexec_b{32,64}: Calculate bitwise AND on the scalar input and the EXEC mask,
            // store the calculated result into the EXEC mask,
            // set SCC iff the calculated result is nonzero and
            // store the original value of the EXEC mask into the scalar destination register.
            // The original EXEC mask is saved to the destination SGPRs before the
            // bitwise operation is performed.
            auto sgpr     = MakeScalarBool(m_context, wavefrontSize);
            auto saveExec = wavefrontSize == 64 ? "s_and_saveexec_b64" : "s_and_saveexec_b32";
            co_yield_(Instruction(saveExec, {sgpr}, {vcc}, {}, ""));

            // If there is an else body, save VCC into an SGPR now, before generating the
            // true body.  Nested conditionals inside the true body may overwrite VCC, making
            // the original condition unavailable for the s_andn1_saveexec transition.
            // This must also be before any branchAndExec branch, since the branch may skip
            // directly to the else label, bypassing the true body.
            Register::ValuePtr savedVcc;
            if(elseBodyFn)
            {
                savedVcc = MakeScalarBool(m_context, wavefrontSize);
                co_yield m_context->copier()->copy(
                    savedVcc, vcc, "Save condition for else transition");
            }

            if(branchAndExec)
            {
                auto EXECZ = m_context->getEXECZ();
                // -------------------------------------------------------------------------------
                // TODO: remove this once we better handle data-flow across branches
                {
                    co_yield Instruction::Wait(
                        WaitCount::Zero(m_context->targetArchitecture(),
                                        "REMOVEME: Wait before branching into EXEC conditional!"));
                }
                // -------------------------------------------------------------------------------
                // if execz == 1 (set), it means EXEC == 0 i.e. the entire execute mask is zero,
                // then skip the then branch and jump to the else branch.
                co_yield m_context->brancher()->branchIfNonZero(
                    elseLabel,
                    EXECZ,
                    concatenate("If EXECZ is set(1), jump to ", elseLabel->toString()));
            }

            co_yield trueBodyFn();

            if(branchAndExec)
            {
                // -------------------------------------------------------------------------------
                // TODO: remove this once we better handle data-flow across branches
                {
                    co_yield Instruction::Wait(
                        WaitCount::Zero(m_context->targetArchitecture(),
                                        "REMOVEME: Wait before EXEC conditional label!"));
                }
                // -------------------------------------------------------------------------------
                co_yield m_context->brancher()->branch(
                    exitLabel, concatenate("THEN: Done, jump to ", exitLabel->toString()));
                co_yield Instruction::Label(elseLabel);
            }

            if(elseBodyFn)
            {
                // restore the original EXEC mask from the scalar destination register.
                auto EXEC = m_context->getEXEC();
                co_yield m_context->copier()->copy(EXEC, sgpr, "restore the EXEC mask");

                // s_andn1_saveexec_b{32,64}: Calculate bitwise AND on the EXEC mask and
                // the negation of the scalar input,
                // store the calculated result into the EXEC mask,
                // set SCC iff the calculated result is nonzero and
                // store the original value of the EXEC mask into the scalar destination register.
                // Use savedVcc (not vcc) so that nested conditionals in the true body cannot
                // corrupt the condition used here.
                auto andn1SaveExec
                    = wavefrontSize == 64 ? "s_andn1_saveexec_b64" : "s_andn1_saveexec_b32";
                co_yield_(Instruction(andn1SaveExec, {sgpr}, {savedVcc}, {}, ""));

                if(branchAndExec)
                {
                    auto EXECZ = m_context->getEXECZ();
                    // -------------------------------------------------------------------------------
                    // TODO: remove this once we better handle data-flow across branches
                    {
                        co_yield Instruction::Wait(WaitCount::Zero(
                            m_context->targetArchitecture(),
                            "REMOVEME: Wait before branching past EXEC else body!"));
                    }
                    // -------------------------------------------------------------------------------
                    // if execz == 1 (set), it means EXEC == 0 i.e. the entire execute mask is zero,
                    // then skip the else branch and jump to the exit.
                    co_yield m_context->brancher()->branchIfNonZero(
                        exitLabel,
                        EXECZ,
                        concatenate("If EXECZ is set(1), jump to ", exitLabel->toString()));
                }

                co_yield elseBodyFn();
            }

            if(branchAndExec)
            {
                // -------------------------------------------------------------------------------
                // TODO: remove this once we better handle data-flow across branches
                {
                    co_yield Instruction::Wait(
                        WaitCount::Zero(m_context->targetArchitecture(),
                                        "REMOVEME: Wait before EXEC conditional exit label!"));
                }
                // -------------------------------------------------------------------------------
                co_yield Instruction::Label(exitLabel);
            }

            // restore the original EXEC mask from the scalar destination register.
            auto EXEC = m_context->getEXEC();
            co_yield m_context->copier()->copy(EXEC, sgpr, "restore the EXEC mask");
            co_yield Instruction::Unlock("Unlock Conditional EXEC");
        }
    }
}
