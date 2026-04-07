// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/ExecuteMaskGenerator.hpp>

#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;

        ExecuteMaskGenerator::ExecuteMaskGenerator(KernelGraphPtr graph, ContextPtr context)
            : m_context(context)
            , m_graph(std::move(graph))
            , m_fastArith(Expression::FastArithmetic(context))
        {
        }

        Generator<Instruction> ExecuteMaskGenerator::genExec(
            int                                                  tag,
            ConditionalOp const&                                 op,
            std::function<Generator<Instruction>(std::set<int>)> generateFn)
        {
            auto const wavefrontSize = m_context->kernel()->wavefront_size();
            AssertFatal(wavefrontSize == 32 || wavefrontSize == 64, ShowValue(wavefrontSize));

            auto expr = m_fastArith(op.condition);
            auto vcc  = m_context->brancher()->resultRegister(expr);

            auto regType = vcc->regType();
            auto varType = vcc->variableType();
            AssertFatal((regType == Register::Type::VCC && varType == DataType::Bool64
                         && wavefrontSize == 64)
                            || (regType == Register::Type::VCC_LO && varType == DataType::Bool32
                                && wavefrontSize == 32),
                        ShowValue(regType),
                        ShowValue(varType),
                        ShowValue(wavefrontSize));

            co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock for Conditional EXEC");
            // code-gen the if-condition
            co_yield Expression::generate(vcc, expr, m_context);

            Register::ValuePtr sgpr;

            // s_and_saveexec_b{32,64}: Calculate bitwise AND on the scalar input and the EXEC mask,
            // store the calculated result into the EXEC mask,
            // set SCC iff the calculated result is nonzero and
            // store the original value of the EXEC mask into the scalar destination register.
            // The original EXEC mask is saved to the destination SGPRs before the
            // bitwise operation is performed.
            if(wavefrontSize == 64)
            {
                sgpr = std::make_shared<Register::Value>(
                    m_context,
                    Register::Type::Scalar,
                    DataType::Bool64,
                    1,
                    Register::AllocationOptions::FullyContiguous());
                co_yield_(Instruction("s_and_saveexec_b64", {sgpr}, {vcc}, {}, ""));
            }
            else
            {
                sgpr = std::make_shared<Register::Value>(
                    m_context,
                    Register::Type::Scalar,
                    DataType::Bool32,
                    1,
                    Register::AllocationOptions::FullyContiguous());
                co_yield_(Instruction("s_and_saveexec_b32", {sgpr}, {vcc}, {}, ""));
            }
            auto trueBody = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
            co_yield generateFn(trueBody);

            auto elseBody = m_graph->control.getOutputNodeIndices<Else>(tag).to<std::set>();
            if(!elseBody.empty())
            {
                // restore the original EXEC mask from the scalar destination register.
                auto EXEC = m_context->getEXEC();
                co_yield m_context->copier()->copy(EXEC, sgpr, "restore the EXEC mask");

                // s_andn1_saveexec_b{32,64}: Calculate bitwise AND on the EXEC mask and
                // the negation of the scalar input,
                // store the calculated result into the EXEC mask,
                // set SCC iff the calculated result is nonzero and
                // store the original value of the EXEC mask into the scalar destination register.
                if(wavefrontSize == 64)
                {
                    co_yield_(Instruction("s_andn1_saveexec_b64", {sgpr}, {vcc}, {}, ""));
                }
                else
                {
                    co_yield_(Instruction("s_andn1_saveexec_b32", {sgpr}, {vcc}, {}, ""));
                }
                co_yield generateFn(elseBody);
            }

            // restore the original EXEC mask from the scalar destination register.
            auto EXEC = m_context->getEXEC();
            co_yield m_context->copier()->copy(EXEC, sgpr, "restore the EXEC mask");
            co_yield Instruction::Unlock("Unlock Conditional EXEC");
        }

        Generator<Instruction> ExecuteMaskGenerator::genBranchAndExec(
            int                                                  tag,
            ConditionalOp const&                                 op,
            std::function<Generator<Instruction>(std::set<int>)> generateFn)
        {
            auto const wavefrontSize = m_context->kernel()->wavefront_size();
            AssertFatal(wavefrontSize == 32 || wavefrontSize == 64, ShowValue(wavefrontSize));

            auto expr = m_fastArith(op.condition);
            auto vcc  = m_context->brancher()->resultRegister(expr);

            auto regType = vcc->regType();
            auto varType = vcc->variableType();
            AssertFatal((regType == Register::Type::VCC && varType == DataType::Bool64
                         && wavefrontSize == 64)
                            || (regType == Register::Type::VCC_LO && varType == DataType::Bool32
                                && wavefrontSize == 32),
                        ShowValue(regType),
                        ShowValue(varType),
                        ShowValue(wavefrontSize));

            auto elseLabel = m_context->labelAllocator()->label(
                fmt::format("ELSE_Conditional_EXECZ_{}", op.conditionName, tag));
            auto exitLabel = m_context->labelAllocator()->label(
                fmt::format("EXIT_Conditional_EXECZ_{}", op.conditionName, tag));

            co_yield Instruction::Lock(Scheduling::Dependency::Branch,
                                       "Lock for Conditional EXECZ");
            // code-gen the if-condition
            co_yield Expression::generate(vcc, expr, m_context);

            Register::ValuePtr sgpr;

            // s_and_saveexec_b{32,64}: Calculate bitwise AND on the scalar input and the EXEC mask,
            // store the calculated result into the EXEC mask,
            // set SCC iff the calculated result is nonzero and
            // store the original value of the EXEC mask into the scalar destination register.
            // The original EXEC mask is saved to the destination SGPRs before the
            // bitwise operation is performed.
            if(wavefrontSize == 64)
            {
                sgpr = std::make_shared<Register::Value>(
                    m_context,
                    Register::Type::Scalar,
                    DataType::Bool64,
                    1,
                    Register::AllocationOptions::FullyContiguous());
                co_yield_(Instruction("s_and_saveexec_b64", {sgpr}, {vcc}, {}, ""));
            }
            else
            {
                sgpr = std::make_shared<Register::Value>(
                    m_context,
                    Register::Type::Scalar,
                    DataType::Bool32,
                    1,
                    Register::AllocationOptions::FullyContiguous());
                co_yield_(Instruction("s_and_saveexec_b32", {sgpr}, {vcc}, {}, ""));
            }

            auto EXECZ = m_context->getEXECZ();
            // if execz == 1 (set), it means EXEC == 0 i.e. the entire execute mask is zero,
            // then skip the then branch and jump to the else branch.
            co_yield m_context->brancher()->branchIfNonZero(
                elseLabel,
                EXECZ,
                concatenate("If EXECZ is set(1), jump to ", elseLabel->toString()));
            auto trueBody = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
            co_yield generateFn(trueBody);
            co_yield m_context->brancher()->branch(
                exitLabel, concatenate("THEN: Done, jump to ", exitLabel->toString()));

            co_yield Instruction::Label(elseLabel);
            auto elseBody = m_graph->control.getOutputNodeIndices<Else>(tag).to<std::set>();
            if(!elseBody.empty())
            {
                // restore the original EXEC mask from the scalar destination register.
                auto EXEC = m_context->getEXEC();
                co_yield m_context->copier()->copy(EXEC, sgpr, "restore the EXEC mask");

                // s_andn1_saveexec_b{32,64}: Calculate bitwise AND on the EXEC mask and
                // the negation of the scalar input,
                // store the calculated result into the EXEC mask,
                // set SCC iff the calculated result is nonzero and
                // store the original value of the EXEC mask into the scalar destination register.
                if(wavefrontSize == 64)
                {
                    co_yield_(Instruction("s_andn1_saveexec_b64", {sgpr}, {vcc}, {}, ""));
                }
                else
                {
                    co_yield_(Instruction("s_andn1_saveexec_b32", {sgpr}, {vcc}, {}, ""));
                }

                auto EXECZ = m_context->getEXECZ();
                // if execz == 1 (set), it means EXEC == 0 i.e. the entire execute mask is zero,
                // then skip the else branch and jump to the exit.
                co_yield m_context->brancher()->branchIfNonZero(
                    exitLabel,
                    EXECZ,
                    concatenate("If EXECZ is set(1), jump to ", exitLabel->toString()));
                co_yield generateFn(elseBody);
            }

            co_yield Instruction::Label(exitLabel);

            // restore the original EXEC mask from the scalar destination register.
            auto EXEC = m_context->getEXEC();
            co_yield m_context->copier()->copy(EXEC, sgpr, "restore the EXEC mask");

            co_yield Instruction::Unlock("Unlock Conditional EXECZ");
        }
    }
}
