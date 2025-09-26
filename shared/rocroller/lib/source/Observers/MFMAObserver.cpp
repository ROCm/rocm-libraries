/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        MFMAObserver::MFMAObserver() {}

        MFMAObserver::MFMAObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        bool MFMAObserver::isTargetedInstruction(Instruction const& inst)
        {
            return GPUInstructionInfo::isMFMA(inst.getOpCode())
                   && !MFMACoexecObserver::isTargetedInstruction(inst);
        }

        InstructionStatus MFMAObserver::peek(Instruction const& inst) const
        {
            InstructionStatus rv;
            if(isTargetedInstruction(inst))
            {
                rv.stallCycles = m_remainingCycles;

                auto aOperands = inst.getSrcs()[0]->getRegisterIds().to<std::vector>();
                if(aOperands == m_aOperands)
                    rv.reusedOperands++;

                auto bOperands = inst.getSrcs()[1]->getRegisterIds().to<std::vector>();
                if(bOperands == m_bOperands)
                    rv.reusedOperands++;
            }
            return rv;
        }

        void MFMAObserver::modify(Instruction& inst) const
        {
            if(m_remainingCycles > 0 && !inst.isCommentOnly()
               && Settings::Get(Settings::LogLvl) >= LogLevel::Info)
                inst.addComment(concatenate("MFMA remaining: ", m_remainingCycles));
        }

        void MFMAObserver::observe(Instruction const& inst)
        {
            const static std::unordered_set<std::string> variableCycleInsts
                = {"v_mfma_f32_16x16x128_f8f6f4",
                   "v_mfma_scale_f32_16x16x128_f8f6f4",
                   "v_mfma_f32_32x32x64_f8f6f4",
                   "v_mfma_scale_f32_32x32x64_f8f6f4"};

            if(isTargetedInstruction(inst))
            {
                auto info
                    = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());

                auto latency        = info.getLatency();
                auto initialLatency = latency;

                if(variableCycleInsts.contains(inst.getOpCode()))
                {
                    bool any8Bits = false;
                    for(auto const& src : inst.getSrcs())
                    {
                        if(!src)
                            continue;
                        auto info    = DataTypeInfo::Get(src->variableType());
                        auto seg     = info.segmentVariableType;
                        auto segInfo = DataTypeInfo::Get(seg);
                        if(segInfo.elementBits == 8 && !segInfo.isIntegral)
                            any8Bits = true;
                    }

                    if(any8Bits)
                    {
                        Log::trace("Found instruction {} with 8-bit src.", inst.getOpCode());
                        latency *= 2;
                    }
                }

                m_remainingCycles = latency;

                m_aOperands = inst.getSrcs()[0]->getRegisterIds().to<std::vector>();
                m_bOperands = inst.getSrcs()[1]->getRegisterIds().to<std::vector>();
            }
            else
            {
                int myCycles = inst.numExecutedInstructions() + inst.peekedStatus().stallCycles;
                m_remainingCycles = std::max(0, m_remainingCycles - myCycles);
            }
        }

        MFMACoexecObserver::MFMACoexecObserver() {}

        MFMACoexecObserver::MFMACoexecObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        bool MFMACoexecObserver::isTargetedInstruction(Instruction const& inst)
        {
            auto isMxInstruction = GPUInstructionInfo::isMFMA(inst.getOpCode())
                                   && inst.getOpCode().find("f8f6f4") != std::string::npos;

            if(!isMxInstruction)
                return false;

            if(inst.getOpCode().find("scale") == std::string::npos)
                return true;

            /**
             * TODO: Remove.
             * Right now, this observer gives slower results unless it is
             * combined with the LinearWeightedSimple cost function. Once we can
             * make this the default cost function we should be able to remove
             * this and just always return true for this instruction.
             */
            if(Settings::Get(Settings::SchedulerCost) == CostFunction::LinearWeightedSimple)
                return true;

            return false;
        }

        DisallowedCycles MFMACoexecObserver::getDisallowedCycles(Instruction const& inst) const
        {
            // When this has info for multiple instructions, it should probably
            // get moved into the GPUInstructionInfo or else where in the
            // architecture info.

            AssertFatal(isTargetedInstruction(inst));

            bool scaled = inst.getOpCode().find("scale") != std::string::npos;

            DisallowedCycles rv = {{1,
                                    {CoexecCategory::VMEM,
                                     CoexecCategory::VALU,
                                     CoexecCategory::VALU_Trans,
                                     CoexecCategory::XDL,
                                     CoexecCategory::XDL_Scale,
                                     CoexecCategory::LDS}},
                                   {2, {CoexecCategory::XDL, CoexecCategory::XDL_Scale}}};

            if(scaled)
            {
                // If this is a `v_mfma_scale_` instruction, it is actually 2
                // instructions and will therefore take 2 instructions to issue.
                // So the next cycle will not be able to issue anything.
                // Shift everything by a cycle.

                EnumBitset<CoexecCategory> anything = {CoexecCategory::NotAnInstruction};
                auto                       nothing  = ~anything;

                DisallowedCycles rvIncludingMFMA = {{1, nothing}};

                for(auto const& [cycle, disallowed] : rv)
                {
                    AssertFatal(cycle > 0);
                    rvIncludingMFMA[cycle + 1] = disallowed;
                }

                return rvIncludingMFMA;
            }

            return rv;
        }

        InstructionStatus MFMACoexecObserver::peek(Instruction const& inst) const
        {
            using bs = EnumBitset<CoexecCategory>;
            InstructionStatus rv;
            if(isTargetedInstruction(inst))
            {

                rv.disallowedCoexec = getDisallowedCycles(inst);

                auto aOperands = inst.getSrcs()[0]->getRegisterIds().to<std::vector>();
                if(aOperands == m_aOperands)
                    rv.reusedOperands++;

                auto bOperands = inst.getSrcs()[1]->getRegisterIds().to<std::vector>();
                if(bOperands == m_bOperands)
                    rv.reusedOperands++;
            }

            auto category = inst.getCategory();

            do
            {
                auto iter = m_disallowedOps.find(m_programCycle + inst.numExecutedInstructions()
                                                 + rv.stallCycles);
                if(iter == m_disallowedOps.end() || !iter->second[category])
                    break;

                ++rv.stallCycles;
            } while(true);

            return rv;
        }

        void MFMACoexecObserver::modify(Instruction& inst) const
        {
            if(!inst.isCommentOnly() && !m_disallowedOps.empty()
               && Settings::Get(Settings::LogLvl) >= LogLevel::Debug)
            {
                auto lastCycle = m_disallowedOps.rbegin()->first;
                inst.addComment(
                    fmt::format("Cycle: {}\nQueue: {}", m_programCycle, toString(m_disallowedOps)));
            }
        }

        void MFMACoexecObserver::observe(Instruction const& inst)
        {
            int myCycles = inst.numExecutedInstructions() + inst.peekedStatus().stallCycles;
            m_programCycle += myCycles;

            auto iter = m_disallowedOps.upper_bound(m_programCycle);
            m_disallowedOps.erase(m_disallowedOps.begin(), iter);

            combineCoexec(
                m_disallowedOps, inst.peekedStatus().disallowedCoexec, m_programCycle - 1);
        }

        std::string MFMACoexecObserver::state() const
        {
            std::string rv = fmt::format("Cycle: {}\n", m_programCycle);

            rv += fmt::format("disallowed:\n{{\n{}}}\n", toString(m_disallowedOps));

            rv += fmt::format(
                "A Operands: {}\nB Operands: {}\n", toString(m_aOperands), toString(m_bOperands));

            return rv;
        }
    }
}
