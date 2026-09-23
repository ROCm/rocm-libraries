// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            return GPUInstructionInfo::isMFMA(inst.getOpCode());
        }

        bool MFMAObserver::runtimeRequired(ContextPtr const& ctx)
        {
            return !MFMACoexecObserver::runtimeRequired(ctx);
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

            auto info = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());

            auto latency = info.getLatency();

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
                    Log::debug("Found instruction {} with 8-bit src. Targeted: {} coexec {}",
                               inst.getOpCode(),
                               isTargetedInstruction(inst),
                               MFMACoexecObserver::isTargetedInstruction(inst));
                    latency *= 2;
                }
            }

            if(isTargetedInstruction(inst))
            {
                auto initialLatency = latency;

                m_remainingCycles = latency;

                m_aOperands = inst.getSrcs()[0]->getRegisterIds().to<std::vector>();
                m_bOperands = inst.getSrcs()[1]->getRegisterIds().to<std::vector>();
            }
            else
            {
                int myCycles      = inst.totalCycles();
                m_remainingCycles = std::max(0, m_remainingCycles - myCycles);
            }
        }

        MFMACoexecObserver::MFMACoexecObserver() {}

        MFMACoexecObserver::MFMACoexecObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        bool MFMACoexecObserver::runtimeRequired(ContextPtr const& ctx)
        {
            /**
             * TODO: Remove.
             * Right now, this observer gives slower results unless it is
             * combined with the LinearWeightedSimple or
             * LinearWeightedSimpleStreamK cost function. Once we can make this
             * the default cost function we should be able to remove this and
             * replace the MFMAObserver with this one entirely.
             */
            auto cost = Settings::Get(Settings::SchedulerCost);
            return cost == CostFunction::LinearWeightedSimple
                   || cost == CostFunction::LinearWeightedSimpleStreamK;
        }

        bool MFMACoexecObserver::isTargetedInstruction(Instruction const& inst)
        {
            return GPUInstructionInfo::isMFMA(inst.getOpCode());
        }

        DisallowedCycles MFMACoexecObserver::getDisallowedCycles(Instruction const& inst) const
        {
            // When this has info for multiple instructions, it should probably
            // get moved into the GPUInstructionInfo or else where in the
            // architecture info.

            AssertFatal(isTargetedInstruction(inst));

            auto isHalfSpeed = inst.getAllSrcs()
                                   .filter([](Register::ValuePtr const& operand) {
                                       return operand != nullptr
                                              && (operand->variableType() == DataType::FP8x4
                                                  || operand->variableType() == DataType::BF8x4);
                                   })
                                   .take(1)
                                   .only()
                                   .has_value();

            bool scaled = inst.getOpCode().find("scale") != std::string::npos;

            DisallowedCycles rv;

            if(isHalfSpeed)
            {
                EnumBitset<CoexecCategory> scalarOnly = {CoexecCategory::VMEM,
                                                         CoexecCategory::VALU,
                                                         CoexecCategory::VALU_Trans,
                                                         CoexecCategory::XDL,
                                                         CoexecCategory::XDL_Scale,
                                                         CoexecCategory::LDS};
                EnumBitset<CoexecCategory> noVALU     = {CoexecCategory::VALU,
                                                     CoexecCategory::VALU_Trans,
                                                     CoexecCategory::XDL,
                                                     CoexecCategory::XDL_Scale};
                EnumBitset<CoexecCategory> noXDL = {CoexecCategory::XDL, CoexecCategory::XDL_Scale};

                rv = {{1, scalarOnly}, {2, noVALU}, {3, noXDL}, {4, noXDL}, {5, noXDL}, {6, noXDL}};
            }
            else
            {
                rv = {{1,
                       {CoexecCategory::VMEM,
                        CoexecCategory::VALU,
                        CoexecCategory::VALU_Trans,
                        CoexecCategory::XDL,
                        CoexecCategory::XDL_Scale,
                        CoexecCategory::LDS}},
                      {2, {CoexecCategory::XDL, CoexecCategory::XDL_Scale}}};
            }

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
            int myCycles = inst.totalCycles();
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
