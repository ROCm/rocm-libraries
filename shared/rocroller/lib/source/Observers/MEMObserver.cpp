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
#include <rocRoller/KernelGraph/Transforms/LDSBankModel.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        VMEMObserver::VMEMObserver(ContextPtr ctx)
            : MEMObserver(ctx, "VMEM", MEMObserver::getWeights(ctx).vmemQueueSize)
        {
        }

        bool VMEMObserver::isMEMInstruction(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVMEM(inst.getOpCode());
        }

        int VMEMObserver::getWait(Instruction const& inst) const
        {
            return inst.getWaitCount().vmcnt();
        }

        int VMEMObserver::getCyclesForInstruction(Instruction const& inst) const
        {
            auto ctx = this->m_context.lock();
            if(!ctx)
                return 1;
            return MEMObserver::getWeights(ctx).vmemCycles;
        }

        DSMEMObserver::DSMEMObserver(ContextPtr ctx)
            : MEMObserver(ctx, "DSMEM", MEMObserver::getWeights(ctx).dsmemQueueSize)
        {
        }

        bool DSMEMObserver::isMEMInstruction(Instruction const& inst) const
        {
            return GPUInstructionInfo::isLDS(inst.getOpCode());
        }

        int DSMEMObserver::getWait(Instruction const& inst) const
        {
            return inst.getWaitCount().dscnt();
        }

        int DSMEMObserver::getCyclesForInstruction(Instruction const& inst) const
        {
            auto ctx = this->m_context.lock();
            if(!ctx)
                return 4;

            // Skip, ds_write, ds_read_u, transposed, ...;
            // Non-gfx9
            if(not inst.getOpCode().starts_with("ds_read_b")
               || inst.getOpCode().find("_tr") != std::string::npos
               || (not ctx->targetArchitecture().target().isCDNAGPU()))
                return MEMObserver::getWeights(ctx).dsmemCycles;

            auto opCode              = inst.getOpCode();
            auto [dwords, direction] = LDSObserver::getLdsInfoFromOpcode(opCode);

            KernelGraph::MemoryTracer::MemoryOpLDS memOp{direction};

            Log::error("{}", inst.toString(LogLevel::Debug));

            // Currently only annotates MemoryType::WAVE
            if(not inst.addresses.has_value())
                return MEMObserver::getWeights(ctx).dsmemCycles;

            AssertFatal(inst.addresses->size() > 0,
                        "LDS instruction missing addresses for cycle prediction");

            KernelGraph::MemoryTracer::RuntimeLDSInstruction runtimeInst{
                memOp, dwords, *inst.addresses};

            auto gfx = ctx->targetArchitecture().target().gfx;

            auto cycles
                = KernelGraph::MemoryTracer::LDSBankModel::getInstructionCycles(runtimeInst, gfx);
            Log::error("cycles: {}, dwords: {}", cycles, dwords);
            return cycles;
        }

        LDSObserver::LDSObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        std::pair<int, KernelGraph::MemoryTracer::LdsDirection>
            LDSObserver::getLdsInfoFromOpcode(const std::string& opCode)
        {
            int dwords = 1; // default to b32

            if(opCode.find("_b64") != std::string::npos)
                dwords = 2;
            else if(opCode.find("_b96") != std::string::npos)
                dwords = 3;
            else if(opCode.find("_b128") != std::string::npos)
                dwords = 4;

            KernelGraph::MemoryTracer::LdsDirection direction
                = opCode.find("ds_write") != std::string::npos
                      ? KernelGraph::MemoryTracer::LdsDirection::Write
                      : KernelGraph::MemoryTracer::LdsDirection::Read;

            return {dwords, direction};
        }

        int LDSObserver::calculateDataSlots(Instruction const& inst) const
        {
            auto opCode              = inst.getOpCode();
            auto [dwords, direction] = getLdsInfoFromOpcode(opCode);

            switch(direction)
            {
            case KernelGraph::MemoryTracer::LdsDirection::Write:
                return 1 + dwords;
            case KernelGraph::MemoryTracer::LdsDirection::Read:
            default:
                return 1; // For addresses
            }
        }

        int LDSObserver::predictCompletionCycles(Instruction const& inst) const
        {
            auto ctx = m_context.lock();
            if(!ctx)
                return 4;

            auto opCode              = inst.getOpCode();
            auto [dwords, direction] = getLdsInfoFromOpcode(opCode);

            KernelGraph::MemoryTracer::MemoryOpLDS memOp{direction};

            KernelGraph::MemoryTracer::RuntimeLDSInstruction runtimeInst{
                memOp, dwords, {} // TODO: addresses should be apart of Instruction
            };

            auto gfx = ctx->targetArchitecture().target().gfx;

            return KernelGraph::MemoryTracer::LDSBankModel::getInstructionCycles(runtimeInst, gfx);
        }

        InstructionStatus LDSObserver::peek(Instruction const& inst) const
        {
            InstructionStatus rv;

            auto opCode              = inst.getOpCode();
            auto [dwords, direction] = getLdsInfoFromOpcode(opCode);

            return rv;
        }

        void LDSObserver::modify(Instruction& inst) const {}

        void LDSObserver::observe(Instruction const& inst)
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
            {
            }
            else
            {
            }
            m_programCycle += inst.numExecutedInstructions() + inst.peekedStatus().stallCycles;
        }
    }
}
