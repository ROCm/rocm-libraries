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

#include <rocRoller/Scheduling/Observers/VGPRIndexingObserver.hpp>
#include <rocRoller/Scheduling/Observers/VGPRIndexingObserver_detail.hpp>

#include <rocRoller/KernelOptions_detail.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void VGPRIndexingObserver::modify(Instruction& inst) const
        {
            using namespace VGPRIndexingObserverDetail;

            auto               context      = m_context.lock();
            auto const&        architecture = context->targetArchitecture();
            GPUInstructionInfo info         = architecture.GetInstructionInfo(inst.getOpCode());

            if(info.isBranch() or inst.isLabel())
            {
                // Conservatively reset MODE register before branches and labels.
                inst.setModeRegister(0);
                return;
            }

            uint8_t modeSet          = 0b0;
            bool    foundVgprOperand = false;

            auto checkOperandsForMode = [&](auto const& operands,
                                            uint        maxOperands,
                                            uint        shift,
                                            uint        iterShift) {
                for(auto const& operand : operands)
                {
                    if(operand and operand->regType() == Register::Type::Vector)
                    {
                        foundVgprOperand = true;

                        int idx = *operand->registerIndices().begin();
                        modeSet |= (GetBankBits(idx) << shift);

                        Log::debug(
                            "[ VGPR indexing observer ]: operand for {}: {}, index: {} bank: {} ",
                            inst.getOpCode(),
                            operand->toString(),
                            idx,
                            GetBankBits(idx));

                        if(--maxOperands == 0)
                            break;
                    }
                    shift += iterShift;
                }
            };

            // Process 1 dst
            checkOperandsForMode(inst.getDsts(), 1, 6, 0);
            // Process 3 srcs
            checkOperandsForMode(inst.getSrcs(), 3, 0, 2);

            if(foundVgprOperand and m_modeSet != modeSet)
            {
                Log::debug("[ VGPR indexing observer ]: MODE reg for {}: 0x{:02X}",
                           inst.getOpCode(),
                           modeSet);
                inst.setModeRegister(modeSet);
            }
        }

        void VGPRIndexingObserver::observe(Instruction const& inst)
        {
            if(inst.getModeRegister())
                m_modeSet = *inst.getModeRegister();
        }
    }
}
