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

#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908/90a/94x rule for V_CMPX Write to EXEC followed by an MFMA requiring 4 NOPs
         *
         * | Arch | 1st Inst           | 2nd Inst        | NOPs |
         * | ---- | ------------------ | --------------- | ---- |
         * | 908  | v_cmpx* write EXEC | v_mfma*         | 4    |
         * | 908  | v_cmpx* write EXEC | v_accvgpr_write | 4    |
         * | 90a  | v_cmpx* write EXEC | v_mfma*         | 4    |
         * | 94x  | v_cmpx* write EXEC | v_mfma*         | 4    |
         *
         */
        class CMPXWriteExec : public WaitStateObserver<CMPXWriteExec>
        {
        public:
            CMPXWriteExec() {}
            CMPXWriteExec(ContextPtr context)
                : WaitStateObserver<CMPXWriteExec>(context)
            {
                m_checkACCVGPR = context->targetArchitecture().target().isCDNA1GPU();
            };

            /**
             * Overriden as we need to target Exec
             */
            void observeHazard(Instruction const& inst) override;

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNAGPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            int                   getNops(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            std::string getComment() const
            {
                return "EXEC Write Hazard";
            }

        private:
            bool      m_checkACCVGPR;
            int const m_maxNops = 4;
        };

        static_assert(CWaitStateObserver<CMPXWriteExec>);
    }
}
