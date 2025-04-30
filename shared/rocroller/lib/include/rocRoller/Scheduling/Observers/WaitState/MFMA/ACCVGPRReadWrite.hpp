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
         * @brief 908 rule for v_accvgpr_read Write
         *
         * | Arch | 1st Inst             | 2nd Inst                  | NOPs |
         * | ---- | -------------------- | ------------------------- | ---- |
         * | 908  | v_accvgpr_read write | VALU read as SrcA/B/C     | 0    |
         * | 908  | v_accvgpr_read write | v_mfma* read SrcA/B       | 2    |
         * | 908  | v_accvgpr_read write | v_accvgpr_write read SrcA | 2    |
         *
         */
        class ACCVGPRReadWrite : public WaitStateObserver<ACCVGPRReadWrite>
        {
        public:
            ACCVGPRReadWrite() {}
            ACCVGPRReadWrite(ContextPtr context)
                : WaitStateObserver<ACCVGPRReadWrite>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA1GPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "v_accvgpr_read Write Hazard";
            }

        private:
            int const m_maxNops = 2;
        };

        static_assert(CWaitStateObserver<ACCVGPRReadWrite>);
    }
}
