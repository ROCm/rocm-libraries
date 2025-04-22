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
         * @brief GFX9 rules for certain VALU Write of SGPR or VCC. Note that if VCC/SGPRs are read as carry, 0 NOPs are required
         *
         * | Arch | 1st Inst                       | 2nd Inst                               | NOPs |
         * | ---- | ------------------------------ | -------------------------------------- | ---- |
         * | 90x  | v_readlane write SGPR/VCC      | v_* read as constant                   | 1    |
         * | 90x  | v_readlane write SGPR/VCC      | v_readlane read as laneselect          | 4    |
         * | 90x  | v_readfirstlane write SGPR/VCC | v_* read as constant                   | 1    |
         * | 90x  | v_readfirstlane write SGPR/VCC | v_readlane read as laneselect          | 4    |
         * | 90x  | v_cmp_* write SGPR/VCC         | v_* read as constant                   | 1    |
         * | 90x  | v_cmp_* write SGPR/VCC         | v_readlane read as laneselect          | 4    |
         * | 90x  | v_add*_i* write SGPR/VCC       | v_* read as constant (except as carry) | 1    |
         * | 90x  | v_add*_i* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 90x  | v_add*_u* write SGPR/VCC       | v_* read as constant (except as carry) | 1    |
         * | 90x  | v_add*_u* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 90x  | v_sub*_i* write SGPR/VCC       | v_* read as constant (except as carry) | 1    |
         * | 90x  | v_sub*_i* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 90x  | v_sub*_u* write SGPR/VCC       | v_* read as constant (except as carry) | 1    |
         * | 90x  | v_sub*_u* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 90x  | v_div_scale* write SGPR/VCC    | v_* read as constant                   | 1    |
         * | 90x  | v_div_scale* write SGPR/VCC    | v_readlane read as laneselect          | 4    |
         * | 94x  | v_readlane write VCC           | v_* read as constant                   | 1    |
         * | 94x  | v_readlane write SGPR          | v_* read as constant                   | 2    |
         * | 94x  | v_readlane write SGPR/VCC      | v_readlane read as laneselect          | 4    |
         * | 94x  | v_readfirstlane write VCC      | v_* read as constant                   | 1    |
         * | 94x  | v_readfirstlane write SGPR     | v_* read as constant                   | 2    |
         * | 94x  | v_readfirstlane write SGPR/VCC | v_readlane read as laneselect          | 4    |
         * | 94x  | v_cmp_* write VCC              | v_* read as constant                   | 1    |
         * | 94x  | v_cmp_* write SGPR             | v_* read as constant                   | 2    |
         * | 94x  | v_cmp_* write SGPR/VCC         | v_readlane read as laneselect          | 4    |
         * | 94x  | v_add*_i* write VCC            | v_* read as constant (except as carry) | 1    |
         * | 94x  | v_add*_i* write SGPR           | v_* read as constant (except as carry) | 2    |
         * | 94x  | v_add*_i* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 94x  | v_add*_u* write VCC            | v_* read as constant (except as carry) | 1    |
         * | 94x  | v_add*_u* write SGPR           | v_* read as constant (except as carry) | 2    |
         * | 94x  | v_add*_u* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 94x  | v_sub*_i* write VCC            | v_* read as constant (except as carry) | 1    |
         * | 94x  | v_sub*_i* write SGPR           | v_* read as constant (except as carry) | 2    |
         * | 94x  | v_sub*_i* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 94x  | v_sub*_u* write VCC            | v_* read as constant (except as carry) | 1    |
         * | 94x  | v_sub*_u* write SGPR           | v_* read as constant (except as carry) | 2    |
         * | 94x  | v_sub*_u* write SGPR/VCC       | v_readlane read as laneselect          | 4    |
         * | 94x  | v_div_scale* write VCC         | v_* read as constant                   | 1    |
         * | 94x  | v_div_scale* write SGPR        | v_* read as constant                   | 2    |
         * | 94x  | v_div_scale* write SGPR/VCC    | v_readlane read as laneselect          | 4    |
         *
         */
        class VALUWriteSGPRVCC : public WaitStateObserver<VALUWriteSGPRVCC>
        {
        public:
            VALUWriteSGPRVCC() {}
            VALUWriteSGPRVCC(ContextPtr context)
                : WaitStateObserver<VALUWriteSGPRVCC>(context)
            {
                auto const& target = context->targetArchitecture().target();
                m_isCDNA1orCDNA2   = target.isCDNA1GPU() || target.isCDNA2GPU();
            };

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNAGPU();
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
                return "VALU Write SGPR/VCC Hazard";
            }

        private:
            int const m_maxNops = 4;
            bool      m_isCDNA1orCDNA2;
        };

        static_assert(CWaitStateObserver<VALUWriteSGPRVCC>);
    }
}
