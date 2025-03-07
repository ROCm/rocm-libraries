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
         * @brief 94x/950 rule for VALU Write followed by a Readlane or Permlane
         *
         * | Arch | 1st Inst  | 2nd Inst             | NOPs |
         * | ---- | --------- | -------------------- | ---- |
         * | 94x  | v_* write | v_readlane_* read    | 1    |
         * | 950  | v_* write | v_permlane* read     | 2    |
         *
         */
        class VALUWriteReadlane94x : public WaitStateObserver<VALUWriteReadlane94x>
        {
        public:
            VALUWriteReadlane94x() {}
            VALUWriteReadlane94x(ContextPtr context)
                : WaitStateObserver<VALUWriteReadlane94x>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA3GPU() || target.isCDNA35GPU();
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
                return "VALU Write Readlane Hazard";
            }

        private:
            bool      m_checkACCVGPR;
            int const m_maxNops = 2;
        };

        static_assert(CWaitStateObserver<VALUWriteReadlane94x>);
    }
}
