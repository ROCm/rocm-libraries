/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
        class WMMAWrite : public WaitStateObserver<WMMAWrite>
        {
        public:
            WMMAWrite() {}
            WMMAWrite(ContextPtr context)
                : WaitStateObserver<WMMAWrite>(context){};

            constexpr static bool required(const GPUArchitectureTarget& target)
            {
                return target.isRDNA4GPU();
            }

            int                   getMaxNops(const Instruction& inst) const;
            bool                  trigger(const Instruction& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            int         getNops(const Instruction& inst) const;
            std::string getComment() const
            {
                return "WMMA Write Hazard";
            }

        private:
            const int m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<WMMAWrite>);
    }
}
