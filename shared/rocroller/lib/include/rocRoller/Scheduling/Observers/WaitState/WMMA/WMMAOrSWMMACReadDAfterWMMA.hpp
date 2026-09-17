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
        /**
         * @brief 125X rules for WMMA/SWMMAC Reads of D
         *
         * | 1st Inst writes vdest         | 2nd Inst reads previous vdest         | NOPs                                 |
         * | ----------------------------- | ------------------------------------- | ------------------------------------ |
         * | v_wmma*_FP16 or               |                                       |                                      |
         * | v_wmma*_BF16 or               |                                       |                                      |
         * | v_wmma*_FP8FP8 or             |                                       |                                      |
         * | v_wmma*_FP8BF8 or             |                                       |  1 + 4, iff co-execution is enabled. |
         * | v_wmma*_BF8FP8 or             |         v_wmma* or v_swmmac*          |  otherwise, 1.                       |
         * | v_wmma*_BF8BF8 or             |                                       |                                      |
         * | v_wmma*_F8F6F4 with A's and   |                                       |                                      |
         * |   B's type are not F8         |                                       |                                      |
         * | write vdest                   |                                       |                                      |
         * | ----------------------------- | ------------------------------------- | ------------------------------------ |
         * | v_wmma*_IU8 or                |                                       |                                      |
         * | v_wmma*_IU4 or                |                                       |  1 + 8, iff co-execution is enabled. |
         * | v_wmma*_F8F6F4 with A's or    |         v_wmma* or v_swmmac*          |  otherwise, 1.                       |
         * |   B's type is F8              |                                       |                                      |
         * | write vdest                   |                                       |                                      |
         * | ----------------------------- | ------------------------------------- | ------------------------------------ |
         *
         */
        class WMMAOrSWMMACReadDAfterWMMA : public WaitStateObserver<WMMAOrSWMMACReadDAfterWMMA>
        {
        public:
            WMMAOrSWMMACReadDAfterWMMA() {}
            WMMAOrSWMMACReadDAfterWMMA(ContextPtr context)
                : WaitStateObserver<WMMAOrSWMMACReadDAfterWMMA>(context){};

            constexpr static bool required(const GPUArchitectureTarget& target)
            {
                return target.isCDNA5GPU();
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
                return "WMMA/SWMMAC Read WMMA Write Hazard";
            }

        private:
            const int m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<WMMAOrSWMMACReadDAfterWMMA>);
    }
}
