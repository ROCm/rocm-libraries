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
         * @brief 908 rules for XDLOP Reads of Src C (WAR)
         *
         * | Arch | 1st Inst                    | 2nd Inst                         | NOPs |
         * | ---- | --------------------------- | -------------------------------- | ---- |
         * | 908  | v_mfma* read SrcC (2 pass)  | v_accvgpr_write write overlapped | 0    |
         * | 908  | v_mfma* read SrcC (8 pass)  | v_accvgpr_write write overlapped | 5    |
         * | 908  | v_mfma* read SrcC (16 pass) | v_accvgpr_write write overlapped | 13   |
         *
         */
        class XDLReadSrcC908 : public WaitStateObserver<XDLReadSrcC908>
        {
        public:
            XDLReadSrcC908() {}
            XDLReadSrcC908(ContextPtr context)
                : WaitStateObserver<XDLReadSrcC908>(context){};

            /**
             * Overriden as we need to target src C only
             */
            void observeHazard(Instruction const& inst) override;

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA1GPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return false;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "XDL Read Hazard";
            }

        private:
            std::unordered_map<int, int> m_latencyAndNops = {{2, 0}, {8, 5}, {16, 13}};
        };

        static_assert(CWaitStateObserver<XDLReadSrcC908>);
    }
}
