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
         * @brief 94x rules for VALU Trans Writes
         *
         * | Arch | 1st Inst (trans op)  | 2nd Inst (non-trans op)  | NOPs |
         * | ---- | -------------------- | ------------------------ | ---- |
         * | 94x  | v_exp_f32            | v_*                      | 1    |
         * | 94x  | v_log_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_iflag_f32      | v_*                      | 1    |
         * | 94x  | v_rsq_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_f64            | v_*                      | 1    |
         * | 94x  | v_rsq_f64            | v_*                      | 1    |
         * | 94x  | v_sqrt_f32           | v_*                      | 1    |
         * | 94x  | v_sqrt_f64           | v_*                      | 1    |
         * | 94x  | v_sin_f32            | v_*                      | 1    |
         * | 94x  | v_cos_f32            | v_*                      | 1    |
         * | 94x  | v_rcp_f16            | v_*                      | 1    |
         * | 94x  | v_sqrt_f16           | v_*                      | 1    |
         * | 94x  | v_rsq_f16            | v_*                      | 1    |
         * | 94x  | v_log_f16            | v_*                      | 1    |
         * | 94x  | v_exp_f16            | v_*                      | 1    |
         * | 94x  | v_sin_f16            | v_*                      | 1    |
         * | 94x  | v_cos_f16            | v_*                      | 1    |
         * | 94x  | v_exp_legacy_f32     | v_*                      | 1    |
         * | 94x  | v_log_legacy_f32     | v_*                      | 1    |
         *
         */
        class VALUTransWrite94x : public WaitStateObserver<VALUTransWrite94x>
        {
        public:
            VALUTransWrite94x() {}
            VALUTransWrite94x(ContextPtr context)
                : WaitStateObserver<VALUTransWrite94x>(context){};

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
                return "VALU Trans Write Hazard";
            }

        private:
            int const m_maxNops = 1;
        };

        static_assert(CWaitStateObserver<VALUTransWrite94x>);
    }
}
