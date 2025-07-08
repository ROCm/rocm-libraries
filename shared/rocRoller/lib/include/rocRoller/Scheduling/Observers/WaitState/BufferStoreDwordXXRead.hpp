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
         * @brief CDNA rule for buffer_store_dwordx3/4 followed by a VALU or non-VALU instruction
         *
         * | Arch | 1st Inst                    | 2nd Inst      | NOPs |
         * | ---- | --------------------------- | ------------- | ---- |
         * | 908  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 908  | buffer_store_dwordx3/4 read | v_* read      | 1    |
         * | 90a  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 90a  | buffer_store_dwordx3/4 read | v_* read      | 1    |
         * | 94x  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 94x  | buffer_store_dwordx3/4 read | v_* read      | 2    |
         * | 950  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 950  | buffer_store_dwordx3/4 read | v_* read      | 2    |
         *
         * NOTE: If soffset argument is an SGPR, no NOPs required
         *
         */
        class BufferStoreDwordXXRead : public WaitStateObserver<BufferStoreDwordXXRead>
        {
        public:
            BufferStoreDwordXXRead() {}
            BufferStoreDwordXXRead(ContextPtr context)
                : WaitStateObserver<BufferStoreDwordXXRead>(context)
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
                return false;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "Buffer Store Read Hazard";
            }

        private:
            bool      m_isCDNA1orCDNA2;
            int const m_maxNops = 2;
        };

        static_assert(CWaitStateObserver<BufferStoreDwordXXRead>);
    }
}
