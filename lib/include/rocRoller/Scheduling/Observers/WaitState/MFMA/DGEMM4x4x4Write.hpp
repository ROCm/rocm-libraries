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
         * @brief 90a rules for v_mfma_f64_4x4x4f64 Write Hazards
         *
         * | Arch | 1st Inst                  | 2nd Inst                           | NOPs |
         * | ---- | ------------------------- | ---------------------------------- | ---- |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_f64_4x4x4f64 read SrcC same | 4    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcC overlapped | 4    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma* read SrcC overlapped       | 0    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcA/B          | 6    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_* read/write                     | 6    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | buffer* read overlapped            | 9    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | ds* read overlapped                | 9    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | flat* read overlapped              | 9    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma_f64_4x4x4f64 read SrcC same | 4    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcC overlapped | 4    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma* read SrcC overlapped       | 0    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcA/B          | 6    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_* read/write                     | 6    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | buffer* read overlapped            | 9    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | ds* read overlapped                | 9    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | flat* read overlapped              | 9    |
         *
         */
        class DGEMM4x4x4Write : public WaitStateObserver<DGEMM4x4x4Write>
        {
        public:
            DGEMM4x4x4Write() {}
            DGEMM4x4x4Write(ContextPtr context)
                : WaitStateObserver<DGEMM4x4x4Write>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA2GPU() || target.isCDNA3GPU() || target.isCDNA35GPU();
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
                return "DGEMM Write Hazard";
            }

        private:
            std::string m_targetOpCode = "v_mfma_f64_4x4x4f64";
            int const   m_maxNops      = 9;
        };

        static_assert(CWaitStateObserver<DGEMM4x4x4Write>);
    }
}
