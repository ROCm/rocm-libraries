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
         * @brief 94x rules for XDL Write Hazards.
         *
         * Note: Excludes DGEMM cases.
         *
         * | Arch | 1st Inst                    | 2nd Inst                           | NOPs |
         * | ---- | --------------------------- | ---------------------------------- | ---- |
         * | 94x  | v_mfma* write (2 pass)      | v_mfma* read SrcC same             | 2    |
         * | 94x  | v_mfma* write (4 pass)      | v_mfma* read SrcC same             | 0    |
         * | 94x  | v_mfma* write (8 pass)      | v_mfma* read SrcC same             | 0    |
         * | 94x  | v_mfma* write (16 pass)     | v_mfma* read SrcC same             | 0    |
         * | 94x  | v_mfma_*f32 write (2 pass)  | v_mfma* read SrcC overlapped       | 2    |
         * | 94x  | v_mfma_*f32 write (4 pass)  | v_mfma* read SrcC overlapped       | 4    |
         * | 94x  | v_mfma_*f32 write (8 pass)  | v_mfma* read SrcC overlapped       | 8    |
         * | 94x  | v_mfma_*f32 write (16 pass) | v_mfma* read SrcC overlapped       | 16   |
         * | 94x  | v_mfma* write (2 pass)      | v_mfma* read SrcC overlapped       | 3    |
         * | 94x  | v_mfma* write (4 pass)      | v_mfma* read SrcC overlapped       | 5    |
         * | 94x  | v_mfma* write (8 pass)      | v_mfma* read SrcC overlapped       | 9    |
         * | 94x  | v_mfma* write (16 pass)     | v_mfma* read SrcC overlapped       | 17   |
         * | 94x  | v_mfma* write (2 pass)      | v_mfma_*f64 read SrcC overlapped   | 3    |
         * | 94x  | v_mfma* write (4 pass)      | v_mfma_*f64 read SrcC overlapped   | 5    |
         * | 94x  | v_mfma* write (8 pass)      | v_mfma_*f64 read SrcC overlapped   | 9    |
         * | 94x  | v_mfma* write (16 pass)     | v_mfma_*f64 read SrcC overlapped   | 17   |
         * | 94x  | v_mfma_*f32 write (2 pass)  | v_mfma* read SrcA/B                | 4    |
         * | 94x  | v_mfma_*f32 write (4 pass)  | v_mfma* read SrcA/B                | 6    |
         * | 94x  | v_mfma_*f32 write (8 pass)  | v_mfma* read SrcA/B                | 10   |
         * | 94x  | v_mfma_*f32 write (16 pass) | v_mfma* read SrcA/B                | 18   |
         * | 94x  | v_mfma_*f32 write (2 pass)  | v_mfma* read SrcA/B                | 5    |
         * | 94x  | v_mfma_*f32 write (4 pass)  | v_mfma* read SrcA/B                | 7    |
         * | 94x  | v_mfma_*f32 write (8 pass)  | v_mfma* read SrcA/B                | 11   |
         * | 94x  | v_mfma_*f32 write (16 pass) | v_mfma* read SrcA/B                | 19   |
         * | 94x  | v_mfma_*f32 write (2 pass)  | buffer* read overlapped            | 5    |
         * | 94x  | v_mfma_*f32 write (4 pass)  | buffer* read overlapped            | 7    |
         * | 94x  | v_mfma_*f32 write (8 pass)  | buffer* read overlapped            | 11   |
         * | 94x  | v_mfma_*f32 write (16 pass) | buffer* read overlapped            | 19   |
         * | 94x  | v_mfma_*f32 write (2 pass)  | ds* read overlapped                | 5    |
         * | 94x  | v_mfma_*f32 write (4 pass)  | ds* read overlapped                | 7    |
         * | 94x  | v_mfma_*f32 write (8 pass)  | ds* read overlapped                | 11   |
         * | 94x  | v_mfma_*f32 write (16 pass) | ds* read overlapped                | 19   |
         * | 94x  | v_mfma_*f32 write (2 pass)  | flat* read overlapped              | 5    |
         * | 94x  | v_mfma_*f32 write (4 pass)  | flat* read overlapped              | 7    |
         * | 94x  | v_mfma_*f32 write (8 pass)  | flat* read overlapped              | 11   |
         * | 94x  | v_mfma_*f32 write (16 pass) | flat* read overlapped              | 19   |
         * | 94x  | v_mfma_*f32 write (2 pass)  | v_* read/write                     | 5    |
         * | 94x  | v_mfma_*f32 write (4 pass)  | v_* read/write                     | 7    |
         * | 94x  | v_mfma_*f32 write (8 pass)  | v_* read/write                     | 11   |
         * | 94x  | v_mfma_*f32 write (16 pass) | v_* read/write                     | 19   |
         * | 94x  | v_mfma* write (2 pass)      | v_mfma* read SrcA/B                | 5    |
         * | 94x  | v_mfma* write (4 pass)      | v_mfma* read SrcA/B                | 7    |
         * | 94x  | v_mfma* write (8 pass)      | v_mfma* read SrcA/B                | 11   |
         * | 94x  | v_mfma* write (16 pass)     | v_mfma* read SrcA/B                | 19   |
         * | 94x  | v_mfma* write (2 pass)      | buffer* read overlapped            | 5    |
         * | 94x  | v_mfma* write (4 pass)      | buffer* read overlapped            | 7    |
         * | 94x  | v_mfma* write (8 pass)      | buffer* read overlapped            | 11   |
         * | 94x  | v_mfma* write (16 pass)     | buffer* read overlapped            | 19   |
         * | 94x  | v_mfma* write (2 pass)      | ds* read overlapped                | 5    |
         * | 94x  | v_mfma* write (4 pass)      | ds* read overlapped                | 7    |
         * | 94x  | v_mfma* write (8 pass)      | ds* read overlapped                | 11   |
         * | 94x  | v_mfma* write (16 pass)     | ds* read overlapped                | 19   |
         * | 94x  | v_mfma* write (2 pass)      | flat* read overlapped              | 5    |
         * | 94x  | v_mfma* write (4 pass)      | flat* read overlapped              | 7    |
         * | 94x  | v_mfma* write (8 pass)      | flat* read overlapped              | 11   |
         * | 94x  | v_mfma* write (16 pass)     | flat* read overlapped              | 19   |
         * | 94x  | v_mfma* write (2 pass)      | v_* read/write                     | 5    |
         * | 94x  | v_mfma* write (4 pass)      | v_* read/write                     | 7    |
         * | 94x  | v_mfma* write (8 pass)      | v_* read/write                     | 11   |
         * | 94x  | v_mfma* write (16 pass)     | v_* read/write                     | 19   |
         *
         */
        class XDLWrite94x : public WaitStateObserver<XDLWrite94x>
        {
        public:
            XDLWrite94x() {}
            XDLWrite94x(ContextPtr context)
                : WaitStateObserver<XDLWrite94x>(context){};

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
                return "XDL Write Hazard";
            }

        private:
            // Excluded as these are handled in other observers
            std::vector<std::string> m_excludedOpCodes
                = {"v_mfma_f64_4x4x4f64", "v_mfma_f64_16x16x4f64"};

            std::unordered_map<int, int> m_latencyAndNops = {{2, 5}, {4, 7}, {8, 11}, {16, 19}};
        };

        static_assert(CWaitStateObserver<XDLWrite94x>);
    }
}
