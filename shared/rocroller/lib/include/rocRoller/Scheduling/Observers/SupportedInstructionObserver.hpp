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

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief Observer that checks for invalid instructions.
         *
         * If the `AllowUnkownInstructions` setting is false, this
         * observer checks every instruction that is scheduled, and if it isn't
         * present in the GPUArchitecture, then an exception is thrown.
         *
         */
        class SupportedInstructionObserver
        {
        public:
            SupportedInstructionObserver() {}
            SupportedInstructionObserver(ContextPtr context)
                : m_context(context){};

            InstructionStatus peek(Instruction const& inst) const;
            void              modify(Instruction& inst) const;
            void              observe(Instruction const& inst);

            static bool runtimeRequired()
            {
                return !Settings::getInstance()->get(Settings::AllowUnkownInstructions);
            }

        private:
            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserverRuntime<SupportedInstructionObserver>);
    }
}
