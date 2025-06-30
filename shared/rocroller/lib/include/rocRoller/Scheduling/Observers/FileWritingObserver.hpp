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

#include <filesystem>
#include <fstream>
#include <stdlib.h>
#include <vector>

#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        class FileWritingObserver
        {
        public:
            FileWritingObserver() {}

            FileWritingObserver(ContextPtr context)
                : m_context(context)
                , m_assemblyFile()
            {
            }

            FileWritingObserver(const FileWritingObserver& input)
                : m_context(input.m_context)
                , m_assemblyFile()
            {
            }

            InstructionStatus peek(Instruction const& inst) const
            {
                return {};
            }

            void modify(Instruction& inst) const
            {
                return;
            }

            void observe(Instruction const& inst)
            {
                auto context = m_context.lock();
                if(!m_assemblyFile.is_open())
                {
                    m_assemblyFile.open(context->assemblyFileName(), std::ios_base::out);
                }
                AssertFatal(m_assemblyFile.is_open(),
                            "Could not open file " + context->assemblyFileName() + " for writing.");
                m_assemblyFile << inst.toString(context->kernelOptions()->logLevel);
                m_assemblyFile.flush();
            }

            static bool runtimeRequired()
            {
                return Settings::getInstance()->get(Settings::SaveAssembly);
            }

        private:
            std::weak_ptr<Context> m_context;
            std::ofstream          m_assemblyFile;
        };

        static_assert(CObserverRuntime<FileWritingObserver>);
    }
}
