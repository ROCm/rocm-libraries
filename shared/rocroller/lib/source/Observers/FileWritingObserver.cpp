// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <fstream>
#include <stdlib.h>

#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>

#include <rocRoller/Context.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        FileWritingObserver::FileWritingObserver() {}

        FileWritingObserver::FileWritingObserver(ContextPtr context)
            : m_context(context)
            , m_assemblyFile()
        {
        }

        FileWritingObserver::FileWritingObserver(FileWritingObserver const& input)
            : m_context(input.m_context)
            , m_assemblyFile()
        {
        }

        void FileWritingObserver::observe(Instruction const& inst)
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

        bool FileWritingObserver::runtimeRequired()
        {
            return Settings::getInstance()->get(Settings::SaveAssembly);
        }

    }
}
