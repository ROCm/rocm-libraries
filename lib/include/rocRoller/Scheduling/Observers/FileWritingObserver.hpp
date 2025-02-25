
#pragma once

#include <filesystem>
#include <fstream>
#include <stdlib.h>
#include <vector>

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
                m_assemblyFile << inst.toString(context->kernelOptions().logLevel);
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
