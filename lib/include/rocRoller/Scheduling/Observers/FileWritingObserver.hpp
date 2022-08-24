
#pragma once

#include <filesystem>
#include <fstream>
#include <stdlib.h>
#include <vector>

#include "../../Utilities/Error.hpp"
#include "../../Utilities/Settings.hpp"
#include "../../Utilities/Utils.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        class FileWritingObserver
        {
        public:
            FileWritingObserver(std::shared_ptr<Context> context)
                : m_context(context)
                , m_assemblyFile()
            {
                m_writing = Settings::getInstance()->get(Settings::SaveAssembly);
            }

            FileWritingObserver(const FileWritingObserver& input)
                : m_context(input.m_context)
                , m_assemblyFile()
                , m_writing(input.m_writing)
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

            InstructionStatus observe(Instruction const& inst)
            {
                if(m_writing)
                {
                    auto context = m_context.lock();
                    if(!m_assemblyFile.is_open())
                    {
                        m_assemblyFile.open(context->assemblyFileName(), std::ios_base::out);
                    }
                    AssertFatal(m_assemblyFile.is_open(),
                                "Could not open file " + context->assemblyFileName()
                                    + " for writing.");
                    m_assemblyFile << inst.toString(context->kernelOptions().logLevel);
                    m_assemblyFile.flush();
                }
                return {};
            }

        private:
            std::weak_ptr<Context> m_context;
            std::ofstream          m_assemblyFile;
            bool                   m_writing;
        };

        static_assert(CObserver<FileWritingObserver>);
    }
}
