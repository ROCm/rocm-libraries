
#pragma once

#include <filesystem>
#include <fstream>
#include <stdlib.h>
#include <vector>

#include "../../Utilities/Error.hpp"
#include "../../Utilities/Utils.hpp"

namespace rocRoller
{
    const std::string ENV_SAVE_ASSEMBLY = "ROCROLLER_SAVE_ASSEMBLY";

    namespace Scheduling
    {
        class FileWritingObserver
        {
        public:
            FileWritingObserver(std::shared_ptr<Context> context)
                : m_context(context)
                , m_assemblyFile()
            {
                char* saveAssembly = getenv(ENV_SAVE_ASSEMBLY.c_str());
                m_writing          = saveAssembly && std::string(saveAssembly) == "1";
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
