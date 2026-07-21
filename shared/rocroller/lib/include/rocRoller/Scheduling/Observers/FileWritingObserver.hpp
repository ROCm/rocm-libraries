// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        class FileWritingObserver
        {
        public:
            FileWritingObserver();
            FileWritingObserver(ContextPtr context);
            FileWritingObserver(FileWritingObserver const& input);

            InstructionStatus peek(Instruction const& inst) const
            {
                return {};
            }

            void modify(Instruction& inst) const
            {
                return;
            }

            void observe(Instruction const& inst);

            static bool runtimeRequired();

        private:
            std::weak_ptr<Context> m_context;
            std::ofstream          m_assemblyFile;
        };

        static_assert(CObserverRuntime<FileWritingObserver>);
    }
}
