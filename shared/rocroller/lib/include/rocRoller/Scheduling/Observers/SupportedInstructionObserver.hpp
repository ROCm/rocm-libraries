// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
         * If the `AllowUnknownInstructions` setting is false, this
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
                return !Settings::getInstance()->get(Settings::AllowUnknownInstructions);
            }

        private:
            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserverRuntime<SupportedInstructionObserver>);
    }
}
