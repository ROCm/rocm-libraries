// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        class MFMAObserver
        {
        public:
            MFMAObserver();
            MFMAObserver(ContextPtr ctx);

            InstructionStatus peek(Instruction const& inst) const;

            void modify(Instruction& inst) const;

            void observe(Instruction const& inst);

            static bool runtimeRequired(ContextPtr const& ctx);

            static bool isTargetedInstruction(Instruction const& inst);

        private:
            int m_remainingCycles = 0;

            std::vector<Register::RegisterId> m_aOperands;
            std::vector<Register::RegisterId> m_bOperands;

            std::weak_ptr<Context> m_context;
        };

        class MFMACoexecObserver
        {
        public:
            MFMACoexecObserver();
            MFMACoexecObserver(ContextPtr ctx);

            InstructionStatus peek(Instruction const& inst) const;

            void modify(Instruction& inst) const;

            void observe(Instruction const& inst);

            static bool runtimeRequired(ContextPtr const& ctx);

            DisallowedCycles getDisallowedCycles(Instruction const& inst) const;

            static bool isTargetedInstruction(Instruction const& inst);

            std::string state() const;

        private:
            int m_programCycle = 0;

            std::map<int, EnumBitset<CoexecCategory>> m_disallowedOps;

            std::vector<Register::RegisterId> m_aOperands;
            std::vector<Register::RegisterId> m_bOperands;

            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserverRuntimeWithContext<MFMAObserver>);
        static_assert(CObserverRuntimeWithContext<MFMACoexecObserver>);

    }
}
