// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        template <typename Derived>
        class MEMObserver
        {
        public:
            MEMObserver();
            MEMObserver(ContextPtr         ctx,
                        std::string const& commentTag,
                        int                cyclesPerInst,
                        int                queueAllotment);

            InstructionStatus peek(Instruction const& inst) const;

            void modify(Instruction& inst) const;

            void observe(Instruction const& inst);

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return true;
            }

            static Scheduling::Weights getWeights(ContextPtr ctx);

        protected:
            virtual bool isMEMInstruction(Instruction const& inst) const = 0;
            virtual int  getWait(Instruction const& inst) const          = 0;

            int m_programCycle = 0;

            const int m_cyclesPerInst;
            const int m_queueAllotment;

            std::string m_commentTag;

            struct Info
            {
                int issuedCycle;
                int expectedCompleteCycle;
            };

            std::deque<Info> m_completeButNotWaited;
            std::deque<Info> m_incomplete;

            size_t queueLen() const;
            Info   queuePop();
            void   queueShift();

            std::weak_ptr<Context> m_context;
        };

        class VMEMObserver : public MEMObserver<VMEMObserver>
        {
        public:
            VMEMObserver(ContextPtr ctx);

            bool isMEMInstruction(Instruction const& inst) const override;
            int  getWait(Instruction const& inst) const override;
        };

        class DSMEMObserver : public MEMObserver<DSMEMObserver>
        {
        public:
            DSMEMObserver(ContextPtr ctx);

            bool isMEMInstruction(Instruction const& inst) const override;
            int  getWait(Instruction const& inst) const override;
        };

        template <typename Derived>
        MEMObserver<Derived>::MEMObserver()
        {
        }

        static_assert(CObserverConst<VMEMObserver>);
        static_assert(CObserverConst<DSMEMObserver>);
    }
}

#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver_impl.hpp>
