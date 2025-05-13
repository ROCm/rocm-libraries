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

        protected:
            bool isMEMInstruction(Instruction const& inst) const override;
            int  getWait(Instruction const& inst) const override;
        };

        class DSMEMObserver : public MEMObserver<DSMEMObserver>
        {
        public:
            DSMEMObserver(ContextPtr ctx);

        protected:
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
