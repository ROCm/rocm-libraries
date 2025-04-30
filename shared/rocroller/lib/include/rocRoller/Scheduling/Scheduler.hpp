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

#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Scheduling/Costs/Cost_fwd.hpp>
#include <rocRoller/Scheduling/Scheduler_fwd.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        class LockState
        {
        public:
            explicit LockState(ContextPtr ctx);
            LockState(ContextPtr ctx, Dependency dependency);

            void add(Instruction const& instr);
            bool isLocked() const;
            void isValid(bool locked = false) const;

            /**
             * @brief Extra checks to verify lock state integrity.
             *
             * Note: disabled in Release mode.
             *
             * @param instr The instruction to verify
             */
            void lockCheck(Instruction const& instr);

            Dependency getDependency() const;
            int        getLockDepth() const;

        private:
            int                               m_lockdepth;
            Dependency                        m_dependency;
            std::weak_ptr<rocRoller::Context> m_ctx;
        };

        /**
         * Yields from the beginning of the range [begin, end) any comment-only instruction(s).
         */
        template <typename Begin, typename End>
        Generator<Instruction> consumeComments(Begin& begin, End const& end);

        /**
         * A `Scheduler` is a base class for the different types of schedulers
         *
         * - This class should be able to be made into `ComponentBase` class
         */
        class Scheduler
        {
        public:
            using Argument = std::tuple<SchedulerProcedure, CostFunction, rocRoller::ContextPtr>;

            Scheduler(ContextPtr);

            static const std::string Basename;
            static const bool        SingleUse = true;

            virtual std::string name() const = 0;

            /**
             * Call operator schedules instructions based on the scheduling algorithm.
             */
            virtual Generator<Instruction> operator()(std::vector<Generator<Instruction>>& streams)
                = 0;

            /**
             * Returns true if `this->operator()` supports having the caller append to the `streams`
             * argument while the coroutine is suspended. Instructions from the new streams should be
             * incorporated according to the same scheduling algorithm, and **MUST** be added to the
             * end of the vector, not anywhere in the middle.
             */
            virtual bool supportsAddingStreams() const;

            LockState getLockState() const;

        protected:
            LockState                         m_lockstate;
            std::weak_ptr<rocRoller::Context> m_ctx;
            std::shared_ptr<Cost>             m_cost;

            /**
             * Yields from `iter`:
             *
             * - At least one instruction
             * - If that first instruction locks the stream, yields until the stream is unlocked.
             */
            Generator<Instruction> yieldFromStream(Generator<Instruction>::iterator& iter);

            /**
             * @brief Handles new nodes being added to the instruction streams being scheduled.
             *
             * @param seqs
             * @param iterators
             * @return Generator<Instruction> Yielding all initial comments from any new instruction streams.
             */
            static Generator<Instruction>
                handleNewNodes(std::vector<Generator<Instruction>>&           seqs,
                               std::vector<Generator<Instruction>::iterator>& iterators);
        };

        std::ostream& operator<<(std::ostream&, SchedulerProcedure const&);
        std::ostream& operator<<(std::ostream&, Dependency const&);
    }
}

#include <rocRoller/Scheduling/Scheduler_impl.hpp>
