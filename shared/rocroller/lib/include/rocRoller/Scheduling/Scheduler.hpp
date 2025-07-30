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

#include <stack>
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
        constexpr bool isNonPreemptibleDependency(Dependency dep);

        /**
         * Locking Rules
         *
         * A scheduler has a number of streams which each will yield a sequence of instructions.
         * The job of the scheduler is to pick (i.e. schedule) the instruction from the beginning 
	 * of one of the streams, and then repeat until there are no more streams with any 
	 * instructions left.
         *
         * - If a scheduler schedules a lock for a non-preemptible dependency, 
	 *   it must continue to select instructions from that same stream 
	 *   until that lock has been unlocked.
         * - That stream might include further lock/unlock instructions which
         *   must occur in a last-in, first-out order, those should be treated
         *   as a stack to track when the original lock has been unlocked.
	 *
	 * - Dependency rank (low-to-high):
	 *   Branch (Non-preemptible)
	 *   M0 (Preemptible)
	 *   VCC (Preemptible)
	 *   SCC (Non-preemptible)
         *
         * - If a stream yields any kind of lock, it cannot yield a lower-ranked
         *   dependency lock until it releases the higher-ranked dependency lock.
         * - If a scheduler schedules a lock for a preemptible dependency,
	 *   it cannot schedule the same kind of lock from any other stream 
	 *   until that lock is released.
         *      Example:
         *          1. Stream 0 locks M0.
         *          2. Stream 1 locks VCC.
         *          3. Stream 0 tries to lock VCC.  It must wait until Stream 1 unlocks VCC.
         *          4. Stream 1 unlocks VCC.
         *          5. Stream 0 locks VCC.
         *          6. Stream 1 locks SCC.  Pull from Stream 1 until SCC is unlocked.
         *          7. Stream 0 locks SCC.  Pull from Stream 0 until SCC is unlocked.
         *          8. Stream 0 unlocks VCC.
         *          9. Stream 0 unlocks M0.
         *
         * - If a scheduler schedules a lock for a preemptible dependency, 
	 *   it cannot schedule a lower-ranked non-preemptible dependency lock 
	 *   from any stream until that lock is released.
         *      Examples:
         *          - If stream 0 locks M0, and then we see stream 3 try to lock
         *            Branch, we can't pull from stream 3 until stream 0
         *            releases M0.
         *          - If stream 2 locks VCC, stream 1 can lock SCC.  We will
         *            then have to pull from stream 1 until SCC is released.
         *
         */
        class LockState
        {
        public:
            explicit LockState(ContextPtr ctx);
            LockState(ContextPtr ctx, Dependency dependency);

            void add(Instruction const& instr, int streamId);
            bool isNonPreemptibleStream(int streamId) const;
            bool isSchedulable(Instruction const& instr, int streamId) const;
            bool isLocked(Dependency dependency, int streamId) const;

            /**
             * @brief Extra checks to verify lock state integrity.
             *
             * Note: disabled in Release mode.
             *
             * @param instr The instruction to verify
         * @param streamId The instruction's stream ID
             */
            void lockCheck(Instruction const& instr, int streamId) const;

            Dependency getTopDependency(int streamId) const;
            int        getLockDepth(int streamId) const;

        private:
            void lock(Dependency dep, int streamId);
            void unlock(Dependency dep, int streamId);

            std::map<int, std::stack<Dependency>> m_streamToStack;
            std::map<Dependency, int>             m_depToStream;
            std::unordered_multiset<Dependency>   m_locks;
            std::optional<int>                    m_nonPreemptibleStream;

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
            Generator<Instruction> yieldFromStream(Generator<Instruction>::iterator& iter,
                                                   int                               streamId);

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

        std::ostream& operator<<(std::ostream&, SchedulerProcedure proc);
        std::ostream& operator<<(std::ostream&, Dependency dep);
        std::ostream& operator<<(std::ostream& stream, LockOperation lockOp);
    }
}

#include <rocRoller/Scheduling/Scheduler_impl.hpp>
