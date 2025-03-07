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

#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        template <typename T>
        using WaitQueueMap  = std::unordered_map<GPUWaitQueue, T, GPUWaitQueue::Hash>;
        using WaitCntQueues = WaitQueueMap<
            std::vector<std::array<Register::ValuePtr, Instruction::MaxDstRegisters>>>;

        /**
         * @brief This struct is used to store the _unallocated_ state of the waitcnt queues.
         *
         */
        struct WaitcntState
        {
        public:
            WaitcntState();

            WaitcntState(WaitQueueMap<bool> const&             needsWaitZero,
                         WaitQueueMap<GPUWaitQueueType> const& typeInQueue,
                         WaitCntQueues const&                  instruction_queues_with_alloc);

            bool operator==(const WaitcntState& rhs) const = default;

            /**
             * @brief Returns true if this WaitcntState and the provided WaitcntState are safe
             *        to see at a label and a branch that goes to that label.
             *
             * @param labelState
             */
            void assertSafeToBranchTo(const WaitcntState& labelState,
                                      std::string const&  label) const;

        private:
            // These members are duplicates of the waitcntobserver members, except we're storing a
            // std::vector<Register::RegisterId> for the registers instead of
            // std::array<Register::ValuePtr, Instruction::MaxDstRegisters>
            // so that we don't maintain the allocations.
            WaitQueueMap<std::vector<std::vector<Register::RegisterId>>> m_instructionQueues;

            WaitQueueMap<bool>             m_needsWaitZero;
            WaitQueueMap<GPUWaitQueueType> m_typeInQueue;
        };

        class WaitcntObserver
        {
        public:
            WaitcntObserver();

            WaitcntObserver(ContextPtr context);

            InstructionStatus peek(Instruction const& inst) const;

            void modify(Instruction& inst) const;

            /**
             * This function handles updating the waitqueues and flags when an instruction is scheduled.
             * 1. If there is a wait included in the instruction, apply it to the queues.
             * 2. Add n copies of the instruction's registers to all the queues that it affects, where n is its waitcnt.
             * 3. If the wait is zero, or the queue already has an instruction of a different type, set the m_needsWaitZero flag.
             * 4. Set the instruction type flag for each queue the instruction affects.
             **/
            void observe(Instruction const& inst);

            constexpr static bool required(GPUArchitectureTarget const& target);

        private:
            std::weak_ptr<Context> m_context;

            WaitCntQueues m_instructionQueues;

            // This member tracks a flag for each queue which indicates that a waitcnt 0 is needed.
            WaitQueueMap<bool> m_needsWaitZero;

            // This member tracks the instruction type that is currently in a given queue.
            // If there are ever multiple instruction types in a queue, and a register intersection occurs,
            // a waitcnt 0 is required.
            WaitQueueMap<GPUWaitQueueType> m_typeInQueue;

            bool m_includeExplanation;
            bool m_displayState;

            // This member tracks, for every label, what the waitcnt state was when that label was encountered.
            std::unordered_map<std::string, WaitcntState> m_labelStates;

            // This member tracks, for every label, what the waitcnt state is everywhere a branch instruction targets that label.
            std::unordered_map<std::string, std::vector<WaitcntState>> m_branchStates;

            /**
             * This function updates the given wait queue by applying the given waitcnt.
             **/
            void applyWaitToQueue(int waitCnt, GPUWaitQueue queue);

            /**
             * @brief This function determines if an instruction needs a wait count inserted before it and provides an explanation as to why it's needed.
             *
             * It searches backwards through each wait queue looking for registers that intersect with the new instruction.
             * If an intersection is found a wait is inserted for the intersection location or 0 if the wait_zero flag is set for the queue.
             *
             * @param inst
             * @param[out] explanation is an output parameter for an explanation of the wait count required.
             * @return WaitCount
             */
            WaitCount computeWaitCount(Instruction const& inst,
                                       std::string*       explanation = nullptr) const;

            /**
             * @brief Get a string representation of the state of the Wait Queues
             *
             * @return std::string
             */
            std::string getWaitQueueState() const;

            /**
             * @brief Add the current waitcnt state to the label state tracker.
             *
             * @param label the currently encountered label.
             */
            void addLabelState(std::string const& label);

            /**
             * @brief Add the current waitcnt state to the branch state tracker.
             *
             * @param label the label currently being branched to.
             */
            void addBranchState(std::string const& label);

            /**
             * @brief Assert that all branch and label waitcnt states are consistent.
             *
             * This function checks that the waitcnt state at every branch to a label is the same
             * as the waitcnt state when that label was encountered.
             */
            void assertLabelConsistency();
        };

        static_assert(CObserverConst<WaitcntObserver>);
    }
}

#include <rocRoller/Scheduling/Observers/WaitcntObserver_impl.hpp>
