#pragma once

#include <sstream>

#include "../Scheduling.hpp"

#include "../../Context.hpp"
#include "../../GPUArchitecture/GPUInstructionInfo.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        template <typename T>
        using WaitQueueMap  = std::unordered_map<GPUWaitQueue, T, GPUWaitQueue::Hash>;
        using WaitCntQueues = WaitQueueMap<std::vector<
            std::array<std::shared_ptr<Register::Value>, Instruction::MaxDstRegisters>>>;

        /**
         * @brief This struct is used to store the _unallocated_ state of the waitcnt queues.
         *
         */
        struct WaitcntState
        {
        public:
            WaitcntState() {}

            WaitcntState(WaitQueueMap<bool> const&             needs_wait_zero,
                         WaitQueueMap<GPUWaitQueueType> const& type_in_queue,
                         WaitCntQueues const&                  instruction_queues_with_alloc)
                : needs_wait_zero(needs_wait_zero)
                , type_in_queue(type_in_queue)
            {
                // Here we're iterating through all of the std::shared_ptr<Register::Value>s and
                // converting them to RegisterIDs
                for(auto& queue : instruction_queues_with_alloc)
                {
                    if(instruction_queues.find(queue.first) == instruction_queues.end())
                    {
                        instruction_queues[queue.first] = {};
                    }
                    for(auto& dsts : queue.second)
                    {
                        instruction_queues[queue.first].emplace_back(
                            std::vector<Register::RegisterId>{});
                        for(auto& dst : dsts)
                        {
                            if(dst)
                            {
                                for(auto& regid : dst->getRegisterIds())
                                {
                                    instruction_queues[queue.first]
                                                      [instruction_queues[queue.first].size() - 1]
                                                          .emplace_back(regid);
                                }
                            }
                        }
                    }
                }
            }

            inline bool operator==(const WaitcntState& rhs) const
            {
                return needs_wait_zero == rhs.needs_wait_zero && type_in_queue == rhs.type_in_queue
                       && instruction_queues == rhs.instruction_queues;
            }

        private:
            // These members are duplicates of the waitcntobserver members, except we're storing a
            // std::vector<Register::RegisterId> for the registers instead of
            // std::array<std::shared_ptr<Register::Value>, Instruction::MaxDstRegisters>
            // so that we don't maintain the allocations.
            WaitQueueMap<std::vector<std::vector<Register::RegisterId>>> instruction_queues;

            WaitQueueMap<bool>             needs_wait_zero;
            WaitQueueMap<GPUWaitQueueType> type_in_queue;
        };

        class WaitcntObserver
        {
        public:
            WaitcntObserver() {}

            WaitcntObserver(std::shared_ptr<Context> context)
                : m_context(context)
            {
                m_displayState = Settings::getInstance()->get(Settings::LogLvl) >= LogLevel::Debug;
                for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
                {
                    GPUWaitQueue waitQueue        = static_cast<GPUWaitQueue>(i);
                    instruction_queues[waitQueue] = {};
                    needs_wait_zero[waitQueue]    = false;
                    type_in_queue[waitQueue]      = GPUWaitQueueType::None;
                }

                AssertFatal(instruction_queues.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                            ShowValue(instruction_queues.size()),
                            ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
                AssertFatal(needs_wait_zero.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                            ShowValue(needs_wait_zero.size()),
                            ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
                AssertFatal(type_in_queue.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                            ShowValue(type_in_queue.size()),
                            ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
            };

            InstructionStatus peek(Instruction const& inst) const
            {
                return InstructionStatus::Wait(computeWaitCount(inst));
            };

            void modify(Instruction& inst) const
            {
                auto        context      = m_context.lock();
                const auto& architecture = context->targetArchitecture();
                inst.addWaitCount(inst.getWaitCount().getAsSaturatedWaitCount(
                    architecture)); // Handle if manually specified waitcnts are over the sat limits.

                std::string explanation;
                inst.addWaitCount(computeWaitCount(inst, &explanation));
                inst.addComment(explanation);

                if(m_displayState && !inst.isCommentOnly())
                {
                    inst.addComment(getWaitQueueState());
                }
            }

            /**
             * This function handles updating the waitqueues and flags when an instruction is scheduled.
             * 1. If there is a wait included in the instruction, apply it to the queues.
             * 2. Add n copies of the instruction's registers to all the queues that it affects, where n is its waitcnt.
             * 3. If the wait is zero, or the queue already has an instruction of a different type, set the needs_wait_zero flag.
             * 4. Set the instruction type flag for each queue the instruction affects.
             **/
            void observe(Instruction const& inst)
            {
                auto               context      = m_context.lock();
                auto const&        architecture = context->targetArchitecture();
                GPUInstructionInfo info         = architecture.GetInstructionInfo(inst.getOpCode());

                if(context->kernelOptions().assertWaitCntState)
                {
                    if(info.isBranch())
                    {
                        addBranchState(inst.getSrcs()[0]->toString());
                    }
                    else if(inst.isLabel())
                    {
                        addLabelState(inst.getLabel());
                    }
                }

                auto instWaitQueues = info.getWaitQueues();

                WaitCount waiting = inst.getWaitCount();

                if(std::find(instWaitQueues.begin(),
                             instWaitQueues.end(),
                             GPUWaitQueueType::FinalInstruction)
                   != instWaitQueues.end())
                {
                    waiting = WaitCount::Zero(context->targetArchitecture());

                    if(context->kernelOptions().assertWaitCntState)
                    {
                        assertLabelConsistency();
                    }
                }

                for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
                {
                    applyWaitToQueue(waiting.getCount(static_cast<GPUWaitQueue>(i)),
                                     static_cast<GPUWaitQueue>(i));
                }

                for(GPUWaitQueueType queueType : instWaitQueues)
                {
                    GPUWaitQueue waitQueue(queueType);
                    if(queueType != GPUWaitQueueType::None
                       && instruction_queues.find(waitQueue) != instruction_queues.end())
                    {
                        int instWaitCnt = info.getWaitCount();
                        if(instWaitCnt >= 0)
                        {
                            if(instWaitCnt == 0)
                            {
                                needs_wait_zero[waitQueue] = true;
                                instWaitCnt                = 1;
                            }
                            else if(type_in_queue[waitQueue] != GPUWaitQueueType::None
                                    && type_in_queue[waitQueue] != queueType)
                            {
                                needs_wait_zero[waitQueue] = true;
                            }
                            for(int i = 0; i < instWaitCnt; i++)
                            {
                                instruction_queues[waitQueue].push_back(inst.getDsts());
                            }
                            type_in_queue[waitQueue] = queueType;
                        }
                    }
                }
            }

            static bool required(std::shared_ptr<Context>)
            {
                return true;
            }

        private:
            std::weak_ptr<Context> m_context;

            WaitCntQueues instruction_queues;

            // This member tracks a flag for each queue which indicates that a waitcnt 0 is needed.
            WaitQueueMap<bool> needs_wait_zero;

            // This member tracks the instruction type that is currently in a given queue.
            // If there are ever multiple instruction types in a queue, and a register intersection occurs,
            // a waitcnt 0 is required.
            WaitQueueMap<GPUWaitQueueType> type_in_queue;

            bool m_displayState;

            // This member tracks, for every label, what the waitcnt state was when that label was encountered.
            std::unordered_map<std::string, WaitcntState> label_states;

            // This member tracks, for every label, what the waitcnt state is everywhere a branch instruction targets that label.
            std::unordered_map<std::string, std::vector<WaitcntState>> branch_states;

            /**
             * This function updates the given wait queue by applying the given waitcnt.
             **/
            inline void applyWaitToQueue(int waitCnt, GPUWaitQueue queue)
            {
                if(waitCnt >= 0 && instruction_queues[queue].size() > (size_t)waitCnt)
                {
                    if(!(needs_wait_zero[queue]
                         && waitCnt
                                > 0)) //Do not partially clear the queue if a waitcnt zero is needed.
                    {
                        instruction_queues[queue].erase(instruction_queues[queue].begin(),
                                                        instruction_queues[queue].begin()
                                                            + instruction_queues[queue].size()
                                                            - waitCnt);
                    }
                    if(instruction_queues[queue].size() == 0)
                    {
                        needs_wait_zero[queue] = false;
                        type_in_queue[queue]   = GPUWaitQueueType::None;
                    }
                }
            }

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
            inline WaitCount computeWaitCount(Instruction const& inst,
                                              std::string*       explanation = nullptr) const
            {
                auto        context      = m_context.lock();
                const auto& architecture = context->targetArchitecture();

                if(inst.getOpCode() == "s_barrier")
                {
                    if(explanation != nullptr)
                    {
                        *explanation = "WaitCnt Needed: Always waitcnt zero before an s_barrier.";
                    }
                    return WaitCount::Zero(architecture);
                }

                WaitCount retval;

                if(inst.getOpCode().size() > 0 && inst.hasRegisters())
                {
                    for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
                    {
                        GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);
                        for(int queue_i = instruction_queues.at(waitQueue).size() - 1; queue_i >= 0;
                            queue_i--)
                        {
                            if(inst.isAfterWriteDependency(
                                   instruction_queues.at(waitQueue)[queue_i]))
                            {
                                if(needs_wait_zero.at(waitQueue))
                                {
                                    retval.combine(WaitCount(waitQueue, 0));
                                    if(explanation != nullptr)
                                    {
                                        *explanation
                                            = "WaitCnt Needed: Intersects with registers in '"
                                              + waitQueue.ToString()
                                              + "', which needs a wait zero.";
                                    }
                                }
                                else
                                {
                                    int waitval
                                        = instruction_queues.at(waitQueue).size() - (queue_i + 1);
                                    retval.combine(WaitCount(waitQueue, waitval));
                                    if(explanation != nullptr)
                                    {
                                        *explanation
                                            = "WaitCnt Needed: Intersects with registers in '"
                                              + waitQueue.ToString() + "', at "
                                              + std::to_string(queue_i) + " and the queue size is "
                                              + std::to_string(
                                                  instruction_queues.at(waitQueue).size())
                                              + ", so a waitcnt of " + std::to_string(waitval)
                                              + " is required.";
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
                return retval.getAsSaturatedWaitCount(architecture);
            }

            /**
             * @brief Get a string representation of the state of the Wait Queues
             *
             * @return std::string
             */
            inline std::string getWaitQueueState() const
            {
                std::stringstream retval;
                for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
                {
                    GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);

                    // Only include state information for wait queues in a non-default state.
                    if(needs_wait_zero.at(waitQueue)
                       || type_in_queue.at(waitQueue) != GPUWaitQueueType::None
                       || instruction_queues.at(waitQueue).size() > 0)
                    {
                        if(retval.rdbuf()->in_avail() == 0)
                        {
                            retval << "Wait Queue State:";
                        }
                        retval << "\n--Queue: " << waitQueue.ToString();
                        retval << "\n----Needs Wait Zero: "
                               << (needs_wait_zero.at(waitQueue) ? "True" : "False");
                        retval << "\n----Type In Queue  : "
                               << type_in_queue.at(waitQueue).ToString();
                        retval << "\n----Registers      : ";

                        for(int queue_i = 0; queue_i < instruction_queues.at(waitQueue).size();
                            queue_i++)
                        {
                            retval << "\n------Dst: {";
                            for(auto& reg : instruction_queues.at(waitQueue)[queue_i])
                            {
                                if(reg)
                                {
                                    retval << reg->toString() << ", ";
                                }
                            }
                            retval << "}";
                        }
                    }
                }
                return retval.str();
            }

            /**
             * @brief Add the current waitcnt state to the label state tracker.
             *
             * @param label the currently encountered label.
             */
            inline void addLabelState(std::string const& label)
            {
                label_states[label]
                    = WaitcntState(needs_wait_zero, type_in_queue, instruction_queues);
            }

            /**
             * @brief Add the current waitcnt state to the branch state tracker.
             *
             * @param label the label currently being branched to.
             */
            inline void addBranchState(std::string const& label)
            {
                if(branch_states.find(label) == branch_states.end())
                {
                    branch_states[label] = {};
                }

                branch_states[label].emplace_back(
                    WaitcntState(needs_wait_zero, type_in_queue, instruction_queues));
            }

            /**
             * @brief Assert that all branch and label waitcnt states are consistent.
             *
             * This function checks that the waitcnt state at every branch to a label is the same
             * as the waitcnt state when that label was encountered.
             */
            inline void assertLabelConsistency()
            {
                for(auto label_state : label_states)
                {
                    if(branch_states.find(label_state.first) != branch_states.end())
                    {
                        for(auto branch_state : branch_states[label_state.first])
                        {
                            AssertFatal(label_state.second == branch_state,
                                        "Branching to label '" + label_state.first
                                            + "' with a different waitcnt state.");
                        }
                    }
                }
            }
        };

        static_assert(CObserver<WaitcntObserver>);
    }
}
