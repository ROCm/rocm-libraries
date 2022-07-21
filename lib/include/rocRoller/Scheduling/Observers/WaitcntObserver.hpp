#pragma once

#include "../Scheduling.hpp"

#include "../../Context.hpp"
#include "../../GPUArchitecture/GPUInstructionInfo.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        class WaitcntObserver
        {
        public:
            WaitcntObserver(std::shared_ptr<Context> context)
                : m_context(context)
            {
                for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
                {
                    instruction_queues[static_cast<GPUWaitQueue>(i)]
                        = std::vector<std::tuple<std::vector<std::shared_ptr<Register::Value>>,
                                                 std::vector<std::shared_ptr<Register::Value>>>>();

                    needs_wait_zero[static_cast<GPUWaitQueue>(i)] = false;

                    type_in_queue[static_cast<GPUWaitQueue>(i)] = GPUWaitQueueType::None;
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
                inst.addWaitCount(computeWaitCount(inst));
                inst.addWaitCount(inst.getWaitCount().getAsSaturatedWaitCount(architecture));
            }

            /**
             * This function handles updating the waitqueues and flags when an instruction is scheduled.
             * 1. If there is a wait included in the instruction, apply it to the queues.
             * 2. Add n copies of the instruction's registers to all the queues that it affects, where n is its waitcnt.
             * 3. If the wait is zero, or the queue already has an instruction of a different type, set the needs_wait_zero flag.
             * 4. Set the instruction type flag for each queue the instruction affects.
             **/
            InstructionStatus observe(Instruction const& inst)
            {
                auto               context      = m_context.lock();
                const auto&        architecture = context->targetArchitecture();
                GPUInstructionInfo info         = architecture.GetInstructionInfo(inst.opCode());

                auto registers      = inst.getRegisters();
                auto instWaitQueues = info.getWaitQueues();

                WaitCount waiting = inst.getWaitCount();

                if(std::find(instWaitQueues.begin(),
                             instWaitQueues.end(),
                             GPUWaitQueueType::FinalInstruction)
                   != instWaitQueues.end())
                {
                    waiting = WaitCount::Zero(context->targetArchitecture());
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
                                instruction_queues[waitQueue].push_back(registers);
                            }
                            type_in_queue[waitQueue] = queueType;
                        }
                    }
                }
                return {};
            }

        private:
            std::weak_ptr<Context> m_context;

            std::unordered_map<
                GPUWaitQueue,
                std::vector<std::tuple<std::vector<std::shared_ptr<Register::Value>>,
                                       std::vector<std::shared_ptr<Register::Value>>>>,
                GPUWaitQueue::Hash>
                instruction_queues;

            // This member tracks a flag for each queue which indicates that a waitcnt 0 is neded.
            std::unordered_map<GPUWaitQueue, bool, GPUWaitQueue::Hash> needs_wait_zero;

            // This member tracks the instruction type that is currently in a given queue.
            // If there are ever multiple instruction types in a queue, and a register intersection occurs,
            // a waitcnt 0 is required.
            std::unordered_map<GPUWaitQueue, GPUWaitQueueType, GPUWaitQueue::Hash> type_in_queue;

            /**
             * This function updates the given wait queue by applying the given waitcnt.
             **/
            inline void applyWaitToQueue(int waitCnt, GPUWaitQueue queue)
            {
                if(waitCnt != -1 && instruction_queues[queue].size() > waitCnt)
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
                    }
                }
            }

            /**
             * This function determines if an instruction needs a wait count inserted before it.
             * It searches backwards through each wait queue looking for registers that intersect with the new instruction.
             * If an intersection is found a wait is inserted for the intersection location or 0 if the wait_zero flag is set for the queue.
             **/
            inline WaitCount computeWaitCount(Instruction const& inst) const
            {
                WaitCount   retval;
                auto        context      = m_context.lock();
                const auto& architecture = context->targetArchitecture();

                if(inst.opCode() == "s_barrier")
                {
                    return WaitCount::Zero(architecture);
                }

                if(inst.opCode().size() > 0 && inst.hasRegisters())
                {
                    for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
                    {
                        GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);
                        for(int queue_i = instruction_queues.at(waitQueue).size() - 1; queue_i >= 0;
                            queue_i--)
                        {
                            if(inst.registersIntersect(
                                   std::get<1>(instruction_queues.at(waitQueue)[queue_i]),
                                   std::get<0>(instruction_queues.at(waitQueue)[queue_i])))
                            {
                                if(needs_wait_zero.at(waitQueue))
                                {
                                    retval.combine(WaitCount(waitQueue, 0));
                                }
                                else
                                {
                                    retval.combine(WaitCount(waitQueue,
                                                             instruction_queues.at(waitQueue).size()
                                                                 - (queue_i + 1)));
                                }
                                break;
                            }
                        }
                    }
                }

                return retval.getAsSaturatedWaitCount(architecture);
            }
        };

        static_assert(CObserver<WaitcntObserver>);
    }
}
