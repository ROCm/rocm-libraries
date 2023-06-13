
#pragma once

#include "../Scheduling.hpp"

#include "../../Context.hpp"
#include "../../GPUArchitecture/GPUInstructionInfo.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        class AllocatingObserver
        {
        public:
            AllocatingObserver() {}
            AllocatingObserver(ContextPtr context)
                : m_context(context)
            {
            }

            InstructionStatus peek(Instruction const& inst) const
            {
                auto              ctx = m_context.lock();
                InstructionStatus rv;

                // Remaining registers after this instruction:
                // currently remaining - newly allocated
                for(int i = 0; i < static_cast<int>(Register::Type::Count); i++)
                {
                    auto regType   = static_cast<Register::Type>(i);
                    auto allocator = ctx->allocator(regType);

                    if(allocator)
                        rv.remainingRegisters[i] = allocator->currentlyFree();
                }

                for(auto alloc : inst.allocations())
                {
                    if(alloc)
                    {
                        // Determine amount of new registers that will be allocated

                        auto regIdx = static_cast<size_t>(alloc->regType());
                        AssertFatal(regIdx < rv.allocatedRegisters.size(),
                                    ShowValue(regIdx),
                                    ShowValue(rv.allocatedRegisters.size()));
                        rv.allocatedRegisters.at(regIdx) += alloc->registerCount();

                        // Determine new highwater mark by simulating the allocation

                        auto allocator = ctx->allocator(alloc->regType());
                        auto newRegs
                            = allocator->findFree(alloc->registerCount(), alloc->options());
                        if(newRegs.size() > 0)
                        {
                            int myHWM = std::max(0, newRegs.back() - allocator->maxUsed());

                            AssertFatal(regIdx < rv.highWaterMarkRegistersDelta.size(),
                                        ShowValue(regIdx),
                                        ShowValue(rv.highWaterMarkRegistersDelta.size()));
                            rv.highWaterMarkRegistersDelta.at(regIdx)
                                = std::max(myHWM, rv.highWaterMarkRegistersDelta.at(regIdx));
                        }

                        rv.outOfRegisters[alloc->regType()]
                            = rv.outOfRegisters[alloc->regType()]
                              || alloc->registerCount() > 0 && newRegs.empty();
                        rv.remainingRegisters[regIdx] -= alloc->registerCount();
                    }
                }
                return rv;
            }

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            void modify(Instruction& inst) const
            {
                inst.allocateNow();
            }

            //> This instruction _will_ be scheduled now, record any side effects.
            void observe(Instruction const& inst) {}

            static bool required(ContextPtr)
            {
                return true;
            }

        private:
            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserver<AllocatingObserver>);
    }
}
