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

#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponentBase(Scheduler);

        std::string toString(SchedulerProcedure proc)
        {
            switch(proc)
            {
            case SchedulerProcedure::Sequential:
                return "Sequential";
            case SchedulerProcedure::RoundRobin:
                return "RoundRobin";
            case SchedulerProcedure::Random:
                return "Random";
            case SchedulerProcedure::Cooperative:
                return "Cooperative";
            case SchedulerProcedure::Priority:
                return "Priority";
            case SchedulerProcedure::Count:
                return "Count";
            }

            Throw<FatalError>("Invalid Scheduler Procedure: ", ShowValue(static_cast<int>(proc)));
        }

        std::ostream& operator<<(std::ostream& stream, SchedulerProcedure proc)
        {
            return stream << toString(proc);
        }

        std::string toString(Dependency dep)
        {
            switch(dep)
            {
            case Dependency::None:
                return "None";
            case Dependency::Branch:
                return "Branch";
            case Dependency::M0:
                return "M0";
            case Dependency::VCC:
                return "VCC";
            case Dependency::SCC:
                return "SCC";
            default:
                break;
            }

            Throw<FatalError>("Invalid Dependency: ", ShowValue(static_cast<int>(dep)));
        }

        std::ostream& operator<<(std::ostream& stream, Dependency dep)
        {
            return stream << toString(dep);
        }

        std::string toString(LockOperation lockOp)
        {
            switch(lockOp)
            {
            case LockOperation::None:
                return "None";
            case LockOperation::Lock:
                return "Lock";
            case LockOperation::Unlock:
                return "Unlock";
            default:
                break;
            }

            Throw<FatalError>("Invalid LockOperation: ", ShowValue(static_cast<int>(lockOp)));
        }

        std::ostream& operator<<(std::ostream& stream, LockOperation lockOp)
        {
            return stream << toString(lockOp);
        }

        constexpr bool isNonPreemptibleDependency(Dependency dep)
        {
            return dep != Dependency::M0 && dep != Dependency::VCC;
        }

        LockState::LockState(ContextPtr ctx)
            : m_ctx(ctx)
        {
        }

        LockState::LockState(ContextPtr ctx, Dependency dependency)
            : m_ctx(ctx)
        {
            lock(dependency, 0);
        }

        void LockState::lock(Dependency dep, int streamId)
        {
            AssertFatal(dep != Dependency::Count && dep != Dependency::None);

            auto topDep = getTopDependency(streamId);
            AssertFatal(topDep <= dep,
                        "Out of order dependency lock can't be acquired.",
                        ShowValue(topDep),
                        ShowValue(dep));

            // Can a stream acquire the same lock (single resource, just the top) multiple times? yes
            // VCC -> SCC -> SCC -> SCC
            if(m_stream.contains(dep))
            {
                AssertFatal(
                    topDep == dep && m_stream.at(dep) == streamId,
                    "Only the same stream can acquire the top dependency lock multiple times.",
                    ShowValue(dep),
                    ShowValue(m_stream[dep]),
                    ShowValue(streamId));
            }

            m_stack[streamId].push(dep);
            m_stream[dep] = streamId;
            m_locks.insert(dep);

            if(isNonPreemptibleDependency(dep))
                m_nonPreemptibleStream = streamId;
        }

        void LockState::unlock(Dependency dep, int streamId)
        {
            AssertFatal(streamId >= 0);
            AssertFatal(m_stack.contains(streamId));
            AssertFatal(m_stack[streamId].size() > 0);
            AssertFatal(dep != Dependency::Count);

            // LIFO
            {
                auto topDep = getTopDependency(streamId);
                if(dep != Dependency::None)
                    AssertFatal(topDep == dep, "locks can only be released in the LIFO order");
                else
                    dep = topDep;
            }

            {
                auto iter = m_stream.find(dep);
                AssertFatal(iter != m_stream.end() && iter->second == streamId,
                            ShowValue(dep),
                            ShowValue(streamId));
            }

            // pop the stack top
            m_stack[streamId].pop();

            // erase one instance of dep from the multiset.
            // if that's the last instance, then erase its streamID mapping.
            {
                auto iter = m_locks.find(dep);
                AssertFatal(iter != m_locks.end());
                m_locks.erase(iter);

                if(!m_locks.contains(dep))
                    m_stream.erase(dep);
            }

            // update m_nonPreemptibleStream state if needed
            // Example: when a stream holds multiple non-preemptible
            // locks like Branch -> VCC -> SCC.
            m_nonPreemptibleStream.reset();
            auto tempDep = getTopDependency(streamId);
            while(tempDep != Dependency::None)
            {
                if(m_stream.contains(tempDep) && m_stream.at(tempDep) == streamId
                   && isNonPreemptibleDependency(tempDep))
                {
                    m_nonPreemptibleStream = streamId;
                    break;
                }

                auto depVal = static_cast<int>(tempDep);
                tempDep     = static_cast<Dependency>(--depVal);
            }
        }

        bool LockState::isSchedulable(Instruction const& instr, int streamId) const
        {
            auto dep = instr.getDependency();
            AssertFatal(dep != Dependency::Count && streamId >= 0);

            auto topDep = getTopDependency(streamId);
            // check if the order of the dependencies satisfies
            AssertFatal(dep == Dependency::None || topDep <= dep,
                        "Out of order dependency lock can't be acquired.",
                        ShowValue(topDep),
                        ShowValue(dep));

            // if the stream itself is non-preemptible,
            if(m_stack.empty())
                return true;

            // it's schedulable
            if(isNonPreemptibleStream(streamId))
                return true;
            // if another stream is non-preemptible,
            // it's not schedulable.
            else if(m_nonPreemptibleStream.has_value())
                return false;

            auto lockOp = instr.getLockValue();
            // if the given instr is not a lock instruction,
            // it's schedulable.
            if(lockOp != LockOperation::Lock)
                return true;

            // if the dependency is already locked and is being
            // scheduled by the same stream again, it's schedulable.
            if(m_locks.contains(dep))
                return m_stream.at(dep) == streamId;

            // If the given stream tries to acquire a non-preemptible lock
            // and another stream currently holds a higher-ranked preemptible lock,
            // the scheduler cannot schedule this lower-ranked non-preemptible
            // lock from streamId until the higher-ranked preemptible lock is released
            // by the another stream.
            if(isNonPreemptibleDependency(dep))
            {
                auto depVal  = static_cast<int>(dep);
                auto tempDep = static_cast<Dependency>(++depVal);
                while(tempDep != Dependency::Count)
                {
                    if(m_locks.contains(tempDep))
                        return false;

                    depVal  = static_cast<int>(tempDep);
                    tempDep = static_cast<Dependency>(++depVal);
                }
            }

            return true;
        }

        void LockState::add(Instruction const& instruction, int streamId)
        {
            //lockCheck(instruction, streamId);

            AssertFatal(isSchedulable(instruction, streamId),
                        "cannot add any instruction from this stream at this point");

            auto lockOp = instruction.getLockValue();

            switch(lockOp)
            {
            case LockOperation::None:
                break;

            case LockOperation::Lock:
                lock(instruction.getDependency(), streamId);
                break;

            case LockOperation::Unlock:
                unlock(instruction.getDependency(), streamId);
                break;

            case LockOperation::Count:
                Throw<FatalError>("Invalid LockOperation ", static_cast<int>(lockOp));
            }
        }

        bool LockState::isNonPreemptibleStream(int streamId) const
        {
            return m_nonPreemptibleStream.has_value() && streamId == m_nonPreemptibleStream.value();
        }

        bool LockState::isLocked(Dependency dep, int streamId) const
        {
            return m_stream.contains(dep) && m_stream.at(dep) == streamId;
        }

        void LockState::lockCheck(Instruction const& instruction, int streamId) const
        {
            auto               context      = m_ctx.lock();
            const auto&        architecture = context->targetArchitecture();
            GPUInstructionInfo info = architecture.GetInstructionInfo(instruction.getOpCode());

            AssertFatal(
                !info.isBranch() || isLocked(Dependency::Branch, streamId),
                concatenate(instruction.getOpCode(),
                            " is a branch instruction, it should only be used within a lock."));

            AssertFatal(
                !info.hasImplicitAccess() || isLocked(Dependency::M0, streamId)
                    || isLocked(Dependency::VCC, streamId) || isLocked(Dependency::SCC, streamId),
                concatenate(instruction.getOpCode(),
                            " implicitly reads a register, it should only be used within a lock."));

            AssertFatal(
                !instruction.readsSpecialRegisters() || isLocked(Dependency::M0, streamId)
                    || isLocked(Dependency::VCC, streamId) || isLocked(Dependency::SCC, streamId),
                concatenate(instruction.getOpCode(),
                            " reads a special register, it should only be used within a lock."));
        }

        Dependency LockState::getTopDependency(int streamId) const
        {
            if(m_stack.contains(streamId) && !(m_stack.at(streamId).empty()))
                return m_stack.at(streamId).top();

            return Dependency::None;
        }

        int LockState::getLockDepth(int streamId) const
        {
            if(m_stack.contains(streamId))
                return m_stack.at(streamId).size();

            return 0;
        }

        Generator<Instruction> Scheduler::yieldFromStream(Generator<Instruction>::iterator& iter,
                                                          int streamId)
        {
            do
            {
                AssertFatal(iter != std::default_sentinel_t{},
                            "End of instruction stream reached without unlocking",
                            ShowValue(streamId));
                m_lockstate.add(*iter, streamId);
                co_yield *iter;
                ++iter;
                co_yield consumeComments(iter, std::default_sentinel_t{});
            } while(m_lockstate.isNonPreemptibleStream(streamId));
        }

        bool Scheduler::supportsAddingStreams() const
        {
            return false;
        }

        Generator<Instruction>
            Scheduler::handleNewNodes(std::vector<Generator<Instruction>>&           seqs,
                                      std::vector<Generator<Instruction>::iterator>& iterators)
        {
            while(seqs.size() != iterators.size())
            {
                AssertFatal(seqs.size() >= iterators.size(),
                            "Sequences cannot shrink!",
                            ShowValue(seqs.size()),
                            ShowValue(iterators.size()));
                iterators.reserve(seqs.size());
                for(size_t i = iterators.size(); i < seqs.size(); i++)
                {
                    iterators.emplace_back(seqs[i].begin());
                    // Consume any comments at the beginning of the stream.
                    // This has the effect of immediately executing Deallocate nodes.
                    co_yield consumeComments(iterators[i], seqs[i].end());
                }
            }
        }
    }
}
