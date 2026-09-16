// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Array.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    inline Instruction::Instruction() = default;

    inline Instruction::Instruction(std::string const&                        opcode,
                                    std::initializer_list<Register::ValuePtr> dst,
                                    std::initializer_list<Register::ValuePtr> src,
                                    std::initializer_list<std::string>        modifiers,
                                    std::string const&                        comment)
        : m_opcode(opcode)
    {
        append(m_dst, dst);
        append(m_src, src);

        AssertFatal(modifiers.size() <= m_modifiers.size(),
                    ShowValue(modifiers.size()),
                    ShowValue(m_modifiers.size()));
        std::copy(modifiers.begin(), modifiers.end(), m_modifiers.begin());

        addComment(comment);

        for(auto& dst : m_dst)
        {
            if(dst && dst->allocationState() == Register::AllocationState::Unallocated)
            {
                dst->allocate(*this);
            }

            if(dst && dst->readOnly())
            {
                addComment(fmt::format(
                    "Read-only register cannot be the destination of an instruction: {}",
                    dst->description()));
            }
        }
    }

    inline Instruction Instruction::Allocate(Register::ValuePtr reg)
    {
        return Allocate(reg->allocation());
    }

    inline Instruction Instruction::Allocate(std::shared_ptr<Register::Allocation> reg)
    {
        return Allocate({reg});
    }

    inline Instruction
        Instruction::Allocate(std::initializer_list<std::shared_ptr<Register::Allocation>> regs)
    {
        AssertFatal(
            regs.size() <= MaxAllocations, ShowValue(regs.size()), ShowValue(MaxAllocations));

        for(auto const& r : regs)
            AssertFatal(r->allocationState() == Register::AllocationState::Unallocated,
                        ShowValue(r->allocationState()),
                        ShowValue(Register::AllocationState::Unallocated));

        Instruction rv;

        std::copy(regs.begin(), regs.end(), rv.m_allocations.begin());

        return rv;
    }

    inline Instruction Instruction::Directive(std::string const& directive)
    {
        return Directive(directive, "");
    }

    inline Instruction Instruction::Directive(std::string const& directive,
                                              std::string const& comment)
    {
        Instruction rv;
        rv.m_directive = directive;
        rv.addComment(comment);
        return rv;
    }

    inline Instruction Instruction::Comment(std::string const& comment)
    {
        Instruction rv;
        rv.addComment(comment);
        return rv;
    }

    inline Instruction Instruction::Warning(std::string const& warning)
    {
        Instruction rv;
        rv.m_warnings = {warning};
        return rv;
    }

    inline Instruction Instruction::Nop()
    {
        return Nop(1);
    }

    inline Instruction Instruction::Nop(int nopCount)
    {
        return Nop(nopCount, "");
    }

    inline Instruction Instruction::Nop(std::string const& comment)
    {
        return Nop(1, comment);
    }

    inline Instruction Instruction::Nop(int nopCount, std::string const& comment)
    {
        Instruction rv;
        rv.addNop(nopCount);
        rv.addComment(comment);
        return rv;
    }

    inline Instruction Instruction::Label(Register::ValuePtr label)
    {
        return Label(label->getLabel());
    }

    inline Instruction Instruction::Label(Register::ValuePtr label, std::string const& comment)
    {
        return Label(label->getLabel(), comment);
    }

    inline Instruction Instruction::Label(std::string const& name)
    {
        return Label(name, "");
    }

    inline Instruction Instruction::Label(std::string&& name)
    {
        return Label(name, "");
    }

    inline Instruction Instruction::Label(std::string const& name, std::string const& comment)
    {
        Instruction rv;
        rv.m_label = name;
        rv.addComment(comment);
        return rv;
    }

    inline Instruction Instruction::Label(std::string&& name, std::string const& comment)
    {
        Instruction rv;
        rv.m_label = std::move(name);
        rv.addComment(comment);
        return rv;
    }

    inline Instruction
        Instruction::InoutInstruction(std::string const&                        opcode,
                                      std::initializer_list<Register::ValuePtr> inout,
                                      std::initializer_list<std::string>        modifiers,
                                      std::string const&                        comment)
    {
        // Store registers as srcs
        Instruction rv = Instruction(opcode, {}, inout, modifiers, comment);
        AssertFatal(inout.size() <= rv.m_inoutDsts.size(),
                    ShowValue(inout.size()),
                    ShowValue(rv.m_inoutDsts.size()));
        // Duplicate registers in m_inoutDsts. Leave m_dst empty.
        std::copy(inout.begin(), inout.end(), rv.m_inoutDsts.begin());
        rv.m_operandsAreInout = true;
        return rv;
    }

    inline Instruction Instruction::Wait(WaitCount const& wait)
    {
        Instruction rv;
        rv.m_waitCount = wait;
        return rv;
    }

    inline Instruction Instruction::Wait(WaitCount&& wait)
    {
        Instruction rv;
        rv.m_waitCount = std::move(wait);
        return rv;
    }

    inline Instruction Instruction::Lock(Scheduling::Dependency dependency, std::string comment)
    {
        Instruction rv;
        rv.lock(dependency, std::move(comment));
        return rv;
    }

    inline Instruction Instruction::Unlock(std::string comment)
    {
        return Unlock(Scheduling::Dependency::None, std::move(comment));
    }

    inline Instruction Instruction::Unlock(Scheduling::Dependency dependency, std::string comment)
    {
        Instruction rv;
        rv.unlock(dependency, std::move(comment));
        return rv;
    }

    inline Instruction& Instruction::addExtraDst(Register::ValuePtr reg)
    {
        append(m_extraDsts, reg);

        return *this;
    }

    inline Instruction& Instruction::addExtraSrc(Register::ValuePtr reg)
    {
        append(m_extraSrcs, reg);

        return *this;
    }

    inline auto const& Instruction::getDsts() const
    {
        // For in/out operands, m_src and m_inoutDsts are populated and m_dst is empty
        return m_operandsAreInout ? m_inoutDsts : m_dst;
    }

    inline auto const& Instruction::getSrcs() const
    {
        return m_src;
    }

    inline auto const& Instruction::getExtraDsts() const
    {
        return m_extraDsts;
    }

    inline auto const& Instruction::getExtraSrcs() const
    {
        return m_extraSrcs;
    }

    inline bool Instruction::hasRegisters() const
    {
        return !getAllOperands().empty();
    }

    inline Generator<Register::ValuePtr> Instruction::getAllDsts() const
    {
        auto notNull = [](Register::ValuePtr val) -> bool { return (bool)val; };

        co_yield filter(notNull, getDsts());
        co_yield filter(notNull, getExtraDsts());
    }

    inline Generator<Register::ValuePtr> Instruction::getAllSrcs() const
    {
        auto notNull = [](Register::ValuePtr val) -> bool { return (bool)val; };

        co_yield filter(notNull, getSrcs());
        co_yield filter(notNull, getExtraSrcs());
    }

    inline Generator<Register::ValuePtr> Instruction::getAllOperands() const
    {
        co_yield getAllDsts();
        co_yield getAllSrcs();
    }

    inline constexpr bool Instruction::readsSpecialRegisters() const
    {
        for(auto& reg : m_src)
        {
            // SPRs which are read-only in the kernel code do not create
            // scheduling dependencies.
            if(reg && reg->isSpecial() && !reg->readOnly())
            {
                return true;
            }
        }
        return false;
    }

    inline constexpr bool Instruction::isCommentOnly() const
    {
        // clang-format off
        return m_opcode.empty()
            && m_label.empty()
            && m_directive.empty()
            && m_nopCount       == 0
            && m_allocations[0] == nullptr
            && m_waitCount      == WaitCount()
            && m_dependency     == Scheduling::Dependency::None
            && m_lockOp         == Scheduling::LockOperation::None;
        // clang-format on
    }

    inline int Instruction::numExecutedInstructions() const
    {
        int rv = m_opcode.empty() ? 0 : 1;

        if(m_nopCount > 0)
            rv += m_nopCount;

        if(m_waitCount != WaitCount())
            rv++;

        return rv;
    }

    inline int Instruction::totalCycles() const
    {
        return numExecutedInstructions() + m_peekedStatus.stallCycles
               + m_peekedStatus.additionalCycles;
    }

    inline constexpr bool Instruction::isLabel() const
    {
        return !m_label.empty();
    }

    inline std::string Instruction::getLabel() const
    {
        return m_label;
    }

    constexpr inline int Instruction::nopCount() const
    {
        return m_nopCount;
    }

    inline WaitCount Instruction::getWaitCount() const
    {
        return m_waitCount;
    }

    template <CForwardRangeOf<Register::ValuePtr> T>
    bool Instruction::isAfterWriteDependency(T const& previousDest) const
    {
        for(auto const& myReg : getAllSrcs())
        {
            for(auto const& prevReg : previousDest)
            {
                if(prevReg && prevReg->intersects(myReg))
                    return true;
            }
        }
        for(auto const& myReg : getAllDsts())
        {
            // dst -> dst dependencies are not counted for LDS for now.
            // Once we split the LDS allocations it should be possible to
            // track this as well.
            if(myReg->regType() != Register::Type::LocalData)
            {
                for(auto const& prevReg : previousDest)
                {
                    if(prevReg && prevReg->intersects(myReg))
                        return true;
                }
            }
        }

        return false;
    }

    inline void Instruction::toStream(std::ostream& os, LogLevel level) const
    {
        preambleString(os, level);
        functionalString(os, level);
        codaString(os, level);
    }

    inline std::string Instruction::toString(LogLevel level) const
    {
        std::ostringstream oss;
        toStream(oss, level);
        return oss.str();
    }

    inline std::string const& Instruction::getOpCode() const
    {
        return m_opcode;
    }

    inline std::array<std::string, Instruction::MaxModifiers> Instruction::getModifiers() const
    {
        return m_modifiers;
    }

    inline void Instruction::addAllocation(std::shared_ptr<Register::Allocation> alloc)
    {
        for(auto& a : m_allocations)
        {
            if(!a)
            {
                a = alloc;
                return;
            }
        }

        throw std::runtime_error("Too many allocations!");
    }

    inline Instruction& Instruction::lock(Scheduling::Dependency dependency, std::string comment)
    {
        if(dependency == Scheduling::Dependency::Count)
            return *this;

        AssertFatal(m_lockOp == Scheduling::LockOperation::None,
                    "An instruction can only lock or unlock once.");

        // AssertFatal(dependency != Scheduling::Dependency::Count,
        //             "Can not create lock instruction with Unlock or Count dependency");

        m_lockOp     = Scheduling::LockOperation::Lock;
        m_dependency = dependency;
        addComment(std::move(comment));
        return *this;
    }

    inline Instruction& Instruction::unlock(std::string comment)
    {
        return unlock(Scheduling::Dependency::None, std::move(comment));
    }

    inline Instruction& Instruction::unlock(Scheduling::Dependency dependency, std::string comment)
    {
        if(dependency == Scheduling::Dependency::Count)
            return *this;

        AssertFatal(m_lockOp == Scheduling::LockOperation::None,
                    "An instruction can only lock or unlock once.");

        m_lockOp     = Scheduling::LockOperation::Unlock;
        m_dependency = dependency;
        addComment(std::move(comment));
        return *this;
    }

    inline constexpr Scheduling::LockOperation Instruction::getLockValue() const
    {
        return m_lockOp;
    }

    inline constexpr Scheduling::Dependency Instruction::getDependency() const
    {
        return m_dependency;
    }

    inline void Instruction::addWaitCount(WaitCount const& wait)
    {
        m_waitCount.combine(wait);
    }

    inline void Instruction::addComment(std::string const& comment)
    {
        if(!comment.empty())
        {
            m_comments.push_back(comment);
        }
    }

    inline void Instruction::addWarning(std::string const& warning)
    {
        m_warnings.push_back(warning);
    }

    inline void Instruction::addNop()
    {
        addNop(1);
    }

    inline void Instruction::addNop(int count)
    {
        m_nopCount += count;
    }

    inline void Instruction::setNopMin(int count)
    {
        m_nopCount = std::max(m_nopCount, count);
    }

    inline void Instruction::setModeRegister(uint8_t mode)
    {
        m_MODESetValue = mode;
    }

    inline std::optional<uint8_t> Instruction::getModeRegister() const
    {
        return m_MODESetValue;
    }

    inline void Instruction::allocateNow()
    {
        // Before allocating the destination register(s), assert that the source register(s)
        // are allocated. This prevents us from accidentally using an unallocated register as
        // both a source and a destination:
        for(auto& s : m_src)
        {
            if(s)
            {
                s->assertCanUseAsOperand();
            }
        }

        for(auto& a : m_allocations)
        {
            if(a)
            {
                if(!m_controlOps.empty())
                    a->setControlOp(m_controlOps.front());

                a->allocateNow();
            }
        }
    }

    inline Instruction::AllocationArray Instruction::allocations() const
    {
        return m_allocations;
    }
}
