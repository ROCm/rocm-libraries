/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

#include "../InstructionValues/Register.hpp"
#include "../Utilities/Error.hpp"

namespace rocRoller
{
    inline Generator<std::string> Instruction::EscapeComment(std::string comment, int indent)
    {
        std::string prefix;
        for(int i = 0; i < indent; i++)
        {
            prefix += " ";
        }
        prefix += "// ";

        size_t beginIndex = 0;
        for(size_t idx = 0; idx < comment.size(); idx++)
        {
            if(comment[idx] == '\n')
            {
                auto n = (idx + 1) - beginIndex;
                co_yield(prefix + comment.substr(beginIndex, n));
                beginIndex = idx + 1;
            }
        }

        if(beginIndex < comment.size())
        {
            auto n = comment.size() - beginIndex;
            co_yield(prefix + comment.substr(beginIndex, n));
        }
    }

    inline Instruction::Instruction() = default;

    inline Instruction::Instruction(std::string const&                                      opcode,
                                    std::initializer_list<std::shared_ptr<Register::Value>> dst,
                                    std::initializer_list<std::shared_ptr<Register::Value>> src,
                                    std::initializer_list<std::string> modifiers,
                                    std::string const&                 comment)
        : m_opcode(opcode)
        , m_comments({comment})
    {
        AssertFatal(dst.size() <= m_dst.size(), ShowValue(dst.size()), ShowValue(m_dst.size()));
        AssertFatal(src.size() <= m_src.size(), ShowValue(src.size()), ShowValue(m_src.size()));
        AssertFatal(modifiers.size() <= m_modifiers.size(),
                    ShowValue(modifiers.size()),
                    ShowValue(m_modifiers.size()));

        std::copy(dst.begin(), dst.end(), m_dst.begin());
        std::copy(src.begin(), src.end(), m_src.begin());
        std::copy(modifiers.begin(), modifiers.end(), m_modifiers.begin());

        for(auto& dst : m_dst)
        {
            if(dst && dst->allocationState() == Register::AllocationState::Unallocated)
            {
                dst->allocate(*this);
            }
        }
    }

    inline Instruction Instruction::Allocate(std::shared_ptr<Register::Value> reg)
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
        if(!comment.empty())
        {
            rv.m_comments = {comment};
        }
        return rv;
    }

    inline Instruction Instruction::Comment(std::string const& comment)
    {
        Instruction rv;

        rv.m_comments = {comment};
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
        if(!comment.empty())
        {
            rv.addComment(comment);
        }

        return rv;
    }

    inline Instruction Instruction::Label(Register::ValuePtr label)
    {
        return Label(label->getLabel());
    }

    inline Instruction Instruction::Label(std::string const& name)
    {
        Instruction rv;
        rv.m_label = name;
        return rv;
    }

    inline Instruction Instruction::Label(std::string&& name)
    {
        Instruction rv;
        rv.m_label = std::move(name);
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

    inline Instruction Instruction::Lock(Scheduling::Dependency const& dependency,
                                         std::string                   comment = "")
    {
        AssertFatal(dependency != Scheduling::Dependency::Unlock
                        && dependency != Scheduling::Dependency::Count,
                    "Can not create lock instruction with Unlock or Count dependency");

        Instruction rv;
        rv.m_dependency = dependency;
        if(!comment.empty())
        {
            rv.m_comments.push_back(comment);
        }
        return rv;
    }

    inline Instruction Instruction::Unlock(std::string comment = "")
    {
        Instruction rv;
        rv.m_dependency = Scheduling::Dependency::Unlock;
        if(!comment.empty())
        {
            rv.m_comments.push_back(comment);
        }
        return rv;
    }

    inline std::tuple<std::vector<std::shared_ptr<Register::Value>>,
                      std::vector<std::shared_ptr<Register::Value>>>
        Instruction::getRegisters() const
    {
        std::vector<std::shared_ptr<Register::Value>> retval_src;
        for(auto& v : m_src)
        {
            if(v)
            {
                retval_src.push_back(v);
            }
        }
        std::vector<std::shared_ptr<Register::Value>> retval_dst;
        for(auto& v : m_dst)
        {
            if(v)
            {
                retval_dst.push_back(v);
            }
        }
        return std::make_tuple(retval_src, retval_dst);
    }

    inline bool Instruction::hasRegisters() const
    {
        return (m_src[0] || m_dst[0]);
    }

    inline WaitCount Instruction::getWaitCount() const
    {
        return m_waitCount;
    }

    //Returns |a.src n b.dest| > 0 or |a.dest n (b.src u b.dest)| > 0
    inline bool
        Instruction::registersIntersect(std::vector<std::shared_ptr<Register::Value>> dst,
                                        std::vector<std::shared_ptr<Register::Value>> src) const
    {
        for(auto& regA : m_src)
        {
            if(regA)
            {
                for(auto& regB : dst)
                {
                    if(regA->intersects(regB))
                    {
                        return true;
                    }
                }
            }
        }
        for(auto& regA : m_dst)
        {
            if(regA)
            {
                for(auto& regB : src)
                {
                    if(regA->intersects(regB))
                    {
                        return true;
                    }
                }
                for(auto& regB : dst)
                {
                    if(regA->intersects(regB))
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    inline void Instruction::toStream(std::ostream& os, Settings::LogLevel level) const
    {
        preambleString(os, level);
        functionalString(os, level);
        codaString(os, level);
    }

    inline std::string Instruction::toString(Settings::LogLevel level) const
    {
        std::ostringstream oss;
        toStream(oss, level);
        return oss.str();
    }

    inline void Instruction::preambleString(std::ostream& os, Settings::LogLevel level) const
    {
        if(level >= Settings::LogLevel::Warning)
        {
            for(auto const& w : m_warnings)
            {
                for(auto const& s : EscapeComment(w))
                {
                    os << s;
                }
                os << "\n";
            }
        }
        allocationString(os, level);
    }

    inline void Instruction::directiveString(std::ostream& os, Settings::LogLevel level) const
    {
        os << m_directive;
    }

    inline void Instruction::functionalString(std::ostream& os, Settings::LogLevel level) const
    {
        auto pos = os.tellp();

        if(!m_label.empty())
        {
            os << m_label << ":\n";
        }

        directiveString(os, level);
        m_waitCount.toStream(os, level);

        if(m_nopCount > 0)
        {
            int count = m_nopCount;
            while(count > 16)
            {
                // s_nop can only handle values from 0 to 0xf
                os << "s_nop 0xf\n";
                count -= 16;
            }
            // Note: "s_nop 0" is 1 nop, "s_nop 0xf" is 16 nops
            os << "s_nop " << (count - 1) << "\n";
        }

        coreInstructionString(os);

        if(level > Settings::LogLevel::Terse && !m_comments.empty())
        {
            for(auto commentsIter = m_comments.begin(); commentsIter != m_comments.end();
                commentsIter++)
            {
                if(commentsIter != m_comments.begin())
                {
                    os << "\n";
                }
                for(auto const& s : EscapeComment(*commentsIter, 1))
                {
                    os << s;
                }
            }
        }

        if(pos != os.tellp())
        {
            os << "\n";
        }
    }

    inline void Instruction::codaString(std::ostream& os, Settings::LogLevel level) const
    {
        if(level >= Settings::LogLevel::Terse && m_comments.size() > 1)
        {
            for(auto const& c : m_comments)
            {
                for(auto const& line : EscapeComment(c))
                {
                    os << line;
                }
                os << "\n";
            }
        }
    }

    inline void Instruction::allocationString(std::ostream& os, Settings::LogLevel level) const
    {
        if(level > Settings::LogLevel::Terse)
        {
            for(auto const& alloc : m_allocations)
            {
                if(alloc)
                {
                    for(auto const& line : EscapeComment(alloc->descriptiveComment("Allocated")))
                    {
                        os << line;
                    }
                }
            }
        }
    }

    inline std::string Instruction::getOpCode() const
    {
        return m_opcode;
    }

    inline void Instruction::coreInstructionString(std::ostream& os) const
    {
        if(m_opcode.empty())
        {
            return;
        }

        os << m_opcode << " ";

        bool firstDstArg = true;
        for(auto const& dst : m_dst)
        {
            if(dst)
            {
                if(!firstDstArg)
                {
                    os << ", ";
                }
                dst->toStream(os);
                firstDstArg = false;
            }
        }

        for(auto const& src : m_src)
        {
            if(src && !firstDstArg)
            {
                os << ", ";
                break;
            }
        }

        bool firstSrcArg = true;
        for(auto const& src : m_src)
        {
            if(src)
            {
                if(!firstSrcArg)
                {
                    os << ", ";
                }
                src->toStream(os);
                firstSrcArg = false;
            }
        }

        for(std::string const& mod : m_modifiers)
        {
            if(!mod.empty())
            {
                os << " " << mod;
            }
        }
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

    inline Instruction Instruction::lock(Scheduling::Dependency const& dependency,
                                         std::string                   comment = "")
    {
        AssertFatal(dependency != Scheduling::Dependency::Unlock
                        && dependency != Scheduling::Dependency::Count,
                    "Can not create lock instruction with Unlock or Count dependency");

        m_dependency = dependency;
        if(!comment.empty())
        {
            m_comments.push_back(comment);
        }
        return *this;
    }

    inline Instruction Instruction::unlock(std::string comment = "")
    {
        m_dependency = Scheduling::Dependency::Unlock;
        if(!comment.empty())
        {
            m_comments.push_back(comment);
        }
        return *this;
    }

    inline int Instruction::getLockValue() const
    {
        if(m_dependency == Scheduling::Dependency::Unlock)
        {
            return -1;
        }
        else if(m_dependency == Scheduling::Dependency::None)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }

    inline Scheduling::Dependency Instruction::getDependency() const
    {
        return m_dependency;
    }

    inline void Instruction::addWaitCount(WaitCount const& wait)
    {
        m_waitCount.combine(wait);
    }

    inline void Instruction::addComment(std::string const& comment)
    {
        m_comments.push_back(comment);
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

    inline void Instruction::allocateNow()
    {
        for(auto& a : m_allocations)
        {
            if(a)
            {
                a->allocateNow();
            }
        }
    }
}
