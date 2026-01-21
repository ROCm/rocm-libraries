// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>

namespace rocRoller
{
    ContextPtr Instruction::findContextFromOperands() const
    {
        // TODO: Reevalute this method for retrieving/passing the context.
        for(auto const& reg : getAllOperands())
            if(reg->context())
                return reg->context();

        return nullptr;
    }

    bool Instruction::requiresVnopForHazard() const
    {
        auto ctx = findContextFromOperands();

        return ctx && ctx->targetArchitecture().target().isGFX12GPU();
    }

    void Instruction::codaString(std::ostream& os, LogLevel level) const
    {
        if(m_lockOp != Scheduling::LockOperation::None && level >= LogLevel::Verbose)
        {
            os << " // " << m_lockOp << " " << m_dependency << std::endl;
        }

        if(level >= LogLevel::Terse && m_comments.size() > 1)
        {
            // Only include everything but the first comment in the coda string.
            for(int i = 1; i < m_comments.size(); i++)
            {
                for(auto const& line : EscapeComment(m_comments[i]))
                {
                    os << line;
                }
                os << "\n";
            }
        }

        if(level >= LogLevel::Info)
        {
            if(m_extraDsts[0])
            {
                std::string comment = "Extra dsts:";
                for(auto const& dst : m_extraDsts)
                {
                    if(dst)
                        comment += " " + dst->description();
                }
                for(auto const& line : EscapeComment(comment))
                    os << line;
                os << "\n";
            }

            if(m_extraSrcs[0])
            {
                std::string comment = "Extra srcs:";
                for(auto const& src : m_extraSrcs)
                {
                    if(src)
                        comment += " " + src->description();
                }
                for(auto const& line : EscapeComment(comment))
                    os << line;
                os << "\n";
            }
        }

        if(level >= LogLevel::Trace)
        {
            auto ctx = findContextFromOperands();
            if(ctx)
            {
                auto status = ctx->observer()->peek(*this);
                for(auto const& line : EscapeComment(m_peekedStatus.toString()))
                    os << line;
                os << "\n";

                auto category = GPUInstructionInfo::getCoexecCategory(m_opcode);
                for(auto const& line : EscapeComment("Category: " + rocRoller::toString(category)))
                    os << line;
                os << "\n";
            }
        }
    }

    void Instruction::addControlOp(int id)
    {
        m_controlOps.push_back(id);
    }

    CoexecCategory Instruction::getCategory() const
    {
        auto category = GPUInstructionInfo::getCoexecCategory(m_opcode);

        if(category == CoexecCategory::NotAnInstruction && getWaitCount() != WaitCount())
        {
            category = CoexecCategory::Scalar;
        }

        return category;
    }

    std::vector<int> const& Instruction::controlOps() const
    {
        return m_controlOps;
    }

    std::optional<int> Instruction::innerControlOp() const
    {
        if(!m_controlOps.empty())
            return m_controlOps.front();

        return std::nullopt;
    }

    void Instruction::setReferencedArg(std::string arg)
    {
        m_referencedArg = std::move(arg);
    }

    std::string const& Instruction::referencedArg() const
    {
        return m_referencedArg;
    }

    std::vector<std::string> const& Instruction::comments() const
    {
        return m_comments;
    }

    Scheduling::InstructionStatus const& Instruction::peekedStatus() const
    {
        return m_peekedStatus;
    }

    void Instruction::setPeekedStatus(Scheduling::InstructionStatus status)
    {
        m_peekedStatus = std::move(status);
    }
}
