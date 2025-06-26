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
                for(auto const& line : EscapeComment(status.toString()))
                    os << line;
                os << "\n";
            }
        }
    }

    void Instruction::addControlOp(int id)
    {
        m_controlOps.push_back(id);
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
