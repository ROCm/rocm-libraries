#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>

namespace rocRoller
{
    void Instruction::codaString(std::ostream& os, LogLevel level) const
    {
        if(level >= LogLevel::Terse && m_comments.size() > 1)
        {
            // Only include everything but the first comment in the code string.
            for(int i = 1; i < m_comments.size(); i++)
            {
                for(auto const& line : EscapeComment(m_comments[i]))
                {
                    os << line;
                }
                os << "\n";
            }
        }

        ContextPtr ctx;
        for(auto const& r : m_src)
            if(r && r->context())
            {
                ctx = r->context();
                break;
            }
        for(auto const& r : m_dst)
            if(r && r->context())
            {
                ctx = r->context();
                break;
            }

        if(ctx && ctx->kernelOptions().logLevel >= LogLevel::Debug)
        {
            auto status = ctx->observer()->peek(*this);
            for(auto const& line : EscapeComment(status.toString()))
                os << line;
            os << "\n";
        }
    }
}
