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
