/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/Operations/Scratch.hpp>

#include <sstream>

namespace rocRoller
{
    namespace Operations
    {
        Scratch::Scratch(OperationTag tag, ScratchPolicy policy)
            : BaseOperation()
            , m_tag(tag)
            , m_policy(policy)
        {
        }

        OperationTag Scratch::getTag() const
        {
            return m_tag;
        }

        ScratchPolicy Scratch::policy() const
        {
            return m_policy;
        }

        std::string Scratch::toString() const
        {
            std::ostringstream rv;
            rv << "Scratch(" << m_policy << ")";
            return rv.str();
        }

        bool Scratch::operator==(Scratch const& rhs) const
        {
            return m_policy == rhs.m_policy;
        }

        std::string toString(ScratchPolicy const& policy)
        {
            switch(policy)
            {
            case ScratchPolicy::None:
                return "None";
            case ScratchPolicy::ZeroedBeforeAndAfter:
                return "ZeroedBeforeAndAfter";
            case ScratchPolicy::Count:;
            }
            return "Invalid";
        }

        std::ostream& operator<<(std::ostream& stream, ScratchPolicy const& policy)
        {
            return stream << toString(policy);
        }

    }
}
