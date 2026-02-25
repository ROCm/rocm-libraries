// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
