// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Operations/Operation.hpp>
#include <rocRoller/Operations/Operations.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace Operations
    {
        std::string name(Operation const& x)
        {
            return std::visit([](auto y) { return name<decltype(y)>(); }, x);
        }

        BaseOperation::BaseOperation() {}

        void BaseOperation::setCommand(CommandPtr command)
        {
            m_command = command;
        }

        OperationTag BaseOperation::getTag() const
        {
            return m_tag;
        }

        void BaseOperation::setTag(OperationTag tag)
        {
            m_tag = tag;
        }

        std::strong_ordering BaseOperation::operator<=>(BaseOperation const& other) const
        {
            return m_tag <=> other.m_tag;
        }

    }
}
