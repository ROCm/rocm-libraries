#include "rocRoller/Operations/Operation.hpp"

namespace rocRoller
{
    namespace Operations
    {
        BaseOperation::BaseOperation() {}

        void BaseOperation::setCommand(CommandPtr command)
        {
            m_command = command;
        }

        int BaseOperation::getTag() const
        {
            return m_tag;
        }

        void BaseOperation::setTag(int tag)
        {
            m_tag = tag;
        }

    }
}
