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

        OperationTag BaseOperation::getTag() const
        {
            return m_tag;
        }

        void BaseOperation::setTag(OperationTag tag)
        {
            m_tag = tag;
        }

    }
}
