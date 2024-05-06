#pragma once

#include "Command_fwd.hpp"
#include "OperationTag.hpp"
#include <memory>

namespace rocRoller
{
    namespace Operations
    {
        class BaseOperation
        {
        public:
            BaseOperation();

            void         setCommand(CommandPtr);
            OperationTag getTag() const;
            void         setTag(OperationTag tag);

        protected:
            OperationTag           m_tag;
            std::weak_ptr<Command> m_command;
        };
    }
} // namespace rocRoller
