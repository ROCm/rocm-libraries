#pragma once

#include "Command_fwd.hpp"

#include <memory>

namespace rocRoller
{
    namespace Operations
    {
        class BaseOperation
        {
        public:
            BaseOperation();

            void setCommand(CommandPtr);
            int  getTag() const;
            void setTag(int tag);

        protected:
            int                    m_tag = -1;
            std::weak_ptr<Command> m_command;
        };
    }
}
