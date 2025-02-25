#pragma once

#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

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

            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
        };
    }
} // namespace rocRoller
