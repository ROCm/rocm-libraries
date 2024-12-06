#pragma once

#include <rocRoller/Operations/CommandArguments.hpp>

namespace rocRoller
{
    inline std::string toString(ArgumentType argType)
    {
        switch(argType)
        {
        case ArgumentType::Value:
            return "Value";
        case ArgumentType::Limit:
            return "Limit";
        case ArgumentType::Size:
            return "Size";
        case ArgumentType::Stride:
            return "Stride";
        default:
            break;
        }
        throw std::runtime_error("Invalid ArgumentType");
    }

    inline std::ostream& operator<<(std::ostream& stream, ArgumentType argType)
    {
        return stream << toString(argType);
    }

    inline CommandArguments::CommandArguments(ArgumentOffsetMapPtr argOffsetMapPtr, int bytes)
        : m_argOffsetMapPtr(argOffsetMapPtr)
        , m_kArgs(false, bytes)
    {
    }

    template <CCommandArgumentValue T>
    void CommandArguments::setArgument(Operations::OperationTag op,
                                       ArgumentType             argType,
                                       int                      dim,
                                       T                        value)
    {
        auto itr = m_argOffsetMapPtr->find(std::make_tuple(op, argType, dim));
        AssertFatal(itr != m_argOffsetMapPtr->end(),
                    "Command argument not found.",
                    ShowValue(op),
                    ShowValue(argType),
                    ShowValue(dim));

        m_kArgs.writeValue(itr->second, value);
    }

    template <CCommandArgumentValue T>
    void CommandArguments::setArgument(Operations::OperationTag op, ArgumentType argType, T value)
    {
        return setArgument(op, argType, -1, value);
    }

    inline RuntimeArguments CommandArguments::runtimeArguments() const
    {
        return m_kArgs.runtimeArguments();
    }
}
