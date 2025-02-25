#pragma once

#include "rocRoller/Operations/CommandArguments.hpp"

namespace rocRoller
{
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
        AssertFatal(itr != m_argOffsetMapPtr->end(), "command argument not found");

        m_kArgs.writeValue(itr->second, value);
    }

    template <CCommandArgumentValue T>
    void CommandArguments::setArgument(Operations::OperationTag op, ArgumentType argType, T value)
    {
        setArgument(op, argType, -1, value);
    }

    inline RuntimeArguments CommandArguments::runtimeArguments() const
    {
        return m_kArgs.runtimeArguments();
    }
}
