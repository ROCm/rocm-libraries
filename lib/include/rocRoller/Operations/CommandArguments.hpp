#pragma once

#include "CommandArgument_fwd.hpp"
#include "CommandArguments_fwd.hpp"
#include "OperationTag.hpp"
#include <rocRoller/KernelArguments.hpp>

namespace rocRoller
{
    class CommandArguments
    {
    public:
        CommandArguments() = delete;
        CommandArguments(ArgumentOffsetMapPtr, int);

        template <CCommandArgumentValue T>
        void setArgument(Operations::OperationTag op, ArgumentType argType, int dimension, T value);
        template <CCommandArgumentValue T>
        void setArgument(Operations::OperationTag op, ArgumentType argType, T value);

        RuntimeArguments runtimeArguments() const;

    private:
        ArgumentOffsetMapPtr m_argOffsetMapPtr;
        KernelArguments      m_kArgs;
    };
}

#include "CommandArguments_impl.hpp"
