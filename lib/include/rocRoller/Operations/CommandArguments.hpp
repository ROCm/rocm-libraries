#pragma once

#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/CommandArguments_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>

namespace rocRoller
{
    std::string   toString(ArgumentType);
    std::ostream& operator<<(std::ostream&, ArgumentType);

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

#include <rocRoller/Operations/CommandArguments_impl.hpp>
