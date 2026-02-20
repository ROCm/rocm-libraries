// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/CommandArguments_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Utilities/Comparison.hpp>

namespace rocRoller
{
    std::string   toString(ArgumentType);
    std::ostream& operator<<(std::ostream&, ArgumentType);

    using ArgumentOffsetMap
        = std::unordered_map<std::tuple<Operations::OperationTag, ArgumentType, int>, int>;
    using ArgumentOffsetMapPtr = std::shared_ptr<const ArgumentOffsetMap>;

    class CommandArguments
    {
    public:
        CommandArguments() = delete;
        CommandArguments(ArgumentOffsetMapPtr, int);

        template <CCommandArgumentValue T>
        void setArgument(Operations::OperationTag op, ArgumentType argType, int dimension, T value);
        template <CCommandArgumentValue T>
        void setArgument(Operations::OperationTag op, ArgumentType argType, T value);

        void setArgument(Operations::OperationTag    op,
                         ArgumentType                argType,
                         int                         dimension,
                         CommandArgumentValue const& value);
        void setArgument(Operations::OperationTag    op,
                         ArgumentType                argType,
                         CommandArgumentValue const& value);

        RuntimeArguments runtimeArguments() const;

    private:
        ArgumentOffsetMapPtr m_argOffsetMapPtr;
        KernelArguments      m_kArgs;
    };
}

#include <rocRoller/Operations/CommandArguments_impl.hpp>
