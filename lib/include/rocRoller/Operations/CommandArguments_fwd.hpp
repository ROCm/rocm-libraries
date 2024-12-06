#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>

#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Utilities/Comparison.hpp>

namespace rocRoller
{
    enum class ArgumentType : int
    {
        Value = 0,
        Limit,
        Size,
        Stride,

        Count
    };

    using ArgumentOffsetMap
        = std::unordered_map<std::tuple<Operations::OperationTag, ArgumentType, int>, int>;
    using ArgumentOffsetMapPtr = std::shared_ptr<const ArgumentOffsetMap>;
}
