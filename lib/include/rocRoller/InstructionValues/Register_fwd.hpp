#pragma once

#include <cassert>
#include <concepts>
#include <memory>

namespace rocRoller
{
    namespace Register
    {
        enum class Type
        {
            Literal = 0,
            Scalar,
            Vector,
            Accumulator,
            LocalData,
            Label,
            Special,
            Count
        };

        enum class AllocationState
        {
            Unallocated = 0,
            Allocated,
            Freed,
            NoAllocation, //< For literals and other values that do not require allocation.
            Error,
            Count
        };

        class Allocation;
        struct Value;

        using AllocationPtr = std::shared_ptr<Value>;
        using ValuePtr      = std::shared_ptr<Value>;
    }

}
