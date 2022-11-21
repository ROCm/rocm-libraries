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
            Count // Always last enum entry
        };
        enum class SpecialType
        {
            SCC = 0,
            VCC,
            VCC_LO,
            VCC_HI,
            EXEC,
            Count // Always last enum entry
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

        struct RegisterId;
        struct RegisterIdHash;

        class Allocation;
        struct Value;

        using AllocationPtr = std::shared_ptr<Allocation>;
        using ValuePtr      = std::shared_ptr<Value>;
    }

}
