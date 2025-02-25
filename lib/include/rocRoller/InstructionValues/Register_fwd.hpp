#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <string>

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
            NullLiteral,
            SCC,
            M0,
            VCC,
            VCC_LO,
            VCC_HI,
            EXEC,
            EXEC_LO,
            EXEC_HI,
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

        std::string   toString(Type t);
        std::ostream& operator<<(std::ostream& stream, Type t);

        std::string   toString(AllocationState state);
        std::ostream& operator<<(std::ostream& stream, AllocationState state);
    }

}
