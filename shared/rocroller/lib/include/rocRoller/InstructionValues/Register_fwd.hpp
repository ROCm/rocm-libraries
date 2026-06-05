// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            EXECZ,
            EXEC,
            EXEC_LO,
            EXEC_HI,
            TTMP7,
            TTMP9,
            Constant,
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

        enum
        {
            /// Contiguity equal to element count
            FULLY_CONTIGUOUS = -3,

            /// Suffucient contiguity to fit the datatype
            VALUE_CONTIGUOUS = -2,

            /// Won't be using allocator
            /// (e.g. taking allocation from elsewhere or assigning particular register numbers)
            MANUAL = -1,

            Count = 255,
        };

        struct AllocationOptions
        {
            /// In units of registers
            int contiguousChunkWidth = VALUE_CONTIGUOUS;

            /// Allocation x must have (x % alignment) == alignmentPhase. -1 means to use default for register type.
            int alignment      = -1;
            int alignmentPhase = 0;

            static AllocationOptions FullyContiguous();

            auto operator<=>(AllocationOptions const& other) const = default;
        };

        std::string toString(RegisterId const& regId);

        // For some reason, GCC will not find the operator declared in Utils.hpp.
        std::ostream& operator<<(std::ostream& stream, RegisterId const& regId);

    }

}
