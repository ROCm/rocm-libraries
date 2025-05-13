/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
            TTMP7,
            TTMP9,
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
