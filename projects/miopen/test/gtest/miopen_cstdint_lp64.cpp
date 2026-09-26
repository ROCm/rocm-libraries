/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

// Regression test for TheRock#6258 / LP64 typedef conflict in miopen_cstdint.hpp.
//
// On LP64 platforms (64-bit Linux, macOS), <stdint.h> defines uint64_t as
// 'unsigned long', but __hip_internal::uint64_t is 'unsigned long long'.
// These are distinct C++ types despite being the same width, so a typedef
// redefinition is a hard error when both are visible.

#include <gtest/gtest.h>
#include <cstdint>
#include <type_traits>

#define MIOPEN_HIP_RUNTIME_COMPILE
#define HIP_PACKAGE_VERSION_FLAT 7015026333ULL

// Provide __hip_internal types as they would appear during HIPRTC compilation
namespace __hip_internal {
typedef unsigned long long uint64_t;
typedef signed long long int64_t;
} // namespace __hip_internal

// Include the header under test — this must not conflict with <cstdint> above
#include "miopen_cstdint.hpp"

#undef MIOPEN_HIP_RUNTIME_COMPILE

TEST(MiopenCstdintLP64, Uint64TypeMatchesSystem)
{
    EXPECT_TRUE((std::is_same<::uint64_t, std::uint64_t>::value))
        << "miopen_cstdint.hpp uint64_t must be the same type as <cstdint> uint64_t";
}

TEST(MiopenCstdintLP64, Int64TypeMatchesSystem)
{
    EXPECT_TRUE((std::is_same<::int64_t, std::int64_t>::value))
        << "miopen_cstdint.hpp int64_t must be the same type as <cstdint> int64_t";
}

TEST(MiopenCstdintLP64, SizesAre64Bit)
{
    EXPECT_EQ(sizeof(::uint64_t), 8u);
    EXPECT_EQ(sizeof(::int64_t), 8u);
}
