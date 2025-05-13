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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include <catch2/matchers/catch_matchers_string.hpp>

#include "CustomSections.hpp"
#include "TestContext.hpp"

#include <common/SourceMatcher.hpp>

using namespace rocRoller;

TEST_CASE("LDSAllocation intersects() works.", "[codegen][lds]")
{
#define CHECK_INTERSECTS(a, b) \
    CHECK(a->intersects(b));   \
    CHECK(b->intersects(a))

#define CHECK_DOESNT_INTERSECT(a, b) \
    CHECK_FALSE(a->intersects(b));   \
    CHECK_FALSE(b->intersects(a))

    auto allocA = std::make_shared<LDSAllocation>(12, 0);
    auto allocB = std::make_shared<LDSAllocation>(1, 0);
    auto allocC = std::make_shared<LDSAllocation>(1, 12);
    auto allocD = std::make_shared<LDSAllocation>(12, 12);
    auto allocE = std::make_shared<LDSAllocation>(12, 6);

    CHECK_INTERSECTS(allocA, allocA);
    CHECK_INTERSECTS(allocA, allocB);
    CHECK_DOESNT_INTERSECT(allocA, allocC);
    CHECK_DOESNT_INTERSECT(allocA, allocD);
    CHECK_INTERSECTS(allocA, allocE);

    CHECK_INTERSECTS(allocB, allocB);
    CHECK_DOESNT_INTERSECT(allocB, allocC);
    CHECK_DOESNT_INTERSECT(allocB, allocD);
    CHECK_DOESNT_INTERSECT(allocB, allocE);

    CHECK_INTERSECTS(allocC, allocC);
    CHECK_INTERSECTS(allocC, allocD);
    CHECK_INTERSECTS(allocC, allocE);

    CHECK_INTERSECTS(allocD, allocD);
    CHECK_INTERSECTS(allocD, allocE);

    CHECK_INTERSECTS(allocE, allocE);
}
