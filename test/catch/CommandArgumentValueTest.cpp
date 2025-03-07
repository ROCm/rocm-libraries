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

#include "CustomMatchers.hpp"
#include "TestContext.hpp"

#include <common/Utilities.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/Utilities/Error.hpp>

TEST_CASE("CommandArgumentValue toString", "[command][toString]")
{
    using cav = rocRoller::CommandArgumentValue;

    SECTION("Values")
    {
        CHECK(toString(cav(5)) == "5");
        CHECK(toString(cav(-5ll)) == "-5");

        CHECK(toString(cav(3643653242342ull)) == "3643653242342");

        CHECK(toString(cav(4.5f)) == "4.50000");
        CHECK(toString(cav(4.5)) == "4.50000");

        CHECK(toString(cav(rocRoller::FP8(4.5))) == "4.50000");
    }

    SECTION("Pointers")
    {
        float x      = 5.f;
        void* ptr    = &x;
        auto  ptrStr = rocRoller::concatenate(ptr);

        CHECK(toString(cav(reinterpret_cast<int32_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<int64_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<uint8_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<uint32_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<uint64_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<float*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<double*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::Half*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::BFloat16*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::FP8*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::BF8*>(ptr))) == ptrStr);
    }
}
