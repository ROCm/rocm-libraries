/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <common/Utilities.hpp>
#include <common/mxDataGen.hpp>
#include <rocRoller/TensorDescriptor.hpp>

using Catch::Matchers::ContainsSubstring;

TEST_CASE("has openmp tag", "[OPENMP]")
{
    SECTION("utilities", "")
    {
        REQUIRE_NOTHROW(normL2(std::vector<int>{1, 2, 3}));
    }

    SECTION("DGenInput", "")
    {
        std::vector<float> hostA, hostB, hostC;
        std::vector<uint8_t> hostScaleA, hostScaleB;

        rocRoller::TensorDescriptor descA(rocRoller::DataType::Float, {64, 32}, "N");
        rocRoller::TensorDescriptor descB(rocRoller::DataType::Float, {32, 64}, "N");
        rocRoller::TensorDescriptor descC(rocRoller::DataType::Float, {64, 64}, "N");

        REQUIRE_NOTHROW(rocRoller::DGenInput(12345u, hostA, descA, hostB, descB, hostC, descC, hostScaleA, hostScaleB));
    }
}

TEST_CASE("doesn't have openmp tag", "")
{
    SECTION("utilities", "")
    {
        REQUIRE_THROWS_WITH(normL2(std::vector<int>{1, 2, 3}), ContainsSubstring("omp"));
    }

    SECTION("DGenInput", "")
    {
        std::vector<float> hostA, hostB, hostC;
        std::vector<uint8_t> hostScaleA, hostScaleB;

        rocRoller::TensorDescriptor descA(rocRoller::DataType::Float, {64, 32}, "N");
        rocRoller::TensorDescriptor descB(rocRoller::DataType::Float, {32, 64}, "N");
        rocRoller::TensorDescriptor descC(rocRoller::DataType::Float, {64, 64}, "N");

        REQUIRE_THROWS_WITH(rocRoller::DGenInput(12345u, hostA, descA, hostB, descB, hostC, descC, hostScaleA, hostScaleB), ContainsSubstring("omp"));
    }
}