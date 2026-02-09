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

#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("getOpselModifiers2xByte works", "[codegen][utils]")
{
    using namespace rocRoller::Arithmetic;

    CHECK(getOpselModifiers2xByte(0, 0) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(0, 1) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(0, 2) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[0,1]"));
    CHECK(getOpselModifiers2xByte(0, 3) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[0,1]"));

    CHECK(getOpselModifiers2xByte(1, 0) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(1, 1) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[0,0]"));
    CHECK(getOpselModifiers2xByte(1, 2) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[0,1]"));
    CHECK(getOpselModifiers2xByte(1, 3) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[0,1]"));

    CHECK(getOpselModifiers2xByte(2, 0) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(2, 1) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(2, 2) == std::make_tuple("op_sel:[0,0]", "op_sel_hi:[1,1]"));
    CHECK(getOpselModifiers2xByte(2, 3) == std::make_tuple("op_sel:[0,1]", "op_sel_hi:[1,1]"));

    CHECK(getOpselModifiers2xByte(3, 0) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(3, 1) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[1,0]"));
    CHECK(getOpselModifiers2xByte(3, 2) == std::make_tuple("op_sel:[1,0]", "op_sel_hi:[1,1]"));
    CHECK(getOpselModifiers2xByte(3, 3) == std::make_tuple("op_sel:[1,1]", "op_sel_hi:[1,1]"));

    CHECK_THROWS(getOpselModifiers2xByte(0, 4));
    CHECK_THROWS(getOpselModifiers2xByte(4, 0));
}
