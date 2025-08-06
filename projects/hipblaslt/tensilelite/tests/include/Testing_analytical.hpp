/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/analytical/AnalyticalGemm.hpp>
#include <Tensile/analytical/Hardware.hpp>
#include <Tensile/analytical/Utils.hpp>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

struct InputWithExpected {
    std::map<std::string, int> values;
    int expected;
};

struct MyTestData {
    std::string name;
    std::vector<InputWithExpected> inputs;
};

// Parameterized test class declaration
class AnalyticalGtest : public ::testing::TestWithParam<MyTestData> {};

void ComputeLoads(int MT_M, int MT_N, int MT_K, int expected, bool debug = false)
{
    auto a_loads  = TensileLite::analytical::compute_A_loads(MT_M, MT_K, debug);
    auto b_loads  = TensileLite::analytical::compute_B_loads(MT_N, MT_K, debug);
    auto cu_loads = TensileLite::analytical::compute_CU_loads(MT_M, MT_N, MT_K, debug);
    EXPECT_EQ(a_loads, expected);
    EXPECT_EQ(b_loads, expected);
    EXPECT_EQ(cu_loads, a_loads + b_loads);
}
