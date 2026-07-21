/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include "HipBackend.hpp"
#include "RunGEMMKernelTest.hpp"

using namespace Tensile;

// Create kernel tests and inputs with HIP backend
using RunGEMMKernelTestHip       = RunGEMMKernelTest<HipBackend>;
using RunGEMMKernelTestHipParams = RunGEMMKernelTestParams<HipBackend>;

// Define which tests we want to run with HIP
TEST_P(RunGEMMKernelTestHip, TestBestSolution)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTestHip, TestAllSolutions)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->TestAllSolutions();
}

TEST_P(RunGEMMKernelTestHip, TestAlphaZero)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->OverrideAlpha(0.0);
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTestHip, TestAlphaZeroSigned)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->OverrideAlpha(std::copysign(0.0, -1.0));
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTestHip, TestAlphaZeroABNull)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->OverrideAlpha(0.0);
    typedTest->NullifyAPtr();
    typedTest->NullifyBPtr();
    auto fail = false;
    try
    {
        ASSERT_NO_THROW();
    }
    catch(...)
    {
        fail = true;
    }
    ASSERT_EQ(fail, false);
}

// Enqueue the hip tests
INSTANTIATE_TEST_SUITE_P(
    HipSolutionAdapter,
    RunGEMMKernelTestHip,
    ::testing::Combine(::testing::ValuesIn(RunGEMMKernelTestHipParams::TypedTests()),
                       ::testing::ValuesIn(RunGEMMKernelTestHipParams::TestProblems()),
                       ::testing::ValuesIn(RunGEMMKernelTestHipParams::TestLibraries()),
                       ::testing::ValuesIn(RunGEMMKernelTestHipParams::TestMemoryAlignments())));

INSTANTIATE_TEST_SUITE_P(
    HipSolutionAdapter_Extended,
    RunGEMMKernelTestHip,
    ::testing::Combine(::testing::ValuesIn(RunGEMMKernelTestHipParams::TypedTests()),
                       ::testing::ValuesIn(RunGEMMKernelTestHipParams::TestProblemsExtended()),
                       ::testing::ValuesIn(RunGEMMKernelTestHipParams::TestLibraries()),
                       ::testing::ValuesIn(RunGEMMKernelTestHipParams::TestMemoryAlignments())));
