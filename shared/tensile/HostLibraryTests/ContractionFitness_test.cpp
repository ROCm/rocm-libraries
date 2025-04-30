/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "TestData.hpp"

#include <tuple>

using namespace Tensile;

TEST(ContractionFitnessTest, MatchingSize)
{
    auto library
        = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("KernelsLite").native());
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, false, 64, 64, 256, 64, 64, 256, 1.0, false, 2);

        double fitness  = -1.0; //Initialize to fail test
        auto   solution = library->findBestSolution(p, hardware, &fitness);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(fitness, 0.0);
    }
}

TEST(ContractionFitnessTest, NonMatchingSize)
{
    auto library
        = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("KernelsLite").native());
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, false, 65, 64, 256, 65, 64, 256, 1.0, false, 2);

        double fitness  = 0.0; //Initialize to fail test
        auto   solution = library->findBestSolution(p, hardware, &fitness);

        ASSERT_NE(solution, nullptr);
        EXPECT_NE(fitness, 0.0);
    }
}
