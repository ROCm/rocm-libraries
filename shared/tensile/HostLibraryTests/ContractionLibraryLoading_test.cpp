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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "TestData.hpp"

using namespace Tensile;

TEST(ContractionLibraryLoadingTest, MultipleKernels)
{
    auto library = LoadLibraryFile<ContractionProblem>(
        TestData::Instance().file("SampleTensileKernels").native());
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Ailk_Bljk_SB_MT128x128x08_K1");
    }

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, true, 4, 4, 4, 4, 4, 4, 1.5, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Ailk_Bjlk_SB_MT128x128x08_K1");
    }

    {
        ContractionProblem p
            = ContractionProblem::GEMM(true, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Alik_Bljk_SB_MT128x128x08_K1");
    }

    {
        ContractionProblem p
            = ContractionProblem::GEMM(true, true, 4, 4, 4, 4, 4, 4, 1.5, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Alik_Bjlk_SB_MT128x128x08_K1");
    }
}

TEST(ContractionLibraryLoadingTest, SGEMM_Kernels_Lite)
{
    auto library
        = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("KernelsLite").native());
    ASSERT_NE(library, nullptr);
}
