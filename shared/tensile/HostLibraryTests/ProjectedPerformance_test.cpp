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
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <cstddef>

using namespace Tensile;

std::map<int, double> makeIdeals()
{
    std::map<int, double> ideals;
    ideals.insert(std::make_pair(32, 2000.0));
    ideals.insert(std::make_pair(64, 3000.0));
    ideals.insert(std::make_pair(128, 4000.0));
    ideals.insert(std::make_pair(256, 5000.0));
    ideals.insert(std::make_pair(512, 6000.0));
    ideals.insert(std::make_pair(1024, 7000.0));
    ideals.insert(std::make_pair(2048, 8000.0));
    ideals.insert(std::make_pair(4096, 9000.0));
    ideals.insert(std::make_pair(8192, 10000.0));
    ideals.insert(std::make_pair(16192, 11000.0));

    return ideals;
}

SizeMapping makeSizeMapping(Tensile::dim3 workGroupSize,
                            Tensile::dim3 threadTile,
                            Tensile::dim3 macroTile,
                            size_t        globalSplitU)
{
    SizeMapping sizeMapping;

    sizeMapping.workGroupSize = workGroupSize;
    sizeMapping.threadTile    = threadTile;
    sizeMapping.macroTile     = macroTile;

    sizeMapping.staggerU           = 32;
    sizeMapping.depthU             = 32;
    sizeMapping.globalSplitU       = globalSplitU;
    sizeMapping.staggerStrideShift = 4;
    sizeMapping.workGroupMapping   = 8;

    sizeMapping.persistentKernel           = 0;
    sizeMapping.persistentKernelAlongBatch = false;

    sizeMapping.globalAccumulation    = 0;
    sizeMapping.workspaceSizePerElemC = 0;

    return sizeMapping;
}

TEST(ContractionPerformance, Problem1)
{
    auto solution = std::make_shared<ContractionSolution>();

    ASSERT_NE(solution, nullptr);

    std::map<int, double> ideals = makeIdeals();

    solution->ideals = ideals;

    Tensile::dim3 workgroupSize = Tensile::dim3(16, 16, 1);
    Tensile::dim3 threadTile    = Tensile::dim3(4, 4, 0);
    Tensile::dim3 macroTile     = Tensile::dim3(64, 64, 16);
    size_t        globalSplitU  = 1;

    SizeMapping sizeMapping = makeSizeMapping(workgroupSize, threadTile, macroTile, globalSplitU);

    solution->sizeMapping = sizeMapping;

    auto problem
        = ContractionProblem::GEMM(false, false, 1536, 1536, 64, 1536, 64, 1536, 1.5, false, 1.0);

    AMDGPU hardware(Tensile::AMDGPU::Processor::gfx906, 64, 0, "gfx906");
    double perf = solution->projectedPerformance(problem, hardware).speedGFlops;

    ASSERT_DOUBLE_EQ(perf, 3000.0);
}

TEST(ContractionPerformance, Problem2)
{
    auto solution = std::make_shared<ContractionSolution>();

    ASSERT_NE(solution, nullptr);

    std::map<int, double> ideals = makeIdeals();

    solution->ideals = ideals;

    Tensile::dim3 workgroupSize = Tensile::dim3(16, 16, 1);
    Tensile::dim3 threadTile    = Tensile::dim3(4, 4, 0);
    Tensile::dim3 macroTile     = Tensile::dim3(64, 64, 16);
    size_t        globalSplitU  = 1;

    SizeMapping sizeMapping = makeSizeMapping(workgroupSize, threadTile, macroTile, globalSplitU);

    solution->sizeMapping = sizeMapping;

    auto problem
        = ContractionProblem::GEMM(false, false, 384, 192, 60, 384, 60, 384, 1.5, false, 1.0);

    AMDGPU hardware(Tensile::AMDGPU::Processor::gfx906, 64, 0, "gfx906");
    double perf = solution->projectedPerformance(problem, hardware).speedGFlops;

    ASSERT_DOUBLE_EQ(perf, 843.75);
}

TEST(ContractionPerformance, Problem3)
{
    auto solution = std::make_shared<ContractionSolution>();

    ASSERT_NE(solution, nullptr);

    std::map<int, double> ideals = makeIdeals();

    solution->ideals = ideals;

    Tensile::dim3 workgroupSize = Tensile::dim3(16, 16, 2);
    Tensile::dim3 threadTile    = Tensile::dim3(8, 8, 0);
    Tensile::dim3 macroTile     = Tensile::dim3(128, 128, 16);
    size_t        globalSplitU  = 1;

    SizeMapping sizeMapping = makeSizeMapping(workgroupSize, threadTile, macroTile, globalSplitU);

    solution->sizeMapping = sizeMapping;

    auto problem
        = ContractionProblem::GEMM(false, false, 384, 192, 60, 384, 60, 384, 1.5, false, 1.0);

    AMDGPU hardware(Tensile::AMDGPU::Processor::gfx906, 64, 0, "gfx906");
    auto   model = solution->projectedPerformance(problem, hardware);

    // std::cout << model << "\n";

    ASSERT_DOUBLE_EQ(model.granularities.tilesPerCu, .1875);
    ASSERT_DOUBLE_EQ(model.granularities.totalGranularity, .140625);
    ASSERT_DOUBLE_EQ(model.speedGFlops, 421.875);
}

TEST(ContractionPerformance, Problem4)
{
    auto solution = std::make_shared<ContractionSolution>();

    ASSERT_NE(solution, nullptr);

    std::map<int, double> ideals = makeIdeals();

    solution->ideals = ideals;

    Tensile::dim3 workgroupSize = Tensile::dim3(16, 16, 4);
    Tensile::dim3 threadTile    = Tensile::dim3(8, 4, 0);
    Tensile::dim3 macroTile     = Tensile::dim3(128, 64, 16);
    size_t        globalSplitU  = 4;

    SizeMapping sizeMapping = makeSizeMapping(workgroupSize, threadTile, macroTile, globalSplitU);

    solution->sizeMapping = sizeMapping;

    auto problem
        = ContractionProblem::GEMM(false, false, 1536, 1575, 64, 1536, 64, 1536, 1.5, false, 3.0);

    AMDGPU hardware(Tensile::AMDGPU::Processor::gfx906, 64, 0, "gfx906");
    auto   model = solution->projectedPerformance(problem, hardware);

    // std::cout << model << "\n";

    ASSERT_DOUBLE_EQ(model.granularities.tilesPerCu, 225);
    ASSERT_NEAR(model.granularities.totalGranularity, .984375, .001);
    ASSERT_DOUBLE_EQ(model.speedGFlops, 2953.125);
}
