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

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/MatchingLibrary.hpp>

#include <memory>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

TEST(MultipleSolutionsPerSize, ArithmeticUnit)
{
    using namespace Tensile;

    auto SolutionMFMA = std::make_shared<ContractionSolution>();
    auto SolutionVALU = std::make_shared<ContractionSolution>();

    SolutionMFMA->problemPredicate
        = std::make_shared<Predicates::Contraction::ArithmeticUnitCompatible>(ArithmeticUnit::MFMA);
    SolutionVALU->problemPredicate
        = std::make_shared<Predicates::Contraction::ArithmeticUnitCompatible>(ArithmeticUnit::VALU);

    SolutionMFMA->index = 0;
    SolutionVALU->index = 1;

    SolutionMap<ContractionSolution> map({{0, SolutionMFMA}, {1, SolutionVALU}});

    auto LibraryMFMA = std::make_shared<SingleContractionLibrary>(SolutionMFMA);
    auto LibraryVALU = std::make_shared<SingleContractionLibrary>(SolutionVALU);

    AMDGPU gpu;

    auto Problem_Size3 = ContractionProblem::GEMM(false, false, 3, 3, 3, 3, 3, 3, 1.2, false, 1);
    auto Problem_Size5 = ContractionProblem::GEMM(false, false, 5, 5, 5, 5, 5, 5, 1.2, false, 1);
    auto Problem_Size7 = ContractionProblem::GEMM(false, false, 7, 7, 7, 7, 7, 7, 1.2, false, 1);
    auto Problem_Size9 = ContractionProblem::GEMM(false, false, 9, 9, 9, 9, 9, 9, 1.2, false, 1);

    using Key = std::array<int64_t, 4>;
    using Table
        = Matching::DistanceMatchingTable<Key,
                                          ContractionProblem,
                                          std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                                          std::shared_ptr<ContractionSolution>,
                                          Matching::EuclideanDistance<Key>>;
    using Properties = std::vector<std::shared_ptr<Property<ContractionProblem>>>;

    Properties properties;

    {
        auto freeSizeA   = std::make_shared<Contraction::FreeSizeA>();
        freeSizeA->index = 0;
        properties.push_back(freeSizeA);
        auto freeSizeB   = std::make_shared<Contraction::FreeSizeB>();
        freeSizeB->index = 0;
        properties.push_back(freeSizeB);
        auto batchSize   = std::make_shared<Contraction::BatchSize>();
        batchSize->index = 0;
        properties.push_back(batchSize);
        auto boundSize   = std::make_shared<Contraction::BoundSize>();
        boundSize->index = 0;
        properties.push_back(boundSize);
    }

    std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

    using Entry
        = Matching::MatchingTableEntry<Key, std::shared_ptr<SolutionLibrary<ContractionProblem>>>;

    std::vector<Entry> table;

    {
        Entry map0{{4, 4, 1, 4}, LibraryMFMA, 2.0};
        table.push_back(map0);
        Entry map1{{4, 4, 1, 4}, LibraryVALU, 1.0};
        table.push_back(map1);
        Entry map2{{8, 8, 1, 8}, LibraryVALU, 2.0};
        table.push_back(map2);
        Entry map3{{8, 8, 1, 8}, LibraryMFMA, 1.0};
        table.push_back(map3);
    }

    matchingTable->table = table;

    ProblemMatchingLibrary<ContractionProblem> lib;

    lib.table = matchingTable;

    auto theSolution0 = lib.findBestSolution(Problem_Size3, gpu);
    EXPECT_EQ(theSolution0, SolutionMFMA);
    auto theSolution1 = lib.findBestSolution(Problem_Size5, gpu);
    EXPECT_EQ(theSolution1, SolutionMFMA);

    auto theSolution2 = lib.findBestSolution(Problem_Size7, gpu);
    EXPECT_EQ(theSolution2, SolutionVALU);
    auto theSolution3 = lib.findBestSolution(Problem_Size9, gpu);
    EXPECT_EQ(theSolution3, SolutionVALU);
}
