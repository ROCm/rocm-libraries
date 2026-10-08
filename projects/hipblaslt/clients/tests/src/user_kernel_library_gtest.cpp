// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Each test uses its own UserKernelLibrary, not the process-wide instance.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/LibraryGeneration.hpp>
#include <Tensile/UserKernelIndex.hpp>
#include <Tensile/UserKernelLibrary.hpp>

using namespace TensileLite;

namespace
{
    using Tier = UserKernelLibrary<ContractionProblemGemm, ContractionSolution>;

    constexpr size_t kM = 128;
    constexpr size_t kN = 256;
    constexpr size_t kK = 64;

    AMDGPU makeGpu()
    {
        return AMDGPU(AMDGPU::Processor::gfx950, 256, "gfx950");
    }

    ContractionProblemGemm makeProblem(size_t m = kM, size_t n = kN, size_t k = kK)
    {
        return ContractionProblemGemm::GEMM(false, false, m, n, k, m, k, m, 1.0, false, 1);
    }

    // Matches makeProblem()'s problem type and has no predicates.
    std::shared_ptr<ContractionSolution> makeSolution(ContractionProblemGemm const& problem)
    {
        auto  solution              = std::make_shared<ContractionSolution>();
        auto& pt                    = solution->problemType;
        pt.transA                   = problem.transA();
        pt.transB                   = problem.transB();
        pt.aType                    = problem.a().dataType();
        pt.bType                    = problem.b().dataType();
        pt.cType                    = problem.c().dataType();
        pt.dType                    = problem.d().dataType();
        pt.computeType              = problem.computeType();
        pt.highPrecisionAccumulate  = problem.highPrecisionAccumulate();
        solution->problemPredicate  = nullptr;
        solution->taskPredicate     = nullptr;
        solution->hardwarePredicate = nullptr;
        return solution;
    }

    UserExactKey keyFor(ContractionProblemGemm const& problem)
    {
        return userExactKeyFor(problem);
    }
}

TEST(UserKernelIndexSpace, smoke_PartitionBoundary)
{
    EXPECT_EQ(UserKernelIndexBase, 3 << 29);

    EXPECT_FALSE(isUserKernelIndex(0));
    EXPECT_FALSE(isUserKernelIndex((1 << 30) - 1)) << "the last shipped index";
    EXPECT_FALSE(isUserKernelIndex(1 << 30)) << "the first JIT index";
    EXPECT_FALSE(isUserKernelIndex(UserKernelIndexBase - 1)) << "the last JIT index";

    EXPECT_TRUE(isUserKernelIndex(UserKernelIndexBase));
    EXPECT_TRUE(isUserKernelIndex(INT32_MAX));
}

// Registration assigns user-range indices and writes them back to the solution.
TEST(UserKernelLibrary, smoke_AddSolutionsAssignsUserIndices)
{
    auto problem = makeProblem();
    Tier tier(std::make_shared<LibraryGeneration>());

    auto a        = makeSolution(problem);
    auto b        = makeSolution(problem);
    a->index      = 7;
    b->index      = 8;
    auto assigned = tier.addSolutions({a, b});

    ASSERT_EQ(assigned.size(), 2u);
    EXPECT_EQ(assigned[0], UserKernelIndexBase);
    EXPECT_EQ(assigned[1], UserKernelIndexBase + 1);
    EXPECT_EQ(a->index, assigned[0]);
    EXPECT_EQ(b->index, assigned[1]);

    auto gpu = makeGpu();
    EXPECT_EQ(tier.getSolutionByIndex(assigned[0]), a);
    EXPECT_EQ(tier.getSolutionByIndex(gpu, assigned[1]), b);
    EXPECT_EQ(tier.getSolutionByIndex(problem, gpu, assigned[0]), a);
    EXPECT_EQ(tier.getSolutionByIndex(7), nullptr) << "a shipped index is not a user kernel";
    EXPECT_EQ(tier.registeredCount(), 2u);
    EXPECT_EQ(tier.selectableCount(), 0u);
}

// A registered kernel is enumerable and dispatchable, but not selectable until
// it is mapped to a shape.
TEST(UserKernelLibrary, smoke_RegistrationDoesNotChangeSelection)
{
    auto problem = makeProblem();
    auto gpu     = makeGpu();
    Tier tier(std::make_shared<LibraryGeneration>());

    auto solution = makeSolution(problem);
    tier.addSolutions({solution});

    EXPECT_EQ(tier.findBestSolution(problem, gpu), nullptr);
    EXPECT_TRUE(tier.findTopSolutions(problem, gpu, 4).empty());
    EXPECT_EQ(tier.findAllSolutions(problem, gpu).count(solution), 1u)
        << "enumeration reports registered kernels whether or not they are mapped";
}

TEST(UserKernelLibrary, smoke_SetExactMatchSelectsOnlyThatShape)
{
    auto problem = makeProblem();
    auto gpu     = makeGpu();
    Tier tier(std::make_shared<LibraryGeneration>());

    auto solution = makeSolution(problem);
    int  index    = tier.addSolutions({solution}).at(0);

    EXPECT_FALSE(tier.setExactMatch(keyFor(problem), index + 1))
        << "an unregistered index should be refused";
    ASSERT_TRUE(tier.setExactMatch(keyFor(problem), index));
    EXPECT_EQ(tier.selectableCount(), 1u);

    double fitness = 1.0;
    EXPECT_EQ(tier.findBestSolution(problem, gpu, &fitness), solution);
    EXPECT_EQ(fitness, 0.0);

    auto top = tier.findTopSolutions(problem, gpu, 4);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0], solution);
    EXPECT_TRUE(tier.lastFindTopAlreadyRetAll()) << "at most one kernel is mapped per shape";

    auto other = makeProblem(kM, kN, kK * 2);
    EXPECT_EQ(tier.findBestSolution(other, gpu), nullptr) << "a different K should not match";
}

// The key omits the problem type, so a mapped kernel must not serve a problem
// of another type with the same shape.
TEST(UserKernelLibrary, smoke_ExactMatchRejectsOtherProblemType)
{
    auto problem = makeProblem();
    auto gpu     = makeGpu();
    Tier tier(std::make_shared<LibraryGeneration>());

    auto solution                = makeSolution(problem);
    solution->problemType.transA = !problem.transA();
    int index                    = tier.addSolutions({solution}).at(0);
    ASSERT_TRUE(tier.setExactMatch(keyFor(problem), index));

    EXPECT_EQ(tier.findBestSolution(problem, gpu), nullptr);
    EXPECT_TRUE(tier.findAllSolutions(problem, gpu).empty());
}

// Every change advances the generation.
TEST(UserKernelLibrary, smoke_PublicationAdvancesGeneration)
{
    auto problem    = makeProblem();
    auto generation = std::make_shared<LibraryGeneration>();
    Tier tier(generation);

    const uint64_t start = generation->value.load();
    int            index = tier.addSolutions({makeSolution(problem)}).at(0);
    const uint64_t added = generation->value.load();
    EXPECT_GT(added, start);

    ASSERT_TRUE(tier.setExactMatch(keyFor(problem), index));
    EXPECT_GT(generation->value.load(), added);

    const uint64_t mapped = generation->value.load();
    EXPECT_FALSE(tier.setExactMatch(keyFor(problem), index + 1));
    EXPECT_EQ(generation->value.load(), mapped)
        << "a refused mapping should not advance the generation";
}
