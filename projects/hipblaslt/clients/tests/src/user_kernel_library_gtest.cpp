// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Each test uses its own UserKernelLibrary, not the process-wide instance.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/UserKernelIndex.hpp>
#include <Tensile/UserKernelLibrary.hpp>

using namespace TensileLite;

namespace
{
    using UserLibrary = UserKernelLibrary<ContractionProblemGemm, ContractionSolution>;
    using Master      = MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>;
    using SolutionPtr = std::shared_ptr<ContractionSolution>;

    constexpr size_t kM = 128;
    constexpr size_t kN = 256;
    constexpr size_t kK = 64;

    AMDGPU makeGpu()
    {
        return AMDGPU(AMDGPU::Processor::gfx950, 256, "gfx950");
    }

    ContractionProblemGemm
        makeProblem(size_t m = kM, size_t n = kN, size_t k = kK, bool transA = false)
    {
        const size_t lda = transA ? k : m;
        return ContractionProblemGemm::GEMM(transA, false, m, n, k, lda, k, m, 1.0, false, 1);
    }

    // Matches the problem type of `problem`; every predicate is True.
    SolutionPtr makeSolution(ContractionProblemGemm const& problem)
    {
        auto  solution             = std::make_shared<ContractionSolution>();
        auto& pt                   = solution->problemType;
        pt.transA                  = problem.transA();
        pt.transB                  = problem.transB();
        pt.aType                   = problem.a().dataType();
        pt.bType                   = problem.b().dataType();
        pt.cType                   = problem.c().dataType();
        pt.dType                   = problem.d().dataType();
        pt.computeType             = problem.computeType();
        pt.highPrecisionAccumulate = problem.highPrecisionAccumulate();
        return solution;
    }

    void rejectEveryProblem(ContractionSolution& solution)
    {
        solution.problemPredicate = std::make_shared<Predicates::False<ContractionProblemGemm>>();
    }

    /// Stands in for the shipped library under a master and records what it
    /// was asked for.
    struct StubLibrary : public SolutionLibrary<ContractionProblemGemm, ContractionSolution>
    {
        std::vector<SolutionPtr> shipped;
        mutable int              lastTopRequest = -1;

        std::shared_ptr<ContractionSolution> getSolutionByIndex(ContractionProblemGemm const&,
                                                                Hardware const&,
                                                                const int) const override
        {
            return nullptr;
        }

        std::shared_ptr<ContractionSolution> findBestSolution(ContractionProblemGemm const&,
                                                              Hardware const&,
                                                              double* fitness) const override
        {
            if(fitness)
                *fitness = 5.0;
            return shipped.empty() ? nullptr : shipped.front();
        }

        SolutionSet<ContractionSolution> findAllSolutions(ContractionProblemGemm const&,
                                                          Hardware const&,
                                                          SolutionLibrarySearchType) const override
        {
            return SolutionSet<ContractionSolution>(shipped.begin(), shipped.end());
        }

        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType) const override
        {
            return {};
        }

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                             Hardware const&,
                                                             int numSolutions) const override
        {
            lastTopRequest = numSolutions;
            const size_t n = std::min(shipped.size(), static_cast<size_t>(numSolutions));
            return SolutionVector<ContractionSolution>(shipped.begin(), shipped.begin() + n);
        }

        std::string type() const override
        {
            return "Stub";
        }
        std::string description() const override
        {
            return "Stub";
        }
    };

    /// A master over four shipped solutions, with its own user library.
    struct MasterFixture
    {
        std::shared_ptr<StubLibrary> stub = std::make_shared<StubLibrary>();
        std::shared_ptr<UserLibrary> user = std::make_shared<UserLibrary>();
        Master                       master;

        explicit MasterFixture(ContractionProblemGemm const& problem)
        {
            for(int i = 0; i < 4; i++)
            {
                auto solution   = makeSolution(problem);
                solution->index = i;
                stub->shipped.push_back(solution);
            }
            master.library     = stub;
            master.userLibrary = user;
        }

        SolutionPtr registerAndMap(ContractionProblemGemm const& problem)
        {
            auto solution = makeSolution(problem);
            int  index    = user->addSolutions({solution}).at(0);
            EXPECT_TRUE(user->setExactMatch(userExactShapeFor(problem), index));
            return solution;
        }
    };
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

TEST(UserKernelLibrary, smoke_GlobalInstanceIsShared)
{
    auto a = globalUserKernelLibrary<ContractionProblemGemm, ContractionSolution>();
    auto b = globalUserKernelLibrary<ContractionProblemGemm, ContractionSolution>();
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a, b);
    EXPECT_EQ(Master().userLibrary, a) << "a master should consult the process-wide library";
}

// Registration assigns user-range indices and writes them back to the solution.
TEST(UserKernelLibrary, smoke_AddSolutionsAssignsUserIndices)
{
    auto        problem = makeProblem();
    UserLibrary library;

    auto a        = makeSolution(problem);
    auto b        = makeSolution(problem);
    a->index      = 7;
    b->index      = 8;
    auto assigned = library.addSolutions({a, b});

    ASSERT_EQ(assigned.size(), 2u);
    EXPECT_EQ(assigned[0], UserKernelIndexBase);
    EXPECT_EQ(assigned[1], UserKernelIndexBase + 1);
    EXPECT_EQ(a->index, assigned[0]);
    EXPECT_EQ(b->index, assigned[1]);

    EXPECT_EQ(library.lookupIndex(assigned[0]), a);
    EXPECT_EQ(library.lookupIndex(assigned[1]), b);
    EXPECT_EQ(library.lookupIndex(7), nullptr) << "a shipped index is not a user kernel";
    EXPECT_EQ(library.registeredCount(), 2u);
    EXPECT_EQ(library.selectableCount(), 0u);
}

// A registered kernel is enumerable, but not selectable until it is mapped.
TEST(UserKernelLibrary, smoke_RegistrationDoesNotChangeSelection)
{
    auto        problem = makeProblem();
    auto        gpu     = makeGpu();
    UserLibrary library;

    auto solution = makeSolution(problem);
    library.addSolutions({solution});

    EXPECT_EQ(library.findExact(problem, gpu), nullptr);
    EXPECT_EQ(library.findAll(problem, gpu).count(solution), 1u);
}

TEST(UserKernelLibrary, smoke_SetExactMatchSelectsOnlyThatShape)
{
    auto        problem = makeProblem();
    auto        gpu     = makeGpu();
    UserLibrary library;

    auto solution = makeSolution(problem);
    int  index    = library.addSolutions({solution}).at(0);

    EXPECT_FALSE(library.setExactMatch(userExactShapeFor(problem), index + 1))
        << "an unregistered index should be refused";
    ASSERT_TRUE(library.setExactMatch(userExactShapeFor(problem), index));
    EXPECT_EQ(library.selectableCount(), 1u);

    EXPECT_EQ(library.findExact(problem, gpu), solution);
    EXPECT_EQ(solution->tag, ContractionSolution::MatchingTag::Equal);

    EXPECT_EQ(library.findExact(makeProblem(kM, kN, kK * 2), gpu), nullptr)
        << "a different K should not match";
}

// A kernel is mapped under its own problem type, so it must not serve a
// problem of another type with the same shape.
TEST(UserKernelLibrary, smoke_ExactMatchRejectsOtherProblemType)
{
    auto        problem = makeProblem();
    auto        gpu     = makeGpu();
    UserLibrary library;

    auto solution                = makeSolution(problem);
    solution->problemType.transA = !problem.transA();
    int index                    = library.addSolutions({solution}).at(0);
    ASSERT_TRUE(library.setExactMatch(userExactShapeFor(problem), index));

    EXPECT_EQ(library.findExact(problem, gpu), nullptr);
    EXPECT_TRUE(library.findAll(problem, gpu).empty());
}

TEST(UserKernelLibrary, smoke_KeyIncludesTransposesAndTypes)
{
    auto problem = makeProblem();
    auto key     = userExactKeyFor(problem);
    EXPECT_EQ(key, userExactKeyFor(userExactShapeFor(problem), makeSolution(problem)->problemType))
        << "a problem is looked up under the key its kernel is mapped under";

    auto other   = key;
    other.transB = !other.transB;
    EXPECT_NE(other, key);
    other       = key;
    other.dType = rocisa::DataType::BFloat16;
    EXPECT_NE(other, key);
    other             = key;
    other.computeType = rocisa::DataType::Double;
    EXPECT_NE(other, key);
}

// Kernels for two transposes mapped to one shape are both selectable.
TEST(UserKernelLibrary, smoke_TwoTransposesAtOneShape)
{
    auto        gpu = makeGpu();
    auto        nn  = makeProblem(kM, kN, kK, false);
    auto        tn  = makeProblem(kM, kN, kK, true);
    UserLibrary library;
    ASSERT_EQ(userExactShapeFor(nn), userExactShapeFor(tn));

    auto forNN = makeSolution(nn);
    auto forTN = makeSolution(tn);
    auto added = library.addSolutions({forNN, forTN});
    ASSERT_TRUE(library.setExactMatch(userExactShapeFor(nn), added[0]));
    ASSERT_TRUE(library.setExactMatch(userExactShapeFor(tn), added[1]));

    EXPECT_EQ(library.findExact(nn, gpu), forNN);
    EXPECT_EQ(library.findExact(tn, gpu), forTN);
    EXPECT_EQ(library.selectableCount(), 2u);
}

// Kernels for two data types mapped to one shape are both selectable.
TEST(UserKernelLibrary, smoke_TwoDataTypesAtOneShape)
{
    auto        gpu      = makeGpu();
    auto        f32      = makeProblem();
    const auto  bf16Type = rocisa::DataType::BFloat16;
    auto        bf16     = ContractionProblemGemm::GEMM_Strides(false,
                                                     false,
                                                     bf16Type,
                                                     bf16Type,
                                                     bf16Type,
                                                     bf16Type,
                                                     kM,
                                                     kN,
                                                     kK,
                                                     1,
                                                     kM,
                                                     kM * kK,
                                                     kK,
                                                     kK * kN,
                                                     kM,
                                                     kM * kN,
                                                     kM,
                                                     kM * kN,
                                                     1.0);
    UserLibrary library;
    ASSERT_EQ(userExactShapeFor(f32), userExactShapeFor(bf16));

    auto forF32  = makeSolution(f32);
    auto forBF16 = makeSolution(bf16);
    auto added   = library.addSolutions({forF32, forBF16});
    ASSERT_TRUE(library.setExactMatch(userExactShapeFor(f32), added[0]));
    ASSERT_TRUE(library.setExactMatch(userExactShapeFor(bf16), added[1]));

    EXPECT_EQ(library.findExact(f32, gpu), forF32);
    EXPECT_EQ(library.findExact(bf16, gpu), forBF16);
    EXPECT_EQ(library.selectableCount(), 2u);
}

// The newest mapping that can serve the problem wins; mapping again restores
// an earlier one.
TEST(UserKernelLibrary, smoke_NewestMappingWins)
{
    auto        problem = makeProblem();
    auto        gpu     = makeGpu();
    auto        shape   = userExactShapeFor(problem);
    UserLibrary library;

    auto first  = makeSolution(problem);
    auto second = makeSolution(problem);
    auto added  = library.addSolutions({first, second});

    ASSERT_TRUE(library.setExactMatch(shape, added[0]));
    EXPECT_EQ(library.findExact(problem, gpu), first);

    ASSERT_TRUE(library.setExactMatch(shape, added[1]));
    EXPECT_EQ(library.findExact(problem, gpu), second);

    ASSERT_TRUE(library.setExactMatch(shape, added[0]));
    EXPECT_EQ(library.findExact(problem, gpu), first);
}

// A newer kernel that cannot serve the problem does not hide an older one that
// can.
TEST(UserKernelLibrary, smoke_FallsBackToOlderMapping)
{
    auto        problem = makeProblem();
    auto        gpu     = makeGpu();
    auto        shape   = userExactShapeFor(problem);
    UserLibrary library;

    auto older = makeSolution(problem);
    auto newer = makeSolution(problem);
    rejectEveryProblem(*newer);
    auto added = library.addSolutions({older, newer});
    ASSERT_TRUE(library.setExactMatch(shape, added[0]));
    ASSERT_TRUE(library.setExactMatch(shape, added[1]));

    EXPECT_EQ(library.findExact(problem, gpu), older);
}

TEST(UserKernelLibrary, smoke_FindAllHonoursSearchType)
{
    auto        problem = makeProblem();
    auto        gpu     = makeGpu();
    UserLibrary library;

    auto serves  = makeSolution(problem);
    auto rejects = makeSolution(problem);
    rejectEveryProblem(*rejects);
    auto otherType                = makeSolution(problem);
    otherType->problemType.transA = !problem.transA();
    library.addSolutions({serves, rejects, otherType});

    EXPECT_EQ(library.findAll(problem, gpu, SolutionLibrarySearchType::DEFAULT),
              SolutionSet<ContractionSolution>({serves}));
    EXPECT_EQ(library.findAll(problem, gpu, SolutionLibrarySearchType::GEMM_TYPE_ONLY),
              SolutionSet<ContractionSolution>({serves, rejects}));
    EXPECT_EQ(library.findAll(problem, gpu, SolutionLibrarySearchType::HARDWARE_ONLY).size(), 3u);
}

// Readers run while kernels are registered and mapped.
TEST(UserKernelLibrary, smoke_ConcurrentReadersAndWriter)
{
    auto        problem = makeProblem();
    auto        gpu     = makeGpu();
    UserLibrary library;

    std::atomic<bool>        done{false};
    std::vector<std::thread> readers;
    for(int t = 0; t < 4; t++)
    {
        readers.emplace_back([&]() {
            while(!done.load())
            {
                auto found = library.findExact(problem, gpu);
                if(found)
                    EXPECT_TRUE(isUserKernelIndex(found->index));
                library.findAll(problem, gpu);
            }
        });
    }

    for(int i = 0; i < 200; i++)
    {
        int index = library.addSolutions({makeSolution(problem)}).at(0);
        ASSERT_TRUE(library.setExactMatch(userExactShapeFor(problem), index));
    }
    done.store(true);
    for(auto& reader : readers)
        reader.join();

    EXPECT_EQ(library.registeredCount(), 200u);
    EXPECT_EQ(library.findExact(problem, gpu)->index, UserKernelIndexBase + 199);
}

// The master puts the mapped kernel first and asks the shipped library for one
// fewer, so the kernel appears once however many shipped rows match.
TEST(UserKernelMaster, smoke_TopSolutionsPrependMappedKernelOnce)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          user = f.registerAndMap(problem);

    auto top = f.master.findTopSolutions(problem, gpu, 4);
    ASSERT_EQ(top.size(), 4u);
    EXPECT_EQ(top[0], user);
    EXPECT_EQ(f.stub->lastTopRequest, 3);
    for(int i = 1; i < 4; i++)
        EXPECT_EQ(top[i], f.stub->shipped[i - 1]) << "shipped order should be kept";
}

TEST(UserKernelMaster, smoke_TopOneSkipsShippedLibrary)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          user = f.registerAndMap(problem);

    auto top = f.master.findTopSolutions(problem, gpu, 1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0], user);
    EXPECT_EQ(f.stub->lastTopRequest, -1);
    EXPECT_FALSE(f.master.lastFindTopAlreadyRetAll());
}

TEST(UserKernelMaster, smoke_UnmappedShapeUsesShippedLibrary)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    f.registerAndMap(makeProblem(kM, kN, kK * 2));

    auto top = f.master.findTopSolutions(problem, gpu, 4);
    EXPECT_EQ(top, f.stub->shipped);
    EXPECT_EQ(f.stub->lastTopRequest, 4);

    double fitness = 0.0;
    EXPECT_EQ(f.master.findBestSolution(problem, gpu, &fitness), f.stub->shipped[0]);
    EXPECT_EQ(fitness, 5.0);
}

TEST(UserKernelMaster, smoke_BestSolutionIsMappedKernel)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          user = f.registerAndMap(problem);

    double fitness = 1.0;
    EXPECT_EQ(f.master.findBestSolution(problem, gpu, &fitness), user);
    EXPECT_EQ(fitness, 0.0);
    EXPECT_EQ(user->tag, ContractionSolution::MatchingTag::Equal);
}

TEST(UserKernelMaster, smoke_FindAllIsUnion)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          mapped   = f.registerAndMap(problem);
    auto          unmapped = makeSolution(problem);
    f.user->addSolutions({unmapped});

    auto all = f.master.findAllSolutions(problem, gpu);
    EXPECT_EQ(all.size(), 6u);
    EXPECT_EQ(all.count(mapped), 1u);
    EXPECT_EQ(all.count(unmapped), 1u);
}

TEST(UserKernelMaster, smoke_ResolveUserIndex)
{
    auto          problem = makeProblem();
    MasterFixture f(problem);
    auto          user = f.registerAndMap(problem);

    EXPECT_EQ(f.master.resolveSolutionByIndex(user->index), user);
    EXPECT_EQ(f.master.resolveSolutionByIndex(user->index + 1), nullptr);
}
