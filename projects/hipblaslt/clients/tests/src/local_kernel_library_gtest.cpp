// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Each test uses its own LocalKernelLibrary, not the process-wide instance.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/LocalKernelIndex.hpp>
#include <Tensile/LocalKernelLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

using namespace TensileLite;

namespace
{
    using LocalLibrary = LocalKernelLibrary<ContractionProblemGemm, ContractionSolution>;
    using Master       = MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>;
    using SolutionPtr  = std::shared_ptr<ContractionSolution>;

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

    // Matches the problem type of `problem`; every predicate is True. It has no
    // index yet, as addSolutions requires.
    SolutionPtr makeSolution(ContractionProblemGemm const& problem)
    {
        auto solution              = std::make_shared<ContractionSolution>();
        solution->index            = -1;
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

    // The part of the problem predicate TensileLite generates from these
    // problem type fields.
    void predicateOnGroupedAndMXBlock(ContractionSolution& solution)
    {
        auto const& pt            = solution.problemType;
        solution.problemPredicate = std::make_shared<Predicates::And<ContractionProblemGemm>>(
            std::initializer_list<std::shared_ptr<Predicates::Predicate<ContractionProblemGemm>>>{
                std::make_shared<Predicates::Contraction::GroupedGemmEqual>(pt.groupedGemm),
                std::make_shared<Predicates::Contraction::MXBlockA>(pt.mxBlockA)});
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

    /// A master over four shipped solutions, with its own local kernel library.
    struct MasterFixture
    {
        std::shared_ptr<StubLibrary>  stub  = std::make_shared<StubLibrary>();
        std::shared_ptr<LocalLibrary> local = std::make_shared<LocalLibrary>();
        Master                        master;

        explicit MasterFixture(ContractionProblemGemm const& problem)
        {
            for(int i = 0; i < 4; i++)
            {
                auto solution   = makeSolution(problem);
                solution->index = i;
                stub->shipped.push_back(solution);
            }
            master.library      = stub;
            master.localLibrary = local;
        }

        SolutionPtr registerAndMap(ContractionProblemGemm const& problem)
        {
            auto solution = makeSolution(problem);
            int  index    = local->addSolutions({solution}).at(0);
            EXPECT_TRUE(local->setExactMatch(localExactShapeFor(problem), index));
            return solution;
        }
    };
}

TEST(LocalKernelIndexSpace, smoke_PartitionBoundary)
{
    EXPECT_EQ(JitKernelIndexBase, 1 << 30);
    EXPECT_EQ(LocalKernelIndexBase, 3 << 29);

    EXPECT_FALSE(isJitKernelIndex(0));
    EXPECT_FALSE(isJitKernelIndex(JitKernelIndexBase - 1)) << "the last shipped index";
    EXPECT_TRUE(isJitKernelIndex(JitKernelIndexBase));
    EXPECT_TRUE(isJitKernelIndex(LocalKernelIndexBase - 1));
    EXPECT_FALSE(isJitKernelIndex(LocalKernelIndexBase)) << "the first local index";
    EXPECT_FALSE(isJitKernelIndex(INT32_MAX));

    EXPECT_FALSE(isLocalKernelIndex(0));
    EXPECT_FALSE(isLocalKernelIndex(JitKernelIndexBase - 1)) << "the last shipped index";
    EXPECT_FALSE(isLocalKernelIndex(JitKernelIndexBase)) << "the first JIT index";
    EXPECT_FALSE(isLocalKernelIndex(LocalKernelIndexBase - 1)) << "the last JIT index";

    EXPECT_TRUE(isLocalKernelIndex(LocalKernelIndexBase));
    EXPECT_TRUE(isLocalKernelIndex(INT32_MAX));
}

TEST(LocalKernelLibrary, smoke_GlobalInstanceIsShared)
{
    auto a = globalLocalKernelLibrary<ContractionProblemGemm, ContractionSolution>();
    auto b = globalLocalKernelLibrary<ContractionProblemGemm, ContractionSolution>();
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a, b);
    EXPECT_EQ(Master().localLibrary, a) << "a master should consult the process-wide library";
}

// Registration assigns local range indices and writes them back to the solution.
TEST(LocalKernelLibrary, smoke_AddSolutionsAssignsLocalIndices)
{
    auto         problem = makeProblem();
    LocalLibrary library;

    auto a        = makeSolution(problem);
    auto b        = makeSolution(problem);
    auto assigned = library.addSolutions({a, b});

    ASSERT_EQ(assigned.size(), 2u);
    EXPECT_EQ(assigned[0], LocalKernelIndexBase);
    EXPECT_EQ(assigned[1], LocalKernelIndexBase + 1);
    EXPECT_EQ(a->index, assigned[0]);
    EXPECT_EQ(b->index, assigned[1]);
    EXPECT_EQ(a->tag, ContractionSolution::MatchingTag::Equal);
    EXPECT_EQ(b->tag, ContractionSolution::MatchingTag::Equal);

    EXPECT_EQ(library.lookupIndex(assigned[0]), a);
    EXPECT_EQ(library.lookupIndex(assigned[1]), b);
    EXPECT_EQ(library.lookupIndex(7), nullptr) << "a shipped index is not a local kernel";
    EXPECT_EQ(library.registeredCount(), 2u);
    EXPECT_EQ(library.selectableCount(), 0u);
}

// A call with a null, repeated, or already registered solution registers
// nothing and leaves the solutions untouched.
TEST(LocalKernelLibrary, smoke_AddSolutionsRejectsRepeats)
{
    auto         problem = makeProblem();
    LocalLibrary library;

    auto a = makeSolution(problem);
    auto b = makeSolution(problem);
    EXPECT_TRUE(library.addSolutions({a, b, a}).empty());
    EXPECT_TRUE(library.addSolutions({a, nullptr}).empty());
    EXPECT_EQ(library.registeredCount(), 0u);
    EXPECT_EQ(a->index, -1);
    EXPECT_EQ(a->tag, ContractionSolution::MatchingTag::Others);

    int index = library.addSolutions({a}).at(0);
    EXPECT_TRUE(library.addSolutions({b, a}).empty());
    EXPECT_EQ(library.registeredCount(), 1u);
    EXPECT_EQ(a->index, index);
    EXPECT_EQ(b->index, -1);

    LocalLibrary other;
    EXPECT_TRUE(other.addSolutions({a}).empty()) << "registered with another library";
    EXPECT_EQ(a->index, index);
    EXPECT_EQ(library.addSolutions({b}).at(0), index + 1)
        << "a rejected call should not consume indices";
}

// A registered kernel is enumerable, but not selectable until it is mapped.
TEST(LocalKernelLibrary, smoke_RegistrationDoesNotChangeSelection)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    LocalLibrary library;

    auto solution = makeSolution(problem);
    library.addSolutions({solution});

    EXPECT_EQ(library.findExact(problem, gpu), nullptr);
    EXPECT_EQ(library.findAll(problem, gpu).count(solution), 1u);
}

TEST(LocalKernelLibrary, smoke_SetExactMatchSelectsOnlyThatShape)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    LocalLibrary library;

    auto solution = makeSolution(problem);
    int  index    = library.addSolutions({solution}).at(0);

    EXPECT_FALSE(library.setExactMatch(localExactShapeFor(problem), index + 1))
        << "an unregistered index should be refused";
    ASSERT_TRUE(library.setExactMatch(localExactShapeFor(problem), index));
    EXPECT_EQ(library.selectableCount(), 1u);

    EXPECT_EQ(library.findExact(problem, gpu), solution);
    EXPECT_EQ(solution->tag, ContractionSolution::MatchingTag::Equal);

    EXPECT_EQ(library.findExact(makeProblem(kM, kN, kK * 2), gpu), nullptr)
        << "a different K should not match";
}

// Concurrent lookups share the selected solution, so selection must not write
// to it.
TEST(LocalKernelLibrary, smoke_SelectionDoesNotWriteToSolution)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    LocalLibrary library;

    auto solution = makeSolution(problem);
    int  index    = library.addSolutions({solution}).at(0);
    ASSERT_TRUE(library.setExactMatch(localExactShapeFor(problem), index));

    solution->tag = ContractionSolution::MatchingTag::Others;
    ASSERT_EQ(library.findExact(problem, gpu), solution);
    EXPECT_EQ(solution->tag, ContractionSolution::MatchingTag::Others);
}

// A kernel is mapped under its own problem type, so it must not serve a
// problem of another type with the same shape.
TEST(LocalKernelLibrary, smoke_ExactMatchRejectsOtherProblemType)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    LocalLibrary library;

    auto solution                = makeSolution(problem);
    solution->problemType.transA = !problem.transA();
    int index                    = library.addSolutions({solution}).at(0);
    ASSERT_TRUE(library.setExactMatch(localExactShapeFor(problem), index));

    EXPECT_EQ(library.findExact(problem, gpu), nullptr);
    EXPECT_TRUE(library.findAll(problem, gpu).empty());
}

TEST(LocalKernelLibrary, smoke_KeyIncludesTransposesAndTypes)
{
    auto problem = makeProblem();
    auto key     = localExactKeyFor(problem);
    EXPECT_EQ(key,
              localExactKeyFor(localExactShapeFor(problem), makeSolution(problem)->problemType))
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
TEST(LocalKernelLibrary, smoke_TwoTransposesAtOneShape)
{
    auto         gpu = makeGpu();
    auto         nn  = makeProblem(kM, kN, kK, false);
    auto         tn  = makeProblem(kM, kN, kK, true);
    LocalLibrary library;
    ASSERT_EQ(localExactShapeFor(nn), localExactShapeFor(tn));

    auto forNN = makeSolution(nn);
    auto forTN = makeSolution(tn);
    auto added = library.addSolutions({forNN, forTN});
    ASSERT_TRUE(library.setExactMatch(localExactShapeFor(nn), added[0]));
    ASSERT_TRUE(library.setExactMatch(localExactShapeFor(tn), added[1]));

    EXPECT_EQ(library.findExact(nn, gpu), forNN);
    EXPECT_EQ(library.findExact(tn, gpu), forTN);
    EXPECT_EQ(library.selectableCount(), 2u);
}

// Kernels for two data types mapped to one shape are both selectable.
TEST(LocalKernelLibrary, smoke_TwoDataTypesAtOneShape)
{
    auto         gpu      = makeGpu();
    auto         f32      = makeProblem();
    const auto   bf16Type = rocisa::DataType::BFloat16;
    auto         bf16     = ContractionProblemGemm::GEMM_Strides(false,
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
    LocalLibrary library;
    ASSERT_EQ(localExactShapeFor(f32), localExactShapeFor(bf16));

    auto forF32  = makeSolution(f32);
    auto forBF16 = makeSolution(bf16);
    auto added   = library.addSolutions({forF32, forBF16});
    ASSERT_TRUE(library.setExactMatch(localExactShapeFor(f32), added[0]));
    ASSERT_TRUE(library.setExactMatch(localExactShapeFor(bf16), added[1]));

    EXPECT_EQ(library.findExact(f32, gpu), forF32);
    EXPECT_EQ(library.findExact(bf16, gpu), forBF16);
    EXPECT_EQ(library.selectableCount(), 2u);
}

// Kernels that differ only outside the key, here in grouped GEMM and the MX
// block size, share a key; each kernel's own problem predicate picks the one
// that serves a problem, so all stay selectable.
TEST(LocalKernelLibrary, smoke_FieldsOutsideTheKeyAreChecked)
{
    auto gpu     = makeGpu();
    auto plain   = makeProblem();
    auto grouped = makeProblem();
    grouped.setGroupedGemm(true);
    auto mx = makeProblem();
    mx.setMXScaleA(rocisa::DataType::E8, 32);
    LocalLibrary library;
    ASSERT_EQ(localExactKeyFor(plain), localExactKeyFor(grouped));
    ASSERT_EQ(localExactKeyFor(plain), localExactKeyFor(mx));

    auto forPlain                       = makeSolution(plain);
    auto forGrouped                     = makeSolution(grouped);
    forGrouped->problemType.groupedGemm = true;
    auto forMX                          = makeSolution(mx);
    forMX->problemType.mxBlockA         = 32;
    for(auto const& solution : {forPlain, forGrouped, forMX})
        predicateOnGroupedAndMXBlock(*solution);

    auto added = library.addSolutions({forPlain, forGrouped, forMX});
    for(int index : added)
        ASSERT_TRUE(library.setExactMatch(localExactShapeFor(plain), index));
    EXPECT_EQ(library.selectableCount(), 1u);

    EXPECT_EQ(library.findExact(plain, gpu), forPlain);
    EXPECT_EQ(library.findExact(grouped, gpu), forGrouped);
    EXPECT_EQ(library.findExact(mx, gpu), forMX);
}

// The newest mapping that can serve the problem wins; mapping again restores
// an earlier one.
TEST(LocalKernelLibrary, smoke_NewestMappingWins)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    auto         shape   = localExactShapeFor(problem);
    LocalLibrary library;

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
TEST(LocalKernelLibrary, smoke_FallsBackToOlderMapping)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    auto         shape   = localExactShapeFor(problem);
    LocalLibrary library;

    auto older = makeSolution(problem);
    auto newer = makeSolution(problem);
    rejectEveryProblem(*newer);
    auto added = library.addSolutions({older, newer});
    ASSERT_TRUE(library.setExactMatch(shape, added[0]));
    ASSERT_TRUE(library.setExactMatch(shape, added[1]));

    EXPECT_EQ(library.findExact(problem, gpu), older);
}

TEST(LocalKernelLibrary, smoke_FindAllHonoursSearchType)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    LocalLibrary library;

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

// The DEFAULT search applies the software predicate the shipped libraries do,
// which drops StreamK dynamic-queue kernels where the XCD count is not known
// to be a power of two. makeGpu() carries no XCD count.
TEST(LocalKernelLibrary, smoke_FindAllAppliesSoftwarePredicate)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    LocalLibrary library;

    auto plain                                       = makeSolution(problem);
    auto dynamicQueue                                = makeSolution(problem);
    dynamicQueue->sizeMapping.tileProcessingStrategy = TileProcessingStrategy::StreamK;
    dynamicQueue->sizeMapping.workAssignment         = WorkAssignment::DynamicWorkQueue;
    library.addSolutions({plain, dynamicQueue});

    EXPECT_EQ(library.findAll(problem, gpu, SolutionLibrarySearchType::DEFAULT),
              SolutionSet<ContractionSolution>({plain}));
    EXPECT_EQ(library.findAll(problem, gpu, SolutionLibrarySearchType::GEMM_TYPE_ONLY),
              SolutionSet<ContractionSolution>({plain, dynamicQueue}));
}

// Readers run while kernels are registered and mapped.
TEST(LocalKernelLibrary, smoke_ConcurrentReadersAndWriter)
{
    auto         problem = makeProblem();
    auto         gpu     = makeGpu();
    LocalLibrary library;

    std::atomic<bool>        done{false};
    std::vector<std::thread> readers;
    struct JoinReaders
    {
        std::atomic<bool>&        done;
        std::vector<std::thread>& readers;

        ~JoinReaders()
        {
            done.store(true);
            for(auto& reader : readers)
                reader.join();
        }
    } joinReaders{done, readers};

    for(int t = 0; t < 4; t++)
    {
        readers.emplace_back([&]() {
            while(!done.load())
            {
                auto found = library.findExact(problem, gpu);
                if(found)
                    EXPECT_TRUE(isLocalKernelIndex(found->index));
                library.findAll(problem, gpu);
            }
        });
    }

    for(int i = 0; i < 200; i++)
    {
        int index = library.addSolutions({makeSolution(problem)}).at(0);
        ASSERT_TRUE(library.setExactMatch(localExactShapeFor(problem), index));
    }
    EXPECT_EQ(library.registeredCount(), 200u);
    auto found = library.findExact(problem, gpu);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->index, LocalKernelIndexBase + 199);
}

// The master puts the mapped kernel first and asks the shipped library for one
// fewer, so the kernel appears once however many shipped rows match.
TEST(LocalKernelMaster, smoke_TopSolutionsPrependMappedKernelOnce)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          local = f.registerAndMap(problem);

    auto top = f.master.findTopSolutions(problem, gpu, 4);
    ASSERT_EQ(top.size(), 4u);
    EXPECT_EQ(top[0], local);
    EXPECT_EQ(f.stub->lastTopRequest, 3);
    for(int i = 1; i < 4; i++)
        EXPECT_EQ(top[i], f.stub->shipped[i - 1]) << "shipped order should be kept";
}

// Registering a solution the shipped library holds is rejected, so it keeps
// its shipped index and tag.
TEST(LocalKernelMaster, smoke_ShippedSolutionsAreNotRegistered)
{
    auto          problem = makeProblem();
    MasterFixture f(problem);

    for(auto const& shipped : f.stub->shipped)
    {
        const int index = shipped->index;
        EXPECT_TRUE(f.local->addSolutions({makeSolution(problem), shipped}).empty());
        EXPECT_EQ(shipped->index, index);
        EXPECT_EQ(shipped->tag, ContractionSolution::MatchingTag::Others);
    }
    EXPECT_EQ(f.local->registeredCount(), 0u);
}

TEST(LocalKernelMaster, smoke_TopOneSkipsShippedLibrary)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          local = f.registerAndMap(problem);

    auto top = f.master.findTopSolutions(problem, gpu, 1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0], local);
    EXPECT_EQ(f.stub->lastTopRequest, -1);
    EXPECT_FALSE(f.master.lastFindTopAlreadyRetAll());
}

TEST(LocalKernelMaster, smoke_UnmappedShapeUsesShippedLibrary)
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

TEST(LocalKernelMaster, smoke_BestSolutionIsMappedKernel)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          local = f.registerAndMap(problem);

    double fitness = 1.0;
    EXPECT_EQ(f.master.findBestSolution(problem, gpu, &fitness), local);
    EXPECT_EQ(fitness, 0.0);
    EXPECT_EQ(local->tag, ContractionSolution::MatchingTag::Equal);
}

TEST(LocalKernelMaster, smoke_FindAllIsUnion)
{
    auto          problem = makeProblem();
    auto          gpu     = makeGpu();
    MasterFixture f(problem);
    auto          mapped   = f.registerAndMap(problem);
    auto          unmapped = makeSolution(problem);
    f.local->addSolutions({unmapped});

    auto all = f.master.findAllSolutions(problem, gpu);
    EXPECT_EQ(all.size(), 6u);
    EXPECT_EQ(all.count(mapped), 1u);
    EXPECT_EQ(all.count(unmapped), 1u);
}

TEST(LocalKernelMaster, smoke_ResolveLocalIndex)
{
    auto          problem = makeProblem();
    MasterFixture f(problem);
    auto          local = f.registerAndMap(problem);

    EXPECT_EQ(f.master.resolveSolutionByIndex(local->index), local);
    EXPECT_EQ(f.master.resolveSolutionByIndex(local->index + 1), nullptr);
}
