// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

namespace
{
    class StubTopLibrary : public ContractionLibrary
    {
    public:
        explicit StubTopLibrary(std::shared_ptr<ContractionSolution> solution)
            : m_solution(std::move(solution))
        {
        }

        std::shared_ptr<ContractionSolution> getSolutionByIndex(ContractionProblemGemm const&,
                                                                  Hardware const&,
                                                                  int) const override
        {
            return {};
        }

        std::shared_ptr<ContractionSolution> findBestSolution(ContractionProblemGemm const&,
                                                              Hardware const&,
                                                              double* = nullptr) const override
        {
            return m_solution;
        }

        SolutionSet<ContractionSolution> findAllSolutions(ContractionProblemGemm const&,
                                                          Hardware const&,
                                                          SolutionLibrarySearchType
                                                          = SolutionLibrarySearchType::DEFAULT) const override
        {
            return m_solution ? SolutionSet<ContractionSolution>({m_solution})
                              : SolutionSet<ContractionSolution>();
        }

        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return {};
        }

        std::string type() const override
        {
            return "StubTop";
        }

        std::string description() const override
        {
            return "StubTop";
        }

        TopSolutionsResult<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                                 Hardware const&,
                                                                 int) const override
        {
            return {m_solution ? SolutionVector<ContractionSolution>({m_solution})
                               : SolutionVector<ContractionSolution>(),
                    true};
        }

    private:
        std::shared_ptr<ContractionSolution> m_solution;
    };

    class ScopedStreamK5ForceMode
    {
    public:
        explicit ScopedStreamK5ForceMode(std::optional<std::string> value)
        {
            m_previous = std::getenv("TENSILE_STREAMK5_FORCE_MODE");
            if(m_previous)
                m_previousValue = m_previous;

            if(value)
                setenv("TENSILE_STREAMK5_FORCE_MODE", value->c_str(), 1);
            else
                unsetenv("TENSILE_STREAMK5_FORCE_MODE");

            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedStreamK5ForceMode()
        {
            if(m_previous)
                setenv("TENSILE_STREAMK5_FORCE_MODE", m_previousValue.c_str(), 1);
            else
                unsetenv("TENSILE_STREAMK5_FORCE_MODE");

            Debug::Instance().reloadDebugBitsForTest();
        }

    private:
        const char* m_previous = nullptr;
        std::string m_previousValue;
    };

    ContractionProblemPredicate makeRowPredicate(
        std::shared_ptr<Predicates::Predicate<ContractionProblemGemm>> predicate)
    {
        return ContractionProblemPredicate(std::move(predicate));
    }

    std::shared_ptr<ContractionProblemSelectionLibrary> buildMatchingRowsLibrary()
    {
        auto lib = std::make_shared<ContractionProblemSelectionLibrary>();

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::EqualityMatching>()),
            std::make_shared<StubTopLibrary>(makeSolution("equality", 1))));

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::RangeMatching>()),
            std::make_shared<StubTopLibrary>(makeSolution("range", 2))));

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::PredictionMatching>()),
            std::make_shared<StubTopLibrary>(makeSolution("prediction", 3))));

        return lib;
    }

    // Stub that records the SolutionLibrarySearchType it was handed, so a test can
    // check both the row set ExactLogicLibrary searched and what it forwarded down.
    class RecordingLibrary : public ContractionLibrary
    {
    public:
        explicit RecordingLibrary(std::shared_ptr<ContractionSolution> solution)
            : m_solution(std::move(solution))
        {
        }

        std::shared_ptr<ContractionSolution>
            getSolutionByIndex(ContractionProblemGemm const&, Hardware const&, int) const override
        {
            return {};
        }

        std::shared_ptr<ContractionSolution> findBestSolution(ContractionProblemGemm const&,
                                                              Hardware const&,
                                                              double* = nullptr) const override
        {
            return m_solution;
        }

        SolutionSet<ContractionSolution>
            findAllSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            m_lastSearchType = searchType;
            return m_solution ? SolutionSet<ContractionSolution>({m_solution})
                              : SolutionSet<ContractionSolution>();
        }

        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            m_lastSearchType = searchType;
            return {};
        }

        std::string type() const override
        {
            return "Recording";
        }

        std::string description() const override
        {
            return "Recording";
        }

        TopSolutionsResult<ContractionSolution>
            findTopSolutions(ContractionProblemGemm const&, Hardware const&, int) const override
        {
            return {m_solution ? SolutionVector<ContractionSolution>({m_solution})
                               : SolutionVector<ContractionSolution>(),
                    true};
        }

        SolutionLibrarySearchType lastSearchType() const
        {
            return m_lastSearchType;
        }

    private:
        std::shared_ptr<ContractionSolution>           m_solution;
        mutable SolutionLibrarySearchType m_lastSearchType = SolutionLibrarySearchType::COUNT;
    };

    // equality / gridbased / prediction rows, where the latter two are the
    // "ranking" matching libraries a no-ranking search must skip. Each call
    // builds its own solutions.
    std::shared_ptr<ContractionProblemSelectionLibrary> buildRankingRowsLibrary()
    {
        auto lib = std::make_shared<ContractionProblemSelectionLibrary>();

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::EqualityMatching>()),
            std::make_shared<RecordingLibrary>(makeSolution("equality", 1))));

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::GridBasedMatching>()),
            std::make_shared<RecordingLibrary>(makeSolution("gridbased", 2))));

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::PredictionMatching>()),
            std::make_shared<RecordingLibrary>(makeSolution("prediction", 3))));

        return lib;
    }

    std::set<std::string> solutionNameSet(SolutionSet<ContractionSolution> const& solutions)
    {
        std::set<std::string> names;
        for(auto const& solution : solutions)
            names.insert(solution->solutionName);
        return names;
    }

    // gfx942 is the only processor in AMDGPU's StandardCUMap, so it is the only
    // way to build a device for which isStandardCU() is false.
    AMDGPU makeNonStandardCuDevice()
    {
        return AMDGPU(AMDGPU::Processor::gfx942, 38, "mi300x-cpx", std::nullopt);
    }

    std::vector<std::string> solutionNames(SolutionVector<ContractionSolution> const& solutions)
    {
        std::vector<std::string> names;
        names.reserve(solutions.size());
        for(auto const& solution : solutions)
            names.push_back(solution->solutionName);
        return names;
    }
} // namespace

// ---------------------------------------------------------------------------
// GEMM_TYPE_ONLY_NO_RANKING (ROCM-31982)
//
// The row skip used to be driven by a process-global StringSet in
// TensileLite::Debug that the heuristic fallback installed before calling
// getAllSolutions and cleared afterwards. It is now a per-call argument, so a
// no-ranking search and a full search are independent.
// ---------------------------------------------------------------------------

TEST(ExactLogicLibraryTest, FindAllSolutionsGemmTypeOnlySearchesRankingRows)
{
    auto         lib     = buildRankingRowsLibrary();
    auto         problem = dummyProblem();
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    ASSERT_TRUE(device.isStandardCU());

    EXPECT_EQ(solutionNameSet(lib->findAllSolutions(
                  problem, device, SolutionLibrarySearchType::GEMM_TYPE_ONLY)),
              (std::set<std::string>{"equality", "gridbased", "prediction"}));
}

TEST(ExactLogicLibraryTest, FindAllSolutionsNoRankingSkipsRankingRows)
{
    auto         lib     = buildRankingRowsLibrary();
    auto         problem = dummyProblem();
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNameSet(lib->findAllSolutions(
                  problem, device, SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING)),
              (std::set<std::string>{"equality"}));
}

// The skip is suppressed on a non-standard-CU GPU, which still needs every row
// searched to find CU-fallback solutions.
TEST(ExactLogicLibraryTest, FindAllSolutionsNoRankingSearchesAllRowsOnNonStandardCuDevice)
{
    auto         lib     = buildRankingRowsLibrary();
    auto         problem = dummyProblem();
    const AMDGPU device  = makeNonStandardCuDevice();
    ASSERT_FALSE(device.isStandardCU());

    EXPECT_EQ(solutionNameSet(lib->findAllSolutions(
                  problem, device, SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING)),
              (std::set<std::string>{"equality", "gridbased", "prediction"}));
}

// The search type must reach the leaf library unchanged, so that the rows that
// are searched still apply the same "same GEMM type" filter GEMM_TYPE_ONLY does.
TEST(ExactLogicLibraryTest, FindAllSolutionsForwardsNoRankingSearchTypeToSubLibrary)
{
    auto         lib     = buildRankingRowsLibrary();
    auto         problem = dummyProblem();
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    lib->findAllSolutions(problem, device, SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING);

    auto const& equalityRow = *static_cast<RecordingLibrary const*>(lib->rows[0].second.get());
    EXPECT_EQ(equalityRow.lastSearchType(), SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING);
    EXPECT_TRUE(isGemmTypeOnlySearch(equalityRow.lastSearchType()));
}

// Interleaving the two search types on the same library must not let one leak
// into the other. With the Debug global, a full search issued while another
// call's window was open silently lost the ranking rows.
TEST(ExactLogicLibraryTest, FindAllSolutionsInterleavedSearchTypesAreIndependent)
{
    auto         lib     = buildRankingRowsLibrary();
    auto         problem = dummyProblem();
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    const std::set<std::string> all{"equality", "gridbased", "prediction"};
    const std::set<std::string> noRanking{"equality"};

    EXPECT_EQ(solutionNameSet(lib->findAllSolutions(
                  problem, device, SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING)),
              noRanking);
    EXPECT_EQ(solutionNameSet(lib->findAllSolutions(
                  problem, device, SolutionLibrarySearchType::GEMM_TYPE_ONLY)),
              all);
    EXPECT_EQ(solutionNameSet(lib->findAllSolutions(
                  problem, device, SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING)),
              noRanking);
    EXPECT_EQ(solutionNameSet(lib->findAllSolutions(problem, device)).count("gridbased"), 1u);
}

TEST(ExactLogicLibraryTest, FindTopSolutionsIncludesEqualityAndRangeByDefault)
{
    auto        lib     = buildMatchingRowsLibrary();
    auto        problem = dummyProblem();
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3).solutions),
              (std::vector<std::string>{"equality", "range", "prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsSkipsEqualityAndRangeWhenStreamKSchedulingOn)
{
    auto lib     = buildMatchingRowsLibrary();
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(1);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3).solutions),
              (std::vector<std::string>{"prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsSkipsEqualityAndRangeWhenStreamKSchedulingAuto)
{
    auto lib     = buildMatchingRowsLibrary();
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(2);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3).solutions),
              (std::vector<std::string>{"prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsSkipsEqualityAndRangeWhenStreamK5ForceDynamic)
{
    ScopedStreamK5ForceMode forceDynamic("1");

    auto        lib     = buildMatchingRowsLibrary();
    auto        problem = dummyProblem();
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3).solutions),
              (std::vector<std::string>{"prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsForceStaticDoesNotSkipEqualityAndRange)
{
    ScopedStreamK5ForceMode forceStatic("0");

    auto        lib     = buildMatchingRowsLibrary();
    auto        problem = dummyProblem();
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3).solutions),
              (std::vector<std::string>{"equality", "range", "prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsForceStaticOverridesStreamKScheduling)
{
    ScopedStreamK5ForceMode forceStatic("0");

    auto lib     = buildMatchingRowsLibrary();
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(1);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3).solutions),
              (std::vector<std::string>{"equality", "range", "prediction"}));
}
