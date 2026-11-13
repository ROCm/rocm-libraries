// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the --best-solution iterator (SolutionIterator::Default with
// best-solution set): what it runs when the library selects no solution, with and
// without --require-selected-solution.

#include <gtest/gtest.h>

#include "ResultReporter.hpp"
#include "SolutionIterator.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    // Returns `selection` from findTopSolutions (the call the --best-solution iterator
    // makes), whatever the problem.
    class FixedSelectionLibrary : public ContractionLibrary
    {
    public:
        SolutionVector<ContractionSolution> selection;

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                             Hardware const&,
                                                             int) const override
        {
            return selection;
        }

        std::shared_ptr<ContractionSolution>
            getSolutionByIndex(ContractionProblemGemm const&, Hardware const&, int) const override
        {
            return nullptr;
        }
        std::shared_ptr<ContractionSolution> findBestSolution(ContractionProblemGemm const&,
                                                              Hardware const&,
                                                              double*) const override
        {
            return nullptr;
        }
        SolutionSet<ContractionSolution> findAllSolutions(ContractionProblemGemm const&,
                                                          Hardware const&,
                                                          SolutionLibrarySearchType) const override
        {
            return {};
        }
        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType) const override
        {
            return {};
        }
        std::string type() const override
        {
            return "FixedSelection";
        }
        std::string description() const override
        {
            return "FixedSelection";
        }
    };

    class FakeResultReporter : public ResultReporter
    {
    public:
        std::string logged;

        void reportValue_string(std::string const&, std::string const&) override {}
        void reportValue_uint(std::string const&, uint64_t) override {}
        void reportValue_int(std::string const&, int64_t) override {}
        void reportValue_double(std::string const&, double) override {}
        void reportValue_sizes(std::string const&, std::vector<size_t> const&) override {}
        void reportValue_vecOfSizes(std::string const&,
                                    std::vector<std::vector<size_t>> const&) override
        {
        }
        void finalizeReport() override {}
        bool logAtLevel(LogLevel) override
        {
            return true;
        }
        void logMessage(LogLevel, std::string const& message) override
        {
            logged += message;
        }
    };

    po::variables_map bestSolutionArgs(bool requireSelectedSolution)
    {
        po::variables_map vm;
        vm["best-solution"].value()             = true;
        vm["PrintWinnersOnly"].value()          = false;
        vm["prediction-threshold"].value()      = 2.0; // the client default: no Formocast ranking
        vm["print-rejected-predicates"].value() = false;
        vm["require-selected-solution"].value() = requireSelectedSolution;
        return vm;
    }

    // Library solution 0 is the fallback; a selection, when there is one, is solution 1.
    struct SolutionIteratorSelectionTest : public ::testing::Test
    {
        std::shared_ptr<ContractionSolution> solution0 = std::make_shared<ContractionSolution>();
        std::shared_ptr<ContractionSolution> solution1 = std::make_shared<ContractionSolution>();
        std::shared_ptr<FixedSelectionLibrary> selector = std::make_shared<FixedSelectionLibrary>();
        std::shared_ptr<MasterContractionLibrary> library
            = std::make_shared<MasterContractionLibrary>();
        std::shared_ptr<Hardware>           hardware = std::make_shared<AMDGPU>();
        std::shared_ptr<FakeResultReporter> reporter = std::make_shared<FakeResultReporter>();

        ContractionProblemGemm problem
            = ContractionProblemGemm::GEMM(true, false, 4, 4, 4, 4, 4, 4, 1.0, false, 1);

        void SetUp() override
        {
            solution0->index = 0;
            solution1->index = 1;

            library->solutions = {{0, solution0}, {1, solution1}};
            library->library   = selector;
        }

        std::shared_ptr<SolutionIterator> makeIterator(bool requireSelectedSolution)
        {
            auto iterator = SolutionIterator::Default(
                library, hardware, bestSolutionArgs(requireSelectedSolution));
            iterator->setReporter(reporter);
            return iterator;
        }
    };
}

TEST_F(SolutionIteratorSelectionTest, SelectedSolutionRunsWithOrWithoutTheOption)
{
    selector->selection = {solution1};
    for(bool require : {false, true})
    {
        SCOPED_TRACE(require);
        auto iterator = makeIterator(require);
        iterator->preProblem(&problem);
        ASSERT_TRUE(iterator->moreSolutionsInProblem());
        EXPECT_EQ(iterator->getSolution(), solution1);
        EXPECT_EQ(iterator->error(), 0);
    }
    EXPECT_EQ(reporter->logged, "");
}

// Default (option off): the iterator falls back to library solution 0.
TEST_F(SolutionIteratorSelectionTest, NothingSelectedFallsBackToSolution0ByDefault)
{
    auto iterator = makeIterator(false);
    iterator->preProblem(&problem);
    ASSERT_TRUE(iterator->moreSolutionsInProblem());
    EXPECT_EQ(iterator->getSolution(), solution0);
    EXPECT_EQ(iterator->error(), 0);
    EXPECT_EQ(reporter->logged, "");
}

// --require-selected-solution: no fallback; the problem is skipped, reported, and counted
// as an error per problem, so the client exits non-zero. A later problem that selects a
// solution still runs.
TEST_F(SolutionIteratorSelectionTest, NothingSelectedIsAnErrorWithTheOption)
{
    auto iterator = makeIterator(true);

    iterator->preProblem(&problem);
    EXPECT_FALSE(iterator->moreSolutionsInProblem());
    EXPECT_EQ(iterator->error(), 1);
    EXPECT_NE(reporter->logged.find("No solution selected for problem "
                                    + problem.operationIdentifier() + " sizes 4 4 1 4"),
              std::string::npos)
        << reporter->logged;
    iterator->postProblem();

    iterator->preProblem(&problem);
    EXPECT_FALSE(iterator->moreSolutionsInProblem());
    EXPECT_EQ(iterator->error(), 2);
    iterator->postProblem();

    selector->selection = {solution1};
    iterator->preProblem(&problem);
    ASSERT_TRUE(iterator->moreSolutionsInProblem());
    EXPECT_EQ(iterator->getSolution(), solution1);
    EXPECT_EQ(iterator->error(), 2);
}
