// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the --best-solution iterator (SolutionIterator::Default with
// best-solution set): what it runs when the library selects no solution, with and
// without --require-selected-solution.

#include <gtest/gtest.h>

#include "ClientArgsTestUtils.hpp"
#include "ClientProblemFactory.hpp"
#include "ResultReporter.hpp"
#include "SolutionIterator.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>

#include <memory>
#include <string>
#include <tuple>
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

// --split-devices: a solution has to be valid for the slice of every device, not only for the full
// problem the library was asked about.
namespace
{
    class ExposedIterator : public TopSolutionIterator
    {
    public:
        using TopSolutionIterator::TopSolutionIterator;
        bool valid(ContractionSolution& solution)
        {
            return checkSolution(solution);
        }
    };
}

TEST_F(SolutionIteratorSelectionTest, ASolutionMustBeValidForEverySlice)
{
    selector->selection = {solution1};
    // M = 128 split over two devices is 64 rows each; this solution only takes multiples of 128.
    solution1->problemPredicate = std::make_shared<Predicates::Contraction::SizeMultiple>(0, 128);

    ClientProblemFactory factory(TestUtils::makeSplitClientArgs({128, 128, 1, 64}, 2, "M"));
    auto*                full = factory.problems()[0].get();

    ExposedIterator split(library, hardware, 2.0, 1, false, false);
    split.setReporter(reporter);
    split.setSplit(&factory);
    split.preProblem(full);
    EXPECT_FALSE(split.valid(*solution1));

    // The same solution is fine for the full problem when it is not split, and a solution without
    // restrictions is fine for every slice.
    ExposedIterator whole(library, hardware, 2.0, 1, false, false);
    whole.setReporter(reporter);
    whole.preProblem(full);
    EXPECT_TRUE(whole.valid(*solution1));
    EXPECT_TRUE(split.valid(*solution0));
}

// With prediction on, the iterator ranks only the solutions that pass checkSolution. When none does
// there is nothing to run, and the iterator must not index into the empty ranking.
TEST_F(SolutionIteratorSelectionTest, PredictionWithNoValidSolutionRunsNothing)
{
    // M = 4 is not a multiple of 128, so neither solution is valid.
    auto sizeMultiple = std::make_shared<Predicates::Contraction::SizeMultiple>(0, 128);
    solution0->problemPredicate = sizeMultiple;
    solution1->problemPredicate = sizeMultiple;

    // Prediction is available: the hardware has an analytical model.
    auto device                = std::make_shared<hip::HipAMDGPU>();
    device->processor          = AMDGPU::Processor::gfx950;
    device->computeUnitCount   = 256;
    device->deviceName         = "prediction-without-valid-solutions-test";
    device->analyticalHardware = std::make_shared<origami::hardware_t>(
        origami::hardware_t::architecture_t::gfx950,
        256, // N_CU
        163840, // lds_capacity
        262144, // rf_capacity
        8, // NUM_XCD
        1.0, // mem1_perf_ratio
        1.0, // mem2_perf_ratio
        1.0, // mem3_perf_ratio
        4000000, // L2_capacity
        1.2, // compute_clock_ghz
        1, // parallel_mi_cu
        std::make_tuple(0.0, 0.008, 0.0)); // mem_bw_per_wg_coefficients

    AllSolutionsIterator iterator(library, device, 1.0, 0, -1, false);
    iterator.setReporter(reporter);
    iterator.preProblem(&problem);
    EXPECT_FALSE(iterator.moreSolutionsInProblem());
}

// Without a prediction (the hardware has no analytical model) nothing can be ranked, so every
// solution is visited in index order whatever the threshold.
TEST_F(SolutionIteratorSelectionTest, WithoutPredictionEverySolutionIsVisited)
{
    for(double threshold : {0.0, 0.5, 1.0, 2.0})
    {
        AllSolutionsIterator iterator(library, hardware, threshold, 0, -1, false);
        iterator.setReporter(reporter);
        iterator.preProblem(&problem);

        std::vector<int32_t> visited;
        while(iterator.moreSolutionsInProblem())
        {
            visited.push_back(iterator.getSolution()->index);
            iterator.postSolution();
        }
        EXPECT_EQ(visited, (std::vector<int32_t>{0, 1})) << "threshold " << threshold;
    }
}
