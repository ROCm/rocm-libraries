// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ProviderRows limits the provider rows ExactLogicLibrary::findTopSolutions walks on
// the calling thread, and CachingLibrary keeps the results of each kind of walk apart.

#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/ProviderRows.hpp>

#include <algorithm>
#include <memory>
#include <vector>

using namespace TensileLite;

namespace
{
    using GemmProblem  = ContractionProblemGemm;
    using GemmSolution = ContractionSolution;
    using Library      = SolutionLibrary<GemmProblem, GemmSolution>;
    using Selection    = ProblemSelectionLibrary<GemmProblem, GemmSolution>;

    // A provider that returns the first solutions of a fixed list.
    struct FixedLibrary : public Library
    {
        SolutionVector<GemmSolution> solutions;
        mutable int                  findTopCalls = 0;

        explicit FixedLibrary(std::vector<int> const& indices)
        {
            for(int index : indices)
            {
                auto solution   = std::make_shared<GemmSolution>();
                solution->index = index;
                solutions.push_back(solution);
            }
        }

        std::shared_ptr<GemmSolution>
            getSolutionByIndex(GemmProblem const&, Hardware const&, const int) const override
        {
            return nullptr;
        }

        std::shared_ptr<GemmSolution>
            findBestSolution(GemmProblem const&, Hardware const&, double*) const override
        {
            return nullptr;
        }

        SolutionSet<GemmSolution> findAllSolutions(GemmProblem const&,
                                                   Hardware const&,
                                                   SolutionLibrarySearchType) const override
        {
            return {};
        }

        SolutionSet<GemmSolution>
            findAllSolutionsGroupedGemm(std::vector<GemmProblem> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType) const override
        {
            return {};
        }

        SolutionVector<GemmSolution>
            findTopSolutions(GemmProblem const&, Hardware const&, int numSolutions) const override
        {
            ++findTopCalls;
            const auto count = std::min<size_t>(numSolutions, solutions.size());
            return {solutions.begin(), solutions.begin() + count};
        }

        std::string type() const override
        {
            return "Fixed";
        }
        std::string description() const override
        {
            return "Fixed";
        }
    };

    template <typename Predicate>
    Selection::Row row(std::shared_ptr<Library> library)
    {
        return {ProblemPredicate<GemmProblem>(std::make_shared<Predicate>()), library};
    }

    std::vector<int> indices(SolutionVector<GemmSolution> const& solutions)
    {
        std::vector<int> rv;
        for(auto const& solution : solutions)
            rv.push_back(solution->index);
        return rv;
    }

    class ProviderRowsTest : public ::testing::Test
    {
    protected:
        std::shared_ptr<FixedLibrary> equality = std::make_shared<FixedLibrary>(std::vector{1, 2});
        std::shared_ptr<FixedLibrary> prediction
            = std::make_shared<FixedLibrary>(std::vector{10, 11, 12});
        std::shared_ptr<Selection> providers = std::make_shared<Selection>(
            std::vector{row<Predicates::Contraction::EqualityMatching>(equality),
                        row<Predicates::Contraction::PredictionMatching>(prediction)});

        GemmProblem problem
            = GemmProblem::GEMM(false, false, 256, 128, 512, 256, 512, 256, 0.0, false, 1);
        AMDGPU hardware = AMDGPU(AMDGPU::Processor::gfx950, 256, "gfx950");

        std::vector<int> find(Library const& library, ProviderRows rows, int count)
        {
            ProviderRowsScope scope(rows);
            return indices(library.findTopSolutions(problem, hardware, count));
        }
    };
}

TEST_F(ProviderRowsTest, smoke_WalkSkipsTheOtherProviders)
{
    EXPECT_EQ(find(*providers, ProviderRows::All, 4), (std::vector{1, 2, 10, 11}));
    EXPECT_EQ(find(*providers, ProviderRows::EqualityOnly, 4), (std::vector{1, 2}));
    EXPECT_EQ(find(*providers, ProviderRows::ExceptEquality, 4), (std::vector{10, 11, 12}));
    EXPECT_EQ(currentProviderRows(), ProviderRows::All);
}

TEST_F(ProviderRowsTest, smoke_RowsLeadingToProvidersAreWalked)
{
    Selection outer({row<Predicates::True<GemmProblem>>(providers)});

    EXPECT_EQ(find(outer, ProviderRows::EqualityOnly, 4), (std::vector{1, 2}));
    EXPECT_EQ(find(outer, ProviderRows::ExceptEquality, 2), (std::vector{10, 11}));
}

TEST_F(ProviderRowsTest, smoke_CacheKeepsWalksApart)
{
    CachingLibrary<GemmProblem, GemmSolution> cache(providers);

    EXPECT_EQ(find(cache, ProviderRows::All, 4), (std::vector{1, 2, 10, 11}));
    EXPECT_EQ(find(cache, ProviderRows::EqualityOnly, 1), (std::vector{1}));
    EXPECT_EQ(find(cache, ProviderRows::ExceptEquality, 2), (std::vector{10, 11}));

    const int equalityCalls = equality->findTopCalls;
    EXPECT_EQ(find(cache, ProviderRows::EqualityOnly, 1), (std::vector{1}));
    EXPECT_EQ(find(cache, ProviderRows::All, 4), (std::vector{1, 2, 10, 11}));
    EXPECT_EQ(equality->findTopCalls, equalityCalls) << "Repeated walks were not cached";
}
