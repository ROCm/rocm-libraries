// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
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

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                             Hardware const&,
                                                             int) const override
        {
            return m_solution ? SolutionVector<ContractionSolution>({m_solution})
                              : SolutionVector<ContractionSolution>();
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

    std::vector<std::string> solutionNames(SolutionVector<ContractionSolution> const& solutions)
    {
        std::vector<std::string> names;
        names.reserve(solutions.size());
        for(auto const& solution : solutions)
            names.push_back(solution->solutionName);
        return names;
    }
} // namespace

TEST(ExactLogicLibraryTest, FindTopSolutionsIncludesEqualityAndRangeByDefault)
{
    auto        lib     = buildMatchingRowsLibrary();
    auto        problem = dummyProblem();
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"equality", "range", "prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsSkipsEqualityAndRangeWhenStreamKSchedulingOn)
{
    auto lib     = buildMatchingRowsLibrary();
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(1);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsSkipsEqualityAndRangeWhenStreamKSchedulingAuto)
{
    auto lib     = buildMatchingRowsLibrary();
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(2);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsSkipsEqualityAndRangeWhenStreamK5ForceDynamic)
{
    ScopedStreamK5ForceMode forceDynamic("1");

    auto        lib     = buildMatchingRowsLibrary();
    auto        problem = dummyProblem();
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsForceStaticDoesNotSkipEqualityAndRange)
{
    ScopedStreamK5ForceMode forceStatic("0");

    auto        lib     = buildMatchingRowsLibrary();
    auto        problem = dummyProblem();
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"equality", "range", "prediction"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsForceStaticOverridesStreamKScheduling)
{
    ScopedStreamK5ForceMode forceStatic("0");

    auto lib     = buildMatchingRowsLibrary();
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(1);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"equality", "range", "prediction"}));
}
