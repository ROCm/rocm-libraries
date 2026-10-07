// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/MapLibrary.hpp>

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

    std::shared_ptr<ContractionProblemSelectionLibrary> buildAllMatchingKindsLibrary()
    {
        auto lib = buildMatchingRowsLibrary();

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::GridBasedMatching>()),
            std::make_shared<StubTopLibrary>(makeSolution("gridbased", 4))));

        lib->rows.push_back(ContractionProblemSelectionLibrary::Row(
            makeRowPredicate(std::make_shared<Predicates::Contraction::FreeSizeMatching>()),
            std::make_shared<StubTopLibrary>(makeSolution("freesize", 5))));

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

    using Types = std::array<rocisa::DataType, 6>;

    Types allOf(rocisa::DataType type)
    {
        return {type, type, type, type, type, type};
    }

    // And(TypesEqual(types), HighPrecisionAccumulate(hpa)).
    ContractionProblemPredicate typesRow(Types const& types, bool hpa)
    {
        auto typesEqual   = std::make_shared<Predicates::Contraction::TypesEqual>();
        typesEqual->value = types;
        auto accumulate = std::make_shared<Predicates::Contraction::HighPrecisionAccumulateEqual>();
        accumulate->value = hpa;
        return makeRowPredicate(std::make_shared<Predicates::And<ContractionProblemGemm>>(
            std::vector<std::shared_ptr<Predicates::Predicate<ContractionProblemGemm>>>{
                typesEqual, accumulate}));
    }

    // Rows selecting on data types, with rows for Float problems in several
    // places, so that the selected rows are not adjacent.
    std::vector<ContractionProblemSelectionLibrary::Row> typedRows()
    {
        using rocisa::DataType;
        std::vector<ContractionProblemSelectionLibrary::Row> rows;
        auto add = [&rows](Types const& types, bool hpa, std::string const& name, int index) {
            rows.emplace_back(typesRow(types, hpa),
                              std::make_shared<StubTopLibrary>(makeSolution(name, index)));
        };
        add(allOf(DataType::Half), false, "half", 1);
        add(allOf(DataType::Float), false, "float-a", 2);
        add(allOf(DataType::BFloat16), true, "bf16", 3);
        add(allOf(DataType::Float), true, "float-hpa", 4);
        add(allOf(DataType::Half), true, "half-hpa", 5);
        add(allOf(DataType::Float), false, "float-b", 6);
        return rows;
    }

    std::vector<ContractionProblemGemm> typedProblems()
    {
        std::vector<ContractionProblemGemm> problems;
        for(auto type : {rocisa::DataType::Float, rocisa::DataType::Half, rocisa::DataType::Double})
        {
            for(bool hpa : {false, true})
            {
                auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                                    false,
                                                                    type,
                                                                    type,
                                                                    type,
                                                                    type,
                                                                    64,
                                                                    64,
                                                                    64,
                                                                    1,
                                                                    64,
                                                                    4096,
                                                                    64,
                                                                    4096,
                                                                    64,
                                                                    4096,
                                                                    64,
                                                                    4096,
                                                                    1.0);
                problem.setComputeInputTypeA(type);
                problem.setComputeInputTypeB(type);
                problem.setHighPrecisionAccumulate(hpa);
                problems.push_back(problem);
            }
        }
        return problems;
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

TEST(ExactLogicLibraryTest, FindTopSolutionsKeepsGridBasedAndFreeSizeWhenStreamKSchedulingOn)
{
    auto lib     = buildAllMatchingKindsLibrary();
    auto problem = dummyProblem();
    problem.setParams().setStreamKTileSchedulingMode(1);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 5)),
              (std::vector<std::string>{"prediction", "gridbased", "freesize"}));
}

TEST(ExactLogicLibraryTest, FindTopSolutionsKeepsGridBasedAndFreeSizeWhenStreamK5ForceDynamic)
{
    ScopedStreamK5ForceMode forceDynamic("1");

    auto         lib     = buildAllMatchingKindsLibrary();
    auto         problem = dummyProblem();
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    EXPECT_EQ(solutionNames(lib->findTopSolutions(problem, device, 5)),
              (std::vector<std::string>{"prediction", "gridbased", "freesize"}));
}

TEST(ExactLogicLibraryTest, PreparedRowsSelectLikeRowsAddedOneByOne)
{
    ContractionProblemSelectionLibrary prepared(typedRows());
    ContractionProblemSelectionLibrary added;
    for(auto const& row : typedRows())
        added.rows.push_back(row);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    for(auto const& problem : typedProblems())
    {
        SCOPED_TRACE(problem.description());
        for(int count : {1, 2, 3})
        {
            auto const expected = solutionNames(added.findTopSolutions(problem, device, count));
            EXPECT_EQ(solutionNames(prepared.findTopSolutions(problem, device, count)), expected);
            EXPECT_EQ(prepared.lastFindTopAlreadyRetAll(), added.lastFindTopAlreadyRetAll());
        }
        auto const best     = prepared.findBestSolution(problem, device);
        auto const expected = added.findBestSolution(problem, device);
        EXPECT_EQ(best ? best->solutionName : "", expected ? expected->solutionName : "");
    }
}

TEST(ExactLogicLibraryTest, RowsOfTheProblemTypesAreReturnedInRowOrder)
{
    ContractionProblemSelectionLibrary lib(typedRows());
    auto                               problem = typedProblems().front();
    ASSERT_EQ(problem.a().dataType(), rocisa::DataType::Float);
    ASSERT_FALSE(problem.highPrecisionAccumulate());

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    EXPECT_EQ(solutionNames(lib.findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"float-a", "float-b"}));
    EXPECT_TRUE(lib.lastFindTopAlreadyRetAll());
    EXPECT_EQ(lib.findBestSolution(problem, device)->solutionName, "float-a");

    problem.setHighPrecisionAccumulate(true);
    EXPECT_EQ(solutionNames(lib.findTopSolutions(problem, device, 3)),
              (std::vector<std::string>{"float-hpa"}));
}

TEST(ExactLogicLibraryTest, RowsWithoutLeadingTypesAreAllEvaluated)
{
    auto rows = typedRows();
    rows.insert(rows.begin() + 2,
                ContractionProblemSelectionLibrary::Row(
                    makeRowPredicate(std::make_shared<Predicates::True<ContractionProblemGemm>>()),
                    std::make_shared<StubTopLibrary>(makeSolution("any", 7))));
    ContractionProblemSelectionLibrary lib(rows);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    EXPECT_EQ(solutionNames(lib.findTopSolutions(typedProblems().front(), device, 4)),
              (std::vector<std::string>{"float-a", "any", "float-b"}));
}

TEST(ExactLogicLibraryTest, PredicateDebuggingEvaluatesEveryRow)
{
    const char*       previous = std::getenv("TENSILE_DB");
    const std::string saved    = previous ? previous : "";
    setenv("TENSILE_DB", "0x40010", 1);
    Debug::Instance().reloadDebugBitsForTest();

    ContractionProblemSelectionLibrary lib(typedRows());
    const AMDGPU                       device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    ::testing::internal::CaptureStdout();
    auto const names = solutionNames(lib.findTopSolutions(typedProblems().front(), device, 3));
    auto const trace = ::testing::internal::GetCapturedStdout();

    if(previous)
        setenv("TENSILE_DB", saved.c_str(), 1);
    else
        unsetenv("TENSILE_DB");
    Debug::Instance().reloadDebugBitsForTest();

    EXPECT_EQ(names, (std::vector<std::string>{"float-a", "float-b"}));
    size_t rows = 0;
    for(size_t at = trace.find("PREDICATE: ExactLogic: Problem"); at != std::string::npos;
        at        = trace.find("PREDICATE: ExactLogic: Problem", at + 1))
        rows++;
    EXPECT_EQ(rows, lib.rows.size());
}

TEST(ExactLogicLibraryTest, RowKindsSkipTheSameRowsAsTheirTypeNames)
{
    auto lib = buildAllMatchingKindsLibrary();
    for(size_t i = 0; i < lib->rows.size(); i++)
        EXPECT_EQ(lib->rowKind(i), libraryRowKind(lib->rows[i].first)) << i;

    ContractionProblemSelectionLibrary prepared(lib->rows);
    auto const                         problem = dummyProblem();
    const AMDGPU                       device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    EXPECT_EQ(solutionNames(prepared.findTopSolutions(problem, device, 5)),
              solutionNames(lib->findTopSolutions(problem, device, 5)));

    EXPECT_EQ(prepared.rowKind(0), LibraryRowKind::EqualityMatching);
    EXPECT_EQ(prepared.rowKind(1), LibraryRowKind::RangeMatching);
    EXPECT_EQ(prepared.rowKind(2), LibraryRowKind::PredictionMatching);
    EXPECT_EQ(prepared.rowKind(3), LibraryRowKind::GridBasedMatching);
    EXPECT_EQ(prepared.rowKind(4), LibraryRowKind::FreeSizeMatching);
}

TEST(ExactLogicLibraryTest, HardwareRowsSelectLikeRowsAddedOneByOne)
{
    using Rows = std::vector<ContractionHardwareSelectionLibrary::Row>;
    auto row   = [](HardwarePredicate predicate, std::string const& name, int index) {
        return ContractionHardwareSelectionLibrary::Row(
            std::move(predicate), std::make_shared<StubTopLibrary>(makeSolution(name, index)));
    };
    Rows const rows{row(makeHwPred(AMDGPU::Processor::gfx950, _MI355_CHIP_ID), "mi355", 1),
                    row(makeHwPred(AMDGPU::Processor::gfx950, _MI350_CHIP_ID, _SPX_CU), "spx", 2),
                    row(makeHwPred(AMDGPU::Processor::gfx950, _MI350_CHIP_ID, _CPX_CU), "cpx", 3),
                    row(makeHwPred(AMDGPU::Processor::gfx950, _MI350_CHIP_ID), "mi350", 4),
                    row(makeHwPred(AMDGPU::Processor::gfx942), "gfx942", 5)};

    ContractionHardwareSelectionLibrary prepared(rows);
    ContractionHardwareSelectionLibrary added;
    for(auto const& r : rows)
        added.rows.push_back(r);

    auto const problem = dummyProblem();
    for(auto const& device : {makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx"),
                              makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx"),
                              makeDevice(_MI355_CHIP_ID, _SPX_CU, "mi355spx"),
                              makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx")})
    {
        SCOPED_TRACE(device.deviceName + " " + std::to_string(device.computeUnitCount));
        // Twice: the second call reads the rows remembered for the device.
        for(int repeat = 0; repeat < 2; repeat++)
        {
            EXPECT_EQ(solutionNames(prepared.findTopSolutions(problem, device, 5)),
                      solutionNames(added.findTopSolutions(problem, device, 5)));
            EXPECT_EQ(prepared.findBestSolution(problem, device)->solutionName,
                      added.findBestSolution(problem, device)->solutionName);
        }
    }
    EXPECT_EQ(solutionNames(prepared.findTopSolutions(
                  problem, makeDevice(_MI355_CHIP_ID, _SPX_CU, "mi355spx"), 5)),
              (std::vector<std::string>{"mi355", "spx", "mi350"}));
}

TEST(ExactLogicLibraryTest, HardwareRowsReadingMoreThanTheDeviceAreEvaluatedEveryCall)
{
    struct NamedDevice : public Predicates::Predicate_CRTP<NamedDevice, AMDGPU>
    {
        enum
        {
            HasIndex = false,
            HasValue = false
        };
        static std::string Type()
        {
            return "NamedDevice";
        }
        bool operator()(AMDGPU const& gpu) const override
        {
            return gpu.deviceName == "first";
        }
    };

    ContractionHardwareSelectionLibrary lib(std::vector<ContractionHardwareSelectionLibrary::Row>{
        {HardwarePredicate(std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(
             std::make_shared<NamedDevice>())),
         std::make_shared<StubTopLibrary>(makeSolution("named", 1))}});

    auto const problem = dummyProblem();
    EXPECT_EQ(solutionNames(
                  lib.findTopSolutions(problem, makeDevice(_MI350_CHIP_ID, _SPX_CU, "first"), 1)),
              (std::vector<std::string>{"named"}));
    EXPECT_TRUE(
        lib.findTopSolutions(problem, makeDevice(_MI350_CHIP_ID, _SPX_CU, "second"), 1).empty());
}

TEST(ProblemMapLibraryTest, OperationIdentifierKeyFindsTheSameLibrary)
{
    auto problem = dummyProblem();
    auto other   = ContractionProblemGemm::GEMM(true, false, 64, 64, 64, 64, 64, 64, 1.0, false, 1);
    ASSERT_NE(problem.operationIdentifier(), other.operationIdentifier());

    ContractionProblemMapLibrary lib;
    lib.property = std::make_shared<Contraction::OperationIdentifier>();
    lib.map.emplace(problem.operationIdentifier(),
                    std::make_shared<StubTopLibrary>(makeSolution("mapped", 1)));

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    for(bool byOperationIdentifier : {false, true})
    {
        SCOPED_TRACE(byOperationIdentifier);
        lib.keyIsOperationIdentifier = byOperationIdentifier;
        EXPECT_EQ(solutionNames(lib.findTopSolutions(problem, device, 1)),
                  (std::vector<std::string>{"mapped"}));
        EXPECT_TRUE(lib.findTopSolutions(other, device, 1).empty());
    }
}

TEST(ExactLogicLibraryTest, FindTopSolutionsReturnsOnlyPredictionWithPredictionLib)
{
    const char*       previous = std::getenv("TENSILE_PREDICTION_LIB");
    const std::string saved    = previous ? previous : "";
    setenv("TENSILE_PREDICTION_LIB", "1", 1);
    Debug::Instance().reloadDebugBitsForTest();

    auto         lib     = buildAllMatchingKindsLibrary();
    auto         problem = dummyProblem();
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    const auto   names   = solutionNames(lib->findTopSolutions(problem, device, 5));

    if(previous)
        setenv("TENSILE_PREDICTION_LIB", saved.c_str(), 1);
    else
        unsetenv("TENSILE_PREDICTION_LIB");
    Debug::Instance().reloadDebugBitsForTest();

    EXPECT_EQ(names, (std::vector<std::string>{"prediction"}));
}
