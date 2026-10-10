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

namespace
{
    // Sets an environment variable (unsets it for nullopt) for the lifetime of
    // the object, and makes Debug read it again.
    class ScopedSetting
    {
    public:
        ScopedSetting(char const* name, std::optional<std::string> value)
            : m_name(name)
        {
            if(char const* previous = std::getenv(name))
                m_previous = previous;
            if(value)
                setenv(name, value->c_str(), 1);
            else
                unsetenv(name);
            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedSetting()
        {
            if(m_previous)
                setenv(m_name, m_previous->c_str(), 1);
            else
                unsetenv(m_name);
            Debug::Instance().reloadDebugBitsForTest();
        }

        ScopedSetting(ScopedSetting const&)            = delete;
        ScopedSetting& operator=(ScopedSetting const&) = delete;

    private:
        char const*                m_name;
        std::optional<std::string> m_previous;
    };

    // Returns its kernels, as many as asked for, and counts the calls that ask
    // and those made in a TilewrightRowOrder; as a Prediction row's library it
    // says whether a tilewright model ranks it.
    class KernelRow : public ContractionLibrary
    {
    public:
        explicit KernelRow(std::vector<std::string> const& names, bool model = false)
            : m_model(model)
        {
            for(auto const& name : names)
                m_solutions.push_back(makeSolution(name, static_cast<int>(m_solutions.size())));
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
            return m_solutions.empty() ? nullptr : m_solutions.front();
        }

        SolutionSet<ContractionSolution>
            findAllSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             SolutionLibrarySearchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            return SolutionSet<ContractionSolution>(m_solutions.begin(), m_solutions.end());
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
            return "KernelRow";
        }

        std::string description() const override
        {
            return "KernelRow";
        }

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                             Hardware const&,
                                                             int count) const override
        {
            m_calls++;
            if(TilewrightRowOrder::current() != nullptr)
                m_ordered++;
            return SolutionVector<ContractionSolution>(
                m_solutions.begin(),
                m_solutions.begin() + std::min<size_t>(m_solutions.size(), std::max(count, 0)));
        }

        bool hasTilewrightModel() const override
        {
            return m_model;
        }

        int calls() const
        {
            return m_calls;
        }

        int ordered() const
        {
            return m_ordered;
        }

    private:
        SolutionVector<ContractionSolution> m_solutions;
        bool                                m_model;
        mutable int                         m_calls   = 0;
        mutable int                         m_ordered = 0;
    };

    using Row       = ContractionProblemSelectionLibrary::Row;
    using Rows      = std::vector<Row>;
    using Equality  = Predicates::Contraction::EqualityMatching;
    using GridBased = Predicates::Contraction::GridBasedMatching;
    using Predicted = Predicates::Contraction::PredictionMatching;
    using AnyRow    = Predicates::True<ContractionProblemGemm>;

    template <typename Kind>
    Row rowOf(std::shared_ptr<ContractionLibrary> library)
    {
        return Row(makeRowPredicate(std::make_shared<Kind>()), std::move(library));
    }

    std::shared_ptr<KernelRow> kernels(std::vector<std::string> const& names, bool model = false)
    {
        return std::make_shared<KernelRow>(names, model);
    }

    // The library files of a problem type as a lazily loaded library holds
    // them: a row of the problem's types for each file, in the given order,
    // holding the rows of that file.
    std::shared_ptr<ContractionProblemSelectionLibrary> files(std::vector<Rows> const& rows)
    {
        Rows typeRows;
        for(auto const& file : rows)
            typeRows.emplace_back(typesRow(allOf(rocisa::DataType::Float), false),
                                  std::make_shared<ContractionProblemSelectionLibrary>(file));
        return std::make_shared<ContractionProblemSelectionLibrary>(typeRows);
    }

    using Names = std::vector<std::string>;

    // The solutions `lib` returns for `count` with tilewright off and on.
    std::pair<Names, Names> bothOrders(ContractionProblemSelectionLibrary const& lib, int count)
    {
        auto const   problem = typedProblems().front();
        const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
        Names        off, on;
        {
            ScopedSetting tilewright("TENSILE_USE_TILEWRIGHT", std::nullopt);
            off = solutionNames(lib.findTopSolutions(problem, device, count));
        }
        {
            ScopedSetting tilewright("TENSILE_USE_TILEWRIGHT", "1");
            on = solutionNames(lib.findTopSolutions(problem, device, count));
        }
        return {off, on};
    }
} // namespace

TEST(ExactLogicLibraryTest, ALoadedTilewrightModelIsSearchedBeforeTheGridBasedRowsBeforeIt)
{
    // An Equality file without the problem's size, a GridBased file, then the
    // file of the Prediction table, which holds a GridBased row of its own.
    auto exact = kernels({});
    auto grid  = kernels({"gridbased"});
    auto table = kernels({"prediction-1", "prediction-2"}, true);
    auto own   = kernels({"gridbased-2"});
    auto lib   = files({{rowOf<Equality>(exact)},
                        {rowOf<GridBased>(grid)},
                        {rowOf<Predicted>(table), rowOf<GridBased>(own)}});

    auto const [off1, on1] = bothOrders(*lib, 1);
    EXPECT_EQ(off1, (Names{"gridbased"}));
    EXPECT_EQ(on1, (Names{"prediction-1"}));
    // With tilewright on, the rows after the model's were not searched.
    EXPECT_EQ(exact->calls(), 2);
    EXPECT_EQ(grid->calls(), 1);
    EXPECT_EQ(table->calls(), 1);
    EXPECT_EQ(own->calls(), 0);
    EXPECT_EQ(exact->ordered(), 1);
    EXPECT_EQ(table->ordered(), 1);

    auto const [off, on] = bothOrders(*lib, 5);
    EXPECT_EQ(off, (Names{"gridbased", "prediction-1", "prediction-2", "gridbased-2"}));
    EXPECT_EQ(on, (Names{"prediction-1", "prediction-2", "gridbased", "gridbased-2"}));
    EXPECT_TRUE(lib->lastFindTopAlreadyRetAll());
    // Each walk searched each row once.
    EXPECT_EQ(exact->calls(), 4);
    EXPECT_EQ(grid->calls(), 3);
    EXPECT_EQ(table->calls(), 3);
    EXPECT_EQ(own->calls(), 2);
}

TEST(ExactLogicLibraryTest, TheRowsKeepTheirOrderWithoutALoadedTilewrightModel)
{
    auto grid  = kernels({"gridbased"});
    auto table = kernels({"prediction-1", "prediction-2"});
    auto own   = kernels({"gridbased-2"});

    auto lib = files({{rowOf<GridBased>(grid)}, {rowOf<Predicted>(table), rowOf<GridBased>(own)}});

    auto const [off1, on1] = bothOrders(*lib, 1);
    EXPECT_EQ(on1, off1);
    EXPECT_EQ(on1, (Names{"gridbased"}));
    auto const [off, on] = bothOrders(*lib, 5);
    EXPECT_EQ(on, off);
    EXPECT_EQ(on, (Names{"gridbased", "prediction-1", "prediction-2", "gridbased-2"}));
    EXPECT_EQ(grid->calls(), 4);
    EXPECT_EQ(table->calls(), 2);
    EXPECT_EQ(own->calls(), 2);
}

TEST(ExactLogicLibraryTest, EqualityRowsAnswerTheirSizesBeforeALoadedTilewrightModel)
{
    for(bool model : {true, false})
    {
        SCOPED_TRACE(model);
        auto exact = kernels({"equality"});
        auto grid  = kernels({"gridbased"});
        auto table = kernels({"prediction"}, model);

        auto lib = files(
            {{rowOf<GridBased>(grid)}, {rowOf<Equality>(exact)}, {rowOf<Predicted>(table)}});

        auto const [off1, on1] = bothOrders(*lib, 1);
        EXPECT_EQ(off1, (Names{"gridbased"}));
        EXPECT_EQ(on1, (model ? Names{"equality"} : Names{"gridbased"}));
        auto const [off, on] = bothOrders(*lib, 3);
        EXPECT_EQ(off, (Names{"gridbased", "equality", "prediction"}));
        EXPECT_EQ(on, (model ? Names{"equality", "prediction", "gridbased"} : off));
        // With a model, the second walk searched the Equality row, which
        // answered a request for one solution.
        EXPECT_EQ(exact->calls(), model ? 3 : 2);
        EXPECT_EQ(table->calls(), 2);
    }

    auto exact = kernels({"equality"});
    auto lib   = files({{rowOf<Equality>(exact)},
                        {rowOf<GridBased>(kernels({"gridbased"}))},
                        {rowOf<Predicted>(kernels({"prediction"}, true))}});
    EXPECT_EQ(bothOrders(*lib, 1).second, (Names{"equality"}));
    EXPECT_EQ(bothOrders(*lib, 3).second, (Names{"equality", "prediction", "gridbased"}));
}

TEST(ExactLogicLibraryTest, TheRowsAfterALoadedTilewrightModelFollowInTheirOrder)
{
    auto lib = files({{rowOf<Predicted>(kernels({"origami"}))},
                      {rowOf<GridBased>(kernels({"gridbased-1"}))},
                      {rowOf<Predicted>(kernels({"prediction"}, true))},
                      {rowOf<GridBased>(kernels({"gridbased-2"}))},
                      {rowOf<Equality>(kernels({"equality"}))}});

    auto const [off, on] = bothOrders(*lib, 8);
    EXPECT_EQ(off, (Names{"origami", "gridbased-1", "prediction", "gridbased-2", "equality"}));
    EXPECT_EQ(on, (Names{"origami", "prediction", "gridbased-1", "gridbased-2", "equality"}));

    // A model whose row the search reaches before any GridBased row is in place.
    lib = files({{rowOf<Predicted>(kernels({"prediction"}, true))},
                 {rowOf<GridBased>(kernels({"gridbased"}))}});

    auto const [off2, on2] = bothOrders(*lib, 4);
    EXPECT_EQ(on2, off2);
    EXPECT_EQ(on2, (Names{"prediction", "gridbased"}));

    // The search looks ahead to the next Prediction row only.
    lib = files({{rowOf<GridBased>(kernels({"gridbased"}))},
                 {rowOf<Predicted>(kernels({"origami"}))},
                 {rowOf<Predicted>(kernels({"prediction"}, true))}});

    auto const [off3, on3] = bothOrders(*lib, 4);
    EXPECT_EQ(on3, off3);
}

TEST(ExactLogicLibraryTest, RowsHoldingKernelsOfOtherKindsKeepTheRowOrder)
{
    // A row of no selection kind that holds kernels before the GridBased row
    // is searched once, in its place.
    auto other = kernels({"other"});
    auto lib   = files({{rowOf<AnyRow>(other)},
                        {rowOf<GridBased>(kernels({"gridbased"}))},
                        {rowOf<Predicted>(kernels({"prediction"}, true))}});
    EXPECT_EQ(bothOrders(*lib, 4).second, (Names{"other", "prediction", "gridbased"}));
    EXPECT_EQ(other->calls(), 2);

    // One between the GridBased row and the model's cannot wait for it: the
    // rows keep their order, and each is searched once more.
    auto between = kernels({"other"});
    auto grid    = kernels({"gridbased"});
    auto table   = kernels({"prediction"}, true);

    lib = files({{rowOf<GridBased>(grid)}, {rowOf<AnyRow>(between)}, {rowOf<Predicted>(table)}});

    auto const [off, on] = bothOrders(*lib, 4);
    EXPECT_EQ(on, off);
    EXPECT_EQ(on, (Names{"gridbased", "other", "prediction"}));
    EXPECT_EQ(grid->calls(), 2);
    EXPECT_EQ(table->calls(), 2);
    EXPECT_EQ(between->calls(), 3);
}

TEST(ExactLogicLibraryTest, PredictionLibWithTilewrightReturnsOnlyPrediction)
{
    ScopedSetting predictionLib("TENSILE_PREDICTION_LIB", "1");

    auto table = kernels({"prediction"}, true);
    auto lib   = files({{rowOf<Equality>(kernels({"equality"}))},
                        {rowOf<GridBased>(kernels({"gridbased"}))},
                        {rowOf<Predicted>(table)}});

    auto const [off, on] = bothOrders(*lib, 3);
    EXPECT_EQ(off, (Names{"prediction"}));
    EXPECT_EQ(on, (Names{"prediction"}));
    // Prediction rows alone answer, in their order: no order is made.
    EXPECT_EQ(table->calls(), 2);
    EXPECT_EQ(table->ordered(), 0);
}

TEST(ExactLogicLibraryTest, AnOrderOfManyRowsSearchesEachRowOnce)
{
    // More rows than an order keeps in place: files of an Equality row
    // without the problem's size before the GridBased row and the model's,
    // and files of Equality rows that answer after them.
    std::vector<std::shared_ptr<KernelRow>> before, after;
    std::vector<Rows>                       rows;
    for(int i = 0; i < 10; i++)
    {
        before.push_back(kernels({}));
        rows.push_back({rowOf<Equality>(before.back())});
    }
    auto grid  = kernels({"gridbased"});
    auto table = kernels({"prediction-1", "prediction-2"}, true);
    rows.push_back({rowOf<GridBased>(grid)});
    rows.push_back({rowOf<Predicted>(table)});
    Names tail;
    for(int i = 0; i < 10; i++)
    {
        tail.push_back("equality-" + std::to_string(i));
        after.push_back(kernels({tail.back()}));
        rows.push_back({rowOf<Equality>(after.back())});
    }
    auto lib = files(rows);

    auto const [off, on] = bothOrders(*lib, 32);
    Names expectedOff{"gridbased", "prediction-1", "prediction-2"};
    Names expectedOn{"prediction-1", "prediction-2", "gridbased"};
    expectedOff.insert(expectedOff.end(), tail.begin(), tail.end());
    expectedOn.insert(expectedOn.end(), tail.begin(), tail.end());
    EXPECT_EQ(off, expectedOff);
    EXPECT_EQ(on, expectedOn);
    // Once with tilewright off, once in the order's walks.
    for(auto const& row : before)
    {
        EXPECT_EQ(row->calls(), 2);
        EXPECT_EQ(row->ordered(), 1);
    }
    for(auto const& row : after)
        EXPECT_EQ(row->calls(), 2);
    EXPECT_EQ(grid->calls(), 2);
    EXPECT_EQ(table->calls(), 2);
}
