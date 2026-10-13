// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A solver's perf-db record is looked up as its own, then under its alternate db id, then -- for a
// TF32 request -- the full-precision record, except for solvers whose performance config chooses
// the precision (the CK grouped solvers).

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_record.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/tensor.hpp>

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

namespace {

using miopen::solver::PerfDbRecord;

/// Holds the records named in `present` ("<perf-db key>:<solver db id>") and logs each lookup.
struct RecordingDb
{
    std::set<std::string> present;
    std::vector<std::string> looked_up;

    template <class Problem, class Config>
    bool Load(const Problem& problem, const std::string& id, Config&)
    {
        looked_up.push_back(Name(problem, id));
        return present.count(looked_up.back()) != 0;
    }

    static std::string Name(const miopen::conv::ProblemDescription& problem, const std::string& id)
    {
        return miopen::DbRecord(miopen::DbKinds::PerfDb, problem).GetKey() + ":" + id;
    }
};

struct FakeSolver
{
    bool selects_precision = false;
    std::string alt_id;

    bool PerfConfigSelectsPrecision(const miopen::ExecutionContext&,
                                    const miopen::conv::ProblemDescription&) const
    {
        return selects_precision;
    }
    const std::string& SolverDbId() const
    {
        static const std::string id = "FakeSolver";
        return id;
    }
    const std::string& AltSolverDbId() const { return alt_id; }
};

miopen::conv::ProblemDescription MakeProblem(miopenMathType_t math_type)
{
    auto conv = miopen::ConvolutionDescriptor{{1, 1}, {1, 1}, {1, 1}};
    conv.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, math_type);
    const auto x = miopen::TensorDescriptor{miopenFloat, {1, 8, 16, 16}};
    const auto w = miopen::TensorDescriptor{miopenFloat, {8, 8, 3, 3}};
    const auto y = conv.GetForwardOutputTensor(x, w, miopenFloat);
    return {x, w, y, conv, miopen::conv::Direction::Forward};
}

PerfDbRecord
Load(const FakeSolver& solver, const miopen::conv::ProblemDescription& problem, RecordingDb& db)
{
    int config     = 0;
    const auto ctx = miopen::ExecutionContext{};
    return miopen::solver::LoadPerfDbRecord(solver, ctx, problem, db, config);
}

const miopen::conv::ProblemDescription& Tf32()
{
    static const auto problem = MakeProblem(miopenMathDefault);
    return problem;
}

const miopen::conv::ProblemDescription& Fp32()
{
    static const auto problem = MakeProblem(miopenMathPedantic);
    return problem;
}

} // namespace

TEST(CPU_ConvTf32PerfDbFallback_NONE, Tf32RequestWithoutOwnRecordLoadsStandIn)
{
    RecordingDb db{{RecordingDb::Name(Fp32(), "FakeSolver")}, {}};
    EXPECT_EQ(Load(FakeSolver{}, Tf32(), db), PerfDbRecord::StandIn);
    EXPECT_EQ(db.looked_up,
              (std::vector<std::string>{RecordingDb::Name(Tf32(), "FakeSolver"),
                                        RecordingDb::Name(Fp32(), "FakeSolver")}));
}

TEST(CPU_ConvTf32PerfDbFallback_NONE, OwnRecordWins)
{
    RecordingDb db{
        {RecordingDb::Name(Tf32(), "FakeSolver"), RecordingDb::Name(Fp32(), "FakeSolver")}, {}};
    EXPECT_EQ(Load(FakeSolver{}, Tf32(), db), PerfDbRecord::Own);
    EXPECT_EQ(db.looked_up.size(), 1u);
}

TEST(CPU_ConvTf32PerfDbFallback_NONE, AlternateIdComesBeforeStandIn)
{
    RecordingDb db{{RecordingDb::Name(Tf32(), "OldFake"), RecordingDb::Name(Fp32(), "FakeSolver")},
                   {}};
    EXPECT_EQ(Load(FakeSolver{false, "OldFake"}, Tf32(), db), PerfDbRecord::Alternate);
    EXPECT_EQ(db.looked_up.size(), 2u);
}

TEST(CPU_ConvTf32PerfDbFallback_NONE, PrecisionSelectingSolverDoesNotFallBack)
{
    RecordingDb db{{RecordingDb::Name(Fp32(), "FakeSolver")}, {}};
    EXPECT_EQ(Load(FakeSolver{true, ""}, Tf32(), db), PerfDbRecord::None);
    EXPECT_EQ(db.looked_up, std::vector<std::string>{RecordingDb::Name(Tf32(), "FakeSolver")});
}

TEST(CPU_ConvTf32PerfDbFallback_NONE, FullPrecisionRequestHasNoStandIn)
{
    RecordingDb db;
    EXPECT_EQ(Load(FakeSolver{}, Fp32(), db), PerfDbRecord::None);
    EXPECT_EQ(db.looked_up, std::vector<std::string>{RecordingDb::Name(Fp32(), "FakeSolver")});
}
