// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_SQLITE
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/sqlite_db.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/tensor.hpp>
#endif

namespace {

#if MIOPEN_ENABLE_SQLITE
miopen::conv::ProblemDescription MakeProblem()
{
    const miopen::TensorDescriptor td{miopenFloat, {1, 1, 1, 1}};
    return {td, td, td, miopen::ConvolutionDescriptor{}, miopen::conv::Direction::Forward};
}

// MIOPEN_DEBUG_PERFDB_OVERRIDE is declared (MIOPEN_DECLARE_ENV_VAR_STR) in
// miopen/sqlite_db.hpp, so it is reused here directly instead of redeclaring it.
struct ScopedPerfDbOverride
{
    explicit ScopedPerfDbOverride(const std::string& value)
    {
        miopen::env::update(MIOPEN_DEBUG_PERFDB_OVERRIDE, value);
    }
    ~ScopedPerfDbOverride() { miopen::env::clear(MIOPEN_DEBUG_PERFDB_OVERRIDE); }
};
#endif

} // namespace

// MIOPEN_DEBUG_PERFDB_OVERRIDE makes FindRecordUnsafe() (miopen/sqlite_db.hpp) return an
// in-memory record built from the env var, short-circuiting the sqlite lookup entirely.
TEST(CPU_SQLitePerfDbOverride_NONE, PerfDbOverride)
{
#if MIOPEN_ENABLE_SQLITE
    ScopedPerfDbOverride override_env("SolverA;paramsA:SolverB;paramsB");

    miopen::TempFile temp_file{"sqlite_perfdb_override"};
    miopen::SQLitePerfDb pdb{miopen::DbKinds::PerfDb, temp_file.Path(), false};

    auto problem = MakeProblem();
    auto record  = pdb.FindRecord(problem);
    ASSERT_TRUE(record.has_value());

    std::string params;
    EXPECT_TRUE(record->GetValues("SolverA", params));
    EXPECT_EQ(params, "paramsA");
    EXPECT_TRUE(record->GetValues("SolverB", params));
    EXPECT_EQ(params, "paramsB");
#else
    GTEST_SKIP() << "Test requires MIOPEN_ENABLE_SQLITE to be enabled";
#endif
}
