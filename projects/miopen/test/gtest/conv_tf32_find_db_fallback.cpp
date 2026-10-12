// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A TF32 request without its own find-db record is ranked by the full-precision record (TF32 is
// a precision floor, so any full-precision solution satisfies it). Requests for full precision
// never look at another record.

#include <miopen/any_solver.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/db_record.hpp>
#include <miopen/find_db.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/tensor.hpp>

#include <gtest/gtest.h>

#include <fstream>

#include "get_handle.hpp"

namespace {

const auto kSolver = miopen::solver::Id{"ConvDirectNaiveConvFwd"};

miopen::conv::ProblemDescription MakeProblem(const miopen::ConvolutionDescriptor& conv,
                                             int channels)
{
    const auto x = miopen::TensorDescriptor{miopenFloat, {1, channels, 16, 16}};
    const auto w = miopen::TensorDescriptor{miopenFloat, {channels, channels, 3, 3}};
    const auto y = conv.GetForwardOutputTensor(x, w, miopenFloat);
    return {x, w, y, conv, miopen::conv::Direction::Forward};
}

miopen::ConvolutionDescriptor MakeConv(miopenMathType_t math_type)
{
    auto conv = miopen::ConvolutionDescriptor{{1, 1}, {1, 1}, {1, 1}};
    conv.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, math_type);
    return conv;
}

class ScopedFindDbPath
{
public:
    ScopedFindDbPath() : previous(miopen::debug::testing_find_db_path_override())
    {
        miopen::debug::testing_find_db_path_override() = file.Path();
        previous_rordb_embed                           = miopen::debug::rordb_embed_fs_override();
        miopen::debug::rordb_embed_fs_override()       = true;
    }
    ~ScopedFindDbPath()
    {
        miopen::debug::testing_find_db_path_override() = previous;
        miopen::debug::rordb_embed_fs_override()       = previous_rordb_embed;
    }
    miopen::fs::path Path() const { return file.Path(); }
    ScopedFindDbPath(const ScopedFindDbPath&)            = delete;
    ScopedFindDbPath& operator=(const ScopedFindDbPath&) = delete;

private:
    miopen::TempFile file{"miopen.test.tf32_find_db"};
    std::optional<miopen::fs::path> previous;
    bool previous_rordb_embed = false;
};

/// Appends a find-db record for problem that names only kSolver.
void StoreRecord(const miopen::fs::path& db_path, const miopen::conv::ProblemDescription& problem)
{
    std::ofstream file{db_path, std::ios::app};
    file << miopen::DbRecord(miopen::DbKinds::FindDb, problem).GetKey() << '=' << kSolver.ToString()
         << ":1,0,miopenConvolutionFwdAlgoDirect\n";
}

/// gtest cannot print a FallbackPath.
int AsInt(miopen::FallbackPath path) { return static_cast<int>(path); }

} // namespace

TEST(GPU_ConvTf32FindDbFallback_FP32, Tf32RequestUsesStandInRecord)
{
    auto&& handle  = get_handle();
    const auto ctx = miopen::ExecutionContext{&handle};
    const ScopedFindDbPath find_db_path;

    const auto fp32_conv = MakeConv(miopenMathPedantic);
    const auto tf32_conv = MakeConv(miopenMathDefault);
    const auto fp32      = MakeProblem(fp32_conv, 8);
    const auto tf32      = MakeProblem(tf32_conv, 8);
    StoreRecord(find_db_path.Path(), fp32);
    ASSERT_NE(miopen::DbRecord(miopen::DbKinds::FindDb, tf32).GetKey(),
              miopen::DbRecord(miopen::DbKinds::FindDb, fp32).GetKey());

    auto fallback        = miopen::FallbackPath::None;
    const auto fp32_sols = fp32_conv.GetSolutions(ctx, fp32, 4, &fallback);
    ASSERT_EQ(fp32_sols.size(), 1u);
    EXPECT_EQ(fp32_sols.front().solution_id, kSolver.Value());
    EXPECT_EQ(AsInt(fallback), AsInt(miopen::FallbackPath::None));

    const auto tf32_sols = tf32_conv.GetSolutions(ctx, tf32, 4, &fallback);
    ASSERT_EQ(tf32_sols.size(), 1u);
    EXPECT_EQ(tf32_sols.front().solution_id, kSolver.Value());
    EXPECT_LT(tf32_sols.front().time, 0.0f) << "a stand-in's time is an estimate for this problem";
    EXPECT_EQ(tf32_sols.front().workspace_size, kSolver.GetSolver().GetWorkspaceSize(ctx, tf32));
    EXPECT_EQ(AsInt(fallback), AsInt(miopen::FallbackPath::StandInRecord));
    EXPECT_EQ(tf32_conv.GetSolutionCount(ctx, tf32), 1u);
}

TEST(GPU_ConvTf32FindDbFallback_FP32, FullPrecisionRequestIgnoresOtherRecords)
{
    auto&& handle  = get_handle();
    const auto ctx = miopen::ExecutionContext{&handle};
    const ScopedFindDbPath find_db_path;

    // Only a TF32 record exists. A full-precision request must not use it.
    const auto fp32_conv = MakeConv(miopenMathPedantic);
    const auto tf32_conv = MakeConv(miopenMathDefault);
    StoreRecord(find_db_path.Path(), MakeProblem(tf32_conv, 16));

    auto fallback = miopen::FallbackPath::None;
    std::ignore   = fp32_conv.GetSolutions(ctx, MakeProblem(fp32_conv, 16), 4, &fallback);
    EXPECT_NE(AsInt(fallback), AsInt(miopen::FallbackPath::None));
    EXPECT_NE(AsInt(fallback), AsInt(miopen::FallbackPath::StandInRecord));
}
