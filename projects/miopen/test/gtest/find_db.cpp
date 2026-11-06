// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "find_db.hpp"
#include "../lib_env_var.hpp"

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

MIOPEN_LIB_ENV_VAR(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME)
MIOPEN_LIB_ENV_VAR(MIOPEN_COMPILE_PARALLEL_LEVEL)

namespace {

void RunFindDbDriver(miopenDataType_t prec)
{
    switch(prec)
    {
    case miopenFloat: break;

    case miopenHalf:
    case miopenBFloat16:
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenHalf, miopenBFloat16, miopenInt8, miopenInt32, miopenDouble, "
                  "miopenFloat8_fnuz, miopenBFloat8_fnuz "
                  "data type not supported by "
                  "find_db test";
    }

    // Set up environment variables with automatic restoration
    ScopedEnvironment<int> logging_elapsed_time(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME, 1);
    ScopedEnvironment<int> log_level(MIOPEN_LOG_LEVEL, 6);
    ScopedEnvironment<int> compile_parallel_level(MIOPEN_COMPILE_PARALLEL_LEVEL, 1);

    // Create driver instance and run the test directly
    miopen::find_db_driver<float> driver;
    driver.run();
}

std::string ReadFile(const miopen::fs::path& path)
{
    std::ifstream file{path};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

// A solver that Find skips for lack of workspace is written to the find-db as a not-measured
// entry with the workspace it needs: the record is reused while the caller provides the same
// workspace, and Find runs again once enough is provided for that solver.
void RunSkippedForWorkspace()
{
    using miopen::FindDbData;
    auto& handle = get_handle();
    tensor<float> x{1, 8, 16, 16};
    tensor<float> w{8, 8, 3, 3};
    miopen::ConvolutionDescriptor conv{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    // Hybrid, as PyTorch uses (Dynamic Hybrid by default): a find-db record is reused through
    // GetSolutions(), and Normal Find runs only without a usable record.
    conv.findMode.Set(miopen::FindMode::Values::Hybrid);
    tensor<float> y{conv.GetForwardOutputTensor(x.desc, w.desc)};
    auto x_dev = handle.Write(x.data);
    auto w_dev = handle.Write(w.data);
    auto y_dev = handle.Write(y.data);

    const miopen::TempFile temp_file{"miopen.test.find_db_skipped"};
    miopen::debug::testing_find_db_path_override() = temp_file;
    miopen::TestRordbEmbedFsOverrideLock rordb_embed_fs_override;

    const auto find = [&](Workspace& ws) {
        int count;
        miopenConvAlgoPerf_t perf[1];
        conv.FindConvFwdAlgorithm(handle,
                                  x.desc,
                                  x_dev.get(),
                                  w.desc,
                                  w_dev.get(),
                                  y.desc,
                                  y_dev.get(),
                                  1,
                                  &count,
                                  perf,
                                  ws.ptr(),
                                  ws.size(),
                                  false);
    };

    // No workspace: GemmFwdRest (im2col) is skipped and recorded as not measured.
    Workspace no_ws{};
    find(no_ws);
    const auto record = ReadFile(temp_file.Path());
    if(record.find("GemmFwdRest:") == std::string::npos)
        GTEST_SKIP() << "GemmFwdRest is not applicable here";
    std::ostringstream not_measured;
    not_measured << "GemmFwdRest:" << FindDbData::NotMeasuredTime() << ',';
    EXPECT_NE(record.find(not_measured.str()), std::string::npos) << record;

    // Same workspace: the record is reused, not regenerated (a new Find would rewrite the times).
    find(no_ws);
    EXPECT_EQ(ReadFile(temp_file.Path()), record);

    // The workspace the not-measured entry records: Find runs again and measures GemmFwdRest.
    const auto ws_pos    = record.find(not_measured.str()) + not_measured.str().size();
    const auto needed_ws = std::stoull(record.substr(ws_pos, record.find(',', ws_pos) - ws_pos));
    EXPECT_GT(needed_ws, 0);
    Workspace full_ws{needed_ws};
    find(full_ws);
    const auto regenerated = ReadFile(temp_file.Path());
    EXPECT_NE(regenerated.find("GemmFwdRest:"), std::string::npos) << regenerated;
    EXPECT_EQ(regenerated.find(not_measured.str()), std::string::npos) << regenerated;

    // Normal Find runs every time; with no workspace again it keeps the GemmFwdRest measurement.
    conv.findMode.Set(miopen::FindMode::Values::Normal);
    find(no_ws);
    const auto normal = ReadFile(temp_file.Path());
    EXPECT_NE(normal.find("GemmFwdRest:"), std::string::npos) << normal;
    EXPECT_EQ(normal.find(not_measured.str()), std::string::npos) << normal;

    miopen::debug::testing_find_db_path_override() = std::nullopt;
}

} // namespace

class GPU_FindDbSkippedWorkspace_FP32 : public testing::Test
{
    void SetUp() override { prng::reset_seed(); }
};

TEST_F(GPU_FindDbSkippedWorkspace_FP32, NotMeasuredEntry) { RunSkippedForWorkspace(); }

class GPU_FindDb_FP32 : public testing::TestWithParam<miopenDataType_t>
{
    void SetUp() override { prng::reset_seed(); }
};

TEST_P(GPU_FindDb_FP32, FloatTest_find_db) { RunFindDbDriver(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_FindDb_FP32, testing::ValuesIn({miopenFloat}));
