// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "miopendriver_common.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

#include <miopen/miopen.h>
#include <miopen/process.hpp>

namespace miopen_conv_deterministic {

// clang-format off
static const std::string shape_3d =
    "-n 1 -c 1 --in_d 10 -H 11 -W 12 -k 1 --fil_d 1 -y 2 -x 5 --pad_d 0 -p 1 -q 4 --conv_stride_d 1 -u 1 -v 1 --dilation_d 2 -l 2 -j 2 --spatial_dim 3 "
    "-m conv -g 1 --iter 1 -V 0";
// clang-format on

struct ConvDeterministicTestCase
{
    std::string base_args;    // args without --deterministic flag
    std::string valid_args;   // args with --deterministic 1
    std::string invalid_args; // args with --deterministic 2 (invalid value)
};

std::vector<ConvDeterministicTestCase>
GetTestCases(const std::string& mode, const std::string& shape, int forw_flag)
{
    const std::string dir  = " -F " + std::to_string(forw_flag);
    const std::string base = mode + " " + shape + dir;
    return {{base, base + " --deterministic 1", base + " --deterministic 2"}};
}

miopen::ProcessEnvironmentMap MakeEnv(const miopen::fs::path& tmp_dir)
{
    miopen::ProcessEnvironmentMap envs;
    envs["MIOPEN_FIND_MODE"]    = "2"; // immediate mode - uses AI heuristics
    envs["MIOPEN_LOG_LEVEL"]    = "5"; // Info level to capture messages
    envs["MIOPEN_USER_DB_PATH"] = tmp_dir.string();
    return envs;
}

// Skip when MIOPEN_BUILD_DRIVER=OFF or the GPU is unsupported.
static void CheckShouldRun()
{
    using e_mask = enabled<Gpu::All>;
    using d_mask = disabled<Gpu::None>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>())
        GTEST_SKIP();
}

// Return a per-test tmp path to avoid collisions when tests run in parallel.
static miopen::fs::path TmpDir(const std::string& suffix)
{
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();
    return miopen::fs::temp_directory_path() /
           (std::string{"miopen_det_"} + info->name() + "_" + suffix);
}

} // namespace miopen_conv_deterministic

class GPU_MIOpenDriverConvDeterministicTest_FP32
    : public testing::TestWithParam<miopen_conv_deterministic::ConvDeterministicTestCase>
{
};

// ----------------------------------------------------------------------------
// Test 1: Without --deterministic being set, log must not mention
// "Restricting convolution to deterministic kernels."
// ----------------------------------------------------------------------------
TEST_P(GPU_MIOpenDriverConvDeterministicTest_FP32, NoDeterministicLog)
{
    miopen_conv_deterministic::CheckShouldRun();
    const auto tmp_dir = miopen_conv_deterministic::TmpDir("db");
    miopen::fs::remove_all(tmp_dir);

    // MIOpenDriver runs as a subprocess; its stderr is redirected into ss via
    // popen(... + " 2>&1"). CaptureStderr() only hooks the test process's own
    // fd 2 and would always return "". Read ss.str() directly instead.
    miopen::Process p{MIOpenDriverExePath().string()};
    std::stringstream ss;
    int rc = 0;
    EXPECT_NO_THROW(
        rc = p(GetParam().base_args, "", &ss, miopen_conv_deterministic::MakeEnv(tmp_dir)));
    EXPECT_EQ(rc, 0);
    EXPECT_THAT(ss.str(),
                Not(testing::HasSubstr("Restricting convolution to deterministic kernels.")));

    miopen::fs::remove_all(tmp_dir);
}

// ----------------------------------------------------------------------------
// Test 2: With --deterministic 1 the log mentions
// "Restricting convolution to deterministic kernels."
// ----------------------------------------------------------------------------
TEST_P(GPU_MIOpenDriverConvDeterministicTest_FP32, RunsSuccessfullyAndLogsOverride)
{
    miopen_conv_deterministic::CheckShouldRun();
    const auto tmp_dir = miopen_conv_deterministic::TmpDir("db");
    miopen::fs::remove_all(tmp_dir);

    miopen::Process p{MIOpenDriverExePath().string()};
    std::stringstream ss;
    int rc = 0;
    EXPECT_NO_THROW(
        rc = p(GetParam().valid_args, "", &ss, miopen_conv_deterministic::MakeEnv(tmp_dir)));
    EXPECT_EQ(rc, 0);
    EXPECT_THAT(ss.str(), testing::HasSubstr("Restricting convolution to deterministic kernels."));

    miopen::fs::remove_all(tmp_dir);
}

// ----------------------------------------------------------------------------
// Test 3: Invalid --deterministic value causes non-zero exit
// ----------------------------------------------------------------------------
TEST_P(GPU_MIOpenDriverConvDeterministicTest_FP32, ExitsOnInvalidValue)
{
    miopen_conv_deterministic::CheckShouldRun();
    const auto tmp_dir = miopen_conv_deterministic::TmpDir("db");
    miopen::fs::remove_all(tmp_dir);

    int result = 0;
    miopen::Process p{MIOpenDriverExePath().string()};
    std::stringstream ss;
    EXPECT_NO_THROW(
        result = p(GetParam().invalid_args, "", &ss, miopen_conv_deterministic::MakeEnv(tmp_dir)));
    EXPECT_NE(result, 0) << "Should exit with a non-zero code on invalid deterministic value";
    EXPECT_THAT(ss.str(), testing::HasSubstr("Invalid deterministic value"));

    miopen::fs::remove_all(tmp_dir);
}

// ----------------------------------------------------------------------------
// Test 4: Reproducibility - two runs with --deterministic 1 produce
//         identical output files (--dump_output)
// ----------------------------------------------------------------------------
TEST_P(GPU_MIOpenDriverConvDeterministicTest_FP32, BitExactAcrossRuns)
{
    miopen_conv_deterministic::CheckShouldRun();
    const auto run1_dir = miopen_conv_deterministic::TmpDir("run1");
    const auto run2_dir = miopen_conv_deterministic::TmpDir("run2");
    miopen::fs::remove_all(run1_dir);
    miopen::fs::remove_all(run2_dir);
    miopen::fs::create_directories(run1_dir);
    miopen::fs::create_directories(run2_dir);

    // RAII guard: clean up even when an ASSERT_ aborts the test body.
    struct Cleanup
    {
        const miopen::fs::path& d1;
        const miopen::fs::path& d2;
        ~Cleanup()
        {
            miopen::fs::remove_all(d1);
            miopen::fs::remove_all(d2);
        }
    } cleanup{run1_dir, run2_dir};

    auto make_env = [](const miopen::fs::path& tmp) {
        miopen::ProcessEnvironmentMap envs;
        envs["MIOPEN_DEBUG_FIND_ONLY_SOLVER"] = "ConvHipImplicitGemm3DGroupWrwXdlops";
        envs["MIOPEN_LOG_LEVEL"]              = "5";
        envs["MIOPEN_USER_DB_PATH"]           = tmp.string();
        return envs;
    };

    // -o 1 (--dump_output) writes binary output files to the working directory.
    // Run the driver twice from separate directories to get separate output files.
    const std::string det_args = GetParam().valid_args + " -o 1";

    {
        auto envs = make_env(run1_dir / "db");
        miopen::fs::create_directories(run1_dir / "db");
        miopen::Process p{MIOpenDriverExePath().string()};
        std::stringstream ss;
        int rc = 0;
        // Change CWD so dump files land in run1_dir
        EXPECT_NO_THROW(rc = p(det_args, run1_dir, &ss, envs));
        ASSERT_EQ(rc, 0) << "First driver run failed:\n" << ss.str();
    }
    {
        auto envs = make_env(run2_dir / "db");
        miopen::fs::create_directories(run2_dir / "db");
        miopen::Process p{MIOpenDriverExePath().string()};
        std::stringstream ss;
        int rc = 0;
        EXPECT_NO_THROW(rc = p(det_args, run2_dir, &ss, envs));
        ASSERT_EQ(rc, 0) << "Second driver run failed:\n" << ss.str();
    }

    // Compare all dump_*.bin files between the two runs.
    int compared = 0;
    for(const auto& entry : miopen::fs::directory_iterator(run1_dir))
    {
        const auto& p1 = entry.path();
        if(p1.extension() != ".bin")
            continue;

        const auto p2 = run2_dir / p1.filename();
        ASSERT_TRUE(miopen::fs::exists(p2))
            << "Output file missing from second run: " << p2.string();

        const auto sz1 = miopen::fs::file_size(p1);
        const auto sz2 = miopen::fs::file_size(p2);
        ASSERT_EQ(sz1, sz2) << "Output file size mismatch: " << p1.filename().string();

        std::ifstream f1(p1, std::ios::binary);
        std::ifstream f2(p2, std::ios::binary);
        ASSERT_TRUE(f1.is_open() && f2.is_open());

        std::vector<char> buf1(sz1), buf2(sz2);
        f1.read(buf1.data(), static_cast<std::streamsize>(sz1));
        f2.read(buf2.data(), static_cast<std::streamsize>(sz2));
        EXPECT_EQ(buf1, buf2) << "Bit-exact mismatch in " << p1.filename().string();
        ++compared;
    }

    ASSERT_GT(compared, 0) << "No .bin output files found — nothing was compared";
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_MIOpenDriverConvDeterministicTest_FP32,
    testing::ValuesIn(miopen_conv_deterministic::GetTestCases(miopendriver::basearg::conv::Float,
                                                              miopen_conv_deterministic::shape_3d,
                                                              4)));