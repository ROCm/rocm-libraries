/* **************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <cstdlib>
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include <fstream>
#include <vector>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "common/misc/client_environment_helpers.hpp"
#include "common/misc/clientcommon.hpp"

using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::UnitTest;

class checkin_misc_LOGGING : public ::testing::Test
{
protected:
    void SetUp() override
    {
        fs::path temp_dir = fs::temp_directory_path();
        std::string test_name = UnitTest::GetInstance()->current_test_info()->name();
        log_filepath = temp_dir / fmt::format("{}.{}.log", test_name, nondeterministic_value());
        fs::path nonexistent_dirpath = temp_dir / fmt::format("nonexistent_{}", test_name);
        ASSERT_FALSE(fs::exists(nonexistent_dirpath));
        invalid_log_filepath = nonexistent_dirpath / "invalid.log";

        ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA * bc), hipSuccess);
        ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP * bc), hipSuccess);
        ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int) * bc), hipSuccess);
    }

    void TearDown() override
    {
        if(fs::exists(log_filepath))
        {
            if(HasFailure() && std::getenv("ROCSOLVER_TEST_DEBUG"))
                fmt::print(stderr, "ROCSOLVER_TEST_DEBUG is set so {} was not removed.\n",
                           log_filepath.string());
            else
                EXPECT_TRUE(fs::remove(log_filepath));
        }

        EXPECT_EQ(hipFree(dA), hipSuccess);
        EXPECT_EQ(hipFree(dP), hipSuccess);
        EXPECT_EQ(hipFree(dinfo), hipSuccess);
    }

    unsigned int nondeterministic_value()
    {
        return rd();
    }

    std::random_device rd;
    fs::path log_filepath;
    fs::path invalid_log_filepath;

    double* dA;
    rocblas_int *dP, *dinfo;

    const rocblas_int m = 25;
    const rocblas_int n = 25;
    const rocblas_int lda = m;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;
    const rocblas_int bc = 3;
};

static void verify_file(const fs::path& filepath, const std::vector<std::string>& expected_lines)
{
    std::ifstream logfile(filepath);
    ASSERT_TRUE(logfile.good()) << "which implies a failure to open " << filepath;
    std::string line;
    size_t line_number = 1;
    for(; std::getline(logfile, line); ++line_number)
    {
        ASSERT_LE(line_number, expected_lines.size())
            << "extra line containing '" << line << "' in " << filepath;
        const std::string& expected_pattern = expected_lines[line_number - 1];
        EXPECT_TRUE(Matcher<std::string>(MatchesRegex(expected_pattern)).Matches(line))
            << "Mismatch at line " << line_number << ":\n     Actual line: " << line
            << "\nExpected pattern: " << expected_pattern;
    }
    ASSERT_EQ(line_number - 1, expected_lines.size()) << "missing lines in " << filepath;
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_path)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PATH", log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    ASSERT_TRUE(fs::exists(log_filepath));
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_trace)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH", log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*ENTER rocsolver_dgetrf_strided_batched trace tree.*",
        ".*getrf.*m: 25, n: 25, shiftA: 0, inca: 1, lda: 25, shiftP: 0, bc: 3.*",
        ".*EXIT rocsolver_dgetrf_strided_batched trace tree.*",
        "\\s*",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_bench)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_BENCH_PATH", log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_bench), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*rocsolver-bench -f getrf_strided_batched -r d -m 25 -n 25 --lda 25 --strideA 625 "
        "--strideP 25 --batch_count 3",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_profile)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH",
                                   log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_write_profile)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH",
                                   log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    EXPECT_EQ(rocsolver_log_write_profile(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_write_profile(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
    };
    verify_file(log_filepath, expected_lines);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success); // reset global state for other tests
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_flush_profile)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH",
                                   log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    EXPECT_EQ(rocsolver_log_flush_profile(), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    EXPECT_EQ(rocsolver_log_flush_profile(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_flush_profile(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 2, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
    };
    verify_file(log_filepath, expected_lines);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success); // reset global state for other tests
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_restore_defaults_resets_layer_mode)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH", log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_restore_defaults(), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_restore_defaults_resets_max_levels)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH", log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(2), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_restore_defaults(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*ENTER rocsolver_dgetrf_strided_batched trace tree.*",
        ".*getrf.*m: 25, n: 25, shiftA: 0, inca: 1, lda: 25, shiftP: 0, bc: 3.*",
        ".*EXIT rocsolver_dgetrf_strided_batched trace tree.*",
        "\\s*",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_trace_tree)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH", log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(2), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*ENTER rocsolver_dgetrf_strided_batched trace tree.*",
        ".*getrf.*m: 25, n: 25, shiftA: 0, inca: 1, lda: 25, shiftP: 0, bc: 3.*",
        "    .*getf2.*m: 25, n: 25, shiftA: 0, inca: 1, lda: 25, shiftP: 0, bc: 3.*",
        ".*EXIT rocsolver_dgetrf_strided_batched trace tree.*",
        "\\s*",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_profile_tree)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH",
                                   log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(2), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        ".*getf2.*Calls: 1, Total Time: .+ .+",
        "\\s*",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_bench_tree)
{
    rocblas_local_handle handle;
    scoped_envvar logpath_variable("ROCSOLVER_LOG_BENCH_PATH", log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_bench), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_set_max_levels(2), rocblas_status_success);
    EXPECT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*rocsolver-bench -f getrf_strided_batched -r d -m 25 -n 25 --lda 25 --strideA 625 "
        "--strideP 25 --batch_count 3",
    };
    verify_file(log_filepath, expected_lines);
}

TEST_F(checkin_misc_LOGGING, invalid_trace_file_open)
{
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH",
                                   invalid_log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_internal_error);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);
}

TEST_F(checkin_misc_LOGGING, invalid_profile_file_open)
{
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH",
                                   invalid_log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_internal_error);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);
}

TEST_F(checkin_misc_LOGGING, invalid_bench_file_open)
{
    scoped_envvar logpath_variable("ROCSOLVER_LOG_BENCH_PATH",
                                   invalid_log_filepath.generic_string().c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_internal_error);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);
}

TEST_F(checkin_misc_LOGGING, begin_twice)
{
    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_begin(), rocblas_status_internal_error);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);
}

TEST_F(checkin_misc_LOGGING, end_twice)
{
    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    EXPECT_EQ(rocsolver_log_end(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_internal_error);
}

TEST_F(checkin_misc_LOGGING, end_before_begin)
{
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_internal_error);
}
