// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <gtest/gtest.h>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <unistd.h>
#endif

#include "../../shared/rocfft_accuracy_test.h"

#include "../../shared/CLI11.hpp"
#include "../../shared/client_except.h"
#include "../../shared/fftw_transform.h"
#include "../../shared/gpubuf.h"
#include "../../shared/rocfft_against_fftw.h"
#include "../../shared/subprocess.h"
#include "rocfft/rocfft.h"

extern std::string mp_launch;
struct user_mp_launch_command
{
    std::string              exe;
    std::vector<std::string> user_mp_argv;
};

// Resolve a launcher given by bare name (e.g. "mpirun") against PATH, as a
// shell would.  This is deliberately done here rather than in
// execute_subprocess, which is also compiled into the rocFFT library and
// must not search PATH.  Returns an empty string if nothing is found.
static std::string resolve_mp_launch_exe(const std::string& exe)
{
#ifdef _WIN32
    // Not implemented: Windows lookup rules differ (';' separator, PATHEXT
    // extensions), so the name is passed through unchanged.  As
    // execute_subprocess's CreateProcessA call does not search PATH (a bare
    // name is only looked up in the current directory), the launcher should
    // be given by its path on Windows.
    return exe;
#else
    if(exe.empty() || exe.find('/') != std::string::npos)
        return exe;

    const char* path_env = std::getenv("PATH");
    if(path_env == nullptr)
        return {};

    std::string_view path(path_env);
    while(true)
    {
        const auto             sep = path.find(':');
        const std::string_view dir = path.substr(0, sep);
        // an empty PATH entry denotes the current directory
        const auto candidate = std::filesystem::path(dir.empty() ? "." : dir) / exe;

        std::error_code ec;
        if(std::filesystem::is_regular_file(candidate, ec) && access(candidate.c_str(), X_OK) == 0)
            return candidate.string();

        if(sep == std::string_view::npos)
            break;
        path.remove_prefix(sep + 1);
    }
    return {};
#endif
}

static user_mp_launch_command get_mp_launch_command()
{
    user_mp_launch_command ret;
    try
    {
        CLI::App command_splitter;
        command_splitter.allow_extras();
        command_splitter.parse(mp_launch, /*program_name_included = */ true);
        ret.exe          = resolve_mp_launch_exe(command_splitter.get_name());
        ret.user_mp_argv = command_splitter.remaining();
        if(ret.exe.empty() && !command_splitter.get_name().empty())
            std::cerr << "multi-process launcher \"" << command_splitter.get_name()
                      << "\" not found on PATH" << std::endl;
    }
    catch(const CLI::Error& e)
    {
        if(verbose)
        {
            if(!mp_launch.empty())
                std::cout << "CLI11 failed to parse the command " << mp_launch << "\n";
            std::cout << "CLI11 exception name: " << e.get_name() << "\n"
                      << "CLI11 exception info: " << e.what() << "\n"
                      << "CLI11 error code:" << e.get_exit_code() << std::endl;
        }
        // returned empty struct
        ret.exe.clear();
        ret.user_mp_argv.clear();
    }
    return ret;
}

void fft_vs_reference(rocfft_params& params, bool round_trip)
{
    switch(params.precision)
    {
    case fft_precision_half:
        fft_vs_reference_impl<rocfft_fp16, rocfft_params>(params, round_trip);
        break;
    case fft_precision_single:
        fft_vs_reference_impl<float, rocfft_params>(params, round_trip);
        break;
    case fft_precision_double:
        fft_vs_reference_impl<double, rocfft_params>(params, round_trip);
        break;
    }
}

// Test for comparison between FFTW and rocFFT.
TEST_P(accuracy_test, vs_fftw)
{
    rocfft_params params(GetParam());

    params.validate();

    // Test that the tokenization works as expected.
    auto       testcase_token = params.token();
    fft_params tokentest;
    tokentest.from_token(testcase_token);
    auto testcase_token1 = tokentest.token();
    EXPECT_EQ(testcase_token, testcase_token1);

    if(!params.valid(verbose))
    {
        GTEST_FAIL() << "Invalid parameters";
    }

    switch(params.mp_lib)
    {
    case fft_params::fft_mp_lib_none:
    {
        // Single-proc FFT.
        // Only do round trip for non-field FFTs
        bool round_trip = params.is_forward() && params.ifields.empty() && params.ofields.empty();

        try
        {
            fft_vs_reference(params, round_trip);
        }
        ROCFFT_CATCH_TEST_EXCEPTIONS;
        break;
    }
    case fft_params::fft_mp_lib_mpi:
    {
        // Multi-proc FFT.
        static const auto mp_launch_command = get_mp_launch_command();

        if(mp_launch_command.exe.empty())
            GTEST_FAIL() << "Cannot proceed due to empty multi-process executable: omitted "
                            "\"--mp_launch\" option or invalid value thereof.";
        auto argv = mp_launch_command.user_mp_argv;

        // append test token and ask for accuracy test
        argv.push_back("--token");
        argv.push_back(testcase_token);
        argv.push_back("--accuracy");

        // throws an exception if launch fails or if subprocess
        // returns nonzero exit code
        execute_subprocess(mp_launch_command.exe, argv, {});
        break;
    }
    default:
        GTEST_FAIL() << "Invalid communicator choice!";
        break;
    }

    SUCCEED();
}
