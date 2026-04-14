// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hipfft/hipfft.h"

#include "../hipfft_params.h"

#include "../../shared/CLI11.hpp"
#include "../../shared/accuracy_test.h"
#include "../../shared/fftw_transform.h"
#include "../../shared/gpubuf.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_against_fftw.h"
#include "../../shared/rocfft_complex.h"
#include "../../shared/subprocess.h"

extern std::string mp_launch;
struct user_mp_launch_command
{
    std::string              exe;
    std::vector<std::string> user_mp_argv;
};
static user_mp_launch_command get_mp_launch_command()
{
    user_mp_launch_command ret;
    try
    {
        CLI::App command_splitter;
        command_splitter.allow_extras();
        command_splitter.parse(mp_launch, /*program_name_included = */ true);
        ret.exe          = command_splitter.get_name();
        ret.user_mp_argv = command_splitter.remaining();
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

extern last_cpu_fft_cache last_cpu_fft_data;

// clang-format off
// tokens of tests found to be symptomatic
static const std::vector<std::string> symptomatic_tokens = {
#ifndef _CUFFT_BACKEND
// cases specific to ROCM backend
#else
    // cases specific to CUFFT backend
    "real_forward_len_16384_half_ip_batch_4_istride_1_R_ostride_1_HI_idist_16386_odist_8193_ioffset_0_0_ooffset_0_0",
    "real_forward_len_32768_half_ip_batch_4_istride_1_R_ostride_1_HI_idist_32770_odist_16385_ioffset_0_0_ooffset_0_0",
    "real_forward_len_65536_half_ip_batch_2_istride_1_R_ostride_1_HI_idist_65538_odist_32769_ioffset_0_0_ooffset_0_0",
    "real_forward_len_65536_half_ip_batch_4_istride_1_R_ostride_1_HI_idist_65538_odist_32769_ioffset_0_0_ooffset_0_0",
    "real_forward_len_16384_half_ip_batch_2_istride_1_R_ostride_1_HI_idist_16386_odist_8193_ioffset_0_0_ooffset_0_0",
#endif
    // common  to both backends
};
// clang-format on

void fft_vs_reference(hipfft_params& params, bool round_trip)
{
    switch(params.precision)
    {
    case fft_precision_half:
        fft_vs_reference_impl<rocfft_fp16, hipfft_params>(params, round_trip);
        break;
    case fft_precision_single:
        fft_vs_reference_impl<float, hipfft_params>(params, round_trip);
        break;
    case fft_precision_double:
        fft_vs_reference_impl<double, hipfft_params>(params, round_trip);
        break;
    }
}

// Test for comparison between FFTW and hipFFT.
TEST_P(accuracy_test, vs_fftw)
{
    hipfft_params params(GetParam());

    params.validate();

    if(!params.valid(verbose))
    {
        if(verbose)
        {
            std::cout << "Invalid parameters, skip this test." << std::endl;
        }
        GTEST_SKIP();
    }

    switch(params.mp_lib)
    {
    case fft_params::fft_mp_lib_none:
    {
        // skipping symptomatic case(s), unless forcefully/knowingly executing normally-disabled
        // test tokens (e.g., by using --gtest_also_run_disabled_tests)
        const char* test_suite_name
            = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
        if(std::strstr(test_suite_name, "DISABLED") == nullptr
           && std::find(symptomatic_tokens.begin(), symptomatic_tokens.end(), params.token())
                  != symptomatic_tokens.end())
        {
            GTEST_SKIP()
                << "Symptomatic test that's currently disabled by default (force-skipping). Use "
                   "CLI arguments '--gtest_also_run_disabled_tests' to force the test execution "
                   "(via another test suite).";
        }
        // only do round trip for forward FFTs
        const bool do_round_trip = params.is_forward();

        try
        {
            fft_vs_reference(params, do_round_trip);
        }
        catch(const std::bad_alloc&)
        {
            // explicitly clear cache
            last_cpu_fft_data = last_cpu_fft_cache();
            GTEST_SKIP() << "host memory allocation failure";
        }
        catch(const HOSTBUF_MEM_USAGE& e)
        {
            // explicitly clear cache
            last_cpu_fft_data = last_cpu_fft_cache();
            GTEST_SKIP() << e.what();
        }
        catch(const DEVICEBUF_MEM_USAGE& e)
        {
            GTEST_SKIP() << e.what();
        }
        catch(const ROCFFT_SKIP& e)
        {
            GTEST_SKIP() << e.what();
        }
        catch(const fft_params::unimplemented_exception& e)
        {
            GTEST_SKIP() << "Unimplemented exception: " << e.what();
        }
        catch(const ROCFFT_FAIL& e)
        {
            GTEST_FAIL() << e.what();
        }
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
        argv.push_back(params.token());
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

INSTANTIATE_TEST_SUITE_P(DISABLED_symptomatic_tokens,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_token(test_prob, symptomatic_tokens)),
                         accuracy_test::TestName);
