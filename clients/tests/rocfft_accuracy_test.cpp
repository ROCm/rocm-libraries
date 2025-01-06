// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <boost/scope_exit.hpp>
#include <boost/tokenizer.hpp>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../shared/rocfft_accuracy_test.h"

#include "../../shared/client_except.h"
#include "../../shared/fftw_transform.h"
#include "../../shared/gpubuf.h"
#include "../../shared/rocfft_against_fftw.h"
#include "../../shared/subprocess.h"
#include "rocfft/rocfft.h"

extern std::string mp_launch;

extern last_cpu_fft_cache last_cpu_fft_data;

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
        bool round_trip = params.ifields.empty() && params.ofields.empty();

        try
        {
            fft_vs_reference(params, round_trip);
        }
        catch(std::bad_alloc&)
        {
            GTEST_SKIP() << "host memory allocation failure";
        }
        catch(HOSTBUF_MEM_USAGE& e)
        {
            // explicitly clear cache
            last_cpu_fft_data = last_cpu_fft_cache();
            GTEST_SKIP() << e.msg;
        }
        catch(ROCFFT_SKIP& e)
        {
            GTEST_SKIP() << e.msg;
        }
        catch(ROCFFT_FAIL& e)
        {
            GTEST_FAIL() << e.msg;
        }
        break;
    }
    case fft_params::fft_mp_lib_mpi:
    {
        // Multi-proc FFT.
        // Split launcher into tokens since the first one is the exe
        // and the remainder is the start of its argv
        boost::escaped_list_separator<char>                   sep('\\', ' ', '\"');
        boost::tokenizer<boost::escaped_list_separator<char>> tokenizer(mp_launch, sep);
        std::string                                           exe;
        std::vector<std::string>                              argv;
        for(auto t : tokenizer)
        {
            if(t.empty())
                continue;

            if(exe.empty())
                exe = t;
            else
                argv.push_back(t);
        }
        // append test token and ask for accuracy test
        argv.push_back("--token");
        argv.push_back(testcase_token);
        argv.push_back("--accuracy");

        // throws an exception if launch fails or if subprocess
        // returns nonzero exit code
        execute_subprocess(exe, argv, {});
        break;
    }
    default:
        GTEST_FAIL() << "Invalid communicator choice!";
        break;
    }

    SUCCEED();
}
