// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/hostbuf.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_complex.h"
#include "../../shared/rocfft_params.h"

#include "../../shared/accuracy_test.h"
#include "../../shared/fftw_transform.h"
#include "../../shared/rocfft_against_fftw.h"
#include <gtest/gtest.h>

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(change_type);

// callback functions to cast data from short to float
__host__ __device__ float
    load_callback_short(short* input, size_t offset, void* cbdata, void* sharedMem)
{
    return static_cast<float>(input[offset]);
}

__host__ __device__ float2 load_callback_short2(short2* input,
                                                size_t  offset,
                                                void*   cbdata,
                                                void*   sharedMem)
{
    return float2{static_cast<float>(input[offset].x), static_cast<float>(input[offset].y)};
}

__device__ auto load_callback_short_dev  = load_callback_short;
__device__ auto load_callback_short2_dev = load_callback_short2;

class change_type : public ::testing::TestWithParam<fft_params>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<change_type::ParamType>& info)
    {
        return info.param.token();
    }
};

// aim for 1D lengths that might need ordinary Stockham, transpose,
// Bluestein kernels to treat real data as complex
std::vector<std::vector<size_t>> callback_type_sizes = {{4}, {60}, {122}, {220}, {8192}, {4500000}};

// test complex + real forward transforms.  real inverse is not a valid
// test case here, because we're allowed to overwrite input on those.
// the input can't be any smaller than what rocFFT thinks it is,
// because the overwrite will fail.
const static std::vector<std::vector<size_t>> stride_range = {{1}};
INSTANTIATE_TEST_SUITE_P(
#ifdef WIN32
    DISABLED_callback,
#else
    callback,
#endif
    change_type,
    ::testing::ValuesIn(param_generator_base(test_prob,
                                             {fft_transform_type_complex_forward,
                                              fft_transform_type_real_forward},
                                             callback_type_sizes,
                                             {fft_precision_single},
                                             {1},
                                             generate_types,
                                             stride_range,
                                             stride_range,
                                             {{0, 0}},
                                             {{0, 0}},
                                             {fft_placement_notinplace},
                                             false,
                                             false)),
    accuracy_test::TestName);

// run an out-of-place transform that casts input from short to float
TEST_P(change_type, short_to_float)
{
    rocfft_params params(GetParam());
    params.run_callbacks = true;

    ASSERT_EQ(params.create_plan(), fft_status_success);

    // input has 2 shorts/floats for complex data, 1 otherwise.
    // output is always complex for these tests.
    const size_t input_complex = params.transform_type != fft_transform_type_real_forward ? 2 : 1;

    // allocate
    gpubuf               gpu_input;
    gpubuf               gpu_output;
    std::vector<hostbuf> cpu_input(1);
    std::vector<hostbuf> cpu_output(1);

    try
    {
        // gpu input is actually shorts, everything else is float
        ASSERT_EQ(gpu_input.alloc(params.isize[0] * sizeof(short) * input_complex), hipSuccess);
        ASSERT_EQ(gpu_output.alloc(params.osize[0] * sizeof(float) * 2), hipSuccess);
        cpu_input[0].alloc(params.isize[0] * sizeof(float) * input_complex);
        cpu_output[0].alloc(params.osize[0] * sizeof(float) * 2);

        // generate short (16-bit) and float (32-bit) input
        std::mt19937                         gen;
        std::uniform_int_distribution<short> dis(-3, 3);
        std::vector<short>                   cpu_input_short(params.isize[0] * input_complex);
        for(auto& i : cpu_input_short)
            i = dis(gen);

        // copy short input to gpubuf
        ASSERT_EQ(hipMemcpy(gpu_input.data(),
                            cpu_input_short.data(),
                            sizeof(short) * cpu_input_short.size(),
                            hipMemcpyHostToDevice),
                  hipSuccess);

        // convert shorts to floats for FFTW input
        std::copy(cpu_input_short.begin(),
                  cpu_input_short.end(),
                  static_cast<float*>(cpu_input[0].data()));

        // get callback function so we can pass it to rocfft
        void* callback_host;
        if(input_complex == 1)
        {
            ASSERT_EQ(hipMemcpyFromSymbol(
                          &callback_host, HIP_SYMBOL(load_callback_short_dev), sizeof(void*)),
                      hipSuccess);
        }
        else
        {
            ASSERT_EQ(hipMemcpyFromSymbol(
                          &callback_host, HIP_SYMBOL(load_callback_short2_dev), sizeof(void*)),
                      hipSuccess);
        }
        ASSERT_EQ(params.set_callbacks(callback_host, nullptr, nullptr, nullptr),
                  fft_status_success);

        // run rocFFT
        void* gpu_input_ptr  = gpu_input.data();
        void* gpu_output_ptr = gpu_output.data();
        ASSERT_EQ(params.execute(&gpu_input_ptr, &gpu_output_ptr), fft_status_success);

        // construct + run FFTW plan
        auto cpu_plan = fftw_plan_via_rocfft<float>(params.length,
                                                    params.istride,
                                                    params.ostride,
                                                    params.nbatch,
                                                    params.idist,
                                                    params.odist,
                                                    params.transform_type,
                                                    cpu_input,
                                                    cpu_output);
        fftw_run<float>(params.transform_type, cpu_plan, cpu_input, cpu_output);

        // copy rocFFT output back to CPU
        std::vector<hostbuf> gpu_output_copy(1);
        gpu_output_copy[0].alloc(gpu_output.size());
        ASSERT_EQ(hipMemcpy(gpu_output_copy[0].data(),
                            gpu_output.data(),
                            gpu_output.size(),
                            hipMemcpyDeviceToHost),
                  hipSuccess);

        auto cpu_output_norm = norm(cpu_output,
                                    params.olength(),
                                    params.nbatch,
                                    params.precision,
                                    params.otype,
                                    params.ostride,
                                    params.odist,
                                    params.ooffset);
        ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));
        ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));

        auto gpu_output_norm = norm(gpu_output_copy,
                                    params.olength(),
                                    params.nbatch,
                                    params.precision,
                                    params.otype,
                                    params.ostride,
                                    params.odist,
                                    params.ooffset);
        ASSERT_TRUE(std::isfinite(gpu_output_norm.l_2));
        ASSERT_TRUE(std::isfinite(gpu_output_norm.l_inf));

        double linf_cutoff
            = type_epsilon(params.precision) * cpu_output_norm.l_inf * log(params.length.front());
        auto diff = distance(cpu_output,
                             gpu_output_copy,
                             params.olength(),
                             params.nbatch,
                             params.precision,
                             params.otype,
                             params.ostride,
                             params.odist,
                             params.otype,
                             params.ostride,
                             params.odist,
                             nullptr,
                             linf_cutoff,
                             params.ioffset,
                             params.ooffset);

        ASSERT_TRUE(diff.l_inf <= linf_cutoff);
    }
    catch(std::bad_alloc&)
    {
        GTEST_SKIP() << "host memory allocation failure";
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        GTEST_SKIP() << e.msg;
    }
}
