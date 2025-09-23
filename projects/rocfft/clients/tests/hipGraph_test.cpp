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

#include "../../shared/accuracy_test.h"
#include "../../shared/arithmetic.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_against_fftw.h"
#include "../../shared/rocfft_params.h"
#include "rocfft/rocfft.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

static const unsigned int KERNEL_THREADS = 64;

__global__ void scale_data_kernel(rocfft_complex<float>* data, size_t length, float scale)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < length)
    {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

template <typename T>
__global__ void offset_data_kernel_complex(T* data, size_t length, T offset)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < length)
    {
        data[idx].x += offset.x;
        data[idx].y += offset.y;
    }
}

template <typename T>
__global__ void offset_data_kernel_real(T* data, size_t length, T offset)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < length)
    {
        data[idx] += offset;
    }
}

static void init_input_data(size_t                              N,
                            size_t                              seed,
                            std::vector<rocfft_complex<float>>& host_data,
                            gpubuf_t<rocfft_complex<float>>&    device_data)
{
    std::minstd_rand                      gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    host_data.resize(N);

    for(size_t i = 0; i < N; i++)
    {
        host_data[i].x = dist(gen);
        host_data[i].y = dist(gen);
    }

    size_t Nbytes = N * sizeof(rocfft_complex<float>);

    if(device_data.alloc(Nbytes) != hipSuccess)
        throw std::bad_alloc();

    EXPECT_EQ(hipMemcpy(device_data.data(), host_data.data(), Nbytes, hipMemcpyHostToDevice),
              hipSuccess);
}

template <typename T>
static void init_data(size_t N, T init_val, std::vector<T>& host_data, gpubuf_t<T>& device_data)
{
    host_data.resize(N);
    std::fill(host_data.begin(), host_data.end(), init_val);

    size_t Nbytes = N * sizeof(T);

    if(device_data.alloc(Nbytes) != hipSuccess)
        throw std::bad_alloc();

    EXPECT_EQ(hipMemcpy(device_data.data(), host_data.data(), Nbytes, hipMemcpyHostToDevice),
              hipSuccess);
}

static void create_forward_fft_plan(size_t N, rocfft_plan& plan)
{
    auto                dim = 1;
    std::vector<size_t> lengths(dim, N);

    ASSERT_EQ(rocfft_plan_create(&plan,
                                 rocfft_placement_notinplace,
                                 rocfft_transform_type_complex_forward,
                                 rocfft_precision_single,
                                 dim,
                                 lengths.data(),
                                 1,
                                 nullptr),
              rocfft_status_success);
}

static void create_inverse_fft_plan(size_t N, rocfft_plan& plan_inv)
{
    auto                dim = 1;
    std::vector<size_t> lengths(dim, N);

    ASSERT_EQ(rocfft_plan_create(&plan_inv,
                                 rocfft_placement_inplace,
                                 rocfft_transform_type_complex_inverse,
                                 rocfft_precision_single,
                                 dim,
                                 lengths.data(),
                                 1,
                                 nullptr),
              rocfft_status_success);
}

static void set_fft_info(hipStream_t stream, rocfft_execution_info& info)
{
    EXPECT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);
    EXPECT_EQ(rocfft_execution_info_set_stream(info, stream), rocfft_status_success);
}

static void
    run_forward_fft(rocfft_execution_info info, const rocfft_plan plan, void* in_ptr, void* out_ptr)
{
    ASSERT_EQ(rocfft_execute(plan, &in_ptr, &out_ptr, info), rocfft_status_success);
}

static void run_inverse_fft(rocfft_execution_info info,
                            const rocfft_plan     plan_inv,
                            void*                 in_ptr,
                            void*                 out_ptr)
{
    // Execute inverse plan in-place
    ASSERT_EQ(rocfft_execute(plan_inv, &in_ptr, &out_ptr, info), rocfft_status_success);
}

static void
    scale_device_data(hipStream_t stream, float scale, size_t N, rocfft_complex<float>* data)
{
    auto blockSize = KERNEL_THREADS;
    auto numBlocks = DivRoundingUp<size_t>(N, blockSize);
    hipLaunchKernelGGL(scale_data_kernel,
                       dim3(numBlocks),
                       dim3(blockSize),
                       0, // sharedMemBytes
                       stream, // stream
                       data,
                       N,
                       scale);
}

template <typename T>
static void offset_device_data_real(hipStream_t stream, T offset, size_t N, T* data)
{
    auto blockSize = KERNEL_THREADS;
    auto numBlocks = DivRoundingUp<size_t>(N, blockSize);
    hipLaunchKernelGGL(offset_data_kernel_real<T>,
                       dim3(numBlocks),
                       dim3(blockSize),
                       0, // sharedMemBytes
                       stream, // stream
                       data,
                       N,
                       offset);
}

template <typename T>
static void offset_device_data_complex(hipStream_t stream, T offset, size_t N, T* data)
{
    auto blockSize = KERNEL_THREADS;
    auto numBlocks = DivRoundingUp<size_t>(N, blockSize);
    hipLaunchKernelGGL(offset_data_kernel_complex<T>,
                       dim3(numBlocks),
                       dim3(blockSize),
                       0, // sharedMemBytes
                       stream, // stream
                       data,
                       N,
                       offset);
}

template <typename T>
static void compare_data_exact_match(hipStream_t           other_stream,
                                     const std::vector<T>& host_data,
                                     const gpubuf_t<T>&    device_data)
{

    std::vector<T> host_data_compare(host_data.size());

    // Copy result back to host
    ASSERT_EQ(hipMemcpyAsync(host_data_compare.data(),
                             device_data.data(),
                             host_data_compare.size() * sizeof(T),
                             hipMemcpyDeviceToHost,
                             other_stream),
              hipSuccess);

    ASSERT_EQ(hipStreamSynchronize(other_stream), hipSuccess);

    ASSERT_EQ(host_data == host_data_compare, true);
}

static void compare_data(const std::vector<rocfft_complex<float>>& original_host_data,
                         const gpubuf_t<rocfft_complex<float>>&    modified_device_data)
{
    std::vector<rocfft_complex<float>> modified_host_data(original_host_data.size());

    // Copy result back to host
    ASSERT_EQ(hipMemcpy(modified_host_data.data(),
                        modified_device_data.data(),
                        modified_host_data.size() * sizeof(rocfft_complex<float>),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    // Compare data we got to the original.
    // We're running 2 transforms (forward+inverse), so we
    // should tolerate 2x the error of a single transform.
    const double MAX_TRANSFORM_ERROR = 2 * type_epsilon<float>();

    auto input_norm
        = norm_complex(reinterpret_cast<const rocfft_complex<float>*>(original_host_data.data()),
                       original_host_data.size(),
                       1,
                       1,
                       original_host_data.size(),
                       {0});
    auto diff = distance_1to1_complex(
        reinterpret_cast<const rocfft_complex<float>*>(original_host_data.data()),
        reinterpret_cast<const rocfft_complex<float>*>(modified_host_data.data()),
        // data is all contiguous, we can treat it as 1d
        original_host_data.size(),
        1,
        1,
        original_host_data.size(),
        1,
        modified_host_data.size(),
        nullptr,
        MAX_TRANSFORM_ERROR,
        {0},
        {0});

    EXPECT_LT(diff.l_2 / input_norm.l_2,
              sqrt(log2(original_host_data.size())) * MAX_TRANSFORM_ERROR);
    EXPECT_LT(diff.l_inf / input_norm.l_inf, log2(original_host_data.size()) * MAX_TRANSFORM_ERROR);
}

TEST(rocfft_UnitTest, hipGraph_execution)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    hipGraph_t     graph      = nullptr;
    hipGraphExec_t graph_exec = nullptr;

    size_t N = 256;

    size_t seed = 100;

    auto offset_1 = rocfft_complex<float>{.1, .1};
    auto offset_2 = rocfft_complex<float>{-.1, -.1};

    float scale     = 2.2;
    float inv_scale = 1. / scale;

    auto output_init_val = rocfft_complex<float>(0., 0.);

    size_t num_kernel_launches = 100;
    size_t num_graph_launches  = 10;

    gpubuf_t<rocfft_complex<float>>    device_mem_in;
    std::vector<rocfft_complex<float>> host_mem_in;
    init_input_data(N, seed, host_mem_in, device_mem_in);
    rocfft_complex<float>* in_ptr = static_cast<rocfft_complex<float>*>(device_mem_in.data());

    gpubuf_t<rocfft_complex<float>>    device_mem_out;
    std::vector<rocfft_complex<float>> host_mem_out;
    init_data<rocfft_complex<float>>(N, output_init_val, host_mem_out, device_mem_out);
    rocfft_complex<float>* out_ptr = static_cast<rocfft_complex<float>*>(device_mem_out.data());

    gpubuf_t<size_t>    device_mem_counter;
    std::vector<size_t> host_mem_counter;
    init_data<size_t>(N, 0, host_mem_counter, device_mem_counter);
    size_t* counter_ptr = static_cast<size_t*>(device_mem_counter.data());

    rocfft_plan plan;
    create_forward_fft_plan(N, plan);

    rocfft_plan plan_inv;
    create_inverse_fft_plan(N, plan_inv);

    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipStream_wrapper_t stream;
    hipStream_wrapper_t other_stream;
    stream.alloc();
    other_stream.alloc();

    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);

    rocfft_execution_info info;
    set_fft_info(stream, info);

    // add offset to device input data
    for(size_t i = 0; i < num_kernel_launches; ++i)
        offset_device_data_complex<rocfft_complex<float>>(stream, offset_1, N, in_ptr);

    // back out the offsets
    for(size_t i = 0; i < num_kernel_launches; ++i)
        offset_device_data_complex<rocfft_complex<float>>(stream, offset_2, N, in_ptr);

    // scale the device input data
    scale_device_data(stream, scale, N, in_ptr);
    // backout the scale
    scale_device_data(stream, inv_scale, N, in_ptr);

    // run forward transform on input data
    run_forward_fft(info, plan, in_ptr, out_ptr);

    // scale the device output data
    scale_device_data(stream, scale, N, out_ptr);
    // backout the scale
    scale_device_data(stream, inv_scale, N, out_ptr);

    // run (in-place) inverse transform on output data
    run_inverse_fft(info, plan_inv, out_ptr, nullptr);

    // normalize results of an inverse transform, so it can be directly
    // compared to the original data before the forward transform
    auto inv_scale_N = 1. / static_cast<float>(N);
    scale_device_data(stream, inv_scale_N, N, out_ptr);

    // add offset to device output data
    for(size_t i = 0; i < num_kernel_launches; ++i)
        offset_device_data_complex<rocfft_complex<float>>(stream, offset_1, N, out_ptr);

    // back out the offsets
    for(size_t i = 0; i < num_kernel_launches; ++i)
        offset_device_data_complex<rocfft_complex<float>>(stream, offset_2, N, out_ptr);

    // increment counter
    offset_device_data_real<size_t>(stream, 1, N, counter_ptr);

    ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);

    // make sure no actual work has been done for
    // the captured stream before graph execution
    compare_data_exact_match<rocfft_complex<float>>(other_stream, host_mem_out, device_mem_out);

    ASSERT_EQ(hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0), hipSuccess);
    ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);

    for(size_t i = 0; i < num_graph_launches; ++i)
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);

    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    stream.free();

    // check for correctness of the output data
    compare_data(host_mem_in, device_mem_out);

    // check for correctness of the counter
    // incremented with multiple graph executions
    std::vector<size_t> host_mem_counter_modified(N);
    fill(host_mem_counter_modified.begin(), host_mem_counter_modified.end(), num_graph_launches);
    compare_data_exact_match<size_t>(other_stream, host_mem_counter_modified, device_mem_counter);

    other_stream.free();
}
