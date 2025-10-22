// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#ifndef ACCURACY_TEST
#define ACCURACY_TEST

#include <algorithm>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

#include "client_except.h"
#include "enum_to_string.h"
#include "fft_params.h"
#include "fftw_transform.h"
#include "gpubuf.h"
#include "rocfft_against_fftw.h"
#include "sys_mem.h"
#include "test_params.h"

extern int  verbose;
extern bool fftw_compare;

// Remember the results of the last FFT we computed with FFTW.  Tests
// are ordered so that later cases can often reuse this result.
struct last_cpu_fft_cache
{
    // keys to the cache
    std::vector<size_t> length;
    size_t              nbatch         = 0;
    fft_transform_type  transform_type = fft_transform_type_complex_forward;
    bool                run_callbacks  = false;
    fft_precision       precision      = fft_precision_single;
    double              scale_factor   = 1.0;

    // FFTW input/output
    std::vector<hostbuf> cpu_input;
    std::vector<hostbuf> cpu_output;
};
extern last_cpu_fft_cache last_cpu_fft_data;

// Perform several checks to make sure buffers will fit in device memory
template <class Tparams>
inline void check_problem_fits_device_memory(Tparams& params, const int verbose)
{

    size_t vram_avail = 0;

    if(vramgb == 0)
    {
        // Check free and total available memory:
        size_t free       = 0;
        size_t total      = 0;
        auto   hip_status = hipMemGetInfo(&free, &total);
        if(hip_status != hipSuccess || total == 0)
        {
            ++n_hip_failures;
            std::stringstream ss;
            if(total == 0)
                ss << "hipMemGetInfo claims there there isn't any vram";
            else
                ss << "hipMemGetInfo failure with error " << hip_status;
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{ss.str()};
            }
            else
            {
                throw ROCFFT_FAIL{ss.str()};
            }
        }
        vram_avail = total;
    }
    else
    {
        vram_avail = vramgb * ONE_GiB;
    }

    // First try a quick estimation of vram footprint, to speed up skipping tests
    // that are too large to fit in the gpu (no plan created with the rocFFT backend)
    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);

    if(!vram_fits_problem(raw_vram_footprint, vram_avail))
    {
        std::stringstream ss;
        ss << "Raw problem size (" << bytes_to_GiB(raw_vram_footprint)
           << " GiB) raw data too large for device";
        throw ROCFFT_SKIP{ss.str()};
    }

    if(verbose > 2)
    {
        std::cout << "Raw problem size: " << raw_vram_footprint << std::endl;
    }

    // If it passed the quick estimation test, go for the more
    // accurate calculation that actually creates the plan and
    // take into account the work buffer size
    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint, vram_avail))
    {
        if(verbose)
        {
            std::cout << "Problem raw data won't fit on device; skipped." << std::endl;
        }
        std::stringstream ss;
        ss << "Problem size (" << bytes_to_GiB(vram_footprint)
           << " GiB) raw data too large for device";
        throw ROCFFT_SKIP{ss.str()};
    }
}

template <typename Tfloat>
bool fftw_plan_uses_bluestein(const typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan)
{
#ifdef FFTW_HAVE_SPRINT_PLAN
    char*       print_plan_c_str = fftw_sprint_plan<Tfloat>(cpu_plan);
    std::string print_plan(print_plan_c_str);
    free(print_plan_c_str);
    return print_plan.find("bluestein") != std::string::npos;
#else
    // assume worst case (bluestein is always used)
    return true;
#endif
}

// Base gtest class for comparison with FFTW.
class accuracy_test : public ::testing::TestWithParam<fft_params>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<accuracy_test::ParamType>& info)
    {
        return info.param.token();
    }
};

struct callback_test_data
{
    // scalar to modify the input/output with
    double scalar;
    // base address of input, to ensure that each callback gets an offset from that base
    void* base;
};

void* get_load_callback_host(fft_array_type itype,
                             fft_precision  precision,
                             bool           round_trip_inverse);
void  apply_load_callback(const fft_params& params, std::vector<hostbuf>& input);
void  apply_store_callback(const fft_params& params, std::vector<hostbuf>& output);
void* get_store_callback_host(fft_array_type otype,
                              fft_precision  precision,
                              bool           round_trip_inverse);

static auto allocate_cpu_fft_buffer(const fft_precision        precision,
                                    const fft_array_type       type,
                                    const std::vector<size_t>& size)
{
    // FFTW does not support half-precision, so we do single instead.
    // So if we need to do a half-precision FFTW transform, allocate
    // enough buffer for single-precision instead.
    return allocate_host_buffer(
        precision == fft_precision_half ? fft_precision_single : precision, type, size);
}

template <typename Tfloat>
inline void execute_cpu_fft(fft_params&                                  params,
                            fft_params&                                  contiguous_params,
                            typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan,
                            std::vector<hostbuf>&                        cpu_input,
                            std::vector<hostbuf>&                        cpu_output)
{
    // CPU output might not be allocated already for us, if FFTW never
    // needed an output buffer during planning
    if(cpu_output.empty())
        cpu_output = allocate_cpu_fft_buffer(
            contiguous_params.precision, contiguous_params.otype, contiguous_params.osize);

    // If this is either C2R or callbacks are enabled, the
    // input will be modified.  So we need to modify the copy instead.
    std::vector<hostbuf>  cpu_input_copy(cpu_input.size());
    std::vector<hostbuf>* input_ptr = &cpu_input;
    if(params.run_callbacks || contiguous_params.transform_type == fft_transform_type_real_inverse)
    {
        for(size_t i = 0; i < cpu_input.size(); ++i)
        {
            cpu_input_copy[i] = cpu_input[i].copy();
        }

        input_ptr = &cpu_input_copy;
    }

    // run FFTW (which may destroy CPU input)
    apply_load_callback(params, *input_ptr);
    fftw_run<Tfloat>(contiguous_params.transform_type, cpu_plan, *input_ptr, cpu_output);
    // clean up
    fftw_destroy_plan_type(cpu_plan);
    // ask FFTW to fully clean up, since it tries to cache plan details
    fftw_cleanup();
    cpu_plan = nullptr;
    apply_store_callback(params, cpu_output);
}

// execute the GPU transform
template <class Tparams>
inline void execute_gpu_fft(Tparams&              params,
                            std::vector<void*>&   pibuffer,
                            std::vector<void*>&   pobuffer,
                            std::vector<gpubuf>&  obuffer,
                            std::vector<hostbuf>& gpu_output,
                            bool                  round_trip_inverse = false)
{
    gpubuf_t<callback_test_data> load_cb_data_dev;
    gpubuf_t<callback_test_data> store_cb_data_dev;
    if(params.run_callbacks)
    {
        void* load_cb_host
            = get_load_callback_host(params.itype, params.precision, round_trip_inverse);

        callback_test_data load_cb_data_host;

        if(round_trip_inverse)
        {
            load_cb_data_host.scalar = params.store_cb_scalar;
        }
        else
        {
            load_cb_data_host.scalar = params.load_cb_scalar;
        }

        load_cb_data_host.base = pibuffer.front();

        auto hip_status = hipSuccess;

        hip_status = load_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            std::stringstream ss;
            ss << "Error occurred when allocating device memory for loading callback";
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{ss.str()};
            }
            else
            {
                throw ROCFFT_FAIL{ss.str()};
            }
        }
        hip_status = hipMemcpy(load_cb_data_dev.data(),
                               &load_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            std::stringstream ss;
            ss << "Error occurred when copying data to device for loading callback";
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{ss.str()};
            }
            else
            {
                throw ROCFFT_FAIL{ss.str()};
            }
        }

        void* store_cb_host
            = get_store_callback_host(params.otype, params.precision, round_trip_inverse);

        callback_test_data store_cb_data_host;

        if(round_trip_inverse)
        {
            store_cb_data_host.scalar = params.load_cb_scalar;
        }
        else
        {
            store_cb_data_host.scalar = params.store_cb_scalar;
        }

        store_cb_data_host.base = pobuffer.front();

        hip_status = store_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            std::stringstream ss;
            ss << "Error occurred when allocating device memory for storing callback";
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{ss.str()};
            }
            else
            {
                throw ROCFFT_FAIL{ss.str()};
            }
        }

        hip_status = hipMemcpy(store_cb_data_dev.data(),
                               &store_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            std::stringstream ss;
            ss << "Error occurred when copying data to device for storing callback";
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{ss.str()};
            }
            else
            {
                throw ROCFFT_FAIL{ss.str()};
            }
        }

        auto fft_status = params.set_callbacks(
            load_cb_host, load_cb_data_dev.data(), store_cb_host, store_cb_data_dev.data());
        if(fft_status != fft_status_success)
            throw std::runtime_error("set callback failure");
    }

    // Execute the transform:
    auto fft_status = params.execute(pibuffer.data(), pobuffer.data());
    if(fft_status != fft_status_success)
        throw std::runtime_error("FFT plan execution failure");

    // if not comparing, then just executing the GPU FFT is all we
    // need to do
    if(!fftw_compare)
        return;

    ASSERT_TRUE(!gpu_output.empty()) << "no output buffers";

    // if output is in multiple bricks, collect it into the
    // host buffer where the results need to go
    params.multi_gpu_finalize(gpu_output, obuffer, pobuffer);

    if(params.ofields.empty())
    {
        // otherwise, copy directly from the device
        for(unsigned int idx = 0; idx < gpu_output.size(); ++idx)
        {
            ASSERT_TRUE(gpu_output[idx].data() != nullptr)
                << "output buffer index " << idx << " is empty";
            auto hip_status = hipMemcpy(gpu_output[idx].data(),
                                        pobuffer.at(idx),
                                        gpu_output[idx].size(),
                                        hipMemcpyDeviceToHost);
            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                std::stringstream ss;
                ss << "hipMemcpy failure";
                if(skip_runtime_fails)
                {
                    throw ROCFFT_SKIP{ss.str()};
                }
                else
                {
                    throw ROCFFT_FAIL{ss.str()};
                }
            }
        }
    }
    if(verbose > 2)
    {
        std::cout << "GPU output:\n";
        params.print_obuffer(gpu_output);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU output:\n";
        params.print_obuffer_flat(gpu_output);
    }
}

template <typename Tfloat>
static void assert_init_value(const std::vector<hostbuf>& output,
                              const size_t                idx,
                              const Tfloat                orig_value);

template <>
void assert_init_value(const std::vector<hostbuf>& output, const size_t idx, const float orig_value)
{
    float actual_value = reinterpret_cast<const float*>(output.front().data())[idx];
    ASSERT_EQ(actual_value, orig_value) << "index " << idx;
}

template <>
void assert_init_value(const std::vector<hostbuf>& output,
                       const size_t                idx,
                       const double                orig_value)
{
    double actual_value = reinterpret_cast<const double*>(output.front().data())[idx];
    ASSERT_EQ(actual_value, orig_value) << "index " << idx;
}

template <>
void assert_init_value(const std::vector<hostbuf>& output,
                       const size_t                idx,
                       const rocfft_complex<float> orig_value)
{
    // if this is interleaved, check directly
    if(output.size() == 1)
    {
        rocfft_complex<float> actual_value
            = reinterpret_cast<const rocfft_complex<float>*>(output.front().data())[idx];
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
    else
    {
        // planar
        rocfft_complex<float> actual_value{
            reinterpret_cast<const float*>(output.front().data())[idx],
            reinterpret_cast<const float*>(output.back().data())[idx]};
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
}

template <>
void assert_init_value(const std::vector<hostbuf>&  output,
                       const size_t                 idx,
                       const rocfft_complex<double> orig_value)
{
    // if this is interleaved, check directly
    if(output.size() == 1)
    {
        rocfft_complex<double> actual_value
            = reinterpret_cast<const rocfft_complex<double>*>(output.front().data())[idx];
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
    else
    {
        // planar
        rocfft_complex<double> actual_value{
            reinterpret_cast<const double*>(output.front().data())[idx],
            reinterpret_cast<const double*>(output.back().data())[idx]};
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
}

static const int OUTPUT_INIT_PATTERN = 0xcd;
template <class Tfloat>
void check_single_output_stride(const std::vector<hostbuf>& output,
                                const size_t                offset,
                                const std::vector<size_t>&  length,
                                const std::vector<size_t>&  stride,
                                const size_t                i)
{
    Tfloat orig;
    memset(static_cast<void*>(&orig), OUTPUT_INIT_PATTERN, sizeof(Tfloat));

    size_t curLength         = length[i];
    size_t curStride         = stride[i];
    size_t nextSmallerLength = i == length.size() - 1 ? 0 : length[i + 1];
    size_t nextSmallerStride = i == stride.size() - 1 ? 0 : stride[i + 1];

    if(nextSmallerLength == 0)
    {
        // this is the fastest dim, indexes that are not multiples of
        // the stride should be the initial value
        for(size_t idx = 0; idx < (curLength - 1) * curStride; ++idx)
        {
            if(idx % curStride != 0)
                assert_init_value<Tfloat>(output, idx, orig);
        }
    }
    else
    {
        for(size_t lengthIdx = 0; lengthIdx < curLength; ++lengthIdx)
        {
            // check that the space after the next smaller dim and the
            // end of this dim is initial value
            for(size_t idx = nextSmallerLength * nextSmallerStride; idx < curStride; ++idx)
                assert_init_value<Tfloat>(output, idx, orig);

            check_single_output_stride<Tfloat>(
                output, offset + lengthIdx * curStride, length, stride, i + 1);
        }
    }
}

template <class Tparams>
void check_output_strides(const std::vector<hostbuf>& output, Tparams& params)
{
    // treat batch+dist like highest length+stride, if batch > 1
    std::vector<size_t> length;
    std::vector<size_t> stride;
    if(params.nbatch > 1)
    {
        length.push_back(params.nbatch);
        stride.push_back(params.odist);
    }

    auto olength = params.olength();
    std::copy(olength.begin(), olength.end(), std::back_inserter(length));
    std::copy(params.ostride.begin(), params.ostride.end(), std::back_inserter(stride));

    if(params.precision == fft_precision_single)
    {
        if(params.otype == fft_array_type_real)
            check_single_output_stride<float>(output, 0, length, stride, 0);
        else
            check_single_output_stride<rocfft_complex<float>>(output, 0, length, stride, 0);
    }
    else
    {
        if(params.otype == fft_array_type_real)
            check_single_output_stride<double>(output, 0, length, stride, 0);
        else
            check_single_output_stride<rocfft_complex<double>>(output, 0, length, stride, 0);
    }
}

// run rocFFT inverse transform
template <class Tparams>
inline void run_round_trip_inverse(Tparams&              params,
                                   std::vector<gpubuf>&  obuffer,
                                   std::vector<void*>&   pibuffer,
                                   std::vector<void*>&   pobuffer,
                                   std::vector<hostbuf>& gpu_output)
{
    params.validate();

    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    // Create FFT plan - this will also allocate work buffer, but will throw a
    // specific exception if that step fails
    auto plan_status = fft_status_success;
    try
    {
        plan_status = params.create_plan();
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        std::stringstream ss;
        ss << "Failed to allocate work buffer (size: " << e.attempted_size << ")";
        ++n_hip_failures;
        if(skip_runtime_fails)
        {
            throw ROCFFT_SKIP{ss.str()};
        }
        else
        {
            throw ROCFFT_FAIL{ss.str()};
        }
    }
    ASSERT_EQ(plan_status, fft_status_success) << "round trip inverse plan creation failed";

    auto obuffer_sizes = params.obuffer_sizes();

    if(params.placement != fft_placement_inplace)
    {
        for(unsigned int i = 0; i < obuffer_sizes.size(); ++i)
        {
            // If we're validating output strides, init the
            // output buffer to a known pattern and we can check
            // that the pattern is untouched in places that
            // shouldn't have been touched.
            if(params.check_output_strides)
            {
                auto hip_status = hipMemset(pobuffer[i], OUTPUT_INIT_PATTERN, obuffer_sizes[i]);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    std::stringstream ss;
                    ss << "hipMemset failure";
                    if(skip_runtime_fails)
                    {
                        throw ROCFFT_SKIP{ss.str()};
                    }
                    else
                    {
                        throw ROCFFT_FAIL{ss.str()};
                    }
                }
            }
        }
    }

    if(params.multiGPU > 1)
    {
        if(verbose > 0)
        {
            std::cout << "scattering data for multi-GPU inverse" << std::endl;
        }
        std::vector<hostbuf> cpu_input;
        params.multi_gpu_prepare(cpu_input, obuffer, pibuffer, pobuffer);
    }

    // execute GPU transform
    execute_gpu_fft(params, pibuffer, pobuffer, obuffer, gpu_output, true);
}

// compare rocFFT inverse transform with forward transform input
template <class Tparams>
inline void compare_round_trip_inverse(Tparams&              params,
                                       fft_params&           contiguous_params,
                                       std::vector<hostbuf>& gpu_output,
                                       std::vector<hostbuf>& cpu_input,
                                       const VectorNorms&    cpu_input_norm,
                                       size_t                total_length)
{
    if(params.check_output_strides)
    {
        check_output_strides<Tparams>(gpu_output, params);
    }

    // compute GPU output norm
    std::shared_future<VectorNorms> gpu_norm = std::async(std::launch::async, [&]() {
        return norm(gpu_output,
                    params.olength(),
                    params.nbatch,
                    params.precision,
                    params.otype,
                    params.ostride,
                    params.odist,
                    params.ooffset);
    });

    // compare GPU inverse output to CPU forward input
    std::unique_ptr<std::vector<std::pair<size_t, size_t>>> linf_failures;
    if(verbose > 1)
        linf_failures = std::make_unique<std::vector<std::pair<size_t, size_t>>>();
    const double linf_cutoff
        = type_epsilon(params.precision) * cpu_input_norm.l_inf * log(total_length);

    VectorNorms diff = distance(cpu_input,
                                gpu_output,
                                params.olength(),
                                params.nbatch,
                                params.precision,
                                contiguous_params.itype,
                                contiguous_params.istride,
                                contiguous_params.idist,
                                params.otype,
                                params.ostride,
                                params.odist,
                                linf_failures.get(),
                                linf_cutoff,
                                {0},
                                params.ooffset,
                                1.0 / total_length);

    if(verbose > 1)
    {
        std::cout << "GPU output Linf norm: " << gpu_norm.get().l_inf << "\n";
        std::cout << "GPU output L2 norm:   " << gpu_norm.get().l_2 << "\n";
        std::cout << "GPU linf norm failures:";
        std::sort(linf_failures->begin(), linf_failures->end());
        for(const auto& i : *linf_failures)
        {
            std::cout << " (" << i.first << "," << i.second << ")";
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_inf)) << params.str();
    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_2)) << params.str();

    switch(params.precision)
    {
    case fft_precision_half:
        max_linf_eps_half
            = std::max(max_linf_eps_half, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_half
            = std::max(max_l2_eps_half, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_single:
        max_linf_eps_single
            = std::max(max_linf_eps_single, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_single
            = std::max(max_l2_eps_single, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double
            = std::max(max_linf_eps_double, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_double
            = std::max(max_l2_eps_double, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    }

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    EXPECT_TRUE(diff.l_inf <= linf_cutoff)
        << "Linf test failed.  Linf:" << diff.l_inf
        << "\tnormalized Linf: " << diff.l_inf / cpu_input_norm.l_inf << "\tcutoff: " << linf_cutoff
        << params.str();

    EXPECT_TRUE(diff.l_2 / cpu_input_norm.l_2
                <= sqrt(log2(total_length)) * type_epsilon(params.precision))
        << "L2 test failed. L2: " << diff.l_2
        << "\tnormalized L2: " << diff.l_2 / cpu_input_norm.l_2
        << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision)
        << params.str();
}

// RAII type to put data into the cache when this object leaves scope
struct StoreCPUDataToCache
{
    StoreCPUDataToCache(std::vector<hostbuf>& cpu_input, std::vector<hostbuf>& cpu_output)
        : cpu_input(cpu_input)
        , cpu_output(cpu_output)
    {
    }
    ~StoreCPUDataToCache()
    {
        last_cpu_fft_data.cpu_output.swap(cpu_output);
        last_cpu_fft_data.cpu_input.swap(cpu_input);
    }
    std::vector<hostbuf>& cpu_input;
    std::vector<hostbuf>& cpu_output;
};

// run CPU + rocFFT transform with the given params and compare
template <class Tfloat, class Tparams>
inline void fft_vs_reference_impl(Tparams& params, bool round_trip)
{
    // Call hipGetLastError to reset any errors
    // returned by previous HIP runtime API calls.
    hipError_t hip_status = hipGetLastError();

    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    auto ibuffer_sizes = params.ibuffer_sizes();
    auto obuffer_sizes = params.obuffer_sizes();

    // Make sure FFT buffers fit in device memory
    check_problem_fits_device_memory(params, verbose);

    // Create FFT plan - this will also allocate work buffer, but
    // will throw a specific exception if that step fails
    auto plan_status = fft_status_success;
    try
    {
        plan_status = params.create_plan();
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        ++n_hip_failures;
        std::stringstream ss;
        ss << "Work buffer allocation failed with size: " << e.attempted_size;
        if(skip_runtime_fails)
        {
            throw ROCFFT_SKIP{ss.str()};
        }
        else
        {
            throw ROCFFT_FAIL{ss.str()};
        }
    }
    ASSERT_EQ(plan_status, fft_status_success) << "plan creation failed";

    fft_params contiguous_params;
    contiguous_params.length         = params.length;
    contiguous_params.precision      = params.precision;
    contiguous_params.placement      = fft_placement_notinplace;
    contiguous_params.transform_type = params.transform_type;
    contiguous_params.nbatch         = params.nbatch;
    contiguous_params.itype          = contiguous_itype(params.transform_type);
    contiguous_params.otype          = contiguous_otype(contiguous_params.transform_type);

    contiguous_params.validate();

    if(!contiguous_params.valid(verbose))
    {
        throw std::runtime_error("Invalid contiguous params");
    }

    if(verbose > 3)
    {
        std::cout << "CPU params:\n";
        std::cout << contiguous_params.str("\n\t") << std::endl;
    }

    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
        if(hip_status != hipSuccess)
        {
            std::stringstream ss;
            if(hip_status == hipErrorOutOfMemory)
            {
                ss << "Input buffer size (" << bytes_to_GiB(ibuffer_sizes[i])
                   << " GiB) raw data too large for device";
            }
            else
            {
                ss << "hipMalloc failure for input buffer " << i << " size " << ibuffer_sizes[i]
                   << "(" << bytes_to_GiB(ibuffer_sizes[i]) << " GiB)"
                   << " with code " << hipError_to_string(hip_status);
            }
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{ss.str()};
            }
            else
            {
                throw ROCFFT_FAIL{ss.str()};
            }
        }
        pibuffer[i] = ibuffer[i].data();
    }

    // allocation counts in elements, ibuffer_sizes is in bytes
    auto ibuffer_sizes_elems = ibuffer_sizes;
    for(auto& buf : ibuffer_sizes_elems)
        buf /= var_size<size_t>(params.precision, params.itype);

    // Check cache first - nbatch is a >= comparison because we compute
    // the largest batch size and cache it.  Smaller batch runs can
    // compare against the larger data.
    std::vector<hostbuf>                 cpu_input;
    std::vector<hostbuf>                 cpu_output;
    std::shared_future<void>             convert_cpu_output_precision;
    std::shared_future<void>             convert_cpu_input_precision;
    bool                                 run_fftw = true;
    std::unique_ptr<StoreCPUDataToCache> store_to_cache;
    if(fftw_compare && last_cpu_fft_data.length == params.length
       && last_cpu_fft_data.transform_type == params.transform_type
       && last_cpu_fft_data.run_callbacks == params.run_callbacks
       && last_cpu_fft_data.scale_factor == params.scale_factor
       && last_cpu_fft_data.precision >= params.precision)
    {
        if(last_cpu_fft_data.nbatch >= params.nbatch)
        {
            // use the cached input/output
            cpu_input.swap(last_cpu_fft_data.cpu_input);
            cpu_output.swap(last_cpu_fft_data.cpu_output);
            run_fftw = false;

            store_to_cache = std::make_unique<StoreCPUDataToCache>(cpu_input, cpu_output);

            if(params.precision != last_cpu_fft_data.precision)
            {
                // Tests should be ordered so we do wider first, then narrower.
                switch(params.precision)
                {
                case fft_precision_double:
                    std::cerr
                        << "test ordering is incorrect: double precision follows a narrower one"
                        << std::endl;
                    abort();
                    break;
                case fft_precision_single:
                    if(last_cpu_fft_data.precision != fft_precision_double)
                    {
                        std::cerr
                            << "test ordering is incorrect: float precision follows a narrower one"
                            << std::endl;
                        abort();
                    }
                    // convert the input/output to single-precision
                    convert_cpu_output_precision = std::async(std::launch::async, [&]() {
                        narrow_precision_inplace<double, float>(cpu_output.front());
                    });
                    convert_cpu_input_precision  = std::async(std::launch::async, [&]() {
                        narrow_precision_inplace<double, float>(cpu_input.front());
                    });
                    break;
                case fft_precision_half:
                    // convert to half precision
                    if(last_cpu_fft_data.precision == fft_precision_double)
                    {
                        convert_cpu_output_precision = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<double, rocfft_fp16>(cpu_output.front());
                        });
                        convert_cpu_input_precision  = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<double, rocfft_fp16>(cpu_input.front());
                        });
                    }
                    else if(last_cpu_fft_data.precision == fft_precision_single)
                    {
                        convert_cpu_output_precision = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<float, rocfft_fp16>(cpu_output.front());
                        });
                        convert_cpu_input_precision  = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<float, rocfft_fp16>(cpu_input.front());
                        });
                    }
                    else
                    {
                        std::cerr << "unhandled previous precision, cannot convert to half"
                                  << std::endl;
                        abort();
                    }
                    break;
                }
                last_cpu_fft_data.precision = params.precision;
            }
        }
        // If the last result has a smaller batch than the new
        // params, that might be a developer error - tests should be
        // ordered to generate the bigger batch first.  But if tests
        // got filtered or skipped due to insufficient memory, we
        // might never have tried to generate the bigger batch first.
        // So just fall through and redo the CPU FFT.
    }
    else
    {
        // Clear cache explicitly so that even if we didn't get a hit,
        // we're not uselessly holding on to cached cpu input/output
        last_cpu_fft_data = last_cpu_fft_cache();
    }

    // Allocate CPU input
    if(run_fftw)
    {
        cpu_input = allocate_cpu_fft_buffer(
            contiguous_params.precision, contiguous_params.itype, contiguous_params.isize);
    }

    // Create FFTW plan - this may write to input, but that's fine
    // since there's nothing in there right now
    typename fftw_trait<Tfloat>::fftw_plan_type cpu_plan = nullptr;
    if(run_fftw)
    {
        // Normally, we would want to defer allocation of CPU output
        // buffer until when we actually do the CPU FFT.  But if we're
        // using FFTW wisdom, FFTW needs an output buffer at plan
        // creation time.
        if(use_fftw_wisdom)
        {
            cpu_output = allocate_cpu_fft_buffer(
                contiguous_params.precision, contiguous_params.otype, contiguous_params.osize);
        }
        cpu_plan = fftw_plan_via_rocfft<Tfloat>(contiguous_params.length,
                                                contiguous_params.istride,
                                                contiguous_params.ostride,
                                                contiguous_params.nbatch,
                                                contiguous_params.idist,
                                                contiguous_params.odist,
                                                contiguous_params.transform_type,
                                                cpu_input,
                                                cpu_output);
    }

    // Host-side buffer used to store input to transfer to GPU, copy GPU input
    // to contiguous layout for FFTW, and reused to store IFFT output.
    std::vector<hostbuf> gpu_input_data;

    auto is_host_gen = (params.igen == fft_input_generator_host
                        || params.igen == fft_input_random_generator_host);

    // allocate and populate the input buffer (cpu/gpu)
    if(run_fftw)
    {
        gpu_input_data = allocate_host_buffer(params.precision, params.itype, ibuffer_sizes_elems);

#ifdef USE_HIPRAND
        if(!is_host_gen)
        {
            // generate the input directly on the gpu
            params.compute_input(ibuffer);

            // Copy the input to CPU
            if(params.itype != contiguous_params.itype
               || params.istride != contiguous_params.istride
               || params.idist != contiguous_params.idist
               || params.isize != contiguous_params.isize)
            {
                // Copy input to CPU
                for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
                {
                    hip_status = hipMemcpy(gpu_input_data.at(idx).data(),
                                           ibuffer[idx].data(),
                                           ibuffer_sizes[idx],
                                           hipMemcpyDeviceToHost);
                    if(hip_status != hipSuccess)
                    {
                        ++n_hip_failures;
                        std::stringstream ss;
                        ss << "hipMemcpy failure with error " << hip_status;
                        if(skip_runtime_fails)
                        {
                            throw ROCFFT_SKIP{ss.str()};
                        }
                        else
                        {
                            throw ROCFFT_FAIL{ss.str()};
                        }
                    }
                }

                copy_buffers(gpu_input_data,
                             cpu_input,
                             params.ilength(),
                             params.nbatch,
                             params.precision,
                             params.itype,
                             params.istride,
                             params.idist,
                             contiguous_params.itype,
                             contiguous_params.istride,
                             contiguous_params.idist,
                             params.ioffset,
                             contiguous_params.ioffset);
            }
            else
            {
                // Copy input to CPU
                for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
                {
                    hip_status = hipMemcpy(cpu_input.at(idx).data(),
                                           ibuffer[idx].data(),
                                           ibuffer_sizes[idx],
                                           hipMemcpyDeviceToHost);
                    if(hip_status != hipSuccess)
                    {
                        ++n_hip_failures;
                        std::stringstream ss;
                        ss << "hipMemcpy failure with error " << hip_status;
                        if(skip_runtime_fails)
                        {
                            throw ROCFFT_SKIP{ss.str()};
                        }
                        else
                        {
                            throw ROCFFT_FAIL{ss.str()};
                        }
                    }
                }
            }
        }

#endif
        if(is_host_gen)
        {
            params.compute_input(gpu_input_data);
            // Copy the input to CPU
            if(params.itype != contiguous_params.itype
               || params.istride != contiguous_params.istride
               || params.idist != contiguous_params.idist
               || params.isize != contiguous_params.isize)
            {
                // Copy input to CPU and make input contiguous
                copy_buffers(gpu_input_data,
                             cpu_input,
                             params.ilength(),
                             params.nbatch,
                             params.precision,
                             params.itype,
                             params.istride,
                             params.idist,
                             contiguous_params.itype,
                             contiguous_params.istride,
                             contiguous_params.idist,
                             params.ioffset,
                             contiguous_params.ioffset);
            }
            else
            {
                // Direct copy input to cpu_input as is
                for(unsigned int idx = 0; idx < gpu_input_data.size(); ++idx)
                {
                    hip_status = hipMemcpy(cpu_input.at(idx).data(),
                                           gpu_input_data.at(idx).data(),
                                           gpu_input_data.at(idx).size(),
                                           hipMemcpyHostToHost);
                    if(hip_status != hipSuccess)
                    {
                        ++n_hip_failures;
                        std::stringstream ss;
                        ss << "hipMemcpy failure with error " << hip_status;
                        if(skip_runtime_fails)
                        {
                            throw ROCFFT_SKIP{ss.str()};
                        }
                        else
                        {
                            throw ROCFFT_FAIL{ss.str()};
                        }
                    }
                }
            }
            // Copy input to GPU
            for(unsigned int idx = 0; idx < gpu_input_data.size(); ++idx)
            {
                hip_status = hipMemcpy(ibuffer[idx].data(),
                                       gpu_input_data.at(idx).data(),
                                       ibuffer_sizes[idx],
                                       hipMemcpyHostToDevice);
            }
        }
    }
    else if(fftw_compare)
    {
        gpu_input_data = allocate_host_buffer(params.precision, params.itype, ibuffer_sizes_elems);

        // In case the cached cpu input needed conversion, wait for it
        if(convert_cpu_input_precision.valid())
            convert_cpu_input_precision.get();

        // gets a pre-computed gpu input buffer from the cpu cache
        std::vector<hostbuf>* gpu_input = &cpu_input;

        if(params.itype != contiguous_params.itype || params.istride != contiguous_params.istride
           || params.idist != contiguous_params.idist || params.isize != contiguous_params.isize)
        {
            copy_buffers(cpu_input,
                         gpu_input_data,
                         params.ilength(),
                         params.nbatch,
                         params.precision,
                         contiguous_params.itype,
                         contiguous_params.istride,
                         contiguous_params.idist,
                         params.itype,
                         params.istride,
                         params.idist,
                         {0},
                         params.ioffset);
            gpu_input = &gpu_input_data;
        }

        // Copy input to GPU
        for(unsigned int idx = 0; idx < gpu_input->size(); ++idx)
        {
            hip_status = hipMemcpy(ibuffer[idx].data(),
                                   gpu_input->at(idx).data(),
                                   ibuffer_sizes[idx],
                                   hipMemcpyHostToDevice);

            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                std::stringstream ss;
                ss << "hipMemcpy failure with error " << hip_status;
                if(skip_runtime_fails)
                {
                    throw ROCFFT_SKIP{ss.str()};
                }
                else
                {
                    throw ROCFFT_FAIL{ss.str()};
                }
            }
        }
    }

    if(verbose > 3)
    {
        std::cout << "CPU input:\n";
        contiguous_params.print_ibuffer(cpu_input);
    }

    // compute input norm
    std::shared_future<VectorNorms> cpu_input_norm;
    if(fftw_compare)
        cpu_input_norm = std::async(std::launch::async, [&]() {
            // in case the cached cpu input needed conversion, wait for it
            if(convert_cpu_input_precision.valid())
                convert_cpu_input_precision.get();

            auto input_norm = norm(cpu_input,
                                   contiguous_params.ilength(),
                                   contiguous_params.nbatch,
                                   contiguous_params.precision,
                                   contiguous_params.itype,
                                   contiguous_params.istride,
                                   contiguous_params.idist,
                                   contiguous_params.ioffset);
            if(verbose > 2)
            {
                std::cout << "CPU Input Linf norm:  " << input_norm.l_inf << "\n";
                std::cout << "CPU Input L2 norm:    " << input_norm.l_2 << "\n";
            }
            return input_norm;
        });

    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    std::vector<void*>   pobuffer;

    // allocate the output buffer

    if(params.placement == fft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            hip_status = obuffer_data[i].alloc(obuffer_sizes[i]);
            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                std::stringstream ss;
                ss << "hipMalloc failure for output buffer " << i << " size " << obuffer_sizes[i]
                   << "(" << bytes_to_GiB(obuffer_sizes[i]) << " GiB)"
                   << " with code " << hipError_to_string(hip_status);
                if(skip_runtime_fails)
                {
                    throw ROCFFT_SKIP{ss.str()};
                }
                else
                {
                    throw ROCFFT_FAIL{ss.str()};
                }
            }

            // If we're validating output strides, init the
            // output buffer to a known pattern and we can check
            // that the pattern is untouched in places that
            // shouldn't have been touched.
            if(params.check_output_strides)
            {
                hip_status
                    = hipMemset(obuffer_data[i].data(), OUTPUT_INIT_PATTERN, obuffer_sizes[i]);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    std::stringstream ss;
                    ss << "hipMemset failure with error " << hip_status;
                    if(skip_runtime_fails)
                    {
                        throw ROCFFT_SKIP{ss.str()};
                    }
                    else
                    {
                        throw ROCFFT_FAIL{ss.str()};
                    }
                }
            }
        }
    }
    pobuffer.resize(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    // scatter data out to multi-GPUs if this is a multi-GPU test
    params.multi_gpu_prepare(cpu_input, ibuffer, pibuffer, pobuffer);

    // Run CPU transform
    //
    // NOTE: This must happen after input is copied to GPU and input
    // norm is computed, since the CPU FFT may overwrite the input.
    VectorNorms              cpu_output_norm;
    std::shared_future<void> cpu_fft;
    if(fftw_compare)
        cpu_fft = std::async(std::launch::async, [&]() {
            // wait for input norm to finish, since we might overwrite input
            cpu_input_norm.get();

            if(run_fftw)
                execute_cpu_fft<Tfloat>(params, contiguous_params, cpu_plan, cpu_input, cpu_output);
            // in case the cached cpu output needed conversion, wait for it
            else if(convert_cpu_output_precision.valid())
                convert_cpu_output_precision.get();

            if(verbose > 3)
            {
                std::cout << "CPU output:\n";
                contiguous_params.print_obuffer(cpu_output);
            }

            cpu_output_norm = norm(cpu_output,
                                   params.olength(),
                                   params.nbatch,
                                   params.precision,
                                   contiguous_params.otype,
                                   contiguous_params.ostride,
                                   contiguous_params.odist,
                                   contiguous_params.ooffset);
            if(verbose > 2)
            {
                std::cout << "CPU Output Linf norm: " << cpu_output_norm.l_inf << "\n";
                std::cout << "CPU Output L2 norm:   " << cpu_output_norm.l_2 << "\n";
            }
        });

    // execute GPU transform
    std::vector<hostbuf> gpu_output
        = allocate_host_buffer(params.precision, params.otype, params.osize);

    execute_gpu_fft(params, pibuffer, pobuffer, *obuffer, gpu_output);

    params.free();

    if(params.check_output_strides)
    {
        check_output_strides<Tparams>(gpu_output, params);
    }

    // compute GPU output norm
    std::shared_future<VectorNorms> gpu_norm;
    if(fftw_compare)
        gpu_norm = std::async(std::launch::async, [&]() {
            return norm(gpu_output,
                        params.olength(),
                        params.nbatch,
                        params.precision,
                        params.otype,
                        params.ostride,
                        params.odist,
                        params.ooffset);
        });

    // compare output
    //
    // Compute the l-infinity and l-2 distance between the CPU and GPU output:
    // wait for cpu FFT so we can compute cutoff

    const auto total_length = product(params.length.begin(), params.length.end());

    std::unique_ptr<std::vector<std::pair<size_t, size_t>>> linf_failures;
    if(verbose > 1)
        linf_failures = std::make_unique<std::vector<std::pair<size_t, size_t>>>();
    double      linf_cutoff;
    VectorNorms diff;

    std::shared_future<void> compare_output;
    if(fftw_compare)
        compare_output = std::async(std::launch::async, [&]() {
            cpu_fft.get();
            linf_cutoff
                = type_epsilon(params.precision) * cpu_output_norm.l_inf * log(total_length);

            diff = distance(cpu_output,
                            gpu_output,
                            params.olength(),
                            params.nbatch,
                            params.precision,
                            contiguous_params.otype,
                            contiguous_params.ostride,
                            contiguous_params.odist,
                            params.otype,
                            params.ostride,
                            params.odist,
                            linf_failures.get(),
                            linf_cutoff,
                            {0},
                            params.ooffset);
        });

    // Update the cache if this current transform is different from
    // what's stored.  But if this transform only has a smaller batch
    // than what's cached, we can still keep the cache around since
    // the input/output we already have is still valid.
    const bool update_last_cpu_fft_data
        = last_cpu_fft_data.length != params.length
          || last_cpu_fft_data.transform_type != params.transform_type
          || last_cpu_fft_data.run_callbacks != params.run_callbacks
          || last_cpu_fft_data.precision != params.precision
          || last_cpu_fft_data.scale_factor != params.scale_factor
          || params.nbatch > last_cpu_fft_data.nbatch;

    // store cpu output in cache
    if(update_last_cpu_fft_data)
    {
        last_cpu_fft_data.length         = params.length;
        last_cpu_fft_data.nbatch         = params.nbatch;
        last_cpu_fft_data.transform_type = params.transform_type;
        last_cpu_fft_data.run_callbacks  = params.run_callbacks;
        last_cpu_fft_data.precision      = params.precision;
        last_cpu_fft_data.scale_factor   = params.scale_factor;
    }

    if(compare_output.valid())
        compare_output.get();

    if(!store_to_cache)
        store_to_cache = std::make_unique<StoreCPUDataToCache>(cpu_input, cpu_output);

    Tparams params_inverse;

    if(round_trip)
    {
        params_inverse.inverse_from_forward(params);

        run_round_trip_inverse<Tparams>(
            params_inverse, *obuffer, pobuffer, pibuffer, gpu_input_data);
    }

    if(fftw_compare)
    {
        ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_2));
        ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_inf));

        ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));
        ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));

        if(verbose > 1)
        {
            std::cout << "GPU output Linf norm: " << gpu_norm.get().l_inf << "\n";
            std::cout << "GPU output L2 norm:   " << gpu_norm.get().l_2 << "\n";
            std::cout << "GPU linf norm failures:";
            std::sort(linf_failures->begin(), linf_failures->end());
            for(const auto& i : *linf_failures)
            {
                std::cout << " (" << i.first << "," << i.second << ")";
            }
            std::cout << std::endl;
        }

        EXPECT_TRUE(std::isfinite(gpu_norm.get().l_inf)) << params.str();
        EXPECT_TRUE(std::isfinite(gpu_norm.get().l_2)) << params.str();
    }

    switch(params.precision)
    {
    case fft_precision_half:
        max_linf_eps_half
            = std::max(max_linf_eps_half, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_half
            = std::max(max_l2_eps_half, diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_single:
        max_linf_eps_single
            = std::max(max_linf_eps_single, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_single = std::max(max_l2_eps_single,
                                     diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double
            = std::max(max_linf_eps_double, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_double = std::max(max_l2_eps_double,
                                     diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    }

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    if(fftw_compare)
    {
        EXPECT_TRUE(diff.l_inf <= linf_cutoff)
            << "Linf test failed.  Linf:" << diff.l_inf
            << "\tnormalized Linf: " << diff.l_inf / cpu_output_norm.l_inf
            << "\tcutoff: " << linf_cutoff << "\n"
            << params.str();

        EXPECT_TRUE(diff.l_2 / cpu_output_norm.l_2
                    <= sqrt(log2(total_length)) * type_epsilon(params.precision))
            << "L2 test failed. L2: " << diff.l_2
            << "\tnormalized L2: " << diff.l_2 / cpu_output_norm.l_2
            << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision) << "\n"
            << params.str();
    }

    if(round_trip && fftw_compare)
    {
        compare_round_trip_inverse<Tparams>(params_inverse,
                                            contiguous_params,
                                            gpu_input_data,
                                            cpu_input,
                                            cpu_input_norm.get(),
                                            total_length);
    }
}

#endif
