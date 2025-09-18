// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef BITWISE_REPRO_TEST_H
#define BITWISE_REPRO_TEST_H

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "../../../shared/accuracy_test.h"
#include "../../../shared/enum_to_string.h"
#include "../../../shared/fft_params.h"
#include "../../../shared/gpubuf.h"
#include "../../../shared/rocfft_params.h"
#include "../../../shared/test_params.h"
#include "bitwise_repro_db.h"

extern int                          verbose;
extern std::unique_ptr<fft_hash_db> repro_db;

// Base gtest class for bitwise reproduction tests
class bitwise_repro_test : public ::testing::TestWithParam<fft_params>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<bitwise_repro_test::ParamType>& info)
    {
        return info.param.token();
    }
};

// execute the GPU transform
template <class Tparams>
inline void execute_fft(Tparams&              params,
                        std::vector<void*>&   pibuffer,
                        std::vector<void*>&   pobuffer,
                        std::vector<gpubuf>&  obuffer,
                        std::vector<hostbuf>& gpu_output)
{
    // Execute the transform:
    auto fft_status = params.execute(pibuffer.data(), pobuffer.data());
    if(fft_status != fft_status_success)
        throw std::runtime_error("rocFFT plan execution failure");

    ASSERT_TRUE(!gpu_output.empty()) << "no output buffers";
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
            std::stringstream msg;
            msg << "hipMemcpy failure";
            if(skip_runtime_fails)
                throw ROCFFT_SKIP{msg.str()};
            else
                throw ROCFFT_FAIL{msg.str()};
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

template <class Tparams>
void compute_fft_data(Tparams&              params,
                      std::vector<hostbuf>& fft_input,
                      std::vector<hostbuf>& fft_output)
{
    // Call hipGetLastError to reset any errors
    // returned by previous HIP runtime API calls.
    hipError_t hip_status = hipGetLastError();

    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    // Make sure FFT buffers fit in device memory
    check_problem_fits_device_memory(params, verbose);

    auto ibuffer_sizes = params.ibuffer_sizes();
    auto obuffer_sizes = params.obuffer_sizes();

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
        std::stringstream msg;
        msg << "Work buffer allocation failed with size: " << params.workbuffersize;
        if(skip_runtime_fails)
            throw ROCFFT_SKIP{msg.str()};
        else
            throw ROCFFT_FAIL{msg.str()};
    }
    ASSERT_EQ(plan_status, fft_status_success) << "plan creation failed";

    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
        if(hip_status != hipSuccess)
        {
            std::stringstream msg;
            msg << "hipMalloc failure for input buffer " << i << " size " << ibuffer_sizes[i] << "("
                << bytes_to_GiB(ibuffer_sizes[i]) << " GiB)"
                << " with code " << hipError_to_string(hip_status);
            ++n_hip_failures;
            if(skip_runtime_fails)
                throw ROCFFT_SKIP{msg.str()};
            else
                throw ROCFFT_FAIL{msg.str()};
        }
        pibuffer[i] = ibuffer[i].data();
    }

    // allocation counts in elements, ibuffer_sizes is in bytes
    auto ibuffer_sizes_elems = ibuffer_sizes;
    for(auto& buf : ibuffer_sizes_elems)
        buf /= var_size<size_t>(params.precision, params.itype);

    fft_input = allocate_host_buffer(params.precision, params.itype, ibuffer_sizes_elems);

// generate input based on either cpu/gpu
#ifdef USE_HIPRAND
    // generate the input directly on the gpu
    params.compute_input(ibuffer);

    // Copy input to CPU
    for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
    {
        hip_status = hipMemcpy(fft_input.at(idx).data(),
                               ibuffer[idx].data(),
                               ibuffer_sizes[idx],
                               hipMemcpyDeviceToHost);
        if(hip_status != hipSuccess)
        {
            std::stringstream msg;
            msg << "hipMemcpy failure with error " << hip_status;

            ++n_hip_failures;
            if(skip_runtime_fails)
                throw ROCFFT_SKIP{msg.str()};
            else
                throw ROCFFT_FAIL{msg.str()};
        }
    }
#else
    // generate the input on the cpu
    params.compute_input(fft_input);

    // Copy input to GPU
    for(unsigned int idx = 0; idx < fft_input.size(); ++idx)
    {
        hip_status = hipMemcpy(ibuffer[idx].data(),
                               fft_input.at(idx).data(),
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
#endif

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
                std::stringstream msg;
                msg << "hipMalloc failure for output buffer " << i << " size " << obuffer_sizes[i]
                    << "(" << bytes_to_GiB(obuffer_sizes[i]) << " GiB)"
                    << " with code " << hipError_to_string(hip_status);
                if(skip_runtime_fails)
                    throw ROCFFT_SKIP{msg.str()};
                else
                    throw ROCFFT_FAIL{msg.str()};
            }
        }
    }
    pobuffer.resize(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    // execute GPU transform
    fft_output = allocate_host_buffer(params.precision, params.otype, params.osize);

    execute_fft(params, pibuffer, pobuffer, *obuffer, fft_output);
}

template <class Tfloat, class Tparams>
inline void bitwise_repro_impl(Tparams& params, Tparams& params_comp)
{
    std::vector<hostbuf> fft_input, fft_output;
    compute_fft_data(params, fft_input, fft_output);

    auto ibuffer_hash_in = hash_input(rocfft_precision_from_fftparams(params.precision),
                                      params.ilength(),
                                      params.istride,
                                      params.idist,
                                      rocfft_array_type_from_fftparams(params.itype),
                                      params.nbatch);

    auto ibuffer_hash_out = hash_output<size_t>();
    compute_hash(fft_input, ibuffer_hash_in, ibuffer_hash_out);

    auto obuffer_hash_in  = hash_input(rocfft_precision_from_fftparams(params.precision),
                                      params.olength(),
                                      params.ostride,
                                      params.odist,
                                      rocfft_array_type_from_fftparams(params.otype),
                                      params.nbatch);
    auto obuffer_hash_out = hash_output<size_t>();
    compute_hash(fft_output, obuffer_hash_in, obuffer_hash_out);

    if(params_comp.token().compare(params.token()) == 0)
    {
        std::stringstream msg;
        msg << "FFT input tokens are identical";
        throw ROCFFT_SKIP{msg.str()};
    }

    std::vector<hostbuf> fft_input_comp, fft_output_comp;
    compute_fft_data(params_comp, fft_input_comp, fft_output_comp);

    auto obuffer_hash_in_comp  = hash_input(rocfft_precision_from_fftparams(params_comp.precision),
                                           params_comp.olength(),
                                           params_comp.ostride,
                                           params_comp.odist,
                                           rocfft_array_type_from_fftparams(params_comp.otype),
                                           params_comp.nbatch);
    auto obuffer_hash_out_comp = hash_output<size_t>();
    compute_hash(fft_output_comp, obuffer_hash_in_comp, obuffer_hash_out_comp);

    params.free();
    params_comp.free();

    // FFT params are not identical and, therefore,
    // must also have different fft outputs.
    ASSERT_FALSE(obuffer_hash_out_comp == obuffer_hash_out)
        << "Different FFT params have the same output hash.";
}

template <class Tfloat, class Tparams>
inline void bitwise_repro_impl(Tparams& params)
{
    std::vector<hostbuf> fft_input, fft_output;
    compute_fft_data(params, fft_input, fft_output);

    auto ibuffer_hash_in  = hash_input(rocfft_precision_from_fftparams(params.precision),
                                      params.ilength(),
                                      params.istride,
                                      params.idist,
                                      rocfft_array_type_from_fftparams(params.itype),
                                      params.nbatch);
    auto ibuffer_hash_out = hash_output<size_t>();
    compute_hash(fft_input, ibuffer_hash_in, ibuffer_hash_out);

    auto obuffer_hash_in  = hash_input(rocfft_precision_from_fftparams(params.precision),
                                      params.olength(),
                                      params.ostride,
                                      params.odist,
                                      rocfft_array_type_from_fftparams(params.otype),
                                      params.nbatch);
    auto obuffer_hash_out = hash_output<size_t>();
    compute_hash(fft_output, obuffer_hash_in, obuffer_hash_out);

    bool hash_entry_found, hash_valid;

    if(verbose)
    {
        std::cout << "input buffer hash:  (" << ibuffer_hash_out.buffer_real << ","
                  << ibuffer_hash_out.buffer_imag << ")" << std::endl;
        std::cout << "output buffer hash: (" << obuffer_hash_out.buffer_real << ","
                  << obuffer_hash_out.buffer_imag << ")" << std::endl;
    }

    repro_db->check_hash_valid(
        ibuffer_hash_out, obuffer_hash_out, params.token(), hash_entry_found, hash_valid);

    params.free();

    if(hash_entry_found)
        ASSERT_TRUE(hash_valid) << "FFT result is not bitwise reproducible.";
    else
    {
        std::stringstream msg;
        msg << "FFT result entry added to the repro-db file. Previously stored reference entry not "
               "found. \n";
        throw ROCFFT_SKIP{msg.str()};
    }
}

inline void bitwise_repro(rocfft_params& params)
{
    switch(params.precision)
    {
    case fft_precision_half:
        bitwise_repro_impl<rocfft_fp16, rocfft_params>(params);
        break;
    case fft_precision_single:
        bitwise_repro_impl<float, rocfft_params>(params);
        break;
    case fft_precision_double:
        bitwise_repro_impl<double, rocfft_params>(params);
        break;
    }
}

inline void bitwise_repro(rocfft_params& params, rocfft_params& params_comp)
{
    switch(params.precision)
    {
    case fft_precision_half:
        bitwise_repro_impl<rocfft_fp16, rocfft_params>(params, params_comp);
        break;
    case fft_precision_single:
        bitwise_repro_impl<float, rocfft_params>(params, params_comp);
        break;
    case fft_precision_double:
        bitwise_repro_impl<double, rocfft_params>(params, params_comp);
        break;
    }
}

#endif // BITWISE_REPRO_TEST_H
