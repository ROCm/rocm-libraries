// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>

#include "bench.h"

#include "../../shared/CLI11.hpp"
#include "../../shared/client_except.h"
#include "../../shared/gpubuf.h"

// initialize static class member of hipfft_params
std::vector<gpubuf> hipfft_params::externally_managed_workareas = std::vector<gpubuf>();

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // Control output verbosity:
    int verbose{};

    // hip Device number for running tests:
    int deviceId{};

    // Number of performance trial samples
    int ntrial{};

    // FFT parameters:
    hipfft_params params;

    // Token string to fully specify fft params.
    std::string token;

    // Declare the supported options.
    CLI::App app{"hipfft-bench command line options"};

    // Declare the supported options. Some option pointers are declared to track passed opts.
    app.add_flag("-v, --version", "Print queryable version information from the rocfft library")
        // ->each([](const std::string&) {
        //     char v[256];
        //     rocfft_get_version_string(v, 256);
        //     std::cout << "version " << v << std::endl;
        //     std::exit(EXIT_SUCCESS);
        // })
        ;

    CLI::Option* opt_token
        = app.add_option("--token", token, "Token to read FFT params from")->default_val("");
    // Group together options that conflict with --token
    auto* non_token = app.add_option_group("Token Conflict", "Options excluded by --token");
    non_token
        ->add_flag("--double", "Double precision transform (deprecated: use --precision double)")
        ->each([&](const std::string&) { params.precision = fft_precision_double; });
    non_token->excludes(opt_token);
    non_token
        ->add_option("-t, --transformType",
                     params.transform_type,
                     "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
                     "forward\n3) real inverse")
        ->default_val(fft_transform_type_complex_forward);
    non_token
        ->add_option("--auto_allocation",
                     params.auto_allocate,
                     "hipFFT's auto-allocation behavior: \"on\", \"off\", or \"default\"")
        ->default_val("default");
    non_token
        ->add_option(
            "--precision", params.precision, "Transform precision: single (default), double, half")
        ->excludes("--double");
    CLI::Option* opt_not_in_place
        = non_token->add_flag("-o, --notInPlace", "Not in-place FFT transform (default: in-place)")
              ->each([&](const std::string&) { params.placement = fft_placement_notinplace; });
    non_token
        ->add_option("--itype",
                     params.itype,
                     "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);
    non_token
        ->add_option("--otype",
                     params.otype,
                     "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);
    CLI::Option* opt_length
        = non_token->add_option("--length", params.length, "Lengths")->required()->expected(1, 3);
    non_token
        ->add_option("-b, --batchSize",
                     params.nbatch,
                     "If this value is greater than one, arrays will be used")
        ->default_val(1);
    CLI::Option* opt_istride = non_token->add_option("--istride", params.istride, "Input strides");
    CLI::Option* opt_ostride = non_token->add_option("--ostride", params.ostride, "Output strides");
    non_token->add_option("--idist", params.idist, "Logical distance between input batches")
        ->default_val(0)
        ->each([&](const std::string& val) { std::cout << "idist: " << val << "\n"; });
    non_token->add_option("--odist", params.odist, "Logical distance between output batches")
        ->default_val(0)
        ->each([&](const std::string& val) { std::cout << "odist: " << val << "\n"; });
    CLI::Option* opt_ioffset = non_token->add_option("--ioffset", params.ioffset, "Input offset");
    CLI::Option* opt_ooffset = non_token->add_option("--ooffset", params.ooffset, "Output offset");

    app.add_option("--device", deviceId, "Select a specific device id")->default_val(0);
    app.add_option("--verbose", verbose, "Control output verbosity")->default_val(0);
    app.add_option("-N, --ntrial", ntrial, "Trial size for the problem")
        ->default_val(1)
        ->each([&](const std::string& val) {
            std::cout << "Running profile with " << val << " samples\n";
        });
    // Default value is set in fft_params.h based on if device-side PRNG was enabled.
    app.add_option("-g, --inputGen",
                   params.igen,
                   "Input data generation:\n0) PRNG sequence (device)\n"
                   "1) PRNG sequence (host)\n"
                   "2) linearly-spaced sequence (device)\n"
                   "3) linearly-spaced sequence (host)");
    app.add_option("--isize", params.isize, "Logical size of input buffer");
    app.add_option("--osize", params.osize, "Logical size of output buffer");
    app.add_option("--scalefactor", params.scale_factor, "Scale factor to apply to output");

    // Parse args and catch any errors here
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    if(!token.empty())
    {
        std::cout << "Reading fft params from token:\n" << token << std::endl;

        try
        {
            params.from_token(token);
        }
        catch(...)
        {
            std::cout << "Unable to parse token." << std::endl;
            return EXIT_FAILURE;
        }
    }
    else
    {
        if(*opt_not_in_place)
        {
            std::cout << "out-of-place\n";
        }
        else
        {
            std::cout << "in-place\n";
        }

        if(*opt_length)
        {
            std::cout << "length:";
            for(auto& i : params.length)
                std::cout << " " << i;
            std::cout << "\n";
        }

        if(*opt_istride)
        {
            std::cout << "istride:";
            for(auto& i : params.istride)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(*opt_ostride)
        {
            std::cout << "ostride:";
            for(auto& i : params.ostride)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(*opt_ioffset)
        {
            std::cout << "ioffset:";
            for(auto& i : params.ioffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(*opt_ooffset)
        {
            std::cout << "ooffset:";
            for(auto& i : params.ooffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
    }

    std::cout << std::flush;

    // Fixme: set the device id properly after the IDs are synced
    // bewteen hip runtime and rocm-smi.
    // HIP_V_THROW(hipSetDevice(deviceId), "set device failed!");

    params.validate();

    if(!params.valid(verbose))
    {
        throw std::runtime_error("Invalid parameters, add --verbose=1 for detail");
    }

    std::cout << "Token: " << params.token() << std::endl;
    if(verbose)
    {
        std::cout << params.str() << std::endl;
        std::cout << "Token: " << params.token() << std::endl;
    }

    // Check free and total available memory:
    size_t free  = 0;
    size_t total = 0;

    if(hipMemGetInfo(&free, &total) != hipSuccess)
        throw std::runtime_error("hipMemGetInfo failed");

    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);
    if(!vram_fits_problem(raw_vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << raw_vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    size_t vram_footprint = 0;

    try
    {
        vram_footprint = params.vram_footprint();
    }
    catch(ROCFFT_SKIP& e)
    {
        std::cout << "SKIPPED: " << e.msg << "\n";
        return EXIT_SUCCESS;
    }
    if(!vram_fits_problem(vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    // Create plans:
    auto ret = params.create_plan();
    if(ret != fft_status_success)
        throw std::runtime_error("Plan creation failed");

    hipError_t hip_rt;

    // GPU input buffer:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_rt = ibuffer[i].alloc(ibuffer_sizes[i]);
        if(hip_rt != hipSuccess)
            throw std::runtime_error("Creating input Buffer failed");
        pibuffer[i] = ibuffer[i].data();
    }

    // CPU-side input buffer
    std::vector<hostbuf> ibuffer_cpu;

    auto is_host_gen = (params.igen == fft_input_generator_host
                        || params.igen == fft_input_random_generator_host);

#ifdef USE_HIPRAND
    if(!is_host_gen)
    {
        // Input data:
        params.compute_input(ibuffer);

        if(verbose > 1)
        {
            // Copy input to CPU
            ibuffer_cpu = allocate_host_buffer(params.precision, params.itype, params.isize);
            for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
            {
                HIP_V_THROW(hipMemcpy(ibuffer_cpu.at(idx).data(),
                                      ibuffer[idx].data(),
                                      ibuffer_sizes[idx],
                                      hipMemcpyDeviceToHost),
                            "hipMemcpy failed");
            }

            std::cout << "GPU input:\n";
            params.print_ibuffer(ibuffer_cpu);
        }
    }
#endif
    if(is_host_gen)
    {
        // Input data:
        ibuffer_cpu = allocate_host_buffer(params.precision, params.itype, params.isize);
        params.compute_input(ibuffer_cpu);

        if(verbose > 1)
        {
            std::cout << "GPU input:\n";
            params.print_ibuffer(ibuffer_cpu);
        }

        for(unsigned int idx = 0; idx < ibuffer_cpu.size(); ++idx)
        {
            HIP_V_THROW(hipMemcpy(pibuffer[idx],
                                  ibuffer_cpu[idx].data(),
                                  ibuffer_cpu[idx].size(),
                                  hipMemcpyHostToDevice),
                        "hipMemcpy failed");
        }
    }

    // GPU output buffer:
    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
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
            hip_rt = obuffer_data[i].alloc(obuffer_sizes[i]);
            if(hip_rt != hipSuccess)
                throw std::runtime_error("Creating output Buffer failed");
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    auto res = params.execute(pibuffer.data(), pobuffer.data());
    if(res != fft_status_success)
        throw std::runtime_error("Execution failed");

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    hip_rt = hipEventCreate(&start);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");

    hip_rt = hipEventCreate(&stop);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");

    for(size_t itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
#ifdef USE_HIPRAND
        // Compute input on default device
        if(!is_host_gen)
            params.compute_input(ibuffer);
#endif
        if(is_host_gen)
        {
            for(unsigned int idx = 0; idx < ibuffer_cpu.size(); ++idx)
            {
                HIP_V_THROW(hipMemcpy(pibuffer[idx],
                                      ibuffer_cpu[idx].data(),
                                      ibuffer_cpu[idx].size(),
                                      hipMemcpyHostToDevice),
                            "hipMemcpy failed");
            }
        }

        hip_rt = hipEventRecord(start);
        if(hip_rt != hipSuccess)
            throw std::runtime_error("hipEventRecord failed");

        res = params.execute(pibuffer.data(), pobuffer.data());

        hip_rt = hipEventRecord(stop);
        if(hip_rt != hipSuccess)
            throw std::runtime_error("hipEventRecord failed");

        hip_rt = hipEventSynchronize(stop);
        if(hip_rt != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");

        if(res != fft_status_success)
            throw std::runtime_error("Execution failed");

        float time;
        hip_rt = hipEventElapsedTime(&time, start, stop);
        if(hip_rt != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failed");

        gpu_time[itrial] = time;

        if(verbose > 2)
        {
            auto output = allocate_host_buffer(params.precision, params.otype, params.osize);
            for(unsigned int idx = 0; idx < output.size(); ++idx)
            {
                hip_rt = hipMemcpy(
                    output[idx].data(), pobuffer[idx], output[idx].size(), hipMemcpyDeviceToHost);
                if(hip_rt != hipSuccess)
                    throw std::runtime_error("hipMemcpy failed");
            }
            std::cout << "GPU output:\n";
            params.print_obuffer(output);
        }
    }

    std::cout << "\nExecution gpu time:";
    for(const auto& i : gpu_time)
    {
        std::cout << " " << i;
    }
    std::cout << " ms" << std::endl;
}
