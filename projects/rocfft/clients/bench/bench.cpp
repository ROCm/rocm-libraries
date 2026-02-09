// Copyright (C) 2016 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstddef>
#include <iostream>
#include <sstream>

#include "../../shared/CLI11.hpp"
#include "../../shared/arithmetic.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"
#include "bench.h"
#include "rocfft/rocfft.h"

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // Control output verbosity:
    int verbose{};

    // number of GPUs to use:
    int ngpus{};

    // hip Device number for running tests:
    int deviceId{};

    // Ignore runtime failures.
    // eg: hipMalloc failing when there isn't enough free vram.
    bool ignore_hip_runtime_failures{true};

    // Number of performance trial samples
    int ntrial{};

    // FFT parameters:
    rocfft_params params;

    // input/output FFT grids
    std::vector<unsigned int> ingrid;
    std::vector<unsigned int> outgrid;

    // Token string to fully specify fft params.
    std::string token;

    CLI::App app{"rocfft-bench command line options"};

    // Declare the supported options. Some option pointers are declared to track passed opts.
    app.add_flag("--version", "Print queryable version information from the rocfft library")
        ->each([](const std::string&) {
            char v[256];
            rocfft_get_version_string(v, 256);
            std::cout << "version " << v << std::endl;
            return EXIT_SUCCESS;
        });

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
                     "rocFFT's auto-allocation behavior: \"on\", \"off\", or \"default\"")
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

    non_token->add_option("--ngpus", ngpus, "Number of GPUs to use")
        ->default_val(1)
        ->check(CLI::NonNegativeNumber);

    // define multi-GPU grids for FFT computation,
    CLI::Option* opt_ingrid
        = non_token->add_option("--ingrid", ingrid, "Single-process grid of GPUs at input")
              ->expected(1, 3)
              ->needs("--ngpus");

    CLI::Option* opt_outgrid
        = non_token->add_option("--outgrid", outgrid, "Single-process grid of GPUs at output")
              ->expected(1, 3)
              ->needs("--ngpus");

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

    app.add_flag("--ignore_runtime_failures,!--no-ignore_runtime_failures",
                 ignore_hip_runtime_failures,
                 "Ignore hip runtime failures");

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

        std::cout << std::flush;
    }
    else // generate token
    {
        if(ngpus > 1)
        {
            // set default GPU grids in case none were given
            params.set_default_grid(ngpus, ingrid, outgrid);

            // split the problem among ngpus
            params.mp_lib = fft_params::fft_mp_lib_none;

            int localDeviceCount = 0;
            if(hipGetDeviceCount(&localDeviceCount) != hipSuccess)
            {
                throw std::runtime_error("hipGetDeviceCount failed");
            }

            // start with all-ones in grids
            std::vector<unsigned int> input_grid(params.length.size() + 1, 1);
            std::vector<unsigned int> output_grid(params.length.size() + 1, 1);

            // create input and output grids and distribute it according to user requirements
            std::copy(ingrid.begin(), ingrid.end(), input_grid.begin() + 1);
            std::copy(outgrid.begin(), outgrid.end(), output_grid.begin() + 1);

            params.distribute_input(localDeviceCount, input_grid);
            params.distribute_output(localDeviceCount, output_grid);
        }

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

        if(*opt_ingrid || !ingrid.empty())
        {
            std::cout << "input  grid:";
            for(auto& i : ingrid)
                std::cout << " " << i;
            std::cout << "\n";
        }

        if(*opt_outgrid || !outgrid.empty())
        {
            std::cout << "output grid:";
            for(auto& i : outgrid)
                std::cout << " " << i;
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;

    rocfft_setup();

    // Set GPU for single-device FFT computation
    rocfft_scoped_device dev(deviceId);

    params.validate();

    if(!params.valid(verbose))
    {
        throw std::runtime_error("Invalid parameters, add --verbose=1 for detail");
    }

    std::cout << "Token: " << params.token() << std::endl;
    if(verbose)
    {
        std::cout << params.str(" ") << std::endl;
    }

    // Check free and total available memory:
    size_t free  = 0;
    size_t total = 0;
    try
    {
        HIP_V_THROW(hipMemGetInfo(&free, &total), "hipMemGetInfo failed");
    }
    catch(rocfft_hip_runtime_error)
    {
        return ignore_hip_runtime_failures ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);
    if(!vram_fits_problem(raw_vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << raw_vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    auto ret = params.create_plan();
    if(ret != fft_status_success)
        LIB_V_THROW(rocfft_status_failure, "Plan creation failed");

    // GPU input buffer:
    std::vector<gpubuf> ibuffer;
    std::vector<void*>  pibuffer;
    // CPU-side input buffer
    std::vector<hostbuf> ibuffer_cpu;

    auto is_host_gen = (params.igen == fft_input_generator_host
                        || params.igen == fft_input_random_generator_host);

    auto ibricks = get_input_bricks(params);
    auto obricks = get_output_bricks(params);

    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = nullptr;
    alloc_bench_bricks(
        params, ibricks, obricks, ibuffer, obuffer_data, obuffer, ibuffer_cpu, is_host_gen);

    pibuffer.resize(ibuffer.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        pibuffer[i] = ibuffer[i].data();
    }

    // print input if requested
    if(verbose > 1)
    {
        if(is_host_gen)
        {
            // data is already on host
            params.print_ibuffer(ibuffer_cpu);
        }
        else
        {
            print_device_buffer(params, ibuffer, true);
        }
    }

    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    init_bench_input(params, ibricks, ibuffer, ibuffer_cpu, is_host_gen);

    // Execute a warm-up call
    params.execute(pibuffer.data(), pobuffer.data());

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_wrapper_t start, stop;
    start.alloc();
    stop.alloc();
    for(unsigned int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
        // Create input at every iteration to avoid overflow
        if(is_host_gen)
        {
            copy_host_input_to_dev(ibuffer_cpu, ibuffer);
        }
        else
        {
            init_bench_input(params, ibricks, ibuffer, ibuffer_cpu, is_host_gen);
        }

        HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

        params.execute(pibuffer.data(), pobuffer.data());

        HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
        HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

        float time;
        HIP_V_THROW(hipEventElapsedTime(&time, start, stop), "hipEventElapsedTime failed");
        gpu_time[itrial] = time;

        // Print result after FFT transform
        if(verbose > 2)
        {
            print_device_buffer(params, *obuffer, false);
        }
    }

    std::cout << "\nExecution gpu time:";
    for(const auto& i : gpu_time)
    {
        std::cout << " " << i;
    }
    std::cout << " ms" << std::endl;

    std::cout << "Execution gflops:  ";
    const double totsize = product(params.length.begin(), params.length.end());
    const double k
        = ((params.itype == fft_array_type_real) || (params.otype == fft_array_type_real)) ? 2.5
                                                                                           : 5.0;
    const double opscount = (double)params.nbatch * k * totsize * log(totsize) / log(2.0);
    for(const auto& i : gpu_time)
    {
        std::cout << " " << opscount / (1e6 * i);
    }
    std::cout << std::endl;

    rocfft_cleanup();
}
