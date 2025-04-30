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

// This file allows one to run tests multiple different rocFFT libraries at the same time.
// This allows one to randomize the execution order for better a better experimental setup
// which produces fewer type 1 errors where one incorrectly rejects the null hypothesis.

#include <algorithm>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <math.h>
#include <vector>

#ifdef WIN32
#include <windows.h>
// psapi.h requires windows.h to be included first
#include <psapi.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

#include "../../shared/CLI11.hpp"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"
#include "bench.h"
#include "rocfft/rocfft.h"

#ifdef WIN32
typedef HMODULE ROCFFT_LIB;
#else
typedef void* ROCFFT_LIB;
#endif

// Load the rocfft library
ROCFFT_LIB rocfft_lib_load(const std::string& path)
{
#ifdef WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_LAZY);
#endif
}

// Return a string describing the error loading rocfft
const char* rocfft_lib_load_error()
{
#ifdef WIN32
    // just return the error number
    static std::string error_str;
    error_str = std::to_string(GetLastError());
    return error_str.c_str();
#else
    return dlerror();
#endif
}

// Get symbol from rocfft lib
void* rocfft_lib_symbol(ROCFFT_LIB libhandle, const char* sym)
{
#ifdef WIN32
    return reinterpret_cast<void*>(GetProcAddress(libhandle, sym));
#else
    return dlsym(libhandle, sym);
#endif
}

void rocfft_lib_close(ROCFFT_LIB libhandle)
{
#ifdef WIN32
    FreeLibrary(libhandle);
#else
    dlclose(libhandle);
#endif
}

// Given a libhandle from dload, return a plan to a rocFFT plan with the given parameters.
rocfft_plan make_plan(ROCFFT_LIB libhandle, const fft_params& params)
{
    auto procfft_setup = (decltype(&rocfft_setup))rocfft_lib_symbol(libhandle, "rocfft_setup");
    if(procfft_setup == NULL)
        throw rocfft_runtime_error("rocfft_setup failed");
    auto procfft_plan_description_create
        = (decltype(&rocfft_plan_description_create))rocfft_lib_symbol(
            libhandle, "rocfft_plan_description_create");
    auto procfft_plan_description_destroy
        = (decltype(&rocfft_plan_description_destroy))rocfft_lib_symbol(
            libhandle, "rocfft_plan_description_destroy");
    auto procfft_plan_description_set_data_layout
        = (decltype(&rocfft_plan_description_set_data_layout))rocfft_lib_symbol(
            libhandle, "rocfft_plan_description_set_data_layout");
    auto procfft_plan_create
        = (decltype(&rocfft_plan_create))rocfft_lib_symbol(libhandle, "rocfft_plan_create");

    procfft_setup();

    rocfft_plan_description desc = NULL;
    LIB_V_THROW(procfft_plan_description_create(&desc), "rocfft_plan_description_create failed");
    LIB_V_THROW(
        procfft_plan_description_set_data_layout(desc,
                                                 rocfft_array_type_from_fftparams(params.itype),
                                                 rocfft_array_type_from_fftparams(params.otype),
                                                 params.ioffset.data(),
                                                 params.ooffset.data(),
                                                 params.istride.size(),
                                                 params.istride.data(),
                                                 params.idist,
                                                 params.ostride.size(),
                                                 params.ostride.data(),
                                                 params.odist),
        "rocfft_plan_description_data_layout failed");
    rocfft_plan plan = NULL;

    LIB_V_THROW(procfft_plan_create(&plan,
                                    rocfft_result_placement_from_fftparams(params.placement),
                                    rocfft_transform_type_from_fftparams(params.transform_type),
                                    rocfft_precision_from_fftparams(params.precision),
                                    params.length.size(),
                                    params.length.data(),
                                    params.nbatch,
                                    desc),
                "rocfft_plan_create failed");

    LIB_V_THROW(procfft_plan_description_destroy(desc), "rocfft_plan_description_destroy failed");

    return plan;
}

// Given a libhandle from dload and a rocFFT plan, destroy the plan.
void destroy_plan(ROCFFT_LIB libhandle, rocfft_plan& plan)
{
    auto procfft_plan_destroy
        = (decltype(&rocfft_plan_destroy))rocfft_lib_symbol(libhandle, "rocfft_plan_destroy");

    LIB_V_THROW(procfft_plan_destroy(plan), "rocfft_plan_destroy failed");

    auto procfft_cleanup
        = (decltype(&rocfft_cleanup))rocfft_lib_symbol(libhandle, "rocfft_cleanup");
    if(procfft_cleanup)
        LIB_V_THROW(procfft_cleanup(), "rocfft_cleanup failed");
}

// Given a libhandle from dload and a rocFFT execution info structure, destroy the info.
void destroy_info(ROCFFT_LIB libhandle, rocfft_execution_info& info)
{
    auto procfft_execution_info_destroy
        = (decltype(&rocfft_execution_info_destroy))rocfft_lib_symbol(
            libhandle, "rocfft_execution_info_destroy");
    LIB_V_THROW(procfft_execution_info_destroy(info), "rocfft_execution_info_destroy failed");
}

// Given a libhandle from dload, and a corresponding rocFFT plan, return how much work
// buffer is required.
size_t get_wbuffersize(ROCFFT_LIB libhandle, const rocfft_plan& plan)
{
    auto procfft_plan_get_work_buffer_size
        = (decltype(&rocfft_plan_get_work_buffer_size))rocfft_lib_symbol(
            libhandle, "rocfft_plan_get_work_buffer_size");

    // Get the buffersize
    size_t workBufferSize = 0;
    LIB_V_THROW(procfft_plan_get_work_buffer_size(plan, &workBufferSize),
                "rocfft_plan_get_work_buffer_size failed");

    return workBufferSize;
}

// Given a libhandle from dload and a corresponding rocFFT plan, print the plan information.
void show_plan(ROCFFT_LIB libhandle, const rocfft_plan& plan)
{
    auto procfft_plan_get_print
        = (decltype(&rocfft_plan_get_print))rocfft_lib_symbol(libhandle, "rocfft_plan_get_print");

    LIB_V_THROW(procfft_plan_get_print(plan), "rocfft_plan_get_print failed");
}

// FIXME: doc
rocfft_execution_info make_execinfo(ROCFFT_LIB libhandle)
{
    auto procfft_execution_info_create = (decltype(&rocfft_execution_info_create))rocfft_lib_symbol(
        libhandle, "rocfft_execution_info_create");
    rocfft_execution_info info = NULL;
    LIB_V_THROW(procfft_execution_info_create(&info), "rocfft_execution_info_create failed");
    return info;
}

// FIXME: doc
void set_work_buffer(const ROCFFT_LIB&      libhandle,
                     rocfft_execution_info& info,
                     const size_t           wbuffersize,
                     void*                  wbuffer)
{
    if(wbuffersize > 0 && wbuffer != NULL)
    {
        auto procfft_execution_info_set_work_buffer
            = (decltype(&rocfft_execution_info_set_work_buffer))rocfft_lib_symbol(
                libhandle, "rocfft_execution_info_set_work_buffer");
        LIB_V_THROW(procfft_execution_info_set_work_buffer(info, wbuffer, wbuffersize),
                    "rocfft_execution_info_set_work_buffer failed");
    }
}

// Given a libhandle from dload and a corresponding rocFFT plan and execution info,
// execute a transform on the given input and output buffers and return the kernel
// execution time.
float run_plan(
    ROCFFT_LIB libhandle, rocfft_plan plan, rocfft_execution_info info, void** in, void** out)
{
    auto procfft_execute
        = (decltype(&rocfft_execute))rocfft_lib_symbol(libhandle, "rocfft_execute");

    hipEvent_wrapper_t start, stop;
    start.alloc();
    stop.alloc();

    HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

    auto rcfft = procfft_execute(plan, in, out, info);

    HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
    HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

    if(rcfft != rocfft_status_success)
    {
        throw std::runtime_error("execution failed");
    }

    float time;
    HIP_V_THROW(hipEventElapsedTime(&time, start, stop), "hipEventElapsedTime failed");
    return time;
}

std::pair<ROCFFT_LIB, rocfft_plan> create_handleplan(const std::string& libstring,
                                                     const fft_params&  params)
{
    auto libhandle = rocfft_lib_load(libstring);
    if(libhandle == NULL)
    {
        std::stringstream ss;
        ss << "Failed to open " << libstring << ", error: " << rocfft_lib_load_error();
        throw std::runtime_error(ss.str());
    }

    auto plan = make_plan(libhandle, params);

    return std::make_pair(libhandle, plan);
}

int main(int argc, char* argv[])
{
    // Control output verbosity:
    int verbose{};

    // number of GPUs to use:
    int ngpus{};

    // hip Device number for running tests:
    int deviceId{};

    // Ignore runtime failures.
    // eg: hipMalloc failing when there isn't enough free vram.
    bool ignore_hip_runtime_failures{true};

    // Number of performance trial samples:
    int ntrial{};

    // Bool to specify whether the libs are loaded in forward or forward+reverse order.
    int reverse{};

    // Test sequence choice:
    int test_sequence{};

    // Vector of test target libraries
    std::vector<std::string> lib_strings;

    // FFT parameters:
    fft_params params;

    // input/output FFT grids
    std::vector<unsigned int> ingrid;
    std::vector<unsigned int> outgrid;

    // Token string to fully specify fft params.
    std::string token;

    CLI::App app{"dyna-rocfft-bench command line options"};

    // Declare the supported options. Some option pointers are declared to track passed opts.
    // FIXME: version needs to be implemented
    app.add_flag("--version",
                 "Print queryable version information from the rocfft library and exit");
    app.add_flag("--reverse", reverse, "Load libs in forward and reverse order")->default_val(1);
    app.add_option(
           "--sequence", test_sequence, "Test sequence:\n0) random\n1) alternating\n2) sequential")
        ->default_val(0);
    app.add_option("--lib", lib_strings, "Set test target library full path (appendable)");
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

    // Check if all the provided libraries are actually there:
    for(const auto& lib_string : lib_strings)
    {
        if(!std::filesystem::exists(lib_string))
        {
            std::cerr << "Error: lib " << lib_string << " does not exist\n";
            return EXIT_FAILURE;
        }
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
    }
    std::cout << std::flush;

    // Set GPU for single-device FFT computation
    rocfft_scoped_device dev(deviceId);

    params.validate();

    if(!params.valid(verbose))
    {
        throw rocfft_runtime_error("Invalid parameters, add --verbose=1 for detail");
    }

    std::cout << "Token: " << params.token() << std::endl;
    if(verbose)
    {
        std::cout << params.str() << std::endl;
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
    init_bench_input(params, ibricks, ibuffer, ibuffer_cpu, is_host_gen);

    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        pibuffer.push_back(ibuffer[i].data());
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

    // Execution times for loaded libraries:
    std::vector<std::vector<double>> time(lib_strings.size());

    // If we are doing a reverse-run, then we need two ntrials; otherwise, just one.
    std::vector<int> ntrial_runs;
    if(reverse == 0)
    {
        ntrial_runs.push_back(ntrial);
    }
    else
    {
        ntrial_runs.push_back((ntrial + 1) / 2);
        ntrial_runs.push_back(ntrial / 2);
    }

    for(size_t ridx = 0; ridx < ntrial_runs.size(); ++ridx)
    {

        std::vector<std::pair<size_t, std::string>> index_lib_string;
        for(size_t i = 0; i < lib_strings.size(); ++i)
        {
            index_lib_string.push_back(std::make_pair(i, lib_strings[i]));
        }
        if(ridx == 1)
        {
            std::reverse(index_lib_string.begin(), index_lib_string.end());
        }

        // Create the handles to the libs and the associated fft plans.
        std::vector<ROCFFT_LIB>  handle;
        std::vector<rocfft_plan> plan;
        // Allocate the work buffer: just one, big enough for any dloaded library.
        std::vector<rocfft_execution_info> info;
        size_t                             wbuffer_size = 0;
        for(unsigned int idx = 0; idx < lib_strings.size(); ++idx)
        {
            std::cout << idx << ": " << lib_strings[idx] << "\n";
            auto libhandle = rocfft_lib_load(lib_strings[idx]);
            if(libhandle == NULL)
            {
                std::cout << "Failed to open " << lib_strings[idx]
                          << ", error: " << rocfft_lib_load_error() << "\n";
                return 1;
            }
            handle.push_back(libhandle);
            plan.push_back(make_plan(handle[idx], params));
            show_plan(handle[idx], plan[idx]);
            wbuffer_size = std::max(wbuffer_size, get_wbuffersize(handle[idx], plan[idx]));
            info.push_back(make_execinfo(handle[idx]));
        }

        std::cout << "Work buffer size: " << wbuffer_size << std::endl;

        if(!vram_fits_problem(raw_vram_footprint + wbuffer_size, free))
        {
            std::cout << "SKIPPED: Problem size (" << raw_vram_footprint << " + " << +wbuffer_size
                      << " = " << raw_vram_footprint + wbuffer_size
                      << " ) data too large for device.\n";
            return EXIT_SUCCESS;
        }

        gpubuf wbuffer;
        if(wbuffer_size)
        {
            try
            {
                HIP_V_THROW(wbuffer.alloc(wbuffer_size), "Creating intermediate Buffer failed");
            }
            catch(rocfft_hip_runtime_error)
            {
                return ignore_hip_runtime_failures ? EXIT_SUCCESS : EXIT_FAILURE;
            }
        }

        // Associate the work buffer to the individual libraries:
        for(unsigned int idx = 0; idx < lib_strings.size(); ++idx)
        {
            set_work_buffer(handle[idx], info[idx], wbuffer_size, wbuffer.data());
        }

        // Run the plan using its associated rocFFT library:
        for(unsigned int idx = 0; idx < handle.size(); ++idx)
        {
            try
            {
                run_plan(handle[idx], plan[idx], info[idx], pibuffer.data(), pobuffer.data());
            }
            catch(rocfft_hip_runtime_error)
            {
                return ignore_hip_runtime_failures ? EXIT_SUCCESS : EXIT_FAILURE;
            }
        }

        std::vector<int> testcase(ntrial_runs[ridx] * index_lib_string.size());

        switch(test_sequence)
        {
        case 0:
        {
            // Random order:
            for(int itrial = 0; itrial < ntrial_runs[ridx]; ++itrial)
            {
                for(size_t ilib = 0; ilib < index_lib_string.size(); ++ilib)
                {

                    testcase[index_lib_string.size() * itrial + ilib] = ilib;
                }
            }
            std::random_device rd;
            std::mt19937       g(rd());
            std::shuffle(testcase.begin(), testcase.end(), g);
            break;
        }
        case 1:
            // Alternating order:
            for(int itrial = 0; itrial < ntrial_runs[ridx]; ++itrial)
            {
                for(size_t ilib = 0; ilib < index_lib_string.size(); ++ilib)
                {
                    testcase[index_lib_string.size() * itrial + ilib] = ilib;
                }
            }
            break;
        case 2:
            // Sequential order:
            for(int itrial = 0; itrial < ntrial_runs[ridx]; ++itrial)
            {
                for(size_t ilib = 0; ilib < index_lib_string.size(); ++ilib)
                {
                    testcase[ilib * ntrial + itrial] = ilib;
                }
            }
            break;
        default:
            throw std::runtime_error("Invalid test sequence choice.");
        }

        if(verbose > 3)
        {
            std::cout << "Test case order:";
            for(const auto val : testcase)
                std::cout << " " << val;
            std::cout << "\n";
        }

        std::cout << "Running the tests...\n";

        for(size_t itest = 0; itest < testcase.size(); ++itest)
        {
            const int tidx = testcase[itest];

            if(verbose > 3)
            {
                std::cout << "running test case " << tidx << " with lib "
                          << index_lib_string[tidx].second << "\n";
            }

#ifdef USE_HIPRAND
            if(!is_host_gen)
                params.compute_input(ibuffer);
#endif
            if(is_host_gen)
            {
                for(unsigned int bidx = 0; bidx < ibuffer_cpu.size(); ++bidx)
                {
                    try
                    {
                        HIP_V_THROW(hipMemcpy(pibuffer[bidx],
                                              ibuffer_cpu[bidx].data(),
                                              ibuffer_cpu[bidx].size(),
                                              hipMemcpyHostToDevice),
                                    "hipMemcpy failed");
                    }
                    catch(rocfft_hip_runtime_error)
                    {
                        return ignore_hip_runtime_failures ? EXIT_SUCCESS : EXIT_FAILURE;
                    }
                }
            }

            // Run the plan using its associated rocFFT library:
            try
            {
                time[tidx].push_back(run_plan(
                    handle[tidx], plan[tidx], info[tidx], pibuffer.data(), pobuffer.data()));
            }
            catch(rocfft_hip_runtime_error)
            {
                return ignore_hip_runtime_failures ? EXIT_SUCCESS : EXIT_FAILURE;
            }

            if(verbose > 2)
            {
                auto output = allocate_host_buffer(params.precision, params.otype, params.osize);
                for(unsigned int iout = 0; iout < output.size(); ++iout)
                {
                    try
                    {
                        HIP_V_THROW(hipMemcpy(output[iout].data(),
                                              pobuffer[iout],
                                              output[iout].size(),
                                              hipMemcpyDeviceToHost),
                                    "hipMemcpy failed");
                    }
                    catch(rocfft_hip_runtime_error)
                    {
                        return ignore_hip_runtime_failures ? EXIT_SUCCESS : EXIT_FAILURE;
                    }
                }
                std::cout << "GPU output:\n";
                params.print_obuffer(output);
            }
        }

        // Clean up:
        for(unsigned int hidx = 0; hidx < handle.size(); ++hidx)
        {
            destroy_info(handle[hidx], info[hidx]);
            destroy_plan(handle[hidx], plan[hidx]);
            rocfft_lib_close(handle[hidx]);
        }
    }

    std::cout << "Execution times in ms:\n";
    for(unsigned int idx = 0; idx < time.size(); ++idx)
    {
        std::cout << "\nExecution gpu time:";
        for(auto& i : time[idx])
        {
            std::cout << " " << i;
        }
        std::cout << " ms" << std::endl;
    }

    return EXIT_SUCCESS;
}
