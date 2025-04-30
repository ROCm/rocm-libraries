// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_BENCHMARK_UTILS_HPP_
#define ROCRAND_BENCHMARK_UTILS_HPP_

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <string>

#include "custom_csv_formater.hpp"
#include <fstream>

#define HIP_CHECK(condition)                                                            \
    do                                                                                  \
    {                                                                                   \
        hipError_t error_ = condition;                                                  \
        if(error_ != hipSuccess)                                                        \
        {                                                                               \
            std::cout << "HIP error: " << error_ << " line: " << __LINE__ << std::endl; \
            exit(error_);                                                               \
        }                                                                               \
    }                                                                                   \
    while(0)

inline void add_common_benchmark_info()
{
    auto str = [](const std::string& name, const std::string& val)
    { benchmark::AddCustomContext(name, val); };

    auto num = [](const std::string& name, const int& value)
    { benchmark::AddCustomContext(name, std::to_string(value)); };

    auto dim2 = [num](const std::string& name, const int* values)
    {
        num(name + "_x", values[0]);
        num(name + "_y", values[1]);
    };

    auto dim3 = [num, dim2](const std::string& name, const int* values)
    {
        dim2(name, values);
        num(name + "_z", values[2]);
    };

    auto num_size_t = [](const std::string& name, const size_t& value)
    { benchmark::AddCustomContext(name, std::to_string(value)); };

    int runtime_version;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    num("hip_runtime_version", runtime_version);

    // On the NVIDIA platform not all members of this struct will be written to
    // Zero-initialize to avoid referencing dangling memory
    hipDeviceProp_t devProp{};
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    str("hdp_name", devProp.name);
    num_size_t("hdp_total_global_mem", devProp.totalGlobalMem);
    num_size_t("hdp_shared_mem_per_block", devProp.sharedMemPerBlock);
    num("hdp_regs_per_block", devProp.regsPerBlock);
    num("hdp_warp_size", devProp.warpSize);
    num("hdp_max_threads_per_block", devProp.maxThreadsPerBlock);
    dim3("hdp_max_threads_dim", devProp.maxThreadsDim);
    dim3("hdp_max_grid_size", devProp.maxGridSize);
    num("hdp_clock_rate", devProp.clockRate);
    num("hdp_memory_clock_rate", devProp.memoryClockRate);
    num("hdp_memory_bus_width", devProp.memoryBusWidth);
    num_size_t("hdp_total_const_mem", devProp.totalConstMem);
    num("hdp_major", devProp.major);
    num("hdp_minor", devProp.minor);
    num("hdp_multi_processor_count", devProp.multiProcessorCount);
    num("hdp_l2_cache_size", devProp.l2CacheSize);
    num_size_t("hdp_max_threads_per_multiprocessor", devProp.maxThreadsPerMultiProcessor);
    num("hdp_compute_mode", devProp.computeMode);
    num("hdp_clock_instruction_rate", devProp.clockInstructionRate);
    num("hdp_concurrent_kernels", devProp.concurrentKernels);
    num("hdp_pci_domain_id", devProp.pciDomainID);
    num("hdp_pci_bus_id", devProp.pciBusID);
    num("hdp_pci_device_id", devProp.pciDeviceID);
    num("hdp_max_shared_memory_per_multi_processor", devProp.maxSharedMemoryPerMultiProcessor);
    num("hdp_is_multi_gpu_board", devProp.isMultiGpuBoard);
    num("hdp_can_map_host_memory", devProp.canMapHostMemory);
    str("hdp_gcn_arch_name", devProp.gcnArchName);
    num("hdp_integrated", devProp.integrated);
    num("hdp_cooperative_launch", devProp.cooperativeLaunch);
    num("hdp_cooperative_multi_device_launch", devProp.cooperativeMultiDeviceLaunch);
    num_size_t("hdp_max_texture_1d_linear", devProp.maxTexture1DLinear);
    num("hdp_max_texture_1d", devProp.maxTexture1D);
    dim2("hdp_max_texture_2d", devProp.maxTexture2D);
    dim3("hdp_max_texture_3d", devProp.maxTexture3D);
    num_size_t("hdp_mem_pitch", devProp.memPitch);
    num("hdp_texture_alignment", devProp.textureAlignment);
    num("hdp_texture_pitch_alignment", devProp.texturePitchAlignment);
    num("hdp_kernel_exec_timeout_enabled", devProp.kernelExecTimeoutEnabled);
    num("hdp_ecc_enabled", devProp.ECCEnabled);
    num("hdp_tcc_driver", devProp.tccDriver);
    num("hdp_cooperative_multi_device_unmatched_func", devProp.cooperativeMultiDeviceUnmatchedFunc);
    num("hdp_cooperative_multi_device_unmatched_grid_dim",
        devProp.cooperativeMultiDeviceUnmatchedGridDim);
    num("hdp_cooperative_multi_device_unmatched_block_dim",
        devProp.cooperativeMultiDeviceUnmatchedBlockDim);
    num("hdp_cooperative_multi_device_unmatched_shared_mem",
        devProp.cooperativeMultiDeviceUnmatchedSharedMem);
    num("hdp_is_large_bar", devProp.isLargeBar);
    num("hdp_asic_revision", devProp.asicRevision);
    num("hdp_managed_memory", devProp.managedMemory);
    num("hdp_direct_managed_mem_access_from_host", devProp.directManagedMemAccessFromHost);
    num("hdp_concurrent_managed_access", devProp.concurrentManagedAccess);
    num("hdp_pageable_memory_access", devProp.pageableMemoryAccess);
    num("hdp_pageable_memory_access_uses_host_page_tables",
        devProp.pageableMemoryAccessUsesHostPageTables);

    const auto arch = devProp.arch;
    num("hdp_arch_has_global_int32_atomics", arch.hasGlobalInt32Atomics);
    num("hdp_arch_has_global_float_atomic_exch", arch.hasGlobalFloatAtomicExch);
    num("hdp_arch_has_shared_int32_atomics", arch.hasSharedInt32Atomics);
    num("hdp_arch_has_shared_float_atomic_exch", arch.hasSharedFloatAtomicExch);
    num("hdp_arch_has_float_atomic_add", arch.hasFloatAtomicAdd);
    num("hdp_arch_has_global_int64_atomics", arch.hasGlobalInt64Atomics);
    num("hdp_arch_has_shared_int64_atomics", arch.hasSharedInt64Atomics);
    num("hdp_arch_has_doubles", arch.hasDoubles);
    num("hdp_arch_has_warp_vote", arch.hasWarpVote);
    num("hdp_arch_has_warp_ballot", arch.hasWarpBallot);
    num("hdp_arch_has_warp_shuffle", arch.hasWarpShuffle);
    num("hdp_arch_has_funnel_shift", arch.hasFunnelShift);
    num("hdp_arch_has_thread_fence_system", arch.hasThreadFenceSystem);
    num("hdp_arch_has_sync_threads_ext", arch.hasSyncThreadsExt);
    num("hdp_arch_has_surface_funcs", arch.hasSurfaceFuncs);
    num("hdp_arch_has_3d_grid", arch.has3dGrid);
    num("hdp_arch_has_dynamic_parallelism", arch.hasDynamicParallelism);
}

inline size_t next_power2(size_t x)
{
    size_t power = 1;
    while(power < x)
    {
        power *= 2;
    }
    return power;
}

inline benchmark::BenchmarkReporter* getConsoleReporter(const std::string format)
{
    benchmark::BenchmarkReporter* reporter;
    if(format == "csv")
    {
        static benchmark::customCSVReporter csv_reporter;
        csv_reporter.SetErrorStream(&std::cout);
        csv_reporter.SetOutputStream(&std::cout);
        reporter = &csv_reporter;
    }
    else if(format == "json")
    {
        static benchmark::customCSVReporter json_reporter;
        json_reporter.SetErrorStream(&std::cout);
        json_reporter.SetOutputStream(&std::cout);
        reporter = &json_reporter;
    }
    else
    {
        static benchmark::ConsoleReporter terminal_reporter;
        terminal_reporter.SetErrorStream(&std::cout);
        terminal_reporter.SetOutputStream(&std::cout);
        reporter = &terminal_reporter;
    }

    return reporter;
}

inline benchmark::BenchmarkReporter* getOutFileReporter(const std::string format)
{
    benchmark::BenchmarkReporter* reporter = nullptr;
    std::ofstream                 output_file;
    if(format == "csv")
    {
        static benchmark::customCSVReporter csv_reporter;
        csv_reporter.SetOutputStream(&output_file);
        csv_reporter.SetErrorStream(&output_file);
        reporter = &csv_reporter;
    }
    else if(format == "json")
    {
        static benchmark::JSONReporter json_reporter;
        json_reporter.SetOutputStream(&output_file);
        json_reporter.SetErrorStream(&output_file);
        reporter = &json_reporter;
    }
    else if(format == "console")
    {
        static benchmark::ConsoleReporter console_reporter;
        console_reporter.SetOutputStream(&output_file);
        console_reporter.SetErrorStream(&output_file);
        reporter = &console_reporter;
    }

    return reporter;
}

inline void getFormats(const int    argc,
                       char*        argv[],
                       std::string& outFormat,
                       std::string& filter,
                       std::string& consoleFormat)
{
    for(int i = 1; i < argc; i++)
    {
        std::string input(argv[i]);
        int         equalPos = input.find("=");

        if(equalPos < 0)
            continue;

        std::string arg    = std::string(input.begin() + 2, input.begin() + equalPos);
        std::string argVal = std::string(input.begin() + 1 + equalPos, input.end());

        if(arg == "benchmark_out_format")
            outFormat = argVal;
        else if(arg == "benchmark_filter")
            filter = argVal;
        else if(arg == "benchmark_format")
            consoleFormat = argVal;
    }
}

#endif // ROCRAND_BENCHMARK_UTILS_HPP_
