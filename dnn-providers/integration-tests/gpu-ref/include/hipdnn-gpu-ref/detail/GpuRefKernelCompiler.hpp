// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_gpu_ref::detail
{

class CompiledKernel
{
public:
    CompiledKernel(const std::string& sourceName,
                   const std::vector<std::string>& compileDefines,
                   const std::string& functionName);
    ~CompiledKernel();

    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;
    CompiledKernel(CompiledKernel&&) = delete;
    CompiledKernel& operator=(CompiledKernel&&) = delete;

    hipFunction_t function() const
    {
        return _function;
    }

private:
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
    std::vector<char> _binary;
};

class GpuRefKernelCompiler
{
public:
    static GpuRefKernelCompiler& instance();

    const CompiledKernel& getOrCompile(const std::string& sourceName,
                                       const std::vector<std::string>& compileDefines,
                                       const std::string& functionName);

    GpuRefKernelCompiler(const GpuRefKernelCompiler&) = delete;
    GpuRefKernelCompiler& operator=(const GpuRefKernelCompiler&) = delete;

private:
    GpuRefKernelCompiler() = default;

    std::mutex _mutex;
    std::unordered_map<std::string, std::unique_ptr<CompiledKernel>> _cache;
};

} // namespace hipdnn_gpu_ref::detail
