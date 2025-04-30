/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "handle.h"
#include "kernel_arguments.hpp"
#include <hip/hip_runtime.h>

#include <map>
#include <mutex>

class SolutionAdapter
{
public:
    SolutionAdapter();
    SolutionAdapter(std::string const& name);
    ~SolutionAdapter();
    std::string name() const
    {
        return m_name;
    }
    hipError_t    loadLibrary(std::string const& path);
    hipError_t    loadCodeObject(const _rocsparselt_handle* handle, std::string const& name);
    hipError_t    loadCodeObject(const _rocsparselt_handle* handle,
                                 const void*                image,
                                 std::string const&         name);
    hipError_t    loadCodeObjectBytes(const _rocsparselt_handle*  handle,
                                      std::vector<uint8_t> const& bytes,
                                      std::string const&          name);
    hipError_t    launchKernel(const _rocsparselt_handle* handle, KernelInvocation const& kernel);
    hipError_t    launchKernel(const _rocsparselt_handle* handle,
                               KernelInvocation const&    kernel,
                               hipStream_t                stream,
                               hipEvent_t                 startEvent,
                               hipEvent_t                 stopEvent,
                               int                        iter = 1);
    hipError_t    launchKernels(const _rocsparselt_handle*           handle,
                                std::vector<KernelInvocation> const& kernels);
    hipError_t    launchKernels(const _rocsparselt_handle*           handle,
                                std::vector<KernelInvocation> const& kernels,
                                hipStream_t                          stream,
                                hipEvent_t                           startEvent,
                                hipEvent_t                           stopEvent);
    hipError_t    launchKernels(const _rocsparselt_handle*           handle,
                                std::vector<KernelInvocation> const& kernels,
                                hipStream_t                          stream,
                                std::vector<hipEvent_t> const&       startEvents,
                                std::vector<hipEvent_t> const&       stopEvents);
    hipError_t    initKernel(std::string const& name);
    size_t        getKernelCounts(std::string const& category);
    KernelParams* getKernelParams(std::string const& category);

private:
    using function_table = std::map<std::string, void*>;

    hipError_t getKernel(hipFunction_t& rv, std::string const& name);
    std::mutex m_access;
    std::unordered_map<std::string, hipModule_t>   m_modules;
    std::unordered_map<std::string, hipFunction_t> m_kernels;
    std::string                                    m_name = "HipSolutionAdapter";
    std::vector<std::string>                       m_loadedModuleNames;
    std::vector<void*>                             m_lib_handles;
    std::vector<function_table>                    m_lib_functions;
    std::vector<std::string>                       m_loadedLibNames;
    friend std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
};
std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr);
