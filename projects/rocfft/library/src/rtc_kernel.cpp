// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rtc_kernel.h"
#include "../../shared/array_predicate.h"
#include "../../shared/device_properties.h"
#include "../../shared/environment.h"
#include "device/generator/stockham_gen.h"

#include "device/kernel-generator-embed.h"
#include "kernel_launch.h"
#include "logging.h"
#include "rtc_bluestein_kernel.h"
#include "rtc_cache.h"
#include "rtc_realcomplex_kernel.h"
#include "rtc_stockham_kernel.h"
#include "rtc_transpose_kernel.h"
#include "tree_node.h"

RTCKernel::RTCKernel(const std::string&       kernel_name,
                     const std::vector<char>& code,
                     dim3                     gridDim,
                     dim3                     blockDim)
    : gridDim(gridDim)
    , blockDim(blockDim)
    , kernel_name(kernel_name)
{
#ifndef ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS
    // if we're only compiling, no need to actually load the code objects
    if(rocfft_getenv("ROCFFT_INTERNAL_COMPILE_ONLY") == "1")
        return;
#endif
    if(hipModuleLoadData(&module, code.data()) != hipSuccess)
        throw std::runtime_error("failed to load module for " + kernel_name);

    if(hipModuleGetFunction(&kernel, module, kernel_name.c_str()) != hipSuccess)
        throw std::runtime_error("failed to get function " + kernel_name);
}

#ifndef ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS
void RTCKernel::launch(DeviceCallIn& data, const hipDeviceProp_t& deviceProp)
{
    RTCKernelArgs kargs = get_launch_args(data);

    const auto& gp = data.gridParam;

    launch(kargs,
           {gp.b_x, gp.b_y, gp.b_z},
           {gp.wgs_x, gp.wgs_y, gp.wgs_z},
           gp.lds_bytes,
           deviceProp,
           data.rocfft_stream);
}
#endif

void RTCKernel::launch(RTCKernelArgs&         kargs,
                       dim3                   gridDim,
                       dim3                   blockDim,
                       unsigned int           lds_bytes,
                       const hipDeviceProp_t& deviceProp,
                       hipStream_t            stream)
{
    launch_limits_check(kernel_name, gridDim, blockDim, deviceProp);
    auto  size     = kargs.size_bytes();
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      kargs.data(),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &size,
                      HIP_LAUNCH_PARAM_END};

#ifndef ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS
    if(LOG_PLAN_ENABLED())
    {
        int        max_blocks_per_sm;
        hipError_t ret = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kernel, blockDim.x * blockDim.y * blockDim.z, lds_bytes);
        rocfft_ostream* kernelplan_stream = LogSingleton::GetInstance().GetPlanOS();
        if(ret == hipSuccess)
            *kernelplan_stream << "Kernel occupancy: " << max_blocks_per_sm << std::endl;
        else
            *kernelplan_stream << "Can not retrieve occupancy info." << std::endl;
    }
#endif

    if(hipModuleLaunchKernel(kernel,
                             gridDim.x,
                             gridDim.y,
                             gridDim.z,
                             blockDim.x,
                             blockDim.y,
                             blockDim.z,
                             lds_bytes,
                             stream,
                             nullptr,
                             config)
       != hipSuccess)
        throw std::runtime_error("hipModuleLaunchKernel failure");
}

bool RTCKernel::get_occupancy(dim3 blockDim, unsigned int lds_bytes, int& occupancy)
{
    hipError_t ret = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, blockDim.x * blockDim.y * blockDim.z, lds_bytes);

    return ret == hipSuccess;
}

std::shared_future<std::unique_ptr<RTCKernel>>
    RTCKernel::runtime_compile(const LeafNode&    node,
                               const std::string& gpu_arch,
                               std::string&       kernel_name,
                               bool               enable_callbacks)
{
#ifndef ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS
    int deviceId = 0;
    if(hipGetDevice(&deviceId) != hipSuccess)
    {
        throw std::runtime_error("failed to get device");
    }

    RTCGenerator generator;
    // try each type of generator until one is valid
    generator = RTCKernelStockham::generate_from_node(node, gpu_arch, enable_callbacks);
    if(!generator.valid())
        generator = RTCKernelTranspose::generate_from_node(node, gpu_arch, enable_callbacks);
    if(!generator.valid())
        generator = RTCKernelRealComplex::generate_from_node(node, gpu_arch, enable_callbacks);
    if(!generator.valid())
        generator = RTCKernelRealComplexEven::generate_from_node(node, gpu_arch, enable_callbacks);
    if(!generator.valid())
        generator = RTCKernelRealComplexEvenTranspose::generate_from_node(
            node, gpu_arch, enable_callbacks);
    if(!generator.valid())
        generator = RTCKernelBluesteinSingle::generate_from_node(node, gpu_arch, enable_callbacks);
    if(!generator.valid())
        generator = RTCKernelBluesteinMulti::generate_from_node(node, gpu_arch, enable_callbacks);

    if(generator.valid())
    {
        kernel_name = generator.generate_name();

        auto compile = [=](std::promise<std::unique_ptr<RTCKernel>> compile_promise) {
            if(hipSetDevice(deviceId) != hipSuccess)
            {
                compile_promise.set_exception(
                    std::make_exception_ptr(std::runtime_error("failed to set device")));
            }
            try
            {
                std::vector<char> code = RTCCache::cached_compile(
                    kernel_name, gpu_arch, generator.generate_src, generator_sum());
                compile_promise.set_value(generator.construct_rtckernel(
                    kernel_name, code, generator.gridDim, generator.blockDim));
            }
            catch(std::exception& e)
            {
                if(LOG_RTC_ENABLED())
                    (*LogSingleton::GetInstance().GetRTCOS()) << e.what() << std::endl;
                compile_promise.set_exception(std::current_exception());
            }
        };

        // compile to code object
        std::promise<std::unique_ptr<RTCKernel>>       compile_promise;
        std::shared_future<std::unique_ptr<RTCKernel>> compile_future
            = compile_promise.get_future();
        std::thread compile_thread(compile, std::move(compile_promise));
        // we'll wait for the future so the thread can continue
        // without being managed by this object
        compile_thread.detach();
        return compile_future;
    }
    // a pre-compiled rtc-stockham-kernel goes here
    else if(generator.is_pre_compiled())
    {
        kernel_name = generator.generate_name();
    }
#endif
    // kernel harness being generated or no kernel found, return
    // null RTCKernel
    std::promise<std::unique_ptr<RTCKernel>> p;
    p.set_value(nullptr);
    return p.get_future();
}
