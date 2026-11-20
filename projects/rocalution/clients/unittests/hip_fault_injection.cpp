/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "hip_fault_injection.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

#include "base/hip/backend_hip.hpp"

namespace
{
    using rocalution_unit::HipFault;

    std::atomic<HipFault> active_fault{HipFault::none};
    std::atomic<int>      injected_count{0};

    const void* rocalution_hip_base()
    {
        static const void* base = [] {
            Dl_info info{};
            dladdr(reinterpret_cast<void*>(&rocalution::rocalution_init_hip), &info);
            return info.dli_fbase;
        }();

        return base;
    }

    // Whether the call that returns to return_address is to be failed
    bool inject(HipFault fault, const void* return_address)
    {
        if(active_fault.load() != fault)
        {
            return false;
        }

        Dl_info info{};
        if(dladdr(return_address, &info) == 0 || info.dli_fbase != rocalution_hip_base())
        {
            return false;
        }

        ++injected_count;
        return true;
    }

    template <typename F>
    F next_definition(const char* name)
    {
        void* symbol = dlsym(RTLD_NEXT, name);

        if(symbol == nullptr)
        {
            std::fprintf(stderr, "hip_fault_injection: %s not found\n", name);
            std::abort();
        }

        return reinterpret_cast<F>(symbol);
    }
}

#define CALLER __builtin_return_address(0)
#define FORWARD(function, type) static const auto real_##function = next_definition<type>(#function)

namespace rocalution_unit
{
    ScopedHipFault::ScopedHipFault(HipFault fault)
    {
        injected_count = 0;
        active_fault   = fault;
    }

    ScopedHipFault::~ScopedHipFault()
    {
        active_fault = HipFault::none;
    }

    int ScopedHipFault::injected() const
    {
        return injected_count.load();
    }
}

extern "C" {

hipError_t hipGetDeviceCount(int* count)
{
    FORWARD(hipGetDeviceCount, hipError_t(*)(int*));

    if(inject(HipFault::device_count_fails, CALLER))
    {
        return hipErrorNoDevice;
    }

    if(inject(HipFault::no_devices, CALLER))
    {
        *count = 0;
        return hipSuccess;
    }

    return real_hipGetDeviceCount(count);
}

hipError_t hipSetDevice(int device)
{
    FORWARD(hipSetDevice, hipError_t(*)(int));

    if(inject(HipFault::set_device_fails, CALLER))
    {
        return hipErrorUnknown;
    }

    if(inject(HipFault::context_in_use, CALLER))
    {
        return hipErrorContextAlreadyInUse;
    }

    return real_hipSetDevice(device);
}

// hipGetDeviceProperties is a macro or an inline function, depending on the HIP
// version, and both call the exported hipGetDevicePropertiesR0600
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int device)
{
    FORWARD(hipGetDevicePropertiesR0600, hipError_t(*)(hipDeviceProp_tR0600*, int));

    hipError_t status = real_hipGetDevicePropertiesR0600(prop, device);

    if(inject(HipFault::low_compute_capability, CALLER))
    {
        prop->major = 2;
    }

    if(inject(HipFault::no_registers_per_block, CALLER))
    {
        prop->regsPerBlock = 0;
    }

    return status;
}

rocblas_status rocblas_create_handle(rocblas_handle* handle)
{
    FORWARD(rocblas_create_handle, rocblas_status(*)(rocblas_handle*));

    if(inject(HipFault::blas_handle_fails, CALLER))
    {
        return rocblas_status_internal_error;
    }

    return real_rocblas_create_handle(handle);
}

rocblas_status rocblas_destroy_handle(rocblas_handle handle)
{
    FORWARD(rocblas_destroy_handle, rocblas_status(*)(rocblas_handle));

    rocblas_status status = real_rocblas_destroy_handle(handle);

    if(inject(HipFault::blas_destroy_fails, CALLER))
    {
        return rocblas_status_internal_error;
    }

    return status;
}

rocsparse_status rocsparse_create_handle(rocsparse_handle* handle)
{
    FORWARD(rocsparse_create_handle, rocsparse_status(*)(rocsparse_handle*));

    if(inject(HipFault::sparse_handle_fails, CALLER))
    {
        return rocsparse_status_internal_error;
    }

    return real_rocsparse_create_handle(handle);
}

rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle)
{
    FORWARD(rocsparse_destroy_handle, rocsparse_status(*)(rocsparse_handle));

    rocsparse_status status = real_rocsparse_destroy_handle(handle);

    if(inject(HipFault::sparse_destroy_fails, CALLER))
    {
        return rocsparse_status_internal_error;
    }

    return status;
}

} // extern "C"
