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

#ifndef ROCFFT_RTC_H
#define ROCFFT_RTC_H

#include "rocfft/rocfft.h"
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "../../../shared/rocfft_complex.h"
#include "../device/kernels/callback.h"
#include "rtc_generator.h"

struct DeviceCallIn;
class TreeNode;
class LeafNode;
struct GridParam;

// Helper class that handles alignment of kernel arguments
class RTCKernelArgs
{
public:
    RTCKernelArgs() = default;
    void append_ptr(const void* ptr)
    {
        append(&ptr, sizeof(void*));
    }
    void append_size_t(size_t s)
    {
        append(&s, sizeof(size_t));
    }
    void append_unsigned_int(unsigned int i)
    {
        append(&i, sizeof(unsigned int));
    }
    void append_int(int i)
    {
        append(&i, sizeof(int));
    }
    void append_double(double d)
    {
        append(&d, sizeof(double));
    }
    void append_float(float f)
    {
        append(&f, sizeof(float));
    }
    void append_half(rocfft_fp16 f)
    {
        append(&f, sizeof(rocfft_fp16));
    }
    template <typename T>
    void append_struct(const T& data)
    {
        append(&data, sizeof(T), 8);
    }

    size_t size_bytes() const
    {
        return buf.size();
    }
    void* data()
    {
        return buf.data();
    }

private:
    void append(const void* src, size_t nbytes, size_t align = 0)
    {
        // values need to be aligned to their width (i.e. 8-byte values
        // need 8-byte alignment, 4-byte needs 4-byte alignment)
        if(align == 0)
            align = nbytes;

        size_t oldsize = buf.size();
        size_t padding = oldsize % align ? align - (oldsize % align) : 0;
        buf.resize(oldsize + padding + nbytes);
        std::copy_n(static_cast<const char*>(src), nbytes, buf.begin() + oldsize + padding);
    }

    std::vector<char> buf;
};

// Base class for a runtime compiled kernel.  Subclassed for
// different kernel types that each have their own details about how
// to be launched.
struct RTCKernel
{
    // try to compile kernel for node, and attach compiled kernel to
    // node if successful.  returns nullptr if there is no matching
    // supported scheme + problem size.  throws runtime_error on
    // error.
    static std::shared_future<std::unique_ptr<RTCKernel>>
        runtime_compile(const LeafNode&    node,
                        const std::string& gpu_arch,
                        std::string&       kernel_name,
                        bool               enable_callbacks = false);

    // take already-compiled code object and prepare to launch the
    // named kernel
    RTCKernel(const std::string&       kernel_name,
              const std::vector<char>& code,
              dim3                     gridDim  = {},
              dim3                     blockDim = {});

    virtual ~RTCKernel()
    {
        kernel = nullptr;
        (void)hipModuleUnload(module);
        module = nullptr;
    }

    // disallow copies, since we expect this to be managed by smart ptr
    RTCKernel(const RTCKernel&) = delete;
    RTCKernel(RTCKernel&&)      = delete;

    void operator=(const RTCKernel&) = delete;

    // normal launch from within rocFFT execution plan
    void launch(DeviceCallIn& data, const hipDeviceProp_t& deviceProp);
    // direct launch with kernel args
    void launch(RTCKernelArgs&         kargs,
                dim3                   gridDim,
                dim3                   blockDim,
                unsigned int           lds_bytes,
                const hipDeviceProp_t& deviceProp,
                hipStream_t            stream = nullptr);

    // normal launch from within rocFFT execution plan
    bool get_occupancy(dim3 blockDim, unsigned int lds_bytes, int& occupancy);

#ifndef ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS
    // Subclasses implement this - each kernel type has different
    // parameters
    virtual RTCKernelArgs get_launch_args(DeviceCallIn& data) = 0;
#endif

    // function to construct the correct RTCKernel object, given a kernel name and its compiled code
    using rtckernel_construct_t = std::function<std::unique_ptr<RTCKernel>(
        const std::string&, const std::vector<char>&, dim3, dim3)>;

    // grid parameters for this kernel.  may be set by runtime
    // compilation, if compilation of this kernel type knows how to.
    // Otherwise, TreeNode::SetupGridParam_internal will do it
    // later.
    dim3 gridDim;
    dim3 blockDim;

    std::string kernel_name;

protected:
#ifndef ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS
    struct RTCGenerator
    {
        kernel_name_gen_t     generate_name;
        kernel_src_gen_t      generate_src;
        rtckernel_construct_t construct_rtckernel;

        virtual bool valid() const
        {
            return generate_name && generate_src && construct_rtckernel;
        }
        // generator is the correct type, but kernel is already compiled
        virtual bool is_pre_compiled() const
        {
            return false;
        }

        // if known at compile time, the grid parameters of the kernel
        // to launch with
        dim3 gridDim;
        dim3 blockDim;
    };
#endif

    hipModule_t   module = nullptr;
    hipFunction_t kernel = nullptr;
};

#ifndef ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS

// helper functions to construct pieces of RTC kernel names
static const char* rtc_array_type_name(rocfft_array_type type)
{
    // hermitian is the same as complex in terms of generated code,
    // so give them the same names in kernels
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        return "_CI";
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        return "_CP";
    case rocfft_array_type_real:
        return "_R";
    default:
        return "_UN";
    }
}

static const char* rtc_precision_name(rocfft_precision precision)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return "_sp";
    case rocfft_precision_double:
        return "_dp";
    case rocfft_precision_half:
        return "_half";
    }
}

static const char* rtc_precision_type_decl(rocfft_precision precision, bool is_complex = true)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return is_complex ? "typedef rocfft_complex<float> scalar_type;\n"
                          : "typedef float scalar_type;\n";
    case rocfft_precision_double:
        return is_complex ? "typedef rocfft_complex<double> scalar_type;\n"
                          : "typedef double scalar_type;\n";
    case rocfft_precision_half:
        return is_complex ? "typedef rocfft_complex<rocfft_fp16> scalar_type;\n"
                          : "typedef rocfft_fp16 scalar_type;\n";
    }
}

static const char* rtc_cbtype_name(CallbackType cbtype)
{
    switch(cbtype)
    {
    case CallbackType::NONE:
        return "";
    case CallbackType::USER_LOAD_STORE:
        return "_CB";
    case CallbackType::USER_LOAD_STORE_R2C:
        return "_CBr2c";
    case CallbackType::USER_LOAD_STORE_C2R:
        return "_CBc2r";
    }
}

// realDataAsComplex is true if we're treating real data as complex
// (in an even-length real-complex FFT)
static const std::string rtc_const_cbtype_decl(CallbackType cbtype)
{
    switch(cbtype)
    {
    case CallbackType::NONE:
        return "static const CallbackType cbtype = CallbackType::NONE;\n";
    case CallbackType::USER_LOAD_STORE:
        return "static const CallbackType cbtype = CallbackType::USER_LOAD_STORE;\n";
    case CallbackType::USER_LOAD_STORE_R2C:
        return "static const CallbackType cbtype = CallbackType::USER_LOAD_STORE_R2C;\n";
    case CallbackType::USER_LOAD_STORE_C2R:
        return "static const CallbackType cbtype = CallbackType::USER_LOAD_STORE_C2R;\n";
    }
}
#endif

#endif
