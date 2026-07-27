// Copyright (C) 2023, 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_LOAD_STORE_OPS_H
#define ROCFFT_LOAD_STORE_OPS_H

#include "../../../shared/rocfft_hip.h"
#include "../device/kernels/callback.h"
#include <hip/hip_runtime_api.h>
#include <hip/linker_types.h>
#include <optional>
#include <string>
#include <vector>

class RTCKernelArgs;
class Function;
class TreeNode;

struct rocfft_spirv_cb_t
{
    rocfft_spirv_cb_t() = default;
    void set(const char* _symbol_name,
             const void* _bitcode_data,
             size_t      _bitcode_len_bytes,
             void**      _cb_data)
    {
        symbol_name       = _symbol_name;
        bitcode_data      = _bitcode_data;
        bitcode_len_bytes = _bitcode_len_bytes;
        // caller gives us a (potentially temporary) array of
        // void*'s, so copy them to an owned vector
        auto device_count = rocfft_scoped_device::device_count();
        cb_data.resize(device_count);
        if(_cb_data)
            std::copy(_cb_data, _cb_data + device_count, cb_data.begin());
    }
    bool enabled() const
    {
        return symbol_name && bitcode_data && bitcode_len_bytes;
    }

    // return a stringified hash of the callback, or empty string if
    // the callback was not specified
    std::string get_hash() const;

    // Non-owning pointers to data provided by users
    const char* symbol_name       = nullptr;
    const void* bitcode_data      = nullptr;
    size_t      bitcode_len_bytes = 0;
    // one pointer per visible HIP device
    std::vector<void*> cb_data;
};

struct LoadOps
{
    LoadOps() = default;

    // user-provided spir-v load callback
    rocfft_spirv_cb_t spirv_cb;

    // returns true if some load operation is enabled
    bool enabled() const
    {
        return spirv_cb.enabled();
    }

    bool has_spirv() const
    {
        return spirv_cb.enabled();
    }

    std::string forward_decls(const CallbackType cbtype, const char* scalar_type) const
    {
        std::string ret;
        if(spirv_cb.enabled())
        {
            // real forward FFT kernel works with complex elements but
            // was passed a real-type callback for loading
            if(cbtype == CallbackType::USER_LOAD_STORE_R2C)
            {
                ret += std::string("extern \"C\" __device__ real_type_t<") + scalar_type + "> "
                       + spirv_cb.symbol_name + "(real_type_t<" + scalar_type
                       + ">*, size_t, void*, void*);\n";
            }
            else
            {
                ret += std::string("extern \"C\" __device__ ") + scalar_type + " "
                       + spirv_cb.symbol_name + "(" + scalar_type + "*, size_t, void*, void*);\n";
            }
            // declare a constant name for the load callback as well
            ret += "__device__ auto load_cb_jit_fn = ";
            ret += spirv_cb.symbol_name;
            ret += ";\n";
        }
        return ret;
    }

    std::string name_suffix() const
    {
        std::string ret;

        if(spirv_cb.enabled())
        {
            ret += "_load" + spirv_cb.get_hash();
        }
        return ret;
    }

    // append kernel arguments to implement the operations defined in
    // *this
    void append_args(RTCKernelArgs& kargs, TreeNode& node) const;
    // transform a global function to implement operations defined in *this
    Function add_ops(const Function& f) const;

    template <typename Tstream>
    void print(Tstream& os, const std::string& indent) const
    {
    }
};

struct hipLink_wrapper_t
{
    hipLink_wrapper_t()
    {
        if(hipLinkCreate(0, nullptr, nullptr, &state) != hipSuccess)
            throw std::runtime_error("failed to link create");
    }

    ~hipLink_wrapper_t()
    {
        (void)hipLinkDestroy(state);
        state = nullptr;
    }

    void link(void* bitcode_data, size_t bitcode_len_bytes, const char* filename)
    {
        // hip/cu link APIs accept non-const data, even though they
        // have no reason to own or modify the data
        if(hipLinkAddData(state,
                          hipJitInputSpirv,
                          const_cast<void*>(bitcode_data),
                          bitcode_len_bytes,
                          filename,
                          0,
                          nullptr,
                          nullptr)
           != hipSuccess)
            throw std::runtime_error("failed to add cb");
    }

    std::vector<char> complete()
    {
        std::vector<char> ret;
        void*             bin     = nullptr;
        size_t            binSize = 0;
        if(hipLinkComplete(state, &bin, &binSize) != hipSuccess)
            throw std::runtime_error("failed to link complete");
        auto bin_char = reinterpret_cast<char*>(bin);
        std::copy(bin_char, bin_char + binSize, std::back_inserter(ret));
        return ret;
    }

    hipLinkState_t state = nullptr;
};

struct StoreOps
{
    StoreOps() = default;

    double scale_factor{1.0};
    // user-provided spir-v store callback
    rocfft_spirv_cb_t spirv_cb;

    // returns true if some store operation is enabled
    bool enabled() const
    {
        return scale_factor != 1.0 || spirv_cb.enabled();
    }

    bool has_spirv() const
    {
        return spirv_cb.enabled();
    }

    std::string forward_decls(const CallbackType cbtype, const char* scalar_type) const
    {
        std::string ret;
        if(spirv_cb.enabled())
        {
            // real inverse FFT kernel works with complex elements but
            // was passed a real-type callback for storing
            if(cbtype == CallbackType::USER_LOAD_STORE_C2R)
            {
                ret += std::string("extern \"C\" __device__ void ") + spirv_cb.symbol_name
                       + "(real_type_t<" + scalar_type + ">*, size_t, real_type_t<" + scalar_type
                       + ">, void*, "
                         "void*);\n";
            }
            else
            {
                ret += std::string("extern \"C\" __device__ void ") + spirv_cb.symbol_name + "("
                       + scalar_type + "*, size_t, " + scalar_type + ", void*, void*);\n";
            }
            // declare a constant name for the load callback as well
            ret += "__device__ auto store_cb_jit_fn = ";
            ret += spirv_cb.symbol_name;
            ret += ";\n";
        }
        return ret;
    }

    std::string name_suffix() const
    {
        std::string ret;
        if(scale_factor != 1.0)
            ret += "_scale";

        if(spirv_cb.enabled())
        {
            ret += "_store" + spirv_cb.get_hash();
        }
        return ret;
    }

    // append kernel arguments to implement the operations defined in
    // *this
    void append_args(RTCKernelArgs& kargs, TreeNode& node) const;
    // transform a global function to implement operations defined in *this
    Function add_ops(const Function& f) const;

    template <typename Tstream>
    void print(Tstream& os, const std::string& indent) const
    {
        if(scale_factor != 1.0)
            os << indent << "scale factor: " << scale_factor << "\n";
    }
};

// helpers to apply both load + store ops together
std::string load_store_name_suffix(const std::optional<LoadOps>&  loadOps,
                                   const std::optional<StoreOps>& storeOps);
void        append_load_store_args(RTCKernelArgs& kargs, TreeNode& node);
void        make_load_store_ops(Function&                      f,
                                const std::optional<LoadOps>&  loadOps,
                                const std::optional<StoreOps>& storeOps);
// forward declarations required by ops (e.g. JIT callbacks)
std::string load_store_decls(const std::optional<LoadOps>&  loadOps,
                             const std::optional<StoreOps>& storeOps,
                             const CallbackType             cbtype,
                             const char*                    load_data_type  = "scalar_type",
                             const char*                    store_data_type = "scalar_type");

#endif
