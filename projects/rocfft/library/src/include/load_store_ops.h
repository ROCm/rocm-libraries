// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string>

class RTCKernelArgs;
class Function;
class TreeNode;

struct LoadOps
{
    LoadOps() = default;

    // returns true if some load operation is enabled
    bool enabled() const
    {
        return false;
    }

    std::string name_suffix() const
    {
        std::string ret;
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

struct StoreOps
{
    StoreOps() = default;

    double scale_factor{1.0};

    // returns true if some store operation is enabled
    bool enabled() const
    {
        return scale_factor != 1.0;
    }

    std::string name_suffix() const
    {
        std::string ret;
        if(scale_factor != 1.0)
            ret += "_scale";
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
std::string load_store_name_suffix(const LoadOps& loadOps, const StoreOps& storeOps);
void        append_load_store_args(RTCKernelArgs& kargs, TreeNode& node);
void        make_load_store_ops(Function& f, const LoadOps& loadOps, const StoreOps& storeOps);

#endif
