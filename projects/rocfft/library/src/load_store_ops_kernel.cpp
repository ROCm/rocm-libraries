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

#include "load_store_ops.h"
#include "rtc_kernel.h"
#include "tree_node.h"

void LoadOps::append_args(RTCKernelArgs& kargs, TreeNode& node) const {}

void StoreOps::append_args(RTCKernelArgs& kargs, TreeNode& node) const
{
    if(scale_factor != 1.0)
    {
        switch(node.precision)
        {
        case rocfft_precision_single:
            kargs.append_float(scale_factor);
            break;
        case rocfft_precision_double:
            kargs.append_double(scale_factor);
            break;
        case rocfft_precision_half:
            // Convert scale factor to float first before truncating it to
            // rocfft_fp16.  Directly truncating a double to rocfft_fp16 introduces
            //  an unwanted symbol (__truncdfhf2) to rocFFT's lib.
            kargs.append_half(static_cast<float>(scale_factor));
            break;
        }
    }
}

void append_load_store_args(RTCKernelArgs& kargs, TreeNode& node)
{
    node.loadOps.append_args(kargs, node);
    node.storeOps.append_args(kargs, node);
}
