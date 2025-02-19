/******************************************************************************
* Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef KERNEL_LAUNCH_SINGLE
#define KERNEL_LAUNCH_SINGLE

#include "../device/kernels/callback.h"
#include "rocfft/rocfft.h"
#include "tree_node.h"

// FIXME: documentation
struct DeviceCallIn
{
    TreeNode* node      = nullptr;
    void*     bufIn[2]  = {nullptr, nullptr};
    void*     bufOut[2] = {nullptr, nullptr};

    // some kernels require a temp buffer that's neither input nor output
    void* bufTemp = nullptr;

    hipStream_t     rocfft_stream = nullptr;
    GridParam       gridParam;
    hipDeviceProp_t deviceProp;

    UserCallbacks callbacks;

    CallbackType get_callback_type() const
    {
        if(callbacks.load_cb_fn || callbacks.store_cb_fn)
            return CallbackType::USER_LOAD_STORE;
        else
            return CallbackType::NONE;
    }
};

#endif // KERNEL_LAUNCH_SINGLE
