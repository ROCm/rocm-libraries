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

#ifndef ENUM_TO_STRING_H
#define ENUM_TO_STRING_H

#include "fft_params.h"

// Return the string of the hipError code.
static std::string hipError_to_string(const hipError_t ret)
{
    switch(ret)
    {
    case hipSuccess:
        return "hipSuccess";
    case hipErrorInvalidContext:
        return "hipErrorInvalidContext";
    case hipErrorInvalidKernelFile:
        return "hipErrorInvalidKernelFile";
    case hipErrorMemoryAllocation:
        return "hipErrorMemoryAllocation";
    case hipErrorInitializationError:
        return "hipErrorInitializationError";
    case hipErrorLaunchFailure:
        return "hipErrorLaunchFailure";
    case hipErrorLaunchOutOfResources:
        return "hipErrorLaunchOutOfResources";
    case hipErrorInvalidDevice:
        return "hipErrorInvalidDevice";
    case hipErrorInvalidValue:
        return "hipErrorInvalidValue";
    case hipErrorInvalidDevicePointer:
        return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection:
        return "hipErrorInvalidMemcpyDirection";
    case hipErrorUnknown:
        return "hipErrorUnknown";
    case hipErrorInvalidResourceHandle:
        return "hipErrorInvalidResourceHandle";
    case hipErrorNotReady:
        return "hipErrorNotReady";
    case hipErrorNoDevice:
        return "hipErrorNoDevice";
    case hipErrorPeerAccessAlreadyEnabled:
        return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled:
        return "hipErrorPeerAccessNotEnabled";
    case hipErrorRuntimeMemory:
        return "hipErrorRuntimeMemory";
    case hipErrorRuntimeOther:
        return "hipErrorRuntimeOther";
    case hipErrorHostMemoryAlreadyRegistered:
        return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
        return "hipErrorHostMemoryNotRegistered";
    case hipErrorMapBufferObjectFailed:
        return "hipErrorMapBufferObjectFailed";
    case hipErrorTbd:
        return "hipErrorTbd";
    default:
        throw std::runtime_error("unknown hipError");
    }
}
#endif
