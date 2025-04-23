/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#if ENABLE_HIP
// device manager functions for HIP backend
#include "device/device_manager_hip.h"

#include <vx_ext_amd.h>

#include <iostream>

#include "pipeline/commons.h"

DeviceManagerHip::~DeviceManagerHip() {
    hipError_t err;
    if (_resources.hip_stream != nullptr) {
        err = hipStreamDestroy(_resources.hip_stream);
        if (err != hipSuccess)
            LOG("hipStreamDestroy failed " + TOSTR(err))
        _resources.hip_stream = nullptr;
    }
    LOG("HIP device resources released")
}

hipError_t DeviceManagerHip::initialize() {
    // TODO:: do any HIP specific initialization here
    return hipSuccess;
}

DeviceResourcesHip *DeviceManagerHip::resources() {
    return &_resources;
}

void DeviceManagerHip::init_hip(vx_context context) {
    hipError_t err;
    hipDevice_t dev_id = -1;
    vx_status vxstatus = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE, &dev_id, sizeof(hipDevice_t));

    if (vxstatus != VX_SUCCESS)
        THROW("init_hip::vxQueryContext failed " + TOSTR(vxstatus))

    hipStream_t stream;
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        THROW("init_hip::hipStreamCreate failed " + TOSTR(err))
    }
    err = hipGetDeviceProperties(&_resources.dev_prop, dev_id);
    if (err != hipSuccess) {
        THROW("init_hip::hipGetDeviceProperties failed " + TOSTR(err))
    }
    _resources.hip_stream = stream;
    _resources.device_id = dev_id;
    err = initialize();
    if (err != hipSuccess) {
        THROW("init_hip::initialize failed " + TOSTR(err))
    }
    LOG("ROCAL HIP initialized ...")
}
#endif