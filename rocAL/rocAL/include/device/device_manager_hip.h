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

// device manager functions for HIP backend
#if ENABLE_HIP
#pragma once
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <memory>

struct DeviceResourcesHip {
    hipStream_t hip_stream;
    int device_id;
    hipDeviceProp_t dev_prop;
    DeviceResourcesHip() {
        hip_stream = nullptr;
        device_id = -1;
    }
};

class DeviceManagerHip {
   public:
    DeviceManagerHip(){};

    hipError_t initialize();

    DeviceResourcesHip *resources();

    void init_hip(vx_context context);

    ~DeviceManagerHip();

   private:
    DeviceResourcesHip _resources;
};

using pRocalHip = std::shared_ptr<DeviceManagerHip>;

#endif