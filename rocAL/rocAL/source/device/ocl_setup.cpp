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

//
// Created by svcbuild on 11/4/19.
//
#if ENABLE_OPENCL
#include "device/ocl_setup.h"

#include <vector>

#include "pipeline/exception.h"
#include "pipeline/log.h"
int get_device_and_context(int devIdx, cl_context *pContext, cl_device_id *pDevice, cl_device_type clDeviceType) {
    cl_int error = CL_DEVICE_NOT_FOUND;

    int status;

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */

    cl_uint numPlatforms = 0;
    cl_platform_id platform = nullptr;
    std::vector<cl_platform_id> platforms;
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS)
        THROW("clGetPlatformIDs returned error: " + TOSTR(status))

    if (0 < numPlatforms) {
        platforms.resize(numPlatforms);
        status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        if (status != CL_SUCCESS)
            THROW("clGetPlatformIDs returned error: " + TOSTR(status))

        for (unsigned i = 0; i < numPlatforms; ++i) {
            char vendor[100];
            status = clGetPlatformInfo(platforms[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(vendor),
                                       vendor,
                                       nullptr);

            const std::string AMD_NAME = "Advanced Micro Devices, Inc.";

            if (status != CL_SUCCESS) {
                LOG("clGetPlatformInfo returned error: " + TOSTR(status))
                continue;
            }
            if (AMD_NAME.compare(0, AMD_NAME.length(), vendor) == 0) {
                LOG("AMD Platform found " + STR(vendor))
                platform = platforms[i];
                break;
            }
        }
    }

    if (!platform)
        THROW("Couldn't find AMD OpenCL platform")

    // enumerate devices

    // To Do: handle multi-GPU case, pick appropriate GPU/APU
    char driverVersion[100] = "\0";

    cl_context_properties contextProps[3] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platform,
            0};

    // Retrieve device
    cl_uint numDevices = 0;
    clGetDeviceIDs(platform, clDeviceType, 0, nullptr, &numDevices);
    if (numDevices == 0)
        THROW("No GPU OpenCL device found on the AMD's platform")

    std::vector<cl_device_id> devices;
    devices.resize(numDevices);

    status = clGetDeviceIDs(platform, clDeviceType, numDevices, devices.data(), &numDevices);
    if (status != CL_SUCCESS)
        THROW("clGetDeviceIDs returned error: " + TOSTR(status))

    clGetDeviceInfo(devices[0], CL_DRIVER_VERSION, sizeof(driverVersion), driverVersion, nullptr);

    // log all devices found:
    LOG("Driver version: " + STR(driverVersion));
    LOG("Devices found " + TOSTR(numDevices));
    for (unsigned int n = 0; n < numDevices; n++) {
        char deviceName[100] = "\0";
        clGetDeviceInfo(devices[n], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        LOG("GPU device " + STR(deviceName))
    }

    if (devIdx >= 0 && devIdx < (int)numDevices) {
        *pDevice = devices[devIdx];
        clRetainDevice(*pDevice);
        *pContext = clCreateContext(contextProps, 1, pDevice, nullptr, nullptr, &error);

        if (error == CL_SUCCESS) {
            char deviceName[100] = "\0";
            clGetDeviceInfo(devices[devIdx], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            LOG("Using GPU device " + STR(deviceName));
        } else {
            THROW("clCreateContext failed: " + TOSTR(error));
        }
    } else {
        THROW("Device id " + TOSTR(devIdx) + " is out of range of available devices " + TOSTR(numDevices))
    }

    for (unsigned int idx = 0; idx < numDevices; idx++) {
        clReleaseDevice(devices[idx]);
    }

    return error;
}
#endif
