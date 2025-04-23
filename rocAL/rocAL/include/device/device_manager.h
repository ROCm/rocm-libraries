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

#pragma once

#if ENABLE_OPENCL

#include <map>
#include <CL/cl.h>
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <memory>
#include "device_data_transfer_code.h"

struct DeviceResources {
    cl_context context;
    cl_device_id device_id;
    cl_command_queue cmd_queue;
    DeviceResources() {
        cmd_queue = nullptr;
        context = nullptr;
        device_id = nullptr;
    }
};

class CLProgram {
   public:
    CLProgram(const DeviceResources* ocl, const DeviceCode& ocl_code) : m_ocl(ocl), m_code(ocl_code) {}

    ~CLProgram();

    cl_int runKernel(const std::string& kernel_name, const std::vector<void*>& args, const std::vector<size_t>& argSize, const std::vector<size_t>& globalWorkSize, const std::vector<size_t>& localWorkSize);

    cl_int buildAll();

    const cl_kernel& operator[](const std::string& kernel_name) const;

    std::string getProgramName();

   private:
    const DeviceResources* m_ocl;

    const DeviceCode& m_code;

    cl_program m_prog;

    std::map<std::string, cl_kernel> m_kernels;
};

class DeviceManager {
   public:
    DeviceManager(){};

    cl_int initialize();

    DeviceResources* resources();

    const CLProgram& operator[](const std::string& prog_name);

    void init_ocl(vx_context context);

    ~DeviceManager();

   private:
    DeviceResources _resources;

    std::map<std::string, CLProgram> m_programs;
};

using pRocalOCL = std::shared_ptr<DeviceManager>;
#endif
