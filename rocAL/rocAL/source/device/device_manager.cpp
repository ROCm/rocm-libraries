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
#if ENABLE_OPENCL

#include "device/device_manager.h"

#include <vx_ext_amd.h>

#include <iostream>

#include "pipeline/commons.h"

DeviceManager::~DeviceManager() {
    if (_resources.cmd_queue != nullptr)
        clReleaseCommandQueue(_resources.cmd_queue);
    if (_resources.device_id != nullptr)
        clReleaseDevice(_resources.device_id);
    if (_resources.context != nullptr)
        clReleaseContext(_resources.context);

    _resources.cmd_queue = nullptr;
    _resources.context = nullptr;
    _resources.device_id = nullptr;
    LOG("OCL context and command queue resources released")
}
CLProgram::~CLProgram() {
    for (auto& kernel_pair : m_kernels)
        if (clReleaseKernel(kernel_pair.second) != CL_SUCCESS)
            ERR("Could not release " + STR(kernel_pair.first))
    m_kernels.clear();
}
cl_int CLProgram::runKernel(const std::string& kernel_name, const std::vector<void*>& args, const std::vector<size_t>& argSize, const std::vector<size_t>& globalWorkSize, const std::vector<size_t>& localWorkSize) {
    cl_int status;
    if (argSize.size() != args.size()) return -1;
    cl_kernel kernel = (*this)[kernel_name];
    for (unsigned argId = 0; argId < args.size(); argId++)
        if ((status = clSetKernelArg(kernel, argId, argSize[argId], args[argId])) != CL_SUCCESS)
            THROW("clSetKernelArg failed " + TOSTR(status));

    if ((status = clEnqueueNDRangeKernel(m_ocl->cmd_queue, kernel, 1, NULL, globalWorkSize.data(), localWorkSize.data(), 0, NULL, NULL)) != CL_SUCCESS)
        THROW("clEnqueueNDRangeKernel failed on " + kernel_name + " error " + TOSTR(status));

    return status;
}

cl_int CLProgram::buildAll() {
    cl_int clerr = CL_SUCCESS;

    auto source_code = m_code.getSourceCode();
    auto program_name = m_code.getName();
    auto kernel_names = m_code.getKernelList();

    size_t code_size = source_code.size();
    const char* code_src = source_code.c_str();

    m_prog = clCreateProgramWithSource(m_ocl->context, 1, &code_src, &code_size, &clerr);

    if (clerr != CL_SUCCESS)
        THROW("Building" + program_name + "program from source failed: " + TOSTR(clerr));

    clerr = clBuildProgram(m_prog, 1, &m_ocl->device_id, NULL, NULL, NULL);

    if (clerr != CL_SUCCESS)
        THROW("Building" + program_name + " failed: " + TOSTR(clerr));

    for (unsigned i = 0; i < kernel_names.size(); ++i) {
        auto kernel = clCreateKernel(m_prog, kernel_names[i].c_str(), &clerr);
        clRetainKernel(kernel);
        if (clerr != CL_SUCCESS)
            THROW("Building kernel" + kernel_names[i] + " failed");

        m_kernels.insert(std::make_pair(kernel_names[i], kernel));
    }
    return clerr;
}

std::string CLProgram::getProgramName() {
    return m_code.getName();
}

const cl_kernel& CLProgram::operator[](const std::string& kernel_name) const {
    const auto it = m_kernels.find(kernel_name);
    if (it != m_kernels.end())
        return it->second;

    THROW("Requested kernel" + kernel_name + " does not exist");
}

cl_int DeviceManager::initialize() {
    std::vector<DeviceCode> ocl_codes = {OCLUtility()};
    for (auto& code : ocl_codes)
        m_programs.insert(make_pair(code.getName(), CLProgram(&_resources, code)));

    cl_int status = CL_SUCCESS;
    for (auto& e : m_programs)
        if ((status = e.second.buildAll()) != CL_SUCCESS)
            THROW("Couldn't build " + e.first);

    return status;
}

DeviceResources* DeviceManager::resources() {
    return &_resources;
}

void DeviceManager::init_ocl(vx_context context) {
    cl_int clerr;
    cl_context clcontext;
    cl_device_id dev_id;
    cl_command_queue cmd_queue;
    vx_status vxstatus = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &clcontext, sizeof(clcontext));

    if (vxstatus != VX_SUCCESS)
        THROW("vxQueryContext failed " + TOSTR(vxstatus))

    cl_int clstatus = clGetContextInfo(clcontext, CL_CONTEXT_DEVICES, sizeof(dev_id), &dev_id, nullptr);

    if (clstatus != CL_SUCCESS)
        THROW("clGetContextInfo failed " + TOSTR(clstatus))

#if defined(CL_VERSION_2_0)
    cmd_queue = clCreateCommandQueueWithProperties(clcontext, dev_id, nullptr, &clerr);
#else
    cmd_queue = clCreateCommandQueue(opencl_context, dev_id, 0, &clerr);
#endif
    if (clerr != CL_SUCCESS)
        THROW("clCreateCommandQueue failed " + TOSTR(clerr))

    _resources.cmd_queue = cmd_queue;
    _resources.context = clcontext;
    _resources.device_id = dev_id;
    clRetainCommandQueue(_resources.cmd_queue);
    clRetainContext(_resources.context);
    clRetainDevice(_resources.device_id);
    // Build CL kernels
    initialize();

    LOG("OpenCL initialized ...")
}

const CLProgram& DeviceManager::operator[](const std::string& prog_name) {
    auto it = m_programs.find(prog_name);
    if (it != m_programs.end())
        return it->second;

    THROW("Requested kernel" + prog_name + "does not exist");
}
#endif
