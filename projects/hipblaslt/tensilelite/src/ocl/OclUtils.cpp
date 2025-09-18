/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <CL/cl2.hpp>
#include <CL/cl_ext.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/ocl/OclHardware.hpp>
#include <Tensile/ocl/OclUtils.hpp>

// std includes
#include <fstream>
#include <string>
#include <vector>

namespace cl
{
    std::ostream& operator<<(std::ostream& stream, Buffer buffer)
    {
        return stream << static_cast<void*>(buffer());
    }

    // Add some definitions for extended device queries (cl_ext.h)
    namespace detail
    {
        CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_BOARD_NAME_AMD, std::string);
        CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD, size_t);
        CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info,
                                     CL_DEVICE_TOPOLOGY_AMD,
                                     cl_device_topology_amd);
    }

} // namespace cl

namespace TensileLite
{

    namespace ocl
    {

        cl::Program clModuleLoad(cl::Context                    context,
                                 std::vector<cl::Device> const& devices,
                                 std::string const&             path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::in | std::ios::ate);
            if(file.fail())
            {
                throw std::runtime_error(concatenate("Failed to load binary file: ", path));
            }

            std::vector<unsigned char> bytes(file.tellg());
            file.seekg(0, std::ios::beg);
            file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            file.close();

            return clModuleLoadData(context, devices, bytes);
        }

        cl::Program clModuleLoadData(cl::Context                       context,
                                     std::vector<cl::Device> const&    devices,
                                     std::vector<unsigned char> const& bytes)
        {
            std::vector<cl_int> binErrs;
            cl::Program         program = cl::Program{context, devices, {bytes}, &binErrs, nullptr};

            for(auto& binErr : binErrs)
            {
                if(binErr != CL_SUCCESS)
                {
                    throw cl::Error(binErr, "Binary error code in Program creation");
                }
            }

            program.build(devices);

            return program;
        }

        oclDeviceProp_t clGetDevicePropertiesAMD()
        {
            auto device = cl::Device::getDefault();
            return clGetDevicePropertiesAMD(device);
        }

        oclDeviceProp_t clGetDevicePropertiesAMD(int deviceId)
        {
            auto context = cl::Context::getDefault();
            return clGetDevicePropertiesAMD(deviceId, context);
        }

        oclDeviceProp_t clGetDevicePropertiesAMD(int deviceId, cl::Context context)
        {
            auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

            if(0 <= deviceId && deviceId < devices.size())
            {
                return clGetDevicePropertiesAMD(devices[deviceId]);
            }
            else
            {
                throw cl::Error(CL_INVALID_DEVICE, "Device index out of range");
            }
        }

        AMDGPU::Processor toProcessorId(std::string const& deviceString)
        {
            if(deviceString.find("gfx803") != std::string::npos)
            {
                return AMDGPU::Processor::gfx803;
            }
            else if(deviceString.find("gfx900") != std::string::npos)
            {
                return AMDGPU::Processor::gfx900;
            }
            else if(deviceString.find("gfx906") != std::string::npos)
            {
                return AMDGPU::Processor::gfx906;
            }
            else if(deviceString.find("gfx908") != std::string::npos)
            {
                return AMDGPU::Processor::gfx908;
            }
            else if(deviceString.find("gfx90a") != std::string::npos)
            {
                return AMDGPU::Processor::gfx90a;
            }
            else if(deviceString.find("gfx940") != std::string::npos)
            {
                return AMDGPU::Processor::gfx940;
            }
            else if(deviceString.find("gfx941") != std::string::npos)
            {
                return AMDGPU::Processor::gfx941;
            }
            else if(deviceString.find("gfx942") != std::string::npos)
            {
                return AMDGPU::Processor::gfx942;
            }
            else if(deviceString.find("gfx950") != std::string::npos)
            {
                return AMDGPU::Processor::gfx950;
            }
            else if(deviceString.find("gfx1010") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1010;
            }
            else if(deviceString.find("gfx1011") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1011;
            }
            else if(deviceString.find("gfx1012") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1012;
            }
            else if(deviceString.find("gfx1030") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1030;
            }
            else if(deviceString.find("gfx1100") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1100;
            }
            else if(deviceString.find("gfx1101") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1101;
            }
            else if(deviceString.find("gfx1102") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1102;
            }
            else if(deviceString.find("gfx1103") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1103;
            }
            else if(deviceString.find("gfx1150") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1150;
            }
            else if(deviceString.find("gfx1151") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1151;
            }
            else if(deviceString.find("gfx1200") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1200;
            }
            else if(deviceString.find("gfx1201") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1201;
            }
            else
            {
                return static_cast<AMDGPU::Processor>(0);
            }
        }

        oclDeviceProp_t clGetDevicePropertiesAMD(cl::Device device)
        {
            auto            topology         = device.getInfo<CL_DEVICE_TOPOLOGY_AMD>();
            auto            maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            auto            maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
            oclDeviceProp_t result           = {
                device.getInfo<CL_DEVICE_BOARD_NAME_AMD>(), //std::string name;
                device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(), // size_t totalGlobalMem;
                device.getInfo<
                    CL_DEVICE_LOCAL_MEM_SIZE>(), // size_t sharedMemPerBlock; CL_DEVICE_LOCAL_MEM_SIZE
                (int)device.getInfo<
                    CL_DEVICE_WAVEFRONT_WIDTH_AMD>(), //int warpSize; CL_WARP_SIZE_NV CL_DEVICE_WAVEFRONT_WIDTH_AMD
                (int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD>(), //int maxThreadsPerBlock;
                {0, 0, 0}, //int maxThreadsDim[3];
                {std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max()}, //int maxGridSize[3];
                (int)device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
                    * 1000, //int clockRate; CL_DEVICE_MAX_CLOCK_FREQUENCY
                (int)device.getInfo<
                    CL_DEVICE_MAX_COMPUTE_UNITS>(), // int multiProcessorCount; CL_MAX_COMPUTE_UNITS
                topology.pcie.bus, //int pciBusID; CL_DEVICE_TOPOLOGY_AMD / CL_DEVICE_PCI_BUS_ID_NV
                topology.pcie.device, // int pciDeviceID;
                device.getInfo<
                    CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD>(), //size_t maxSharedMemoryPerMultiProcessor;
                (int)toProcessorId(device.getInfo<CL_DEVICE_NAME>()) //int gcnArch;
            };

            result.maxThreadsDim[0] = maxWorkItemSizes[0];
            result.maxThreadsDim[1] = maxWorkItemSizes[1];
            result.maxThreadsDim[2] = maxWorkItemSizes[2];

            return result;
        }

    } // namespace ocl

} // namespace TensileLite
