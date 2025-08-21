/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef OCL_UTILS_HPP
#define OCL_UTILS_HPP

/*
* A place to put utilities to help with OpenCL
* application workflow.
* Approximates Tensile 'module' loads of
* ELF binaries from file and formatting them as
* executable kernels
*/

#include <Tensile/ocl/OclFwd.hpp>
#include <iostream>

#define CL_CHECK(x)                                                                     \
    if(CL_SUCCESS != (x))                                                               \
    {                                                                                   \
        throw std::runtime_error(                                                       \
            concatenate("File: ", __FILE__, " (", __LINE__, ") CL Error code: ", (x))); \
    }

namespace TensileLite
{

    namespace ocl
    {
        class oclDeviceProp_t;

        cl::Program clModuleLoad(cl::Context                    context,
                                 std::vector<cl::Device> const& devices,
                                 std::string const&             path);
        cl::Program clModuleLoadData(cl::Context                       context,
                                     std::vector<cl::Device> const&    devices,
                                     std::vector<unsigned char> const& bytes);

        oclDeviceProp_t clGetDevicePropertiesAMD();
        oclDeviceProp_t clGetDevicePropertiesAMD(int deviceId);
        oclDeviceProp_t clGetDevicePropertiesAMD(int deviceId, cl::Context context);
        oclDeviceProp_t clGetDevicePropertiesAMD(cl::Device device);

    } // namespace ocl

} // namespace TensileLite

namespace cl
{
    std::ostream& operator<<(std::ostream& stream, Buffer buffer);
}

#endif // OCL_UTILS_HPP
