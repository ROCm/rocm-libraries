/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_TEST_ROCKE_TEST_DEVICE_HPP
#define HIPTENSOR_TEST_ROCKE_TEST_DEVICE_HPP

#include <hip/hip_runtime.h>

#include <cstdio>
#include <sstream>
#include <string>

#ifndef HIPTENSOR_ROCKE_TEST_TARGETS
#error "HIPTENSOR_ROCKE_TEST_TARGETS (rocKE's targets, comma separated) is not defined"
#endif

namespace hiptensor_test
{
    // The base architecture of the current device when it is not one of rocKE's targets, on
    // which the rocKE backend declines every problem; empty on a rocKE target (where a missing
    // kernel pack fails the tests) or when the device cannot be queried.
    inline std::string nonRockeDevice()
    {
        int             device = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&props, device) != hipSuccess)
        {
            return {};
        }
        std::string arch = props.gcnArchName;
        arch             = arch.substr(0, arch.find(':'));

        std::istringstream targets(HIPTENSOR_ROCKE_TEST_TARGETS);
        for(std::string target; std::getline(targets, target, ',');)
        {
            if(target == arch)
            {
                return {};
            }
        }
        return arch;
    }

    // Reports a device that is not a rocKE target with the text the ctest skip expression
    // matches; true if the tests must be skipped.
    inline bool skipNonRockeDevice()
    {
        auto const arch = nonRockeDevice();
        if(arch.empty())
        {
            return false;
        }
        std::fprintf(stderr,
                     "rocKE: unsupported host device %s (rocKE targets: %s). Skipping tests.\n",
                     arch.c_str(),
                     HIPTENSOR_ROCKE_TEST_TARGETS);
        return true;
    }
} // namespace hiptensor_test

#endif // HIPTENSOR_TEST_ROCKE_TEST_DEVICE_HPP
