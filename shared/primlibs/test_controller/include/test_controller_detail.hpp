// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// Internal helpers used by test_utils_controller.hpp.
// These are deliberately self-contained (they only depend on the standard library
// and the HIP runtime) so that the test controller can be used by any of the
// primitives libraries, regardless of what their own test utility headers provide.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <hip/hip_runtime.h>

// Checks the result of a HIP runtime call and exits on failure.
// This is named differently from the HIP_CHECK macros that the individual libraries
// define in order to avoid redefinition conflicts.
#define TEST_CONTROLLER_HIP_CHECK(condition)                                               \
    {                                                                                      \
        hipError_t error = condition;                                                      \
        if(error != hipSuccess)                                                            \
        {                                                                                  \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " file: " << __FILE__ \
                      << " line: " << __LINE__ << std::endl;                               \
            std::exit(error);                                                              \
        }                                                                                  \
    }

namespace test_controller
{
namespace detail
{

    // Returns the value of the given environment variable, or std::nullopt if it is not set.
    inline std::optional<std::string> get_env(const char* name)
    {
#ifdef _MSC_VER
        char*   env = nullptr;
        size_t  len = 0;
        errno_t err = _dupenv_s(&env, &len, name);
        if(err || env == nullptr)
        {
            return std::nullopt;
        }
        std::string result(env);
        free(env);
        return result;
#else
        const char* env = std::getenv(name);
        if(env == nullptr)
        {
            return std::nullopt;
        }
        return std::string(env);
#endif
    }

    // Returns the id of the device that ctest has assigned to the currently running test
    // (via resource groups), or 0 if no resource group is set.
    inline int obtain_device_from_ctest()
    {
        static const std::string rg0 = "CTEST_RESOURCE_GROUP_0";

        const std::optional<std::string> env = get_env(rg0.c_str());
        if(!env)
        {
            return 0;
        }

        std::string amdgpu_target(env.value());
        std::transform(amdgpu_target.cbegin(),
                       amdgpu_target.cend(),
                       amdgpu_target.begin(),
                       // Feeding std::toupper plainly results in implicitly truncating conversions
                       // between int and char triggering warnings.
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

        const std::optional<std::string> reqs = get_env((rg0 + "_" + amdgpu_target).c_str());
        if(!reqs)
        {
            return 0;
        }

        const std::string& r = reqs.value();
        return std::atoi(r.substr(r.find(':') + 1, r.find(',') - (r.find(':') + 1)).c_str());
    }

} // namespace detail
} // namespace test_controller
