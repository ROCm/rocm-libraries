// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_TEST_UTILS_HPP_
#define TEST_TEST_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>
#include <cstdlib>

#include <rocprim.hpp>

#include "../test_utils_host.hpp"

namespace test_utils
{

#ifdef ROCPRIM_HC_API
inline
size_t get_max_tile_size(hc::accelerator acc = hc::accelerator())
{
    return acc.get_max_tile_static_size();
}
#endif

#ifdef ROCPRIM_HIP_API
inline
size_t hip_get_max_block_size()
{
    hipDeviceProp_t device_properties;
    hipError_t error = hipGetDeviceProperties(&device_properties, 0);
    if(error != hipSuccess)
    {
        std::cout << "HIP error: " << error
                  << " file: " << __FILE__
                  << " line: " << __LINE__
                  << std::endl;
        std::exit(error);
    }
    return device_properties.maxThreadsPerBlock;
}
#endif

#if defined(ROCPRIM_HC_API) || defined(ROCPRIM_HIP_API)
// Custom type used in tests
template<class T>
struct custom_test_type
{
    T x;
    T y;

    ROCPRIM_HOST_DEVICE
    custom_test_type(T xx = 0, T yy = 0) : x(xx), y(yy) {}

    ROCPRIM_HOST_DEVICE
    ~custom_test_type() {}

    ROCPRIM_HOST_DEVICE
    custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(x + other.x, y + other.y);
    }

    ROCPRIM_HOST_DEVICE
    bool operator==(const custom_test_type& other) const
    {
        return (x == other.x && y == other.y);
    }
};
#endif

} // end test_utils namespace

#endif // TEST_TEST_UTILS_HPP_
