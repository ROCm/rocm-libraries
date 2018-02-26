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

#ifndef ROCPRIM_TEST_TEST_UTILS_HPP_
#define ROCPRIM_TEST_TEST_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>
#include <cstdlib>

#include <hipcub.hpp>

#include "../test_utils_host.hpp"

namespace test_utils
{
    HIPCUB_HOST_DEVICE inline
    constexpr unsigned int warp_size()
    {
        return HIPCUB_WARP_THREADS;
    }

    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T max(const T& a, const T& b)
    {
        return a < b ? b : a;
    }

    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T min(const T& a, const T& b)
    {
        return a < b ? a : b;
    }

    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr bool is_power_of_two(const T x)
    {
        static_assert(std::is_integral<T>::value, "T must be integer type");
        return (x > 0) && ((x & (x - 1)) == 0);
    }

    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T next_power_of_two(const T x, const T acc = 1)
    {
        static_assert(std::is_unsigned<T>::value, "T must be unsigned type");
        return acc >= x ? acc : next_power_of_two(x, 2 * acc);
    }

    // Return thread id in a "logical warp", which can be smaller than a hardware warp size.
    template<unsigned int LogicalWarpSize>
    HIPCUB_DEVICE inline
    auto logical_lane_id()
        -> typename std::enable_if<is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return hipcub::LaneId() & (LogicalWarpSize-1); // same as land_id()%WarpSize
    }

    template<unsigned int LogicalWarpSize>
    HIPCUB_DEVICE inline
    auto logical_lane_id()
        -> typename std::enable_if<!is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return hipcub::LaneId()%LogicalWarpSize;
    }

    template<>
    HIPCUB_DEVICE inline
    unsigned int logical_lane_id<HIPCUB_WARP_THREADS>()
    {
        return hipcub::LaneId();
    }

    // Return id of "logical warp" in a block
    template<unsigned int LogicalWarpSize>
    HIPCUB_DEVICE inline
    unsigned int logical_warp_id()
    {
        return hipcub::RowMajorTid(1, 1, 1)/LogicalWarpSize;
    }

    template<>
    HIPCUB_DEVICE inline
    unsigned int logical_warp_id<HIPCUB_WARP_THREADS>()
    {
        return hipcub::WarpId();
    }

    inline
    size_t get_max_block_size()
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

    // Select the minimal warp size for block of size block_size, it's
    // useful for blocks smaller than maximal warp size.
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T get_min_warp_size(const T block_size, const T max_warp_size)
    {
        static_assert(std::is_unsigned<T>::value, "T must be unsigned type");
        return block_size >= max_warp_size ? max_warp_size : next_power_of_two(block_size);
    }

    // Custom type used in tests
    template<class T>
    struct custom_test_type
    {
        T x;
        T y;

        HIPCUB_HOST_DEVICE
        custom_test_type(T xx = 0, T yy = 0) : x(xx), y(yy) {}

        HIPCUB_HOST_DEVICE
        ~custom_test_type() {}

        HIPCUB_HOST_DEVICE inline
        custom_test_type& operator=(const custom_test_type& other)
        {
            x = other.x;
            y = other.y;
            return *this;
        }

        HIPCUB_HOST_DEVICE inline
        custom_test_type operator+(const custom_test_type& other) const
        {
            return custom_test_type(x + other.x, y + other.y);
        }

        HIPCUB_HOST_DEVICE inline
        bool operator==(const custom_test_type& other) const
        {
            return (x == other.x && y == other.y);
        }
    };
} // end test_util namespace

#endif // ROCPRIM_TEST_HIPCUB_TEST_UTILS_HPP_
