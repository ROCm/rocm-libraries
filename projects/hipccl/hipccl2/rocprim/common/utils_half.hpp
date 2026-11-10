// MIT License
//
// Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_UTILS_HALF_HPP_
#define COMMON_UTILS_HALF_HPP_

#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/types.hpp>

#include <ostream>

namespace common
{

// Support half operators on host side
ROCPRIM_HOST
inline rocprim::native_half half_to_native(const rocprim::half& x)
{
    return *reinterpret_cast<const rocprim::native_half*>(&x);
}

ROCPRIM_HOST
inline rocprim::half native_to_half(const rocprim::native_half& x)
{
    return *reinterpret_cast<const rocprim::half*>(&x);
}

} // namespace common

// For better Google Test reporting and debug output of half values
inline std::ostream& operator<<(std::ostream& stream, const rocprim::half& value)
{
    stream << static_cast<float>(value);
    return stream;
}

#endif // COMMON_UTILS_HALF_HPP_
