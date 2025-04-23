// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_TEST_UTILS_ROCRAND_CPP_WRAPPER_TRAITS_HPP_
#define TEST_TEST_UTILS_ROCRAND_CPP_WRAPPER_TRAITS_HPP_

#include <rocrand/rocrand.hpp>

template<class GeneratorType>
struct supports_offset
{
    // Always false
    static_assert(sizeof(GeneratorType) == 0,
                  "supports_offset is not implemented for this generator");
};

template<>
struct supports_offset<rocrand_cpp::mrg31k3p> : std::true_type
{};
template<>
struct supports_offset<rocrand_cpp::mrg32k3a> : std::true_type
{};
template<>
struct supports_offset<rocrand_cpp::philox4x32_10> : std::true_type
{};
template<>
struct supports_offset<rocrand_cpp::threefry2x32> : std::true_type
{};
template<>
struct supports_offset<rocrand_cpp::threefry2x64> : std::true_type
{};
template<>
struct supports_offset<rocrand_cpp::threefry4x32> : std::true_type
{};
template<>
struct supports_offset<rocrand_cpp::threefry4x64> : std::true_type
{};
template<>
struct supports_offset<rocrand_cpp::xorwow> : std::true_type
{};
// lfsr113, mt19937 and mtgp32 do not support offset
template<>
struct supports_offset<rocrand_cpp::lfsr113> : std::false_type
{};
template<>
struct supports_offset<rocrand_cpp::mt19937> : std::false_type
{};
template<>
struct supports_offset<rocrand_cpp::mtgp32> : std::false_type
{};
// sobol generators do not support offset
template<>
struct supports_offset<rocrand_cpp::sobol32> : std::false_type
{};
template<>
struct supports_offset<rocrand_cpp::sobol64> : std::false_type
{};
template<>
struct supports_offset<rocrand_cpp::scrambled_sobol32> : std::false_type
{};
template<>
struct supports_offset<rocrand_cpp::scrambled_sobol64> : std::false_type
{};

template<class GeneratorType>
struct is_qrng
{
    // Always false
    static_assert(sizeof(GeneratorType) == 0, "is_qrng is not implemented for this generator");
};

template<>
struct is_qrng<rocrand_cpp::sobol32> : std::true_type
{};
template<>
struct is_qrng<rocrand_cpp::sobol64> : std::true_type
{};
template<>
struct is_qrng<rocrand_cpp::scrambled_sobol32> : std::true_type
{};
template<>
struct is_qrng<rocrand_cpp::scrambled_sobol64> : std::true_type
{};
template<>
struct is_qrng<rocrand_cpp::lfsr113> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::mt19937> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::mtgp32> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::mrg31k3p> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::mrg32k3a> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::philox4x32_10> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::threefry2x32> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::threefry2x64> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::threefry4x32> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::threefry4x64> : std::false_type
{};
template<>
struct is_qrng<rocrand_cpp::xorwow> : std::false_type
{};

template<class GeneratorType>
struct is_64bit
{
    // Always false
    static_assert(sizeof(GeneratorType) == 0, "is_64bit is not implemented for this generator");
};

template<>
struct is_64bit<rocrand_cpp::lfsr113> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::mrg31k3p> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::mrg32k3a> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::mt19937> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::mtgp32> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::philox4x32_10> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::sobol32> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::scrambled_sobol32> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::sobol64> : std::true_type
{};
template<>
struct is_64bit<rocrand_cpp::scrambled_sobol64> : std::true_type
{};
template<>
struct is_64bit<rocrand_cpp::threefry2x32> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::threefry2x64> : std::true_type
{};
template<>
struct is_64bit<rocrand_cpp::threefry4x32> : std::false_type
{};
template<>
struct is_64bit<rocrand_cpp::threefry4x64> : std::true_type
{};
template<>
struct is_64bit<rocrand_cpp::xorwow> : std::false_type
{};

#endif // TEST_TEST_UTILS_ROCRAND_CPP_WRAPPER_TRAITS_HPP_
