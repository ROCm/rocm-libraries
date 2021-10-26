// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/device/device_radix_sort.hpp>

// required test headers
#include "test_utils_types.hpp"
#include "test_sort_comparator.hpp"

// Special floats sort test

template<
    class Key,
    bool Descending = false
>
struct params_special
{
    using key_type = Key;
    static constexpr bool descending = Descending;
};

template<class Params>
class RocprimDeviceRadixSortSpecial : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params_special<double>,
    params_special<double, true>,
    params_special<float>,
    params_special<float, true>,
    params_special<half>,
    params_special<half, true>,
    params_special<rocprim::bfloat16>,
    params_special<rocprim::bfloat16, true>
> Params_special;

namespace vectors
{
    void put_special_values(std::vector<half>& keys)
    {
        keys[0] = +0.0;
        keys[1] = -0.0;
        uint16_t p = 0xffff; // -NaN
        keys[2] = *(reinterpret_cast<half*>(&p));
        p = 0x7fff; // +NaN
        keys[3] = *(reinterpret_cast<half*>(&p));
        p = 0x7C00; // +inf
        keys[4] = *(reinterpret_cast<half*>(&p));
        p = 0xFC00; // -inf
        keys[5] = *(reinterpret_cast<half*>(&p));
    }

    void put_special_values(std::vector<rocprim::bfloat16>& keys)
    {
        uint16_t p = 0x0000; //+0.0
        keys[0] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0x8000; // -0.0
        keys[1] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0xffff; // -NaN
        keys[2] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0x7fff; // +NaN
        keys[3] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0x7F80; // +inf
        keys[4] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0xFF80; // -inf
        keys[5] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
    }

    void put_special_values(std::vector<float>& keys)
    {
        keys[0] = +0.0;
        keys[1] = -0.0;
        uint32_t p = 0xffffffff; // -NaN
        keys[2] = *(reinterpret_cast<float*>(&p));
        p = 0x7fffffff; // +NaN
        keys[3] = *(reinterpret_cast<float*>(&p));
        p = 0x7F800000; // +inf
        keys[4] = *(reinterpret_cast<float*>(&p));
        p = 0xFF800000; // -inf
        keys[5] = *(reinterpret_cast<float*>(&p));
    }

    void put_special_values(std::vector<double>& keys)
    {
        keys[0] = +0.0;
        keys[1] = -0.0;
        uint64_t p = 0xffffffffffffffff; // -NaN
        keys[2] = *(reinterpret_cast<double*>(&p));
        p = 0x7fffffffffffffff; // +NaN
        keys[3] = *(reinterpret_cast<double*>(&p));
        p = 0x7ff0000000000000; // +inf
        keys[4] = *(reinterpret_cast<double*>(&p));
        p = 0xfff0000000000000; // -inf
        keys[5] = *(reinterpret_cast<double*>(&p));
    }

    template<typename Real>
    void put_other_values(std::vector<Real>& keys)
    {
        for(auto a : {3.5, 0.5, -1.0, -3.5, -2.5, 0.0, 2.5, 3.5, 4.0, 4.5, 2.0, \
                    -3.0, -2.5, 4.5, -3.5, 0.5, -1.0, 1.0, 3.0, -4.0, 3.0, 2.0, \
                    -0.5, -1.5, 4.0, -1.0})
        {
            keys.push_back(Real(a));
        }
    }

    template<typename Real>
    void put_values(std::vector<Real>& keys)
    {
        keys.resize(6);
        put_special_values(keys);
        put_other_values(keys);
    }

    void first_special_values_descending(std::vector<half>& keys)
    {
        uint16_t p = 0x7fff; // +NaN
        keys[0] = *(reinterpret_cast<half*>(&p));
        p = 0x7C00; // +inf
        keys[1] = *(reinterpret_cast<half*>(&p));
    }

    void last_special_values_descending(std::vector<half>& keys)
    {
        uint16_t p = 0xFC00; // -inf
        keys[30] = *(reinterpret_cast<half*>(&p));
        p = 0xffff; // -NaN
        keys[31] = *(reinterpret_cast<half*>(&p));
    }

    void first_special_values_descending(std::vector<rocprim::bfloat16>& keys)
    {
        uint16_t p = 0x7fff; // +NaN
        keys[0] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0x7F80; // +inf
        keys[1] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
    }

    void last_special_values_descending(std::vector<rocprim::bfloat16>& keys)
    {
        uint16_t p = 0xFF80; // -inf
        keys[30] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0xffff; // -NaN
        keys[31] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
    }

    void first_special_values_descending(std::vector<float>& keys)
    {
        uint32_t p = 0x7fffffff; // +NaN
        keys[0] = *(reinterpret_cast<float*>(&p));
        p = 0x7F800000; // +inf
        keys[1] = *(reinterpret_cast<float*>(&p));
    }

    void last_special_values_descending(std::vector<float>& keys)
    {
        uint32_t p = 0xFF800000; // -inf
        keys[30] = *(reinterpret_cast<float*>(&p));
        p = 0xffffffff; // -NaN
        keys[31] = *(reinterpret_cast<float*>(&p));
    }

    void first_special_values_descending(std::vector<double>& keys)
    {
        uint64_t p = 0x7fffffffffffffff; // +NaN
        keys[0] = *(reinterpret_cast<double*>(&p));
        p = 0x7ff0000000000000; // +inf
        keys[1] = *(reinterpret_cast<double*>(&p));
    }

    void last_special_values_descending(std::vector<double>& keys)
    {
        uint64_t p = 0xfff0000000000000; // -inf
        keys[30] = *(reinterpret_cast<double*>(&p));
        p = 0xffffffffffffffff; // -NaN
        keys[31] = *(reinterpret_cast<double*>(&p));
    }

    template<typename Real>
    void expected_values_descending(std::vector<Real>& keys)
    {
        keys.resize(2);
        first_special_values_descending(keys);
        for(auto a : {4.5, 4.5, 4.0, 4.0, 3.5, 3.5, 3.0, 3.0, 2.5, 2.0, 2.0, 1.0, 0.5, 0.5, \
                    0.0, -0.0, 0.0, -0.5, -1.0, -1.0, -1.0, -1.5, -2.5, -2.5, -3.0, -3.5, -3.5, -4.0})
            keys.push_back(Real(a));
        keys.resize(32);
        last_special_values_descending(keys);
    }

    void first_special_values_ascending(std::vector<half>& keys)
    {
        uint16_t p = 0xffff; // -NaN
        keys[0] = *(reinterpret_cast<half*>(&p));
        p = 0xFC00; // -inf
        keys[1] = *(reinterpret_cast<half*>(&p));
    }

    void last_special_values_ascending(std::vector<half>& keys)
    {
        uint16_t p = 0x7C00; // +inf
        keys[30] = *(reinterpret_cast<half*>(&p));
        p = 0x7fff; // +NaN
        keys[31] = *(reinterpret_cast<half*>(&p));
    }

    void first_special_values_ascending(std::vector<rocprim::bfloat16>& keys)
    {
        uint16_t p = 0xffff; // -NaN
        keys[0] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0xFF80; // -inf
        keys[1] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
    }

    void last_special_values_ascending(std::vector<rocprim::bfloat16>& keys)
    {
        uint16_t p = 0x7F80; // +inf
        keys[30] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
        p = 0x7fff; // +NaN
        keys[31] = *(reinterpret_cast<rocprim::bfloat16*>(&p));
    }

    void first_special_values_ascending(std::vector<float>& keys)
    {
        uint32_t p = 0xffffffff; // -NaN
        keys[0] = *(reinterpret_cast<float*>(&p));
        p = 0xFF800000; // -inf
        keys[1] = *(reinterpret_cast<float*>(&p));
    }

    void last_special_values_ascending(std::vector<float>& keys)
    {
        uint32_t p = 0x7F800000; // +inf
        keys[30] = *(reinterpret_cast<float*>(&p));
        p = 0x7fffffff; // +NaN
        keys[31] = *(reinterpret_cast<float*>(&p));
    }

    void first_special_values_ascending(std::vector<double>& keys)
    {
        uint64_t p = 0xffffffffffffffff; // -NaN
        keys[0] = *(reinterpret_cast<double*>(&p));
        p = 0xfff0000000000000; // -inf
        keys[1] = *(reinterpret_cast<double*>(&p));
    }

    void last_special_values_ascending(std::vector<double>& keys)
    {
        uint64_t p = 0x7ff0000000000000; // +inf
        keys[30] = *(reinterpret_cast<double*>(&p));
        p = 0x7fffffffffffffff; // +NaN
        keys[31] = *(reinterpret_cast<double*>(&p));
    }

    template<typename Real>
    void expected_values_ascending(std::vector<Real>& keys)
    {
        keys.resize(2);
        first_special_values_ascending(keys);
        for(auto a : {-4.0, -3.5, -3.5, -3.0, -2.5, -2.5, -1.5, -1.0, -1.0, -1.0, -0.5, \
                    0.0, -0.0, 0.0, 0.5, 0.5, 1.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.0, 4.5, 4.5})
            keys.push_back(Real(a));
        keys.resize(32);
        last_special_values_ascending(keys);
    }
}

TYPED_TEST_SUITE(RocprimDeviceRadixSortSpecial, Params_special);

TYPED_TEST(RocprimDeviceRadixSortSpecial, SortKeys)
{
    using key_type = typename TestFixture::params::key_type;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int start_bit = 0;
    constexpr unsigned int end_bit = sizeof(key_type) * 8;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    size_t size = 32;

    // Generate data
    std::vector<key_type> keys_input;
    vectors::put_values(keys_input);

    key_type *d_keys_input;
    key_type * d_keys_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    // Expected results on host
    std::vector<key_type> expected;
    if(descending)
        vectors::expected_values_descending(expected);
    else
        vectors::expected_values_ascending(expected);

    // Use custom config
    using config = rocprim::radix_sort_config<8, 5, rocprim::kernel_config<256, 3>, rocprim::kernel_config<256, 8>>;

    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rocprim::radix_sort_keys<config>(
            nullptr, temporary_storage_bytes,
            d_keys_input, d_keys_output, size,
            start_bit, end_bit
        )
    );

    ASSERT_GT(temporary_storage_bytes, 0U);

    void * d_temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

    if(descending)
    {
        HIP_CHECK(
            rocprim::radix_sort_keys_desc<config>(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                start_bit, end_bit,
                stream, debug_synchronous
            )
        );
    }
    else
    {
        HIP_CHECK(
            rocprim::radix_sort_keys<config>(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                start_bit, end_bit,
                stream, debug_synchronous
            )
        );
    }

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input));

    std::vector<key_type> keys_output(size);
    HIP_CHECK(
        hipMemcpy(
            keys_output.data(), d_keys_output,
            size * sizeof(key_type),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(hipFree(d_keys_output));

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output, expected));
}
