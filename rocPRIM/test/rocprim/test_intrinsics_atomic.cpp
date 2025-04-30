// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocprim/intrinsics/atomic.hpp"
#include "rocprim/types.hpp"

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"

#include "test_utils.hpp"

#include <cstdint>

template<typename GetPtr>
ROCPRIM_INLINE ROCPRIM_DEVICE
void test_atomics(
    GetPtr get_ptr, bool* error, const uint32_t* test_data, size_t random_size, bool delay)
{
    // This test attempts to test some atomicity properties of the operation by repeatedly writing
    // some random data to the value, distributed over the atomic bits. This way, we can attempt to
    // test that the write really is atomic: If it happens in multiple parts, there is a chance
    // that another write will happen at the same time and mess up the parts. Manually replacing
    // rocprim's 128-bit atomic store with 2 64-bit atomic stores seems to trigger this as expected.
    //
    // Unfortunately, this test cannot really check caching mechanisms, reordering, etc, for now.

    for(int i = 0; i < 1000; ++i)
    {
        const rocprim::uint128_t new_value = rocprim::detail::atomic_load(get_ptr(i * 2));
        const unsigned int       j         = new_value >> 96;
        const unsigned int       a         = (new_value >> 64) & 0xFFFF'FFFF;
        const unsigned int       b         = (new_value >> 32) & 0xFFFF'FFFF;
        const unsigned int       c         = new_value & 0xFFFF'FFFF;

        if(a != test_data[j * 3] || b != test_data[j * 3 + 1] || c != test_data[j * 3 + 2])
        {
            *error = true;
        }

        if(delay)
        {
            __builtin_amdgcn_s_sleep(63);
        }

        const unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x + i) % random_size;
        rocprim::uint128_t value = rocprim::uint128_t{index} << 96;
        value |= rocprim::uint128_t{test_data[index * 3]} << 64;
        value |= rocprim::uint128_t{test_data[index * 3 + 1]} << 32;
        value |= rocprim::uint128_t{test_data[index * 3 + 2]};
        rocprim::detail::atomic_store(get_ptr(i * 2 + 1), value);
    }
}

__global__
void test_global(rocprim::uint128_t* ptr,
                 bool*               error,
                 const uint32_t*     test_data,
                 size_t              random_size)
{
    test_atomics([=](int /* i */) { return ptr; }, error, test_data, random_size, true);
}

__global__
void test_shared(bool* error, const uint32_t* test_data, size_t random_size)
{
    __shared__ rocprim::uint128_t shared_data;
    if(threadIdx.x == 0)
    {
        shared_data = 0;
    }

    __syncthreads();

    test_atomics([&](int /* i */) { return &shared_data; }, error, test_data, random_size, false);
}

__global__
void test_flat(rocprim::uint128_t* global_ptr,
               bool*               error,
               const uint32_t*     test_data,
               size_t              random_size)
{
    __shared__ rocprim::uint128_t shared_data;
    if(threadIdx.x == 0)
    {
        shared_data = 0;
    }

    __syncthreads();

    test_atomics([&](int i) { return test_data[i % random_size] & 1 ? &shared_data : global_ptr; },
                 error,
                 test_data,
                 random_size,
                 false);
}

template<typename F>
void generic_atomic_test(F cbk)
{
    static constexpr uint32_t block_size = 1024;
    static constexpr uint32_t grid_size  = 1024;
    static constexpr uint32_t size       = block_size * grid_size;

    common::device_ptr<bool> d_error(1);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        auto input = test_utils::get_random_data<uint32_t>(size * 3,
                                                           rocprim::numeric_limits<uint32_t>::min(),
                                                           rocprim::numeric_limits<uint32_t>::max(),
                                                           seed_value);
        // Set the correct initial value if the initial atomic value is 0.
        input[0] = 0;
        input[1] = 0;
        input[2] = 0;

        HIP_CHECK(hipMemset(d_error.get(), 0, sizeof(bool)));
        common::device_ptr<uint32_t> d_input(input);

        cbk(block_size, grid_size, d_error.get(), d_input.get(), size);
        HIP_CHECK(hipGetLastError());

        bool result;
        HIP_CHECK(hipMemcpy(&result, d_error.get(), sizeof(bool), hipMemcpyDeviceToHost));
        ASSERT_EQ(result, false);
    }
}

TEST(RocprimAtomicTests, Global128Bits)
{
    common::device_ptr<rocprim::uint128_t> d_ptr(1);
    generic_atomic_test(
        [&](auto block_size, auto grid_size, auto* d_error, auto* d_input, auto size)
        {
            HIP_CHECK(hipMemset(d_ptr.get(), 0, sizeof(rocprim::uint128_t)));
            test_global<<<grid_size, block_size>>>(d_ptr.get(), d_error, d_input, size);
        });
}

TEST(RocprimAtomicTests, Shared128Bits)
{
    generic_atomic_test(
        [&](auto block_size, auto grid_size, auto* d_error, auto* d_input, auto size)
        { test_shared<<<grid_size, block_size>>>(d_error, d_input, size); });
}

TEST(RocprimAtomicTests, Flat128Bits)
{
    common::device_ptr<rocprim::uint128_t> d_ptr(1);
    generic_atomic_test(
        [&](auto block_size, auto grid_size, auto* d_error, auto* d_input, auto size)
        {
            HIP_CHECK(hipMemset(d_ptr.get(), 0, sizeof(rocprim::uint128_t)));
            test_flat<<<grid_size, block_size>>>(d_ptr.get(), d_error, d_input, size);
        });
}
