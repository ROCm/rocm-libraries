/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019, 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#define PARALLEL_FOR thrust::hip_rocprim::parallel_for
#elif THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#define PARALLEL_FOR thrust::cuda_cub::parallel_for
#endif

#ifdef PARALLEL_FOR

using TestsParams = ::testing::Types<Params<unsigned int>, Params<unsigned long long>>;

TESTS_DEFINE(ParallelForTests, TestsParams)

template <typename T>
struct add_functor
{
    T*       ptr;
    __host__ __device__ void operator()(size_t x)
    {
        atomicAdd(ptr, T(x + 1));
    }
};

TYPED_TEST(ParallelForTests, HostPathSimpleTest)
{
    thrust::device_system_tag tag;
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    const std::vector<size_t> sizes = { (1ull << 31) + 65535, (1ull << 32) + (1ull << 20) - 1 };

    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        auto ptr      = thrust::malloc<T>(tag, sizeof(T));
        auto raw_ptr  = thrust::raw_pointer_cast(ptr);
        ASSERT_NE(raw_ptr, nullptr);

        // Zero input memory
        HIP_CHECK(hipMemset(raw_ptr, 0, sizeof(T)));

        // Create unary function
        add_functor<T> func;
        func.ptr = raw_ptr;

        // Add all numbers: 1+2+...+size = size * (size+1) / 2
        PARALLEL_FOR(tag, func, size);

        T output;
        HIP_CHECK(hipMemcpy(&output, raw_ptr, sizeof(T), hipMemcpyDeviceToHost));
        output *= 2;

        ASSERT_EQ(output, T(size * (size + 1)));

        // Free
        thrust::free(tag, ptr);
    }
}

#undef PARALLEL_FOR
#endif
