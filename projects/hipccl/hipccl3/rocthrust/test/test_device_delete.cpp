/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

struct Foo
{
    __host__ __device__ Foo(void)
        : set_me_upon_destruction(0)
    {
    }

    __host__ __device__
    ~Foo(void)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // __device__ overload
        if(set_me_upon_destruction != 0)
            *set_me_upon_destruction = true;
#endif
    }

    bool* set_me_upon_destruction;
};

TEST(DeviceDelete, TestDeviceDeleteDestructorInvocation)
{
    thrust::device_vector<bool> destructor_flag(1, false);

    thrust::device_ptr<Foo> foo_ptr = thrust::device_new<Foo>();

    Foo exemplar;
    exemplar.set_me_upon_destruction = thrust::raw_pointer_cast(&destructor_flag[0]);
    *foo_ptr                         = exemplar;

    ASSERT_EQ(false, destructor_flag[0]);

    thrust::device_delete(foo_ptr);

    ASSERT_EQ(true, destructor_flag[0]);
}
