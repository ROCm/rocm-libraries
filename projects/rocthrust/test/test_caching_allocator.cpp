/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/config.h>

#include <thrust/detail/caching_allocator.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

template <typename Allocator>
void test_implementation(Allocator alloc)
{
  using Traits = typename thrust::detail::allocator_traits<Allocator>;
  using Ptr    = typename Allocator::pointer;

  Ptr p = Traits::allocate(alloc, 123);
  Traits::deallocate(alloc, p, 123);

  Ptr p2 = Traits::allocate(alloc, 123);
  ASSERT_EQ(p, p2);
}

TEST(CachingAllocatorTests, TestSingleDeviceTLSCachingAllocator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  test_implementation(thrust::detail::single_device_tls_caching_allocator());
}
