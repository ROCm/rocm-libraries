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

#include <thrust/system/detail/internal/decompose.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

using thrust::system::detail::internal::uniform_decomposition;

TEST(DecomposeTests, TestUniformDecomposition)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    uniform_decomposition<int> ud(10, 10, 1);

    // [0,10)
    ASSERT_EQ(ud.size(), 1);
    ASSERT_EQ(ud[0].begin(), 0);
    ASSERT_EQ(ud[0].end(), 10);
    ASSERT_EQ(ud[0].size(), 10);
  }

  {
    uniform_decomposition<int> ud(10, 20, 1);

    // [0,10)
    ASSERT_EQ(ud.size(), 1);
    ASSERT_EQ(ud[0].begin(), 0);
    ASSERT_EQ(ud[0].end(), 10);
    ASSERT_EQ(ud[0].size(), 10);
  }

  {
    uniform_decomposition<int> ud(8, 5, 2);

    // [0,5)[5,8)
    ASSERT_EQ(ud.size(), 2);
    ASSERT_EQ(ud[0].begin(), 0);
    ASSERT_EQ(ud[0].end(), 5);
    ASSERT_EQ(ud[0].size(), 5);
    ASSERT_EQ(ud[1].begin(), 5);
    ASSERT_EQ(ud[1].end(), 8);
    ASSERT_EQ(ud[1].size(), 3);
  }

  {
    uniform_decomposition<int> ud(8, 5, 3);

    // [0,5)[5,8)
    ASSERT_EQ(ud.size(), 2);
    ASSERT_EQ(ud[0].begin(), 0);
    ASSERT_EQ(ud[0].end(), 5);
    ASSERT_EQ(ud[0].size(), 5);
    ASSERT_EQ(ud[1].begin(), 5);
    ASSERT_EQ(ud[1].end(), 8);
    ASSERT_EQ(ud[1].size(), 3);
  }

  {
    uniform_decomposition<int> ud(10, 1, 2);

    // [0,5)[5,10)
    ASSERT_EQ(ud.size(), 2);
    ASSERT_EQ(ud[0].begin(), 0);
    ASSERT_EQ(ud[0].end(), 5);
    ASSERT_EQ(ud[0].size(), 5);
    ASSERT_EQ(ud[1].begin(), 5);
    ASSERT_EQ(ud[1].end(), 10);
    ASSERT_EQ(ud[1].size(), 5);
  }

  {
    // [0,4)[4,8)[8,10)
    uniform_decomposition<int> ud(10, 2, 3);

    ASSERT_EQ(ud.size(), 3);
    ASSERT_EQ(ud[0].begin(), 0);
    ASSERT_EQ(ud[0].end(), 4);
    ASSERT_EQ(ud[0].size(), 4);
    ASSERT_EQ(ud[1].begin(), 4);
    ASSERT_EQ(ud[1].end(), 8);
    ASSERT_EQ(ud[1].size(), 4);
    ASSERT_EQ(ud[2].begin(), 8);
    ASSERT_EQ(ud[2].end(), 10);
    ASSERT_EQ(ud[2].size(), 2);
  }
}
