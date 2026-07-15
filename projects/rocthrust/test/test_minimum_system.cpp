/*
 *  Copyright 2026 NVIDIA Corporation
 *  Modifications Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/iterator/detail/minimum_system.h>

#include _THRUST_STD_INCLUDE(type_traits)

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

template <typename System, typename SeqSystem>
void check()
{
  EXPECT_TRUE(_THRUST_STD::is_convertible_v<System, SeqSystem>);
  EXPECT_TRUE(!_THRUST_STD::is_convertible_v<SeqSystem, System>);
  EXPECT_TRUE(_THRUST_STD::is_same_v<thrust::detail::minimum_system_t<SeqSystem, System>, SeqSystem>);
  EXPECT_TRUE(_THRUST_STD::is_same_v<thrust::detail::minimum_system_t<System, SeqSystem>, SeqSystem>);
}

TEST(MinimumSystemTests, MinimumSystem)
{
  using seq      = decltype(thrust::seq);
  using seq_tag  = decltype(thrust::seq)::tag_type;
  using dev      = decltype(thrust::device);
  using dev_tag  = thrust::device_system_tag;
  using host     = decltype(thrust::host);
  using host_tag = thrust::host_system_tag;

  check<dev, seq>();
  check<dev_tag, seq>();
  check<dev, seq_tag>();
  check<dev_tag, seq_tag>();
  check<host, seq_tag>();
  check<host_tag, seq_tag>();
}
