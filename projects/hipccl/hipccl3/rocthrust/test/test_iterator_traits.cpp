// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include _THRUST_STD_INCLUDE(type_traits)

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

TEST(IteratorTraitsTests, IteratorSystemPropagationAnySystem)
{
  [[maybe_unused]] auto counting_it = thrust::make_counting_iterator(0);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);

  [[maybe_unused]] auto transform_it = thrust::make_transform_iterator(counting_it, thrust::square<>{});
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::any_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(counting_it, transform_it);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::any_system_tag>);
}

TEST(IteratorTraitsTests, IteratorSystemPropagationHostSystem)
{
  [[maybe_unused]] auto h_vec    = thrust::host_vector<int>{};
  [[maybe_unused]] auto h_vec_it = h_vec.begin();
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(h_vec_it)>, thrust::host_system_tag>);

  [[maybe_unused]] auto transform_it = thrust::make_transform_iterator(h_vec_it, thrust::square<>{});
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::host_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(h_vec_it, transform_it);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::host_system_tag>);
}

TEST(IteratorTraitsTests, IteratorSystemPropagationDeviceSystem)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  [[maybe_unused]] auto d_vec    = thrust::device_vector<int>{};
  [[maybe_unused]] auto d_vec_it = d_vec.begin();
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(d_vec_it)>, thrust::device_system_tag>);

  [[maybe_unused]] auto transform_it = thrust::make_transform_iterator(d_vec_it, thrust::square<>{});
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::device_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(d_vec_it, transform_it);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::device_system_tag>);
}

TEST(IteratorTraitsTests, IteratorSystemPropagationAnyAndDeviceSystem)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  [[maybe_unused]] auto d_vec    = thrust::device_vector<int>{};
  [[maybe_unused]] auto d_vec_it = d_vec.begin();
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(d_vec_it)>, thrust::device_system_tag>);

  [[maybe_unused]] auto counting_it = thrust::make_counting_iterator(0);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);

  [[maybe_unused]] auto zip_it = thrust::make_zip_iterator(d_vec_it, counting_it);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::device_system_tag>);
}
