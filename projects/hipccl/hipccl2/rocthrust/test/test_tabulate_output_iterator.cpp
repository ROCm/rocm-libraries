/*
 *  Copyright 2024 NVIDIA Corporation
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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(type_traits)

using VectorTestsParams = ::testing::Types<
  Params<thrust::host_vector<signed char>>,
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::host_vector<float>>,
  Params<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>,
  Params<thrust::device_vector<signed char>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>,
  Params<thrust::universal_vector<int>>,
  Params<thrust::universal_host_pinned_vector<int>>>;

TESTS_DEFINE(TabulateOutputIteratorTests, FullTestsParams);
TESTS_DEFINE(TabulateOutputIteratorVectorTests, VectorTestsParams);

template <typename OutItT>
struct host_write_op
{
  OutItT out;

  template <typename IndexT, typename T>
  THRUST_HOST void operator()(IndexT index, T val)
  {
    out[index] = val;
  }
};

template <typename OutItT>
struct host_write_first_op
{
  OutItT out;

  template <typename IndexT, typename T>
  THRUST_HOST void operator()(IndexT index, T val)
  {
    // val is a thrust::tuple(value, input_index). Only write out the value part.
    out[index] = thrust::get<0>(val);
  }
};

template <typename OutItT>
struct device_write_first_op
{
  OutItT out;

  template <typename IndexT, typename T>
  THRUST_DEVICE void operator()(IndexT index, T val)
  {
    // val is a thrust::tuple(value, input_index). Only write out the value part.
    out[index] = thrust::get<0>(val);
  }
};

struct select_op
{
  std::size_t select_every_nth;

  template <typename T, typename IndexT>
  THRUST_HOST_DEVICE bool operator()(thrust::tuple<T, IndexT> key_index_pair)
  {
    // Select every n-th item
    return (thrust::get<1>(key_index_pair) % select_every_nth == 0);
  }
};

struct index_to_gather_index_op
{
  std::size_t gather_stride;

  template <typename IndexT>
  THRUST_HOST_DEVICE IndexT operator()(IndexT index)
  {
    // Gather the i-th output item from input[i*3]
    return index * static_cast<IndexT>(gather_stride);
  }
};

TYPED_TEST(TabulateOutputIteratorVectorTests, TestTabulateOutputIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;
  using it_t   = typename Vector::iterator;
  using space  = typename thrust::iterator_system<typename Vector::iterator>::type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  static constexpr std::size_t num_items = 240;
  Vector input(num_items);
  Vector output(num_items, T{42});

  // Use operator type that supports the targeted system
  using op_t = typename _THRUST_STD::conditional<(_THRUST_STD::is_same<space, thrust::host_system_tag>::value),
                                                 host_write_first_op<it_t>,
                                                 device_write_first_op<it_t>>::type;

  // Construct tabulate_output_iterator
  op_t op{output.begin()};
  auto tabulate_out_it = thrust::make_tabulate_output_iterator(op);

  // Prepare input
  thrust::sequence(input.begin(), input.end(), 1);
  auto iota_it   = thrust::make_counting_iterator(0);
  auto zipped_in = thrust::make_zip_iterator(input.begin(), iota_it);

  // Run copy_if using tabulate_output_iterator as the output iterator
  static constexpr std::size_t select_every_nth = 3;
  auto selected_it_end =
    thrust::copy_if(zipped_in, zipped_in + num_items, tabulate_out_it, select_op{select_every_nth});
  const auto num_selected = static_cast<std::size_t>(thrust::distance(tabulate_out_it, selected_it_end));

  // Prepare expected data
  Vector expected_output(num_items, T{42});
  const std::size_t expected_num_selected = (num_items + select_every_nth - 1) / select_every_nth;
  auto gather_index_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), index_to_gather_index_op{select_every_nth});
  thrust::gather(gather_index_it, gather_index_it + expected_num_selected, input.cbegin(), expected_output.begin());

  ASSERT_EQ(expected_num_selected, num_selected);
  ASSERT_EQ(output, expected_output);
}

TEST(TabulateOutputIteratorTests, TestTabulateOutputIterator)
{
  using vector_t = thrust::host_vector<int>;
  using vec_it_t = typename vector_t::iterator;
  using op_t     = host_write_op<vec_it_t>;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  vector_t out(4, 42);
  thrust::tabulate_output_iterator<op_t> tabulate_out_it{op_t{out.begin()}};

  tabulate_out_it[1] = 2;
  vector_t ref{42, 2, 42, 42};
  ASSERT_EQ(out, ref);

  tabulate_out_it[3] = 0;
  ref                = {42, 2, 42, 0};
  ASSERT_EQ(out, ref);

  tabulate_out_it[1] = 4;
  ref                = {42, 4, 42, 0};
  ASSERT_EQ(out, ref);
}
