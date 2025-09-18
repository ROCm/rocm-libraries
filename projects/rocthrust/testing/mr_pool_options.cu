/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/mr/pool_options.h>

#include <unittest/unittest.h>

void TestPoolOptionsBasicValidity()
{
  thrust::mr::pool_options options = thrust::mr::pool_options();
  ASSERT_EQUAL(options.validate(), false);

  options.max_blocks_per_chunk = 1024;
  options.max_bytes_per_chunk  = 1024 * 1024;
  options.smallest_block_size  = 8;
  options.largest_block_size   = 1024;
  ASSERT_EQUAL(options.validate(), true);

  // the minimum number of blocks per chunk is bigger than the max
  options.min_blocks_per_chunk = 1025;
  ASSERT_EQUAL(options.validate(), false);
  options.min_blocks_per_chunk = 128;
  ASSERT_EQUAL(options.validate(), true);

  // the minimum number of bytes per chunk is bigger than the max
  options.min_bytes_per_chunk = 1025 * 1024;
  ASSERT_EQUAL(options.validate(), false);
  options.min_bytes_per_chunk = 1024;
  ASSERT_EQUAL(options.validate(), true);

  // smallest block size is bigger than the largest block size
  options.smallest_block_size = 2048;
  ASSERT_EQUAL(options.validate(), false);
  options.smallest_block_size = 8;
  ASSERT_EQUAL(options.validate(), true);
}
DECLARE_UNITTEST(TestPoolOptionsBasicValidity);

void TestPoolOptionsComplexValidity()
{
  thrust::mr::pool_options options = thrust::mr::pool_options();
  ASSERT_EQUAL(options.validate(), false);

  options.max_blocks_per_chunk = 1024;
  options.max_bytes_per_chunk  = 1024 * 1024;
  options.smallest_block_size  = 8;
  options.largest_block_size   = 1024;
  ASSERT_EQUAL(options.validate(), true);

  options.min_bytes_per_chunk = 2 * 1024;
  options.max_bytes_per_chunk = 256 * 1024;

  // the biggest allowed allocation (deduced from blocks in chunks)
  // is smaller than the minimal allowed one (defined in bytes)
  options.max_blocks_per_chunk = 1;
  ASSERT_EQUAL(options.validate(), false);
  options.max_blocks_per_chunk = 1024;
  ASSERT_EQUAL(options.validate(), true);

  // the smallest allowed allocation (deduced from blocks in chunks)
  // is bigger than the maximum allowed one (defined in bytes)
  options.min_blocks_per_chunk = 1024 * 1024;
  ASSERT_EQUAL(options.validate(), false);
  options.min_blocks_per_chunk = 128;
  ASSERT_EQUAL(options.validate(), true);
}
DECLARE_UNITTEST(TestPoolOptionsComplexValidity);
