// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>

// required test headers
#include "test_utils.hpp"
#include "test_utils_types.hpp"

// kernel definitions
#include "test_block_load_store.kernels.hpp"

template<class Params>
class RocprimVectorizationTests : public ::testing::Test {
public:
    using params = Params;
};

TYPED_TEST_SUITE(RocprimVectorizationTests, VectorParams);

TYPED_TEST(RocprimVectorizationTests, IsVectorizable)
{
    using T = typename TestFixture::params::type;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool should_be_vectorized = TestFixture::params::should_be_vectorized;
    bool input = rocprim::detail::is_vectorizable<T, items_per_thread>();
    ASSERT_EQ(input, should_be_vectorized);
}

TYPED_TEST(RocprimVectorizationTests, MatchVectorType)
{
    using T = typename TestFixture::params::type;
    using U = typename TestFixture::params::vector_type;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    typedef typename rocprim::detail::match_vector_type<T, items_per_thread>::type Vector;
    bool input = std::is_same<Vector, U>::value;
    ASSERT_TRUE(input);
}

// Start stamping out tests
struct RocprimBlockLoadStoreClassTests;

struct FirstPart;
#define suite_name RocprimBlockLoadStoreClassTests
#define warp_params ClassParamsFirstPart
#define name_suffix FirstPart

#include "test_block_load_store.hpp"

#undef suite_name
#undef warp_params
#undef name_suffix

struct SecondPart;
#define suite_name RocprimBlockLoadStoreClassTests
#define warp_params ClassParamsSecondPart
#define name_suffix SecondPart

#include "test_block_load_store.hpp"
