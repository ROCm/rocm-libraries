// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include "test_utils.hpp"

using Params = ::testing::Types<char,
                                int8_t,
                                short,
                                uint16_t,
                                rocprim::half,
                                rocprim::bfloat16,
                                int,
                                unsigned int,
                                float,
                                long long,
                                unsigned long long,
                                int64_t,
                                double,
                                rocprim::int128_t,
                                rocprim::uint128_t>;

template<class T>
ROCPRIM_HOST_DEVICE
inline T get_random_full_range(std::uint32_t seed = std::rand())
{
    return test_utils::get_random_value<T>(rocprim::numeric_limits<T>::min(),
                                           rocprim::numeric_limits<T>::max(),
                                           seed);
}

template <typename T>
class DoubleBufferTest : public ::testing::Test
{
protected:
    T value1{};
    T value2{};

    rocprim::double_buffer<T> db{&value1, &value2};
};

TYPED_TEST_SUITE(DoubleBufferTest, Params);

TYPED_TEST(DoubleBufferTest, CurrentBufferTest)
{
    EXPECT_EQ(this->db.current(), &this->value1);
}

TYPED_TEST(DoubleBufferTest, AlternateBufferTest)
{
    EXPECT_EQ(this->db.alternate(), &this->value2);
}

TYPED_TEST(DoubleBufferTest, SwapBuffersTest)
{
    this->db.swap();
    EXPECT_EQ(this->db.current(), &this->value2);
    EXPECT_EQ(this->db.alternate(), &this->value1);
}

template <typename T>
class FutureValueTest : public ::testing::Test
{
protected:
    using value_type = T;

    value_type value{};
    
    rocprim::future_value<value_type> fv{&value};
};

TYPED_TEST_SUITE(FutureValueTest, Params);

TYPED_TEST(FutureValueTest, MemberAccessTest)
{
    using T = typename TestFixture::value_type;

    this->value = get_random_full_range<T>();
    EXPECT_EQ(static_cast<TypeParam>(this->fv), this->value);
}

TYPED_TEST(FutureValueTest, GetInputValuePlainTest)
{
    using T = typename TestFixture::value_type;
    
    T val = get_random_full_range<T>();
    EXPECT_EQ(rocprim::detail::get_input_value(val), val);
}

TYPED_TEST(FutureValueTest, GetInputValueFutureTest)
{
    using T = typename TestFixture::value_type;
    
    this->value = get_random_full_range<T>();
    EXPECT_EQ(rocprim::detail::get_input_value(this->fv), this->value);
}

template<class K, class V>
struct kv_tag
{
    using key_type   = K;
    using value_type = V;
};

using TestPairs = ::testing::Types<
    kv_tag<char, int>,
//  __half breaks down currently
//  kv_tag<int, rocprim::half>,
    kv_tag<unsigned int, double>,
    kv_tag<long long, float>,
    kv_tag<unsigned long long, rocprim::uint128_t>
>;

template<class Pair>
class KeyValuePairTest : public ::testing::Test
{
protected:
    using key_type   = typename Pair::key_type;
    using value_type = typename Pair::value_type;
    using kv_type    = rocprim::key_value_pair<key_type, value_type>;
};
TYPED_TEST_SUITE(KeyValuePairTest, TestPairs);

TYPED_TEST(KeyValuePairTest, MemberAccessTest)
{
    using K = typename TestFixture::key_type;
    using V = typename TestFixture::value_type;
    using kv_type = typename TestFixture::kv_type;

    K k = get_random_full_range<K>();
    V v = get_random_full_range<V>();

    kv_type kv{k, v};

    EXPECT_EQ(kv.key,   k);
    EXPECT_EQ(kv.value, v);
}

TYPED_TEST(KeyValuePairTest, EqualityOperatorTest)
{
    using K = typename TestFixture::key_type;
    using V = typename TestFixture::value_type;
    using kv_type = typename TestFixture::kv_type;

    K k = get_random_full_range<K>();
    V v = get_random_full_range<V>();

    kv_type kv1{k, v};
    kv_type kv2{k, v};

    K k_diff;
    V v_diff;

    do
    {
        k_diff = get_random_full_range<K>();
    } while(k_diff == k);
    
    do
    {
        v_diff = get_random_full_range<V>();
    } while(v_diff == v);

    kv_type kv_diff_key{k_diff, v};
    kv_type kv_diff_val{k, v_diff};

    EXPECT_TRUE (kv1 == kv2);
    EXPECT_FALSE(kv1 != kv2);

    EXPECT_FALSE(kv1 == kv_diff_key);
    EXPECT_TRUE (kv1 != kv_diff_key);

    EXPECT_FALSE(kv1 == kv_diff_val);
    EXPECT_TRUE (kv1 != kv_diff_val);
}
