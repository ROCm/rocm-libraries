// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
inline T get_random_full_range(unsigned int seed = std::rand())
{
    return test_utils::get_random_value<T>(rocprim::numeric_limits<T>::min(),
                                           rocprim::numeric_limits<T>::max(),
                                           seed);
}

template<typename T>
class DoubleBufferTest : public ::testing::Test
{
protected:
    T value1{};
    T value2{};

    rocprim::double_buffer<T> db{&value1, &value2};
};

TYPED_TEST_SUITE(DoubleBufferTest, Params);

TYPED_TEST(DoubleBufferTest, TestDoubleBuffer)
{
    using T = TypeParam;

    // Test default construction
    rocprim::double_buffer<T> db_default;
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(db_default.current(), static_cast<T*>(nullptr)));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(db_default.alternate(), static_cast<T*>(nullptr)));

    // Test current buffer
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(this->db.current(), &this->value1));

    // Test alternate buffer
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(this->db.alternate(), &this->value2));

    // Test swap buffers
    this->db.swap();
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(this->db.current(), &this->value2));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(this->db.alternate(), &this->value1));
}

template<typename T>
class FutureValueTest : public ::testing::Test
{
protected:
    T value{};

    rocprim::future_value<T> fv{&value};
};

TYPED_TEST_SUITE(FutureValueTest, Params);

TYPED_TEST(FutureValueTest, TestFutureValue)
{
    using T = TypeParam;

    // Test const future value
    const auto cfv = this->fv;
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(static_cast<T>(cfv), this->value));

    // Test value access
    this->value = get_random_full_range<T>();
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(static_cast<TypeParam>(this->fv), this->value));

    // Test plain input value
    T val = get_random_full_range<T>();
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(rocprim::detail::get_input_value(val), val));

    // Test future input value
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(rocprim::detail::get_input_value(this->fv), this->value));
}

template<class K, class V>
struct kv_tag
{
    using key_type   = K;
    using value_type = V;
};

using TestPairs = ::testing::Types<kv_tag<char, int>,
                                   //  __half breaks down currently
                                   //  kv_tag<int, rocprim::half>,
                                   kv_tag<unsigned int, double>,
                                   kv_tag<long long, float>,
                                   kv_tag<unsigned long long, rocprim::uint128_t>>;

template<class Pair>
class KeyValuePairTest : public ::testing::Test
{
protected:
    using key_type   = typename Pair::key_type;
    using value_type = typename Pair::value_type;
    using kv_type    = rocprim::key_value_pair<key_type, value_type>;
};
TYPED_TEST_SUITE(KeyValuePairTest, TestPairs);

TYPED_TEST(KeyValuePairTest, TestKeyValuePair)
{
    using K       = typename TestFixture::key_type;
    using V       = typename TestFixture::value_type;
    using kv_type = typename TestFixture::kv_type;

    K k = get_random_full_range<K>();
    V v = get_random_full_range<V>();

    kv_type kv1{k, v};
    kv_type kv2{k, v};

    // Test value access
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(kv1.key, k));
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(kv1.value, v));

    K k_diff;
    V v_diff;

    do
    {
        k_diff = get_random_full_range<K>();
    }
    while(k_diff == k);

    do
    {
        v_diff = get_random_full_range<V>();
    }
    while(v_diff == v);

    kv_type kv_diff_key{k_diff, v};
    kv_type kv_diff_val{k, v_diff};

    // Test operator == and !=
    EXPECT_TRUE(kv1 == kv2);
    EXPECT_FALSE(kv1 != kv2);

    EXPECT_FALSE(kv1 == kv_diff_key);
    EXPECT_TRUE(kv1 != kv_diff_key);

    EXPECT_FALSE(kv1 == kv_diff_val);
    EXPECT_TRUE(kv1 != kv_diff_val);
}

template<class T>
class UninitializedArrayTest : public ::testing::Test
{
protected:
    static constexpr unsigned int Count = 10;
    using ua_type                       = rocprim::uninitialized_array<T, Count>;

    ua_type ua;
};
TYPED_TEST_SUITE(UninitializedArrayTest, Params);

TYPED_TEST(UninitializedArrayTest, EmplaceConstructsCorrectValue)
{
    // Test class traits
    using ua = typename TestFixture::ua_type;
    static_assert(!std::is_copy_constructible<ua>::value, "Should not be copy-constructible");
    static_assert(!std::is_copy_assignable<ua>::value, "Should not be copy-assignable");
    static_assert(std::is_move_constructible<ua>::value, "Should be move-constructible");
    static_assert(std::is_move_assignable<ua>::value, "Should be move-assignable");

    // Test emplace construction
    using V = TypeParam;
    for(unsigned int i = 0; i < TestFixture::Count; ++i)
    {
        V  val = get_random_full_range<V>();
        V& ref = this->ua.emplace(i, val); // Emplace construction
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(ref, val)); // Same value by reference
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(this->ua.get_unsafe_array()[i], val)); // Same value in array
    }

    // Test memory consistency
    V v0 = get_random_full_range<V>();
    this->ua.emplace(0, v0);

    auto& arr = this->ua.get_unsafe_array();
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(&arr[0], &this->ua.get_unsafe_array()[0])); // Same address
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(arr[0], v0)); // Same content

    for(unsigned int i = 0; i < TestFixture::Count; ++i)
    {
        this->ua.emplace(i, V(i + 1));
    }

    // Test move construction
    typename TestFixture::ua_type moved{std::move(this->ua)};
    auto&                         arr_moved = moved.get_unsafe_array();
    for(unsigned int i = 0; i < TestFixture::Count; ++i)
    {
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(arr_moved[i], V(i + 1)));
    }
}
