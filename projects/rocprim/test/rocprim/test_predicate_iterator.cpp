// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <common_test_header.hpp>

#include "../../common/predicate_iterator.hpp"
#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"
#include "rocprim/iterator/transform_iterator.hpp"

#include "test_utils.hpp"
#include <rocprim/device/device_transform.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/predicate_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

struct is_odd
{
    // While this can be "constexpr T(const T&) const", we want to verify that
    // it compiles without the constness.
    template<class T>
    __device__ __host__
    bool operator()(T& a)
    {
        return a % 2;
    }
};

template<int V>
struct set_to
{
    template<class T>
    __device__ __host__
    constexpr T
        operator()(const T&) const
    {
        return V;
    }
};

// Params for tests
template<class InputType>
struct RocprimPredicateIteratorParams
{
    using input_type = InputType;
};

template<class Params>
class RocprimPredicateIteratorTests : public ::testing::Test
{
public:
    using input_type             = typename Params::input_type;
    const bool debug_synchronous = false;
};

using RocprimPredicateIteratorTestsParams
    = ::testing::Types<RocprimPredicateIteratorParams<int>,
                       RocprimPredicateIteratorParams<unsigned int>,
                       RocprimPredicateIteratorParams<unsigned long>>;

TYPED_TEST_SUITE(RocprimPredicateIteratorTests, RocprimPredicateIteratorTestsParams);

TYPED_TEST(RocprimPredicateIteratorTests, Basic)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::input_type;
    using Iterator = rocprim::predicate_iterator<T*, T*, is_odd>;
    using Proxy    = typename Iterator::value_type;

    for(size_t seed_index = 0; seed_index < number_of_runs; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        std::vector<T> input = test_utils::get_random_data_wrapped<T>(10, 1, 100, seed_value);

        std::vector<T> flags = input; // using input itself to apply is_odd

        Iterator begin = rocprim::make_predicate_iterator(input.data(), flags.data(), is_odd{});
        Iterator mid   = begin + 5;
        Iterator end   = begin + 10;

        // Pre-increment
        Iterator it = begin;
        ++it;
        ASSERT_EQ(static_cast<T>(*it), input[1] % 2 ? input[1] : T{});

        // Post-increment
        Iterator post = it++;
        ASSERT_EQ(static_cast<T>(*post), input[1] % 2 ? input[1] : T{});
        ASSERT_EQ(static_cast<T>(*it), input[2] % 2 ? input[2] : T{});

        // Pre-decrement
        it = begin + 2;
        --it;
        ASSERT_EQ(static_cast<T>(*it), input[1] % 2 ? input[1] : T{});

        // Post-decrement
        Iterator post_dec = it--;
        ASSERT_EQ(static_cast<T>(*post_dec), input[1] % 2 ? input[1] : T{});
        ASSERT_EQ(static_cast<T>(*it), input[0] % 2 ? input[0] : T{});

        // operator+
        Iterator plus_it = begin + 3;
        ASSERT_EQ(static_cast<T>(*plus_it), input[3] % 2 ? input[3] : T{});
        Iterator plus_i_rev = 3 + begin;
        ASSERT_EQ(static_cast<T>(*plus_i_rev), input[3] % 2 ? input[3] : T{});

        // operator-
        Iterator minus_it = end - 3;
        ASSERT_EQ(static_cast<T>(*minus_it), input[7] % 2 ? input[7] : T{});

        // compound assignment +=
        Iterator a = begin;
        a += 4;
        ASSERT_EQ(static_cast<T>(*a), input[4] % 2 ? input[4] : T{});

        // compound assignment -=
        a -= 2;
        ASSERT_EQ(static_cast<T>(*a), input[2] % 2 ? input[2] : T{});

        // Subtraction of iterators (distance)
        ASSERT_EQ(end - begin, 10);
        ASSERT_EQ(mid - begin, 5);
        ASSERT_EQ(begin - mid, -5);

        // Indexing operator[]
        for(int i = 0; i < 10; ++i)
        {
            Proxy proxy    = begin[i];
            T     expected = input[i] % 2 ? input[i] : T{};
            ASSERT_EQ(static_cast<T>(proxy), expected);
        }

        // Comparisons
        ASSERT_TRUE(begin == begin);
        ASSERT_TRUE(begin != end);
        ASSERT_TRUE(begin < end);
        ASSERT_TRUE(end > begin);
        ASSERT_TRUE(begin <= begin);
        ASSERT_TRUE(begin <= end);
        ASSERT_TRUE(end >= begin);
        ASSERT_TRUE(end >= end);

        // Arrow operator (returns proxy, which we can test through conversion)
        Proxy p = *begin;
        ASSERT_EQ(static_cast<T>(p), input[0] % 2 ? input[0] : T{});

        // Assignment via proxy: only odd-indexed data should be modified
        for(int i = 0; i < 10; ++i)
        {
            begin[i] = T{999};
        }

        for(int i = 0; i < 10; ++i)
        {
            if(input[i] % 2)
            {
                ASSERT_EQ(input[i], T{999});
            }

            else
            {
                ASSERT_NE(input[i], T{999});
            }
        }
    }
}

TEST(RocprimPredicateIteratorTests, TypeTraits)
{
    using value_type = int;

    value_type* data{};
    bool*       mask{};

    auto m_it = rocprim::make_mask_iterator(data, mask);

    using m_it_t  = decltype(m_it);
    using proxy_t = m_it_t::proxy;

    static_assert(std::is_assignable<proxy_t, value_type>::value,
                  "discard type is not assignable with underlying type, even though it should be!");
    static_assert(std::is_assignable<decltype(*m_it), value_type>::value,
                  "iterator is not assignable with underlying type via dereference, even though it "
                  "should be!");
    static_assert(std::is_assignable<decltype(m_it[0]), value_type>::value,
                  "iterator is not assignablle with underlying type via array index, even though "
                  "is should be!");

    // Check if we can apply predicate iterator on a constant iterator
    auto c_it = rocprim::make_constant_iterator(0);
    auto p_it = rocprim::make_predicate_iterator(c_it, is_odd{});

    static_assert(
        std::is_convertible<decltype(*p_it), value_type>::value,
        "predicate iterator is not convertible to underlying type, even though it should be!");
}

// Test that we are only writing if predicate holds
TEST(RocprimPredicateIteratorTests, HostWrite)
{
    using T                      = int;
    static constexpr size_t size = 100;

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), 0);

    // Make iterator that only writes to odd values
    auto odd_it = rocprim::make_predicate_iterator(data.begin(), is_odd{});

    // Increment all values in that iterator
    std::transform(data.begin(), data.end(), odd_it, [](auto v) { return v + 1; });

    // Such that none of data is odd
    ASSERT_TRUE(std::none_of(data.begin(), data.end(), is_odd{}));
}

// Test that we are only reading if predicate holds, excluding the required read for the predicate
TEST(RocprimPredicateIteratorTests, HostRead)
{
    using T                      = int;
    static constexpr size_t size = 100;

    auto is_odd_or_default = [](T v) { return v % 2 || v == T{}; };

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), 0);

    // Make iterator that only reads odd values
    auto odd_it = rocprim::make_predicate_iterator(data.begin(), is_odd{});

    // Read all values from that iterator
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = odd_it[i];
    }

    // Such that all of data is odd or default
    ASSERT_TRUE(std::all_of(data.begin(), data.end(), is_odd_or_default));
}

// Test that we are only writing if predicate holds
TEST(RocprimPredicateIteratorTests, HostMaskWrite)
{
    using T                      = int;
    static constexpr size_t size = 100;

    std::vector<T>    data(size);
    std::vector<bool> mask = test_utils::get_random_data<bool>(size, false, true, 0);
    std::iota(data.begin(), data.end(), 0);
    test_utils::get_random_data<bool>(size, false, true, 0);

    using identity_type = typename std::iterator_traits<decltype(mask.begin())>::value_type;

    auto masked_it = rocprim::make_predicate_iterator(data.begin(),
                                                      mask.begin(),
                                                      rocprim::identity<identity_type>{});
    std::transform(data.begin(), data.end(), masked_it, set_to<-1>{});

    for(size_t i = 0; i < size; ++i)
    {
        if(mask[i])
        {
            ASSERT_EQ(data[i], -1);
        }
        else
        {
            ASSERT_EQ(data[i], i);
        }
    }
}

// Test that we are only reading if predicate holds, excluding the required read for the predicate
TEST(RocprimPredicateIteratorTests, HostMaskRead)
{
    using T                      = int;
    static constexpr size_t size = 100;

    std::vector<T>    data(size);
    std::vector<bool> mask = test_utils::get_random_data<bool>(size, false, true, 0);
    std::iota(data.begin(), data.end(), 0);

    auto masked_it = rocprim::make_mask_iterator(data.begin(), mask.begin());

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = masked_it[i];
    }

    for(size_t i = 0; i < size; ++i)
    {
        if(mask[i])
        {
            ASSERT_EQ(data[i], i);
        }
        else
        {
            ASSERT_EQ(data[i], T{});
        }
    }
}

// Test if predicate iterator can be used on device
TEST(RocprimPredicateIteratorTests, DeviceInplace)
{
    using T         = int;
    using predicate = is_odd;
    using transform = common::increment_by<5>;

    constexpr size_t size = 100;

    std::vector<T> h_data(size);
    std::iota(h_data.begin(), h_data.end(), 0);

    common::device_ptr<T> d_data(h_data);

    auto w_it = rocprim::make_predicate_iterator(d_data.get(), predicate{});

    HIP_CHECK(rocprim::transform(d_data.get(), w_it, size, transform{}));
    h_data = d_data.load();

    for(T i = 0; i < T{size}; ++i)
    {
        if(predicate{}(i))
        {
            ASSERT_EQ(h_data[i], transform{}(i));
        }
        else
        {
            ASSERT_EQ(h_data[i], i);
        }
    }
}

// Test if predicate iterator can be used on device
TEST(RocprimPredicateIteratorTests, DeviceRead)
{
    using T         = int;
    using predicate = is_odd;
    using transform = common::increment_by<5>;

    constexpr size_t size = 100;

    std::vector<T> h_data(size);
    std::iota(h_data.begin(), h_data.end(), 0);

    common::device_ptr<T> d_input(h_data);
    common::device_ptr<T> d_output(size);

    auto t_it = rocprim::make_transform_iterator(d_input.get(), transform{});
    auto r_it = rocprim::make_predicate_iterator(t_it, d_input.get(), predicate{});

    using identity_type = typename std::iterator_traits<decltype(r_it)>::value_type;

    HIP_CHECK(rocprim::transform(r_it, d_output.get(), size, rocprim::identity<identity_type>{}));

    h_data = d_output.load();

    for(T i = 0; i < T{size}; ++i)
    {
        if(predicate{}(i))
        {
            ASSERT_EQ(h_data[i], transform{}(i));
        }
        else
        {
            ASSERT_EQ(h_data[i], T{});
        }
    }
}
