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
