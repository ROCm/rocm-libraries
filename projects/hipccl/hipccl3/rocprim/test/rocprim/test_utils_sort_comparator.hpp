// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_UTILS_SORT_COMPARATOR_HPP_
#define TEST_UTILS_SORT_COMPARATOR_HPP_

#include <rocprim/type_traits.hpp>

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"

namespace test_utils
{

template<class T>
constexpr bool is_floating_nan_host(const T& a)
{
    return (a != a);
}

template<class Key,
         bool         Descending,
         unsigned int StartBit,
         unsigned int EndBit,
         class Enable = void>
struct key_comparator
{};

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key,
                      Descending,
                      StartBit,
                      EndBit,
                      typename std::enable_if<rocprim::is_integral<Key>::value>::type>
{
    static constexpr Key radix_mask_upper
        = EndBit == 8 * sizeof(Key) ? ~Key(0) : (Key(1) << EndBit) - 1;
    static constexpr Key radix_mask_bottom = (Key(1) << StartBit) - 1;
    static constexpr Key radix_mask = radix_mask_upper ^ radix_mask_bottom;

    bool operator()(const Key& lhs, const Key& rhs) const
    {
        Key l = lhs & radix_mask;
        Key r = rhs & radix_mask;
        return Descending ? (r < l) : (l < r);
    }
};

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key,
                      Descending,
                      StartBit,
                      EndBit,
                      typename std::enable_if<rocprim::is_floating_point<Key>::value>::type>
{
    using unsigned_bits_type = typename rocprim::get_unsigned_bits_type<Key>::unsigned_type;

    bool operator()(const Key& lhs, const Key& rhs) const
    {
        return key_comparator<unsigned_bits_type, Descending, StartBit, EndBit>()(
            this->to_bits(lhs),
            this->to_bits(rhs));
    }

    unsigned_bits_type to_bits(const Key& key) const
    {
        unsigned_bits_type bit_key;
        memcpy(&bit_key, &key, sizeof(Key));

        // Remove signed zero, this case is supposed to be treated the same as
        // unsigned zero in rocprim sorting algorithms.
        constexpr unsigned_bits_type minus_zero = unsigned_bits_type{1} << (8 * sizeof(Key) - 1);
        // Positive and negative zero should compare the same.
        if(bit_key == minus_zero)
        {
            bit_key = 0;
        }
        // Flip bits mantissa and exponent if the key is negative, so as to make
        // 'more negative' values compare before 'less negative'.
        if(bit_key & minus_zero)
        {
            bit_key ^= ~minus_zero;
        }
        // Make negatives compare before positives.
        bit_key ^= minus_zero;
        return bit_key;
    }
};

template<class Key, class Value, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_value_comparator
{
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return key_comparator<Key, Descending, StartBit, EndBit>()(lhs.first, rhs.first);
    }
};

template <bool Descending>
struct key_comparator<rocprim::half, Descending, 0, sizeof(rocprim::half) * 8>
{
    bool operator()(const rocprim::half& lhs, const rocprim::half& rhs)
    {
        // HIP's half doesn't have __host__ comparison operators, use floats instead
        return key_comparator<float, Descending, 0, sizeof(float) * 8>()(lhs, rhs);
    }
};

template <bool Descending>
struct key_comparator<rocprim::bfloat16, Descending, 0, sizeof(rocprim::bfloat16) * 8>
{
    bool operator()(const rocprim::bfloat16& lhs, const rocprim::bfloat16& rhs)
    {
        // HIP's bfloat16 doesn't have __host__ comparison operators, use floats instead
        return key_comparator<float, Descending, 0, sizeof(float) * 8>()(lhs, rhs);
    }
};

}
#endif // TEST_UTILS_SORT_COMPARATOR_HPP_
