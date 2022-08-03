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

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit, bool ShiftLess = (StartBit == 0 && EndBit == sizeof(Key) * 8), class Enable = void>
struct key_comparator {};

template <class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, false, typename std::enable_if<std::is_integral<Key>::value>::type>
{
    static constexpr Key radix_mask_upper  = (Key(1) << EndBit) - 1;
    static constexpr Key radix_mask_bottom = (Key(1) << StartBit) - 1;
    static constexpr Key radix_mask = radix_mask_upper ^ radix_mask_bottom;

    bool operator()(const Key& lhs, const Key& rhs)
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
                      false,
                      typename std::enable_if<!std::is_integral<Key>::value>::type>
{
    // Floating-point types do not support StartBit and EndBit.
    bool operator()(const Key&, const Key&)
    {
        return false;
    }
};

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, true, typename std::enable_if<std::is_integral<Key>::value>::type>
{
    bool operator()(const Key& lhs, const Key& rhs)
    {
        return Descending ? (rhs < lhs) : (lhs < rhs);
    }
};

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, true, typename std::enable_if<!std::is_integral<Key>::value>::type>
{
    bool operator()(const Key& lhs, const Key& rhs)
    {
        using namespace std;
        if(is_floating_nan_host(lhs) && is_floating_nan_host(rhs) && signbit(lhs) == signbit(rhs))
        {
            return false;
        }
        if(Descending){
            if(is_floating_nan_host(lhs))
                return !signbit(lhs);
            if(is_floating_nan_host(rhs))
                return signbit(rhs);
            return (rhs < lhs);
        }else{
            if(is_floating_nan_host(lhs))
                return signbit(lhs);
            if(is_floating_nan_host(rhs))
                return !signbit(rhs);
            return (lhs < rhs);
        }
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
