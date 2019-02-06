// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DETAIL_RADIX_SORT_HPP_
#define ROCPRIM_DETAIL_RADIX_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../type_traits.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

// Encode and decode integral and floating point values for radix sort in such a way that preserves
// correct order of negative and positive keys (i.e. negative keys go before positive ones,
// which is not true for a simple reinterpetation of the key's bits).

template<class Key, class BitKey, class Enable = void>
struct radix_key_codec_integral { };

template<class Key, class BitKey>
struct radix_key_codec_integral<Key, BitKey, typename std::enable_if<::rocprim::is_unsigned<Key>::value>::type>
{
    using bit_key_type = BitKey;

    ROCPRIM_DEVICE inline
    static bit_key_type encode(Key key)
    {
        return *reinterpret_cast<bit_key_type *>(&key);
    }

    ROCPRIM_DEVICE inline
    static Key decode(bit_key_type bit_key)
    {
        return *reinterpret_cast<Key *>(&bit_key);
    }
};

template<class Key, class BitKey>
struct radix_key_codec_integral<Key, BitKey, typename std::enable_if<::rocprim::is_signed<Key>::value>::type>
{
    using bit_key_type = BitKey;

    static constexpr bit_key_type sign_bit = bit_key_type(1) << (sizeof(bit_key_type) * 8 - 1);

    ROCPRIM_DEVICE inline
    static bit_key_type encode(Key key)
    {
        return sign_bit ^ *reinterpret_cast<bit_key_type *>(&key);
    }

    ROCPRIM_DEVICE inline
    static Key decode(bit_key_type bit_key)
    {
        bit_key ^= sign_bit;
        return *reinterpret_cast<Key *>(&bit_key);
    }
};

template<class Key, class BitKey>
struct radix_key_codec_floating
{
    using bit_key_type = BitKey;

    static constexpr bit_key_type sign_bit = bit_key_type(1) << (sizeof(bit_key_type) * 8 - 1);

    ROCPRIM_DEVICE inline
    static bit_key_type encode(Key key)
    {
        bit_key_type bit_key = *reinterpret_cast<bit_key_type *>(&key);
        bit_key ^= (sign_bit & bit_key) == 0 ? sign_bit : bit_key_type(-1);
        return bit_key;
    }

    ROCPRIM_DEVICE inline
    static Key decode(bit_key_type bit_key)
    {
        bit_key ^= (sign_bit & bit_key) == 0 ? bit_key_type(-1) : sign_bit;
        return *reinterpret_cast<Key *>(&bit_key);
    }
};

template<class Key, class Enable = void>
struct radix_key_codec_base
{
    static_assert(sizeof(Key) == 0,
        "Only integral and floating point types supported as radix sort keys");
};

template<class Key>
struct radix_key_codec_base<
    Key,
    typename std::enable_if<::rocprim::is_integral<Key>::value>::type
> : radix_key_codec_integral<Key, typename std::make_unsigned<Key>::type> { };

template<>
struct radix_key_codec_base<bool>
{
    using bit_key_type = unsigned char;

    ROCPRIM_DEVICE inline
    static bit_key_type encode(bool key)
    {
        return static_cast<bit_key_type>(key);
    }

    ROCPRIM_DEVICE inline
    static bool decode(bit_key_type bit_key)
    {
        return static_cast<bool>(bit_key);
    }
};

template<>
struct radix_key_codec_base<::rocprim::half> : radix_key_codec_floating<::rocprim::half, unsigned short> { };

template<>
struct radix_key_codec_base<float> : radix_key_codec_floating<float, unsigned int> { };

template<>
struct radix_key_codec_base<double> : radix_key_codec_floating<double, unsigned long long> { };

template<class Key, bool Descending = false>
class radix_key_codec : protected radix_key_codec_base<Key>
{
    using base_type = radix_key_codec_base<Key>;

public:
    using bit_key_type = typename base_type::bit_key_type;

    ROCPRIM_DEVICE inline
    static bit_key_type encode(Key key)
    {
        bit_key_type bit_key = base_type::encode(key);
        return (Descending ? ~bit_key : bit_key);
    }

    ROCPRIM_DEVICE inline
    static Key decode(bit_key_type bit_key)
    {
        bit_key = (Descending ? ~bit_key : bit_key);
        return base_type::decode(bit_key);
    }
};

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_RADIX_SORT_HPP_
