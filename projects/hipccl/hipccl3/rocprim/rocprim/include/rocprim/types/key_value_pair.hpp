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

#ifndef ROCPRIM_TYPES_KEY_VALUE_PAIR_HPP_
#define ROCPRIM_TYPES_KEY_VALUE_PAIR_HPP_

#include "../config.hpp"

/// \addtogroup utilsmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

template<
    class Key_,
    class Value_
>
struct key_value_pair
{
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using Key = Key_;
    using Value = Value_;
    #endif

    using key_type = Key_;
    using value_type = Value_;

    key_type key;
    value_type value;

    ROCPRIM_HOST_DEVICE inline
    key_value_pair() = default;

    ROCPRIM_HOST_DEVICE inline
    ~key_value_pair() = default;

    ROCPRIM_HOST_DEVICE inline
    key_value_pair(const key_type key, const value_type value) : key(key), value(value)
    {
    }

    #if __hcc_major__ < 1 || __hcc_major__ == 1 && __hcc_minor__ < 2
    ROCPRIM_HOST_DEVICE inline
    key_value_pair& operator =(const key_value_pair& kvb)
    {
        key = kvb.key;
        value = kvb.value;
        return *this;
    }
    #endif

    ROCPRIM_HOST_DEVICE inline
    bool operator !=(const key_value_pair& kvb)
    {
        return (key != kvb.key) || (value != kvb.value);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule

#endif // ROCPRIM_TYPES_KEY_VALUE_PAIR_HPP_
