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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_BY_KEY_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../types/tuple.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class V, class K, class BinaryFunction, class KCompare>
struct scan_by_key_wrapper_op
{
    #ifdef __cpp_lib_is_invocable
    using value_type = typename std::invoke_result<BinaryFunction, V, V>::type;
    #else
    using value_type = typename std::result_of<BinaryFunction(V, V)>::type;
    #endif

    ROCPRIM_HOST_DEVICE inline
    scan_by_key_wrapper_op() = default;

    ROCPRIM_HOST_DEVICE inline
    scan_by_key_wrapper_op(BinaryFunction scan_op, KCompare compare_keys_op)
        : scan_op_(scan_op), compare_keys_op_(compare_keys_op)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~scan_by_key_wrapper_op() = default;

    ROCPRIM_HOST_DEVICE inline
    rocprim::tuple<value_type, K> operator()(const rocprim::tuple<value_type, K>& t1,
                                             const rocprim::tuple<value_type, K>& t2) const
    {
        if(compare_keys_op_(rocprim::get<1>(t1), rocprim::get<1>(t2)))
        {
            return rocprim::make_tuple(
                scan_op_(rocprim::get<0>(t1), rocprim::get<0>(t2)),
                rocprim::get<1>(t2)
            );
        }
        return t2;
    }

private:
    BinaryFunction scan_op_;
    KCompare compare_keys_op_;
};

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_BY_KEY_HPP_
