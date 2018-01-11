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

#ifndef ROCPRIM_WARP_WARP_REDUCE_HPP_
#define ROCPRIM_WARP_WARP_REDUCE_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/warp_reduce_shuffle.hpp"
#include "detail/warp_reduce_shared_mem.hpp"

/// \addtogroup collectivewarpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Select warp_scan implementation based WarpSize
template<class T, unsigned int WarpSize>
struct select_warp_reduce_impl
{
    typedef typename std::conditional<
        // can we use shuffle-based implementation?
        detail::is_warpsize_shuffleable<WarpSize>::value,
        detail::warp_reduce_shuffle<T, WarpSize>, // yes
        detail::warp_reduce_shared_mem<T, WarpSize> // no
    >::type type;
};

} // end namespace detail

template<
    class T,
    unsigned int WarpSize = warp_size(),
    bool UseAllReduce = false
>
class warp_reduce
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_warp_reduce_impl<T, WarpSize>::type
#endif
{
    using base_type = typename detail::select_warp_reduce_impl<T, WarpSize>::type;

public:
    using storage_type = typename base_type::storage_type;
    
    void sum(T input,
             T& output,
             storage_type& storage) [[hc]]
    {
        reduce<UseAllReduce>(input, output, storage, ::rocprim::plus<T>());
    }
    
    void sum(T input,
             T& output,
             int valid_items,
             storage_type& storage) [[hc]]
    {
        reduce<UseAllReduce>(input, output, valid_items, storage, ::rocprim::plus<T>());
    }
    
    template<class BinaryFunction = ::rocprim::plus<T>>
    void reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction()) [[hc]]
    {
        reduce<UseAllReduce>(input, output, storage, reduce_op);
    }
    
    template<class BinaryFunction = ::rocprim::plus<T>>
    void reduce(T input,
                    T& output,
                    int valid_items,
                    storage_type& storage,
                    BinaryFunction reduce_op = BinaryFunction()) [[hc]]
    {
        reduce<UseAllReduce>(input, output, valid_items, storage, reduce_op);
    }
    
private:
    template<bool Switch, class BinaryFunction>
    typename std::enable_if<(Switch == false)>::type
    reduce(T input,
           T& output,
           storage_type& storage,
           BinaryFunction reduce_op) [[hc]]
    {
        base_type::reduce(input, output, storage, reduce_op);
    }
    
    template<bool Switch, class BinaryFunction>
    typename std::enable_if<(Switch == true)>::type
    reduce(T input,
           T& output,
           storage_type& storage,
           BinaryFunction reduce_op) [[hc]]
    {
        base_type::all_reduce(input, output, storage, reduce_op);
    }
    
    template<bool Switch, class BinaryFunction>
    typename std::enable_if<(Switch == false)>::type
    reduce(T input,
           T& output,
           int valid_items,
           storage_type& storage,
           BinaryFunction reduce_op) [[hc]]
    {
        base_type::reduce(input, output, valid_items, storage, reduce_op);
    }
    
    template<bool Switch, class BinaryFunction>
    typename std::enable_if<(Switch == true)>::type
    reduce(T input,
           T& output,
           int valid_items,
           storage_type& storage,
           BinaryFunction reduce_op) [[hc]]
    {
        base_type::all_reduce(input, output, valid_items, storage, reduce_op);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group collectivewarpmodule

#endif // ROCPRIM_WARP_WARP_REDUCE_HPP_
