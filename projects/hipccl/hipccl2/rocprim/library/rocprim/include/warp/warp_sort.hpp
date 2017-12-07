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

#ifndef ROCPRIM_WARP_WARP_SORT_HPP_
#define ROCPRIM_WARP_WARP_SORT_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize
>
class warp_sort_shuffle
{
private:
    template<class BinaryFunction>
    T shuffle_swap(const T a, int mask, int dir, BinaryFunction sort_function) [[hc]]
    {
        T b = warp_shuffle_xor(a, mask, WarpSize);
        return sort_function(a, b) == dir ? b : a;
    }
    
    template<int warp, class BinaryFunction>
    typename std::enable_if<!(WarpSize > warp), T>::type
    swap(const T a, int mask, int dir, BinaryFunction sort_function) [[hc]]
    {
        return a;
    }
    
    template<int warp, class BinaryFunction>
    typename std::enable_if<(WarpSize > warp), T>::type
    swap(const T a, int mask, int dir, BinaryFunction sort_function) [[hc]]
    {
        return shuffle_swap(a, mask, dir, sort_function);
    }
    
public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    typedef detail::empty_type storage;

    template<class BinaryFunction>
    T sort(T thread_value, BinaryFunction sort_function) [[hc]]
    {
        unsigned int id = lane_id();
        thread_value = swap<2, BinaryFunction>(thread_value, 0x01, 
                                               get_bit(id, 1) ^ get_bit(id, 0),
                                               sort_function);
        
        thread_value = swap<4, BinaryFunction>(thread_value, 0x02, 
                                               get_bit(id, 2) ^ get_bit(id, 1),
                                               sort_function);
        thread_value = swap<4, BinaryFunction>(thread_value, 0x01, 
                                               get_bit(id, 2) ^ get_bit(id, 0),
                                               sort_function);
        
        thread_value = swap<8, BinaryFunction>(thread_value, 0x04, 
                                               get_bit(id, 3) ^ get_bit(id, 2),
                                               sort_function);
        thread_value = swap<8, BinaryFunction>(thread_value, 0x02, 
                                               get_bit(id, 3) ^ get_bit(id, 1),
                                               sort_function);
        thread_value = swap<8, BinaryFunction>(thread_value, 0x01, 
                                               get_bit(id, 3) ^ get_bit(id, 0),
                                               sort_function);
        
        thread_value = swap<16, BinaryFunction>(thread_value, 0x08, 
                                                get_bit(id, 4) ^ get_bit(id, 3),
                                                sort_function);
        thread_value = swap<16, BinaryFunction>(thread_value, 0x04, 
                                                get_bit(id, 4) ^ get_bit(id, 2),
                                                sort_function);
        thread_value = swap<16, BinaryFunction>(thread_value, 0x02, 
                                                get_bit(id, 4) ^ get_bit(id, 1),
                                                sort_function);
        thread_value = swap<16, BinaryFunction>(thread_value, 0x01, 
                                                get_bit(id, 4) ^ get_bit(id, 0),
                                                sort_function);
        
        thread_value = swap<32, BinaryFunction>(thread_value, 0x10, 
                                                get_bit(id, 5) ^ get_bit(id, 4),
                                                sort_function);
        thread_value = swap<32, BinaryFunction>(thread_value, 0x08, 
                                                get_bit(id, 5) ^ get_bit(id, 3),
                                                sort_function);
        thread_value = swap<32, BinaryFunction>(thread_value, 0x04, 
                                                get_bit(id, 5) ^ get_bit(id, 2),
                                                sort_function);
        thread_value = swap<32, BinaryFunction>(thread_value, 0x02, 
                                                get_bit(id, 5) ^ get_bit(id, 1),
                                                sort_function);
        thread_value = swap<32, BinaryFunction>(thread_value, 0x01, 
                                                get_bit(id, 5) ^ get_bit(id, 0),
                                                sort_function);
        
        thread_value = swap<32, BinaryFunction>(thread_value, 0x20, 
                                                get_bit(id, 5),
                                                sort_function);
        thread_value = swap<16, BinaryFunction>(thread_value, 0x10, 
                                                get_bit(id, 4),
                                                sort_function);
        thread_value = swap<8, BinaryFunction>(thread_value, 0x08, 
                                               get_bit(id, 3),
                                               sort_function);
        thread_value = swap<4, BinaryFunction>(thread_value, 0x04, 
                                               get_bit(id, 2),
                                               sort_function);
        thread_value = swap<2, BinaryFunction>(thread_value, 0x02, 
                                               get_bit(id, 1),
                                               sort_function);
        
        thread_value = shuffle_swap<BinaryFunction>(thread_value, 0x01, 
                                                    get_bit(id, 0),
                                                    sort_function);
        
        return thread_value;
    }

    template<class BinaryFunction>
    T sort(T thread_value,
           storage& temporary_storage,
           BinaryFunction sort_function) [[hc]]
    {
        (void) temporary_storage;
        return sort(thread_value, sort_function);
    }
    
    template<class BinaryFunction>
    T sort_by_key(T thread_key, T thread_value, BinaryFunction sort_function) [[hc]]
    {
        //unsigned int id = lane_id();
                
        return thread_value;
    }

    template<class BinaryFunction>
    T sort_by_key(T thread_key, T thread_value,
           storage& temporary_storage,
           BinaryFunction sort_function) [[hc]]
    {
        (void) temporary_storage;
        return sort_by_key(thread_key, thread_value, sort_function);
    }
};
    
template<
    class T,
    unsigned int WarpSize
>
class warp_sort_shared_mem
{
public:
    static_assert(
        detail::is_power_of_two(WarpSize),
        "warp_sort is not implemented for WarpSizes that are not power of two."
    );

    typedef detail::empty_type storage;
};

// Select warp_sort implementation based WarpSize
template<class T, unsigned int WarpSize>
struct select_warp_sort_impl
{
    typedef typename std::conditional<
        // can we use shuffle-based implementation?
        detail::is_warpsize_shuffleable<WarpSize>::value,
        detail::warp_sort_shuffle<T, WarpSize>, // yes
        detail::warp_sort_shared_mem<T, WarpSize> // no
    >::type type;
};
    
} // end namespace detail

/// \brief Parallel sort primitive for warp.
template<
    class T,
    unsigned int WarpSize = warp_size()
>
class warp_sort : detail::select_warp_sort_impl<T, WarpSize>::type
{
    typedef typename detail::select_warp_sort_impl<T, WarpSize>::type base_type;

public:
    typedef typename base_type::storage storage;

    template<class BinaryFunction = ::rocprim::less<T>>
    T sort(T thread_value, BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        return base_type::sort(thread_value, sort_function);
    }

    template<class BinaryFunction = ::rocprim::less<T>>
    T sort(T thread_value,
           storage& temporary_storage,
           BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        return base_type::sort(
            thread_value, temporary_storage, sort_function
        );
    }
};

template<
    class T,
    class U,
    unsigned int WarpSize = warp_size()
>
class warp_sort_by_key : detail::select_warp_sort_impl<Pair<T, U>, WarpSize>::type
{
    typedef typename detail::select_warp_sort_impl<Pair<T, U>, WarpSize>::type base_type;

public:
    typedef typename base_type::storage storage;

    template<class BinaryFunction = ::rocprim::less<Pair<T, U>>>
    void sort(T & thread_key, U & thread_value, BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        Pair<T, U> pair(thread_key, thread_value);
        pair = base_type::sort(pair, sort_function);
        thread_key = pair.x;
        thread_value = pair.y;
    }

    template<class BinaryFunction = ::rocprim::less<Pair<T, U>>>
    void sort(T & thread_key, U & thread_value,
           storage& temporary_storage,
           BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        Pair<T, U> pair(thread_key, thread_value);
        pair = base_type::sort(
            pair, temporary_storage, sort_function
        );
        thread_key = pair.x;
        thread_value = pair.y;
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_WARP_SORT_HPP_
