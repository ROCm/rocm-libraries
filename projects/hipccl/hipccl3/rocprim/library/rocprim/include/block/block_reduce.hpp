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

#ifndef ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
#define ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_reduce_warp_reduce.hpp"

/// \addtogroup collectiveblockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for block_reduce primitive.
enum class block_reduce_algorithm
{
    /// \brief A warp_reduce based algorithm.
    using_warp_reduce,

    /// \brief Default block_reduce algorithm.
    default_algorithm = using_warp_reduce,
};

namespace detail
{

// Selector for block_reduce algorithm which gives block reduce implementation
// type based on passed block_reduce_algorithm enum
template<block_reduce_algorithm Algorithm>
struct select_block_reduce_impl;

template<>
struct select_block_reduce_impl<block_reduce_algorithm::using_warp_reduce>
{
    template<class T, unsigned int BlockSize>
    using type = block_reduce_warp_reduce<T, BlockSize>;
};

} // end namespace detail

template<
    class T,
    unsigned int BlockSize,
    block_reduce_algorithm Algorithm = block_reduce_algorithm::default_algorithm
>
class block_reduce
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_reduce_impl<Algorithm>::template type<T, BlockSize>
#endif
{
    using base_type = typename detail::select_block_reduce_impl<Algorithm>::template type<T, BlockSize>;
public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt> in HIP or \p tile_static in HC. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    using storage_type = typename base_type::storage_type;

    template<class BinaryFunction = ::rocprim::plus<T>>
    void reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction()) [[hc]]
    {
        base_type::reduce(input, output, storage, reduce_op);
    }
    
    template<class BinaryFunction = ::rocprim::plus<T>>
    void reduce(T input,
                T& output,
                BinaryFunction reduce_op = BinaryFunction()) [[hc]]
    {
        base_type::reduce(input, output, reduce_op);
    }
    
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction()) [[hc]]
    {
        base_type::reduce(input, output, storage, reduce_op);
    }
    
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                BinaryFunction reduce_op = BinaryFunction()) [[hc]]
    {
        base_type::reduce(input, output, reduce_op);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group collectiveblockmodule

#endif // ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
