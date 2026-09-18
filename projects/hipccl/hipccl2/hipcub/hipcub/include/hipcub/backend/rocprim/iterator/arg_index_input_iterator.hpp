/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"

#include "iterator_category.hpp"
#include "iterator_wrapper.hpp"

#include <rocprim/iterator/arg_index_iterator.hpp> // IWYU pragma: export

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

template<class InputIterator,
         class Difference     = std::ptrdiff_t,
         class InputValueType = typename std::iterator_traits<InputIterator>::value_type>
class ArgIndexInputIterator
    : public detail::IteratorWrapper<
          rocprim::arg_index_iterator<InputIterator, Difference, InputValueType>,
          ArgIndexInputIterator<InputIterator, Difference, InputValueType>>
{
    using Iterator = rocprim::arg_index_iterator<InputIterator, Difference, InputValueType>;
    using Base
        = detail::IteratorWrapper<Iterator,
                                  ArgIndexInputIterator<InputIterator, Difference, InputValueType>>;

public:
    using iterator_category = typename detail::IteratorCategory<typename Iterator::value_type,
                                                                typename Iterator::reference>::type;
    using self_type         = typename Iterator::self_type;

    __host__ __device__ __forceinline__ ArgIndexInputIterator(
        InputIterator iterator, typename Iterator::difference_type offset = 0)
        : Base(Iterator(iterator, offset))
    {}

    // Cast from wrapped iterator to class itself
    __host__ __device__ __forceinline__ explicit ArgIndexInputIterator(Iterator iterator)
        : Base(iterator)
    {}
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_
