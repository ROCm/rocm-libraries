/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_ITERATOR_CONSTANT_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_CONSTANT_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include "iterator_category.hpp"
#include "iterator_wrapper.hpp"

#include <rocprim/iterator/constant_iterator.hpp> // IWYU pragma: export

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

template<class ValueType, class Difference = std::ptrdiff_t>
class HIPCUB_DEPRECATED_BECAUSE(
    "Use rocprim::constant_iterator or rocthrust::constant_iterator instead") ConstantInputIterator
    : public detail::IteratorWrapper<rocprim::constant_iterator<ValueType, Difference>,
                                     ConstantInputIterator<ValueType, Difference>>
{
    using Iterator = rocprim::constant_iterator<ValueType, Difference>;
    using Base = detail::IteratorWrapper<Iterator, ConstantInputIterator<ValueType, Difference>>;

public:
    using iterator_category = typename detail::IteratorCategory<typename Iterator::value_type,
                                                                typename Iterator::reference>::type;
    using self_type         = typename Iterator::self_type;

    __host__ __device__ __forceinline__ ConstantInputIterator(
        const typename Iterator::value_type value, const size_t index = 0)
        : Base(Iterator(value, index))
    {}

    // Cast from wrapped iterator to class itself
    __host__ __device__ __forceinline__ explicit ConstantInputIterator(Iterator iterator)
        : Base(iterator)
    {}
};

#endif

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_CONSTANT_INPUT_ITERATOR_HPP_
