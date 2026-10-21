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

#ifndef HIPCUB_ROCPRIM_ITERATOR_TEX_OBJ_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_TEX_OBJ_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"

#include "iterator_category.hpp"

#include <rocprim/iterator/texture_cache_iterator.hpp> // IWYU pragma: export

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    typename OffsetT = std::ptrdiff_t
>
class TexObjInputIterator : public ::rocprim::texture_cache_iterator<T, OffsetT>
{
public:
    using iterator_category = typename detail::IteratorCategory<
        typename rocprim::texture_cache_iterator<T, OffsetT>::value_type,
        typename rocprim::texture_cache_iterator<T, OffsetT>::reference,
        false>::type; ///< The iterator category

    template<class Qualified>
    inline
    hipError_t BindTexture(Qualified* ptr,
                           size_t bytes = size_t(-1),
                           size_t texture_offset = 0)
    {
        return ::rocprim::texture_cache_iterator<T, OffsetT>::bind_texture(ptr, bytes, texture_offset);
    }

    inline hipError_t UnbindTexture()
    {
        return ::rocprim::texture_cache_iterator<T, OffsetT>::unbind_texture();
    }

    HIPCUB_HOST_DEVICE inline
    ~TexObjInputIterator() = default;

    HIPCUB_HOST_DEVICE inline TexObjInputIterator()
        : ::rocprim::texture_cache_iterator<T, OffsetT>()
    {}

    HIPCUB_HOST_DEVICE inline TexObjInputIterator(
        const ::rocprim::texture_cache_iterator<T, OffsetT> other)
        : ::rocprim::texture_cache_iterator<T, OffsetT>(other)
    {}
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_TEX_OBJ_INPUT_ITERATOR_HPP_
