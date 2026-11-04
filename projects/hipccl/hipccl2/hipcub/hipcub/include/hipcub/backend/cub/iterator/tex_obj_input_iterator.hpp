// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_CUB_ITERATOR_TEX_OBJ_INPUT_ITERATOR_HPP_
#define HIPCUB_CUB_ITERATOR_TEX_OBJ_INPUT_ITERATOR_HPP_

#include "../../../config.hpp"

#include <cub/iterator/tex_obj_input_iterator.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

template<typename T, typename OffsetT = std::ptrdiff_t>
class TexObjInputIterator : public ::cub::TexObjInputIterator<T, OffsetT>
{
public:
    template<class Qualified>
    inline hipError_t
        BindTexture(Qualified* ptr, size_t bytes = size_t(-1), size_t texture_offset = 0)
    {
        return hipCUDAErrorTohipError(
            ::cub::TexObjInputIterator<T, OffsetT>::BindTexture(ptr, bytes, texture_offset));
    }

    inline hipError_t UnbindTexture()
    {
        return hipCUDAErrorTohipError(::cub::TexObjInputIterator<T, OffsetT>::UnbindTexture());
    }

    HIPCUB_HOST_DEVICE inline TexObjInputIterator() : ::cub::TexObjInputIterator<T, OffsetT>() {}

    HIPCUB_HOST_DEVICE inline TexObjInputIterator(
        const ::cub::TexObjInputIterator<T, OffsetT> other)
        : ::cub::TexObjInputIterator<T, OffsetT>(other)
    {}
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_ITERATOR_TEX_OBJ_INPUT_ITERATOR_HPP_
