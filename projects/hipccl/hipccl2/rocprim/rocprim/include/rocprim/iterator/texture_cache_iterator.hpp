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

#ifndef ROCPRIM_ITERATOR_TEXTURE_CACHE_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_TEXTURE_CACHE_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
// Takes a scalar type T and matches to a texture type based on NumElements.
template <class T, unsigned int NumElements>
struct make_texture_type
{
    using type = void;
};

template <>
struct make_texture_type<char, 1>
{
    using type = char;
};

template <>
struct make_texture_type<int, 1>
{
    using type = int;
};

template <>
struct make_texture_type<short, 1>
{
    using type = short;
};

#define DEFINE_MAKE_TEXTURE_TYPE(base, suffix) \
\
template<> \
struct make_texture_type<base, suffix> \
{ \
    using type = ::base##suffix; \
};

DEFINE_MAKE_TEXTURE_TYPE(char, 2);
DEFINE_MAKE_TEXTURE_TYPE(char, 4);
DEFINE_MAKE_TEXTURE_TYPE(int, 2);
DEFINE_MAKE_TEXTURE_TYPE(int, 4);
DEFINE_MAKE_TEXTURE_TYPE(short, 2);
DEFINE_MAKE_TEXTURE_TYPE(short, 4);

// Selects an appropriate vector_type based on the input T and size N.
// The byte size is calculated and used to select an appropriate vector_type.
template<class T>
struct match_texture_type
{
    static constexpr unsigned int size = sizeof(T);
    using texture_base_type =
        typename std::conditional<
            sizeof(T) >= 4,
            int,
            typename std::conditional<
                sizeof(T) >= 2,
                short,
                char
            >::type
        >::type;

    using texture_4 = typename make_texture_type<texture_base_type, 4>::type;
    using texture_2 = typename make_texture_type<texture_base_type, 2>::type;
    using texture_1 = typename make_texture_type<texture_base_type, 1>::type;

    using type =
        typename std::conditional<
            size % sizeof(texture_4) == 0,
            texture_4,
            typename std::conditional<
                size % sizeof(texture_2) == 0,
                texture_2,
                texture_1
            >::type
        >::type;
};
}

/// \class texture_cache_iterator
/// \brief A random-access input (read-only) iterator adaptor for dereferencing array values
/// through texture cache.
///
/// \par Overview
/// * A texture_cache_iterator wraps a device pointer of type T, where values are obtained
/// by dereferencing through texture cache.
/// * Can be exchanged and manipulated within and between host and device functions.
/// * Can only be constructed within host functions, and can only be dereferenced within
/// device functions.
/// * Accepts any data type from memory, and loads through texture cache.
/// 
/// \tparam T - type of value that can be obtained by dereferencing the iterator.
/// \tparam Difference - a type used for identify distance between iterators.
template<
    class T,
    class Difference = std::ptrdiff_t
>
class texture_cache_iterator
{
public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename std::remove_const<T>::type;
    /// \brief A reference type of the type iterated over (\p value_type).
    using reference = const value_type&;
    /// \brief A pointer type of the type iterated over (\p value_type).
    using pointer = const value_type*;
    /// A type used for identify distance between iterators.
    using difference_type = Difference;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = texture_cache_iterator;
#endif

    ROCPRIM_HOST_DEVICE inline
    ~texture_cache_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    texture_cache_iterator()
        : ptr(NULL), texture_offset(0), texture_object(0)
    {
    }

    template<class Qualified>
    inline
    hipError_t bind_texture(Qualified* ptr,
                            size_t bytes = size_t(-1),
                            size_t texture_offset = 0)
    {
        this->ptr = const_cast<typename std::remove_cv<Qualified>::type*>(ptr);
        this->texture_offset = texture_offset;

        hipChannelFormatDesc channel_desc = hipCreateChannelDesc<texture_type>();
        hipResourceDesc resourse_desc;
        hipTextureDesc texture_desc;
        memset(&resourse_desc, 0, sizeof(hipResourceDesc));
        memset(&texture_desc, 0, sizeof(hipTextureDesc));
        resourse_desc.resType = hipResourceTypeLinear;
        resourse_desc.res.linear.devPtr = this->ptr;
        resourse_desc.res.linear.desc = channel_desc;
        resourse_desc.res.linear.sizeInBytes = bytes;
        texture_desc.readMode = hipReadModeElementType;

        return hipCreateTextureObject(&texture_object, &resourse_desc, &texture_desc, NULL);
    }

    inline
    hipError_t unbind_texture()
    {
        return hipDestroyTextureObject(texture_object);
    }

    ROCPRIM_HOST_DEVICE inline
    texture_cache_iterator& operator++()
    {
        ptr++;
        texture_offset++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    texture_cache_iterator operator++(int)
    {
        texture_cache_iterator old_tc = *this;
        ptr++;
        texture_offset++;
        return old_tc;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        #ifndef __HIP_DEVICE_COMPILE__
        return ptr[texture_offset];
        #else
        texture_type words[multiple];

        #pragma unroll
        for(unsigned int i = 0; i < multiple; i++)
        {
            tex1Dfetch(
                &words[i],
                texture_object,
                (texture_offset * multiple) + i
            );
        }

        return *reinterpret_cast<value_type*>(words);
        #endif
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &(*(*this));
    }

    ROCPRIM_HOST_DEVICE inline
    texture_cache_iterator operator+(difference_type distance) const
    {
        self_type retval;
        retval.ptr = ptr + distance;
        retval.texture_object = texture_object;
        retval.texture_offset = texture_offset + distance;
        return retval;
    }

    ROCPRIM_HOST_DEVICE inline
    texture_cache_iterator& operator+=(difference_type distance)
    {
        ptr += distance;
        texture_offset += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    texture_cache_iterator operator-(difference_type distance) const
    {
        self_type retval;
        retval.ptr = ptr - distance;
        retval.texture_object = texture_object;
        retval.texture_offset = texture_offset - distance;
        return retval;
    }

    ROCPRIM_HOST_DEVICE inline
    texture_cache_iterator& operator-=(difference_type distance)
    {
        ptr -= distance;
        texture_offset -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(texture_cache_iterator other) const
    {
        return ptr - other.ptr;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        texture_cache_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(texture_cache_iterator other) const
    {
        return (ptr == other.ptr) && (texture_offset == other.texture_offset);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(texture_cache_iterator other) const
    {
        return (ptr != other.ptr) || (texture_offset != other.texture_offset);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(texture_cache_iterator other) const
    {
        return (ptr - other.ptr) > 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(texture_cache_iterator other) const
    {
        return (ptr - other.ptr) >= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(texture_cache_iterator other) const
    {
        return (ptr - other.ptr) < 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(texture_cache_iterator other) const
    {
        return (ptr - other.ptr) <= 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const texture_cache_iterator& /* iter */)
    {
        return os;
    }

private:
    using texture_type = typename ::rocprim::detail::match_texture_type<T>::type;
    static constexpr unsigned int multiple = sizeof(T) / sizeof(texture_type);
    value_type* ptr;
    difference_type texture_offset;
    hipTextureObject_t texture_object;
};

template<
    class T,
    class Difference
>
ROCPRIM_HOST_DEVICE inline
texture_cache_iterator<T, Difference>
operator+(typename texture_cache_iterator<T, Difference>::difference_type distance,
          const texture_cache_iterator<T, Difference>& iterator)
{
    return iterator + distance;
}

/// @}
// end of group iteratormodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_TEXTURE_CACHE_ITERATOR_HPP_