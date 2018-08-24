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

#ifndef ROCPRIM_ITERATOR_ZIP_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_ZIP_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../types/tuple.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T>
struct tuple_of_references;

template<class... Types>
struct tuple_of_references<::rocprim::tuple<Types...>>
{
    using type = ::rocprim::tuple<typename std::iterator_traits<Types>::reference...>;
};

template<class T>
struct tuple_of_values;

template<class... Types>
struct tuple_of_values<::rocprim::tuple<Types...>>
{
    using type = ::rocprim::tuple<typename std::iterator_traits<Types>::value_type...>;
};

template<class... Types, class Function, size_t... Indices>
ROCPRIM_HOST_DEVICE inline
void for_each_in_tuple_impl(::rocprim::tuple<Types...>& t,
                            Function f,
                            ::rocprim::index_sequence<Indices...>)
{
    auto swallow = { (f(::rocprim::get<Indices>(t)), 0)... };
    (void) swallow;
}

template<class... Types, class Function>
ROCPRIM_HOST_DEVICE inline
void for_each_in_tuple(::rocprim::tuple<Types...>& t, Function f)
{
    for_each_in_tuple_impl(t, f, ::rocprim::index_sequence_for<Types...>());
}

struct increment_iterator
{
    template<class Iterator>
    ROCPRIM_HOST_DEVICE inline
    void operator()(Iterator& it)
    {
        ++it;
    }
};

struct decrement_iterator
{
    template<class Iterator>
    ROCPRIM_HOST_DEVICE inline
    void operator()(Iterator& it)
    {
        --it;
    }
};

template<class Difference>
struct advance_iterator
{
    ROCPRIM_HOST_DEVICE inline
    advance_iterator(Difference distance)
        : distance_(distance)
    {
    }

    template<class Iterator>
    ROCPRIM_HOST_DEVICE inline
    void operator()(Iterator& it)
    {
        using it_distance_type = typename std::iterator_traits<Iterator>::difference_type;
        it += static_cast<it_distance_type>(distance_);
    }

private:
    Difference distance_;
};

template<class ReferenceTuple, class... Types, size_t... Indices>
ROCPRIM_HOST_DEVICE inline
ReferenceTuple dereference_iterator_tuple_impl(const ::rocprim::tuple<Types...>& t,
                                               ::rocprim::index_sequence<Indices...>)
{
    ReferenceTuple rt { *::rocprim::get<Indices>(t)... };
    return rt;
}

template<class ReferenceTuple, class... Types>
ROCPRIM_HOST_DEVICE inline
ReferenceTuple dereference_iterator_tuple(const ::rocprim::tuple<Types...>& t)
{
    return dereference_iterator_tuple_impl<ReferenceTuple>(
        t, ::rocprim::index_sequence_for<Types...>()
    );
}

} // end detail namespace

/// \class zip_iterator
/// \brief TBD
///
/// \par Overview
/// * TBD
///
/// \tparam IteratorTuple -
template<class IteratorTuple>
class zip_iterator
{
public:
    /// \brief A reference type of the type iterated over.
    ///
    /// The type of the tuple made of the reference types of the iterator
    /// types in the IteratorTuple argument.
    using reference = typename detail::tuple_of_references<IteratorTuple>::type;
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename detail::tuple_of_values<IteratorTuple>::type;
    /// \brief A pointer type of the type iterated over (\p value_type).
    using pointer = value_type*;
    /// \brief A type used for identify distance between iterators.
    ///
    /// The difference_type member of zip_iterator is the difference_type of
    /// the first of the iterator types in the IteratorTuple argument.
    using difference_type = typename std::iterator_traits<
        typename ::rocprim::tuple_element<0, IteratorTuple>::type
    >::difference_type;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline
    ~zip_iterator() = default;

    /// \brief Creates a new zip_iterator.
    ///
    /// \param iterator_tuple tuple of iterators
    ROCPRIM_HOST_DEVICE inline
    zip_iterator(IteratorTuple iterator_tuple)
        : iterator_tuple_(iterator_tuple)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator& operator++()
    {
        detail::for_each_in_tuple(iterator_tuple_, detail::increment_iterator());
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator operator++(int)
    {
        zip_iterator old = *this;
        ++(*this);
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator& operator--()
    {
        detail::for_each_in_tuple(iterator_tuple_, detail::decrement_iterator());
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator operator--(int)
    {
        zip_iterator old = *this;
        --(*this);
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    reference operator*() const
    {
        return detail::dereference_iterator_tuple<reference>(iterator_tuple_);
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &(*(*this));
    }

    ROCPRIM_HOST_DEVICE inline
    reference operator[](difference_type distance) const
    {
        zip_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator operator+(difference_type distance) const
    {
        zip_iterator copy = *this;
        copy += distance;
        return copy;
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator& operator+=(difference_type distance)
    {
        detail::for_each_in_tuple(
            iterator_tuple_,
            detail::advance_iterator<difference_type>(distance)
        );
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator operator-(difference_type distance) const
    {
        auto copy = *this;
        copy -= distance;
        return copy;
    }

    ROCPRIM_HOST_DEVICE inline
    zip_iterator& operator-=(difference_type distance)
    {
        *this += -distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(zip_iterator other) const
    {
        return ::rocprim::get<0>(iterator_tuple_) - ::rocprim::get<0>(other.iterator_tuple_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(zip_iterator other) const
    {
        return iterator_tuple_ == other.iterator_tuple_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(zip_iterator other) const
    {
        return !(*this == other);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(zip_iterator other) const
    {
        return ::rocprim::get<0>(iterator_tuple_) < ::rocprim::get<0>(other.iterator_tuple_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(zip_iterator other) const
    {
        return !(other < *this);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(zip_iterator other) const
    {
        return other < *this;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(zip_iterator other) const
    {
        return !(*this < other);
    }

    friend std::ostream& operator<<(std::ostream& os, const zip_iterator& /* iter */)
    {
        return os;
    }

private:
    IteratorTuple iterator_tuple_;
};

template<class IteratorTuple>
ROCPRIM_HOST_DEVICE inline
zip_iterator<IteratorTuple>
operator+(typename zip_iterator<IteratorTuple>::difference_type distance,
          const zip_iterator<IteratorTuple>& iterator)
{
    return iterator + distance;
}

/// make_zip_iterator creates a zip_iterator using \p iterator_tuple as
/// the underlying tuple of iterator.
///
/// \tparam IteratorTuple - iterator tuple type
///
/// \param iterator_tuple - tuple of iterators to use
/// \return A new zip_iterator object
template<class IteratorTuple>
ROCPRIM_HOST_DEVICE inline
zip_iterator<IteratorTuple>
make_zip_iterator(IteratorTuple iterator_tuple)
{
    return zip_iterator<IteratorTuple>(iterator_tuple);
}

/// @}
// end of group iteratormodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_ZIP_ITERATOR_HPP_
