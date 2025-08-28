// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_ITERATOR_TRANSFORM_OUTPUT_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_TRANSFORM_OUTPUT_ITERATOR_HPP_

#include "rocprim/config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
template<typename OutputIterator, typename UnaryFunction>
class transform_output_iterator_proxy
{
public:
    ROCPRIM_HOST_DEVICE transform_output_iterator_proxy(const OutputIterator& iterator,
                                                        UnaryFunction         func)
        : it(iterator), f(func)
    {}

    template<typename T>
    ROCPRIM_HOST_DEVICE
    transform_output_iterator_proxy
        operator=(const T& value)
    {
        *it = f(value);
        return *this;
    }

private:
    OutputIterator it;
    UnaryFunction  f;
};
} // namespace detail

template<class OutputIterator, class UnaryFunction>
class transform_output_iterator
{
public:
    // We define most of the traits as void as these generally aren't defined
    // for write-only operators. Example: std::back_insert_iterator.

    /// The type of the value that can be obtained by dereferencing the iterator.
    /// It's `void` since transform_output_iterator is a write-only iterator.
    using value_type = void;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's `void` since transform_output_iterator is a write-only iterator.
    using reference = void;
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `void` since transform_output_iterator is a write-only iterator.
    using pointer = void;
    /// A type used for identify distance between iterators.
    /// It's `void` since transform_output_iterator is a write-only iterator.
    using difference_type = typename std::iterator_traits<OutputIterator>::difference_type;
    /// The category of the iterator.
    using iterator_category = std::output_iterator_tag;
    /// The type of unary function used to transform input range.
    using unary_function = UnaryFunction;

    using proxy_type = detail::transform_output_iterator_proxy<OutputIterator, unary_function>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = transform_output_iterator;
#endif

    ROCPRIM_HOST_DEVICE inline ~transform_output_iterator() = default;

    /// \brief Creates a new transform_output_iterator.
    ///
    /// \param iterator input iterator to iterate over and transform.
    /// \param transform unary function used to transform values obtained
    /// from range pointed by \p iterator.
    ROCPRIM_HOST_DEVICE inline transform_output_iterator(OutputIterator iterator,
                                                         UnaryFunction  transform)
        : iterator_(iterator), transform_(transform)
    {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator&
        operator++()
    {
        iterator_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator
        operator++(int)
    {
        transform_output_iterator old = *this;
        iterator_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator&
        operator--()
    {
        iterator_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator
        operator--(int)
    {
        transform_output_iterator old = *this;
        iterator_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE
    inline proxy_type
        operator*() const
    {
        return proxy_type(iterator_, transform_);
    }

    // We can't really define this as '&(*(*this));', se we delete this.
    ROCPRIM_HOST_DEVICE
    inline pointer
        operator->() const
        = delete;

    ROCPRIM_HOST_DEVICE
    inline proxy_type
        operator[](difference_type distance) const
    {
        transform_output_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator
        operator+(difference_type distance) const
    {
        return transform_output_iterator(iterator_ + distance, transform_);
    }

    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator&
        operator+=(difference_type distance)
    {
        iterator_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator
        operator-(difference_type distance) const
    {
        return transform_output_iterator(iterator_ - distance, transform_);
    }

    ROCPRIM_HOST_DEVICE
    inline transform_output_iterator&
        operator-=(difference_type distance)
    {
        iterator_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    inline difference_type
        operator-(transform_output_iterator other) const
    {
        return iterator_ - other.iterator_;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator==(transform_output_iterator other) const
    {
        return iterator_ == other.iterator_;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator!=(transform_output_iterator other) const
    {
        return iterator_ != other.iterator_;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator<(transform_output_iterator other) const
    {
        return iterator_ < other.iterator_;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator<=(transform_output_iterator other) const
    {
        return iterator_ <= other.iterator_;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator>(transform_output_iterator other) const
    {
        return iterator_ > other.iterator_;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator>=(transform_output_iterator other) const
    {
        return iterator_ >= other.iterator_;
    }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
    OutputIterator iterator_;
    UnaryFunction  transform_;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class InputIterator, class UnaryFunction>
ROCPRIM_HOST_DEVICE
inline transform_output_iterator<InputIterator, UnaryFunction>
    operator+(
        typename transform_output_iterator<InputIterator, UnaryFunction>::difference_type distance,
        const transform_output_iterator<InputIterator, UnaryFunction>&                    iterator)
{
    return iterator + distance;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

template<typename OutputIterator, typename UnaryFunction>
transform_output_iterator<
    OutputIterator,
    UnaryFunction> ROCPRIM_HOST_DEVICE make_transform_output_iterator(OutputIterator out,
                                                                      UnaryFunction  func)
{
    return transform_output_iterator<OutputIterator, UnaryFunction>(out, func);
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_TRANSFORM_OUTPUT_ITERATOR_HPP_