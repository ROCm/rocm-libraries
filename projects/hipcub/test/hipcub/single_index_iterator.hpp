// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_SINGLE_INDEX_ITERATOR_HPP_
#define TEST_SINGLE_INDEX_ITERATOR_HPP_

namespace test_utils
{

// Output iterator used in tests to check situations when size of output is too large to be stored
// in memory so only the last value is actually stored
template<typename T>
class single_index_iterator
{
public:
    class conditional_discard_value
    {
    public:
        HIPCUB_HOST_DEVICE inline explicit conditional_discard_value(T* const value, bool keep)
            : value_{value}, keep_{keep}
        {}

        HIPCUB_HOST_DEVICE
        inline conditional_discard_value&
            operator=(T value)
        {
            if(keep_)
            {
                *value_ = value;
            }
            return *this;
        }

    private:
        T* const   value_;
        const bool keep_;
    };

    T*     value_;
    size_t expected_index_;
    size_t index_;

public:
    using value_type        = conditional_discard_value;
    using reference         = conditional_discard_value;
    using pointer           = conditional_discard_value*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;

    HIPCUB_HOST_DEVICE inline single_index_iterator(T*     value,
                                                    size_t expected_index,
                                                    size_t index = 0)
        : value_{value}, expected_index_{expected_index}, index_{index}
    {}

    single_index_iterator(const single_index_iterator&)            = default;
    single_index_iterator& operator=(const single_index_iterator&) = default;

    ~single_index_iterator() = default;

    HIPCUB_HOST_DEVICE
    inline bool
        operator==(const single_index_iterator& rhs) const
    {
        return index_ == rhs.index_;
    }
    HIPCUB_HOST_DEVICE
    inline bool
        operator!=(const single_index_iterator& rhs) const
    {
        return !(this == rhs);
    }

    HIPCUB_HOST_DEVICE
    inline reference
        operator*()
    {
        return value_type{value_, index_ == expected_index_};
    }

    HIPCUB_HOST_DEVICE
    inline reference
        operator[](const difference_type distance) const
    {
        return *(*this + distance);
    }

    HIPCUB_HOST_DEVICE
    inline single_index_iterator&
        operator+=(const difference_type rhs)
    {
        index_ += rhs;
        return *this;
    }
    HIPCUB_HOST_DEVICE
    inline single_index_iterator&
        operator-=(const difference_type rhs)
    {
        index_ -= rhs;
        return *this;
    }

    HIPCUB_HOST_DEVICE
    inline difference_type
        operator-(const single_index_iterator& rhs) const
    {
        return index_ - rhs.index_;
    }

    HIPCUB_HOST_DEVICE
    inline single_index_iterator
        operator+(const difference_type rhs) const
    {
        return single_index_iterator(*this) += rhs;
    }
    HIPCUB_HOST_DEVICE
    inline single_index_iterator
        operator-(const difference_type rhs) const
    {
        return single_index_iterator(*this) -= rhs;
    }

    HIPCUB_HOST_DEVICE
    inline single_index_iterator&
        operator++()
    {
        ++index_;
        return *this;
    }
    HIPCUB_HOST_DEVICE
    inline single_index_iterator&
        operator--()
    {
        --index_;
        return *this;
    }

    HIPCUB_HOST_DEVICE
    inline single_index_iterator
        operator++(int)
    {
        return ++single_index_iterator{*this};
    }
    HIPCUB_HOST_DEVICE
    inline single_index_iterator
        operator--(int)
    {
        return --single_index_iterator{*this};
    }
};

} // namespace test_utils

#endif // TEST_SINGLE_INDEX_ITERATOR_HPP_
