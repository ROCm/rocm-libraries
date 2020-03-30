// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_BOUNDS_CHECKING_ITERATOR_HPP_
#define TEST_BOUNDS_CHECKING_ITERATOR_HPP_

namespace test_utils
{

// Output iterator checking out of bounds situations
template<class T>
class bounds_checking_iterator
{
public:
    // Iterator traits
    using difference_type = std::ptrdiff_t;
    using value_type = void;
    using pointer = void;
    using reference = T&;

    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator(T * ptr, T * start_ptr, bool * out_of_bounds_flag, size_t size)
        : ptr_(ptr), start_ptr_(start_ptr), out_of_bounds_flag_(out_of_bounds_flag), size_(size)
    { }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator(T * ptr, bool * out_of_bounds_flag, size_t size)
        : bounds_checking_iterator(ptr, ptr, out_of_bounds_flag, size)
    { }

    ROCPRIM_HOST_DEVICE inline
    ~bounds_checking_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator& operator++()
    {
        ptr_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator operator++(int)
    {
        bounds_checking_iterator old = *this;
        ptr_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator& operator--()
    {
        ptr_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator operator--(int)
    {
        bounds_checking_iterator old = *this;
        ptr_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    reference operator*() const
    {
        if((ptr_ < start_ptr_) || (ptr_ >= start_ptr_ + size_))
        {
            *out_of_bounds_flag_ = true;
        }
        return *ptr_;
    }

    ROCPRIM_HOST_DEVICE inline
    reference operator[](difference_type n) const
    {
        if(((ptr_ + n) < start_ptr_) || ((ptr_ + n) >= start_ptr_ + size_))
        {
            *out_of_bounds_flag_ = true;
        }
        return *(ptr_ + n);
    }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator operator+(difference_type distance) const
    {
        auto i = ptr_ + distance;
        return bounds_checking_iterator(i, start_ptr_, out_of_bounds_flag_, size_);
    }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator& operator+=(difference_type distance)
    {
        ptr_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator operator-(difference_type distance) const
    {
        auto i = ptr_ - distance;
        return bounds_checking_iterator(i, start_ptr_, out_of_bounds_flag_, size_);
    }

    ROCPRIM_HOST_DEVICE inline
    bounds_checking_iterator& operator-=(difference_type distance)
    {
        ptr_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(bounds_checking_iterator other) const
    {
        return ptr_ - other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(bounds_checking_iterator other) const
    {
        return ptr_ == other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(bounds_checking_iterator other) const
    {
        return ptr_ != other.ptr_;
    }

private:
    T * ptr_;
    T * start_ptr_;
    bool * out_of_bounds_flag_;
    size_t size_;
};

class out_of_bounds_flag
{
public:
    out_of_bounds_flag()
    {
        hipMalloc(&device_pointer_, sizeof(bool));
        hipMemset(device_pointer_, 0, sizeof(bool));
    }

    ~out_of_bounds_flag()
    {
        hipFree(device_pointer_);
    }

    bool get() const
    {
        bool value;
        hipMemcpy(&value, device_pointer_, sizeof(bool), hipMemcpyDeviceToHost);
        return value;
    }

    bool * device_pointer() const
    {
        return device_pointer_;
    }

private:
    bool * device_pointer_;
};

} // end test_utils namespace

#endif // TEST_BOUNDS_CHECKING_ITERATOR_HPP_
