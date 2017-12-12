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

#ifndef ROCPRIM_ITERATOR_OUTPUT_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_OUTPUT_ITERATOR_HPP_

#include <type_traits>
#include <iterator>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

template <typename T>
class output_iterator
{
    public:
        typedef output_iterator self_type;
        typedef T value_type;
        typedef T& reference;
        typedef T* pointer;
        typedef std::forward_iterator_tag iterator_category;
        typedef int difference_type;
    
        output_iterator(pointer ptr) [[hc]] : ptr_(ptr) 
        { 
        }
    
        self_type operator++() [[hc]] 
        { 
            self_type i = *this; 
            ptr_++; 
            return i; 
        }
    
        self_type operator++(int junk) [[hc]] 
        { 
            ptr_++; 
            return *this; 
        }
    
        template <typename Distance>
        self_type operator+(Distance n) [[hc]] 
        { 
            self_type retval(ptr_ + n);
            return retval;
        }
    
        template <typename Distance>
        self_type operator-(Distance n) [[hc]] 
        { 
            self_type retval(ptr_ - n);
            return retval;
        }
    
        template <typename Distance>
        reference operator[](Distance n) [[hc]] 
        { 
            return *(ptr_ + n);
        }
    
        reference operator*() [[hc]] 
        { 
            return *ptr_; 
        }
    
        pointer operator->() [[hc]] 
        { 
            return ptr_; 
        }
    
        bool operator==(const self_type& rhs) [[hc]] 
        { 
            return ptr_ == rhs.ptr_; 
        }
    
        bool operator!=(const self_type& rhs) 
        { 
            return ptr_ != rhs.ptr_; 
        }
    
    private:
        pointer ptr_;
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_OUTPUT_ITERATOR_HPP_
