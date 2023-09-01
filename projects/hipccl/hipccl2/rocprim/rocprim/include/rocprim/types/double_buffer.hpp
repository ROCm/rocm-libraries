// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_
#define ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_

#include "../config.hpp"

/// \addtogroup utilsmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief This class provides an convenient way to do double buffering
///
/// It tracks two versions of the buffer ("current" and "alternate"),
/// which can be accessed with functions of the same name.
/// You can also swap the buffers.
template<class T>
class double_buffer
{
    T * buffers[2];

    unsigned int selector;

public:

    /// \brief Constructs an empty double buffer object, initializing the 
    /// buffer pointers to nullptr.
    ROCPRIM_HOST_DEVICE inline
    double_buffer()
    {
        selector = 0;
        buffers[0] = nullptr;
        buffers[1] = nullptr;
    }

    /// \brief Contructs a double buffer object using the supplied buffer pointers.
    ///
    /// \param current - Pointer to the buffer to designate as "current" (in use).
    /// \param alternate - Pointer to the buffer to designate as "alternate" (not in use)
    ROCPRIM_HOST_DEVICE inline
    double_buffer(T * current, T * alternate)
    {
        selector = 0;
        buffers[0] = current;
        buffers[1] = alternate;
    }

    /// \brief Returns a pointer to the current buffer.
    ROCPRIM_HOST_DEVICE inline
    T * current() const
    {
        return buffers[selector];
    }

    /// \brief Returns a pointer to the alternate buffer.
    ROCPRIM_HOST_DEVICE inline
    T * alternate() const
    {
        return buffers[selector ^ 1];
    }

    /// \brief Swaps the current and alternate buffers.
    ROCPRIM_HOST_DEVICE inline
    void swap()
    {
        selector ^= 1;
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule

#endif // ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_
