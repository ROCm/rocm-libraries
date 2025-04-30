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

#ifndef ROCPRIM_TYPES_UNINITIALIZED_ARRAY_HPP_
#define ROCPRIM_TYPES_UNINITIALIZED_ARRAY_HPP_

#include "../config.hpp"

#include <cstddef>
#include <iterator>
#include <type_traits>

#include <stdint.h>

BEGIN_ROCPRIM_NAMESPACE

/// \brief Provides indexed & typed access to uninitialized memory.
/// To be used with `ROCPRIM_SHARED_MEMORY`
/// \note This class should be used to ensure that writes to the uninitialized memory block
/// occur only via placement new.
/// \note This class is non-copyable.
/// \note The recommended pattern for usage is to first fill the array via calls to `emplace`,
/// then read-only via `get_unsafe_array()`. Writing to the array reference returned by `get_unsafe_array()`
/// should be avoided, if possible.
/// \note The value of `Alignment` MUST be a valid alignment for `T`.
/// \tparam T The item type which is provided via the accessors.
/// \tparam Count The number of T items to store.
/// \tparam Alignment The alignment of the backing storage, in bytes.
template<typename T, unsigned int Count, size_t Alignment = alignof(T)>
class uninitialized_array
{
public:
    /// \brief Type of the represented C array.
    using c_array_t = T[Count];

    ROCPRIM_HOST_DEVICE uninitialized_array()                           = default;
    ROCPRIM_HOST_DEVICE uninitialized_array(const uninitialized_array&) = delete;

    /// \brief Default move constructor.
    ROCPRIM_HOST_DEVICE uninitialized_array(uninitialized_array&&) = default;

    ROCPRIM_HOST_DEVICE uninitialized_array& operator=(const uninitialized_array&) = delete;

    /// \brief Default move assignment.
    ROCPRIM_HOST_DEVICE uninitialized_array& operator=(uninitialized_array&&) = default;

    /// \brief Constructs a value in-place at the specified array index.
    /// \note This function calls the constructor of `T` with the specified arguments.
    /// If an instance of `T&` is passed, the copy constructor is called, whereas in the case
    /// of a `T&&`, the move constructor is called.
    /// \note If an object is created at the same index more than once, the old object's destructor is
    /// **not** called. If `T` is not trivially destructible, the behaviour is undefined.
    /// \tparam Args The types of the argument values.
    /// \param index The index in the array where the new object is constructed.
    /// \param args The arguments to call the constructor with.
    /// \returns A reference to the newly constructed element.
    template<typename... Args>
    [[maybe_unused]] ROCPRIM_HOST_DEVICE T& emplace(const unsigned int index, Args&&... args)
    {
        T* ptr = reinterpret_cast<T*>(&storage[0]) + index;
        return *new(ptr) T(std::forward<Args>(args)...);
    }

    /// \brief Returns a reference to the underlying memory as a typed array.
    /// \note Manipulating items in the returned array reference at indices that
    /// were not previously filled by calls to `emplace` MUST be avoided.
    [[nodiscard]] ROCPRIM_HOST_DEVICE c_array_t& get_unsafe_array()
    {
        return *reinterpret_cast<c_array_t*>(&storage[0]);
    }

private:
    // The type of the backing storage MUST be either [unsigned] char or std::byte
    // otherwise the aliasing rules are violated and the program behaviour is undefined.
    alignas(Alignment) unsigned char storage[sizeof(T) * Count];
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_TYPES_UNINITIALIZED_ARRAY_HPP_
