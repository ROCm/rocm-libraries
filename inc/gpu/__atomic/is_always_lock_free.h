// -*- C++ -*-

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __GPU___ATOMIC_IS_ALWAYS_LOCK_FREE_H__
#define __GPU___ATOMIC_IS_ALWAYS_LOCK_FREE_H__

#include <type_traits>

namespace gpu::internal {

template <class T>
struct is_hip_native
    : std::integral_constant<bool,
                             std::is_same<float, typename std::remove_cv<T>::type>::value ||
                                 // std::is_same<double, typename std::remove_cv<T>::type>::value || // requires MI-200
                                 std::is_same<int, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<long, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<long long, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<unsigned int, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<unsigned long, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<unsigned long long, typename std::remove_cv<T>::type>::value> {};

template <class T>
inline constexpr bool is_hip_native_v = is_hip_native<T>::value;

template <class _Tp>
struct __libcpp_is_always_lock_free {
    // TODO: If/when HIP gets an equivalent to __atomic_always_lock_free uncomment this and remove the following line
    // static const bool __value = __atomic_always_lock_free(sizeof(_Tp), nullptr);
    static const bool __value = gpu::internal::is_hip_native_v<_Tp>;
};

} // namespace gpu::internal

#endif // __GPU___ATOMIC_IS_ALWAYS_LOCK_FREE_H__
