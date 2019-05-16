// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_
#define ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

// invoke_result is based on https://en.cppreference.com/w/cpp/types/result_of
// The main difference is using ROCPRIM_HOST_DEVICE, this allows to
// use invoke_result with device-only lambdas/functors in host-only functions
// on HIP-clang.

template <class T>
struct is_reference_wrapper : std::false_type {};
template <class U>
struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

template<class T>
struct invoke_impl {
    template<class F, class... Args>
    ROCPRIM_HOST_DEVICE
    static auto call(F&& f, Args&&... args)
        -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));
};

template<class B, class MT>
struct invoke_impl<MT B::*>
{
    template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<std::is_base_of<B, Td>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto get(T&& t) -> T&&;

    template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<is_reference_wrapper<Td>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto get(T&& t) -> decltype(t.get());

    template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<!std::is_base_of<B, Td>::value>::type,
        class = typename std::enable_if<!is_reference_wrapper<Td>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto get(T&& t) -> decltype(*std::forward<T>(t));

    template<class T, class... Args, class MT1,
        class = typename std::enable_if<std::is_function<MT1>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto call(MT1 B::*pmf, T&& t, Args&&... args)
        -> decltype((invoke_impl::get(std::forward<T>(t)).*pmf)(std::forward<Args>(args)...));

    template<class T>
    ROCPRIM_HOST_DEVICE
    static auto call(MT B::*pmd, T&& t)
        -> decltype(invoke_impl::get(std::forward<T>(t)).*pmd);
};

template<class F, class... Args, class Fd = typename std::decay<F>::type>
ROCPRIM_HOST_DEVICE
auto INVOKE(F&& f, Args&&... args)
    -> decltype(invoke_impl<Fd>::call(std::forward<F>(f), std::forward<Args>(args)...));

// Conforming C++14 implementation (is also a valid C++11 implementation):
template <typename AlwaysVoid, typename, typename...>
struct invoke_result_impl { };
template <typename F, typename...Args>
struct invoke_result_impl<decltype(void(INVOKE(std::declval<F>(), std::declval<Args>()...))), F, Args...>
{
    using type = decltype(INVOKE(std::declval<F>(), std::declval<Args>()...));
};

template <class F, class... ArgTypes>
struct invoke_result : invoke_result_impl<void, F, ArgTypes...> {};

template<class InputType, class BinaryFunction>
struct match_result_type
{
    using type = typename invoke_result<BinaryFunction, InputType, InputType>::type;
};

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_
