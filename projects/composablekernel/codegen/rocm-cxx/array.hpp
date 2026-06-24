/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef ROCM_GUARD_ROCM_ARRAY_HPP
#define ROCM_GUARD_ROCM_ARRAY_HPP

#include <rocm/config.hpp>
#include <rocm/stdint.hpp>
#include <rocm/type_traits.hpp>
#include <rocm/utility/swap.hpp>
#include <rocm/utility/integer_sequence.hpp>
#include <rocm/iterator/reverse_iterator.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T, size_t N>
struct array
{
    using value_type             = T;
    using pointer                = T*;
    using const_pointer          = const T*;
    using reference              = T&;
    using const_reference        = const T&;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = rocm::reverse_iterator<iterator>;
    using const_reverse_iterator = rocm::reverse_iterator<const_iterator>;

    T elems[N]; // NOLINT

    // fill
    constexpr void fill(const T& u)
    {
        for(size_type i = 0; i < N; ++i)
            elems[i] = u;
    }

    // swap
    constexpr void swap(array& other) noexcept
    {
        for(size_type i = 0; i < N; ++i)
            rocm::swap(elems[i], other.elems[i]);
    }

    // iterators
    constexpr iterator begin() noexcept { return elems; }
    constexpr const_iterator begin() const noexcept { return elems; }
    constexpr iterator end() noexcept { return elems + N; }
    constexpr const_iterator end() const noexcept { return elems + N; }

    constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    constexpr const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }
    constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    constexpr const_iterator cbegin() const noexcept { return begin(); }
    constexpr const_iterator cend() const noexcept { return end(); }
    constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
    constexpr const_reverse_iterator crend() const noexcept { return rend(); }

    // capacity
    constexpr size_type size() const noexcept { return N; }
    constexpr size_type max_size() const noexcept { return N; }
    constexpr bool empty() const noexcept { return N == 0; }

    // element access
    constexpr reference operator[](size_type n) { return elems[n]; }
    constexpr const_reference operator[](size_type n) const { return elems[n]; }

    constexpr reference at(size_type n) { return elems[n]; }
    constexpr const_reference at(size_type n) const { return elems[n]; }

    constexpr reference front() { return elems[0]; }
    constexpr const_reference front() const { return elems[0]; }
    constexpr reference back() { return elems[N - 1]; }
    constexpr const_reference back() const { return elems[N - 1]; }

    constexpr T* data() noexcept { return elems; }
    constexpr const T* data() const noexcept { return elems; }

    // comparison operators
    friend constexpr bool operator==(const array& x, const array& y)
    {
        for(size_type i = 0; i < N; ++i)
        {
            if(not(x.elems[i] == y.elems[i]))
                return false;
        }
        return true;
    }

    friend constexpr bool operator!=(const array& x, const array& y) { return not(x == y); }

    friend constexpr bool operator<(const array& x, const array& y)
    {
        for(size_type i = 0; i < N; ++i)
        {
            if(x.elems[i] < y.elems[i])
                return true;
            if(y.elems[i] < x.elems[i])
                return false;
        }
        return false;
    }

    friend constexpr bool operator>(const array& x, const array& y) { return y < x; }

    friend constexpr bool operator<=(const array& x, const array& y) { return not(y < x); }

    friend constexpr bool operator>=(const array& x, const array& y) { return not(x < y); }
};

// zero-size specialization
template <class T>
struct array<T, 0>
{
    using value_type             = T;
    using pointer                = T*;
    using const_pointer          = const T*;
    using reference              = T&;
    using const_reference        = const T&;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = rocm::reverse_iterator<iterator>;
    using const_reverse_iterator = rocm::reverse_iterator<const_iterator>;

    constexpr void fill(const T&) {}
    constexpr void swap(array&) noexcept {}

    constexpr iterator begin() noexcept { return nullptr; }
    constexpr const_iterator begin() const noexcept { return nullptr; }
    constexpr iterator end() noexcept { return nullptr; }
    constexpr const_iterator end() const noexcept { return nullptr; }

    constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    constexpr const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }
    constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    constexpr const_iterator cbegin() const noexcept { return begin(); }
    constexpr const_iterator cend() const noexcept { return end(); }
    constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
    constexpr const_reverse_iterator crend() const noexcept { return rend(); }

    constexpr size_type size() const noexcept { return 0; }
    constexpr size_type max_size() const noexcept { return 0; }
    constexpr bool empty() const noexcept { return true; }

    constexpr T* data() noexcept { return nullptr; }
    constexpr const T* data() const noexcept { return nullptr; }

    friend constexpr bool operator==(const array&, const array&) { return true; }
    friend constexpr bool operator!=(const array&, const array&) { return false; }
    friend constexpr bool operator<(const array&, const array&) { return false; }
    friend constexpr bool operator>(const array&, const array&) { return false; }
    friend constexpr bool operator<=(const array&, const array&) { return true; }
    friend constexpr bool operator>=(const array&, const array&) { return true; }
};

// CTAD
template <class T, class... U>
ROCM_HIP_HOST_DEVICE_DEDUCTION_GUIDE array(T, U...) -> array<T, 1 + sizeof...(U)>;

// swap
template <class T, size_t N>
constexpr void swap(array<T, N>& x, array<T, N>& y) noexcept(noexcept(x.swap(y)))
{
    x.swap(y);
}

// to_array
namespace detail {

template <class T, size_t N, size_t... Is>
constexpr array<remove_cv_t<T>, N> to_array_lvalue(T (&a)[N], rocm::index_sequence<Is...>) // NOLINT
{
    return {{a[Is]...}};
}

template <class T, size_t N, size_t... Is>
constexpr array<remove_cv_t<T>, N>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
to_array_rvalue(T (&&a)[N], rocm::index_sequence<Is...>) // NOLINT
{
    return {{static_cast<T&&>(a[Is])...}};
}

} // namespace detail

template <class T, size_t N>
constexpr array<remove_cv_t<T>, N> to_array(T (&a)[N]) // NOLINT
{
    return detail::to_array_lvalue(a, rocm::make_index_sequence<N>{});
}

template <class T, size_t N>
constexpr array<remove_cv_t<T>, N> to_array(T (&&a)[N]) // NOLINT
{
    return detail::to_array_rvalue(static_cast<T(&&)[N]>(a),
                                   rocm::make_index_sequence<N>{}); // NOLINT
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ARRAY_HPP
