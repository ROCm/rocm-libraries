// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_MOVE_ITERATOR_H
#define __GPU___ITERATOR_MOVE_ITERATOR_H

#include "gpu/__config"
#include <type_traits>
#include <iterator>

#include "gpu/__iterator/iterator_traits.h"
#include "gpu/__utility/declval.h"

namespace gpu {

#if _LIBGPU_STD_VER >= 20
template<class _Iter, class = void>
struct __move_iter_category_base {};

template<class _Iter>
  requires requires { typename std::iterator_traits<_Iter>::iterator_category; }
struct __move_iter_category_base<_Iter> {
    using iterator_category = std::conditional_t<
        derived_from<typename std::iterator_traits<_Iter>::iterator_category, std::random_access_iterator_tag>,
        std::random_access_iterator_tag,
        typename std::iterator_traits<_Iter>::iterator_category
    >;
};

template<class _Iter, class _Sent>
concept __move_iter_comparable = requires {
    { gpu::declval<const _Iter&>() == gpu::declval<_Sent>() } -> std::convertible_to<bool>;
};
#endif // _LIBGPU_STD_VER >= 20

template <class _Iter>
class _LIBGPU_TEMPLATE_VIS move_iterator
#if _LIBGPU_STD_VER >= 20
    : public __move_iter_category_base<_Iter>
#endif
{
    #if _LIBGPU_STD_VER >= 20
private:
    __device__ _LIBGPU_HIDE_FROM_ABI
    static constexpr auto __get_iter_concept() {
        if constexpr (random_access_iterator<_Iter>) {
            return std::random_access_iterator_tag{};
        } else if constexpr (bidirectional_iterator<_Iter>) {
            return std::bidirectional_iterator_tag{};
        } else if constexpr (forward_iterator<_Iter>) {
            return std::forward_iterator_tag{};
        } else {
            return std::input_iterator_tag{};
        }
    }
#endif // _LIBGPU_STD_VER >= 20
public:
#if _LIBGPU_STD_VER >= 20
    using iterator_type = _Iter;
    using iterator_concept = decltype(__get_iter_concept());
    // iterator_category is inherited and not always present
    using value_type = iter_value_t<_Iter>;
    using difference_type = iter_difference_t<_Iter>;
    using pointer = _Iter;
    using reference = iter_rvalue_reference_t<_Iter>;
#else
    typedef _Iter iterator_type;
    typedef std::conditional_t<
        __is_cpp17_random_access_iterator<_Iter>::value,
        std::random_access_iterator_tag,
        typename std::iterator_traits<_Iter>::iterator_category
    > iterator_category;
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    typedef typename std::iterator_traits<iterator_type>::difference_type difference_type;
    typedef iterator_type pointer;

    typedef typename std::iterator_traits<iterator_type>::reference __reference;
    typedef typename std::conditional<
            std::is_reference<__reference>::value,
            std::remove_reference_t<__reference>&&,
            __reference
        >::type reference;
#endif // _LIBGPU_STD_VER >= 20

    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    explicit move_iterator(_Iter __i) : __current_(std::move(__i)) {}

    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator& operator++() { ++__current_; return *this; }

    __device__ _LIBGPU_DEPRECATED_IN_CXX20 _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    pointer operator->() const { return __current_; }

#if _LIBGPU_STD_VER >= 20
    __device__ _LIBGPU_HIDE_FROM_ABI constexpr
    move_iterator() requires std::is_constructible_v<_Iter> : __current_() {}

    template <class _Up>
        requires (!_IsSame<_Up, _Iter>::value) && std::convertible_to<const _Up&, _Iter>
    __device__ _LIBGPU_HIDE_FROM_ABI constexpr
    move_iterator(const move_iterator<_Up>& __u) : __current_(__u.base()) {}

    template <class _Up>
        requires (!_IsSame<_Up, _Iter>::value) &&
                 std::convertible_to<const _Up&, _Iter> &&
                 assignable_from<_Iter&, const _Up&>
    __device__ _LIBGPU_HIDE_FROM_ABI constexpr
    move_iterator& operator=(const move_iterator<_Up>& __u) {
        __current_ = __u.base();
        return *this;
    }

    __device__ _LIBGPU_HIDE_FROM_ABI constexpr const _Iter& base() const & noexcept { return __current_; }
    __device__ _LIBGPU_HIDE_FROM_ABI constexpr _Iter base() && { return std::move(__current_); }

    __device__ _LIBGPU_HIDE_FROM_ABI constexpr
    reference operator*() const { return ranges::iter_move(__current_); }
    __device__ _LIBGPU_HIDE_FROM_ABI constexpr
    reference operator[](difference_type __n) const { return ranges::iter_move(__current_ + __n); }

    __device__ _LIBGPU_HIDE_FROM_ABI constexpr
    auto operator++(int)
        requires forward_iterator<_Iter>
    {
        move_iterator __tmp(*this); ++__current_; return __tmp;
    }

    __device__ _LIBGPU_HIDE_FROM_ABI constexpr
    void operator++(int) { ++__current_; }
#else
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator() : __current_() {}

    template <class _Up, class = std::enable_if_t<
        !std::is_same<_Up, _Iter>::value && std::is_convertible<const _Up&, _Iter>::value
    > >
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator(const move_iterator<_Up>& __u) : __current_(__u.base()) {}

    template <class _Up, class = std::enable_if_t<
        !std::is_same<_Up, _Iter>::value &&
        std::is_convertible<const _Up&, _Iter>::value &&
        std::is_assignable<_Iter&, const _Up&>::value
    > >
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator& operator=(const move_iterator<_Up>& __u) {
        __current_ = __u.base();
        return *this;
    }

    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    _Iter base() const { return __current_; }

    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    reference operator*() const { return static_cast<reference>(*__current_); }
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    reference operator[](difference_type __n) const { return static_cast<reference>(__current_[__n]); }

    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator operator++(int) { move_iterator __tmp(*this); ++__current_; return __tmp; }
#endif // _LIBGPU_STD_VER >= 20

    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator& operator--() { --__current_; return *this; }
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator operator--(int) { move_iterator __tmp(*this); --__current_; return __tmp; }
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator operator+(difference_type __n) const { return move_iterator(__current_ + __n); }
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator& operator+=(difference_type __n) { __current_ += __n; return *this; }
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator operator-(difference_type __n) const { return move_iterator(__current_ - __n); }
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
    move_iterator& operator-=(difference_type __n) { __current_ -= __n; return *this; }

#if _LIBGPU_STD_VER >= 20
    template<sentinel_for<_Iter> _Sent>
    __device__ friend _LIBGPU_HIDE_FROM_ABI constexpr
    bool operator==(const move_iterator& __x, const move_sentinel<_Sent>& __y)
        requires __move_iter_comparable<_Iter, _Sent>
    {
        return __x.base() == __y.base();
    }

    template<sized_sentinel_for<_Iter> _Sent>
    __device__ friend _LIBGPU_HIDE_FROM_ABI constexpr
    iter_difference_t<_Iter> operator-(const move_sentinel<_Sent>& __x, const move_iterator& __y)
    {
        return __x.base() - __y.base();
    }

    template<sized_sentinel_for<_Iter> _Sent>
    __device__ friend _LIBGPU_HIDE_FROM_ABI constexpr
    iter_difference_t<_Iter> operator-(const move_iterator& __x, const move_sentinel<_Sent>& __y)
    {
        return __x.base() - __y.base();
    }

    __device__ friend _LIBGPU_HIDE_FROM_ABI constexpr
    iter_rvalue_reference_t<_Iter> iter_move(const move_iterator& __i)
        noexcept(noexcept(ranges::iter_move(__i.__current_)))
    {
        return ranges::iter_move(__i.__current_);
    }

    template<indirectly_swappable<_Iter> _It2>
    __device__ friend _LIBGPU_HIDE_FROM_ABI constexpr
    void iter_swap(const move_iterator& __x, const move_iterator<_It2>& __y)
        noexcept(noexcept(ranges::iter_swap(__x.__current_, __y.__current_)))
    {
        return ranges::iter_swap(__x.__current_, __y.__current_);
    }
#endif // _LIBGPU_STD_VER >= 20

private:
    template<class _It2> friend class move_iterator;

    _Iter __current_;
};
_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(move_iterator);

template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
bool operator==(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() == __y.base();
}

#if _LIBGPU_STD_VER <= 17
template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
bool operator!=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() != __y.base();
}
#endif // _LIBGPU_STD_VER <= 17

template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
bool operator<(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
bool operator>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
bool operator<=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
bool operator>=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() >= __y.base();
}

#if _LIBGPU_STD_VER >= 20
template <class _Iter1, three_way_comparable_with<_Iter1> _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI constexpr
auto operator<=>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
    -> compare_three_way_result_t<_Iter1, _Iter2>
{
    return __x.base() <=> __y.base();
}
#endif // _LIBGPU_STD_VER >= 20

#ifndef _LIBGPU_CXX03_LANG
template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
auto operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
    -> decltype(__x.base() - __y.base())
{
    return __x.base() - __y.base();
}
#else
template <class _Iter1, class _Iter2>
__device__ inline _LIBGPU_HIDE_FROM_ABI
typename move_iterator<_Iter1>::difference_type
operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() - __y.base();
}
#endif // !_LIBGPU_CXX03_LANG

#if _LIBGPU_STD_VER >= 20
template <class _Iter>
__device__ inline _LIBGPU_HIDE_FROM_ABI constexpr
move_iterator<_Iter> operator+(iter_difference_t<_Iter> __n, const move_iterator<_Iter>& __x)
    requires requires { { __x.base() + __n } -> same_as<_Iter>; }
{
    return __x + __n;
}
#else
template <class _Iter>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
move_iterator<_Iter>
operator+(typename move_iterator<_Iter>::difference_type __n, const move_iterator<_Iter>& __x)
{
    return move_iterator<_Iter>(__x.base() + __n);
}
#endif // _LIBGPU_STD_VER >= 20

template <class _Iter>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
move_iterator<_Iter>
make_move_iterator(_Iter __i)
{
    return move_iterator<_Iter>(std::move(__i));
}

} // namespace gpu

#endif // __GPU___ITERATOR_MOVE_ITERATOR_H
