#ifndef __GPU___ITERATOR_REVERSE_ITERATOR_H__
#define __GPU___ITERATOR_REVERSE_ITERATOR_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::reverse_iterator
//====================================================================================================================//

_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template <class _Iter>
class _LIBGPU_TEMPLATE_VIS reverse_iterator
#if _LIBGPU_STD_VER <= 14 || !defined(_LIBGPU_ABI_NO_ITERATOR_BASES)
    : public iterator<typename std::iterator_traits<_Iter>::iterator_category,
                      typename std::iterator_traits<_Iter>::value_type,
                      typename std::iterator_traits<_Iter>::difference_type,
                      typename std::iterator_traits<_Iter>::pointer,
                      typename std::iterator_traits<_Iter>::reference>
#endif
{
_LIBGPU_SUPPRESS_DEPRECATED_POP
private:
#if _LIBGPU_STD_VER >= 20
    static_assert(__is_cpp17_bidirectional_iterator<_Iter>::value || bidirectional_iterator<_Iter>,
        "reverse_iterator<It> requires It to be a bidirectional iterator.");
#endif // _LIBGPU_STD_VER >= 20

protected:
    _Iter current;
public:
    using iterator_type = _Iter;

    using iterator_category = std::conditional<__is_cpp17_random_access_iterator<_Iter>::value,
                                  std::random_access_iterator_tag,
                                  typename std::iterator_traits<_Iter>::iterator_category>::type;
    using pointer = typename std::iterator_traits<_Iter>::pointer;
#if _LIBGPU_STD_VER >= 20
    using iterator_concept = std::conditional_t<random_access_iterator<_Iter>, std::random_access_iterator_tag, std::bidirectional_iterator_tag>;
    using value_type = iter_value_t<_Iter>;
    using difference_type = iter_difference_t<_Iter>;
    using reference = iter_reference_t<_Iter>;
#else
    using value_type = typename std::iterator_traits<_Iter>::value_type;
    using difference_type = typename std::iterator_traits<_Iter>::difference_type;
    using reference = typename std::iterator_traits<_Iter>::reference;
#endif

#ifndef _LIBGPU_ABI_NO_ITERATOR_BASES
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator() : current() {}

    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    explicit reverse_iterator(_Iter __x) : current(__x) {}

    template <class _Up, class = typename std::enable_if<
        !std::is_same<_Up, _Iter>::value && std::is_convertible<_Up const&, _Iter>::value
    >::type >
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator(const reverse_iterator<_Up>& __u)
        : current(__u.base())
    { }

    template <class _Up, class = typename std::enable_if<
        !std::is_same<_Up, _Iter>::value &&
        std::is_convertible<_Up const&, _Iter>::value &&
        std::is_assignable<_Iter&, _Up const&>::value
    >::type >
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator=(const reverse_iterator<_Up>& __u) {
        current = __u.base();
        return *this;
    }
#else
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator() : current() {}

    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    explicit reverse_iterator(_Iter __x) : current(__x) {}

    template <class _Up, class = typename std::enable_if<
        !std::is_same<_Up, _Iter>::value && std::is_convertible<_Up const&, _Iter>::value
    >::type >
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator(const reverse_iterator<_Up>& __u)
        : current(__u.base())
    { }

    template <class _Up, class = typename std::enable_if<
        !std::is_same<_Up, _Iter>::value &&
        std::is_convertible<_Up const&, _Iter>::value &&
        std::is_assignable<_Iter&, _Up const&>::value
    >::type >
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator=(const reverse_iterator<_Up>& __u) {
        current = __u.base();
        return *this;
    }
#endif
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    _Iter base() const {return current;}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reference operator*() const {_Iter __tmp = current; return *--__tmp;}

#if _LIBGPU_STD_VER >= 20
    _LIBGPU_INLINE_VISIBILITY
    constexpr pointer operator->() const
      requires std::is_pointer_v<_Iter> || requires(const _Iter __i) { __i.operator->(); }
    {
      if constexpr (std::is_pointer_v<_Iter>) {
        return std::prev(current);
      } else {
        return std::prev(current).operator->();
      }
    }
#else
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    pointer operator->() const {
      return std::addressof(operator*());
    }
#endif // _LIBGPU_STD_VER >= 20

    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator++() {--current; return *this;}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator++(int) {reverse_iterator __tmp(*this); --current; return __tmp;}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator--() {++current; return *this;}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator--(int) {reverse_iterator __tmp(*this); ++current; return __tmp;}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator+(difference_type __n) const {return reverse_iterator(current - __n);}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator+=(difference_type __n) {current -= __n; return *this;}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator operator-(difference_type __n) const {return reverse_iterator(current + __n);}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reverse_iterator& operator-=(difference_type __n) {current += __n; return *this;}
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    reference operator[](difference_type __n) const {return *(*this + __n);}

#if _LIBGPU_STD_VER >= 20
    _LIBGPU_HIDE_FROM_ABI friend constexpr
    iter_rvalue_reference_t<_Iter> iter_move(const reverse_iterator& __i)
      noexcept(std::is_nothrow_copy_constructible_v<_Iter> &&
          noexcept(ranges::iter_move(--std::declval<_Iter&>()))) {
      auto __tmp = __i.base();
      return ranges::iter_move(--__tmp);
    }

    template <indirectly_swappable<_Iter> _Iter2>
    _LIBGPU_HIDE_FROM_ABI friend constexpr
    void iter_swap(const reverse_iterator& __x, const reverse_iterator<_Iter2>& __y)
      noexcept(std::is_nothrow_copy_constructible_v<_Iter> &&
          std::is_nothrow_copy_constructible_v<_Iter2> &&
          noexcept(ranges::iter_swap(--std::declval<_Iter&>(), --std::declval<_Iter2&>()))) {
      auto __xtmp = __x.base();
      auto __ytmp = __y.base();
      ranges::iter_swap(--__xtmp, --__ytmp);
    }
#endif // _LIBGPU_STD_VER >= 20
};

template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
bool
operator==(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBGPU_STD_VER >= 20
    requires requires {
      { __x.base() == __y.base() } -> convertible_to<bool>;
    }
#endif // _LIBGPU_STD_VER >= 20
{
    return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
bool
operator<(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBGPU_STD_VER >= 20
    requires requires {
        { __x.base() > __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBGPU_STD_VER >= 20
{
    return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
bool
operator!=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBGPU_STD_VER >= 20
    requires requires {
      { __x.base() != __y.base() } -> convertible_to<bool>;
    }
#endif // _LIBGPU_STD_VER >= 20
{
    return __x.base() != __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
bool
operator>(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBGPU_STD_VER >= 20
    requires requires {
        { __x.base() < __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBGPU_STD_VER >= 20
{
    return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
bool
operator>=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBGPU_STD_VER >= 20
    requires requires {
        { __x.base() <= __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBGPU_STD_VER >= 20
{
    return __x.base() <= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
bool
operator<=(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
#if _LIBGPU_STD_VER >= 20
    requires requires {
        { __x.base() >= __y.base() } -> convertible_to<bool>;
      }
#endif // _LIBGPU_STD_VER >= 20
{
    return __x.base() >= __y.base();
}

#if _LIBGPU_STD_VER >= 20
template <class _Iter1, three_way_comparable_with<_Iter1> _Iter2>
_LIBGPU_HIDE_FROM_ABI constexpr
compare_three_way_result_t<_Iter1, _Iter2>
operator<=>(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
{
    return __y.base() <=> __x.base();
}
#endif // _LIBGPU_STD_VER >= 20

#ifndef _LIBGPU_CXX03_LANG
template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto
operator-(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
-> decltype(__y.base() - __x.base())
{
    return __y.base() - __x.base();
}
#else
template <class _Iter1, class _Iter2>
inline _LIBGPU_INLINE_VISIBILITY
typename reverse_iterator<_Iter1>::difference_type
operator-(const reverse_iterator<_Iter1>& __x, const reverse_iterator<_Iter2>& __y)
{
    return __y.base() - __x.base();
}
#endif

template <class _Iter>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
reverse_iterator<_Iter>
operator+(typename reverse_iterator<_Iter>::difference_type __n, const reverse_iterator<_Iter>& __x)
{
    return reverse_iterator<_Iter>(__x.base() - __n);
}

#if _LIBGPU_STD_VER >= 20
template <class _Iter1, class _Iter2>
  requires (!sized_sentinel_for<_Iter1, _Iter2>)
inline constexpr bool disable_sized_sentinel_for<reverse_iterator<_Iter1>, reverse_iterator<_Iter2>> = true;
#endif // _LIBGPU_STD_VER >= 20

#if _LIBGPU_STD_VER >= 14
template <class _Iter>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
reverse_iterator<_Iter> make_reverse_iterator(_Iter __i)
{
    return reverse_iterator<_Iter>(__i);
}
#endif


// TODO: get this working
#if _LIBGPU_STD_VER <= 17
template <class _Iter>
using __unconstrained_reverse_iterator = reverse_iterator<_Iter>;
#else

// __unconstrained_reverse_iterator allows us to use reverse iterators in the implementation of algorithms by working
// around a language issue in C++20.
// In C++20, when a reverse iterator wraps certain C++20-hostile iterators, calling comparison operators on it will
// result in a compilation error. However, calling comparison operators on the pristine hostile iterator is not
// an error. Thus, we cannot use reverse_iterators in the implementation of an algorithm that accepts a
// C++20-hostile iterator. This class is an internal workaround -- it is a copy of reverse_iterator with
// tweaks to make it support hostile iterators.
//
// A C++20-hostile iterator is one that defines a comparison operator where one of the arguments is an exact match
// and the other requires an implicit conversion, for example:
//   friend bool operator==(const BaseIter&, const DerivedIter&);
//
// C++20 rules for rewriting equality operators create another overload of this function with parameters reversed:
//   friend bool operator==(const DerivedIter&, const BaseIter&);
//
// This creates an ambiguity in overload resolution.
//
// Clang treats this ambiguity differently in different contexts. When operator== is actually called in the function
// body, the code is accepted with a warning. When a concept requires operator== to be a valid expression, however,
// it evaluates to false. Thus, the implementation of reverse_iterator::operator== can actually call operator== on its
// base iterators, but the constraints on reverse_iterator::operator== prevent it from being considered during overload
// resolution. This class simply removes the problematic constraints from comparison functions.
template <class _Iter>
class __unconstrained_reverse_iterator {
  _Iter __iter_;

public:
  static_assert(__is_cpp17_bidirectional_iterator<_Iter>::value || bidirectional_iterator<_Iter>);

  using iterator_type = _Iter;
  using iterator_category =
      _If<__is_cpp17_random_access_iterator<_Iter>::value, std::random_access_iterator_tag, __iterator_category_type<_Iter>>;
  using pointer = __iterator_pointer_type<_Iter>;
  using value_type = iter_value_t<_Iter>;
  using difference_type = iter_difference_t<_Iter>;
  using reference = iter_reference_t<_Iter>;

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator() = default;
  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator(const __unconstrained_reverse_iterator&) = default;
  _LIBGPU_HIDE_FROM_ABI constexpr explicit __unconstrained_reverse_iterator(_Iter __iter) : __iter_(__iter) {}

  _LIBGPU_HIDE_FROM_ABI constexpr _Iter base() const { return __iter_; }
  _LIBGPU_HIDE_FROM_ABI constexpr reference operator*() const {
    auto __tmp = __iter_;
    return *--__tmp;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr pointer operator->() const {
    if constexpr (std::is_pointer_v<_Iter>) {
      return std::prev(__iter_);
    } else {
      return std::prev(__iter_).operator->();
    }
  }

  _LIBGPU_HIDE_FROM_ABI friend constexpr
  iter_rvalue_reference_t<_Iter> iter_move(const __unconstrained_reverse_iterator& __i)
    noexcept(std::is_nothrow_copy_constructible_v<_Iter> &&
        noexcept(ranges::iter_move(--std::declval<_Iter&>()))) {
    auto __tmp = __i.base();
    return ranges::iter_move(--__tmp);
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator& operator++() {
    --__iter_;
    return *this;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator operator++(int) {
    auto __tmp = *this;
    --__iter_;
    return __tmp;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator& operator--() {
    ++__iter_;
    return *this;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator operator--(int) {
    auto __tmp = *this;
    ++__iter_;
    return __tmp;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator& operator+=(difference_type __n) {
    __iter_ -= __n;
    return *this;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator& operator-=(difference_type __n) {
    __iter_ += __n;
    return *this;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator operator+(difference_type __n) const {
    return __unconstrained_reverse_iterator(__iter_ - __n);
  }

  _LIBGPU_HIDE_FROM_ABI constexpr __unconstrained_reverse_iterator operator-(difference_type __n) const {
    return __unconstrained_reverse_iterator(__iter_ + __n);
  }

  _LIBGPU_HIDE_FROM_ABI constexpr difference_type operator-(const __unconstrained_reverse_iterator& __other) const {
    return __other.__iter_ - __iter_;
  }

  _LIBGPU_HIDE_FROM_ABI constexpr auto operator[](difference_type __n) const { return *(*this + __n); }

  // Deliberately unconstrained unlike the comparison functions in `reverse_iterator` -- see the class comment for the
  // rationale.
  _LIBGPU_HIDE_FROM_ABI friend constexpr bool
  operator==(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs) {
    return __lhs.base() == __rhs.base();
  }

  _LIBGPU_HIDE_FROM_ABI friend constexpr bool
  operator!=(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs) {
    return __lhs.base() != __rhs.base();
  }

  _LIBGPU_HIDE_FROM_ABI friend constexpr bool
  operator<(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs) {
    return __lhs.base() > __rhs.base();
  }

  _LIBGPU_HIDE_FROM_ABI friend constexpr bool
  operator>(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs) {
    return __lhs.base() < __rhs.base();
  }

  _LIBGPU_HIDE_FROM_ABI friend constexpr bool
  operator<=(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs) {
    return __lhs.base() >= __rhs.base();
  }

  _LIBGPU_HIDE_FROM_ABI friend constexpr bool
  operator>=(const __unconstrained_reverse_iterator& __lhs, const __unconstrained_reverse_iterator& __rhs) {
    return __lhs.base() <= __rhs.base();
  }
};

#endif // _LIBGPU_STD_VER <= 17

template <template <class> class _RevIter1, template <class> class _RevIter2, class _Iter>
struct __unwrap_reverse_iter_impl {
  using _UnwrappedIter = decltype(__unwrap_iter_impl<_Iter>::__unwrap(std::declval<_Iter>()));
  using _ReverseWrapper = _RevIter1<_RevIter2<_Iter> >;

  static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _ReverseWrapper
  __rewrap(_ReverseWrapper __orig_iter, _UnwrappedIter __unwrapped_iter) {
    return _ReverseWrapper(
        _RevIter2<_Iter>(__unwrap_iter_impl<_Iter>::__rewrap(__orig_iter.base().base(), __unwrapped_iter)));
  }

  static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _UnwrappedIter __unwrap(_ReverseWrapper __i) _NOEXCEPT {
    return __unwrap_iter_impl<_Iter>::__unwrap(__i.base().base());
  }
};

#if _LIBGPU_STD_VER >= 20
template <ranges::bidirectional_range _Range>
_LIBGPU_HIDE_FROM_ABI constexpr ranges::
    subrange<reverse_iterator<ranges::iterator_t<_Range>>, reverse_iterator<ranges::iterator_t<_Range>>>
    __reverse_range(_Range&& __range) {
  auto __first = ranges::begin(__range);
  return {std::make_reverse_iterator(ranges::next(__first, ranges::end(__range))), std::make_reverse_iterator(__first)};
}
#endif

template <class _Iter, bool __b>
struct __unwrap_iter_impl<reverse_iterator<reverse_iterator<_Iter> >, __b>
    : __unwrap_reverse_iter_impl<reverse_iterator, reverse_iterator, _Iter> {};

#if _LIBGPU_STD_VER >= 20

template <class _Iter, bool __b>
struct __unwrap_iter_impl<reverse_iterator<__unconstrained_reverse_iterator<_Iter>>, __b>
    : __unwrap_reverse_iter_impl<reverse_iterator, __unconstrained_reverse_iterator, _Iter> {};

template <class _Iter, bool __b>
struct __unwrap_iter_impl<__unconstrained_reverse_iterator<reverse_iterator<_Iter>>, __b>
    : __unwrap_reverse_iter_impl<__unconstrained_reverse_iterator, reverse_iterator, _Iter> {};

template <class _Iter, bool __b>
struct __unwrap_iter_impl<__unconstrained_reverse_iterator<__unconstrained_reverse_iterator<_Iter>>, __b>
    : __unwrap_reverse_iter_impl<__unconstrained_reverse_iterator, __unconstrained_reverse_iterator, _Iter> {};

#endif // _LIBGPU_STD_VER >= 20

    
} // namespace gpu


#endif // __GPU___ITERATOR_REVERSE_ITERATOR_H__
