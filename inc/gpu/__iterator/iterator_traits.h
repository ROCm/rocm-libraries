#ifndef __GPU___ITERATOR_ITERATOR_TRAITS_H__
#define __GPU___ITERATOR_ITERATOR_TRAITS_H__

#include <iterator>
#include <type_traits>

#include "gpu/__utility/pair.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::iterator_traits.h
//====================================================================================================================//

template <class _Tp>
struct __has_iterator_category
{
private:
    template <class _Up> static std::false_type __test(...);
    template <class _Up> static std::true_type __test(typename _Up::iterator_category* = nullptr);
public:
    static const bool value = decltype(__test<_Tp>(nullptr))::value;
};

template <class _Tp, class _Up, bool = __has_iterator_category<std::iterator_traits<_Tp> >::value>
struct __has_iterator_category_convertible_to
    : std::is_convertible<typename std::iterator_traits<_Tp>::iterator_category, _Up>
{};

template <class _Tp, class _Up>
struct __has_iterator_category_convertible_to<_Tp, _Up, false> : std::false_type {};

template <class _Tp>
struct __is_cpp17_input_iterator : public __has_iterator_category_convertible_to<_Tp, std::input_iterator_tag> {};

template <class _Tp>
struct __is_cpp17_forward_iterator : public __has_iterator_category_convertible_to<_Tp, std::forward_iterator_tag> {};

template <class _Tp>
struct __is_cpp17_bidirectional_iterator : public __has_iterator_category_convertible_to<_Tp, std::bidirectional_iterator_tag> {};

template <class _Tp>
struct __is_cpp17_random_access_iterator : public __has_iterator_category_convertible_to<_Tp, std::random_access_iterator_tag> {};

// __is_cpp17_contiguous_iterator determines if an iterator is known by
// libc++ to be contiguous, either because it advertises itself as such
// (in C++20) or because it is a pointer type or a known trivial wrapper
// around a (possibly fancy) pointer type, such as __wrap_iter<T*>.
// Such iterators receive special "contiguous" optimizations in
// std::copy and std::sort.
//
#if _LIBCPP_STD_VER >= 20
template <class _Tp>
struct __is_cpp17_contiguous_iterator : std::disjunction_v<
    __has_iterator_category_convertible_to<_Tp, std::contiguous_iterator_tag>,
    __has_iterator_concept_convertible_to<_Tp, std::contiguous_iterator_tag>
> {};
#else
template <class _Tp>
struct __is_cpp17_contiguous_iterator : std::false_type {};
#endif

// Any native pointer which is an iterator is also a contiguous iterator.
template <class _Up>
struct __is_cpp17_contiguous_iterator<_Up*> : std::true_type {};

template <class _Tp>
struct __is_exactly_cpp17_input_iterator
    : public std::integral_constant<bool,
         __has_iterator_category_convertible_to<_Tp, std::input_iterator_tag>::value &&
        !__has_iterator_category_convertible_to<_Tp, std::forward_iterator_tag>::value> {};

template <class _Tp>
struct __is_exactly_cpp17_forward_iterator
    : public std::integral_constant<bool,
         __has_iterator_category_convertible_to<_Tp, std::forward_iterator_tag>::value &&
        !__has_iterator_category_convertible_to<_Tp, std::bidirectional_iterator_tag>::value> {};

template <class _Tp>
struct __is_exactly_cpp17_bidirectional_iterator
    : public std::integral_constant<bool,
         __has_iterator_category_convertible_to<_Tp, std::bidirectional_iterator_tag>::value &&
        !__has_iterator_category_convertible_to<_Tp, std::random_access_iterator_tag>::value> {};

template<class _InputIterator>
using __iter_value_type = typename std::iterator_traits<_InputIterator>::value_type;

template<class _InputIterator>
using __iter_key_type = std::remove_const_t<typename std::iterator_traits<_InputIterator>::value_type::first_type>;

template<class _InputIterator>
using __iter_mapped_type = typename std::iterator_traits<_InputIterator>::value_type::second_type;

template<class _InputIterator>
using __iter_to_alloc_type = pair<
    typename std::add_const<typename std::iterator_traits<_InputIterator>::value_type::first_type>::type,
    typename std::iterator_traits<_InputIterator>::value_type::second_type>;

template <class _Iter>
using __iterator_category_type = typename std::iterator_traits<_Iter>::iterator_category;

template <class _Iter>
using __iterator_pointer_type = typename std::iterator_traits<_Iter>::pointer;

template <class _Iter>
using __iter_diff_t = typename std::iterator_traits<_Iter>::difference_type;

template<class _InputIterator>
using __iter_value_type = typename std::iterator_traits<_InputIterator>::value_type;

} // namespace gpu


#endif // __GPU___ITERATOR_ITERATOR_TRAITS_H__
