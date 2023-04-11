#ifndef __GPU___ITERATOR_ITERATOR_TRAITS_H__
#define __GPU___ITERATOR_ITERATOR_TRAITS_H__

#include <iterator>
#include <type_traits>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ iterator_traits.h
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

} // namespace gpu


#endif // __GPU___ITERATOR_ITERATOR_TRAITS_H__
