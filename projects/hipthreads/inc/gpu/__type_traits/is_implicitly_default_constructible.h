#ifndef __GPU___TYPE_TRAITS_IS_IMPLICITLY_DEFAULT_CONSTRUCTIBLE_H__
#define __GPU___TYPE_TRAITS_IS_IMPLICITLY_DEFAULT_CONSTRUCTIBLE_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __type_traits/is_implicitly_default_constructible.h
//====================================================================================================================//

#ifndef _LIBGPU_CXX03_LANG
// First of all, we can't implement this check in C++03 mode because the {}
// default initialization syntax isn't valid.
// Second, we implement the trait in a funny manner with two defaulted template
// arguments to workaround Clang's PR43454.
template <class _Tp>
void __test_implicit_default_constructible(_Tp);

template <class _Tp, class = void, class = typename std::is_default_constructible<_Tp>::type>
struct __is_implicitly_default_constructible
    : std::false_type
{ };

template <class _Tp>
struct __is_implicitly_default_constructible<_Tp, decltype(gpu::__test_implicit_default_constructible<_Tp const&>({})), std::true_type>
    : std::true_type
{ };

template <class _Tp>
struct __is_implicitly_default_constructible<_Tp, decltype(gpu::__test_implicit_default_constructible<_Tp const&>({})), std::false_type>
    : std::false_type
{ };
#endif // !C++03


} // namespace gpu

#endif // __GPU___TYPE_TRAITS_IS_IMPLICITLY_DEFAULT_CONSTRUCTIBLE_H__