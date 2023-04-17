#ifndef __GPU___TYPE_TRAITS_IS_CALLABLE_H__
#define __GPU___TYPE_TRAITS_IS_CALLABLE_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __type_traits/is_callable.h
//====================================================================================================================//

template<class _Func, class... _Args, class = decltype(std::declval<_Func>()(std::declval<_Args>()...))>
std::true_type __is_callable_helper(int);
template<class...>
std::false_type __is_callable_helper(...);

template<class _Func, class... _Args>
struct __is_callable : decltype(std::__is_callable_helper<_Func, _Args...>(0)) {};

} // namespace gpu

#endif // __GPU___TYPE_TRAITS_IS_CALLABLE_H__
