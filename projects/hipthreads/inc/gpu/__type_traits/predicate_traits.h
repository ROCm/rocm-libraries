#ifndef __GPU___TYPE_TRAITS_PREDICATE_TRAITS_H__
#define __GPU___TYPE_TRAITS_PREDICATE_TRAITS_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __type_traits/predicate_traits.h
//====================================================================================================================//

template <class _Pred, class _Lhs, class _Rhs>
struct __is_trivial_equality_predicate : std::false_type {};

} // namespace gpu

#endif // __GPU___TYPE_TRAITS_PREDICATE_TRAITS_H__
