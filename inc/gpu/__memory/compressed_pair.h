#ifndef __GPU___MEMORY_COMPRESSED_PAIR_H__
#define __GPU___MEMORY_COMPRESSED_PAIR_H__

#include <type_traits>
#include "gpu/__config"
#include "gpu/__type_traits/dependent_type.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __compressed_pair
//====================================================================================================================//

// Tag used to default initialize one or both of the pair's elements.
struct __default_init_tag {};
struct __value_init_tag {};

template <class _Tp, int _Idx, bool _CanBeEmptyBase = std::is_empty<_Tp>::value && !std::is_final<_Tp>::value>
struct __compressed_pair_elem {
  using _ParamT = _Tp;
  using reference = _Tp&;
  using const_reference = const _Tp&;

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR explicit __compressed_pair_elem(__default_init_tag) {}
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR explicit __compressed_pair_elem(__value_init_tag) : __value_() {}

  template <class _Up, class = std::enable_if_t<!std::is_same<__compressed_pair_elem, typename std::decay<_Up>::type>::value> >
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR
  explicit __compressed_pair_elem(_Up&& __u) : __value_(std::forward<_Up>(__u)) {}

#if 0 // TODO: 
#ifndef _LIBGPU_CXX03_LANG
  template <class... _Args, size_t... _Indices>
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
  explicit __compressed_pair_elem(piecewise_construct_t, tuple<_Args...> __args, __tuple_indices<_Indices...>)
      : __value_(std::forward<_Args>(std::get<_Indices>(__args))...) {}
#endif
#endif

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 reference __get() _NOEXCEPT { return __value_; }
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR const_reference __get() const _NOEXCEPT { return __value_; }

private:
  _Tp __value_;
};

template <class _Tp, int _Idx>
struct __compressed_pair_elem<_Tp, _Idx, true> : private _Tp {
  using _ParamT = _Tp;
  using reference = _Tp&;
  using const_reference = const _Tp&;
  using __value_type = _Tp;

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR explicit __compressed_pair_elem() = default;
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR explicit __compressed_pair_elem(__default_init_tag) {}
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR explicit __compressed_pair_elem(__value_init_tag) : __value_type() {}

  template <class _Up, class = std::enable_if_t<!std::is_same<__compressed_pair_elem, typename std::decay<_Up>::type>::value> >
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR
  explicit __compressed_pair_elem(_Up&& __u) : __value_type(std::forward<_Up>(__u)) {}

#if 0 // TODO: 
#ifndef _LIBGPU_CXX03_LANG
  template <class... _Args, size_t... _Indices>
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
  __compressed_pair_elem(piecewise_construct_t, tuple<_Args...> __args, __tuple_indices<_Indices...>)
      : __value_type(std::forward<_Args>(std::get<_Indices>(__args))...) {}
#endif
#endif

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 reference __get() _NOEXCEPT { return *this; }
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR const_reference __get() const _NOEXCEPT { return *this; }
};

template <class _T1, class _T2>
class __compressed_pair : private __compressed_pair_elem<_T1, 0>,
                          private __compressed_pair_elem<_T2, 1> {
public:
  // NOTE: This static assert should never fire because __compressed_pair
  // is *almost never* used in a scenario where it's possible for T1 == T2.
  // (The exception is std::function where it is possible that the function
  //  object and the allocator have the same type).
  static_assert((!std::is_same<_T1, _T2>::value),
    "__compressed_pair cannot be instantiated when T1 and T2 are the same type; "
    "The current implementation is NOT ABI-compatible with the previous implementation for this configuration");

  using _Base1 _LIBGPU_NODEBUG = __compressed_pair_elem<_T1, 0>;
  using _Base2 _LIBGPU_NODEBUG = __compressed_pair_elem<_T2, 1>;

  template <bool _Dummy = true,
    class = std::enable_if_t<
        __dependent_type<std::is_default_constructible<_T1>, _Dummy>::value &&
        __dependent_type<std::is_default_constructible<_T2>, _Dummy>::value
    >
  >
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR
  explicit __compressed_pair() : _Base1(__value_init_tag()), _Base2(__value_init_tag()) {}

  template <class _U1, class _U2>
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR
  explicit __compressed_pair(_U1&& __t1, _U2&& __t2) : _Base1(std::forward<_U1>(__t1)), _Base2(std::forward<_U2>(__t2)) {}

#if 0 // TODO: 
#ifndef _LIBGPU_CXX03_LANG
  template <class... _Args1, class... _Args2>
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17
  explicit __compressed_pair(piecewise_construct_t __pc, tuple<_Args1...> __first_args,
                             tuple<_Args2...> __second_args)
      : _Base1(__pc, std::move(__first_args), typename __make_tuple_indices<sizeof...(_Args1)>::type()),
        _Base2(__pc, std::move(__second_args), typename __make_tuple_indices<sizeof...(_Args2)>::type()) {}
#endif
#endif

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  typename _Base1::reference first() _NOEXCEPT {
    return static_cast<_Base1&>(*this).__get();
  }

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR
  typename _Base1::const_reference first() const _NOEXCEPT {
    return static_cast<_Base1 const&>(*this).__get();
  }

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  typename _Base2::reference second() _NOEXCEPT {
    return static_cast<_Base2&>(*this).__get();
  }

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR
  typename _Base2::const_reference second() const _NOEXCEPT {
    return static_cast<_Base2 const&>(*this).__get();
  }

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR static
  _Base1* __get_first_base(__compressed_pair* __pair) _NOEXCEPT {
    return static_cast<_Base1*>(__pair);
  }
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR static
  _Base2* __get_second_base(__compressed_pair* __pair) _NOEXCEPT {
    return static_cast<_Base2*>(__pair);
  }

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  void swap(__compressed_pair& __x)
      _NOEXCEPT_(std::is_nothrow_swappable<_T1>::value && std::is_nothrow_swappable<_T2>::value) {
    using std::swap;
    swap(first(), __x.first());
    swap(second(), __x.second());
  }
};

template <class _T1, class _T2>
__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
void swap(__compressed_pair<_T1, _T2>& __x, __compressed_pair<_T1, _T2>& __y)
    _NOEXCEPT_(std::is_nothrow_swappable<_T1>::value && std::is_nothrow_swappable<_T2>::value) {
  __x.swap(__y);
}

} // namespace gpu

#endif // __GPU___MEMORY_COMPRESSED_PAIR_H__
