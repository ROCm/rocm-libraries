#ifndef __GPU___MEMORY_UNIQUE_PTR_H__
#define __GPU___MEMORY_UNIQUE_PTR_H__

#include "hip/hip_runtime_api.h"
#include <cstddef>
#include <type_traits>

#include "gpu/__config"
#include "gpu/__functional/operations.h"
#include "gpu/__memory/__pointer.h"
#include "gpu/__memory/compressed_pair.h"
#include "gpu/__type_traits/dependent_type.h"
#include "gpu/__type_traits/type_identity.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::unique_ptr
//====================================================================================================================//

template <class _Tp>
struct _LIBGPU_TEMPLATE_VIS default_delete {
    static_assert(!std::is_function<_Tp>::value, "default_delete cannot be instantiated for function types");
#ifndef _LIBGPU_CXX03_LANG
    __device__ _LIBGPU_INLINE_VISIBILITY constexpr default_delete() _NOEXCEPT = default;
#else
    __device__ _LIBGPU_INLINE_VISIBILITY default_delete() {}
#endif
    template <class _Up>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    default_delete(const default_delete<_Up> &,
                   typename std::enable_if<std::is_convertible<_Up *, _Tp *>::value>::type * = 0) _NOEXCEPT {}

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void operator()(_Tp *__ptr) const _NOEXCEPT {
        static_assert(sizeof(_Tp) >= 0, "cannot delete an incomplete type");
        static_assert(!std::is_void<_Tp>::value, "cannot delete an incomplete type");
        delete __ptr;
    }
};

template <class _Tp>
struct _LIBGPU_TEMPLATE_VIS default_delete<_Tp[]> {
  private:
    template <class _Up>
    struct _EnableIfConvertible : std::enable_if<std::is_convertible<_Up (*)[], _Tp (*)[]>::value> {};

  public:
#ifndef _LIBGPU_CXX03_LANG
    __device__ _LIBGPU_INLINE_VISIBILITY constexpr default_delete() _NOEXCEPT = default;
#else
    __device__ _LIBGPU_INLINE_VISIBILITY default_delete() {}
#endif

    template <class _Up>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    default_delete(const default_delete<_Up[]> &, typename _EnableIfConvertible<_Up>::type * = 0) _NOEXCEPT {}

    template <class _Up>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 typename _EnableIfConvertible<_Up>::type
    operator()(_Up *__ptr) const _NOEXCEPT {
        static_assert(sizeof(_Up) >= 0, "cannot delete an incomplete type");
        delete[] __ptr;
    }
};

template <class _Deleter>
struct __unique_ptr_deleter_sfinae {
    static_assert(!std::is_reference<_Deleter>::value, "incorrect specialization");
    typedef const _Deleter &__lval_ref_type;
    typedef _Deleter &&__good_rval_ref_type;
    typedef std::true_type __enable_rval_overload;
};

template <class _Deleter>
struct __unique_ptr_deleter_sfinae<_Deleter const &> {
    typedef const _Deleter &__lval_ref_type;
    typedef const _Deleter &&__bad_rval_ref_type;
    typedef std::false_type __enable_rval_overload;
};

template <class _Deleter>
struct __unique_ptr_deleter_sfinae<_Deleter &> {
    typedef _Deleter &__lval_ref_type;
    typedef _Deleter &&__bad_rval_ref_type;
    typedef std::false_type __enable_rval_overload;
};

#if defined(_LIBGPU_ABI_ENABLE_UNIQUE_PTR_TRIVIAL_ABI)
#define _LIBGPU_UNIQUE_PTR_TRIVIAL_ABI __attribute__((__trivial_abi__))
#else
#define _LIBGPU_UNIQUE_PTR_TRIVIAL_ABI
#endif

template <class _Tp, class _Dp = default_delete<_Tp>>
class _LIBGPU_UNIQUE_PTR_TRIVIAL_ABI _LIBGPU_TEMPLATE_VIS unique_ptr {
  public:
    typedef _Tp element_type;
    typedef _Dp deleter_type;
    typedef _LIBGPU_NODEBUG typename __pointer<_Tp, deleter_type>::type pointer;

    static_assert(!std::is_rvalue_reference<deleter_type>::value,
                  "the specified deleter type cannot be an rvalue reference");

  private:
    // Use Empty Base Optimization to eliminate any padding from an empty deleter class
    __compressed_pair<pointer, deleter_type> __ptr_;

    struct __nat {
        int __for_bool_;
    };

    typedef _LIBGPU_NODEBUG __unique_ptr_deleter_sfinae<_Dp> _DeleterSFINAE;

    template <bool _Dummy>
    using _LValRefType _LIBGPU_NODEBUG = typename __dependent_type<_DeleterSFINAE, _Dummy>::__lval_ref_type;

    template <bool _Dummy>
    using _GoodRValRefType _LIBGPU_NODEBUG = typename __dependent_type<_DeleterSFINAE, _Dummy>::__good_rval_ref_type;

    template <bool _Dummy>
    using _BadRValRefType _LIBGPU_NODEBUG = typename __dependent_type<_DeleterSFINAE, _Dummy>::__bad_rval_ref_type;

    template <bool _Dummy, class _Deleter = typename __dependent_type<__type_identity<deleter_type>, _Dummy>::type>
    using _EnableIfDeleterDefaultConstructible _LIBGPU_NODEBUG =
        typename std::enable_if<std::is_default_constructible<_Deleter>::value &&
                                !std::is_pointer<_Deleter>::value>::type;

    template <class _ArgType>
    using _EnableIfDeleterConstructible _LIBGPU_NODEBUG =
        typename std::enable_if<std::is_constructible<deleter_type, _ArgType>::value>::type;

    template <class _UPtr, class _Up>
    using _EnableIfMoveConvertible _LIBGPU_NODEBUG =
        typename std::enable_if<std::is_convertible<typename _UPtr::pointer, pointer>::value &&
                                !std::is_array<_Up>::value>::type;

    template <class _UDel>
    using _EnableIfDeleterConvertible _LIBGPU_NODEBUG =
        typename std::enable_if<(std::is_reference<_Dp>::value && std::is_same<_Dp, _UDel>::value) ||
                                (!std::is_reference<_Dp>::value && std::is_convertible<_UDel, _Dp>::value)>::type;

    template <class _UDel>
    using _EnableIfDeleterAssignable = typename std::enable_if<std::is_assignable<_Dp &, _UDel &&>::value>::type;

  public:
    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr() _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr(std::nullptr_t) _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(pointer __p) _NOEXCEPT
        : __ptr_(__p, __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer __p,
                                                                                  _LValRefType<_Dummy> __d) _NOEXCEPT
        : __ptr_(__p, __d) {}

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    unique_ptr(pointer __p, _GoodRValRefType<_Dummy> __d) _NOEXCEPT : __ptr_(__p, std::move(__d)) {
        static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference");
    }

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_BadRValRefType<_Dummy>>>
    __device__ _LIBGPU_INLINE_VISIBILITY unique_ptr(pointer __p, _BadRValRefType<_Dummy> __d) = delete;

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<deleter_type>(__u.get_deleter())) {}

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterConvertible<_Ep>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<_Ep>(__u.get_deleter())) {}

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<deleter_type>(__u.get_deleter());
        return *this;
    }

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterAssignable<_Ep>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<_Ep>(__u.get_deleter());
        return *this;
    }

#ifdef _LIBGPU_CXX03_LANG
    __device__ unique_ptr(unique_ptr const &) = delete;
    __device__ unique_ptr &operator=(unique_ptr const &) = delete;
#endif

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 ~unique_ptr() { reset(); }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &operator=(std::nullptr_t) _NOEXCEPT {
        reset();
        return *this;
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 std::add_lvalue_reference_t<_Tp> operator*() const {
        return *__ptr_.first();
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer operator->() const _NOEXCEPT {
        return __ptr_.first();
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer get() const _NOEXCEPT {
        return __ptr_.first();
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 deleter_type &get_deleter() _NOEXCEPT {
        return __ptr_.second();
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 const deleter_type &
    get_deleter() const _NOEXCEPT {
        return __ptr_.second();
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit operator bool() const _NOEXCEPT {
        return __ptr_.first() != nullptr;
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer release() _NOEXCEPT {
        pointer __t = __ptr_.first();
        __ptr_.first() = pointer();
        return __t;
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void reset(pointer __p = pointer()) _NOEXCEPT {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = __p;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr &__u) _NOEXCEPT {
        __ptr_.swap(__u.__ptr_);
    }
};

template <class _Tp, class _Dp>
class _LIBGPU_UNIQUE_PTR_TRIVIAL_ABI _LIBGPU_TEMPLATE_VIS unique_ptr<_Tp[], _Dp> {
  public:
    typedef _Tp element_type;
    typedef _Dp deleter_type;
    typedef typename __pointer<_Tp, deleter_type>::type pointer;

  private:
    __compressed_pair<pointer, deleter_type> __ptr_;

    template <class _From>
    struct _CheckArrayPointerConversion : std::is_same<_From, pointer> {};

    template <class _FromElem>
    struct _CheckArrayPointerConversion<_FromElem *>
        : std::integral_constant<bool, std::is_same<_FromElem *, pointer>::value ||
                                      (std::is_same<pointer, element_type *>::value &&
                                       std::is_convertible<_FromElem (*)[], element_type (*)[]>::value)> {};

    typedef __unique_ptr_deleter_sfinae<_Dp> _DeleterSFINAE;

    template <bool _Dummy>
    using _LValRefType _LIBGPU_NODEBUG = typename __dependent_type<_DeleterSFINAE, _Dummy>::__lval_ref_type;

    template <bool _Dummy>
    using _GoodRValRefType _LIBGPU_NODEBUG = typename __dependent_type<_DeleterSFINAE, _Dummy>::__good_rval_ref_type;

    template <bool _Dummy>
    using _BadRValRefType _LIBGPU_NODEBUG = typename __dependent_type<_DeleterSFINAE, _Dummy>::__bad_rval_ref_type;

    template <bool _Dummy, class _Deleter = typename __dependent_type<__type_identity<deleter_type>, _Dummy>::type>
    using _EnableIfDeleterDefaultConstructible _LIBGPU_NODEBUG =
        typename std::enable_if<std::is_default_constructible<_Deleter>::value &&
                                !std::is_pointer<_Deleter>::value>::type;

    template <class _ArgType>
    using _EnableIfDeleterConstructible _LIBGPU_NODEBUG =
        typename std::enable_if<std::is_constructible<deleter_type, _ArgType>::value>::type;

    template <class _Pp>
    using _EnableIfPointerConvertible _LIBGPU_NODEBUG =
        typename std::enable_if<_CheckArrayPointerConversion<_Pp>::value>::type;

    template <class _UPtr, class _Up, class _ElemT = typename _UPtr::element_type>
    using _EnableIfMoveConvertible _LIBGPU_NODEBUG =
        typename std::enable_if<std::is_array<_Up>::value && std::is_same<pointer, element_type *>::value &&
                                std::is_same<typename _UPtr::pointer, _ElemT *>::value &&
                                std::is_convertible<_ElemT (*)[], element_type (*)[]>::value>::type;

    template <class _UDel>
    using _EnableIfDeleterConvertible _LIBGPU_NODEBUG =
        typename std::enable_if<(std::is_reference<_Dp>::value && std::is_same<_Dp, _UDel>::value) ||
                                (!std::is_reference<_Dp>::value && std::is_convertible<_UDel, _Dp>::value)>::type;

    template <class _UDel>
    using _EnableIfDeleterAssignable _LIBGPU_NODEBUG =
        typename std::enable_if<std::is_assignable<_Dp &, _UDel &&>::value>::type;

  public:
    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr() _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr(std::nullptr_t) _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>,
              class = _EnableIfPointerConvertible<_Pp>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(_Pp __p) _NOEXCEPT
        : __ptr_(__p, __value_init_tag()) {}

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>,
              class = _EnableIfPointerConvertible<_Pp>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(_Pp __p,
                                                                                  _LValRefType<_Dummy> __d) _NOEXCEPT
        : __ptr_(__p, __d) {}

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(std::nullptr_t,
                                                                                  _LValRefType<_Dummy> __d) _NOEXCEPT
        : __ptr_(nullptr, __d) {}

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>,
              class = _EnableIfPointerConvertible<_Pp>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    unique_ptr(_Pp __p, _GoodRValRefType<_Dummy> __d) _NOEXCEPT : __ptr_(__p, std::move(__d)) {
        static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference");
    }

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    unique_ptr(std::nullptr_t, _GoodRValRefType<_Dummy> __d) _NOEXCEPT : __ptr_(nullptr, std::move(__d)) {
        static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference");
    }

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterConstructible<_BadRValRefType<_Dummy>>,
              class = _EnableIfPointerConvertible<_Pp>>
    __device__ _LIBGPU_INLINE_VISIBILITY unique_ptr(_Pp __p, _BadRValRefType<_Dummy> __d) = delete;

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<deleter_type>(__u.get_deleter())) {}

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<deleter_type>(__u.get_deleter());
        return *this;
    }

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterConvertible<_Ep>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<_Ep>(__u.get_deleter())) {}

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterAssignable<_Ep>>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<_Ep>(__u.get_deleter());
        return *this;
    }

#ifdef _LIBGPU_CXX03_LANG
    __device__ unique_ptr(unique_ptr const &) = delete;
    __device__ unique_ptr &operator=(unique_ptr const &) = delete;
#endif
  public:
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 ~unique_ptr() { reset(); }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &operator=(std::nullptr_t) _NOEXCEPT {
        reset();
        return *this;
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 std::add_lvalue_reference_t<_Tp>
    operator[](size_t __i) const {
        return __ptr_.first()[__i];
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer get() const _NOEXCEPT {
        return __ptr_.first();
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 deleter_type &get_deleter() _NOEXCEPT {
        return __ptr_.second();
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 const deleter_type &
    get_deleter() const _NOEXCEPT {
        return __ptr_.second();
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit operator bool() const _NOEXCEPT {
        return __ptr_.first() != nullptr;
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer release() _NOEXCEPT {
        pointer __t = __ptr_.first();
        __ptr_.first() = pointer();
        return __t;
    }

    template <class _Pp>
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
        typename std::enable_if<_CheckArrayPointerConversion<_Pp>::value>::type
        reset(_Pp __p) _NOEXCEPT {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = __p;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void reset(std::nullptr_t = nullptr) _NOEXCEPT {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = nullptr;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr &__u) _NOEXCEPT {
        __ptr_.swap(__u.__ptr_);
    }
};

template <class _Tp, class _Dp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
typename std::enable_if<std::is_swappable<_Dp>::value, void>::type
swap(unique_ptr<_Tp, _Dp> &__x, unique_ptr<_Tp, _Dp> &__y) _NOEXCEPT {
    __x.swap(__y);
}

template <class _T1, class _D1, class _T2, class _D2>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator==(const unique_ptr<_T1, _D1> &__x, const unique_ptr<_T2, _D2> &__y) {
    return __x.get() == __y.get();
}

#if _LIBGPU_STD_VER <= 17
template <class _T1, class _D1, class _T2, class _D2>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator!=(const unique_ptr<_T1, _D1> &__x,
                                                            const unique_ptr<_T2, _D2> &__y) {
    return !(__x == __y);
}
#endif

template <class _T1, class _D1, class _T2, class _D2>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator<(const unique_ptr<_T1, _D1> &__x,
                                                           const unique_ptr<_T2, _D2> &__y) {
    typedef typename unique_ptr<_T1, _D1>::pointer _P1;
    typedef typename unique_ptr<_T2, _D2>::pointer _P2;
    typedef typename std::common_type<_P1, _P2>::type _Vp;
    return less<_Vp>()(__x.get(), __y.get());
}

template <class _T1, class _D1, class _T2, class _D2>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator>(const unique_ptr<_T1, _D1> &__x,
                                                           const unique_ptr<_T2, _D2> &__y) {
    return __y < __x;
}

template <class _T1, class _D1, class _T2, class _D2>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator<=(const unique_ptr<_T1, _D1> &__x,
                                                            const unique_ptr<_T2, _D2> &__y) {
    return !(__y < __x);
}

template <class _T1, class _D1, class _T2, class _D2>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator>=(const unique_ptr<_T1, _D1> &__x,
                                                            const unique_ptr<_T2, _D2> &__y) {
    return !(__x < __y);
}

#if _LIBGPU_STD_VER >= 20
template <class _T1, class _D1, class _T2, class _D2>
    requires three_way_comparable_with<typename unique_ptr<_T1, _D1>::pointer, typename unique_ptr<_T2, _D2>::pointer>
_LIBGPU_HIDE_FROM_ABI
__device__ compare_three_way_result_t<typename unique_ptr<_T1, _D1>::pointer, typename unique_ptr<_T2, _D2>::pointer>
operator<=>(const unique_ptr<_T1, _D1> &__x, const unique_ptr<_T2, _D2> &__y) {
    return compare_three_way()(__x.get(), __y.get());
}
#endif

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator==(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) _NOEXCEPT {
    return !__x;
}

#if _LIBGPU_STD_VER <= 17
template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator==(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) _NOEXCEPT {
    return !__x;
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator!=(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) _NOEXCEPT {
    return static_cast<bool>(__x);
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY bool operator!=(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) _NOEXCEPT {
    return static_cast<bool>(__x);
}
#endif // _LIBGPU_STD_VER <= 17

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    typedef typename unique_ptr<_T1, _D1>::pointer _P1;
    return less<_P1>()(__x.get(), nullptr);
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    typedef typename unique_ptr<_T1, _D1>::pointer _P1;
    return less<_P1>()(nullptr, __x.get());
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return nullptr < __x;
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    return __x < nullptr;
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<=(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return !(nullptr < __x);
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<=(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    return !(__x < nullptr);
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>=(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return !(__x < nullptr);
}

template <class _T1, class _D1>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>=(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    return !(nullptr < __x);
}

#if _LIBGPU_STD_VER >= 20
template <class _T1, class _D1>
    requires three_way_comparable<typename unique_ptr<_T1, _D1>::pointer>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX23 compare_three_way_result_t<typename unique_ptr<_T1, _D1>::pointer>
operator<=>(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return compare_three_way()(__x.get(), static_cast<typename unique_ptr<_T1, _D1>::pointer>(nullptr));
}
#endif

#if _LIBGPU_STD_VER >= 14

template <class _Tp>
struct __unique_if {
    typedef unique_ptr<_Tp> __unique_single;
};

template <class _Tp>
struct __unique_if<_Tp[]> {
    typedef unique_ptr<_Tp[]> __unique_array_unknown_bound;
};

template <class _Tp, size_t _Np>
struct __unique_if<_Tp[_Np]> {
    typedef void __unique_array_known_bound;
};

template <class _Tp, class... _Args>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 typename __unique_if<_Tp>::__unique_single
make_unique(_Args &&...__args) {
    return unique_ptr<_Tp>(new _Tp(std::forward<_Args>(__args)...));
}

template <class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
typename __unique_if<_Tp>::__unique_array_unknown_bound
make_unique(size_t __n) {
    typedef std::remove_extent_t<_Tp> _Up;
    return unique_ptr<_Tp>(new _Up[__n]());
}

template <class _Tp, class... _Args>
__device__ typename __unique_if<_Tp>::__unique_array_known_bound make_unique(_Args &&...) = delete;

#endif // _LIBGPU_STD_VER >= 14

#if _LIBGPU_STD_VER >= 20

template <class _Tp>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX23 typename __unique_if<_Tp>::__unique_single
make_unique_for_overwrite() {
    return unique_ptr<_Tp>(new _Tp);
}

template <class _Tp>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX23 typename __unique_if<_Tp>::__unique_array_unknown_bound
make_unique_for_overwrite(size_t __n) {
    return unique_ptr<_Tp>(new std::remove_extent_t<_Tp>[__n]);
}

template <class _Tp, class... _Args>
__device__ typename __unique_if<_Tp>::__unique_array_known_bound make_unique_for_overwrite(_Args &&...) = delete;

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___MEMORY_UNIQUE_PTR_H__
