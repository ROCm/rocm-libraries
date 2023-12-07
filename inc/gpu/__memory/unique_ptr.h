#ifndef __GPU___MEMORY_UNIQUE_PTR_H__
#define __GPU___MEMORY_UNIQUE_PTR_H__

#include "hip/hip_runtime.h"
#include <cstddef>
#include <type_traits>
#include <memory>

#include "gpu/__config"
#include "gpu/__clib/malloc.h"
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
    __device__ _LIBGPU_INLINE_VISIBILITY default_delete() _NOEXCEPT = default;
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
    __device__ _LIBGPU_INLINE_VISIBILITY default_delete() _NOEXCEPT = default;
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

template <class _Tp>
struct _LIBGPU_TEMPLATE_VIS host_delete {
    static_assert(!std::is_function<_Tp>::value, "host_delete cannot be instantiated for function types");
    static_assert(std::is_trivially_destructible<_Tp>::value, "host_delete can only be instantiated for trivially destructible types");
#ifndef _LIBGPU_CXX03_LANG
    __host__ _LIBGPU_INLINE_VISIBILITY constexpr host_delete() _NOEXCEPT = default;
#else
    __host__ _LIBGPU_INLINE_VISIBILITY host_delete() {}
#endif
    template <class _Up>
    __host__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    host_delete(const host_delete<_Up> &,
                   typename std::enable_if<std::is_convertible<_Up *, _Tp *>::value>::type * = 0) _NOEXCEPT {}

    __host__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void operator()(_Tp *__ptr) const _NOEXCEPT {
        static_assert(sizeof(_Tp) >= 0, "cannot delete an incomplete type");
        static_assert(!std::is_void<_Tp>::value, "cannot delete an incomplete type");
        gpu::free(const_cast<typename std::remove_const<_Tp>::type *>(__ptr));
    }
};

template <class _Tp>
struct _LIBGPU_TEMPLATE_VIS host_delete<_Tp[]> {
    static_assert(std::is_trivially_destructible<_Tp>::value, "host_delete can only be instantiated for trivially destructible types");
  private:
    template <class _Up>
    struct _EnableIfConvertible : std::enable_if<std::is_convertible<_Up (*)[], _Tp (*)[]>::value> {};

  public:
#ifndef _LIBGPU_CXX03_LANG
    __host__ _LIBGPU_INLINE_VISIBILITY constexpr host_delete() _NOEXCEPT = default;
#else
    __host__ _LIBGPU_INLINE_VISIBILITY host_delete() {}
#endif

    template <class _Up>
    __host__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    host_delete(const host_delete<_Up[]> &, typename _EnableIfConvertible<_Up>::type * = 0) _NOEXCEPT {}

    template <class _Up>
    __host__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 typename _EnableIfConvertible<_Up>::type
    operator()(_Up *__ptr) const _NOEXCEPT {
        static_assert(sizeof(_Up) >= 0, "cannot delete an incomplete type");
        gpu::free(const_cast<typename std::remove_const<_Up>::type *>(__ptr));
    }
};

template <class _Tp>
__global__ void offload_delete_kernel(_Tp *__ptr) {
    __ptr->~_Tp();
}

// TODO: maybe rename this to offload_destruct
// Launches a kernel which calls the destructor for the pointed to data, and queues an async free into the same stream after the kernel
template <class _Tp>
struct _LIBGPU_TEMPLATE_VIS offload_delete {
    static_assert(!std::is_function<_Tp>::value, "offload_delete cannot be instantiated for function types");
    static_assert(!std::is_array<_Tp>::value, "offload_delete cannot be instantiated for array types");
#ifndef _LIBGPU_CXX03_LANG
    __host__ _LIBGPU_INLINE_VISIBILITY constexpr offload_delete() _NOEXCEPT = default;
#else
    __host__ _LIBGPU_INLINE_VISIBILITY offload_delete() {}
#endif
    template <class _Up>
    __host__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    offload_delete(const offload_delete<_Up> &,
                   typename std::enable_if<std::is_convertible<_Up *, _Tp *>::value>::type * = 0) _NOEXCEPT {}

    __host__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void operator()(_Tp *__ptr) const _NOEXCEPT {
        static_assert(sizeof(_Tp) >= 0, "cannot delete an incomplete type");
        static_assert(!std::is_void<_Tp>::value, "cannot delete an incomplete type");
        // TODO: Clean this up. It's kind of ugly, but I can't put my finger on why or how to fix it
        if (__ptr != nullptr)
        {
            hipLaunchKernelGGL(offload_delete_kernel, dim3(1), dim3(1), 0, internal::getEnqueingStream(), __ptr);
            // TODO: figure out how to handle hipLaunchKernelGGL failures, since this function is noexcept. Maybe remove
            // the noexcept designation?
            // __LIBGPU_HIP_CHECK__(hipGetLastError());

            // gpu::free queues an async free into the enqueuing stream, so it's guaranteed not to perform the free before
            // offload_delete_kernel finishes. Thus we don't need to call hipStreamSynchronize
            gpu::free(const_cast<typename std::remove_const<_Tp>::type *>(__ptr));
        }
    }
};

// TODO: maybe also provide a 'skip_destruct' deleter which acts like host_delete, but accepts
// non-trivially-destructible classes

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
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr() _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr(std::nullptr_t) _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(pointer __p) _NOEXCEPT
        : __ptr_(__p, __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(pointer __p,
                                                                                  _LValRefType<_Dummy> __d) _NOEXCEPT
        : __ptr_(__p, __d) {}

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    unique_ptr(pointer __p, _GoodRValRefType<_Dummy> __d) _NOEXCEPT : __ptr_(__p, std::move(__d)) {
        static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference");
    }

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_BadRValRefType<_Dummy>>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY unique_ptr(pointer __p, _BadRValRefType<_Dummy> __d) = delete;

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<deleter_type>(__u.get_deleter())) {}

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterConvertible<_Ep>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<_Ep>(__u.get_deleter())) {}

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<deleter_type>(__u.get_deleter());
        return *this;
    }

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterAssignable<_Ep>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<_Ep>(__u.get_deleter());
        return *this;
    }

    // Note that we are intentionally not accepting any std::unique_ptr with a non-default deleter
    template <class _Up, class = _EnableIfMoveConvertible<std::unique_ptr<_Up>, _Up>,
              class = typename std::enable_if<std::is_trivially_copyable<_Up>::value &&
                                              std::is_same<deleter_type, gpu::host_delete<element_type>>::value>::type>
    __host__ _LIBGPU_INLINE_VISIBILITY unique_ptr(std::unique_ptr<_Up> &&__u)
        : __ptr_(static_cast<pointer>(gpu::malloc(sizeof(element_type) == 0 ? 1 : sizeof(element_type))),
                 __value_init_tag()) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast" // Suppress "old-style-cast" warnings because hipStreamPerThread uses one
        __LIBGPU_HIP_CHECK__(hipMemcpyAsync(__ptr_.first(), static_cast<pointer>(__u.get()), sizeof(element_type),
                                            hipMemcpyHostToDevice, hipStreamPerThread));
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(hipStreamPerThread));
#pragma clang diagnostic pop
        // Avoid calling any destructor by calling `operator delete` directly instead of using a delete-expression. This
        // isn't strictly necessary since TriviallyCopyable implies a trivial deleter.
        // Note that we also aren't invoking the gpu::unique_ptr's deleter. Even if we allowed a deleter other than
        // gpu::host_delete, we wouldn't want to invoke it here because we're only "moving" the object.
        operator delete(__u.release());
    }
    // TODO: Provide a template for a wrapper class that uses a class-specific overriden delete operator to free the memory
    // then allow the above conversion for gpu::unique_ptr<wrapper<T>, D> where D is any delete operator?

    // TODO: provide a constructor to convert gpu::unique_ptr<T, gpu::host_delete> to gpu::unique_ptr<T, gpu::default_delete>.
    // It would need to run on host, but launch a kernel to copy the memory to a block allocated with device-side new/malloc

    // Also maybe do the same for converting std::unique_ptr<T, std::default_delete> to gpu::unique_ptr<T, gpu::default_delete>?

    template <
        class _Up = element_type, bool _Dummy = true,
        class = typename std::enable_if<std::is_convertible<pointer, _Up *>::value &&
                                        __dependent_type<std::is_trivially_copyable<element_type>, _Dummy>::value &&
                                        std::is_same<deleter_type, gpu::host_delete<element_type>>::value>::type>
    __host__ _LIBGPU_INLINE_VISIBILITY operator std::unique_ptr<_Up>() && {
        // Avoid calling any constructor by calling `operator new` directly instead of using a new-expression.
        void* __buf = operator new(sizeof(element_type));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast" // Suppress "old-style-cast" warnings because hipStreamPerThread uses one
        __LIBGPU_HIP_CHECK__(
            hipMemcpyAsync(__buf, __ptr_.first(), sizeof(element_type), hipMemcpyDeviceToHost, hipStreamPerThread));
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(hipStreamPerThread));
#pragma clang diagnostic pop
        // Even if we allowed a deleter other than gpu::host_delete, we wouldn't want to invoke it here because we're
        // only "moving" the object.
        gpu::free(release());
        return std::unique_ptr<_Up>(static_cast<pointer>(__buf));
    }
    template <
        class _Up = element_type, bool _Dummy = true,
        class = typename std::enable_if<std::is_convertible<pointer, _Up *>::value &&
                                        __dependent_type<std::is_trivially_copyable<element_type>, _Dummy>::value &&
                                        std::is_same<deleter_type, gpu::host_delete<element_type>>::value>::type>
    __host__ _LIBGPU_INLINE_VISIBILITY std::unique_ptr<_Up> move_to_host() && {
        return std::move(*this);
    }

    // TODO: Do we also need assignment operators for std::vector to gpu::vector?

#ifdef _LIBGPU_CXX03_LANG
    __host__ __device__ unique_ptr(unique_ptr const &) = delete;
    __host__ __device__ unique_ptr &operator=(unique_ptr const &) = delete;
#endif

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 ~unique_ptr() { reset(); }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &operator=(std::nullptr_t) _NOEXCEPT {
        reset();
        return *this;
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 std::add_lvalue_reference_t<_Tp> operator*() const {
        return *__ptr_.first();
    }
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer operator->() const _NOEXCEPT {
        return __ptr_.first();
    }

  private:
    // Helper class that fetches a copy of the data from the device
    struct MemberAccessHelper {
        // Avoid calling any constructor of element_type by using a char buffer for storage.
        char __buf[sizeof(element_type)];
        __host__ _LIBGPU_INLINE_VISIBILITY MemberAccessHelper(pointer __p) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast" // Suppress "old-style-cast" warnings because hipStreamPerThread uses one
            __LIBGPU_HIP_CHECK__(hipMemcpyAsync(__buf, __p, sizeof(element_type), hipMemcpyDeviceToHost, hipStreamPerThread));
            __LIBGPU_HIP_CHECK__(hipStreamSynchronize(hipStreamPerThread));
#pragma clang diagnostic pop
        }
        __host__ _LIBGPU_INLINE_VISIBILITY typename std::add_const<element_type>::type get() _NOEXCEPT {
            // We use std::move to avoid calling any constructors (the dereference operation produces an lvalue).
            return std::move(*reinterpret_cast<std::remove_const_t<element_type> *>(__buf));
        }
        __host__ _LIBGPU_INLINE_VISIBILITY typename std::add_const<element_type>::type *operator->() _NOEXCEPT {
            return reinterpret_cast<element_type *>(__buf);
        }
    };
  public:
    template <bool _Dummy = true, class = typename std::enable_if<std::is_scalar<element_type>::value ||
                                      __dependent_type<std::is_trivially_copyable<element_type>, _Dummy>::value>::type>
    __host__ _LIBGPU_INLINE_VISIBILITY const element_type operator*() const {
        return MemberAccessHelper(__ptr_.first()).get();
    }
    template <bool _Dummy = true, class = typename std::enable_if<std::is_scalar<element_type>::value ||
                                      __dependent_type<std::is_trivially_copyable<element_type>, _Dummy>::value>::type>
    __host__ _LIBGPU_INLINE_VISIBILITY MemberAccessHelper operator->() const _NOEXCEPT {
        return MemberAccessHelper(__ptr_.first());
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer get() const _NOEXCEPT {
        return __ptr_.first();
    }
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 deleter_type &get_deleter() _NOEXCEPT {
        return __ptr_.second();
    }
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 const deleter_type &
    get_deleter() const _NOEXCEPT {
        return __ptr_.second();
    }
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit operator bool() const _NOEXCEPT {
        return __ptr_.first() != nullptr;
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer release() _NOEXCEPT {
        pointer __t = __ptr_.first();
        __ptr_.first() = pointer();
        return __t;
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void reset(pointer __p = pointer()) _NOEXCEPT {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = __p;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr &__u) _NOEXCEPT {
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
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr() _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR unique_ptr(std::nullptr_t) _NOEXCEPT
        : __ptr_(__value_init_tag(), __value_init_tag()) {}

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>,
              class = _EnableIfPointerConvertible<_Pp>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit unique_ptr(_Pp __p) _NOEXCEPT
        : __ptr_(__p, __value_init_tag()) {}

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>,
              class = _EnableIfPointerConvertible<_Pp>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(_Pp __p,
                                                                                  _LValRefType<_Dummy> __d) _NOEXCEPT
        : __ptr_(__p, __d) {}

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(std::nullptr_t,
                                                                                  _LValRefType<_Dummy> __d) _NOEXCEPT
        : __ptr_(nullptr, __d) {}

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>,
              class = _EnableIfPointerConvertible<_Pp>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    unique_ptr(_Pp __p, _GoodRValRefType<_Dummy> __d) _NOEXCEPT : __ptr_(__p, std::move(__d)) {
        static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference");
    }

    template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
    unique_ptr(std::nullptr_t, _GoodRValRefType<_Dummy> __d) _NOEXCEPT : __ptr_(nullptr, std::move(__d)) {
        static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference");
    }

    template <class _Pp, bool _Dummy = true, class = _EnableIfDeleterConstructible<_BadRValRefType<_Dummy>>,
              class = _EnableIfPointerConvertible<_Pp>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY unique_ptr(_Pp __p, _BadRValRefType<_Dummy> __d) = delete;

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<deleter_type>(__u.get_deleter())) {}

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<deleter_type>(__u.get_deleter());
        return *this;
    }

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterConvertible<_Ep>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT
        : __ptr_(__u.release(), std::forward<_Ep>(__u.get_deleter())) {}

    template <class _Up, class _Ep, class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
              class = _EnableIfDeleterAssignable<_Ep>>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &
    operator=(unique_ptr<_Up, _Ep> &&__u) _NOEXCEPT {
        reset(__u.release());
        __ptr_.second() = std::forward<_Ep>(__u.get_deleter());
        return *this;
    }

    template <class _Up = element_type[], class = _EnableIfMoveConvertible<std::unique_ptr<_Up>, _Up>,
              class = typename std::enable_if<std::is_trivially_copyable<_Up>::value &&
                                              std::is_same<deleter_type, gpu::host_delete<element_type[]>>::value>::type>
    __host__ _LIBGPU_INLINE_VISIBILITY unique_ptr(std::unique_ptr<_Up> &&__u, std::size_t __n)
        : __ptr_(static_cast<pointer>(gpu::malloc(sizeof(element_type[__n]) == 0 ? 1 : sizeof(element_type[__n]))), __value_init_tag()) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast" // Suppress "old-style-cast" warnings because hipStreamPerThread uses one
        __LIBGPU_HIP_CHECK__(hipMemcpyAsync(__ptr_.first(), static_cast<pointer>(__u.get()), sizeof(element_type[__n]),
                                            hipMemcpyHostToDevice, hipStreamPerThread));
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(hipStreamPerThread));
#pragma clang diagnostic pop
        // Avoid calling any destructor by calling `operator delete[]` directly instead of using a delete-expression. This
        // isn't strictly necessary since TriviallyCopyable implies a trivial deleter.
        // Note that we also aren't invoking the gpu::unique_ptr's deleter. Even if we allowed a deleter other than
        // gpu::host_delete, we wouldn't want to invoke it here because we're only "moving" the object.
        operator delete[](__u.release());
    }
    // TODO: Provide all the same options for converting types as we do for the non-array version

    template <
        class _Up = element_type, bool _Dummy = true,
        class = typename std::enable_if<std::is_convertible<element_type (*)[], _Up (*)[]>::value &&
                                        __dependent_type<std::is_trivially_copyable<element_type[]>, _Dummy>::value &&
                                        std::is_same<deleter_type, gpu::host_delete<element_type[]>>::value>::type>
    __host__ _LIBGPU_INLINE_VISIBILITY std::unique_ptr<_Up[]> move_to_host(std::size_t __n) && {
        // Avoid calling any constructor by calling `operator new[]` directly instead of using a new-expression
        void* __buf = operator new[](sizeof(element_type[__n]));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast" // Suppress "old-style-cast" warnings because hipStreamPerThread uses one
        __LIBGPU_HIP_CHECK__(hipMemcpyAsync(__buf, __ptr_.first(), sizeof(element_type[__n]), hipMemcpyDeviceToHost, hipStreamPerThread));
        __LIBGPU_HIP_CHECK__(hipStreamSynchronize(hipStreamPerThread));
#pragma clang diagnostic pop
        // Even if we allowed a deleter other than gpu::host_delete, we wouldn't want to invoke it here because we're
        // only "moving" the object.
        gpu::free(release());
        return std::unique_ptr<_Up[]>(static_cast<pointer>(__buf));
    }

#ifdef _LIBGPU_CXX03_LANG
    __host__ __device__ unique_ptr(unique_ptr const &) = delete;
    __host__ __device__ unique_ptr &operator=(unique_ptr const &) = delete;
#endif
  public:
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 ~unique_ptr() { reset(); }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 unique_ptr &operator=(std::nullptr_t) _NOEXCEPT {
        reset();
        return *this;
    }

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 std::add_lvalue_reference_t<_Tp>
    operator[](std::size_t __i) const {
        return __ptr_.first()[__i];
    }
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer get() const _NOEXCEPT {
        return __ptr_.first();
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 deleter_type &get_deleter() _NOEXCEPT {
        return __ptr_.second();
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 const deleter_type &
    get_deleter() const _NOEXCEPT {
        return __ptr_.second();
    }
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 explicit operator bool() const _NOEXCEPT {
        return __ptr_.first() != nullptr;
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 pointer release() _NOEXCEPT {
        pointer __t = __ptr_.first();
        __ptr_.first() = pointer();
        return __t;
    }

    template <class _Pp>
    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
        typename std::enable_if<_CheckArrayPointerConversion<_Pp>::value>::type
        reset(_Pp __p) _NOEXCEPT {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = __p;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void reset(std::nullptr_t = nullptr) _NOEXCEPT {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = nullptr;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    __host__ __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 void swap(unique_ptr &__u) _NOEXCEPT {
        __ptr_.swap(__u.__ptr_);
    }
};

template <class _Tp>
using unique_ptr_h = unique_ptr<_Tp, host_delete<_Tp>>;

template <class _Tp, class _Dp>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
typename std::enable_if<std::is_swappable<_Dp>::value, void>::type
swap(unique_ptr<_Tp, _Dp> &__x, unique_ptr<_Tp, _Dp> &__y) _NOEXCEPT {
    __x.swap(__y);
}

template <class _T1, class _D1, class _T2, class _D2>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator==(const unique_ptr<_T1, _D1> &__x, const unique_ptr<_T2, _D2> &__y) {
    return __x.get() == __y.get();
}

#if _LIBGPU_STD_VER <= 17
template <class _T1, class _D1, class _T2, class _D2>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator!=(const unique_ptr<_T1, _D1> &__x,
                                                                     const unique_ptr<_T2, _D2> &__y) {
    return !(__x == __y);
}
#endif

template <class _T1, class _D1, class _T2, class _D2>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator<(const unique_ptr<_T1, _D1> &__x,
                                                           const unique_ptr<_T2, _D2> &__y) {
    typedef typename unique_ptr<_T1, _D1>::pointer _P1;
    typedef typename unique_ptr<_T2, _D2>::pointer _P2;
    typedef typename std::common_type<_P1, _P2>::type _Vp;
    return less<_Vp>()(__x.get(), __y.get());
}

template <class _T1, class _D1, class _T2, class _D2>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator>(const unique_ptr<_T1, _D1> &__x,
                                                           const unique_ptr<_T2, _D2> &__y) {
    return __y < __x;
}

template <class _T1, class _D1, class _T2, class _D2>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator<=(const unique_ptr<_T1, _D1> &__x,
                                                            const unique_ptr<_T2, _D2> &__y) {
    return !(__y < __x);
}

template <class _T1, class _D1, class _T2, class _D2>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator>=(const unique_ptr<_T1, _D1> &__x,
                                                            const unique_ptr<_T2, _D2> &__y) {
    return !(__x < __y);
}

#if _LIBGPU_STD_VER >= 20
template <class _T1, class _D1, class _T2, class _D2>
    requires three_way_comparable_with<typename unique_ptr<_T1, _D1>::pointer, typename unique_ptr<_T2, _D2>::pointer>
_LIBGPU_HIDE_FROM_ABI
__host__ __device__ compare_three_way_result_t<typename unique_ptr<_T1, _D1>::pointer, typename unique_ptr<_T2, _D2>::pointer>
operator<=>(const unique_ptr<_T1, _D1> &__x, const unique_ptr<_T2, _D2> &__y) {
    return compare_three_way()(__x.get(), __y.get());
}
#endif

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator==(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) _NOEXCEPT {
    return !__x;
}

#if _LIBGPU_STD_VER <= 17
template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator==(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) _NOEXCEPT {
    return !__x;
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator!=(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) _NOEXCEPT {
    return static_cast<bool>(__x);
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY bool operator!=(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) _NOEXCEPT {
    return static_cast<bool>(__x);
}
#endif // _LIBGPU_STD_VER <= 17

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    typedef typename unique_ptr<_T1, _D1>::pointer _P1;
    return less<_P1>()(__x.get(), nullptr);
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    typedef typename unique_ptr<_T1, _D1>::pointer _P1;
    return less<_P1>()(nullptr, __x.get());
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return nullptr < __x;
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    return __x < nullptr;
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<=(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return !(nullptr < __x);
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator<=(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    return !(__x < nullptr);
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>=(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return !(__x < nullptr);
}

template <class _T1, class _D1>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 bool
operator>=(std::nullptr_t, const unique_ptr<_T1, _D1> &__x) {
    return !(nullptr < __x);
}

#if _LIBGPU_STD_VER >= 20
template <class _T1, class _D1>
    requires three_way_comparable<typename unique_ptr<_T1, _D1>::pointer>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX23 compare_three_way_result_t<typename unique_ptr<_T1, _D1>::pointer>
operator<=>(const unique_ptr<_T1, _D1> &__x, std::nullptr_t) {
    return compare_three_way()(__x.get(), static_cast<typename unique_ptr<_T1, _D1>::pointer>(nullptr));
}
#endif

#if _LIBGPU_STD_VER >= 14

template <class _Tp>
struct __unique_if {
    typedef unique_ptr<_Tp> __unique_single;
    typedef unique_ptr<_Tp, host_delete<_Tp>> __unique_single_host;
};

template <class _Tp>
struct __unique_if<_Tp[]> {
    typedef unique_ptr<_Tp[]> __unique_array_unknown_bound;
    typedef unique_ptr<_Tp[], host_delete<_Tp[]>> __unique_array_unknown_bound_host;
};

template <class _Tp, std::size_t _Np>
struct __unique_if<_Tp[_Np]> {
    typedef void __unique_array_known_bound;
};

template <class _Tp, class... _Args>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23 typename __unique_if<_Tp>::__unique_single
make_unique(_Args &&...__args) {
    return unique_ptr<_Tp>(new _Tp(std::forward<_Args>(__args)...));
}

template <class _Tp>
__host__ inline _LIBGPU_INLINE_VISIBILITY typename __unique_if<_Tp>::__unique_single_host
make_unique() {
    static_assert(std::is_trivially_default_constructible<_Tp>::value,
                  "Host code can't invoke a non-trivial constructor for objects in device memory");
    void *__buf = gpu::malloc(sizeof(_Tp) == 0 ? 1 : sizeof(_Tp));
    return unique_ptr<_Tp, host_delete<_Tp>>(static_cast<_Tp *>(__buf));
}

template <class _T1, class _T2>
__host__ inline _LIBGPU_INLINE_VISIBILITY typename __unique_if<_T1>::__unique_single_host
make_unique(_T2 &&__arg) {
    using __RefType = decltype(std::forward<_T2>(__arg));
    static_assert(std::is_constructible<_T1, __RefType>::value, "No valid constructor found");
    static_assert(std::is_trivially_constructible<_T1, __RefType>::value,
                  "Host code can't invoke a non-trivial constructor for objects in device memory");
    void *__buf = gpu::malloc(sizeof(_T1) == 0 ? 1 : sizeof(_T1));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast" // Suppress "old-style-cast" warnings because hipStreamPerThread uses one
    __LIBGPU_HIP_CHECK__(hipMemcpyAsync(__buf, &__arg, sizeof(_T1), hipMemcpyHostToDevice, hipStreamPerThread));
    __LIBGPU_HIP_CHECK__(hipStreamSynchronize(hipStreamPerThread));
#pragma clang diagnostic pop
    return unique_ptr<_T1, host_delete<_T1>>(static_cast<_T1 *>(__buf));
}

template <class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX23
typename __unique_if<_Tp>::__unique_array_unknown_bound
make_unique(std::size_t __n) {
    typedef std::remove_extent_t<_Tp> _Up;
    return unique_ptr<_Tp>(new _Up[__n]());
}

template <class _Tp>
__host__ inline _LIBGPU_INLINE_VISIBILITY
typename __unique_if<_Tp>::__unique_array_unknown_bound_host
make_unique(std::size_t __n) {
    typedef std::remove_extent_t<_Tp> _Up;
    static_assert(std::is_trivially_default_constructible<_Up>::value,
                  "Host code can't invoke a non-trivial constructor for objects in device memory");
    void *__buf = gpu::malloc(sizeof(_Up[__n]) == 0 ? 1 : sizeof(_Up[__n]));
    return unique_ptr<_Tp, host_delete<_Tp>>(static_cast<_Up *>(__buf));
}

template <class _Tp, class... _Args>
__host__ __device__ typename __unique_if<_Tp>::__unique_array_known_bound make_unique(_Args &&...) = delete;

#endif // _LIBGPU_STD_VER >= 14

#if _LIBGPU_STD_VER >= 20

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX23 typename __unique_if<_Tp>::__unique_single
make_unique_for_overwrite() {
    return unique_ptr<_Tp>(new _Tp);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX23 typename __unique_if<_Tp>::__unique_array_unknown_bound
make_unique_for_overwrite(std::size_t __n) {
    return unique_ptr<_Tp>(new std::remove_extent_t<_Tp>[__n]);
}

template <class _Tp, class... _Args>
__host__ __device__ typename __unique_if<_Tp>::__unique_array_known_bound make_unique_for_overwrite(_Args &&...) = delete;

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___MEMORY_UNIQUE_PTR_H__
