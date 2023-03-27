#ifndef __GPU___ATOMIC_ATOMIC_H__
#define __GPU___ATOMIC_ATOMIC_H__

#include "hip/hip_runtime_api.h"
#include <atomic>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "gpu/__atomic/atomic_base.h"
#include "gpu/__atomic/cxx_atomic_impl.h"
#include "gpu/__atomic/memory_order.h"
#include "gpu/__memory/addressof.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::atomic
//====================================================================================================================//

// TODO: add a thread_scope template parameter

// atomic<T>
template <class _Tp>
struct atomic : public gpu::internal::__atomic_base<_Tp> {
    using __base = gpu::internal::__atomic_base<_Tp>;
    using value_type = _Tp;
    using difference_type = value_type;

#if _LIBGPU_STD_VER >= 20
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI atomic() = default;
#else
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI atomic() _NOEXCEPT = default;
#endif

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR atomic(_Tp __d) _NOEXCEPT : __base(__d) {}

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp operator=(_Tp __d) volatile _NOEXCEPT {
        __base::store(__d);
        return __d;
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp operator=(_Tp __d) _NOEXCEPT {
        __base::store(__d);
        return __d;
    }

    atomic &operator=(const atomic &) = delete;
    atomic &operator=(const atomic &) volatile = delete;
};

// atomic<T*>

template <class _Tp>
struct atomic<_Tp *> : public gpu::internal::__atomic_base<_Tp *> {
    using __base = gpu::internal::__atomic_base<_Tp *>;
    using value_type = _Tp *;
    using difference_type = ptrdiff_t;

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI atomic() _NOEXCEPT = default;

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR atomic(_Tp *__d) _NOEXCEPT : __base(__d) {}

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp *operator=(_Tp *__d) volatile _NOEXCEPT {
        __base::store(__d);
        return __d;
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp *operator=(_Tp *__d) _NOEXCEPT {
        __base::store(__d);
        return __d;
    }

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function<std::remove_pointer_t<_Tp>>::value, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_add(gpu::addressof(this->__a_), __op, __m);
    }

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function<std::remove_pointer_t<_Tp>>::value, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_add(gpu::addressof(this->__a_), __op, __m);
    }

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
     _Tp *fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function<std::remove_pointer_t<_Tp>>::value, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_sub(gpu::addressof(this->__a_), __op, __m);
    }

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function<std::remove_pointer_t<_Tp>>::value, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_sub(gpu::addressof(this->__a_), __op, __m);
    }

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator++(int) volatile _NOEXCEPT               { return fetch_add(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator++(int) _NOEXCEPT                        { return fetch_add(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator--(int) volatile _NOEXCEPT               { return fetch_sub(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator--(int) _NOEXCEPT                        { return fetch_sub(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator++() volatile _NOEXCEPT                  { return fetch_add(1) + 1; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator++() _NOEXCEPT                           { return fetch_add(1) + 1; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator--() volatile _NOEXCEPT                  { return fetch_sub(1) - 1; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator--() _NOEXCEPT                           { return fetch_sub(1) - 1; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator+=(ptrdiff_t __op) volatile _NOEXCEPT    { return fetch_add(__op) + __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator+=(ptrdiff_t __op) _NOEXCEPT             { return fetch_add(__op) + __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator-=(ptrdiff_t __op) volatile _NOEXCEPT    { return fetch_sub(__op) - __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp *operator-=(ptrdiff_t __op) _NOEXCEPT             { return fetch_sub(__op) - __op; }

    __host__ __device__ atomic &operator=(const atomic &) = delete;
    __host__ __device__ atomic &operator=(const atomic &) volatile = delete;
};

// atomic_is_lock_free

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool atomic_is_lock_free(const volatile atomic<_Tp> *__o) _NOEXCEPT {
    return __o->is_lock_free();
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool atomic_is_lock_free(const atomic<_Tp> *__o) _NOEXCEPT {
    return __o->is_lock_free();
}

// atomic_init

template <class _Tp>
_LIBGPU_DEPRECATED_IN_CXX20 __host__ __device__ _LIBGPU_HIDE_FROM_ABI void
atomic_init(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    gpu::internal::__cxx_atomic_init(gpu::addressof(__o->__a_), __d);
}

template <class _Tp>
_LIBGPU_DEPRECATED_IN_CXX20 __host__ __device__ _LIBGPU_HIDE_FROM_ABI void
atomic_init(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    gpu::internal::__cxx_atomic_init(gpu::addressof(__o->__a_), __d);
}

// atomic_store

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI void atomic_store(volatile atomic<_Tp> *__o,
                                                            typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    __o->store(__d);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI void atomic_store(atomic<_Tp> *__o,
                                                            typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    __o->store(__d);
}

// atomic_store_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI void
atomic_store_explicit(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __d, memory_order __m) _NOEXCEPT {
    __o->store(__d, __m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI void
atomic_store_explicit(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __d, memory_order __m) _NOEXCEPT {
    __o->store(__d, __m);
}

// atomic_load

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_load(const volatile atomic<_Tp> *__o) _NOEXCEPT {
    return __o->load();
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_load(const atomic<_Tp> *__o) _NOEXCEPT {
    return __o->load();
}

// atomic_load_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_load_explicit(const volatile atomic<_Tp> *__o,
                                                                   memory_order __m) _NOEXCEPT {
    return __o->load(__m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_load_explicit(const atomic<_Tp> *__o, memory_order __m) _NOEXCEPT {
    return __o->load(__m);
}

// atomic_exchange

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_exchange(volatile atomic<_Tp> *__o,
                                                              typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    return __o->exchange(__d);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_exchange(atomic<_Tp> *__o,
                                                              typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    return __o->exchange(__d);
}

// atomic_exchange_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_exchange_explicit(volatile atomic<_Tp> *__o,
                                                                       typename atomic<_Tp>::value_type __d,
                                                                       memory_order __m) _NOEXCEPT {
    return __o->exchange(__d, __m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_exchange_explicit(atomic<_Tp> *__o,
                                                                       typename atomic<_Tp>::value_type __d,
                                                                       memory_order __m) _NOEXCEPT {
    return __o->exchange(__d, __m);
}

// atomic_compare_exchange_weak

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_weak(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                             typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    return __o->compare_exchange_weak(*__e, __d);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_weak(atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                             typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    return __o->compare_exchange_weak(*__e, __d);
}

// atomic_compare_exchange_strong

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_strong(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                               typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    return __o->compare_exchange_strong(*__e, __d);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_strong(atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                               typename atomic<_Tp>::value_type __d) _NOEXCEPT {
    return __o->compare_exchange_strong(*__e, __d);
}

// atomic_compare_exchange_weak_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_weak_explicit(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                                      typename atomic<_Tp>::value_type __d, memory_order __s,
                                      memory_order __f) _NOEXCEPT {
    return __o->compare_exchange_weak(*__e, __d, __s, __f);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_weak_explicit(atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                                      typename atomic<_Tp>::value_type __d, memory_order __s,
                                      memory_order __f) _NOEXCEPT {
    return __o->compare_exchange_weak(*__e, __d, __s, __f);
}

// atomic_compare_exchange_strong_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_strong_explicit(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                                        typename atomic<_Tp>::value_type __d, memory_order __s,
                                        memory_order __f) _NOEXCEPT {
    return __o->compare_exchange_strong(*__e, __d, __s, __f);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI bool
atomic_compare_exchange_strong_explicit(atomic<_Tp> *__o, typename atomic<_Tp>::value_type *__e,
                                        typename atomic<_Tp>::value_type __d, memory_order __s,
                                        memory_order __f) _NOEXCEPT {
    return __o->compare_exchange_strong(*__e, __d, __s, __f);
}

// atomic_fetch_add

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_add(volatile atomic<_Tp> *__o,
                                                               typename atomic<_Tp>::difference_type __op) _NOEXCEPT {
    return __o->fetch_add(__op);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_add(atomic<_Tp> *__o,
                                                               typename atomic<_Tp>::difference_type __op) _NOEXCEPT {
    return __o->fetch_add(__op);
}

// atomic_fetch_add_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_add_explicit(volatile atomic<_Tp> *__o,
                                                                        typename atomic<_Tp>::difference_type __op,
                                                                        memory_order __m) _NOEXCEPT {
    return __o->fetch_add(__op, __m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_add_explicit(atomic<_Tp> *__o,
                                                                        typename atomic<_Tp>::difference_type __op,
                                                                        memory_order __m) _NOEXCEPT {
    return __o->fetch_add(__op, __m);
}

// atomic_fetch_sub

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_sub(volatile atomic<_Tp> *__o,
                                                               typename atomic<_Tp>::difference_type __op) _NOEXCEPT {
    return __o->fetch_sub(__op);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_sub(atomic<_Tp> *__o,
                                                               typename atomic<_Tp>::difference_type __op) _NOEXCEPT {
    return __o->fetch_sub(__op);
}

// atomic_fetch_sub_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_sub_explicit(volatile atomic<_Tp> *__o,
                                                                        typename atomic<_Tp>::difference_type __op,
                                                                        memory_order __m) _NOEXCEPT {
    return __o->fetch_sub(__op, __m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp atomic_fetch_sub_explicit(atomic<_Tp> *__o,
                                                                        typename atomic<_Tp>::difference_type __op,
                                                                        memory_order __m) _NOEXCEPT {
    return __o->fetch_sub(__op, __m);
}

// atomic_fetch_and

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_and(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op) _NOEXCEPT {
    return __o->fetch_and(__op);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_and(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op) _NOEXCEPT {
    return __o->fetch_and(__op);
}

// atomic_fetch_and_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_and_explicit(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op,
                            memory_order __m) _NOEXCEPT {
    return __o->fetch_and(__op, __m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_and_explicit(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op, memory_order __m) _NOEXCEPT {
    return __o->fetch_and(__op, __m);
}

// atomic_fetch_or

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_or(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op) _NOEXCEPT {
    return __o->fetch_or(__op);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_or(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op) _NOEXCEPT {
    return __o->fetch_or(__op);
}

// atomic_fetch_or_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_or_explicit(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op, memory_order __m) _NOEXCEPT {
    return __o->fetch_or(__op, __m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_or_explicit(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op, memory_order __m) _NOEXCEPT {
    return __o->fetch_or(__op, __m);
}

// atomic_fetch_xor

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_xor(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op) _NOEXCEPT {
    return __o->fetch_xor(__op);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_xor(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op) _NOEXCEPT {
    return __o->fetch_xor(__op);
}

// atomic_fetch_xor_explicit

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_xor_explicit(volatile atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op, memory_order __m) _NOEXCEPT {
    return __o->fetch_xor(__op, __m);
}

template <class _Tp>
__host__ __device__ _LIBGPU_HIDE_FROM_ABI
typename std::enable_if<std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value, _Tp>::type
atomic_fetch_xor_explicit(atomic<_Tp> *__o, typename atomic<_Tp>::value_type __op, memory_order __m) _NOEXCEPT {
    return __o->fetch_xor(__op, __m);
}

} // namespace gpu

#endif // __GPU___ATOMIC_ATOMIC_H__
