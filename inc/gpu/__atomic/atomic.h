#ifndef __GPU___ATOMIC_ATOMIC_H__
#define __GPU___ATOMIC_ATOMIC_H__

#include "hip/hip_runtime_api.h"
#include <atomic>
#include <cstddef>
#include <cstring>

#include "gpu/__atomic/atomic_base.h"
#include "gpu/__atomic/cxx_atomic_impl.h"
#include "gpu/__atomic/memory_order.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::atomic
//====================================================================================================================//

// TODO: add a thread_scope template parameter

// atomic<T>
template <class _Tp>
struct atomic : public gpu::internal::__atomic_base<_Tp> {
    typedef gpu::internal::__atomic_base<_Tp> __base;
    typedef _Tp value_type;
    typedef value_type difference_type;

    inline __host__ __device__ atomic() noexcept = default;

    inline constexpr __host__ __device__ atomic(_Tp __d) noexcept : __base(__d) {}

    inline __host__ __device__ _Tp operator=(_Tp __d) volatile noexcept {
        __base::store(__d);
        return __d;
    }
    inline __host__ __device__ _Tp operator=(_Tp __d) noexcept {
        __base::store(__d);
        return __d;
    }

    atomic &operator=(const atomic &) = delete;
    atomic &operator=(const atomic &) volatile = delete;
};

// atomic<T*>

template <class _Tp>
struct atomic<_Tp *> : public gpu::internal::__atomic_base<_Tp *> {
    typedef gpu::internal::__atomic_base<_Tp *> __base;
    typedef _Tp *value_type;
    typedef ptrdiff_t difference_type;

    inline __host__ __device__ atomic() noexcept = default;

    inline constexpr __host__ __device__ atomic(_Tp *__d) noexcept : __base(__d) {}

    inline __host__ __device__ _Tp *operator=(_Tp *__d) volatile noexcept {
        __base::store(__d);
        return __d;
    }
    inline __host__ __device__ _Tp *operator=(_Tp *__d) noexcept {
        __base::store(__d);
        return __d;
    }

    inline __host__ __device__ _Tp *fetch_add(ptrdiff_t __op,
                                              memory_order __m = memory_order_seq_cst) volatile noexcept {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function_v<std::remove_pointer_t<_Tp>>, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_add(&this->__a_, __op, __m);
    }

    inline __host__ __device__ _Tp *fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) noexcept {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function_v<std::remove_pointer_t<_Tp>>, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_add(&this->__a_, __op, __m);
    }

    inline __host__ __device__ _Tp *fetch_sub(ptrdiff_t __op,
                                              memory_order __m = memory_order_seq_cst) volatile noexcept {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function_v<std::remove_pointer_t<_Tp>>, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_sub(&this->__a_, __op, __m);
    }

    inline __host__ __device__ _Tp *fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) noexcept {
        // __atomic_fetch_add accepts function pointers, guard against them.
        static_assert(!std::is_function_v<std::remove_pointer_t<_Tp>>, "Pointer to function isn't allowed");
        return gpu::internal::__cxx_atomic_fetch_sub(&this->__a_, __op, __m);
    }

    inline __host__ __device__ _Tp *operator++(int) volatile noexcept               { return fetch_add(1); }
    inline __host__ __device__ _Tp *operator++(int) noexcept                        { return fetch_add(1); }
    inline __host__ __device__ _Tp *operator--(int) volatile noexcept               { return fetch_sub(1); }
    inline __host__ __device__ _Tp *operator--(int) noexcept                        { return fetch_sub(1); }
    inline __host__ __device__ _Tp *operator++() volatile noexcept                  { return fetch_add(1) + 1; }
    inline __host__ __device__ _Tp *operator++() noexcept                           { return fetch_add(1) + 1; }
    inline __host__ __device__ _Tp *operator--() volatile noexcept                  { return fetch_sub(1) - 1; }
    inline __host__ __device__ _Tp *operator--() noexcept                           { return fetch_sub(1) - 1; }
    inline __host__ __device__ _Tp *operator+=(ptrdiff_t __op) volatile noexcept    { return fetch_add(__op) + __op; }
    inline __host__ __device__ _Tp *operator+=(ptrdiff_t __op) noexcept             { return fetch_add(__op) + __op; }
    inline __host__ __device__ _Tp *operator-=(ptrdiff_t __op) volatile noexcept    { return fetch_sub(__op) - __op; }
    inline __host__ __device__ _Tp *operator-=(ptrdiff_t __op) noexcept             { return fetch_sub(__op) - __op; }

    __host__ __device__ atomic &operator=(const atomic &) = delete;
    __host__ __device__ atomic &operator=(const atomic &) volatile = delete;
};

} // namespace gpu

#endif // __GPU___ATOMIC_ATOMIC_H__
