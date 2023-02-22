#ifndef __GPU_ATOMIC_H__
#define __GPU_ATOMIC_H__

#include "hip/hip_runtime.h"
#include <atomic>
#include <cstddef>
#include <cstring>

#include "gpu/__clib/memcmp.h"
#include "gpu/__clib/memcpy.h"

namespace gpu {

using std::memory_order;
using std::memory_order_acq_rel;
using std::memory_order_acquire;
using std::memory_order_consume;
using std::memory_order_relaxed;
using std::memory_order_release;
using std::memory_order_seq_cst;

namespace internal {

template <class T>
struct is_hip_native
    : std::integral_constant<bool,
                             std::is_same<float, typename std::remove_cv<T>::type>::value ||
                                 // std::is_same<double, typename std::remove_cv<T>::type>::value || // requires MI-200
                                 std::is_same<int, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<long, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<long long, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<unsigned int, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<unsigned long, typename std::remove_cv<T>::type>::value ||
                                 std::is_same<unsigned long long, typename std::remove_cv<T>::type>::value> {};

template <class T>
inline constexpr bool is_hip_native_v = is_hip_native<T>::value;

//====================================================================================================================//
//      Adapted from libc++ std::atomic
//====================================================================================================================//

typedef std::underlying_type_t<std::memory_order> __memory_order_underlying_t;

// [atomics.types.generic]p1 guarantees _Tp is trivially copyable. Because
// the default operator= in an object is not volatile, a byte-by-byte copy
// is required.
template <typename _Tp, typename _Tv>
inline __host__ __device__ typename std::enable_if_t<std::is_assignable_v<_Tp &, _Tv>>
__cxx_atomic_assign_volatile(_Tp &__a_value, _Tv const &__val) {
    __a_value = __val;
}
template <typename _Tp, typename _Tv>
inline __host__ __device__ typename std::enable_if_t<std::is_assignable_v<_Tp &, _Tv>>
__cxx_atomic_assign_volatile(_Tp volatile &__a_value, _Tv volatile const &__val) {
    volatile char *__to = reinterpret_cast<volatile char *>(&__a_value);
    volatile char *__end = __to + sizeof(_Tp);
    volatile const char *__from = reinterpret_cast<volatile const char *>(&__val);
    while (__to != __end)
        *__to++ = *__from++;
}

// Provides a base implementation that uses lock-free native operations.
template <typename _Tp>
struct __cxx_atomic_base_impl {

    inline __host__ __device__ __cxx_atomic_base_impl() noexcept = default;
    constexpr explicit __cxx_atomic_base_impl(_Tp value) noexcept : __a_value(value) {}
    _Tp __a_value;
};

// TODO: If/when __hip_atomic_is_lock_free gets implemented, uncomment this
// #define __cxx_atomic_is_lock_free(__s) __hip_atomic_is_lock_free(__s)

template <class _Tp>
inline __host__ __device__ void __cxx_atomic_init(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __val) noexcept {
    __hip_atomic_init(&__a->__a_value, __val);
}
template <class _Tp>
inline __host__ __device__ void __cxx_atomic_init(__cxx_atomic_base_impl<_Tp> *__a, _Tp __val) noexcept {
    __hip_atomic_init(&__a->__a_value, __val);
}

template <class _Tp>
inline __host__ __device__ void __cxx_atomic_store(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __val,
                                                   memory_order __order) noexcept {
    __hip_atomic_store(&__a->__a_value, __val, static_cast<__memory_order_underlying_t>(__order),
                       __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ void __cxx_atomic_store(__cxx_atomic_base_impl<_Tp> *__a, _Tp __val,
                                                   memory_order __order) noexcept {
    __hip_atomic_store(&__a->__a_value, __val, static_cast<__memory_order_underlying_t>(__order),
                       __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_load(__cxx_atomic_base_impl<_Tp> const volatile *__a,
                                                 memory_order __order) noexcept {
    using __ptr_type = std::remove_const_t<decltype(__a->__a_value)> *;
    return __hip_atomic_load(const_cast<__ptr_type>(&__a->__a_value), static_cast<__memory_order_underlying_t>(__order),
                             __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_load(__cxx_atomic_base_impl<_Tp> const *__a,
                                                 memory_order __order) noexcept {
    using __ptr_type = std::remove_const_t<decltype(__a->__a_value)> *;
    return __hip_atomic_load(const_cast<__ptr_type>(&__a->__a_value), static_cast<__memory_order_underlying_t>(__order),
                             __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_exchange(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __value,
                                                     memory_order __order) noexcept {
    return __hip_atomic_exchange(&__a->__a_value, __value, static_cast<__memory_order_underlying_t>(__order),
                                 __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_exchange(__cxx_atomic_base_impl<_Tp> *__a, _Tp __value,
                                                     memory_order __order) noexcept {
    return __hip_atomic_exchange(&__a->__a_value, __value, static_cast<__memory_order_underlying_t>(__order),
                                 __HIP_MEMORY_SCOPE_AGENT);
}

inline constexpr __host__ __device__ memory_order __to_failure_order(memory_order __order) {
    // Avoid switch statement to make this a constexpr.
    return __order == memory_order_release ? memory_order_relaxed
                                           : (__order == memory_order_acq_rel ? memory_order_acquire : __order);
}

template <class _Tp>
inline __host__ __device__ bool
__cxx_atomic_compare_exchange_strong(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp *__expected, _Tp __value,
                                     memory_order __success, memory_order __failure) noexcept {
    return __hip_atomic_compare_exchange_strong(
        &__a->__a_value, __expected, __value, static_cast<__memory_order_underlying_t>(__success),
        static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)), __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_base_impl<_Tp> *__a, _Tp *__expected,
                                                                     _Tp __value, memory_order __success,
                                                                     memory_order __failure) noexcept {
    return __hip_atomic_compare_exchange_strong(
        &__a->__a_value, __expected, __value, static_cast<__memory_order_underlying_t>(__success),
        static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)), __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_impl<_Tp> volatile *__a,
                                                                   _Tp *__expected, _Tp __value, memory_order __success,
                                                                   memory_order __failure) noexcept {
    return __hip_atomic_compare_exchange_weak(
        &__a->__a_value, __expected, __value, static_cast<__memory_order_underlying_t>(__success),
        static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)), __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_impl<_Tp> *__a, _Tp *__expected,
                                                                   _Tp __value, memory_order __success,
                                                                   memory_order __failure) noexcept {
    return __hip_atomic_compare_exchange_weak(
        &__a->__a_value, __expected, __value, static_cast<__memory_order_underlying_t>(__success),
        static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)), __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __delta,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp> *__a, _Tp __delta,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp *__cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp *> volatile *__a, ptrdiff_t __delta,
                                                       memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp *__cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp *> *__a, ptrdiff_t __delta,
                                                       memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __delta,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, -__delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp> *__a, _Tp __delta,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, -__delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp *__cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp *> volatile *__a, ptrdiff_t __delta,
                                                       memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, -__delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp *__cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp *> *__a, ptrdiff_t __delta,
                                                       memory_order __order) noexcept {
    return __hip_atomic_fetch_add(&__a->__a_value, -__delta, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_and(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __pattern,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_and(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_and(__cxx_atomic_base_impl<_Tp> *__a, _Tp __pattern,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_and(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_or(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __pattern,
                                                     memory_order __order) noexcept {
    return __hip_atomic_fetch_or(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order),
                                 __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_or(__cxx_atomic_base_impl<_Tp> *__a, _Tp __pattern,
                                                     memory_order __order) noexcept {
    return __hip_atomic_fetch_or(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order),
                                 __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_impl<_Tp> volatile *__a, _Tp __pattern,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_xor(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}
template <class _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_impl<_Tp> *__a, _Tp __pattern,
                                                      memory_order __order) noexcept {
    return __hip_atomic_fetch_xor(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order),
                                  __HIP_MEMORY_SCOPE_AGENT);
}

template <class _Tp>
struct __libcpp_is_always_lock_free {
    // TODO: If/when HIP gets an equivalent to __atomic_always_lock_free uncomment this and remove the following line
    // static const bool __value = __atomic_always_lock_free(sizeof(_Tp), nullptr);
    static const bool __value = gpu::internal::is_hip_native_v<_Tp>;
};

// For types without a native operation, we'll provide overloads of the __cxx_atomic_*** functions that use a lock.
// The lock is implemented using the native/base atomic operations on an unsigned int.
template <typename _Tp>
struct __cxx_atomic_lock_impl {

    inline __host__ __device__ __cxx_atomic_lock_impl() noexcept : __a_value(), __a_lock(0) {}
    inline constexpr __host__ __device__ explicit __cxx_atomic_lock_impl(_Tp value) noexcept
        : __a_value(value), __a_lock(0) {}

    _Tp __a_value;
    mutable __cxx_atomic_base_impl<unsigned int> __a_lock;

    inline __host__ __device__ void __lock() const volatile {
        while (1 == __cxx_atomic_exchange(&__a_lock, unsigned(true), memory_order_acquire))
            /*spin*/;
    }
    inline __host__ __device__ void __lock() const {
        while (1 == __cxx_atomic_exchange(&__a_lock, unsigned(true), memory_order_acquire))
            /*spin*/;
    }
    inline __host__ __device__ void __unlock() const volatile {
        __cxx_atomic_store(&__a_lock, unsigned(false), memory_order_release);
    }
    inline __host__ __device__ void __unlock() const {
        __cxx_atomic_store(&__a_lock, unsigned(false), memory_order_release);
    }
    inline __host__ __device__ _Tp __read() const volatile {
        __lock();
        _Tp __old;
        __cxx_atomic_assign_volatile(__old, __a_value);
        __unlock();
        return __old;
    }
    inline __host__ __device__ _Tp __read() const {
        __lock();
        _Tp __old = __a_value;
        __unlock();
        return __old;
    }
};

template <typename _Tp>
inline __host__ __device__ void __cxx_atomic_init(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Tp __val) {
    __cxx_atomic_assign_volatile(__a->__a_value, __val);
}
template <typename _Tp>
inline __host__ __device__ void __cxx_atomic_init(__cxx_atomic_lock_impl<_Tp> *__a, _Tp __val) {
    __a->__a_value = __val;
}

template <typename _Tp>
inline __host__ __device__ void __cxx_atomic_store(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Tp __val, memory_order) {
    __a->__lock();
    __cxx_atomic_assign_volatile(__a->__a_value, __val);
    __a->__unlock();
}
template <typename _Tp>
inline __host__ __device__ void __cxx_atomic_store(__cxx_atomic_lock_impl<_Tp> *__a, _Tp __val, memory_order) {
    __a->__lock();
    __a->__a_value = __val;
    __a->__unlock();
}

template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_load(const volatile __cxx_atomic_lock_impl<_Tp> *__a, memory_order) {
    return __a->__read();
}
template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_load(const __cxx_atomic_lock_impl<_Tp> *__a, memory_order) {
    return __a->__read();
}

template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_exchange(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Tp __value,
                                                     memory_order) {
    __a->__lock();
    _Tp __old;
    __cxx_atomic_assign_volatile(__old, __a->__a_value);
    __cxx_atomic_assign_volatile(__a->__a_value, __value);
    __a->__unlock();
    return __old;
}
template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_exchange(__cxx_atomic_lock_impl<_Tp> *__a, _Tp __value, memory_order) {
    __a->__lock();
    _Tp __old = __a->__a_value;
    __a->__a_value = __value;
    __a->__unlock();
    return __old;
}

template <typename _Tp>
inline __host__ __device__ bool __cxx_atomic_compare_exchange_strong(volatile __cxx_atomic_lock_impl<_Tp> *__a,
                                                                     _Tp *__expected, _Tp __value, memory_order,
                                                                     memory_order) {
    _Tp __temp;
    __a->__lock();
    __cxx_atomic_assign_volatile(__temp, __a->__a_value);
    bool __ret = (gpu::memcmp(&__temp, __expected, sizeof(_Tp)) == 0);
    if (__ret)
        __cxx_atomic_assign_volatile(__a->__a_value, __value);
    else
        __cxx_atomic_assign_volatile(*__expected, __a->__a_value);
    __a->__unlock();
    return __ret;
}
template <typename _Tp>
inline __host__ __device__ bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_lock_impl<_Tp> *__a, _Tp *__expected,
                                                                     _Tp __value, memory_order, memory_order) {
    __a->__lock();
    bool __ret = (gpu::memcmp(&__a->__a_value, __expected, sizeof(_Tp)) == 0);
    if (__ret)
        gpu::memcpy(&__a->__a_value, &__value, sizeof(_Tp));
    else
        gpu::memcpy(__expected, &__a->__a_value, sizeof(_Tp));
    __a->__unlock();
    return __ret;
}

template <typename _Tp>
inline __host__ __device__ bool __cxx_atomic_compare_exchange_weak(volatile __cxx_atomic_lock_impl<_Tp> *__a,
                                                                   _Tp *__expected, _Tp __value, memory_order,
                                                                   memory_order) {
    _Tp __temp;
    __a->__lock();
    __cxx_atomic_assign_volatile(__temp, __a->__a_value);
    bool __ret = (gpu::memcmp(&__temp, __expected, sizeof(_Tp)) == 0);
    if (__ret)
        __cxx_atomic_assign_volatile(__a->__a_value, __value);
    else
        __cxx_atomic_assign_volatile(*__expected, __a->__a_value);
    __a->__unlock();
    return __ret;
}
template <typename _Tp>
inline __host__ __device__ bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_lock_impl<_Tp> *__a, _Tp *__expected,
                                                                   _Tp __value, memory_order, memory_order) {
    __a->__lock();
    bool __ret = (gpu::memcmp(&__a->__a_value, __expected, sizeof(_Tp)) == 0);
    if (__ret)
        gpu::memcpy(&__a->__a_value, &__value, sizeof(_Tp));
    else
        gpu::memcpy(__expected, &__a->__a_value, sizeof(_Tp));
    __a->__unlock();
    return __ret;
}

template <typename _Tp, typename _Td>
inline __host__ __device__ _Tp __cxx_atomic_fetch_add(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Td __delta,
                                                      memory_order) {
    __a->__lock();
    _Tp __old;
    __cxx_atomic_assign_volatile(__old, __a->__a_value);
    __cxx_atomic_assign_volatile(__a->__a_value, _Tp(__old + __delta));
    __a->__unlock();
    return __old;
}
template <typename _Tp, typename _Td>
inline __host__ __device__ _Tp __cxx_atomic_fetch_add(__cxx_atomic_lock_impl<_Tp> *__a, _Td __delta, memory_order) {
    __a->__lock();
    _Tp __old = __a->__a_value;
    __a->__a_value += __delta;
    __a->__unlock();
    return __old;
}

template <typename _Tp, typename _Td>
inline __host__ __device__ _Tp *__cxx_atomic_fetch_add(volatile __cxx_atomic_lock_impl<_Tp *> *__a, ptrdiff_t __delta,
                                                       memory_order) {
    __a->__lock();
    _Tp *__old;
    __cxx_atomic_assign_volatile(__old, __a->__a_value);
    __cxx_atomic_assign_volatile(__a->__a_value, __old + __delta);
    __a->__unlock();
    return __old;
}
template <typename _Tp, typename _Td>
inline __host__ __device__ _Tp *__cxx_atomic_fetch_add(__cxx_atomic_lock_impl<_Tp *> *__a, ptrdiff_t __delta,
                                                       memory_order) {
    __a->__lock();
    _Tp *__old = __a->__a_value;
    __a->__a_value += __delta;
    __a->__unlock();
    return __old;
}

template <typename _Tp, typename _Td>
inline __host__ __device__ _Tp __cxx_atomic_fetch_sub(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Td __delta,
                                                      memory_order) {
    __a->__lock();
    _Tp __old;
    __cxx_atomic_assign_volatile(__old, __a->__a_value);
    __cxx_atomic_assign_volatile(__a->__a_value, _Tp(__old - __delta));
    __a->__unlock();
    return __old;
}
template <typename _Tp, typename _Td>
inline __host__ __device__ _Tp __cxx_atomic_fetch_sub(__cxx_atomic_lock_impl<_Tp> *__a, _Td __delta, memory_order) {
    __a->__lock();
    _Tp __old = __a->__a_value;
    __a->__a_value -= __delta;
    __a->__unlock();
    return __old;
}

template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_and(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Tp __pattern,
                                                      memory_order) {
    __a->__lock();
    _Tp __old;
    __cxx_atomic_assign_volatile(__old, __a->__a_value);
    __cxx_atomic_assign_volatile(__a->__a_value, _Tp(__old & __pattern));
    __a->__unlock();
    return __old;
}
template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_and(__cxx_atomic_lock_impl<_Tp> *__a, _Tp __pattern, memory_order) {
    __a->__lock();
    _Tp __old = __a->__a_value;
    __a->__a_value &= __pattern;
    __a->__unlock();
    return __old;
}

template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_or(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Tp __pattern,
                                                     memory_order) {
    __a->__lock();
    _Tp __old;
    __cxx_atomic_assign_volatile(__old, __a->__a_value);
    __cxx_atomic_assign_volatile(__a->__a_value, _Tp(__old | __pattern));
    __a->__unlock();
    return __old;
}
template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_or(__cxx_atomic_lock_impl<_Tp> *__a, _Tp __pattern, memory_order) {
    __a->__lock();
    _Tp __old = __a->__a_value;
    __a->__a_value |= __pattern;
    __a->__unlock();
    return __old;
}

template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_xor(volatile __cxx_atomic_lock_impl<_Tp> *__a, _Tp __pattern,
                                                      memory_order) {
    __a->__lock();
    _Tp __old;
    __cxx_atomic_assign_volatile(__old, __a->__a_value);
    __cxx_atomic_assign_volatile(__a->__a_value, _Tp(__old ^ __pattern));
    __a->__unlock();
    return __old;
}
template <typename _Tp>
inline __host__ __device__ _Tp __cxx_atomic_fetch_xor(__cxx_atomic_lock_impl<_Tp> *__a, _Tp __pattern, memory_order) {
    __a->__lock();
    _Tp __old = __a->__a_value;
    __a->__a_value ^= __pattern;
    __a->__unlock();
    return __old;
}

// This is how we'll decide which of the two overloads of the __cxx_atomic_*** functions we'll use.
// __cxx_atomic_impl will conditionally inherit from base or lock, depending on whether the compilation target supports
// native atomics for _Tp. So, calling __cxx_atomic_*** with a pointer to an __cxx_atomic_impl will call the correct
// overload.
template <typename _Tp,
          typename _Base = typename std::conditional_t<__libcpp_is_always_lock_free<_Tp>::__value,
                                                       __cxx_atomic_base_impl<_Tp>, __cxx_atomic_lock_impl<_Tp>>>
struct __cxx_atomic_impl : public _Base {
    static_assert(std::is_trivially_copyable_v<_Tp>, "std::atomic<T> requires that 'T' be a trivially copyable type");

    inline __host__ __device__ __cxx_atomic_impl() noexcept = default;
    inline constexpr __host__ __device__ explicit __cxx_atomic_impl(_Tp __value) noexcept : _Base(__value) {}
};

#if defined(__linux__) || (defined(_AIX) && !defined(__64BIT__))
using __cxx_contention_t = int32_t;
#else
using __cxx_contention_t = int64_t;
#endif // __linux__ || (_AIX && !__64BIT__)

using __cxx_atomic_contention_t = __cxx_atomic_impl<__cxx_contention_t>;

// general atomic<T>
// For bool and non-integral types, we don't provide the arithmetic operations (e.g. fetch_add and operator++).
// We provide those using a partial template specialization of __atomic_base<_Tp, true> further down.
template <class _Tp, bool = std::is_integral_v<_Tp> && !std::is_same_v<_Tp, bool>>
struct __atomic_base // <_Tp, false>
{
    mutable __cxx_atomic_impl<_Tp> __a_;

#if defined(__cpp_lib_atomic_is_always_lock_free)
    static constexpr bool is_always_lock_free = __libcpp_is_always_lock_free<__cxx_atomic_impl<_Tp>>::__value;
#endif
    inline __host__ __device__ bool is_lock_free() const volatile noexcept {
        // TODO: If/when __hip_atomic_is_lock_free gets implemented, uncomment this and remove the following line
        // return __cxx_atomic_is_lock_free(sizeof(_Tp));
        return gpu::internal::is_hip_native_v<_Tp>;
    }
    inline __host__ __device__ bool is_lock_free() const noexcept {
        return static_cast<__atomic_base const volatile *>(this)->is_lock_free();
    }
    inline __host__ __device__ void store(_Tp __d, memory_order __m = memory_order_seq_cst) volatile noexcept {
        gpu::internal::__cxx_atomic_store(&__a_, __d, __m);
    }
    inline __host__ __device__ void store(_Tp __d, memory_order __m = memory_order_seq_cst) noexcept {
        gpu::internal::__cxx_atomic_store(&__a_, __d, __m);
    }
    inline __host__ __device__ _Tp load(memory_order __m = memory_order_seq_cst) const volatile noexcept {
        return gpu::internal::__cxx_atomic_load(&__a_, __m);
    }
    inline __host__ __device__ _Tp load(memory_order __m = memory_order_seq_cst) const noexcept {
        return gpu::internal::__cxx_atomic_load(&__a_, __m);
    }
    inline __host__ __device__ operator _Tp() const volatile noexcept { return load(); }
    inline __host__ __device__ operator _Tp() const noexcept { return load(); }
    inline __host__ __device__ _Tp exchange(_Tp __d, memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_exchange(&__a_, __d, __m);
    }
    inline __host__ __device__ _Tp exchange(_Tp __d, memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_exchange(&__a_, __d, __m);
    }
    inline __host__ __device__ bool compare_exchange_weak(_Tp &__e, _Tp __d, memory_order __s,
                                                          memory_order __f) volatile noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(&__a_, &__e, __d, __s, __f);
    }
    inline __host__ __device__ bool compare_exchange_weak(_Tp &__e, _Tp __d, memory_order __s,
                                                          memory_order __f) noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(&__a_, &__e, __d, __s, __f);
    }
    inline __host__ __device__ bool compare_exchange_strong(_Tp &__e, _Tp __d, memory_order __s,
                                                            memory_order __f) volatile noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(&__a_, &__e, __d, __s, __f);
    }
    inline __host__ __device__ bool compare_exchange_strong(_Tp &__e, _Tp __d, memory_order __s,
                                                            memory_order __f) noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(&__a_, &__e, __d, __s, __f);
    }
    inline __host__ __device__ bool compare_exchange_weak(_Tp &__e, _Tp __d,
                                                          memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(&__a_, &__e, __d, __m, __m);
    }
    inline __host__ __device__ bool compare_exchange_weak(_Tp &__e, _Tp __d,
                                                          memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(&__a_, &__e, __d, __m, __m);
    }
    inline __host__ __device__ bool compare_exchange_strong(_Tp &__e, _Tp __d,
                                                            memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(&__a_, &__e, __d, __m, __m);
    }
    inline __host__ __device__ bool compare_exchange_strong(_Tp &__e, _Tp __d,
                                                            memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(&__a_, &__e, __d, __m, __m);
    }

    inline __host__ __device__ __atomic_base() noexcept = default;

    inline constexpr __host__ __device__ __atomic_base(_Tp __d) noexcept : __a_(__d) {}

    __atomic_base(const __atomic_base &) = delete;
};

#if defined(__cpp_lib_atomic_is_always_lock_free)
template <class _Tp, bool __b>
constexpr bool __atomic_base<_Tp, __b>::is_always_lock_free;
#endif

// atomic<Integral>
// Use a partial template specialization to provide arithmetic operations like fetch_add for integral types.
// However, we still inherit from __atomic_base<_Tp, false> so we don't have to re-implement the others
template <class _Tp>
struct __atomic_base<_Tp, true> : public __atomic_base<_Tp, false> {
    typedef __atomic_base<_Tp, false> __base;

    inline __host__ __device__ __atomic_base() noexcept = default;

    inline constexpr __host__ __device__ __atomic_base(_Tp __d) noexcept : __base(__d) {}

    inline __host__ __device__ _Tp fetch_add(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_fetch_add(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_add(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_fetch_add(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_sub(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_fetch_sub(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_sub(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_fetch_sub(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_and(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_fetch_and(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_and(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_fetch_and(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_or(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_fetch_or(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_or(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_fetch_or(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_xor(_Tp __op, memory_order __m = memory_order_seq_cst) volatile noexcept {
        return gpu::internal::__cxx_atomic_fetch_xor(&this->__a_, __op, __m);
    }
    inline __host__ __device__ _Tp fetch_xor(_Tp __op, memory_order __m = memory_order_seq_cst) noexcept {
        return gpu::internal::__cxx_atomic_fetch_xor(&this->__a_, __op, __m);
    }

    inline __host__ __device__ _Tp operator++(int) volatile noexcept        { return fetch_add(_Tp(1)); }
    inline __host__ __device__ _Tp operator++(int) noexcept                 { return fetch_add(_Tp(1)); }
    inline __host__ __device__ _Tp operator--(int) volatile noexcept        { return fetch_sub(_Tp(1)); }
    inline __host__ __device__ _Tp operator--(int) noexcept                 { return fetch_sub(_Tp(1)); }
    inline __host__ __device__ _Tp operator++() volatile noexcept           { return fetch_add(_Tp(1)) + _Tp(1); }
    inline __host__ __device__ _Tp operator++() noexcept                    { return fetch_add(_Tp(1)) + _Tp(1); }
    inline __host__ __device__ _Tp operator--() volatile noexcept           { return fetch_sub(_Tp(1)) - _Tp(1); }
    inline __host__ __device__ _Tp operator--() noexcept                    { return fetch_sub(_Tp(1)) - _Tp(1); }
    inline __host__ __device__ _Tp operator+=(_Tp __op) volatile noexcept   { return fetch_add(__op) + __op; }
    inline __host__ __device__ _Tp operator+=(_Tp __op) noexcept            { return fetch_add(__op) + __op; }
    inline __host__ __device__ _Tp operator-=(_Tp __op) volatile noexcept   { return fetch_sub(__op) - __op; }
    inline __host__ __device__ _Tp operator-=(_Tp __op) noexcept            { return fetch_sub(__op) - __op; }
    inline __host__ __device__ _Tp operator&=(_Tp __op) volatile noexcept   { return fetch_and(__op) & __op; }
    inline __host__ __device__ _Tp operator&=(_Tp __op) noexcept            { return fetch_and(__op) & __op; }
    inline __host__ __device__ _Tp operator|=(_Tp __op) volatile noexcept   { return fetch_or(__op) | __op; }
    inline __host__ __device__ _Tp operator|=(_Tp __op) noexcept            { return fetch_or(__op) | __op; }
    inline __host__ __device__ _Tp operator^=(_Tp __op) volatile noexcept   { return fetch_xor(__op) ^ __op; }
    inline __host__ __device__ _Tp operator^=(_Tp __op) noexcept            { return fetch_xor(__op) ^ __op; }
};

} // namespace internal

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

#endif // __GPU_ATOMIC_H__
