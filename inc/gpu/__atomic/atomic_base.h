#ifndef __GPU___ATOMIC_ATOMIC_BASE_H__
#define __GPU___ATOMIC_ATOMIC_BASE_H__

#include <type_traits>

#include "gpu/__atomic/cxx_atomic_impl.h"
#include "gpu/__atomic/is_always_lock_free.h"
#include "gpu/__atomic/memory_order.h"

namespace gpu::internal {

//====================================================================================================================//
//      Adapted from libc++ std::atomic
//====================================================================================================================//

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

} // namespace gpu::internal

#endif // __GPU___ATOMIC_ATOMIC_BASE_H__
