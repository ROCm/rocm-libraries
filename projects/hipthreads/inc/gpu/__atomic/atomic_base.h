#ifndef __GPU___ATOMIC_ATOMIC_BASE_H__
#define __GPU___ATOMIC_ATOMIC_BASE_H__

#include <type_traits>

#include "gpu/__config"

#include "gpu/__atomic/cxx_atomic_impl.h"
#include "gpu/__atomic/is_always_lock_free.h"
#include "gpu/__atomic/memory_order.h"
#include "gpu/__memory/addressof.h"

namespace gpu::internal {

//====================================================================================================================//
//      Adapted from libc++ std::atomic
//====================================================================================================================//

// general atomic<T>
// For bool and non-integral types, we don't provide the arithmetic operations (e.g. fetch_add and operator++).
// We provide those using a partial template specialization of __atomic_base<_Tp, true> further down.
template <class _Tp, bool = std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value>
struct __atomic_base // false
{
    mutable __cxx_atomic_impl<_Tp> __a_;

#if defined(__cpp_lib_atomic_is_always_lock_free)
    static _LIBGPU_CONSTEXPR bool is_always_lock_free = __libcpp_is_always_lock_free<__cxx_atomic_impl<_Tp>>::__value;
#endif
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    bool is_lock_free() const volatile _NOEXCEPT {
        // TODO: If/when __hip_atomic_is_lock_free gets implemented, uncomment this and remove the following line
        // return __cxx_atomic_is_lock_free(sizeof(_Tp));
        return gpu::internal::is_hip_native_v<_Tp>;
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    bool is_lock_free() const _NOEXCEPT {
        return static_cast<__atomic_base const volatile *>(this)->is_lock_free();
    }

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI void store(_Tp __d, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        gpu::internal::__cxx_atomic_store(gpu::addressof(__a_), __d, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI void store(_Tp __d, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        gpu::internal::__cxx_atomic_store(gpu::addressof(__a_), __d, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp load(memory_order __m = memory_order_seq_cst) const volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_load(gpu::addressof(__a_), __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp load(memory_order __m = memory_order_seq_cst) const _NOEXCEPT {
        return gpu::internal::__cxx_atomic_load(gpu::addressof(__a_), __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    operator _Tp() const volatile _NOEXCEPT { return load(); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    operator _Tp() const _NOEXCEPT { return load(); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp exchange(_Tp __d, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_exchange(gpu::addressof(__a_), __d, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp exchange(_Tp __d, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_exchange(gpu::addressof(__a_), __d, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_weak(_Tp &__e, _Tp __d, memory_order __s,
                                                     memory_order __f) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(gpu::addressof(__a_), gpu::addressof(__e), __d, __s, __f);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_weak(_Tp &__e, _Tp __d, memory_order __s, memory_order __f) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(gpu::addressof(__a_), gpu::addressof(__e), __d, __s, __f);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_strong(_Tp &__e, _Tp __d, memory_order __s,
                                                       memory_order __f) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(gpu::addressof(__a_), gpu::addressof(__e), __d, __s, __f);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_strong(_Tp &__e, _Tp __d, memory_order __s,
                                                       memory_order __f) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(gpu::addressof(__a_), gpu::addressof(__e), __d, __s, __f);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_weak(_Tp &__e, _Tp __d,
                                                     memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(gpu::addressof(__a_), gpu::addressof(__e), __d, __m, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_weak(_Tp &__e, _Tp __d,
                                                     memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_weak(gpu::addressof(__a_), gpu::addressof(__e), __d, __m, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_strong(_Tp &__e, _Tp __d,
                                                       memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(gpu::addressof(__a_), gpu::addressof(__e), __d, __m, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI bool compare_exchange_strong(_Tp &__e, _Tp __d,
                                                       memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_compare_exchange_strong(gpu::addressof(__a_), gpu::addressof(__e), __d, __m, __m);
    }

#if _LIBGPU_STD_VER >= 20
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI constexpr __atomic_base() noexcept(std::is_nothrow_default_constructible_v<_Tp>)
        : __a_(_Tp()) {}
#else
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    __atomic_base() _NOEXCEPT = default;
#endif

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR __atomic_base(_Tp __d) _NOEXCEPT : __a_(__d) {}

    __atomic_base(const __atomic_base &) = delete;
};

#if defined(__cpp_lib_atomic_is_always_lock_free)
template <class _Tp, bool __b>
_LIBGPU_CONSTEXPR bool __atomic_base<_Tp, __b>::is_always_lock_free;
#endif

// atomic<Integral>

template <class _Tp>
struct __atomic_base<_Tp, true> : public __atomic_base<_Tp, false> {
    using __base = __atomic_base<_Tp, false>;

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 __atomic_base() _NOEXCEPT = default;

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _LIBGPU_CONSTEXPR __atomic_base(_Tp __d) _NOEXCEPT : __base(__d) {}

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_add(_Tp __op, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_add(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_add(_Tp __op, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_add(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_sub(_Tp __op, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_sub(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_sub(_Tp __op, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_sub(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_and(_Tp __op, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_and(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_and(_Tp __op, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_and(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_or(_Tp __op, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_or(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_or(_Tp __op, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_or(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_xor(_Tp __op, memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_xor(gpu::addressof(this->__a_), __op, __m);
    }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI _Tp fetch_xor(_Tp __op, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
        return gpu::internal::__cxx_atomic_fetch_xor(gpu::addressof(this->__a_), __op, __m);
    }

    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator++(int) volatile _NOEXCEPT      { return fetch_add(_Tp(1)); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator++(int) _NOEXCEPT               { return fetch_add(_Tp(1)); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator--(int) volatile _NOEXCEPT      { return fetch_sub(_Tp(1)); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator--(int) _NOEXCEPT               { return fetch_sub(_Tp(1)); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator++() volatile _NOEXCEPT         { return fetch_add(_Tp(1)) + _Tp(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator++() _NOEXCEPT                  { return fetch_add(_Tp(1)) + _Tp(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator--() volatile _NOEXCEPT         { return fetch_sub(_Tp(1)) - _Tp(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator--() _NOEXCEPT                  { return fetch_sub(_Tp(1)) - _Tp(1); }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator+=(_Tp __op) volatile _NOEXCEPT { return fetch_add(__op) + __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator+=(_Tp __op) _NOEXCEPT          { return fetch_add(__op) + __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator-=(_Tp __op) volatile _NOEXCEPT { return fetch_sub(__op) - __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator-=(_Tp __op) _NOEXCEPT          { return fetch_sub(__op) - __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator&=(_Tp __op) volatile _NOEXCEPT { return fetch_and(__op) & __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator&=(_Tp __op) _NOEXCEPT          { return fetch_and(__op) & __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator|=(_Tp __op) volatile _NOEXCEPT { return fetch_or(__op) | __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator|=(_Tp __op) _NOEXCEPT          { return fetch_or(__op) | __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator^=(_Tp __op) volatile _NOEXCEPT { return fetch_xor(__op) ^ __op; }
    __host__ __device__ _LIBGPU_HIDE_FROM_ABI
    _Tp operator^=(_Tp __op) _NOEXCEPT          { return fetch_xor(__op) ^ __op; }
};

} // namespace gpu::internal

#endif // __GPU___ATOMIC_ATOMIC_BASE_H__
