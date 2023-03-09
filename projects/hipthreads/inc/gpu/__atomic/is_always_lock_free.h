#ifndef __GPU___ATOMIC_IS_ALWAYS_LOCK_FREE_H__
#define __GPU___ATOMIC_IS_ALWAYS_LOCK_FREE_H__

#include <type_traits>

namespace gpu::internal {

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

template <class _Tp>
struct __libcpp_is_always_lock_free {
    // TODO: If/when HIP gets an equivalent to __atomic_always_lock_free uncomment this and remove the following line
    // static const bool __value = __atomic_always_lock_free(sizeof(_Tp), nullptr);
    static const bool __value = gpu::internal::is_hip_native_v<_Tp>;
};

} // namespace gpu::internal

#endif // __GPU___ATOMIC_IS_ALWAYS_LOCK_FREE_H__
