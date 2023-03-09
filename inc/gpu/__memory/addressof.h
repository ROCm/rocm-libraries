#ifndef __GPU___ATOMIC___MEMORY_ADDRESSOF_H__
#define __GPU___ATOMIC___MEMORY_ADDRESSOF_H__

#include "hip/hip_runtime_api.h"
#include <type_traits>

namespace gpu {

template <class _Tp>
__host__ __device__ inline constexpr _Tp *addressof(_Tp &__x) noexcept {
    return __builtin_addressof(__x);
}

template <class _Tp>
__host__ __device__ _Tp *addressof(const _Tp &&) noexcept = delete;

} // namespace gpu

#endif // __GPU___ATOMIC___MEMORY_ADDRESSOF_H__
