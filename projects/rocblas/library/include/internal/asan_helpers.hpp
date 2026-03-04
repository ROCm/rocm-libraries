// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCBLAS_ASAN_HELPERS_HPP
#define ROCBLAS_ASAN_HELPERS_HPP

#include <iostream>

#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
inline constexpr bool rocblas_enable_asan = true;
#else
inline constexpr bool rocblas_enable_asan = false;
#endif

namespace rocblas {

// Value-level analog of std::conditional_t<B, T, F>
template <bool B, auto IfTrue, auto IfFalse>
inline constexpr auto conditional_v = B ? IfTrue : IfFalse;

/// One-time startup warning when ASAN kernel-parameter reductions are active.
/// @param program_name  e.g. "rocblas-bench" or "rocblas-test"
inline void print_asan_kernel_warning(const char* program_name = "rocblas")
{
    if constexpr(rocblas_enable_asan)
    {
        std::cout << program_name
                  << " WARNING: AddressSanitizer build active; some kernel launch "
                     "configurations are reduced for stability and may not match production "
                     "performance."
                  << std::endl;
    }
}

} // namespace rocblas

#endif
