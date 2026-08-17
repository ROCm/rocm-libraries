// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCSOLVER_ASAN_HELPERS_HPP
#define ROCSOLVER_ASAN_HELPERS_HPP

#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
inline constexpr bool rocsolver_enable_asan = true;
/// Selects asan_val when building with AddressSanitizer, normal_val otherwise.
/// Works in preprocessor #define, __launch_bounds__, and constexpr initializers.
#define ROCSOLVER_ASAN_VALUE(asan_val, normal_val) (asan_val)
#else
inline constexpr bool rocsolver_enable_asan = false;
#define ROCSOLVER_ASAN_VALUE(asan_val, normal_val) (normal_val)
#endif

#endif
