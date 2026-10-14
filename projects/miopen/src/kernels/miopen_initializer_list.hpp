// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// Freestanding std::initializer_list for MIOpen runtime-compiled kernels
/// (clang lowers braced-init-lists to field stores on this exact layout,
/// so field order and types are fixed by the ABI).
///
/// The real <initializer_list> is preferred whenever it is reachable; the
/// fallback below is selected only when kernels are compiled at runtime
/// (MIOPEN_HIP_RUNTIME_COMPILE) and no host C++ standard library is
/// installed (see miopen_type_traits.hpp for the rationale).

#if defined(MIOPEN_HIP_RUNTIME_COMPILE) && !__has_include(<initializer_list>)

#if !defined(__clang__)
#error "the initializer_list fallback relies on clang braced-init-list lowering"
#endif

namespace std {

template <class E>
class initializer_list
{
    const E* __begin_;
    __SIZE_TYPE__ __size_;

    constexpr initializer_list(const E* b, __SIZE_TYPE__ s) noexcept : __begin_(b), __size_(s) {}

public:
    constexpr initializer_list() noexcept : __begin_(nullptr), __size_(0) {}

    constexpr const E* begin() const noexcept { return __begin_; }
    constexpr const E* end() const noexcept { return __begin_ + __size_; }
    constexpr __SIZE_TYPE__ size() const noexcept { return __size_; }
};

} // namespace std

#else

#include <initializer_list>

#endif
