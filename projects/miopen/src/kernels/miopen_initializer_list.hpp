// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// Minimal std::initializer_list for runtime compilation without a host
/// C++ standard library (field order fixed by the clang ABI).

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
