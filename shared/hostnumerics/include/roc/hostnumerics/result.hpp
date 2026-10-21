// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/compat/expected.hpp>

namespace roc::hostnumerics::detail {
template <typename T, typename E>
using Expected = compat::expected<T, E>;

template <typename E>
constexpr auto Unexpected(E&& error) noexcept(
    std::is_nothrow_constructible_v<std::decay_t<E>, E&&>) {
    return compat::unexpected<std::decay_t<E>>(std::forward<E>(error));
}
}  // namespace roc::hostnumerics::detail
