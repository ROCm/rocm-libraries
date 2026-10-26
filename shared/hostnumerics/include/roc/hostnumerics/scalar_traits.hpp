// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <concepts>
#include <type_traits>

namespace roc::hostnumerics::detail {
template <typename T>
struct RuntimeIsComplex : std::false_type {};

// std::complex is specified only for floating component types.
template <std::floating_point T>
struct RuntimeIsComplex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool RuntimeIsComplexV = RuntimeIsComplex<T>::value;

template <typename>
inline constexpr bool AlwaysFalseV = false;

}  // namespace roc::hostnumerics::detail
