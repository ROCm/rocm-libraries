// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

namespace roc::hostnumerics {
// A dimension permutation from slowest to fastest. The identity [0, 1, ...]
// gives NumPy/C row-major enumeration; its reverse gives Fortran enumeration.
// An empty order selects the identity for any rank. Explicit orders must contain
// every dimension exactly once. Logical enumeration is independent of strides.
using IndexOrder = std::vector<size_t>;
}  // namespace roc::hostnumerics
