// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_COMMON_UTILS_FLOAT8_HPP
#define GUARD_COMMON_UTILS_FLOAT8_HPP

// HOST-SIDE entry point for the FP8/BF8 type (miopen_f8::hip_f8<T>).
//
// The canonical implementation lives at src/kernels/hip_float8.hpp and is
// registered with HIPRTC for runtime GPU kernel compilation. That file must
// remain in src/kernels/ because HIPRTC resolves includes by flat filename.
//
// This wrapper centralizes the relative-path include so that driver, test,
// and miopen_utils code can use the FP8 types without each hard-coding a path
// into MIOpen internals. Only this file should contain the relative path.
// It is a documented, whitelisted layering exception (RTC flat-include
// constraint). This header is intentionally NOT installed (see
// common_utils/CMakeLists.txt) because src/kernels/ is not part of the
// installed tree.
//
// WARNING: Do not duplicate the FP8 conversion math. The implementation in
// src/kernels/hip_float8.hpp and hip_f8_impl.hpp is non-trivial and must
// remain the single source of truth for both host and device code.

#include "../../../src/kernels/hip_float8.hpp"

using float8_fnuz  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8_fnuz = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

#endif // GUARD_COMMON_UTILS_FLOAT8_HPP
