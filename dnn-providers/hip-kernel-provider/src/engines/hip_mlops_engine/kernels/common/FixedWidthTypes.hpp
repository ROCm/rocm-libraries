// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Runtime-only ROCm installs (notably on Windows) do not ship the host STL headers hipRTC would
// otherwise find, so fall back to the compiler's predefined types when <cstdint> is unavailable.
#if defined(__has_include)
#if __has_include(<cstdint>)
#include <cstdint>
#define HIP_PLUGIN_HAS_CSTDINT 1
#endif
#endif

#ifndef HIP_PLUGIN_HAS_CSTDINT
using int8_t = __INT8_TYPE__;
using int16_t = __INT16_TYPE__;
using int32_t = __INT32_TYPE__;
using int64_t = __INT64_TYPE__;
using uint8_t = __UINT8_TYPE__;
using uint16_t = __UINT16_TYPE__;
using uint32_t = __UINT32_TYPE__;
using uint64_t = __UINT64_TYPE__;
#endif
