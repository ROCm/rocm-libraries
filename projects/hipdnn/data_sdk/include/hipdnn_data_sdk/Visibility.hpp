// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file Visibility.hpp
 * @brief Symbol visibility macros for per-shared-object isolation
 *
 * On Linux, `inline` functions with `static` local variables in header files
 * are subject to symbol deduplication across shared objects (due to ELF vague
 * linkage). This causes unintended sharing of state between components
 * (backend.so, application, plugin.so) when each component should have its
 * own copy.
 *
 * The HIPDNN_HIDDEN macro adds `__attribute__((visibility("hidden")))` to
 * force each shared object to have its own copy of the static variables
 * on Linux. This prevents the dynamic linker from deduplicating the symbols.
 *
 * On Windows, each DLL naturally gets its own copy of these statics, so
 * the macro is empty.
 */

#if defined(__GNUC__) || defined(__clang__)
#define HIPDNN_HIDDEN __attribute__((visibility("hidden")))
#else
#define HIPDNN_HIDDEN
#endif
