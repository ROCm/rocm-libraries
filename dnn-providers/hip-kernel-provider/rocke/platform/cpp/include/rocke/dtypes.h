// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Target-independent dtype naming API. Mirrors rocke.core.dtypes:
 * normalize_dtype -> rocke_normalize_dtype.
 * Recognized names do not imply IR scalar or target instruction support.
 */
#ifndef ROCKE_DTYPES_H
#define ROCKE_DTYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical integer dtype key shared by normalization and its consumers. */
#define ROCKE_DTYPE_I32 "i32"

/* ============================== dtype keys ============================== */

/* Map a dtype spelling ("f16"/"half"/"fp16" -> "fp16", ...) to its canonical
 * catalog key. Returns a pointer to a static, interned canonical string when the
 * spelling is recognised; otherwise returns the *lowercased* spelling stored in
 * `scratch` (caller-provided buffer of >= scratch_cap bytes), so unknown
 * spellings pass through Python-identically. `scratch` may be NULL only if the
 * caller guarantees a known spelling; pass a buffer to be safe.
 *
 * Mirrors core/dtypes.py::normalize_dtype (strip + lower + _DTYPE_ALIASES.get). */
const char* rocke_normalize_dtype(const char* name, char* scratch, size_t scratch_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_DTYPES_H */
