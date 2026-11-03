// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Engine-internal dtype lookups that raise. NOT a public API.
 *
 * rocke/dtypes.h is the C contract: rocke_dtype_info returns NULL for an unknown
 * name. Engine code that mirrors a Python `raise` calls the twin below instead,
 * so the refusal text has one home in the C++ engine as it has in Python
 * (core/dtypes.py::dtype_info).
 */
#ifndef ROCKE_DTYPES_INTERNAL_H
#define ROCKE_DTYPES_INTERNAL_H

#include "rocke/dtypes.h"

/* The encoding of `name` (any accepted spelling), as rocke_dtype_info returns
 * it. Throws ckc::ValueError "unknown dtype '<name>'", naming the spelling the
 * caller passed, when no encoding has that name. */
const rocke_dtype_info_t* rocke_require_dtype_info(const char* name);

#endif /* ROCKE_DTYPES_INTERNAL_H */
