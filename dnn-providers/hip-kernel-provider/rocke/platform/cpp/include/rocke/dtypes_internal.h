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

#include <string>

#include "rocke/dtypes.h"

/* The encoding of `name` (any accepted spelling), as rocke_dtype_info returns
 * it. Throws ckc::ValueError "unknown dtype '<name>'", naming the spelling the
 * caller passed, when no encoding has that name. */
const rocke_dtype_info_t* rocke_require_dtype_info(const char* name);

/* The canonical name of every dtype in `category`, in table order, joined by
 * ", " (Python ", ".join(dtype_names(category))). Error texts that list the
 * dtypes an operation accepts build the list from here. */
std::string rocke_dtype_names(rocke_dtype_category_t category);

#endif /* ROCKE_DTYPES_INTERNAL_H */
