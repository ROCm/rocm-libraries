// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief hipblaslt-experimental.h declares the hipBLASLt experimental API.
 *
 *  Everything declared here ships in every hipBLASLt build but is NOT covered
 *  by the stability guarantees of hipblaslt.h or hipblaslt-ext.hpp. Any
 *  declaration in this file may change or be removed in any release, including
 *  minor and patch releases.
 *
 *  This header is not included by hipblaslt.h. To use it, define
 *  ``HIPBLASLT_ENABLE_EXPERIMENTAL_API`` to the experimental API version you
 *  have reviewed, before including this header:
 *
 *  \code{.c}
 *  #define HIPBLASLT_ENABLE_EXPERIMENTAL_API 1
 *  #include <hipblaslt/hipblaslt-experimental.h>
 *  \endcode
 *
 *  Compilation fails if the value does not match
 *  ``HIPBLASLT_EXPERIMENTAL_API_VERSION``, so any change to the experimental
 *  surface forces every consumer to re-review and opt in again.
 *
 *  \defgroup experimental_module Experimental API
 *  Unstable APIs that are shipped for evaluation and may change or be removed
 *  in any release.
 */

#pragma once
#ifndef _HIPBLASLT_EXPERIMENTAL_H_
#define _HIPBLASLT_EXPERIMENTAL_H_

/*! \ingroup experimental_module
 *  \brief Version of the experimental API surface declared in this header.
 *
 *  Incremented on every addition, change, graduation, or removal of an
 *  experimental declaration. Unrelated to the hipBLASLt library version.
 */
#define HIPBLASLT_EXPERIMENTAL_API_VERSION 1

#ifndef HIPBLASLT_ENABLE_EXPERIMENTAL_API
#error "hipblaslt-experimental.h: define HIPBLASLT_ENABLE_EXPERIMENTAL_API to the experimental API version you have reviewed (see HIPBLASLT_EXPERIMENTAL_API_VERSION). These APIs may change or be removed in any release."
#elif HIPBLASLT_ENABLE_EXPERIMENTAL_API != HIPBLASLT_EXPERIMENTAL_API_VERSION
#error "hipblaslt-experimental.h: HIPBLASLT_ENABLE_EXPERIMENTAL_API does not match HIPBLASLT_EXPERIMENTAL_API_VERSION. The experimental API has changed; review the hipBLASLt CHANGELOG and update your code and the define."
#endif

// Checked before including hipblaslt.h so a missing or stale opt-in fails fast.
#include "hipblaslt/hipblaslt.h"

/*! \ingroup experimental_module
 *  \brief Marker bit for experimental values of stable enum types.
 *
 *  Experimental enumerators for stable enums (for example, new epilogues or
 *  descriptor attributes) are declared in this header as
 *  ``(EnumType)(HIPBLASLT_EXPERIMENTAL_ENUM_BIT | n)``, so they can be passed to
 *  the existing stable entry points (such as hipblasLtMatmulDescSetAttribute)
 *  without adding enumerators to hipblaslt.h. Stable enums must never use this
 *  bit. When an experimental value graduates it receives a new value in the
 *  stable enum.
 */
#define HIPBLASLT_EXPERIMENTAL_ENUM_BIT 0x40000000

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup experimental_module
 *  \brief Returns the experimental API version implemented by the loaded
 *  hipBLASLt library.
 *
 *  \details
 *  The shared-library version does not protect the experimental ABI. Callers
 *  should check at runtime that this matches the
 *  ``HIPBLASLT_EXPERIMENTAL_API_VERSION`` they compiled against before calling
 *  any other experimental function.
 *
 *  @retval int The experimental API version of the loaded library.
 */
HIPBLASLT_EXPORT
int hipblasLtExperimentalGetApiVersion(void);

#ifdef __cplusplus
}
#endif

#endif // _HIPBLASLT_EXPERIMENTAL_H_
