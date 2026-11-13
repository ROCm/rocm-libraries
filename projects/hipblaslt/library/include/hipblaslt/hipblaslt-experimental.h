// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief hipblaslt-experimental.h declares the hipBLASLt experimental API.
 *
 *  Everything declared here ships in every hipBLASLt build but is NOT covered
 *  by the stability guarantees of hipblaslt.h or hipblaslt-ext.hpp. Any
 *  experimental feature may change or be removed in any ROCm release.
 *
 *  This header is not included by hipblaslt.h, and it declares nothing unless
 *  the consumer defines ``ROCM_ENABLE_EXPERIMENTAL_APIS`` to 1 before including
 *  it. The switch is ROCm-wide: it enables the experimental headers of every
 *  ROCm library the consumer includes.
 *
 *  Each feature defines ``HIPBLASLT_EXPERIMENTAL_<FEATURE>`` to its revision.
 *  Check that it has the revision your code was written for, and check the
 *  loaded library with hipblasLtExperimentalGetFeatureRevision(); on a
 *  mismatch, either fall back to the stable API or fail:
 *
 *  \code{.c}
 *  #define ROCM_ENABLE_EXPERIMENTAL_APIS 1
 *  #include <hipblaslt/hipblaslt-experimental.h>
 *
 *  #if defined(HIPBLASLT_EXPERIMENTAL_DEMO) && HIPBLASLT_EXPERIMENTAL_DEMO == 1
 *      if(hipblasLtExperimentalGetFeatureRevision("DEMO") == 1) { ... }
 *      else { ... fallback ... }
 *  #else
 *      ... fallback ...
 *  #endif
 *  \endcode
 *
 *  \defgroup experimental_module Experimental API
 *  Unstable APIs that are shipped for evaluation and may change or be removed
 *  in any release.
 */

#pragma once
#ifndef _HIPBLASLT_EXPERIMENTAL_H_
#define _HIPBLASLT_EXPERIMENTAL_H_

// Without the switch (or with it set to 0), this header declares nothing, so consumers can
// include it unconditionally and their feature checks take the fallback path.
#if defined(ROCM_ENABLE_EXPERIMENTAL_APIS) && ROCM_ENABLE_EXPERIMENTAL_APIS

#include "hipblaslt/hipblaslt.h"

/*! \ingroup experimental_module
 *  \brief Marker bit for experimental values of stable enum types.
 *
 *  Experimental enumerators for stable enums (for example, new epilogues or
 *  descriptor attributes) are declared in the feature headers as
 *  ``(EnumType)(HIPBLASLT_EXPERIMENTAL_ENUM_BIT | n)``, so they can be passed to
 *  the existing stable entry points (such as hipblasLtMatmulDescSetAttribute)
 *  without adding enumerators to hipblaslt.h. Stable enums must never use this
 *  bit. When an experimental value graduates it receives a new value in the
 *  stable enum.
 */
#define HIPBLASLT_EXPERIMENTAL_ENUM_BIT 0x40000000

/*! \ingroup experimental_module
 *  \brief Highest experimental enumerator value ever used, ignoring
 *  HIPBLASLT_EXPERIMENTAL_ENUM_BIT.
 *
 *  Experimental enumerator values are never reused, so a binary built against
 *  a retired value cannot silently get a different meaning. A new experimental
 *  enumerator, in any experimental enum, takes a value above this mark, and
 *  the mark is raised to match. It never goes down.
 *  tools/scripts/check_experimental_api.py enforces this.
 */
#define HIPBLASLT_EXPERIMENTAL_ENUM_HIGH_WATER 1

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup experimental_module
 *  \brief Returns the revision of an experimental feature in the loaded
 *  hipBLASLt library.
 *
 *  \details
 *  The shared-library version does not track experimental changes. Before
 *  calling a feature's functions, check that this returns the revision your
 *  code was compiled against (``HIPBLASLT_EXPERIMENTAL_<FEATURE>``).
 *
 *  This function never changes and is never removed.
 *
 *  @param[in] feature The feature name without the prefix, for example "DEMO".
 *
 *  @retval int The feature's revision, or 0 if the library does not have the
 *  feature or \p feature is null.
 */
HIPBLASLT_EXPORT
int hipblasLtExperimentalGetFeatureRevision(const char* feature);

#ifdef __cplusplus
}
#endif

// One header per experimental feature, generated from library/experimental/features.cmake.
#include "hipblaslt/hipblaslt-experimental-features.h"

#endif // ROCM_ENABLE_EXPERIMENTAL_APIS

#endif // _HIPBLASLT_EXPERIMENTAL_H_
