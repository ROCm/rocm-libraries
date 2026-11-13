// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief Experimental feature DEMO (spike only): a trivial API that exercises the
 *  experimental-API deadline and revision-bump checks. Not for real use.
 *
 *  Contract tests: clients/tests/src/experimental/contract_demo_gtest.cpp.
 */

#pragma once

#ifndef HIPBLASLT_EXPERIMENTAL_ENUM_BIT
#error "include <hipblaslt/hipblaslt-experimental.h> instead"
#endif

/*! \ingroup experimental_module
 *  \brief Revision of experimental feature DEMO declared in this header.
 *
 *  Incremented on every breaking change to the feature: a removed declaration, a function or
 *  type replaced by a renamed version, a retired enumerator, or a changed contract test.
 *  Additions do not increment it. tools/scripts/check_experimental_api.py enforces this.
 */
#define HIPBLASLT_EXPERIMENTAL_DEMO 1

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup experimental_module
 *  \brief Rounding direction for hipblasLtExperimentalDemoAlignSize.
 */
typedef enum
{
    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_DOWN = 0, /**< Round down to a multiple of the alignment. */
    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP   = 1, /**< Round up to a multiple of the alignment. */
} hipblasLtExperimentalDemoRounding_t;

/*! \ingroup experimental_module
 *  \brief Round \p size to a multiple of \p alignment.
 *
 *  @param[in]  size       Value to round.
 *  @param[in]  alignment  Alignment; must be a nonzero power of two.
 *  @param[in]  rounding   Rounding direction.
 *  @param[out] result     Rounded value.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS        \p result holds the rounded value.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE  \p result is null, \p alignment is not a nonzero
 *                                        power of two, \p rounding is unknown, or rounding up
 *                                        overflows size_t. \p result is not written.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtExperimentalDemoAlignSize(size_t                              size,
                                                   size_t                              alignment,
                                                   hipblasLtExperimentalDemoRounding_t rounding,
                                                   size_t*                             result);

#ifdef __cplusplus
}
#endif
