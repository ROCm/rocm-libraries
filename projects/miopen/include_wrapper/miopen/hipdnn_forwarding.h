// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <miopen/miopen.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup handle
 *
 *  @{
 */

#ifdef MIOPEN_BETA_API
/*! @brief Gets the status of this thread's last call if it was forwarded to hipDNN and failed
 *
 * A call that MIOpen hands to hipDNN can fail with the same status MIOpen itself would
 * return. If the most recent MIOpen call on the calling thread was forwarded to hipDNN and
 * failed, @p forwardedStatus receives that call's status; otherwise it receives
 * miopenStatusSuccess. Calling this function or miopenGetErrorString does not change the
 * answer.
 *
 * Available only in MIOpen builds configured with MIOPEN_ENABLE_HIPDNN_WRAPPER=ON.
 *
 * @param forwardedStatus  Status of the failed forwarded call, or miopenStatusSuccess (output)
 * @return                 miopenStatusSuccess, or miopenStatusBadParm if forwardedStatus is NULL
 */
MIOPEN_EXPORT miopenStatus_t miopenGetLastForwardedError(miopenStatus_t* forwardedStatus);
#endif

/** @} */

#ifdef __cplusplus
}
#endif
