// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnNormFwdPhase.h
 * @brief Normalization forward phase enumeration for hipDNN backend operations
 *
 * Defines the forward phase (inference or training) used when setting the
 * HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT attribute on normalization
 * operation descriptors.
 */

#pragma once

/**
 * @enum hipdnnNormFwdPhase_t
 * @brief Forward phase for normalization operations
 *
 * Determines whether the normalization runs in inference mode (no saved
 * statistics) or training mode (saves inverse-RMS for backward pass).
 */
typedef enum
{
    HIPDNN_NORM_FWD_INFERENCE = 1, ///< Inference mode (no saved statistics)
    HIPDNN_NORM_FWD_TRAINING = 2 ///< Training mode (saves inv_rms for backward)
} hipdnnNormFwdPhase_t;
