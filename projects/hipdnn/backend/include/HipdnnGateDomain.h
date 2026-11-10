// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnGateDomain.h
 * @brief GateDomain enumeration for hipDNN backend operations
 *
 * Defines the GateDomain used when setting the
 * HIPDNN_ATTR_OPERATION_GDN_GATE_DOMAIN_EXT attribute on Gdn descriptors.
 */

#pragma once

/**
 * @enum hipdnnGateDomain_t
 * @brief GateDomain for backend Gdn operations
 */
typedef enum
{
    HIPDNN_GATE_DOMAIN_LOG_EXT = 0, ///< g holds ln(alpha)
    HIPDNN_GATE_DOMAIN_LINEAR_EXT = 1 ///< g holds alpha in (0, 1]
} hipdnnGateDomain_t;
