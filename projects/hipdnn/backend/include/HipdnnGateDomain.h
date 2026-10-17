// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnGateDomain.h
 * @brief GateDomain enumeration for hipDNN backend operations
 *
 * Defines the GateDomain used when setting the HIPDNN_ATTR_OPERATION_GDN_GATE_DOMAIN_EXT
 * and HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT attributes on GDN and KDA descriptors.
 */

#pragma once

/**
 * @enum hipdnnGateDomain_t
 * @brief How the g tensor of a GDN or KDA operation encodes the decay alpha
 */
typedef enum
{
    HIPDNN_GATE_DOMAIN_LOG_EXT = 0, ///< g holds ln(alpha), the log of the per-token decay
    HIPDNN_GATE_DOMAIN_LINEAR_EXT = 1 ///< g holds alpha in (0, 1] directly
} hipdnnGateDomain_t;
