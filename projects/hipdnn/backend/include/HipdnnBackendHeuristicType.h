// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnBackendHeuristicType.h
 * @brief Heuristic mode types for engine selection
 *
 * This file defines the heuristic modes used when querying available
 * engines for an operation graph. Different modes provide different
 * trade-offs between search time and result quality.
 */

#pragma once

/**
 * @enum hipdnnBackendHeurMode_t
 * @brief Heuristic modes for engine discovery
 *
 * When creating an engine heuristic descriptor, specify a mode to
 * control how engines are discovered and ranked. The mode affects
 * both the search algorithm and the quality of performance estimates.
 *
 * @see HIPDNN_ATTR_ENGINEHEUR_MODE
 * @see hipdnnBackendSetAttribute()
 */
typedef enum
{
    /**
     * @brief Fallback mode - functional support only
     *
     * Returns engines that are functionally correct for the given
     * operation graph. No performance tuning or optimization is
     * performed. This is the most reliable mode for finding a
     * working configuration.
     *
     * Use this mode when:
     * - You need guaranteed functional correctness
     * - Performance is not critical
     * - You want the fastest engine discovery
     */
    HIPDNN_HEUR_MODE_FALLBACK = 0,

} hipdnnBackendHeurMode_t;
