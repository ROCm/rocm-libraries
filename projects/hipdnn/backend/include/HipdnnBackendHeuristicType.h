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
 * control how engines are ranked. Every mode returns the same applicable
 * engines; the mode selects the heuristic policy list that orders them,
 * unless HIPDNN_HEUR_POLICY_ORDER or HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT
 * overrides that list.
 * @see HIPDNN_ATTR_ENGINEHEUR_MODE
 * @see hipdnnBackendSetAttribute()
 */
typedef enum
{
    /**
     * @brief Fallback mode - model-free ranking
     *
     * Ranks the applicable engines without a trained model: policy list
     * [SelectionHeuristic::Config, SelectionHeuristic::StaticOrdering].
     * Every applicable engine is returned. Use it for a deterministic order
     * that no installed model can change.
     */
    HIPDNN_HEUR_MODE_FALLBACK = 0,

    /**
     * @brief Mode A - rank engines by their engine-level (L1) prediction
     *
     * Policy list [SelectionHeuristic::Config, SelectionHeuristic::ModeA,
     * SelectionHeuristic::StaticOrdering]. ModeA ranks the applicable engines
     * by their L1 prediction in the ranking metric; engines without one follow
     * in static order, and when no engine has one ModeA declines and
     * StaticOrdering decides. Returns the same engines as FALLBACK. Added in
     * hipdnn_backend 0.5.0.
     */
    HIPDNN_HEUR_MODE_A = 1,

    /**
     * @brief Mode B - rank engines by their configuration-level (L2) prediction
     *
     * Policy list [SelectionHeuristic::Config, SelectionHeuristic::ModeB,
     * SelectionHeuristic::StaticOrdering]. ModeB prefers each engine's L2
     * prediction and falls back to its L1 prediction. Added in hipdnn_backend
     * 0.5.0.
     */
    HIPDNN_HEUR_MODE_B = 2,

    /** @brief Alias of HIPDNN_HEUR_MODE_A, as cuDNN's A aliases INSTANT. Added in hipdnn_backend 0.5.0 */
    HIPDNN_HEUR_MODE_INSTANT = HIPDNN_HEUR_MODE_A,

} hipdnnBackendHeurMode_t;
