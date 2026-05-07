// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BehaviorNote.h
 * @brief Behavior note values for engine metadata.
 *
 * Behavior notes describe advisory, non-numerical execution behavior for an
 * engine. Engine plugins report these values through EngineDetails metadata,
 * and backend/frontend APIs expose them to callers.
 */

#pragma once

// This file's definitions are duplicated by HipdnnBackendBehaviorNote.h.
// Please ensure any updates are synced between the two files.

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

#ifndef HIPDNN_BEHAVIOR_NOTE_TYPES_DEFINED
#define HIPDNN_BEHAVIOR_NOTE_TYPES_DEFINED

/**
 * @enum hipdnnBackendBehaviorNote_t
 * @brief Advisory behavior notes for hipDNN engines.
 */
typedef enum
{
    HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION = 0,
    HIPDNN_BEHAVIOR_NOTE_REQUIRES_LAYOUT_TRANSFORM = 1,
    HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE = 2,
    HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY = 3,
    HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION = 4,
    HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT = 5
} hipdnnBackendBehaviorNote_t;

#endif // HIPDNN_BEHAVIOR_NOTE_TYPES_DEFINED

#ifdef __cplusplus
}
#endif
// NOLINTEND
