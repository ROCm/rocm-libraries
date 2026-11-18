// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

/**
 * @file NumericalNote.h
 * @brief Numerical note values for engine metadata.
 *
 * Numerical notes describe the numerical properties of an engine's results. Engine
 * plugins report these values through EngineDetails metadata, and backend/frontend
 * APIs expose them to callers.
 */

// This file's definitions are duplicated by HipdnnBackendNumericalNote.h.
// Please ensure any updates are synced between the two files.

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

#ifndef HIPDNN_NUMERICAL_NOTE_TYPES_DEFINED
#define HIPDNN_NUMERICAL_NOTE_TYPES_DEFINED

/**
 * @brief Numerical notes for hipDNN engines.
 *
 * Values match the cuDNN NumericalNote_t enumerators.
 *
 * The transport type is intentionally an int32_t so newer plugins can pass
 * future note values through older backends/frontends without truncation.
 */
typedef int32_t hipdnnBackendNumericalNote_t;

enum
{
    HIPDNN_NUMERICAL_NOTE_TENSOR_CORE
    = 0, ///< Uses tensor-core (matrix) instructions; performance only.
    HIPDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS
    = 1, ///< Converts inputs to a lower precision before computing.
    HIPDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION
    = 2, ///< Accumulates reductions in a lower precision than the output.
    HIPDNN_NUMERICAL_NOTE_FFT = 3, ///< Computes with a Fourier-transform algorithm.
    HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC
    = 4, ///< Repeated runs on identical inputs may produce different results.
    HIPDNN_NUMERICAL_NOTE_WINOGRAD = 5, ///< Computes with a Winograd algorithm.
    HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4
    = 6, ///< Computes with a Winograd algorithm on 4x4 tiles.
    HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6
    = 7, ///< Computes with a Winograd algorithm on 6x6 tiles.
    HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13
    = 8, ///< Computes with a Winograd algorithm on 13x13 tiles.
    HIPDNN_NUMERICAL_NOTE_STRICT_NAN_PROP = 9, ///< Propagates NaN inputs strictly to the outputs.
    HIPDNN_NUMERICAL_NOTE_TYPE_COUNT = 10 ///< Number of cuDNN-defined notes; not a note.
};

#endif // HIPDNN_NUMERICAL_NOTE_TYPES_DEFINED

#ifdef __cplusplus
}
#endif
// NOLINTEND
