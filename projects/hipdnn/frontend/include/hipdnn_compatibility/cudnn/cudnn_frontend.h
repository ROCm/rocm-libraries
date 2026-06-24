// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file cudnn_frontend.h
 * @brief Umbrella header for the hipDNN cuDNN-compatibility shim (v9-only).
 *
 * Entry point for the shim. Mirrors the filename of NVIDIA's `cudnn_frontend.h`
 * so consumer source can be ported by swapping the include and aliasing the
 * namespace:
 *
 * @code{.cpp}
 * #include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
 * namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;
 * @endcode
 *
 * Scope: the cuDNN frontend v9 graph API only; the v0.x / v8 builder surface is
 * not reconstructed. Must remain includable standalone with no extra defines.
 */

#pragma once

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
