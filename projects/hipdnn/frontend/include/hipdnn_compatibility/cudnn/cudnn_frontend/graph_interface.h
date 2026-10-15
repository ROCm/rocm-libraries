// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend/graph_interface.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file graph_interface.h
 * @brief cuDNN-shaped graph wrapper for the hipDNN compatibility shim.
 *
 * This header (via the umbrella cudnn_frontend.h) is the supported entry point
 * for the graph API. `detail/graph_wrapper.h` holds the implementation and is
 * not a supported direct include.
 */

#pragma once

#include <hipdnn_compatibility/cudnn/detail/graph_wrapper.h>
