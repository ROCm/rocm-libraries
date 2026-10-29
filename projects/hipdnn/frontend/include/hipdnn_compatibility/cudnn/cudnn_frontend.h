// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend.h
 * @brief Umbrella header for the hipDNN cuDNN frontend compatibility shim (v9 graph API and
 * v0.x builders).
 *
 * The v0.x builders were introduced in hipdnn_frontend 0.5.0.
 *
 * @code{.cpp}
 * #include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
 * namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;
 * @endcode
 *
 * `cudnn_frontend_EngineConfigGenerator.h`, `cudnn_frontend_get_plan.h` and
 * `cudnn_frontend_find_plan.h` are not included, as upstream.
 */

#pragma once

#include <hipdnn_compatibility/cudnn/cudnn.h>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_ConvDesc.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Engine.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineFallbackList.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Errata.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlanCache.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Filters.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Heuristics.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Logging.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_MatMulDesc.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Operation.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_PointWiseDesc.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Reorder_Tensor.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Resample.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Tensor.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_VariantPack.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_interface.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
// NOLINTBEGIN(readability-identifier-naming)
using ConvDesc = ConvDesc_v8;
using ConvDescBuilder = ConvDescBuilder_v8;
using ReductionDesc = ReductionDesc_v8;
using ReductionDescBuilder = ReductionDescBuilder_v8;
using EngineHeuristicsBuilder = EngineHeuristicsBuilder_v8;
using EngineHeuristics = EngineHeuristics_v8;
using EngineBuilder = EngineBuilder_v8;
using Engine = Engine_v8;
using EngineConfig = EngineConfig_v8;
using EngineConfigBuilder = EngineConfigBuilder_v8;
using EngineFallbackList = EngineFallbackList_v8;
using EngineFallbackListBuilder = EngineFallbackListBuilder_v8;
using ResampleDesc = ResampleDesc_v8;
using ResampleDescBuilder = ResampleDescBuilder_v8;
// NOLINTEND(readability-identifier-naming)
} // namespace hipdnn_frontend::compatibility::cudnn_frontend
