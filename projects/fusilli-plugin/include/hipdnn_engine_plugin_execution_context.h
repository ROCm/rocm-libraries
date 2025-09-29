// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the fusilli plugin's definition of
// HipdnnEnginePluginExecutionContext. To hipDNN this type is opaque, it deals
// in hipdnnEnginePluginExecutionContext_t which is a pointer to the undefined
// HipdnnEnginePluginExecutionContext. Each plugin must define
// HipdnnEnginePluginExecutionContext in order to create something when hipDNN
// asks for an execution context.
//
// The execution context should store what's needed to execute a given kernel
// (plan in hipDNN parlance) in a hot loop without any overhead. For fusilli
// plugin, that maps to constructing and storing a fusilli::Graph based on
// hipDNN graph. When an execution is requested, it should be a simple lookup
// for uid -> tensor attribute, then a graph execution.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_PLUGIN_SRC_HIPDNN_ENGINE_PLUGIN_EXECUTION_CONTEXT_H
#define FUSILLI_PLUGIN_SRC_HIPDNN_ENGINE_PLUGIN_EXECUTION_CONTEXT_H

#include <fusilli.h>

struct HipdnnEnginePluginExecutionContext {
  // Fusilli graph.
  fusilli::Graph graph;

  // Map from hipDNN tensor UID to fusilli::TensorAttrs for graph boundary
  // tensors (inputs and outputs).
  std::unordered_map<int64_t, std::shared_ptr<fusilli::TensorAttr>>
      uidToFusilliTensorAttr;
};

#endif // FUSILLI_PLUGIN_SRC_HIPDNN_ENGINE_PLUGIN_EXECUTION_CONTEXT_H
