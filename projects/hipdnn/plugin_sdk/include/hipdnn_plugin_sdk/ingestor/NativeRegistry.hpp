// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/// @file NativeRegistry.hpp
/// @brief The ingestor's native-hook registries: `ingestor::NativeRegistry<T>` and the
/// typed registries over it (GraphMatchRegistry, GraphCriterionRegistry,
/// KernelMatcherRegistry, ScoreRegistry, DispatchRegistry). The registry template itself is
/// in hipdnn_plugin_sdk/NativeRegistry.hpp and the hook types in NativeHooks.hpp; this
/// header includes both. GraphMatchRegistry serves only the `native` arm of a UED's
/// `graph_match`. A `nodes` pattern is compiled from the descriptor and needs no
/// registration.

#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
