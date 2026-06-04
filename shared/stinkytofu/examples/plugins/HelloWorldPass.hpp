// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file HelloWorldPass.hpp
/// @brief Example plugin pass demonstrating the StinkyTofu pass plugin API.
///
/// This pass reads pluginDataStr("greeting") from StinkyAsmModule and
/// writes it back to pluginDataStr("greeting_result") to prove execution.
///
/// Build: compiled into a shared library (.so) with extern "C" registerPlugin().
/// Load:  rocisa.loadPlugin("path/to/libstinkytofu-plugin-helloworld.so")
/// Use:   stModule.registerPassAtExtensionPoint(EP_AfterRegionPasses, "HelloWorldPass")

#pragma once

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/pipeline/PassBuilder.hpp"

#ifdef _WIN32
#define HELLOWORLD_EXPORT __declspec(dllexport)
#else
#define HELLOWORLD_EXPORT __attribute__((visibility("default")))
#endif

namespace stinkytofu {
HELLOWORLD_EXPORT void registerHelloWorldPassPlugin();
}  // namespace stinkytofu
