// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Runtime discovery of the plugin module directory.
//
// Returns the directory containing the hip_kernel_provider shared library,
// used to locate sibling assets (.kpack archives) without compile-time path
// baking. Cross-platform: delegates to hipdnn_data_sdk's dladdr / GetModuleHandleExW
// wrappers.

#pragma once

#include <filesystem>

namespace asm_sdpa_engine::asm_kernels
{

std::filesystem::path currentPluginDirectory();

} // namespace asm_sdpa_engine::asm_kernels
