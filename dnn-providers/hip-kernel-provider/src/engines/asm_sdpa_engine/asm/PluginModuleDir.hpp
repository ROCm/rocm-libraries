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
#include <string>

namespace asm_sdpa_engine::asm_kernels
{

/// Replaces the plugin-relative ASM SDPA archive root when it names an existing
/// directory. Laid out like the default root: <dir>/<arch>/hip_kernel_provider_sdpa_<arch>.kpack.
inline constexpr const char* ASM_SDPA_KPACK_DIR_ENV = "HIPDNN_ASM_SDPA_KPACK_DIR";

std::filesystem::path currentPluginDirectory();

/// The directory holding the per-arch ASM SDPA .kpack archives, from the first of two
/// sources that answers: HIPDNN_ASM_SDPA_KPACK_DIR if it names a real directory, else
/// arch_content/asm_sdpa beside the loaded plugin module. Nothing is compiled in, so a
/// relocated, repackaged, or DESTDIR-staged install finds its own archives.
/// Throws HipdnnPluginException when the plugin module directory cannot be resolved.
std::filesystem::path asmSdpaKpackRoot();

/// The .kpack archive for @p arch under asmSdpaKpackRoot().
std::filesystem::path asmSdpaKpackArchivePath(const std::string& arch);

} // namespace asm_sdpa_engine::asm_kernels
