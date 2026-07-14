// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

namespace rocke_client::dispatcher
{

/// Return the directory that contains the rocke-client plugin DSO (or DLL on
/// Windows). The result is weakly-canonical and absolute, matching the
/// convention used by defaultArtifactRoot() for bundle resolution.
///
/// Mirrors hipdnn_backend::platform_utilities::getCurrentModuleDirectory()
/// from projects/hipdnn/backend/src/PlatformUtils.linux/windows.cpp.
/// rocke deliberately does not link the backend (platform-utils split
/// pending) — keep this implementation in sync with the backend source.
///
/// @throws std::runtime_error if the module path cannot be determined.
std::filesystem::path currentPluginDirectory();

} // namespace rocke_client::dispatcher
