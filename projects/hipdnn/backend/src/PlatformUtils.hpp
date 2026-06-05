// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <string>

#if !defined(_WIN32) && !defined(__linux__)

#error "Unsupported platform"

#endif

namespace hipdnn_backend::platform_utilities
{

using PluginLibHandle = hipdnn_data_sdk::utilities::SharedLibraryHandle;

std::filesystem::path getCurrentModuleDirectory();

PluginLibHandle openLibrary(const std::filesystem::path& libraryPath);
void closeLibrary(PluginLibHandle handle);
void* getSymbol(PluginLibHandle handle, const char* symbolName);
std::string getSystemInfo();

}
