// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <string>

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>

namespace hipdnn_backend::platform_utilities
{

typedef HMODULE PluginLibHandle;

}

#elif defined(__linux__)

namespace hipdnn_backend::platform_utilities
{

typedef void* PluginLibHandle;

}

#else

#error "Unsupported platform"

#endif

namespace hipdnn_backend::platform_utilities
{

std::filesystem::path getCurrentModuleDirectory();

PluginLibHandle openLibrary(const std::filesystem::path& libraryPath);
void closeLibrary(PluginLibHandle handle);
void* getSymbol(PluginLibHandle handle, const char* symbolName);

}
