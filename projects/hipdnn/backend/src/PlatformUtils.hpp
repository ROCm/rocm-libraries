// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
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
std::array<uint8_t, 16> generateUuidV4();

/// True while the process is exiting and the loader is unloading libraries. On Windows,
/// ExitProcess has terminated every other thread by then, so nothing may wait for one; the
/// answer comes from RtlDllShutdownInProgress, an undocumented ntdll export, and is false
/// when ntdll or the export is missing, which keeps the normal, waiting logger teardown.
/// Always false on Linux, where exit handlers run while the other threads are still alive.
bool isProcessTerminating();

}
