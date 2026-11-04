// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#ifdef _WIN32

#include "PlatformUtils.windows.hpp"

#elif defined(__linux__)

#include "PlatformUtils.linux.hpp"

#else

#error "Unsupported platform"

#endif

// expandUser() is declared and implemented per-platform, in PlatformUtils.linux.hpp and
// PlatformUtils.windows.hpp (included above per platform) -- see those headers for the
// platform-specific contract. It follows the same per-platform split as
// getEnv()/setEnv()/unsetEnv().

namespace hipdnn_data_sdk::utilities
{

inline std::string getLibraryName(const char* libraryBaseName)
{
    return std::string(LIB_PREFIX) + libraryBaseName + SHARED_LIB_EXT;
}

inline std::string getExecutableName(const char* executableBaseName)
{
    return std::string(executableBaseName) + EXECUTABLE_EXT;
}

} // namespace hipdnn_data_sdk::utilities
