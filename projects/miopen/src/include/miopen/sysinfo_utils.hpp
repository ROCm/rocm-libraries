// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_SYSINFO_UTILS_HPP
#define MIOPEN_SYSINFO_UTILS_HPP

#include <string>

namespace miopen {
namespace sysinfo {

/// Retrieves the system hostname for logging and identification purposes
std::string GetSystemHostname();

} // namespace sysinfo
} // namespace miopen

#endif // MIOPEN_SYSINFO_UTILS_HPP
