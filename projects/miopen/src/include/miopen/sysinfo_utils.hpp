// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace miopen {
namespace sysinfo {

/// Retrieves the system hostname for logging and identification purposes
std::string GetSystemHostname();

} // namespace sysinfo
} // namespace miopen
