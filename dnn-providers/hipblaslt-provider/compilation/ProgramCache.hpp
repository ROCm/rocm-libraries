// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Program.hpp"

namespace hipblaslt_plugin::compilation
{

// The Program for `sourceName` and `options` on the current device, compiled for that
// device's architecture (--offload-arch=<gcnArchName> is added to `options`) on the
// first request and kept for the life of the process. Like hipBLASLt's handle, it
// relies on the caller keeping the handle's device current. A failed compile throws
// and is not cached, so a later request compiles again.
std::shared_ptr<const Program> getProgram(const std::string& sourceName,
                                          const std::vector<std::string>& options);

} // namespace hipblaslt_plugin::compilation
