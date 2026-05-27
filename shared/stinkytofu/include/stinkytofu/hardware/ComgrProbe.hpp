// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {

/// Try to assemble the given string against the specified ISA using comgr.
/// Returns true if assembly succeeds, false on failure or if comgr is unavailable.
STINKYTOFU_EXPORT bool tryAssembleWithComgr(const std::string& asmString,
                                            const std::string& isaName, uint32_t wavefrontSize);

/// Returns true if comgr support was compiled into this build.
STINKYTOFU_EXPORT bool hasComgrSupport();

}  // namespace stinkytofu
