/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// Copies the module's versioned GIR frame contract onto a scoped temporary Function.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGirFrameContractImportPass(
    const StinkyAsmModule& module);

}  // namespace stinkytofu
