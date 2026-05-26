/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/**
 * @brief Attaches implicit registers to instructions for dependency tracking.
 *
 * Two kinds of implicit dependencies are materialized as registers so that the
 * def-use chain builder can see them:
 *
 * 1) Special registers (SCC, VCC, EXEC) declared via HW flags
 *    (Flags.def: IF_ImplicitRead/WriteSCC, IF_ImplicitReadVCC,
 *     IF_ImplicitRead/WriteEXEC). The corresponding singleton register is
 *    added to src/dest if not already present.
 *
 * 2) RegType::LDS pseudo-registers keyed by MemTokenData token IDs:
 *      LDS writers (tensor_load, ds_write) — token to dest (defines)
 *      LDS readers (ds_read)               — token to src  (uses)
 *      Barriers                            — token to both src and dest
 *    This creates the dependency chain:  writer → barrier → reader.
 */
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyBuildImplicitDependencyPass();

}  // namespace stinkytofu
