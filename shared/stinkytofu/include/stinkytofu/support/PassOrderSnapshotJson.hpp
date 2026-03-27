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

#include "stinkytofu/core/Types.hpp"

#include <string>
#include <vector>

namespace stinkytofu
{
    class DAGScheduleJsonCollector;
    class Function;
    class PassContext;
    class StinkyInstruction;

    /// True if \p passName is in the configured allow-list and a non-empty snapshot path is set.
    bool shouldEmitPassOrderSnapshotAfterPass(const PassFeatureConfig& cfg, const std::string& passName);

    /// Linearize all processed Stinky instructions in function basic-block order (pointer stability).
    void snapshotProgramOrderStinkyLinear(
        Function& func, const PassContext& passCtx, std::vector<StinkyInstruction*>& outOrder);

    /// One viewer region for the whole function (register deps + before/after order); skipped if size changes.
    void appendPassOrderSnapshotJsonAfterPass(
        Function&                                      func,
        const std::vector<StinkyInstruction*>&         beforeOrder,
        const PassContext&                             passCtx,
        const std::string&                             passNameJustRan,
        DAGScheduleJsonCollector&                      collector);
} // namespace stinkytofu
