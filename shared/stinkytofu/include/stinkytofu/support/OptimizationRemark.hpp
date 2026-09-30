/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <iostream>
#include <string>

#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {

/// Lightweight optimization remark inspired by LLVM's OptimizationRemark.
///
/// Three categories mirror LLVM's -Rpass=, -Rpass-missed=, -Rpass-analysis=:
///   - Passed:   optimization successfully applied
///   - Missed:   optimization attempted but could not be applied
///   - Analysis: informational diagnostic about the generated code
struct OptimizationRemark {
    enum class Kind { Passed, Missed, Analysis };

    Kind kind;
    std::string passName;
    std::string remarkName;
    std::string message;
};

inline void emitRemark(const PassContext& passCtx, const OptimizationRemark& remark) {
    if (!passCtx.getRemarksEnabled()) return;

    const char* prefix = nullptr;
    switch (remark.kind) {
        case OptimizationRemark::Kind::Passed:
            prefix = "remark";
            break;
        case OptimizationRemark::Kind::Missed:
            prefix = "missed";
            break;
        case OptimizationRemark::Kind::Analysis:
            prefix = "analysis";
            break;
    }
    if (!prefix) return;
    std::cerr << prefix << ": " << remark.passName << ": " << remark.message << "\n";
}

}  // namespace stinkytofu
