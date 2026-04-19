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

#include <string>

namespace stinkytofu {
class Function;
class PassContext;

/// Callback interface for observing PassManager pass execution.
class PassInstrumentation {
   public:
    virtual ~PassInstrumentation() = default;

    /// Called once at the beginning of PassManager::run(), before any pass executes.
    virtual void runBegin(Function& F, PassContext& ctx) {}

    /// Called immediately before each pass executes.
    virtual void beforePass(const std::string& passName, Function& F, PassContext& ctx) {}

    /// Called immediately after each pass executes.
    virtual void afterPass(const std::string& passName, Function& F, PassContext& ctx) {}

    /// Called once at the end of PassManager::run(), after all passes have executed.
    virtual void runEnd(Function& F, PassContext& ctx) {}
};

}  // namespace stinkytofu
