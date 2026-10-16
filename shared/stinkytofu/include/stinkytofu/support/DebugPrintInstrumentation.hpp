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
#include <ostream>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassInstrumentation.hpp"

namespace stinkytofu {
class PassManagerDebugConfig;
class Function;
class StinkyAsmModule;

/// Print IR before/after passes using PassManagerDebugConfig settings.
///
/// When a StinkyAsmModule is supplied, dumps of the module's entry Function are
/// followed by every callee Function, so whole-module debug output includes
/// callee bodies (the PassManager only runs passes on, and hands this observer,
/// the entry Function). Without a module it dumps only the Function it is given.
class STINKYTOFU_EXPORT DebugPrintInstrumentation : public PassInstrumentation {
   public:
    explicit DebugPrintInstrumentation(std::unique_ptr<PassManagerDebugConfig> cfg,
                                       const StinkyAsmModule* module = nullptr);
    ~DebugPrintInstrumentation() override;

    void runBegin(Function& F, PassContext& ctx) override;
    void beforePass(const std::string& passName, Function& F, PassContext& ctx) override;
    void afterPass(const std::string& passName, Function& F, PassContext& ctx) override;

    void beforeModulePass(const std::string& passName, StinkyAsmModule& M,
                          PassContext& ctx) override;
    void afterModulePass(const std::string& passName, StinkyAsmModule& M,
                         PassContext& ctx) override;

   private:
    /// Dump F to out; if a module is set and F is its entry Function,
    /// also dump each callee Function under a labeled header.
    void dumpWithCallees(Function& F, std::ostream& out) const;

    /// Dump every function of M (entry + callees) to out.
    void dumpModule(StinkyAsmModule& M, std::ostream& out) const;

    std::unique_ptr<PassManagerDebugConfig> dbgCfg;
    const StinkyAsmModule* module_ = nullptr;
};

}  // namespace stinkytofu
