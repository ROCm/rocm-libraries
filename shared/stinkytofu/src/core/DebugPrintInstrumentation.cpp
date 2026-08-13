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
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"

#include <iostream>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
DebugPrintInstrumentation::DebugPrintInstrumentation(std::unique_ptr<PassManagerDebugConfig> cfg,
                                                     const StinkyAsmModule* module)
    : dbgCfg(std::move(cfg)), module_(module) {}

DebugPrintInstrumentation::~DebugPrintInstrumentation() = default;

void DebugPrintInstrumentation::dumpWithCallees(Function& F, std::ostream& out) const {
    F.dump(out);
    if (module_ == nullptr || &F != module_->getFunction(/*name=*/"")) return;
    for (const Function* fn : module_->getFunctions()) {
        if (fn == nullptr || !fn->getIsCallable()) continue;
        out << "\n; --- callee Function: " << fn->getName() << " ---\n";
        fn->dump(out);
    }
}

void DebugPrintInstrumentation::runBegin(Function& F, PassContext& /*ctx*/) {
    if (dbgCfg->shouldDumpInitialIR()) {
        dbgCfg->getOutputStreamInBefore() << "\n*** Initial IR (before all passes) ***\n";
        dumpWithCallees(F, dbgCfg->getOutputStreamInBefore());
        dbgCfg->getOutputStreamInBefore().flush();
    }
}

void DebugPrintInstrumentation::beforePass(const std::string& passName, Function& F,
                                           PassContext& /*ctx*/) {
    if (dbgCfg->shouldPrintPassName())
        std::cerr << "[StinkyTofu] Running pass: " << passName << "\n";

    if (dbgCfg->shouldPrintBefore(passName)) {
        dbgCfg->getOutputStreamInBefore() << "\n*** Before Pass: " << passName << " ***\n";
        dumpWithCallees(F, dbgCfg->getOutputStreamInBefore());
        dbgCfg->getOutputStreamInBefore().flush();
    }
}

void DebugPrintInstrumentation::afterPass(const std::string& passName, Function& F,
                                          PassContext& /*ctx*/) {
    if (dbgCfg->shouldPrintAfter(passName)) {
        dbgCfg->getOutputStreamInAfter() << "\n*** After Pass: " << passName << " ***\n";
        dumpWithCallees(F, dbgCfg->getOutputStreamInAfter());
        dbgCfg->getOutputStreamInAfter().flush();
    }
}

void DebugPrintInstrumentation::dumpModule(StinkyAsmModule& M, std::ostream& out) const {
    for (const Function* fn : M.getFunctions()) {
        if (fn == nullptr) continue;
        if (fn->getIsCallable()) out << "\n; --- callee Function: " << fn->getName() << " ---\n";
        fn->dump(out);
    }
}

void DebugPrintInstrumentation::beforeModulePass(const std::string& passName, StinkyAsmModule& M,
                                                 PassContext& /*ctx*/) {
    if (dbgCfg->shouldPrintPassName())
        std::cerr << "[StinkyTofu] Running module pass: " << passName << "\n";

    if (dbgCfg->shouldPrintBefore(passName)) {
        dbgCfg->getOutputStreamInBefore() << "\n*** Before Pass: " << passName << " ***\n";
        dumpModule(M, dbgCfg->getOutputStreamInBefore());
        dbgCfg->getOutputStreamInBefore().flush();
    }
}

void DebugPrintInstrumentation::afterModulePass(const std::string& passName, StinkyAsmModule& M,
                                                PassContext& /*ctx*/) {
    if (dbgCfg->shouldPrintAfter(passName)) {
        dbgCfg->getOutputStreamInAfter() << "\n*** After Pass: " << passName << " ***\n";
        dumpModule(M, dbgCfg->getOutputStreamInAfter());
        dbgCfg->getOutputStreamInAfter().flush();
    }
}

}  // namespace stinkytofu
