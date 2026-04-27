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

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
DebugPrintInstrumentation::DebugPrintInstrumentation(std::unique_ptr<PassManagerDebugConfig> cfg)
    : dbgCfg(std::move(cfg)) {}

DebugPrintInstrumentation::~DebugPrintInstrumentation() = default;

void DebugPrintInstrumentation::runBegin(Function& F, PassContext& /*ctx*/) {
    if (dbgCfg->shouldDumpInitialIR()) {
        dbgCfg->getOutputStreamInBefore() << "\n*** Initial IR (before all passes) ***\n";
        F.dump(dbgCfg->getOutputStreamInBefore());
        dbgCfg->getOutputStreamInBefore().flush();
    }
}

void DebugPrintInstrumentation::beforePass(const std::string& passName, Function& F,
                                           PassContext& /*ctx*/) {
    if (dbgCfg->shouldPrintPassName())
        std::cerr << "[StinkyTofu] Running pass: " << passName << "\n";

    if (dbgCfg->shouldPrintBefore(passName)) {
        dbgCfg->getOutputStreamInBefore() << "\n*** Before Pass: " << passName << " ***\n";
        F.dump(dbgCfg->getOutputStreamInBefore());
        dbgCfg->getOutputStreamInBefore().flush();
    }
}

void DebugPrintInstrumentation::afterPass(const std::string& passName, Function& F,
                                          PassContext& /*ctx*/) {
    if (dbgCfg->shouldPrintAfter(passName)) {
        dbgCfg->getOutputStreamInAfter() << "\n*** After Pass: " << passName << " ***\n";
        F.dump(dbgCfg->getOutputStreamInAfter());
        dbgCfg->getOutputStreamInAfter().flush();
    }
}

}  // namespace stinkytofu
