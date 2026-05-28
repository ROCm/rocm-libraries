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

#include "stinkytofu/ir/DumpStinkyFunctionPass.hpp"

#include <cassert>
#include <fstream>
#include <string>

namespace {
std::string pathWithSuffix(const std::string& path, const std::string& newExtWithDot) {
    const auto dot = path.find_last_of('.');
    const auto slash = path.find_last_of("/\\");
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
        return path.substr(0, dot) + newExtWithDot;
    return path + newExtWithDot;
}
}  // namespace

namespace stinkytofu {
char DumpStinkyFunctionPass::ID = 0;

PreservedAnalyses DumpStinkyFunctionPass::run(Function& func, PassContext&,
                                              AnalysisManager& /*AM*/) {
    if (!config_.stirPath.empty()) {
        std::ofstream out(config_.stirPath, std::ios::out | std::ios::trunc);

        // use assert
        assert(out && "[DumpStinkyFunctionPass] Failed to open stirPath");
        AsmPrinter printer(out, config_.printerOptions);
        printer.print(func);
    }

    if (config_.emitAsm) {
        std::string asmPath = config_.asmPath;
        if (asmPath.empty()) {
            if (!config_.stirPath.empty())
                asmPath = pathWithSuffix(config_.stirPath, ".s");
            else
                asmPath = func.getName().empty() ? "dump.s" : func.getName() + ".s";
        }

        std::ofstream out(asmPath, std::ios::out | std::ios::trunc);
        assert(out && "[DumpStinkyFunctionPass] Failed to open asmPath");

        config_.emitterOptions.emitComments = true;
        config_.emitterOptions.emitCycleInfo = false;
        config_.emitterOptions.indent = 0;
        config_.emitterOptions.emitBlankLines = false;
        config_.emitterOptions.useSymbolicNames = false;  // Enable symbolic register names

        StinkyAsmEmitter emitter(config_.emitterOptions);
        emitter.emit(out, func);
    }
    return PreservedAnalyses::all();
}

std::unique_ptr<Pass> createDumpStinkyFunctionPass(DumpStinkyFunctionPassConfig config) {
    return std::make_unique<DumpStinkyFunctionPass>(std::move(config));
}
}  // namespace stinkytofu
