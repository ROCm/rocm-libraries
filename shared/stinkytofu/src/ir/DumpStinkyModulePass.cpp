// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/ir/DumpStinkyModulePass.hpp"

#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "stinkytofu/bindings/python/Module.hpp"

namespace {
std::string pathWithSuffix(const std::string& path, const std::string& newExtWithDot) {
    const auto dot = path.find_last_of('.');
    const auto slash = path.find_last_of("/\\");
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
        return path.substr(0, dot) + newExtWithDot;
    return path + newExtWithDot;
}

stinkytofu::AsmEmitterOptions normalizedEmitterOptions(
    stinkytofu::AsmEmitterOptions emitterOptions) {
    emitterOptions.emitComments = true;
    emitterOptions.emitCycleInfo = false;
    emitterOptions.indent = 0;
    emitterOptions.emitBlankLines = false;
    emitterOptions.useSymbolicNames = false;
    return emitterOptions;
}

void dumpAssembly(const std::vector<const stinkytofu::Function*>& functions,
                  const std::string& fallbackName,
                  const stinkytofu::DumpStinkyModulePassConfig& config) {
    if (!config.emitAsm) return;

    std::string asmPath = config.asmPath;
    if (asmPath.empty()) {
        if (!config.stirPath.empty())
            asmPath = pathWithSuffix(config.stirPath, ".s");
        else
            asmPath = fallbackName.empty() ? "dump.s" : fallbackName + ".s";
    }

    std::ofstream out(asmPath, std::ios::out | std::ios::trunc);
    assert(out && "[DumpStinkyModulePass] Failed to open asmPath");

    stinkytofu::StinkyAsmEmitter emitter(normalizedEmitterOptions(config.emitterOptions));
    for (const stinkytofu::Function* function : functions) emitter.emit(out, *function);
}

void dumpFunctions(const std::vector<const stinkytofu::Function*>& functions,
                   const std::string& fallbackName,
                   const stinkytofu::DumpStinkyModulePassConfig& config) {
    if (!config.stirPath.empty()) {
        std::ofstream out(config.stirPath, std::ios::out | std::ios::trunc);

        // use assert
        assert(out && "[DumpStinkyModulePass] Failed to open stirPath");
        stinkytofu::AsmPrinter printer(out, config.printerOptions);
        for (size_t i = 0; i < functions.size(); ++i) {
            if (i > 0) out << "\n";
            printer.print(*functions[i]);
        }
    }

    dumpAssembly(functions, fallbackName, config);
}

void dumpModule(const stinkytofu::StinkyAsmModule& module,
                const stinkytofu::DumpStinkyModulePassConfig& config) {
    if (!config.stirPath.empty()) {
        std::ofstream out(config.stirPath, std::ios::out | std::ios::trunc);
        assert(out && "[DumpStinkyModulePass] Failed to open stirPath");
        stinkytofu::AsmPrinter printer(out, config.printerOptions);
        printer.print(module);
    }

    dumpAssembly(module.getFunctions(), module.getName(), config);
}
}  // namespace

namespace stinkytofu {
char DumpStinkyModulePass::ID = 0;

PreservedAnalyses DumpStinkyModulePass::run(Function& func, PassContext& passCtx,
                                            AnalysisManager& AM) {
    if (module_ != nullptr) return run(*module_, passCtx, AM);

    std::vector<const Function*> functions{&func};
    dumpFunctions(functions, func.getName(), config_);
    return PreservedAnalyses::all();
}

PreservedAnalyses DumpStinkyModulePass::run(const StinkyAsmModule& module, PassContext&,
                                            AnalysisManager& /*AM*/) {
    dumpModule(module, config_);
    return PreservedAnalyses::all();
}

std::unique_ptr<Pass> createDumpStinkyModulePass(DumpStinkyModulePassConfig config) {
    return std::make_unique<DumpStinkyModulePass>(std::move(config));
}

std::unique_ptr<Pass> createDumpStinkyModulePass(const StinkyAsmModule& module,
                                                 DumpStinkyModulePassConfig config) {
    return std::make_unique<DumpStinkyModulePass>(module, std::move(config));
}
}  // namespace stinkytofu
