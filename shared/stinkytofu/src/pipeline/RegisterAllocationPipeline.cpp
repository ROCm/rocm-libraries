// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/pipeline/RegisterAllocationPipeline.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/DumpStinkyModulePass.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/DefUseAnalysisCleanup.hpp"
#include "stinkytofu/transforms/asm/FlattenCFGPass.hpp"
#include "stinkytofu/transforms/asm/StinkyUnreachableBlockElimPass.hpp"
#include "stinkytofu/transforms/asm/TieExecMaskedWritesPass.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationConfig.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

namespace stinkytofu {
namespace {

/// Where RegisterAllocationDump writes: <outputDir>/<kernel>/ra/, or
/// ./<kernel>/ra/ when the module has no output directory.
std::filesystem::path registerAllocationDumpDir(const StinkyAsmModule& module) {
    const std::string kernel =
        module.getOutputName().empty() ? module.getName() : module.getOutputName();
    const std::string outputDir = module.getOutputDir();
    const std::filesystem::path root =
        outputDir.empty() ? std::filesystem::path(".") : std::filesystem::path(outputDir);
    std::filesystem::path dir = root / kernel / "ra";
    std::error_code ignored;
    std::filesystem::create_directories(dir, ignored);
    return dir;
}

}  // namespace

void addRegisterAllocationPasses(PassManager& pm, const StinkyAsmModule& module) {
    const StinkyAsmModule::ModuleOptions& moduleOptions = module.getModuleOptions();
    Expected<RegisterAllocationConfig> config = kernelRegisterAllocationConfig(
        moduleOptions.RegisterAllocation, moduleOptions.RegisterAllocationConfig);
    if (config.hasError()) {
        pm.addPass(createRegisterAllocationNotRunPass("register allocation config '" +
                                                      moduleOptions.RegisterAllocationConfig +
                                                      "' does not parse: " + config.getError()));
        return;
    }
    if (config->mode == RegisterAllocationMode::Off) return;

    RegisterAllocationOptions options = std::move(config->options);
    options.report = true;
    options.emitSymbolBreadcrumbs = true;

    // Rebuild the CFG in case a label was left inside a block, where it is a
    // branch target with no edges. Building only runs on a flat function.
    pm.addPass(createFlattenCFGPass());
    pm.addPass(createCFGBuilderPass());

    pm.addPass(createStinkyUnreachableBlockElimPass());
    pm.addPass(createRemoveDefUseAnalysisPass());

    // Must precede the lift, which binds the operand it adds. Otherwise a write
    // that only some lanes execute looks like a full definition of its
    // destination, and the allocator moves it off the value those lanes keep.
    pm.addPass(createTieExecMaskedWritesPass());

    std::optional<std::filesystem::path> dumpDir;
    if (moduleOptions.RegisterAllocationDump) dumpDir = registerAllocationDumpDir(module);
    if (dumpDir.has_value()) {
        DumpStinkyModulePassConfig before;
        before.stirPath = (*dumpDir / "before_lift.stir").string();
        pm.addPass(createDumpStinkyModulePass(std::move(before)));
    }

    // Lift only the classes allocation may move. A class left physical keeps
    // the producer's registers.
    LiftAsmRegistersToSSAOptions liftOptions;
    liftOptions.classes = options.allocate;
    pm.addPass(createLiftAsmRegistersToSSAPass(liftOptions));

    if (dumpDir.has_value()) {
        DumpStinkyModulePassConfig ssa;
        ssa.stirPath = (*dumpDir / "ssa.stir").string();
        ssa.printerOptions.ssaForm = true;
        ssa.ssaLiveOut = (*dumpDir / "live_ranges.txt").string();
        pm.addPass(createDumpStinkyModulePass(std::move(ssa)));
        options.blockDumpPrefix = (*dumpDir / "blocks.").string();
    }

    pm.addPass(createRegisterAllocationPass(std::move(options)));
}

}  // namespace stinkytofu
