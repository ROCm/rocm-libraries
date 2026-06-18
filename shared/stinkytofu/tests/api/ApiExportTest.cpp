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

// API export surface test.
//
// This binary links stinkytofu as a SHARED library. If a symbol used by a
// real consumer (rocisa, stinkytofu-opt) is accidentally un-exported, this
// file will fail to LINK — catching the regression before it reaches downstream.
//
// Rule: only add tests here for symbols that rocisa or stinkytofu-opt actually
// call. Everything else should stay hidden.

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <sstream>
#include <string>

// --- rocisa consumer headers ---
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

// --- stinkytofu-opt consumer headers ---
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/ir/DumpStinkyFunctionPass.hpp"
#include "stinkytofu/pipeline/Backend.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/RawAsmParser.hpp"
#include "stinkytofu/support/DAGScheduleJsonWriter.hpp"
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"
#include "stinkytofu/support/PassOrderSnapshotJson.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/LongBranchLoweringPass.hpp"
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"
#include "stinkytofu/transforms/asm/RaiseVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"
#include "stinkytofu/transforms/asm/RemoveDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/RemoveWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveNopPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

using namespace stinkytofu;

namespace {
constexpr std::array<int, 3> kArch{12, 5, 0};

std::unique_ptr<StinkyAsmModule> makeModule() {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    return std::make_unique<StinkyAsmModule>("api_test", kArch, opts);
}
}  // namespace

// =============================================================================
// BackendRegistry (rocisa + stinkytofu-opt)
// =============================================================================

TEST(ApiExport, BackendRegistryRegisterAllBackends) {
    // rocisa: main.cpp calls registerAllBackends() at startup
    BackendRegistry::registerAllBackends();
}

TEST(ApiExport, BackendRegistryGetArchPipeline) {
    // stinkytofu-opt: getArchPipeline() to validate --arch flag
    BackendRegistry::registerAllBackends();
    // May return null for unknown arch — that's fine; the symbol must link.
    (void)BackendRegistry::getArchPipeline(kArch);
}

TEST(ApiExport, BackendRegistryParseArchKey) {
    std::array<int, 3> arch{};
    (void)BackendRegistry::parseArchKey("gfx1250", arch);
}

TEST(ApiExport, BackendRegistryGetRegisteredArchKeys) {
    BackendRegistry::registerAllBackends();
    auto keys = BackendRegistry::getRegisteredArchKeys();
    EXPECT_FALSE(keys.empty());
}

// =============================================================================
// StinkyAsmModule (rocisa + stinkytofu-opt)
// =============================================================================

TEST(ApiExport, StinkyAsmModuleConstruction) {
    auto module = makeModule();
    EXPECT_NE(module, nullptr);
}

TEST(ApiExport, StinkyAsmModuleGetFunction) {
    auto module = makeModule();
    // getFunction() returns the internal Function; verify it's accessible (symbol exported).
    Function& func = module->getFunction();
    (void)func;
}

// =============================================================================
// Backend (stinkytofu-opt pipeline mode)
// =============================================================================

TEST(ApiExport, BackendRunOptimization) {
    auto module = makeModule();
    Backend backend(*module);
    EXPECT_TRUE(backend.runOptimization());
}

// =============================================================================
// PassManager + AnalysisManager (stinkytofu-opt individual pass mode)
// =============================================================================

TEST(ApiExport, PassManagerConstruction) {
    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
}

TEST(ApiExport, PassManagerDebugConfig) {
    PassManagerDebugConfig cfg;
    cfg.setPrintBeforeAll(true);
    cfg.setPrintAfterAll(true);
}

TEST(ApiExport, DebugPrintInstrumentation) {
    auto cfg = std::make_unique<PassManagerDebugConfig>();
    cfg->setPrintBeforeAll(false);
    auto instr = std::make_shared<DebugPrintInstrumentation>(std::move(cfg));
    EXPECT_NE(instr, nullptr);
}

// =============================================================================
// Serialization (stinkytofu-opt)
// =============================================================================

TEST(ApiExport, ParseSourceStringWithDiagnostics) {
    auto result = parseSourceStringWithDiagnostics("func f() {}");
    // May fail to parse — we only care that the symbol links.
    (void)result;
}

TEST(ApiExport, ParseAllSourceStringsWithDiagnostics) {
    auto result = parseAllSourceStringsWithDiagnostics("");
    (void)result;
}

TEST(ApiExport, StinkyIRConverterPopulateFromParsed) {
    // Symbol must be exported; call with minimal args to verify linkage.
    ParsedFunction pf;
    pf.funcName = "f";
    pf.blocks.push_back(std::make_unique<ParsedBlock>());
    pf.blocks[0]->blockId = "entry";
    Function func("f");
    GfxArchID arch = getGfxArchID(kArch[0], kArch[1], kArch[2]);
    (void)StinkyIRConverter::populateFunctionFromParsed(pf, func, arch);
}

TEST(ApiExport, StinkyAsmEmitter) {
    Function func("f");
    func.createBasicBlock("entry");
    AsmEmitterOptions opts{};
    StinkyAsmEmitter emitter(opts);
    std::ostringstream oss;
    emitter.emit(oss, func);
}

// =============================================================================
// ToolchainCaps (stinkytofu-opt)
// =============================================================================

TEST(ApiExport, ToolchainCapsProbe) {
    GfxArchID arch = getGfxArchID(kArch[0], kArch[1], kArch[2]);
    auto caps = ToolchainCaps::probe(arch);
    (void)caps;
}

// =============================================================================
// DAGScheduleJsonCollector + PassOrderSnapshotInstrumentation (stinkytofu-opt)
// =============================================================================

TEST(ApiExport, DAGScheduleJsonCollector) {
    auto collector = std::make_shared<DAGScheduleJsonCollector>("", "f");
    EXPECT_NE(collector, nullptr);
}

TEST(ApiExport, PassOrderSnapshotInstrumentation) {
    auto collector = std::make_shared<DAGScheduleJsonCollector>("", "f");
    auto instr = std::make_shared<PassOrderSnapshotInstrumentation>(std::move(collector));
    EXPECT_NE(instr, nullptr);
}

// =============================================================================
// SignatureBase (rocisa ToStinkyTofuUtils)
// =============================================================================

TEST(ApiExport, SignatureBaseConstruction) {
    SignatureBase sig("k", kArch, 2, "v5", 0, std::array<int, 3>{1, 1, 1}, 1, 256);
    EXPECT_FALSE(sig.toString().empty());
}

// =============================================================================
// ArchHelper (rocisa + stinkytofu-opt)
// =============================================================================

TEST(ApiExport, GetGfxArchID) {
    GfxArchID arch = getGfxArchID(12, 5, 0);
    EXPECT_EQ(arch, GfxArchID::Gfx1250);
}

// =============================================================================
// Pass factory functions (stinkytofu-opt availablePasses table)
// =============================================================================

TEST(ApiExport, PassFactories) {
    EXPECT_NE(createStinkyDAGSchedulerPass(), nullptr);
    EXPECT_NE(createSetMatrixReusePass(), nullptr);
    EXPECT_NE(createStinkyBuildImplicitDependencyPass(), nullptr);
    EXPECT_NE(createStinkyRemoveWaitCntPass(), nullptr);
    EXPECT_NE(createStinkyRemoveNopPass(), nullptr);
    EXPECT_NE(createStinkyWaitCntInsertionPass(), nullptr);
    EXPECT_NE(createBuildUseDefChainPass(true, false), nullptr);
    EXPECT_NE(createCFGBuilderPass(), nullptr);
    EXPECT_NE(createDumpStinkyFunctionPass({}), nullptr);
    EXPECT_NE(createPeepholeOptimizationPass(), nullptr);
    EXPECT_NE(createDeadCodeEliminationPass(), nullptr);
    EXPECT_NE(createRedundantMovEliminationPass(), nullptr);
    EXPECT_NE(createStinkyIRVerifierPass(), nullptr);
    EXPECT_NE(createRemoveDelayAluPass(), nullptr);
    EXPECT_NE(createInsertDelayAluPass(), nullptr);
    EXPECT_NE(createLoopRegionRemarkPass(), nullptr);
    EXPECT_NE(createMemTokenConsistencyCheckPass(), nullptr);
    EXPECT_NE(createRaiseVgprMsbPass(), nullptr);
    EXPECT_NE(createInsertVgprMsbPass(), nullptr);
    EXPECT_NE(createLongBranchLoweringPass(), nullptr);
    EXPECT_NE(createInsertClusterBarrierPass(true, 1, 1), nullptr);
    EXPECT_NE(createRemoveWaitAluPass(), nullptr);
    EXPECT_NE(createInsertWaitAluPass(), nullptr);
}
