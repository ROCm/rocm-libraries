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

#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/analysis/asm/HazardGapAnalysisPass.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/DumpMemTokenIRStructurePass.hpp"
#include "stinkytofu/ir/DumpStinkyModulePass.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/AsmMovePropagationPass.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"
#include "stinkytofu/transforms/asm/DefUseAnalysisCleanup.hpp"
#include "stinkytofu/transforms/asm/EpilogueStoreSinkPass.hpp"
#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"
#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/InsertInitialUnclausedVmemPass.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/LongBranchLoweringPass.hpp"
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"
#include "stinkytofu/transforms/asm/RaiseVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"
#include "stinkytofu/transforms/asm/RegionClonePass.hpp"
#include "stinkytofu/transforms/asm/RemoveDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/RemoveDscntPass.hpp"
#include "stinkytofu/transforms/asm/RemoveInstructionPass.hpp"
#include "stinkytofu/transforms/asm/RemoveWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyMergeBarrierPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveNopPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyUnreachableBlockElimPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelDynamicPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelStaticPass.hpp"
#include "stinkytofu/transforms/asm/TDMLoadWaveSyncPass.hpp"
#include "stinkytofu/transforms/asm/WaitAwareScheduleRepairPass.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;

// Structure to hold pass information.
//
// The creator receives the comma-separated argument list that was supplied
// via `--PassName=arg1,arg2`. Passes that don't accept arguments simply
// ignore the vector. Each pass is responsible for documenting and parsing
// its own arguments (typically simple flag-name strings or key=value pairs).
struct PassInfo {
    const char* name;
    std::function<std::unique_ptr<Pass>(const std::vector<std::string>& args)> creator;
};

// Helper: returns true if `args` contains the literal flag name `flag`
// (case-sensitive, exact match).
inline bool hasPassArg(const std::vector<std::string>& args, const char* flag) {
    for (const auto& a : args)
        if (a == flag) return true;
    return false;
}

// Helper: value of `key=value` in `args`, or `defaultValue` when absent.
inline std::string passArgValue(const std::vector<std::string>& args, const char* key,
                                std::string defaultValue = {}) {
    const std::string prefix = std::string(key) + "=";
    for (const auto& a : args) {
        if (a.starts_with(prefix)) return a.substr(prefix.size());
    }
    return defaultValue;
}

// Helper: every value of `key=value` in `args`, in order.
//
// Pass arguments are comma-separated, so a comma cannot also separate items
// inside one value. A list is therefore written either as repeated keys
// (`rules=A,rules=B`) or joined with '+' (`rules=A+B`); both are accepted.
inline std::vector<std::string> passArgValues(const std::vector<std::string>& args,
                                              const char* key) {
    const std::string prefix = std::string(key) + "=";
    std::vector<std::string> values;
    for (const auto& a : args) {
        if (!a.starts_with(prefix)) continue;
        const std::string rest = a.substr(prefix.size());
        size_t start = 0;
        while (start <= rest.size()) {
            const size_t plus = rest.find('+', start);
            std::string tok =
                rest.substr(start, plus == std::string::npos ? std::string::npos : plus - start);
            if (!tok.empty()) values.push_back(std::move(tok));
            if (plus == std::string::npos) break;
            start = plus + 1;
        }
    }
    return values;
}

// Helper: parse a class list like "vs", "v", or "s" into a RegClassSet. Used for
// both the lift scope and the allocation scope. Returns nullopt for an unknown
// class letter, so a typo is reported rather than silently selecting something
// else.
inline std::optional<RegClassSet> parseRegClasses(const std::string& spec) {
    RegClassSet classes;
    for (char c : spec) {
        switch (c) {
            case 'v':
                classes.add(RegType::V);
                break;
            case 's':
                classes.add(RegType::S);
                break;
            default:
                return std::nullopt;
        }
    }
    if (classes.empty()) return std::nullopt;
    return classes;
}

// Helper: parse "s0" as one register or "s0:19" as the inclusive run s0..s19.
// Returns nullopt for anything else, so a typo is reported rather than holding
// registers nobody named.
inline std::optional<AllocationScope::HeldRange> parseHeldRange(const std::string& spec) {
    if (spec.size() < 2) return std::nullopt;
    RegType type = RegType::UNKNOWN;
    if (spec[0] == 's')
        type = RegType::S;
    else if (spec[0] == 'v')
        type = RegType::V;
    else
        return std::nullopt;

    const auto digits = [](const std::string& text) -> std::optional<uint32_t> {
        if (text.empty()) return std::nullopt;
        uint32_t value = 0;
        for (char c : text) {
            if (c < '0' || c > '9') return std::nullopt;
            value = value * 10 + static_cast<uint32_t>(c - '0');
        }
        return value;
    };

    const std::string body = spec.substr(1);
    const size_t colon = body.find(':');
    if (colon == std::string::npos) {
        const std::optional<uint32_t> only = digits(body);
        if (!only.has_value()) return std::nullopt;
        return AllocationScope::HeldRange{type, *only, *only};
    }

    const std::optional<uint32_t> start = digits(body.substr(0, colon));
    const std::optional<uint32_t> end = digits(body.substr(colon + 1));
    if (!start.has_value() || !end.has_value() || *end < *start) return std::nullopt;
    return AllocationScope::HeldRange{type, *start, *end};
}

// List of available passes
const std::vector<PassInfo> availablePasses = {
    {"StinkyDAGSchedulerPass", [](const auto&) { return createStinkyDAGSchedulerPass(); }},
    // HazardGapAnalysisPass accepts optional arg: verbose
    {"HazardGapAnalysisPass",
     [](const std::vector<std::string>& args) {
         return createHazardGapAnalysisPass(hasPassArg(args, "verbose"));
     }},
    {"StinkyMergeBarrierPass", [](const auto&) { return createStinkyMergeBarrierPass(); }},
    {"SetMatrixReusePass", [](const auto&) { return createSetMatrixReusePass(); }},
    {"SwInstructionPrefetchRelStaticPass",
     [](const auto&) { return createSwInstructionPrefetchRelStaticPass(std::string{}); }},
    {"SwInstructionPrefetchRelDynamicPass",
     [](const auto&) { return createSwInstructionPrefetchRelDynamicPass(std::string{}); }},
    {"AccumulateInstructionSizePass",
     [](const auto&) { return createAccumulateInstructionSizePass(""); }},
    {"AccumulateInstructionSizeDebugPass",
     [](const auto&) { return createAccumulateInstructionSizePassWithDebug(); }},
    {"StinkyBuildImplicitDependencyPass",
     [](const auto&) { return createStinkyBuildImplicitDependencyPass(); }},
    // StinkyRemoveWaitCntPass accepts:
    //   keepTensor   — leave s_wait_tensorcnt in place (default strips it)
    //   removeXcnt   — also strip s_wait_xcnt (the O3 backend policy)
    //   removeKmcnt  — also strip s_wait_kmcnt
    {"StinkyRemoveWaitCntPass",
     [](const std::vector<std::string>& args) {
         RemoveWaitCntOptions options;
         options.removeTensor = !hasPassArg(args, "keepTensor");
         options.removeXcnt = hasPassArg(args, "removeXcnt");
         options.removeKmcnt = hasPassArg(args, "removeKmcnt");
         return createStinkyRemoveWaitCntPass(options);
     }},
    {"StinkyRemoveNopPass", [](const auto&) { return createStinkyRemoveNopPass(); }},
    {"RemoveDscntPass", [](const auto&) { return createRemoveDscntPass(); }},
    {"StinkyWaitCntInsertionPass",
     [](const std::vector<std::string>& args) {
         WaitCntInsertionOptions options;
         options.enableLoopCarriedTokenDeps = hasPassArg(args, "enableLoopCarriedTokenDeps");
         return createStinkyWaitCntInsertionPass(options);
     }},
    // Gfx1250HazardPass accepts:
    //   profile — print the xcnt drain summary (per rule and drain site) to stderr
    {"Gfx1250HazardPass",
     [](const std::vector<std::string>& args) {
         return createGfx1250HazardPass(hasPassArg(args, "profile"));
     }},
    {"WaitAwareScheduleRepairPass",
     [](const std::vector<std::string>& args) {
         constexpr int kDefaultSlotsToMovePastAnchor = 1;
         const std::string prefix = "kSlotsToMovePastAnchor=";
         for (const auto& arg : args) {
             if (arg.starts_with(prefix))
                 return createWaitAwareScheduleRepairPass(
                     std::atoi(arg.substr(prefix.size()).c_str()));
         }
         return createWaitAwareScheduleRepairPass(kDefaultSlotsToMovePastAnchor);
     }},
    // BuildUseDefChainPass accepts:
    //   includePseudo    — also build chains for pseudo registers (memtokens)
    //   noClearExisting  — keep any existing PHIs/chains
    {"BuildUseDefChainPass",
     [](const std::vector<std::string>& args) {
         bool clearExisting = !hasPassArg(args, "noClearExisting");
         bool includePseudo = hasPassArg(args, "includePseudo");
         return createBuildUseDefChainPass(clearExisting, includePseudo);
     }},
    {"CFGBuilderPass", [](const auto&) { return createCFGBuilderPass(); }},
    // Erases blocks not reachable from the entry along CFG successor edges.
    // Run after CFGBuilderPass / LongBranchLoweringPass; an incomplete CFG
    // would make this delete live targets.
    {"StinkyUnreachableBlockElimPass",
     [](const auto&) { return createStinkyUnreachableBlockElimPass(); }},
    // Discards physical-register PHIs and def-use chains. Lifting rejects a
    // leftover analysis PHI, so this runs immediately before
    // LiftAsmRegistersToSSAPass rather than inside it.
    {"RemoveDefUseAnalysisPass", [](const auto&) { return createRemoveDefUseAnalysisPass(); }},
    // LiftAsmRegistersToSSAPass accepts:
    //   classes=<vs>   — register classes to lift (default vs). Anything left out
    //                    stays physical and no later pass rewrites it, so
    //                    classes=s allocates SGPRs with every VGPR untouched.
    //   strictLiveIns  — reject a read with no reaching definition instead of
    //                    inferring a function live-in
    //   noVerify       — skip attached SSA verification after construction
    {"LiftAsmRegistersToSSAPass",
     [](const std::vector<std::string>& args) -> std::unique_ptr<Pass> {
         LiftAsmRegistersToSSAOptions options;
         const std::optional<RegClassSet> classes =
             parseRegClasses(passArgValue(args, "classes", "vs"));
         if (!classes.has_value()) return nullptr;
         options.classes = *classes;
         options.allowInferredLiveIns = !hasPassArg(args, "strictLiveIns");
         options.verify = !hasPassArg(args, "noVerify");
         return createLiftAsmRegistersToSSAPass(options);
     }},
    // RegisterAllocationPass accepts:
    //   allocator=<name>  — registry name: greedy (default) or legacy
    //   classes=<vs>      — classes the policy may move (default v). Anything else
    //                       keeps the register it was lifted from, so classes=s
    //                       reallocates SGPRs and leaves every VGPR alone. Must be
    //                       a subset of what the lift covered.
    //   regionEnd=<label> — only values whose live range lies entirely before this
    //                       block's end slot may move (^ prefix optional)
    //   pinReg=<range>    — leave these registers exactly as found: each keeps
    //                       the value lifted into it and takes no other. One as
    //                       pinReg=s0, an inclusive run as pinReg=s0:19. Repeat
    //                       the key for disjoint runs
    //   apply             — write the colouring through destroyAttachedSSA
    //                       (also runs syncRegisterSymbols; see
    //                       docs/developer/register-allocation.md §11.1)
    //   report            — emit the shadow comparison (peak, highest, regionPeak)
    //   emitRegisterMap   — after apply, insert a producer→allocated TEXTBLOCK
    //   emitSymbolBreadcrumbs — after apply, append "// s18 was sgprTmp" comments
    //                       on operands whose symbolic name was stripped
    //   noVerify          — skip AllocationVerifier
    //   rules=<name>      — force this architecture rule Active, ignoring the
    //                       chip's own capability gate. Repeat the key or join
    //                       with '+' for several; an unknown name is an error,
    //                       because a test that enables nothing still passes
    //   rules=all         — force every rule the triple declares
    //   ruleAudit         — force every declared rule to Audit, which reports
    //                       against the producer's colouring without enforcing
    //   noRules           — behave as if the chip declared no rules
    {"RegisterAllocationPass",
     [](const std::vector<std::string>& args) -> std::unique_ptr<Pass> {
         RegisterAllocationOptions options;
         options.allocator = passArgValue(args, "allocator", options.allocator);
         const std::optional<RegClassSet> classes =
             parseRegClasses(passArgValue(args, "classes", "v"));
         if (!classes.has_value()) return nullptr;
         options.allocate = *classes;
         options.regionEnd = passArgValue(args, "regionEnd", "");
         for (const std::string& name : passArgValues(args, "pinReg")) {
             const std::optional<AllocationScope::HeldRange> range = parseHeldRange(name);
             if (!range.has_value()) return nullptr;
             options.pinRegisters.push_back(*range);
         }
         options.applyToOperands = hasPassArg(args, "apply");
         options.report = hasPassArg(args, "report");
         options.emitRegisterMap = hasPassArg(args, "emitRegisterMap");
         options.emitSymbolBreadcrumbs = hasPassArg(args, "emitSymbolBreadcrumbs");
         options.verify = !hasPassArg(args, "noVerify");
         options.rules.disableAll = hasPassArg(args, "noRules");
         options.rules.auditAll = hasPassArg(args, "ruleAudit");
         for (std::string& name : passArgValues(args, "rules")) {
             if (name == "all") {
                 options.rules.activateAll = true;
                 continue;
             }
             options.rules.activate.push_back(std::move(name));
         }
         return createRegisterAllocationPass(std::move(options));
     }},
    // DumpStinkyModulePass accepts:
    //   ssaForm  — print attached SSA values instead of physical registers
    //   ssaLive  — also dump SSA live ranges and peak pressure
    //   stdout   — print to stdout instead of dump_module.stir /
    //              ssa_live_intervals.txt
    {"DumpStinkyModulePass",
     [](const std::vector<std::string>& args) {
         const bool toStdout = hasPassArg(args, "stdout");
         DumpStinkyModulePassConfig config;
         config.stirPath = toStdout ? std::string{} : "dump_module.stir";
         config.stirToStdout = toStdout;
         config.printerOptions.ssaForm = hasPassArg(args, "ssaForm");
         if (hasPassArg(args, "ssaLive"))
             config.ssaLiveOut = toStdout ? "-" : "ssa_live_intervals.txt";
         return createDumpStinkyModulePass(std::move(config));
     }},
    {"DumpMemTokenIRStructurePass",
     [](const auto&) {
         return createDumpMemTokenIRStructurePass({.path = "dump_memtoken_ir_structure.txt"});
     }},
    // EpilogueStoreSinkPass accepts: noStoreGroup (maxStoreGroupSize=0),
    // noGuard (avoidMsbXcntDrain=false)
    {"EpilogueStoreSinkPass",
     [](const std::vector<std::string>& args) {
         EpilogueStoreSinkOptions options;
         if (hasPassArg(args, "noStoreGroup")) options.maxStoreGroupSize = 0;
         if (hasPassArg(args, "noGuard")) options.avoidMsbXcntDrain = false;
         return createEpilogueStoreSinkPass(options);
     }},
    {"PeepholeOptimizationPass", [](const auto&) { return createPeepholeOptimizationPass(); }},
    {"DeadCodeEliminationPass", [](const auto&) { return createDeadCodeEliminationPass(); }},
    {"RedundantMovEliminationPass",
     [](const auto&) { return createRedundantMovEliminationPass(); }},
    {"AsmMovePropagationPass", [](const auto&) { return createAsmMovePropagationPass(); }},
    {"StinkyIRVerifierPass", [](const auto&) { return createStinkyIRVerifierPass(); }},
    {"RemoveDelayAluPass", [](const auto&) { return createRemoveDelayAluPass(); }},
    // RemoveInstructionPass accepts one or more mnemonics:
    //   --RemoveInstructionPass=s_wait_alu,tensor_load_to_lds,s_nop
    {"RemoveInstructionPass",
     [](const std::vector<std::string>& args) {
         if (args.empty()) return std::unique_ptr<Pass>{};
         return createRemoveInstructionPass(args);
     }},
    {"InsertDelayAluPass", [](const auto&) { return createInsertDelayAluPass(); }},
    {"LoopRegionRemarkPass", [](const auto&) { return createLoopRegionRemarkPass(); }},
    {"MemTokenConsistencyCheckPass",
     [](const auto&) { return createMemTokenConsistencyCheckPass(); }},
    {"RaiseVgprMsbPass", [](const auto&) { return createRaiseVgprMsbPass(); }},
    {"InsertVgprMsbPass", [](const auto&) { return createInsertVgprMsbPass(); }},
    {"InsertInitialUnclausedVmemPass",
     [](const auto&) { return createInsertInitialUnclausedVmemPass(); }},
    {"LongBranchLoweringPass", [](const auto&) { return createLongBranchLoweringPass(); }},
    {"InsertClusterBarrierPass", [](const auto&) { return createInsertClusterBarrierPass(); }},
    {"TDMLoadWaveSyncPass", [](const auto&) { return createTDMLoadWaveSyncPass(); }},
    {"RemoveWaitAluPass", [](const auto&) { return createRemoveWaitAluPass(); }},
    {"InsertWaitAluPass",
     [](const std::vector<std::string>& args) {
         return createInsertWaitAluPass(hasPassArg(args, "enableESM2TrackValuVsrc"));
     }},
    {"InsertCoexecHazardPass", [](const auto&) { return createInsertCoexecHazardPass(); }},
    {"RegionClonePass",
     [](const auto&) {
         return createRegionClonePass({CloneSpec{"InitCIterWmma", "label_LoopBeginL"}});
     }},
};

/**
 * Create default DebugPrintInstrumentation for stinkytofu-opt.
 */
std::shared_ptr<stinkytofu::PassInstrumentation> createDebugPrintInstrumentation() {
    auto streams = std::make_shared<stinkytofu::DebugOutputStreams>();
    auto debugConfig = std::make_unique<stinkytofu::PassManagerDebugConfig>();
    debugConfig->setPrintBeforeAll(true);
    debugConfig->setPrintAfterAll(true);
    debugConfig->setDumpStreamBefore(streams->getOrCreate("before.txt"));
    debugConfig->setDumpStreamAfter(streams->getOrCreate("after.txt"));
    return std::make_shared<stinkytofu::DebugPrintInstrumentation>(std::move(debugConfig));
}

/**
 * Get default PassFeatureConfig configuration.
 */
stinkytofu::PassFeatureConfig getPassFeatureConfig() {
    stinkytofu::PassFeatureConfig config;
    config.loopConfig.unrollGemm = true;
    config.dagFeatures.distributeGlobalRead = true;
    return config;
}
