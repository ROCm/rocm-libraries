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

#include "stinkytofu/transforms/logical/LowerLogicalModulePipeline.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

namespace stinkytofu {

namespace {

// gfx1250 logical-IR migration (Step B): asm-layer port of rocisa's
// removeDuplicateAssignment (rocisa/src/pass/graph.cpp). On the native path that
// pass runs on the rocisa tree inside rocIsaPass, before toStinkyTofuModule; a
// mixed-tree logical leaf is converted from rocisa eagerly at build time, before
// rocIsaPass, so its s_mov_b32 opcodes never see that dedup and a redundant move
// survives into the leaf. Re-running the identical algorithm here, over the whole
// asm function once operands are concrete (after resolveSymbolicOperands above),
// drops exactly those leftover leaf duplicates. It is idempotent on code rocisa
// already deduplicated (native parts), so a kernel with no logical leaves is left
// byte-identical.
//
// Faithful transcription of rocisa's per-sgpr-lane tracking, linearised:
//   * a branch / call / label / s_wait resets every lane (value becomes unknown);
//   * s_mov_b32 s[i], V -> if lane i already holds V, drop the move (replacing it
//     with the same " (dup assign opt.)" comment line rocisa leaves when the move
//     carried a comment); either way lane i now holds V, and (as in rocisa) the
//     source lane of a reg-to-reg move is recorded as holding its own value;
//   * any other instruction that writes an sgpr invalidates those lanes.
// s_mov_b64 and every non-s_mov_b32 opcode are plain writers here, exactly as
// rocisa classifies them (only SMovB32 is the dedup-eligible kind).
void removeDuplicateSgprAssignment(Function& func) {
    // Value last written into each sgpr lane by an s_mov_b32; absent == unknown.
    std::unordered_map<uint32_t, StinkyRegister> assignValue;

    auto invalidateLanes = [&](const StinkyRegister& r) {
        if (!r.isRegister() || r.reg.type != RegType::S) return;
        for (uint32_t k = 0; k < r.reg.num; ++k) assignValue.erase(r.reg.idx + k);
    };

    for (BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end();) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            // Directives (.set / textblock / .if) carry no sgpr dataflow, matching
            // rocisa, which records neither in its graph.
            if (inst == nullptr) {
                ++it;
                continue;
            }

            // rocisa's Reset kind: control transfer and wait barriers clear every
            // tracked lane. (Call is added over rocisa's set purely as a safe
            // superset -- it only ever drops a would-be removal, never adds one.)
            if (isBranch(*inst) || isCall(*inst) || isLabel(*inst) || isWaitCnt(*inst)) {
                assignValue.clear();
                ++it;
                continue;
            }

            if (inst->getUnifiedOpcode() == GFX::s_mov_b32 && inst->getNumDestRegs() >= 1 &&
                inst->getNumSrcRegs() >= 1) {
                // Copy operands out before any erase below invalidates `inst`.
                const StinkyRegister dst = inst->getDestReg(0);
                const StinkyRegister src = inst->getSrcReg(0);
                bool erased = false;

                if (dst.isRegister() && dst.reg.type == RegType::S) {
                    const uint32_t lane = dst.reg.idx;
                    const auto held = assignValue.find(lane);
                    if (held != assignValue.end() && held->second == src) {
                        // Lane already holds this exact value: drop the duplicate move.
                        const CommentData* c = inst->getModifier<CommentData>();
                        if (c != nullptr && !c->comment.empty()) {
                            AsmDirective* dir = IRBase::createIR<AsmDirective>();
                            dir->kind = AsmDirectiveKind::TEXTBLOCK;
                            // Byte-for-byte the rocisa slash50(comment + " (dup assign opt.)").
                            dir->value = std::string(50, ' ') + " // " + c->comment +
                                         " (dup assign opt.)" + "\n";
                            bb.insertIR(it, dir);
                        }
                        it = bb.eraseIR(it);
                        erased = true;
                    }
                    assignValue[lane] = src;
                }
                // rocisa records the source lane of a reg-to-reg move as holding
                // its own value too; mirror it for exactness.
                if (src.isRegister() && src.reg.type == RegType::S)
                    for (uint32_t k = 0; k < src.reg.num; ++k) assignValue[src.reg.idx + k] = src;

                if (!erased) ++it;
                continue;
            }

            // Any other instruction writing an sgpr makes those lanes unknown.
            for (const StinkyRegister& d : inst->getDestRegs()) invalidateLanes(d);
            ++it;
        }
    }
}

}  // namespace

void runLogicalLoweringPipeline(Function& func, const GemmTileConfig& config,
                                const AsmCapsConfig& caps, bool removeDupAssign) {
    PassManager pm;
    pm.setGemmTileConfig(config);
    pm.setAsmCapsConfig(caps);
    pm.addPass(createCompositeInstructionLoweringPass());
    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    // gfx1250 logical-IR migration (Step A): now that ToStinkyAsmPass has lowered
    // the function to asm-form StinkyInstructions, resolve symbolic SGPR operand
    // names to their physical `.set` index -- the logical-layer counterpart of
    // rocisa's convertTextVariablesToRegisters, which runs on the rocisa tree
    // before toStinkyTofuModule on the native path. A mixed-tree logical leaf is
    // converted from rocisa eagerly at build time -- before that rocisa pass --
    // so its named SGPR operands still carry the unresolved placeholder index and
    // would otherwise emit as s[sgprName] instead of sN. Scoped to that
    // placeholder (resolveUnresolvedSentinel), so native operands (already
    // resolved) and real virtual registers are untouched: for a kernel with no
    // logical leaves this changes nothing, keeping the native path byte-identical.
    std::vector<SymbolicOperandFix> symbolicFixes;
    resolveSymbolicOperands(func, symbolicFixes, /*resolveUnresolvedSentinel=*/true);

    // gfx1250 logical-IR migration (Step B): with operands now concrete, drop the
    // duplicate s_mov_b32 assignments a logical leaf carried in (the rocisa dedup
    // pass could not see into the leaf). No-op on code rocisa already deduplicated,
    // so a leaf-free kernel stays byte-identical. Runs after resolveSymbolicOperands
    // so value comparisons see resolved indices, mirroring rocisa's pass order
    // (convertTextVariablesToRegisters before removeDuplicateAssignment). Gated on
    // removeDupAssign so a kernel where rocisa skips its own dedup (ActivationType
    // =="all") is not deduplicated here either.
    if (removeDupAssign) removeDuplicateSgprAssignment(func);
}

namespace {

GemmTileConfig configFromOptions(std::array<int, 3> arch,
                                 const StinkyAsmModule::ModuleOptions& opts) {
    GemmTileConfig cfg;
    cfg.arch = arch;
    // Carry the concrete stepping identity so ToStinkyAsmPass can disambiguate
    // steppings that share an ISA triple (e.g. gfx1250 vs gfx1250v0) and select
    // the matching per-arch cost table, mirroring rocisa's toStinkyTofuModule.
    cfg.archName = opts.ArchName;
    cfg.TileA0 = static_cast<uint32_t>(opts.TileA0);
    cfg.TileB0 = static_cast<uint32_t>(opts.TileB0);
    cfg.TileM0 = static_cast<uint32_t>(opts.TileM0);
    cfg.NumGRA = opts.NumGRA;
    cfg.NumGRB = opts.NumGRB;
    cfg.NumGRM = opts.NumGRM;
    cfg.NumWaves = static_cast<uint32_t>(opts.WaveGroup0 * opts.WaveGroup1);
    return cfg;
}

// Mirror of Backend::configurePassManager(): take the explicitly requested mode
// when TensileLite forwarded rocisa's asmCaps, else auto-probe via comgr. The
// asm-side pipeline derives its caps this way, and ToStinkyAsmPass must see the
// same verdict or a split ds_* half crossing a VGPR MSB bank loses its offset.
AsmCapsConfig capsFromOptions(std::array<int, 3> arch, const StinkyAsmModule::ModuleOptions& opts) {
    auto msbVal = opts.VgprMsbMode;
    if (msbVal < 0 || msbVal > static_cast<int>(VgprMsbMode::Msb16)) msbVal = 0;

    AsmCapsConfig caps;
    caps.vgprMsbMode = static_cast<VgprMsbMode>(msbVal);
    if (caps.vgprMsbMode == VgprMsbMode::None) {
        caps = ToolchainCaps::probe(resolveArchId(arch, opts.ArchName));
    }
    caps.requiresXCntForVolatileVMEM = opts.RequiresXCntForVolatileVMEM;
    caps.enableXnackReplay = opts.EnableXnackReplay;
    return caps;
}

}  // anonymous namespace

std::shared_ptr<StinkyAsmModule> lowerLogicalModuleToAsm(
    PyLogicalModule& module, std::array<int, 3> arch,
    const StinkyAsmModule::ModuleOptions& moduleOptions) {
    auto asmModule = std::make_shared<StinkyAsmModule>(module.getName(), arch, moduleOptions);

    // Register instruction groups from the target backend's pipeline.
    if (auto* pipeline = BackendRegistry::getArchPipeline(arch)) {
        for (const auto& groupName : pipeline->groupNames) {
            asmModule->addGroup(groupName);
        }
    }

    Function& func = asmModule->getFunction();
    BasicBlock* entryBB = func.getEntryBlock();
    assert(entryBB && "StinkyAsmModule must have an entry basic block");

    // Honor the concrete stepping identity (ModuleOptions.ArchName) when the two
    // steppings share an ISA triple (gfx1250 vs gfx1250v0), so the emitted asm
    // uses the correct per-arch cost table -- matching rocisa's toStinkyTofuModule.
    GfxArchID archId = resolveArchId(arch, moduleOptions.ArchName);

    {
        const auto& instructions = module.getInstructions();
        const auto& directives = module.getSetDirectives();
        const auto& conditionalDirectives = module.getConditionalDirectives();
        const auto& labels = module.getLabels();
        const auto& textBlocks = module.getTextBlocks();
        const auto& groupMarkers = module.getGroupMarkers();
        const auto& callableMarkers = module.getCallableMarkers();
        size_t dirIdx = 0;
        size_t condIdx = 0;
        size_t lblIdx = 0;
        size_t tbIdx = 0;
        size_t gmIdx = 0;
        size_t cmIdx = 0;

        // Active group names stack — tracks which groups the current
        // instruction position is inside.  Only groups that are also
        // registered (via addGroup above) will actually be updated.
        std::vector<std::string> activeGroups;
        std::vector<std::string> entryActiveGroups;
        std::string currentCallableName;

        BasicBlock* currentBB = entryBB;
        auto irBuilder = std::make_unique<AsmIRBuilder>(*currentBB, archId);
        std::vector<std::unique_ptr<PyLogicalFunction>> pyFunctions;
        pyFunctions.push_back(std::make_unique<PyLogicalFunction>(&func));

        // Process group markers whose position/order are eligible at the
        // current emission point.  Group markers toggle the active-group
        // stack but do not emit IR.
        auto processGroupMarkers = [&](size_t pos, size_t maxOrder) {
            while (gmIdx < groupMarkers.size() && groupMarkers[gmIdx].position <= pos &&
                   groupMarkers[gmIdx].order < maxOrder) {
                if (groupMarkers[gmIdx].isBegin) {
                    activeGroups.push_back(groupMarkers[gmIdx].name);
                } else {
                    auto it = std::find(activeGroups.rbegin(), activeGroups.rend(),
                                        groupMarkers[gmIdx].name);
                    if (it != activeGroups.rend()) {
                        activeGroups.erase(std::next(it).base());
                    }
                }
                ++gmIdx;
            }
        };

        // Build the pointer vector for updateInstructionGroups from active groups.
        auto buildGroupPtrs = [&]() -> std::vector<const std::string*> {
            std::vector<const std::string*> ptrs;
            ptrs.reserve(activeGroups.size());
            for (auto& g : activeGroups) {
                ptrs.push_back(&g);
            }
            return ptrs;
        };

        auto emitNextItem = [&](int type) {
            const auto instsCountBefore = currentBB->size();
            switch (type) {
                case 0: {
                    AsmDirective* dir = IRBase::createIR<AsmDirective>();
                    dir->kind = AsmDirectiveKind::SET;
                    dir->name = ".set";
                    dir->symbol = directives[dirIdx].symbol;
                    dir->value = directives[dirIdx].value;
                    currentBB->appendIR(dir);
                    ++dirIdx;
                    break;
                }
                case 1: {
                    StinkyInstruction* labelInst =
                        irBuilder->createLabel(labels[lblIdx].labelName, labels[lblIdx].alignment);
                    if (!labels[lblIdx].comment.empty()) {
                        labelInst->addModifier<CommentData>(CommentData{labels[lblIdx].comment});
                    }
                    ++lblIdx;
                    break;
                }
                case 2: {
                    AsmDirective* dir = IRBase::createIR<AsmDirective>();
                    dir->kind = AsmDirectiveKind::TEXTBLOCK;
                    dir->value = textBlocks[tbIdx].text;
                    currentBB->appendIR(dir);
                    ++tbIdx;
                    break;
                }
                case 3: {
                    const auto& marker = callableMarkers[cmIdx++];
                    if (marker.isBegin) {
                        assert(currentBB == entryBB && "nested callable functions are unsupported");
                        irBuilder->createFunctionAsmPlacementMarker(marker.name);
                        asmModule->updateInstructionGroups(buildGroupPtrs(), instsCountBefore);
                        entryActiveGroups = activeGroups;
                        activeGroups.clear();
                        currentCallableName = marker.name;
                        Function& callable = asmModule->createFunction(marker.name, true);
                        currentBB = callable.getEntryBlock();
                        assert(currentBB && "callable Function must have an entry basic block");
                        pyFunctions.push_back(std::make_unique<PyLogicalFunction>(&callable));
                        irBuilder = std::make_unique<AsmIRBuilder>(*currentBB, archId);
                    } else {
                        assert(currentBB != entryBB &&
                               "callable end marker without a begin marker");
                        assert(marker.name == currentCallableName &&
                               "mismatched callable end marker");
                        currentBB = entryBB;
                        activeGroups = entryActiveGroups;
                        entryActiveGroups.clear();
                        currentCallableName.clear();
                        irBuilder = std::make_unique<AsmIRBuilder>(*currentBB, archId);
                    }
                    return;
                }
                case 4: {
                    const auto& entry = conditionalDirectives[condIdx++];
                    AsmDirective* dir = IRBase::createIR<AsmDirective>();
                    if (entry.kind == ConditionalDirectiveKind::IF) {
                        dir->kind = AsmDirectiveKind::IF;
                        dir->name = ".if";
                        dir->condition = entry.payload;
                    } else {
                        dir->kind = AsmDirectiveKind::ENDIF;
                        dir->name = ".endif";
                        dir->comment = entry.payload;
                    }
                    currentBB->appendIR(dir);
                    break;
                }
                default:
                    break;
            }
            if (currentBB == entryBB) {
                asmModule->updateInstructionGroups(buildGroupPtrs(), instsCountBefore);
            }
        };

        auto emitItemsAtPosition = [&](size_t pos) {
            while (true) {
                size_t bestOrder = SIZE_MAX;
                int bestType = -1;

                if (dirIdx < directives.size() && directives[dirIdx].position <= pos &&
                    directives[dirIdx].order < bestOrder) {
                    bestOrder = directives[dirIdx].order;
                    bestType = 0;
                }
                if (lblIdx < labels.size() && labels[lblIdx].position <= pos &&
                    labels[lblIdx].order < bestOrder) {
                    bestOrder = labels[lblIdx].order;
                    bestType = 1;
                }
                if (tbIdx < textBlocks.size() && textBlocks[tbIdx].position <= pos &&
                    textBlocks[tbIdx].order < bestOrder) {
                    bestOrder = textBlocks[tbIdx].order;
                    bestType = 2;
                }
                if (cmIdx < callableMarkers.size() && callableMarkers[cmIdx].position <= pos &&
                    callableMarkers[cmIdx].order < bestOrder) {
                    bestOrder = callableMarkers[cmIdx].order;
                    bestType = 3;
                }
                if (condIdx < conditionalDirectives.size() &&
                    conditionalDirectives[condIdx].position <= pos &&
                    conditionalDirectives[condIdx].order < bestOrder) {
                    bestOrder = conditionalDirectives[condIdx].order;
                    bestType = 4;
                }

                if (bestType == -1) break;
                // Process any group markers that precede this item.
                processGroupMarkers(pos, bestOrder);
                emitNextItem(bestType);
            }
        };

        for (size_t i = 0; i < instructions.size(); ++i) {
            emitItemsAtPosition(i);
            // Process group markers at this instruction position.
            processGroupMarkers(i, SIZE_MAX);

            const auto instsCountBefore = currentBB->size();
            currentBB->appendIR(static_cast<IRBase*>(instructions[i].get()));
            if (currentBB == entryBB) {
                asmModule->updateInstructionGroups(buildGroupPtrs(), instsCountBefore);
            }
        }
        // Trailing items (after all instructions)
        emitItemsAtPosition(SIZE_MAX);
        // Process any remaining group markers.
        processGroupMarkers(SIZE_MAX, SIZE_MAX);

        // Debug: report instruction group ranges after population.
        if (std::getenv("DEBUG_STINKY_GROUPS")) {
            auto* pipeline = BackendRegistry::getArchPipeline(arch);
            if (pipeline) {
                std::cerr << "[DEBUG_STINKY_GROUPS] Instruction groups after population:\n";
                for (const auto& groupName : pipeline->groupNames) {
                    auto range = asmModule->findGroupRange(groupName);
                    if (range) {
                        size_t count = 0;
                        for (auto it = range->first; it != range->second; ++it) ++count;
                        ++count;  // include last
                        std::cerr << "  " << groupName << ": populated (" << count
                                  << " instructions)\n";
                    } else {
                        std::cerr << "  " << groupName << ": EMPTY (no range)\n";
                    }
                }
                std::cerr << "  Total BB size: " << entryBB->size() << "\n";
            }
        }

        assert(currentBB == entryBB && "unterminated callable function body");
        const GemmTileConfig loweringConfig = configFromOptions(arch, moduleOptions);
        const AsmCapsConfig loweringCaps = capsFromOptions(arch, moduleOptions);
        for (Function* function : asmModule->getFunctions()) {
            runLogicalLoweringPipeline(*function, loweringConfig, loweringCaps,
                                       moduleOptions.RemoveDupAssign);
        }
    }

    return asmModule;
}

}  // namespace stinkytofu
