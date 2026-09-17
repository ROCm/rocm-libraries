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

#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

namespace stinkytofu {

void runLogicalLoweringPipeline(Function& func, const GemmTileConfig& config,
                                const AsmCapsConfig& caps) {
    PassManager pm;
    pm.setGemmTileConfig(config);
    pm.setAsmCapsConfig(caps);
    pm.addPass(createCompositeInstructionLoweringPass());
    pm.addPass(createToStinkyAsmPass());
    pm.run(func);
}

namespace {

GemmTileConfig configFromOptions(std::array<int, 3> arch,
                                 const StinkyAsmModule::ModuleOptions& opts) {
    GemmTileConfig cfg;
    cfg.arch = arch;
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
AsmCapsConfig capsFromOptions(std::array<int, 3> arch,
                              const StinkyAsmModule::ModuleOptions& opts) {
    auto msbVal = opts.VgprMsbMode;
    if (msbVal < 0 || msbVal > static_cast<int>(VgprMsbMode::Msb16)) msbVal = 0;

    AsmCapsConfig caps;
    caps.vgprMsbMode = static_cast<VgprMsbMode>(msbVal);
    if (caps.vgprMsbMode == VgprMsbMode::None) {
        caps = ToolchainCaps::probe(getGfxArchID(arch[0], arch[1], arch[2]));
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

    GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

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
                        assert(currentBB != entryBB && "callable end marker without a begin marker");
                        assert(marker.name == currentCallableName && "mismatched callable end marker");
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
            runLogicalLoweringPipeline(*function, loweringConfig, loweringCaps);
        }
    }

    return asmModule;
}

}  // namespace stinkytofu
