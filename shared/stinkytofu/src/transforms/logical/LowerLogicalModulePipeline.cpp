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

#include <cassert>

#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

namespace stinkytofu {

void runLogicalLoweringPipeline(Function& func, const GemmTileConfig& config) {
    PassManager pm;
    pm.setGemmTileConfig(config);
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

}  // anonymous namespace

std::shared_ptr<StinkyAsmModule> lowerLogicalModuleToAsm(
    PyLogicalModule& module, std::array<int, 3> arch,
    const StinkyAsmModule::ModuleOptions& moduleOptions) {
    auto asmModule = std::make_shared<StinkyAsmModule>(module.getName(), arch, moduleOptions);

    Function& func = asmModule->getFunction();
    BasicBlock* entryBB = func.getEntryBlock();
    assert(entryBB && "StinkyAsmModule must have an entry basic block");

    GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

    {
        PyLogicalFunction pyFunc(&func);

        const auto& instructions = module.getInstructions();
        const auto& directives = module.getSetDirectives();
        const auto& labels = module.getLabels();
        size_t dirIdx = 0;
        size_t lblIdx = 0;

        AsmIRBuilder irBuilder(*entryBB, archId);

        for (size_t i = 0; i < instructions.size(); ++i) {
            // Insert any .set directives whose position <= current instruction index
            while (dirIdx < directives.size() && directives[dirIdx].position <= i) {
                AsmDirective* dir = IRBase::createIR<AsmDirective>();
                dir->kind = AsmDirectiveKind::SET;
                dir->name = ".set";
                dir->symbol = directives[dirIdx].symbol;
                dir->value = directives[dirIdx].value;
                entryBB->appendIR(dir);
                ++dirIdx;
            }
            // Insert any labels whose position <= current instruction index
            while (lblIdx < labels.size() && labels[lblIdx].position <= i) {
                StinkyInstruction* labelInst =
                    irBuilder.createLabel(labels[lblIdx].labelName, labels[lblIdx].alignment);
                if (!labels[lblIdx].comment.empty()) {
                    labelInst->addModifier<CommentData>(CommentData{labels[lblIdx].comment});
                }
                ++lblIdx;
            }
            entryBB->appendIR(static_cast<IRBase*>(instructions[i].get()));
        }
        // Trailing .set directives (after all instructions)
        while (dirIdx < directives.size()) {
            AsmDirective* dir = IRBase::createIR<AsmDirective>();
            dir->kind = AsmDirectiveKind::SET;
            dir->name = ".set";
            dir->symbol = directives[dirIdx].symbol;
            dir->value = directives[dirIdx].value;
            entryBB->appendIR(dir);
            ++dirIdx;
        }
        // Trailing labels (after all instructions)
        while (lblIdx < labels.size()) {
            StinkyInstruction* labelInst =
                irBuilder.createLabel(labels[lblIdx].labelName, labels[lblIdx].alignment);
            if (!labels[lblIdx].comment.empty()) {
                labelInst->addModifier<CommentData>(CommentData{labels[lblIdx].comment});
            }
            ++lblIdx;
        }

        runLogicalLoweringPipeline(func, configFromOptions(arch, moduleOptions));
    }

    return asmModule;
}

}  // namespace stinkytofu
