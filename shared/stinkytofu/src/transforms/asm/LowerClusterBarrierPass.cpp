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
#include "stinkytofu/transforms/asm/LowerClusterBarrierPass.hpp"

#include <cstdint>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"

namespace stinkytofu {
namespace {

/// Cluster-scope split-barrier literal id (`s_barrier_signal/wait -3`).
constexpr int kClusterBarrierId = -3;
/// Symbolic SGPR holding the wave index; only wave 0 issues the cluster handshake.
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
/// Prefixes for the WaveIdx-gated skip labels. Signal and wait use distinct prefixes so a
/// SignalWait placeholder can emit two independently-guarded blocks with unique labels.
constexpr const char* kSkipSignalLabelPrefix = "label_skipCBPreSignal_";
constexpr const char* kSkipWaitLabelPrefix = "label_skipCBPreWait_";

/// Single-dword symbolic SGPR reference, emitted as `s[<name>]`.
StinkyRegister makeSymbolicSgpr(const std::string& name) {
    StinkyRegister reg(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
    reg.setSymbolicName(name);
    return reg;
}

/// Expand one PSEUDO_CLUSTER_BARRIER (\p pseudo) in place: insert the concrete
/// handshake immediately before it. The caller erases the placeholder.
void expandPseudoClusterBarrier(StinkyInstruction* pseudo, AsmIRBuilder& irBuilder,
                                GfxArchID archId, uint64_t gen) {
    const auto* data = pseudo->getModifier<PseudoClusterBarrierData>();
    const PseudoClusterBarrierData::Kind kind =
        data ? data->kind : PseudoClusterBarrierData::Kind::SignalWait;

    const HwInstDesc* cmpDesc = getMCIDByUOp(GFX::s_cmp_eq_u32, archId);
    const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_cbranch_scc0, archId);
    const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
    assert(cmpDesc && brDesc && signalDesc && waitDesc &&
           "Cluster-barrier opcodes are not supported on this architecture");

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};

    // Emit one WaveIdx-gated cluster handshake block (signal or wait). Only wave 0 issues
    // the cluster-scope barrier, so both halves carry the same guard:
    //   s_cmp_eq_u32 s[sgprWaveIdx], 0   -- SCC = (WaveIdx == 0), clobbers SCC
    //   s_cbranch_scc0 <label>           -- non-zero waves skip, reads SCC
    //   s_barrier_signal/wait -3
    //   <label>:
    // The cmp/cbranch make the block a self-contained SCC def->use; the scheduler's SCC
    // self-contained gate keeps any other live SCC range from straddling either block.
    auto emitGuardedBarrier = [&](const HwInstDesc* barrierDesc, const std::string& labelName,
                                  const char* barrierComment, const char* branchComment) {
        StinkyInstruction* cmpInst = irBuilder.create(cmpDesc, pseudo);
        cmpInst->addSrcReg(makeSymbolicSgpr(kWaveIdxSymbol));
        cmpInst->addSrcReg(StinkyRegister(0));
        cmpInst->addModifier<CommentData>(CommentData{"Check for waveID 0"});

        StinkyInstruction* brInst = irBuilder.create(brDesc, pseudo);
        brInst->addSrcReg(StinkyRegister(labelName));
        brInst->addModifier<LabelData>(LabelData{labelName});
        brInst->addModifier<CommentData>(CommentData{branchComment});

        StinkyInstruction* barInst = irBuilder.create(barrierDesc, pseudo);
        barInst->addSrcReg(StinkyRegister(kClusterBarrierId));
        barInst->addModifier<CommentData>(CommentData{barrierComment});

        StinkyInstruction* lblInst = irBuilder.create(&labelMCID, pseudo);
        lblInst->addModifier<LabelData>(LabelData{labelName, /*alignment=*/1});
    };

    const bool emitSignal = kind != PseudoClusterBarrierData::Kind::WaitOnly;
    const bool emitWait = kind != PseudoClusterBarrierData::Kind::SignalOnly;

    if (emitSignal) {
        emitGuardedBarrier(signalDesc, std::string(kSkipSignalLabelPrefix) + std::to_string(gen),
                           "cluster_barrier signal",
                           "Execute cluster barrier signal for waveID 0");
    }

    if (emitWait) {
        emitGuardedBarrier(waitDesc, std::string(kSkipWaitLabelPrefix) + std::to_string(gen),
                           "cluster barrier wait", "Execute cluster barrier wait for waveID 0");
    }
}

class LowerClusterBarrierPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "Lower Cluster Barrier";
    }

    Pass::ID getPassID() const override {
        return &LowerClusterBarrierPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        bool changed = false;
        for (BasicBlock& bb : func) {
            AsmIRBuilder irBuilder(bb, archId);
            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst != nullptr && isPseudoClusterBarrier(*inst)) {
                    expandPseudoClusterBarrier(inst, irBuilder, archId, labelCounter_++);
                    it = bb.eraseIR(it);  // drop the placeholder, advance
                    changed = true;
                } else {
                    ++it;
                }
            }
        }

        return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }

   private:
    // Monotonic across functions so every skip label is unique in the kernel.
    uint64_t labelCounter_ = 0;
};

char LowerClusterBarrierPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createLowerClusterBarrierPass() {
    return std::make_unique<LowerClusterBarrierPassImpl>();
}

}  // namespace stinkytofu
