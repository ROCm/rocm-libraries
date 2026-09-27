// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/ir/DumpMemTokenIRStructurePass.hpp"

#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

namespace stinkytofu {
namespace {

/// An instruction is kept iff it is a memtoken CANDIDATE — a kind that participates
/// in wave-group structure — regardless of whether a MemTokenData is actually
/// attached. Keying on presence would drop the very instructions that matter when
/// untagged: e.g. a source-fence `s_wait_tensorcnt 0` outside the region-tagged pass
/// has no memtoken but is exactly what the reader needs to reason about FIFO drains.
/// Covered: TDM loads, tensorcnt/other waits, ds read/write/atomic, barriers.
bool isStructural(const StinkyInstruction& inst) {
    return isTensorLoad(inst) || inst.is(InstFlag::IF_WaitTensorCnt) || isWaitCnt(inst) ||
           isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst) || isBarrier(inst);
}

/// Trailing comment of an instruction ("" if none).
std::string commentOf(const StinkyInstruction& inst) {
    const auto* c = inst.getModifier<CommentData>();
    return c ? c->comment : std::string{};
}

void dumpFunction(std::ostream& os, const Function& func) {
    os << "st.func @" << func.getName() << " {\n";
    for (const BasicBlock& bb : func) {
        os << "  ^" << bb.getLabel() << ":\n";
        for (const auto& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr || !isStructural(*inst)) continue;
            // Reuse AsmPrinter for the op line (mnemonic + operands + memtoken); it
            // ends in a newline and omits comments, so trim the newline and append
            // the comment ourselves so each instruction is one line.
            std::string line = toString(*inst);
            while (!line.empty() && (line.back() == '\n' || line.back() == ' ')) line.pop_back();
            os << "    " << line;
            const std::string comment = commentOf(*inst);
            if (!comment.empty()) os << "  // " << comment;
            os << "\n";
        }
        const auto& succs = bb.getSuccessors();
        if (!succs.empty()) {
            os << "    -> ";
            for (size_t i = 0; i < succs.size(); ++i) {
                if (i > 0) os << ", ";
                os << "^" << succs[i]->getLabel();
            }
            os << "\n";
        }
    }
    os << "}\n";
}

class DumpMemTokenIRStructurePass : public Pass {
   public:
    static char ID;

    explicit DumpMemTokenIRStructurePass(DumpMemTokenIRStructurePassConfig config)
        : config_(std::move(config)) {}

    DumpMemTokenIRStructurePass(const StinkyAsmModule& module,
                                DumpMemTokenIRStructurePassConfig config)
        : module_(&module), config_(std::move(config)) {}

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return "DumpMemTokenIRStructurePass";
    }

    // With a module, dump every function; without one, dump just the function the
    // PassManager hands us (opt-tool / per-function use). Both cover the full picture
    // for the module hook, which is the intended before/after bracket.
    PreservedAnalyses run(Function& func, PassContext& /*ctx*/, AnalysisManager& /*AM*/) override {
        if (config_.path.empty()) return PreservedAnalyses::all();
        std::ofstream out(config_.path, std::ios::out | std::ios::trunc);
        if (out) {
            if (module_ != nullptr) {
                const auto functions = module_->getFunctions();
                for (size_t i = 0; i < functions.size(); ++i) {
                    if (i > 0) out << "\n";
                    dumpFunction(out, *functions[i]);
                }
            } else {
                dumpFunction(out, func);
            }
        }
        return PreservedAnalyses::all();
    }

   private:
    const StinkyAsmModule* module_ = nullptr;
    DumpMemTokenIRStructurePassConfig config_;
};

char DumpMemTokenIRStructurePass::ID = 0;
}  // namespace

std::unique_ptr<Pass> createDumpMemTokenIRStructurePass(const StinkyAsmModule& module,
                                                        DumpMemTokenIRStructurePassConfig config) {
    return std::make_unique<DumpMemTokenIRStructurePass>(module, std::move(config));
}

std::unique_ptr<Pass> createDumpMemTokenIRStructurePass(DumpMemTokenIRStructurePassConfig config) {
    return std::make_unique<DumpMemTokenIRStructurePass>(std::move(config));
}
}  // namespace stinkytofu
