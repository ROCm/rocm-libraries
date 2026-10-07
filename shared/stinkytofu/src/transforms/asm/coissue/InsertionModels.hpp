// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Insertion models: what each pass after the co-issue repair will insert, predicted for an
// instruction order that is not in the IR yet. Each model runs its pass's own planner:
//   VgprMsbModel      planVgprMsb               (InsertVgprMsbPass)
//   BridgeModel       planPrefetchBridge        (PrefetchBridgeSubstitutionPass)
//   WaitAluModel      WaitAluTracker            (InsertWaitAluPass)
//   CoexecNopModel    planCoexecNops            (InsertCoexecHazardPass)
// so model output equals pass output by construction; InsertionModelTest checks it on the
// passes' filecheck inputs. InsertDelayAluPass is not modeled: it emits nothing on the
// kernels the repair targets (fewer than 2 waves per SIMD).

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/transforms/asm/CoexecNopPlanner.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

namespace stinkytofu {
class BasicBlock;
class PassContext;
struct HWModel;
struct StinkyInstruction;
}  // namespace stinkytofu

namespace stinkytofu::coissue {

/// The instructions the models add, owned by a scratch function. One object per distinct
/// instruction, so a prediction allocates nothing once the pool is warm.
class InstructionPool {
   public:
    explicit InstructionPool(GfxArchID arch);
    const StinkyInstruction* bankSwitch(int immediate, int requiredMsb);
    const StinkyInstruction* nop0();
    const StinkyInstruction* vnop();
    const StinkyInstruction* waitAlu(int vaVdst, int vmVsrc, int holdCnt);
    /// `prefetch` as PrefetchBridgeSubstitutionPass rewrites it to flat_prefetch.
    const StinkyInstruction* flatBridge(const StinkyInstruction& prefetch);
    /// True if `inst` came from this pool.
    bool owns(const StinkyInstruction* inst) const {
        return owned_.count(inst) != 0;
    }

   private:
    StinkyInstruction* create(int opcode);

    GfxArchID arch_;
    Function scratch_;
    BasicBlock* block_ = nullptr;
    std::map<int, const StinkyInstruction*> bankSwitches_;
    std::map<std::tuple<int, int, int>, const StinkyInstruction*> waitAlus_;
    std::unordered_map<const StinkyInstruction*, const StinkyInstruction*> bridges_;
    const StinkyInstruction* nop0_ = nullptr;
    const StinkyInstruction* vnop_ = nullptr;
    std::unordered_set<const StinkyInstruction*> owned_;
};

/// What the later passes insert, counted over the predicted blocks.
struct InsertedCounts {
    int bankSwitches = 0;  ///< s_set_vgpr_msb
    int waitAlus = 0;      ///< s_wait_alu
    int nops = 0;          ///< s_nop and v_nop
    int bridges = 0;       ///< global_prefetch rewritten to flat_prefetch

    bool operator==(const InsertedCounts&) const = default;
};

/// A scope of blocks as the later passes will leave them.
struct PredictedBlock {
    std::vector<const BasicBlock*> blocks;  ///< layout order
    std::vector<std::vector<const StinkyInstruction*>> seqs;
    InsertedCounts counts;
};

class InsertionModel {
   public:
    virtual ~InsertionModel() = default;
    virtual const char* name() const = 0;
    /// Insert into `block` exactly what the model's pass would.
    virtual void apply(PredictedBlock& block) = 0;
};

/// The single models, for testing each against its pass. `scope` are blocks of `func` in
/// layout order; the models read every other block in its IR order.
std::unique_ptr<InsertionModel> makeVgprMsbModel(VgprMsbMode mode, InstructionPool& pool);
std::unique_ptr<InsertionModel> makeBridgeModel(Function& func,
                                                const std::vector<const BasicBlock*>& scope,
                                                int required, InstructionPool& pool);
std::unique_ptr<InsertionModel> makeWaitAluModel(const PassContext& passCtx,
                                                 InsertWaitAluOptions opts, InstructionPool& pool);
std::unique_ptr<InsertionModel> makeCoexecNopModel(Function& func, const HWModel& hw,
                                                   InstructionPool& pool);

/// The models the gfx1250 pipeline runs after the repair, in pipeline order.
struct InsertionConfig {
    VgprMsbMode msbMode = VgprMsbMode::None;
    /// InsertWaitAlu and the prefetch bridge only run with expert scheduling mode 2.
    bool esm2 = false;
    InsertWaitAluOptions waitAlu;
};

class InsertionPipeline {
   public:
    /// `scope` are blocks of `func` in layout order; every other block keeps its IR order.
    InsertionPipeline(Function& func, std::vector<const BasicBlock*> scope,
                      const PassContext& passCtx, const InsertionConfig& config);
    ~InsertionPipeline();

    /// The scope with everything the later passes insert, for `orders` (one per block).
    PredictedBlock predict(const std::vector<std::vector<const StinkyInstruction*>>& orders);

    InstructionPool& pool() {
        return *pool_;
    }
    const std::vector<std::unique_ptr<InsertionModel>>& models() const {
        return models_;
    }

   private:
    std::vector<const BasicBlock*> scope_;
    std::unique_ptr<InstructionPool> pool_;
    std::vector<std::unique_ptr<InsertionModel>> models_;
};

}  // namespace stinkytofu::coissue
