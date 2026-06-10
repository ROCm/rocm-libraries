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

#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"

#define DEBUG_TYPE "InsertDelayAluPass"

// ---------------------------------------------------------------------------
// S_CLAUSE interaction
// ---------------------------------------------------------------------------
// Emitting s_delay_alu inside a clause has no effect because the hardware
// does not allow other waves to interleave during a clause.
//
// This pass does not explicitly track clause regions. It works correctly for
// memory clauses because:
//   1. s_clause itself -> getDelayType() returns OTHER -> skipped.
//   2. Memory ops inside the clause -> instructionWaitsForVALU() clears all
//      VALU/TRANS/SALU state, so no s_delay_alu is emitted between them.
//
// TODO: if VALU clause support is added in the future, this pass would need
// clause-region awareness to suppress s_delay_alu emission inside them.
// ---------------------------------------------------------------------------

namespace {
using namespace stinkytofu;

static const char* delayTypeName(int type) {
    static const char* names[] = {"VALU", "TRANS", "SALU", "OTHER"};
    return names[type];
}

// ---------------------------------------------------------------------------
// Delay type classification (maps to scoreboard entries)
// ---------------------------------------------------------------------------
enum DelayType { VALU, TRANS, SALU, OTHER };

DelayType getDelayType(const StinkyInstruction& inst) {
    // 16/32-bit transcendentals -> TRANS pipe
    if (isTranscendental(inst) && !isTrans64(inst)) return TRANS;
    // XDL WMMA/SWMMAC -> TRANS pipe
    if (isXDLWMMA(inst)) return TRANS;
    // Non-XDL WMMA/SWMMA/MXWMMA -> VALU pipe
    if (isWMMA(inst) || isSWMMA(inst) || isMXWMMA(inst)) return VALU;
    // Regular VALU, 64-bit trans -> VALU pipe
    if (isVectorALU(inst) || isTrans64(inst)) return VALU;
    // SALU
    if (isScalarALU(inst)) return SALU;
    return OTHER;
}

// Return the number of wait states (elapsed cycles) for an instruction.
// s_nop N occupies N+1 cycles; all other instructions occupy 1 cycle.
unsigned getNumWaitStates(const StinkyInstruction& inst) {
    if (inst.getHwInstDesc()->unifiedOpcode == GFX::s_nop) {
        const auto& srcs = inst.getSrcRegs();
        if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt)
            return static_cast<unsigned>(srcs[0].getLiteralInt()) + 1;
    }
    return 1;
}

// Instructions that implicitly wait for VA_VDST==0 before issuing.
// All outstanding VALU results are guaranteed complete after these.
bool instructionWaitsForVALU(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst) || isFLATLoad(inst) ||
           isFLATStore(inst) || isFLATAtomic(inst) || isMUBUFLoad(inst) || isMUBUFStore(inst) ||
           isMUBUFAtomic(inst) || isGLOBALLoad(inst) || isGLOBALStore(inst) || isTensorLoad(inst);
}

// ---------------------------------------------------------------------------
// Per-register delay info
// ---------------------------------------------------------------------------
struct DelayInfo {
    static constexpr unsigned VALU_MAX = 5;         // scoreboard: 4 entries (DEP_1..4)
    static constexpr unsigned TRANS_MAX = 4;        // scoreboard: 3 entries (DEP_1..3)
    static constexpr unsigned SALU_CYCLES_MAX = 4;  // SALU_CYCLE_1..3

    uint8_t VALUCycles = 0;
    uint8_t VALUNum = VALU_MAX;

    uint8_t TRANSCycles = 0;
    uint8_t TRANSNum = TRANS_MAX;
    uint8_t TRANSNumVALU = VALU_MAX;  // VALU count since TRANS (for encoding priority)

    uint8_t SALUCycles = 0;

    DelayInfo() = default;

    DelayInfo(DelayType type, unsigned cycles) {
        switch (type) {
            case VALU:
                VALUCycles = cycles;
                VALUNum = 0;
                break;
            case TRANS:
                TRANSCycles = cycles;
                TRANSNum = 0;
                TRANSNumVALU = 0;
                break;
            case SALU:
                SALUCycles = std::min(cycles, SALU_CYCLES_MAX);
                break;
            default:
                break;
        }
    }

    bool operator==(const DelayInfo& rhs) const {
        return VALUCycles == rhs.VALUCycles && VALUNum == rhs.VALUNum &&
               TRANSCycles == rhs.TRANSCycles && TRANSNum == rhs.TRANSNum &&
               TRANSNumVALU == rhs.TRANSNumVALU && SALUCycles == rhs.SALUCycles;
    }

    bool operator!=(const DelayInfo& rhs) const {
        return !(*this == rhs);
    }

    // Merge another DelayInfo, taking worst-case (smallest position / largest cycles).
    void merge(const DelayInfo& rhs) {
        VALUCycles = std::max(VALUCycles, rhs.VALUCycles);
        VALUNum = std::min(VALUNum, rhs.VALUNum);
        TRANSCycles = std::max(TRANSCycles, rhs.TRANSCycles);
        TRANSNum = std::min(TRANSNum, rhs.TRANSNum);
        TRANSNumVALU = std::min(TRANSNumVALU, rhs.TRANSNumVALU);
        SALUCycles = std::max(SALUCycles, rhs.SALUCycles);
    }

    // Advance after issuing an instruction. Returns true if entry can be erased.
    bool advance(DelayType type, unsigned cycles) {
        bool erase = true;

        VALUNum += (type == VALU);
        if (VALUNum >= VALU_MAX || VALUCycles <= cycles) {
            VALUNum = VALU_MAX;
            VALUCycles = 0;
        } else {
            VALUCycles -= cycles;
            erase = false;
        }

        TRANSNum += (type == TRANS);
        TRANSNumVALU += (type == VALU);
        if (TRANSNum >= TRANS_MAX || TRANSCycles <= cycles) {
            TRANSNum = TRANS_MAX;
            TRANSNumVALU = VALU_MAX;
            TRANSCycles = 0;
        } else {
            TRANSCycles -= cycles;
            erase = false;
        }

        if (SALUCycles <= cycles) {
            SALUCycles = 0;
        } else {
            SALUCycles -= cycles;
            erase = false;
        }

        return erase;
    }
};

// ---------------------------------------------------------------------------
// Per-BB delay state: register -> DelayInfo
// ---------------------------------------------------------------------------
struct DelayState : RegKeyMap<DelayInfo> {
    // Merge another state (worst-case per register).
    void merge(const DelayState& rhs) {
        for (const auto& [key, info] : rhs) {
            auto [it, inserted] = insert({key, info});
            if (!inserted) it->second.merge(info);
        }
    }

    // Advance all entries after issuing an instruction. Erase expired entries.
    void advance(DelayType type, unsigned cycles) {
        for (auto it = begin(); it != end();) {
            if (it->second.advance(type, cycles))
                it = erase(it);
            else
                ++it;
        }
    }
};

// ---------------------------------------------------------------------------
// InsertDelayAlu pass
// ---------------------------------------------------------------------------
class InsertDelayAluPassImpl : public Pass {
    // Saved delay state at exit of each basic block (for cross-BB propagation).
    std::unordered_map<BasicBlock*, DelayState> blockExitState;

    int minWavesPerSimd_;

    // delay_alu only pays off when sibling waves exist to hide ALU latency.
    // Missing metadata => false (run the pass).
    //
    // Assumes static VGPR allocation (granule is fixed per arch). Dynamic VGPR
    // kernels (`.dynamic_vgpr_en true`) would need separate handling.
    //
    // Note: this only reflects what codegen can see (per-wave VGPR allocation).
    // Whether the actual dispatch fills the GPU depends on runtime problem size
    // (M, N, batch), which codegen does not have — small GEMMs may still under-
    // utilize even when this predicate says occupancy is fine.
    bool shouldSkipForLowOccupancy(const Function& func, GfxArchID archId) const {
        auto kernelVgprs = func.getMetaData(kSigTotalVgprsMetaKey);
        if (!kernelVgprs || *kernelVgprs == 0) return false;

        const int waves = getWavesPerSimd(archId, static_cast<int>(*kernelVgprs));
        if (waves >= minWavesPerSimd_) return false;

        PASS_DEBUG(std::cerr << "[DelayAlu] skipping: kernelVgprs=" << *kernelVgprs
                             << " -> wavesPerSimd=" << waves
                             << " < minWavesPerSimd=" << minWavesPerSimd_ << "\n");
        return true;
    }

   public:
    explicit InsertDelayAluPassImpl(int minWavesPerSimd) : minWavesPerSimd_(minWavesPerSimd) {}

    // Emit s_delay_alu or pack into a previous one.
    // Returns the last s_delay_alu that still has room for packing (instid1 slot empty),
    // or nullptr if no packing is possible.
    StinkyInstruction* emitDelayAlu(BasicBlock& bb, StinkyInstruction* insertBefore,
                                    const DelayInfo& delay, StinkyInstruction* lastDelayAlu,
                                    AsmIRBuilder& irBuilder, GfxArchID archId) {
        SDelayAluData::InstType id0Type = SDelayAluData::InstType::NO_DEP;
        int8_t id0Dist = 0;
        SDelayAluData::InstType id1Type = SDelayAluData::InstType::NO_DEP;
        int8_t id1Dist = 0;
        bool hasId1 = false;

        // TRANS dep -> first slot
        if (delay.TRANSNum < DelayInfo::TRANS_MAX) {
            id0Type = SDelayAluData::InstType::TRANS;
            id0Dist = static_cast<int8_t>(delay.TRANSNum);
        }

        // VALU dep -> first or second slot (first if no TRANS, second if TRANS present).
        // Only encode if VALU is more recent than any TRANS dep we're also waiting on.
        if (delay.VALUNum < DelayInfo::VALU_MAX && delay.VALUNum <= delay.TRANSNumVALU) {
            if (id0Dist != 0) {
                // TRANS already in id0 -> put VALU in id1
                id1Type = SDelayAluData::InstType::VALU;
                id1Dist = static_cast<int8_t>(delay.VALUNum);
                hasId1 = true;
            } else {
                id0Type = SDelayAluData::InstType::VALU;
                id0Dist = static_cast<int8_t>(delay.VALUNum);
            }
        }

        // SALU dep -> fills remaining slot
        if (delay.SALUCycles > 0) {
            assert(delay.SALUCycles < DelayInfo::SALU_CYCLES_MAX);
            if (hasId1) {
                // Both slots used (TRANS + VALU), drop SALU
            } else if (id0Dist != 0) {
                // id0 used, put SALU in id1
                id1Type = SDelayAluData::InstType::SALU;
                id1Dist = static_cast<int8_t>(delay.SALUCycles);
                hasId1 = true;
            } else {
                id0Type = SDelayAluData::InstType::SALU;
                id0Dist = static_cast<int8_t>(delay.SALUCycles);
            }
        }

        // Nothing to emit
        if (id0Dist == 0) return lastDelayAlu;

        // Try packing into previous s_delay_alu's instid1 slot.
        if (!hasId1 && lastDelayAlu) {
            auto* prevData = lastDelayAlu->getModifier<SDelayAluData>();
            if (prevData && !prevData->hasInstId1) {
                // Count skip distance (non-pseudo instructions between prev delay_alu and here)
                unsigned skip = 0;
                for (auto it = IRList::iterator(lastDelayAlu);;) {
                    ++it;
                    if (it.getNodePtr() == insertBefore) break;
                    auto* si = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if (si && !isPseudoInst(si)) ++skip;
                }
                if (skip < 6) {
                    prevData->hasInstId1 = true;
                    prevData->instSkip = static_cast<int8_t>(skip);
                    prevData->instid1Type = id0Type;
                    prevData->instid1Distance = id0Dist;
                    return nullptr;  // packed, no new instruction
                }
            }
        }

        // Emit a new s_delay_alu instruction
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_delay_alu, archId);
        StinkyInstruction* delayInst = irBuilder.create(desc, insertBefore);
        if (hasId1) {
            delayInst->addModifier<SDelayAluData>(
                SDelayAluData(id0Type, id0Dist, 0 /*skip=SAME*/, id1Type, id1Dist));
        } else {
            delayInst->addModifier<SDelayAluData>(SDelayAluData(id0Type, id0Dist));
        }

        // Only remember for future packing if instid1 slot is still free
        return hasId1 ? nullptr : delayInst;
    }

    // Process a single basic block.
    // Phase 1 (emit=false): compute delay state, return true if exit state changed.
    // Phase 2 (emit=true): insert s_delay_alu instructions.
    bool runOnBasicBlock(BasicBlock& bb, bool emit, GfxArchID archId) {
        // Merge predecessor exit states into entry state
        DelayState state;
        for (auto* pred : bb.getPredecessors()) state.merge(blockExitState[pred]);

        PASS_DEBUG(std::cerr << "[DelayAlu] " << (emit ? "emit" : "analyze") << " bb=\""
                             << bb.getLabel() << "\"" << " preds=" << bb.getPredecessors().size()
                             << " state_entries=" << state.size() << "\n");

        StinkyInstruction* lastDelayAlu = nullptr;
        AsmIRBuilder irBuilder(bb, archId);

        for (auto it = bb.begin(); it != bb.end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst) {
                PASS_DEBUG({
                    auto* dir = dyn_cast<AsmDirective>(it.getNodePtr());
                    if (dir && dir->kind == AsmDirectiveKind::TEXTBLOCK)
                        std::cerr << "[DelayAlu] TEXTBLOCK: " << dir->value.substr(0, 80) << "\n";
                });
                continue;
            }
            if (isPseudoInst(inst)) continue;

            // In the pipeline, InsertVgprMsb runs after this pass so no
            // s_set_vgpr_msb should be present. When processing raw .s input
            // that already contains them, skip them here because they co-issue
            // with the previous VALU/SALU instruction and consume 0 cycles.
            // The instruction remains in the IR so the packing skip-count
            // loop still counts it (hardware counts s_set_vgpr_msb for skip).
            if (inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) continue;

            DelayType type = getDelayType(*inst);

            PASS_DEBUG(std::cerr << "[DelayAlu] " << inst->getHwInstDesc()->mnemonic << " type="
                                 << delayTypeName(type) << " state_sz=" << state.size() << "\n");
            PASS_DEBUG(for (const auto& [key, info]
                            : state) {
                if (info.SALUCycles > 0 || info.VALUNum < DelayInfo::VALU_MAX ||
                    info.TRANSNum < DelayInfo::TRANS_MAX) {
                    std::cerr << "  state[" << (key.type == RegType::S ? "s" : "v") << key.idx
                              << "]" << " SALU=" << (int)info.SALUCycles
                              << " VALUNum=" << (int)info.VALUNum
                              << " TRANSNum=" << (int)info.TRANSNum << "\n";
                }
            });

            // Memory ops wait for all outstanding VALU -> clear state
            if (instructionWaitsForVALU(*inst)) {
                PASS_DEBUG(std::cerr << "[DelayAlu]   mem inst clears state: "
                                     << inst->getHwInstDesc()->mnemonic << "\n");
                state = DelayState();
            } else if (type != OTHER) {
                // Collect delay info from source register dependencies
                DelayInfo delay;
                for (const auto& srcReg : inst->getSrcRegs()) {
                    forEachRegUnit(srcReg, [&](const RegKey& key) {
                        if (!isAllocatableReg(key.type)) return;  // skip implicit registers
                        auto stateIt = state.find(key);
                        if (stateIt != state.end()) {
                            PASS_DEBUG(std::cerr << "[DelayAlu]   src hit: "
                                                 << (key.type == RegType::S ? "s" : "v") << key.idx
                                                 << " SALU=" << (int)stateIt->second.SALUCycles
                                                 << " VALUNum=" << (int)stateIt->second.VALUNum
                                                 << " TRANSNum=" << (int)stateIt->second.TRANSNum
                                                 << "\n");
                            delay.merge(stateIt->second);
                            state.erase(stateIt);
                        }
                    });
                }

                PASS_DEBUG(if (delay.SALUCycles > 0 || delay.VALUNum < DelayInfo::VALU_MAX ||
                               delay.TRANSNum < DelayInfo::TRANS_MAX) {
                    std::cerr << "[DelayAlu]   emit: SALU=" << (int)delay.SALUCycles
                              << " VALUNum=" << (int)delay.VALUNum
                              << " TRANSNum=" << (int)delay.TRANSNum << "\n";
                });

                // Emit s_delay_alu if needed
                if (emit) {
                    auto* prev = lastDelayAlu;
                    lastDelayAlu = emitDelayAlu(bb, inst, delay, lastDelayAlu, irBuilder, archId);
                    PASS_DEBUG(if (lastDelayAlu != prev) {
                        if (lastDelayAlu)
                            std::cerr << "[DelayAlu]   new s_delay_alu before "
                                      << inst->getHwInstDesc()->mnemonic << "\n";
                        else if (!prev)
                            std::cerr << "[DelayAlu]   packed into prev before "
                                      << inst->getHwInstDesc()->mnemonic << "\n";
                    });
                }
            }

            // Record dest registers with fresh delay info
            if (type != OTHER) {
                unsigned latency = inst->latencyCycles;
                PASS_DEBUG(std::cerr << "[DelayAlu]   " << delayTypeName(type)
                                     << " def: " << inst->getHwInstDesc()->mnemonic
                                     << " latency=" << latency << "\n");
                for (const auto& destReg : inst->getDestRegs()) {
                    forEachRegUnit(destReg, [&](const RegKey& key) {
                        if (!isAllocatableReg(key.type)) return;  // skip implicit registers
                        PASS_DEBUG(std::cerr << "[DelayAlu]     dest: "
                                             << (key.type == RegType::S ? "s" : "v") << key.idx
                                             << "\n");
                        state[key] = DelayInfo(type, latency);
                    });
                }
            }

            // Advance all state entries by the number of wait states
            // (1 for most instructions, N+1 for s_nop).
            unsigned cycles = getNumWaitStates(*inst);
            state.advance(type, cycles);
        }

        // Save or compare exit state
        if (emit) {
            return false;
        }
        DelayState& saved = blockExitState[&bb];
        if (state != saved) {
            PASS_DEBUG(std::cerr << "[DelayAlu]   state changed for bb=\"" << bb.getLabel()
                                 << "\", re-queue successors\n");
            saved = std::move(state);
            return true;  // state changed, successors need re-processing
        }
        return false;
    }

   public:
    static char ID;

    const char* getName() const override {
        return "InsertDelayAluPass";
    }

    Pass::ID getPassID() const override {
        return &InsertDelayAluPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        auto arch = passCtx.getGemmTileConfig().arch;
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        if (shouldSkipForLowOccupancy(func, archId)) return PreservedAnalyses::all();

        // Get RPO ordering from BBIndexAnalysis for efficient convergence.
        const auto& bbIndex = AM.getResult<BBIndexAnalysis>(func);
        const auto& rpoOrder = bbIndex.rpo;

        PASS_DEBUG(std::cerr << "[DelayAlu] Phase 1: fixed-point state computation ("
                             << rpoOrder.size() << " BBs in RPO)\n");

        // Phase 1: fixed-point iteration to compute stable delay state per BB.
        // Initialize worklist in reverse RPO so pop_back gives RPO processing order.
        std::vector<BasicBlock*> workList;
        std::unordered_set<BasicBlock*> inWorkList;
        for (auto it = rpoOrder.rbegin(); it != rpoOrder.rend(); ++it) {
            workList.push_back(*it);
            inWorkList.insert(*it);
        }

        unsigned iteration = 0;
        while (!workList.empty()) {
            BasicBlock* bb = workList.back();
            workList.pop_back();
            inWorkList.erase(bb);
            ++iteration;

            bool changed = runOnBasicBlock(*bb, /*emit=*/false, archId);
            if (changed) {
                for (auto* succ : bb->getSuccessors()) {
                    if (inWorkList.insert(succ).second) workList.push_back(succ);
                }
            }
        }

        PASS_DEBUG(std::cerr << "[DelayAlu] Phase 1 converged after " << iteration
                             << " BB visits\n");
        PASS_DEBUG(std::cerr << "[DelayAlu] Phase 2: emitting s_delay_alu instructions\n");

        // Phase 2: emit s_delay_alu instructions using converged state.
        for (auto* bb : rpoOrder) {
            runOnBasicBlock(*bb, /*emit=*/true, archId);
        }

        blockExitState.clear();
        return preserveCFGAnalyses();
    }
};

char InsertDelayAluPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertDelayAluPass(int minWavesPerSimd) {
    return std::make_unique<InsertDelayAluPassImpl>(minWavesPerSimd);
}
}  // namespace stinkytofu
