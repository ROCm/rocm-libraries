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
#include "stinkytofu/transforms/asm/EstimateAsmCyclesPass.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "EstimateAsmCyclesPass"

namespace {
using namespace stinkytofu;
constexpr const char* kEstimateAsmTotalCyclesMetadataKey = "EstimateAsmCyclesPass.totalCycles";

class AsmCycleEstimator {
   public:
    void setLocalReadLatencyByArch(stinkytofu::GfxArchID arch) {
        if (arch == GfxArchID::Gfx1250) {
            // FIXME: need to verify
            LocalReadBaseLatencyB128 = 1;
            LocalReadBaseLatencyB64 = 1;
            LocalReadBaseLatencyB32 = 1;
            LocalReadConflictMultiplierB128 = 1;
            LocalReadConflictMultiplierB64 = 1;
            LocalReadConflictMultiplierB32 = 1;
            LocalWriteBaseLatencyB128 = 1;
            LocalWriteBaseLatencyB64 = 1;
            LocalWriteBaseLatencyB32 = 1;
            LocalWriteConflictMultiplierB128 = 1;
            LocalWriteConflictMultiplierB64 = 1;
            LocalWriteConflictMultiplierB32 = 1;
        } else {
            // default setting
            LocalReadBaseLatencyB128 = 1;
            LocalReadBaseLatencyB64 = 1;
            LocalReadBaseLatencyB32 = 1;
            LocalReadConflictMultiplierB128 = 1;
            LocalReadConflictMultiplierB64 = 1;
            LocalReadConflictMultiplierB32 = 1;
            LocalWriteBaseLatencyB128 = 1;
            LocalWriteBaseLatencyB64 = 1;
            LocalWriteBaseLatencyB32 = 1;
            LocalWriteConflictMultiplierB128 = 1;
            LocalWriteConflictMultiplierB64 = 1;
            LocalWriteConflictMultiplierB32 = 1;
        }
    }
    int getLocalWriteLatency(int baseLatency, int conflictMultiplier, double bankConflict) {
        int conflictPenalty = bankConflict * conflictMultiplier;
        return baseLatency + conflictPenalty;
    }
    int getLocalReadLatency(int baseLatency, int conflictMultiplier, double bankConflict) {
        int conflictPenalty = (bankConflict - 1) * conflictMultiplier;
        return baseLatency + conflictPenalty;
    }
    int getLocalReadQueueFullStallCycles(int currentCycle, std::queue<int>& fifo, int bpRead,
                                         int numWaves, bool isStall, double bankConflict) {
        int lrStallLatencyBuffer;
        if (!isStall) {
            lrStallLatencyBuffer = 1;
        } else if (bpRead == 16) {
            lrStallLatencyBuffer = getLocalReadLatency(
                LocalReadBaseLatencyB128, LocalReadConflictMultiplierB128, bankConflict);
        } else if (bpRead == 8) {
            lrStallLatencyBuffer = getLocalReadLatency(
                LocalReadBaseLatencyB64, LocalReadConflictMultiplierB64, bankConflict);
        } else {
            lrStallLatencyBuffer = getLocalReadLatency(
                LocalReadBaseLatencyB32, LocalReadConflictMultiplierB32, bankConflict);
        }
        return getLocalReadStallCycles(currentCycle, fifo, bpRead, numWaves, lrStallLatencyBuffer);
    }
    void pushLocalReadWrite(int currentCycle, std::queue<int>& fifo, int bpr, double bankConflict,
                            bool isLocalRead, int numPreviousLRs) {
        int lrMemLatency;
        if (isLocalRead) {
            if (bpr == 16) {
                lrMemLatency = getLocalReadLatency(LocalReadBaseLatencyB128,
                                                   LocalReadConflictMultiplierB128, bankConflict);
                if (numPreviousLRs <= 4) {
                    lrMemLatency += 2 * numPreviousLRs;
                } else {
                    // Maximum 4 * 2 latency for previous local reads
                    lrMemLatency += 2 * 4;
                }
            } else if (bpr == 8) {
                lrMemLatency = getLocalReadLatency(LocalReadBaseLatencyB64,
                                                   LocalReadConflictMultiplierB64, bankConflict);
            } else {
                lrMemLatency = getLocalReadLatency(LocalReadBaseLatencyB32,
                                                   LocalReadConflictMultiplierB32, bankConflict);
            }
        } else {
            // Local write latency
            if (bpr == 16) {
                lrMemLatency = getLocalWriteLatency(LocalWriteBaseLatencyB128,
                                                    LocalWriteConflictMultiplierB128, bankConflict);
            } else if (bpr == 8) {
                lrMemLatency = getLocalWriteLatency(LocalWriteBaseLatencyB64,
                                                    LocalWriteConflictMultiplierB64, bankConflict);
            } else {
                lrMemLatency = getLocalWriteLatency(LocalWriteBaseLatencyB32,
                                                    LocalWriteConflictMultiplierB32, bankConflict);
            }
        }
        fifo.push(currentCycle + lrMemLatency);
    }
    int getGlobalReadQueueFullStallCycles(int currentCycle, std::deque<int>& fifo, int bpRead,
                                          int numWaves, bool isStall, bool isSgprOffset) {
        int extraIssueCycles = 0;
        if (isSgprOffset) {
            extraIssueCycles = 1;
        }
        (void)fifo;
        (void)bpRead;
        (void)numWaves;
        (void)isStall;
        return currentCycle + extraIssueCycles;
    }
    int getLocalWriteQueueFullStallCycles(int currentCycle, int previousLW, int issueCycles,
                                          int bpr, int numWaves) {
        return currentCycle;
    }
    int getLocalReadCompletionCycle(int currentCycle, std::queue<int>& fifo, std::size_t numLR) {
        if (fifo.size() <= numLR) return currentCycle;
        int finalCycle = currentCycle;
        // pop finisned LR
        while (fifo.size() > numLR) {
            int oldCycle = fifo.front();
            if (oldCycle < currentCycle)
                fifo.pop();
            else
                break;
        }
        // check non-finished LR
        while (fifo.size() > numLR) {
            int oldCycle = fifo.front();
            finalCycle = std::max(finalCycle, oldCycle);
            fifo.pop();
        }
        return finalCycle;
    }
    int getLocalReadStallCycles(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves,
                                int lrStallLatencyBuffer) {
        (void)bpRead;
        (void)numWaves;
        int finalCycle = currentCycle + lrStallLatencyBuffer;
        fifo.push(finalCycle);
        return currentCycle;
    }

   private:
    int LocalReadBaseLatencyB128;
    int LocalReadBaseLatencyB64;
    int LocalReadBaseLatencyB32;
    int LocalReadConflictMultiplierB128;
    int LocalReadConflictMultiplierB64;
    int LocalReadConflictMultiplierB32;
    int LocalWriteBaseLatencyB128;
    int LocalWriteBaseLatencyB64;
    int LocalWriteBaseLatencyB32;
    int LocalWriteConflictMultiplierB128;
    int LocalWriteConflictMultiplierB64;
    int LocalWriteConflictMultiplierB32;
};

/// Implementation of the EstimateAsmCycles pass
///
/// This pass estimates cycle counts for asm-related operations
/// by analyzing instruction sequences and their characteristics.
class EstimateAsmCyclesPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "EstimateAsmCyclesPass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        (void)AM;
        // Reset total cycles before processing
        totalCycles_ = 0;

        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            // std::cout << "Processing basic block: " << bb.getLabel() << "\n";
            // Only process the loopWithPrefetch block

            // if(bb.getLabel() == "LocalReadDoA_I0" || bb.getLabel() == "LocalReadDoB_I0")
            // {
            //     calculateLocalReadBytes(bb, passCtx);

            // }
            // else if(bb.getLabel() == "LocalRead")
            // {
            //     analyzeBankConflicts(bb, passCtx);
            // }
            // else if(bb.getLabel() == "label_LoopBeginL")
            // {
            //     calculateMathClocksInUnrolledLoop(bb, passCtx);
            // }
            calculateMathClocksInUnrolledLoop(bb, passCtx);
        }
        // std::cout << "[EstimateAsmCycles] Total Asm Cycles: " << totalCycles_ << "\n";
        func.setMetaData(kEstimateAsmTotalCyclesMetadataKey, totalCycles_);

        return PreservedAnalyses::all();
    }

    /// Get the total cycles calculated by this pass
    unsigned int getTotalCycles() const {
        return totalCycles_;
    }

   private:
    // (For future extension: If analysis needs to track cycles per basic block/label,
    // consider using a map like below)
    // std::unordered_map<std::string, unsigned int> labelCycles_;
    void appendComment(StinkyInstruction* inst, const std::string& suffix) {
        if (!inst || suffix.empty()) return;
        if (auto* c = inst->getModifier<CommentData>()) {
            if (!c->comment.empty()) c->comment += " ";
            c->comment += suffix;
        } else {
            inst->addModifier<CommentData>(CommentData{suffix});
        }
    }

    struct WmmaCoExecProfile {
        int windowCycles = 0;
        std::vector<bool> valuCoExecSlots;
    };

    static std::string toUpperASCII(std::string_view s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            if (c >= 'a' && c <= 'z')
                out.push_back(static_cast<char>(c - ('a' - 'A')));
            else
                out.push_back(c);
        }
        return out;
    }

    static std::optional<WmmaCoExecProfile> getWmmaCoExecProfile(const StinkyInstruction& inst) {
        WmmaCoExecProfile profile;
        // Default profile for unknown/non-WMMA instructions:
        // one-cycle window with no co-exec slot.
        profile.windowCycles = 1;
        profile.valuCoExecSlots.assign(1, false);

        if (!inst.getHwInstDesc()) return profile;

        const std::string mnemonic = toUpperASCII(inst.getHwInstDesc()->mnemonic);
        if (mnemonic.find("V_WMMA_") != 0) return profile;

        // Keep a valid default when latency is absent/invalid.
        if (inst.latencyCycles <= 0) return profile;

        profile.windowCycles = inst.latencyCycles;
        profile.valuCoExecSlots.assign(static_cast<size_t>(profile.windowCycles), true);
        // First execute cycle is non-insertable, the remaining cycles are co-exec slots.
        profile.valuCoExecSlots[0] = false;
        profile.windowCycles = static_cast<int>(profile.valuCoExecSlots.size());
        if (profile.windowCycles == 0) {
            profile.windowCycles = 1;
            profile.valuCoExecSlots.assign(1, false);
        }
        return profile;
    }

    static int getActiveWmmaElapsedCycle(int cycles, int activeWmmaStartCycle,
                                         int activeWmmaCoExecAdvance) {
        if (activeWmmaStartCycle < 0) return -1;
        return (cycles - activeWmmaStartCycle) + activeWmmaCoExecAdvance;
    }

    static bool canCoExecAtCurrentCycle(int cycles, int activeWmmaStartCycle,
                                        int& activeWmmaCoExecAdvance,
                                        const std::vector<bool>& valuSlots, bool isValu) {
        if (activeWmmaStartCycle < 0 || valuSlots.empty()) return false;
        if (!isValu) return true;
        int elapsed =
            getActiveWmmaElapsedCycle(cycles, activeWmmaStartCycle, activeWmmaCoExecAdvance);
        if (elapsed < 0 || elapsed >= static_cast<int>(valuSlots.size())) return false;
        if (valuSlots[static_cast<size_t>(elapsed)]) return true;

        // For VALU, skip forward to the next insertable co-exec slot.
        for (int idx = elapsed + 1; idx < static_cast<int>(valuSlots.size()); ++idx) {
            if (valuSlots[static_cast<size_t>(idx)]) {
                activeWmmaCoExecAdvance += (idx - elapsed);
                return true;
            }
        }
        return false;
    }

    /// Get a string key for a StinkyRegister for use in vgprState/sgprState maps.
    /// Uses symbolic name if set, otherwise "prefix" + reg index (e.g. "v0", "s1").
    static std::string getRegisterKey(const StinkyRegister& reg) {
        if (reg.dataType != StinkyRegister::Type::Register) return "";
        if (reg.hasSymbolicName()) return reg.getSymbolicName();
        return regTypeToString(reg.reg.type) + std::to_string(reg.reg.idx);
    }

    /// Resolve an instruction source operand to an int64 value for simulation.
    /// - LiteralInt/LiteralDouble: returns the literal value.
    /// - Register: looks up current value in vgprState (V/AGPR) or sgprState (S); returns 0 if not
    /// found.
    /// - Other types: returns 0.
    static int64_t getInstructionInputValue(
        const StinkyRegister& src, const std::unordered_map<std::string, int64_t>& vgprState,
        const std::unordered_map<std::string, int64_t>& sgprState) {
        switch (src.dataType) {
            case StinkyRegister::Type::LiteralInt:
                return static_cast<int64_t>(src.getLiteralInt());
            case StinkyRegister::Type::LiteralDouble:
                return static_cast<int64_t>(src.getLiteralDouble());
            case StinkyRegister::Type::Register: {
                std::string key = getRegisterKey(src);
                if (key.empty()) return 0;
                // VGPR/AGPR: per-thread state
                if (src.reg.type == RegType::V || src.reg.type == RegType::AGPR) {
                    auto it = vgprState.find(key);
                    return it != vgprState.end() ? it->second : 0;
                }
                // SGPR and other scalar-like: shared state
                auto it = sgprState.find(key);
                return it != sgprState.end() ? it->second : 0;
            }
            case StinkyRegister::Type::LiteralString:
            case StinkyRegister::Type::Invalid:
            default:
                return 0;
        }
    }

    // Simulate VALU instruction using dynamic casting (type-safe approach)
    void simulateInstructionTyped(StinkyInstruction* inst,
                                  std::unordered_map<std::string, int64_t>& vgprState,
                                  const std::unordered_map<std::string, int64_t>& sgprState) {
        // Try to cast to CommonInstruction first (most VALU instructions inherit from this)
        auto dstReg = inst->getDestRegs();

        // v_add_co_u32: dst = src0 + src1 (with carry out, dst1 is vcc)
        if (GFX::v_add_co_u32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 + src1;
            }
        }
        // v_add_u32, v_add_i32
        // else if(GFX::v_add_u32 == inst->getUnifiedOpcode() || GFX::v_add_i32 ==
        // inst->getUnifiedOpcode())
        // {
        //     if(inst->getSrcRegs().size() >= 2)
        //     {
        //         int64_t src0 = getInstructionInputValue(inst->getSrcRegs()[0], vgprState,
        //         sgprState); int64_t src1 = getInstructionInputValue(inst->getSrcRegs()[1],
        //         vgprState, sgprState); vgprState[getRegisterKey(dstReg[0])] = src0 + src1;
        //     }
        // }
        // else if(GFX::v_add_i32 == inst->getUnifiedOpcode())
        // {
        //     if(inst->getSrcRegs().size() >= 2)
        //     {
        //         int64_t src0 = getInstructionInputValue(inst->getSrcRegs()[0], vgprState,
        //         sgprState); int64_t src1 = getInstructionInputValue(inst->getSrcRegs()[1],
        //         vgprState, sgprState); vgprState[getRegisterKey(dstReg[0])] = src0 + src1;
        //     }
        // }
        // v_sub_u32, v_sub_i32
        else if (GFX::v_sub_u32 == inst->getUnifiedOpcode() ||
                 GFX::v_sub_i32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 - src1;
            }
        } else if (GFX::v_sub_i32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 - src1;
            }
        }
        // v_mul_lo_u32
        else if (GFX::v_mul_lo_u32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 * src1;
            }
        }
        // v_mul_hi_u32: dst = high 32 bits of (src0 * src1)
        else if (GFX::v_mul_hi_u32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                // Cast to uint64_t for unsigned multiplication, then take high 32 bits
                uint64_t result = static_cast<uint64_t>(static_cast<uint32_t>(src0)) *
                                  static_cast<uint64_t>(static_cast<uint32_t>(src1));
                vgprState[getRegisterKey(dstReg[0])] = static_cast<int64_t>(result >> 32);
            }
        }
        // v_mul_hi_i32: dst = high 32 bits of (src0 * src1) signed
        else if (GFX::v_mul_hi_i32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                // Cast to int64_t for signed multiplication, then take high 32 bits
                int64_t result = static_cast<int64_t>(static_cast<int32_t>(src0)) *
                                 static_cast<int64_t>(static_cast<int32_t>(src1));
                vgprState[getRegisterKey(dstReg[0])] = result >> 32;
            }
        }
        // v_lshlrev_b32 or v_lshl_b32: logical shift left
        else if (GFX::v_lshlrev_b32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                // VLShiftLeftB32 format: dst, shiftAmount, src
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src << shiftAmount;
            }
        }
        // v_lshrrev_b32 or v_lshr_b32: logical shift right (32-bit)
        else if (GFX::v_lshrrev_b32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                // VLShiftRightB32 format: dst, shiftAmount, src
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = (uint64_t)src >> (uint64_t)shiftAmount;
            }
        }
        // v_lshlrev_b64 or v_lshl_b64: logical shift left (64-bit)
        else if (GFX::v_lshlrev_b64 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                // VLShiftLeftB64 format: dst, shiftAmount, src
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] =
                    static_cast<int64_t>(static_cast<uint64_t>(src) << shiftAmount);
            }
        }
        // v_lshrrev_b64 or v_lshr_b64: logical shift right (64-bit)
        else if (GFX::v_lshrrev_b64 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                // VLShiftRightB64 format: dst, shiftAmount, src
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src = getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] =
                    static_cast<int64_t>(static_cast<uint64_t>(src) >> shiftAmount);
            }
        }
        // v_and_b32
        else if (GFX::v_and_b32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 & src1;
            }
        }
        // v_or_b32
        else if (GFX::v_or_b32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 | src1;
            }
        }
        // v_xor_b32
        else if (GFX::v_xor_b32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 2) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 ^ src1;
            }
        }
        // v_mad_u32_u24: dst = src0 * src1 + src2
        else if (GFX::v_mad_u32_u24 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 3) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                int64_t src2 =
                    getInstructionInputValue(inst->getSrcRegs()[2], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 * src1 + src2;
            }
        }
        // v_mad_i32_i24: dst = src0 * src1 + src2
        else if (GFX::v_mad_i32_i24 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 3) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                int64_t src2 =
                    getInstructionInputValue(inst->getSrcRegs()[2], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0 * src1 + src2;
            }
        }
        // v_mov_b32: dst = src0
        else if (GFX::v_mov_b32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 1) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = src0;
            }
        }
        // v_lshl_add_u32 (actual instruction, CommonInstruction): dst = (src0 << shiftAmount) +
        // src1 srcs = {src0, shiftHex, src1} - note the order!
        else if (GFX::v_lshl_add_u32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 3) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[2], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = (src0 << shiftAmount) + src1;
            }
        }
        // v_add_lshl_u32 (actual instruction, CommonInstruction): dst = (src0 + src1) <<
        // shiftAmount srcs = {src0, src1, shiftHex} - standard order
        else if (GFX::v_add_lshl_u32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 3) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[2], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = (src0 + src1) << shiftAmount;
            }
        }
        // v_lshl_add_u32 (CompositeInstruction wrapper): dst = (src0 << shiftAmount) + src1
        // srcs = {src0, src1, shiftHex} based on constructor
        else if (GFX::v_lshl_add_u32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 3) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[2], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = (src0 << shiftAmount) + src1;
            }
        }
        // v_add_lshl_u32 (CompositeInstruction wrapper): dst = (src0 + src1) << shiftAmount
        // srcs = {src0, src1, shiftHex} based on constructor
        else if (GFX::v_add_lshl_u32 == inst->getUnifiedOpcode()) {
            if (inst->getSrcRegs().size() >= 3) {
                int64_t src0 =
                    getInstructionInputValue(inst->getSrcRegs()[0], vgprState, sgprState);
                int64_t src1 =
                    getInstructionInputValue(inst->getSrcRegs()[1], vgprState, sgprState);
                int64_t shiftAmount =
                    getInstructionInputValue(inst->getSrcRegs()[2], vgprState, sgprState);
                vgprState[getRegisterKey(dstReg[0])] = (src0 + src1) << shiftAmount;
            }
        } else {
            std::cout << "Unsupported instruction: " << (inst->getHwInstDesc()->mnemonic)
                      << ", Opcode: " << inst->getUnifiedOpcode() << "\n";
        }
    }
    void analyzeBankConflicts(BasicBlock& bb, PassContext& passCtx) {
        // default bank conflicts
        bankConflictA_ = 1.0;
        bankConflictB_ = 1.0;

        std::vector<std::unordered_map<std::string, int64_t>> vgprState(getWaveFrontSize(arch_));
        std::unordered_map<std::string, int64_t> sgprState;  // SGPRs are shared across all threads

        // Initialize thread IDs (v[vgprSerial])
        std::string vgprSerial = "vgprSerial";
        std::string vgprLocalReadAddrA = "vgprLocalReadAddrA";
        std::string vgprLocalReadAddrB = "vgprLocalReadAddrB";

        // parse the basic block to find the local read addresses
        std::vector<StinkyInstruction*> instructions;
        for (IRBase& irNode : bb) {
            if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                instructions.push_back(cast<StinkyInstruction>(&irNode));
            }
        }

        // Simulate real behavior: propagate vgpr/sgpr state as instructions are executed.
        // 'sgprState' is a map<string, int64_t>
        // 'vgprState' is a vector<map<string, int64_t>>, one for each lane/thread

        // initial vgprSerial
        const auto waveFrontSize = getWaveFrontSize(arch_);
        for (uint32_t tid = 0; tid < waveFrontSize; tid++) {
            vgprState[tid][vgprSerial] = tid;
        }

        for (StinkyInstruction* inst : instructions) {
            // Simplified simulation: handle basic assignment and arithmetic for sgpr/vgpr
            const std::string& opcode = inst->getHwInstDesc()->mnemonic;
            const auto& dsts = inst->getDestRegs();
            const auto& srcs = inst->getSrcRegs();
            // std::cout << "Instruction: " << opcode << "\n";

            // Handle SGPR assignments (global)
            if (opcode.find("s_mov") != std::string::npos) {
                sgprState[getRegisterKey(dsts[0])] =
                    getInstructionInputValue(srcs[0], vgprState[0], sgprState);
            }
            // Handle VGPR assignments (per-thread)
            else if (opcode.find("v_") != std::string::npos) {
                // Simulate for each thread using type-safe dynamic casting
                for (uint32_t tid = 0; tid < waveFrontSize; tid++) {
                    simulateInstructionTyped(inst, vgprState[tid], sgprState);
                }
            }
            // (ignore operations that do not write to sgpr/vgpr)

            // Additional behaviors (address, etc.) could be simulated here if needed.

            // print all registers
            // std::cout << "Registers: ";
            // for(int tid = 0; tid < getWaveFrontSize(arch_); tid++)
            // {
            //     std::cout << "Thread " << tid << ": ";
            //     for(const auto& vgpr : vgprState[tid])
            //     {
            //         std::cout << vgpr.first << " = " << vgpr.second << " " << std::endl;
            //     }
            //     std::cout << std::endl;
            // }
        }
    }

    void calculateLocalReadBytes(BasicBlock& bb, PassContext& passCtx) {
        if (bb.getLabel() == "LocalReadDoA_I0") {
            LocalReadBytesA_ = 16;
            for (IRBase& irNode : bb) {
                if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                    StinkyInstruction* inst = cast<StinkyInstruction>(&irNode);
                    auto dstRegs = inst->getDestRegs();
                    if (dstRegs[0].reg.num == 4) {
                        LocalReadBytesA_ = 16;
                    } else if (dstRegs[0].reg.num == 2) {
                        LocalReadBytesA_ = 8;
                    } else {
                        LocalReadBytesA_ = 4;
                    }
                }
            }
        } else if (bb.getLabel() == "LocalReadDoB_I0") {
            LocalReadBytesB_ = 16;
            for (IRBase& irNode : bb) {
                if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                    StinkyInstruction* inst = cast<StinkyInstruction>(&irNode);
                    auto dstRegs = inst->getDestRegs();
                    if (dstRegs[0].reg.num == 4) {
                        LocalReadBytesB_ = 16;
                    } else if (dstRegs[0].reg.num == 2) {
                        LocalReadBytesB_ = 8;
                    } else {
                        LocalReadBytesB_ = 4;
                    }
                }
            }
        }
    }

    /// Process a single basic block to estimate asm cycles
    void calculateMathClocksInUnrolledLoop(BasicBlock& bb, PassContext& passCtx) {
        arch_ =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        AsmCycleEstimator asmCycleEstimator;
        asmCycleEstimator.setLocalReadLatencyByArch(arch_);

        if (arch_ != GfxArchID::Gfx1250) {
            // FIXME: Add support for gfx1201
            return;
        }

        // Collect all instructions in this basic block
        std::vector<StinkyInstruction*> instructions;
        for (IRBase& irNode : bb) {
            if (irNode.getType() == IRBase::IRType::StinkyTofu) {
                instructions.push_back(cast<StinkyInstruction>(&irNode));
            }
        }

        if (instructions.empty()) return;

        // Estimate cycles for each instruction
        // initial values
        int cycles = 0;
        int hwMFMA = -99;
        int jumpOverhead = 6;
        int previousLW = 0;
        std::queue<int> hwLRFIFO;
        std::queue<int> lgkmLRFIFO;
        std::deque<int> hwGRFIFO;
        int numPreviousLRs = 0;
        int previousBarrierSignal = -11;  // gfx1250 barrier signal latency is 11 cycles
        int activeWmmaStartCycle = -1;
        int activeWmmaCoExecAdvance = 0;
        std::vector<bool> activeWmmaValuSlots;

        // Find vgprLocalReadAddrA and vgprLocalReadAddrB names
        std::string vgprLocalReadAddrA = "vgprLocalReadAddrA";
        std::string vgprLocalReadAddrB = "vgprLocalReadAddrB";

        uint32_t totalCycles = 0;
        // When IR is hierarchical (e.g. populateFunctionFromString), ^block: is only a block
        // boundary. "LoopBeginL" avoids parser conflating label_* with branch target;
        // "label_LoopBeginL" for compat.
        bool isLoopBeginL = (bb.getLabel() == "label_LoopBeginL" || bb.getLabel() == "LoopBeginL");
        for (StinkyInstruction* inst : instructions) {
            if (isLoopBeginL == false) {
                if (isLabel(*inst) && inst->getModifier<LabelData>()->label == "label_LoopBeginL") {
                    isLoopBeginL = true;
                } else
                    continue;
            }
            bool isCoIssued =
                (!isMatrixInstruction(*inst) &&
                 canCoExecAtCurrentCycle(cycles, activeWmmaStartCycle, activeWmmaCoExecAdvance,
                                         activeWmmaValuSlots, isVectorALU(*inst)));
            if (isCoIssued) {
                // VALU is inserted into an "I" slot in active WMMA window, so this instruction does
                // not advance the global cycle counter.
                activeWmmaCoExecAdvance += std::max(1, inst->issueCycles);
            } else if (isBarrier(*inst)) {
                const std::string& opcode = inst->getHwInstDesc()->mnemonic;
                if (opcode.find("s_barrier_signal") != std::string::npos)
                    previousBarrierSignal = cycles;
                if (opcode.find("s_barrier_wait") != std::string::npos)
                    cycles = std::max(cycles + inst->issueCycles, previousBarrierSignal + 11);
                else
                    cycles += inst->issueCycles;
            } else if (isDSRead(*inst)) {
                // get bpr of destination register
                auto dstRegs = inst->getDestRegs();
                int bpr = dstRegs.empty() ? 16 : (dstRegs[0].reg.num * 4);

                // get bank conflict from source address registers
                auto srcRegs = inst->getSrcRegs();
                std::string srcStr;
                double bankConflict = 1.0;
                for (const auto& srcReg : srcRegs) {
                    if (srcReg.isRegister()) {
                        srcStr = srcReg.getSymbolicName();
                        if (srcStr.find(vgprLocalReadAddrA) != std::string::npos) {
                            bankConflict = bankConflictA_;  // Use A's bank conflict
                            break;
                        } else if (srcStr.find(vgprLocalReadAddrB) != std::string::npos) {
                            bankConflict = bankConflictB_;  // Use B's bank conflict
                            break;
                        }
                    }
                }
                // Determine which bank conflict value to use based on source register
                int stallcycle = asmCycleEstimator.getLocalReadQueueFullStallCycles(
                    cycles, hwLRFIFO, bpr, passCtx.getGemmTileConfig().NumWaves, true,
                    bankConflict);
                if (stallcycle == cycles) {
                    // no stall
                    // heck LR fifo
                    auto currCycles = cycles + inst->issueCycles;
                    if (numPreviousLRs > 0 &&
                        bpr >= 4) {  // Access to LDS is shared by a pair of SIMDs
                        currCycles += inst->issueCycles;
                    }
                    cycles = currCycles;
                } else {
                    cycles = stallcycle;
                }
                asmCycleEstimator.pushLocalReadWrite(cycles, lgkmLRFIFO, bpr, bankConflict, true,
                                                     numPreviousLRs);
            } else if (isMatrixInstruction(*inst)) {
                auto mfmaLatency = inst->latencyCycles;
                if (cycles - hwMFMA >= (mfmaLatency - 1)) {
                    cycles += inst->issueCycles;
                } else {
                    cycles = hwMFMA + mfmaLatency;
                }
                hwMFMA = cycles;

                auto wmmaCoExec = getWmmaCoExecProfile(*inst);
                assert(wmmaCoExec.has_value() &&
                       "Missing WMMA co-exec profile. Add this WMMA type into slot table.");
                activeWmmaStartCycle = cycles;
                activeWmmaCoExecAdvance = 0;
                activeWmmaValuSlots = wmmaCoExec->valuCoExecSlots;
            } else if (isBranch(*inst)) {
                cycles = std::max(cycles + jumpOverhead, hwMFMA + 4);
                // End of loop (LabelData set when branch comes from rocisa; IR-from-string may omit
                // it)
                const LabelData* labelData = inst->getModifier<LabelData>();
                if (labelData != nullptr) {
                    const std::string& labelName = labelData->label;
                    auto pos = labelName.find("label_LoopBeginL");
                    if (pos != std::string::npos && pos == 0) {
                        break;
                    }
                }
            } else if (isWaitCnt(*inst)) {
                // std::cout << "Estimate WaitCnt: " << inst->getHwInstDesc()->mnemonic << "\n";
                auto waitCntData = inst->getModifier<SWaitCntData>();
                auto tensorCntData = inst->getModifier<SWaitTensorCntData>();
                auto storeCntData = inst->getModifier<SWaitStoreCntData>();
                if (waitCntData != nullptr) {
                    int dlcnt = waitCntData->dlcnt;
                    int dscnt = waitCntData->dscnt;
                    std::size_t numWaits = static_cast<std::size_t>(dlcnt + dscnt);
                    cycles = asmCycleEstimator.getLocalReadCompletionCycle(cycles + 1, lgkmLRFIFO,
                                                                           numWaits);
                } else if (tensorCntData != nullptr) {
                    cycles += inst->issueCycles;
                } else if (storeCntData != nullptr) {
                    cycles += inst->issueCycles;
                } else {
                    // std::cout << "Estimate WaitCnt: " << inst->getHwInstDesc()->mnemonic << "\n";
                    // std::cout << "WaitCntData is nullptr\n";
                    cycles += inst->issueCycles;
                }
            } else if (isTensorLoad(*inst)) {
                // TODO: Estimate cycles for TensorLoad
                cycles += inst->issueCycles;
            } else if (isDSWrite(*inst)) {
                // get bpr of data size (ds_write_b128 = 16 bytes); dest may be address or data
                // depending on IR
                auto dstRegs = inst->getDestRegs();
                int bpr = 16;
                if (!dstRegs.empty()) bpr = dstRegs[0].reg.num * 4;
                cycles = asmCycleEstimator.getLocalWriteQueueFullStallCycles(
                    cycles, previousLW, inst->issueCycles, bpr, getWaveFrontSize(arch_));
                previousLW = cycles;
                asmCycleEstimator.pushLocalReadWrite(cycles, lgkmLRFIFO, bpr, 1.0, false, 0);
            } else if (isGlobalMemLoad(*inst)) {
                auto currCycles = cycles + inst->issueCycles;
                // get bpr of destination register
                auto dstRegs = inst->getDestRegs();
                int bpr = dstRegs.empty() ? 16 : (dstRegs[0].reg.num * 4);

                bool hasSgprOffset = false;
                auto srcRegs = inst->getSrcRegs();
                for (const auto& srcReg : srcRegs) {
                    // std::cout<<"srcReg: "<<srcReg.getSymbolicName()<<std::endl;
                    if (srcReg.isRegister()) {
                        auto srcStr = srcReg.getSymbolicName();
                        if (srcStr.find("s") != std::string::npos) {
                            hasSgprOffset = true;
                        }
                    }
                }
                cycles = asmCycleEstimator.getGlobalReadQueueFullStallCycles(
                    currCycles, hwGRFIFO, bpr, getWaveFrontSize(arch_),
                    passCtx.getGemmTileConfig().arch[0] == 9, hasSgprOffset);
            } else if (isLabel(*inst)) {
                // TODO: Estimate cycles for Label
            } else {
                cycles += inst->issueCycles;
            }

            // Set Flags
            if (isMatrixInstruction(*inst)) {
                numPreviousLRs = 0;
            } else if (isDSRead(*inst)) {
                numPreviousLRs++;
            } else if (isVectorALU(*inst)) {
                numPreviousLRs = 0;
            } else if (isScalarALU(*inst)) {
                numPreviousLRs = 0;
            } else {
                numPreviousLRs = 0;
            }
            // Accumulate issue cycles from each instruction
            // totalCycles += estimateInstructionCycles(inst, passCtx);

            // Update total cycles
            if (!isLabel(*inst)) {
                // inst->dump(std::cout, false, "AsmCycles "+std::to_string(cycles - totalCycles));
                if (auto* c = inst->getModifier<CommentData>()) {
                    c->comment = "<This is " + std::to_string(cycles) + "-cycle>";
                } else {
                    inst->addModifier<CommentData>(
                        CommentData{"<This is " + std::to_string(cycles) + "-cycle>"});
                }
            }
            // std::cout << "cycles: " << cycles - totalCycles << std::endl;
            totalCycles = cycles;
        }
        totalCycles_ = totalCycles;
    }

    /// Estimate cycles for a single instruction
    /// Returns the issueCycles from the StinkyInstruction
    unsigned estimateInstructionCycles(StinkyInstruction* inst, PassContext& passCtx) {
        if (!inst) return 0;

        // Return the issueCycles from the instruction
        // issueCycles is an int, convert to unsigned (clamp negative values to 0)
        return static_cast<unsigned>(inst->issueCycles > 0 ? inst->issueCycles : 0);
    }

    GfxArchID arch_ = GfxArchID::Gfx1250;

    unsigned int totalCycles_ = 0;

    double bankConflictA_ = 0.0;
    double bankConflictB_ = 0.0;
    int LocalReadBytesA_ = 4;
    int LocalReadBytesB_ = 4;
};

char EstimateAsmCyclesPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createEstimateAsmCyclesPass() {
    return std::make_unique<EstimateAsmCyclesPassImpl>();
}

EstimateAsmCyclesAnalysis::Result EstimateAsmCyclesAnalysis::run(Function& func,
                                                                 AnalysisManager& AM) {
    (void)AM;
    PassContext passCtx;
    passCtx.setGemmTileConfig(func.getGemmTileConfig());
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::all());
    return calculateEstimateAsmCycles(func, passCtx);
}

unsigned int calculateEstimateAsmCycles(Function& func, PassContext& passCtx) {
    EstimateAsmCyclesPassImpl pass;
    AnalysisManager AM;
    (void)pass.run(func, passCtx, AM);
    return pass.getTotalCycles();
}
}  // namespace stinkytofu
