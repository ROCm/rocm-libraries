/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Configurable WaitCnt Pass - Policy-Based Design
 *
 * Allows configuring what to wait for at barriers and dependencies
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/StinkyConfigurableWaitCntPass.hpp"

#include <algorithm>
#include <map>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"

namespace {
using namespace stinkytofu;

constexpr int WAIT_COMPLETE = 0;
constexpr int WAIT_IGNORE = -1;
constexpr int MAX_WAITCNT = 255;

// Note: BarrierWaitPolicy, DependencyTrackingPolicy, and WaitCntConfig
// are now defined in StinkyConfigurableWaitCntPass.hpp

/**
 * @brief Tracks outstanding memory operations
 */
struct MemoryOperationState {
    int globalLoadCount = 0;
    int globalStoreCount = 0;
    int dsLoadCount = 0;
    int dsStoreCount = 0;
    int tensorLoadCount = 0;
    int atomicCount = 0;

    // Track specific registers with outstanding DS loads (in issue order)
    std::vector<StinkyRegister> outstandingDSLoads;

    // Track specific registers with outstanding global loads (in issue order)
    std::vector<StinkyRegister> outstandingGlobalLoads;

    void incrementForInst(const StinkyInstruction& inst) {
        if (isGlobalMemLoad(inst)) {
            // Track destination registers - remove any previous load to same register
            for (const auto& destReg : inst.getDestRegs()) {
                // Check if this register already has an outstanding load
                auto oldSize = outstandingGlobalLoads.size();
                outstandingGlobalLoads.erase(
                    std::remove_if(
                        outstandingGlobalLoads.begin(), outstandingGlobalLoads.end(),
                        [&destReg](const StinkyRegister& r) { return r.isOverlap(destReg); }),
                    outstandingGlobalLoads.end());

                // Only increment count if this is a new outstanding load (not a replacement)
                if (outstandingGlobalLoads.size() == oldSize) globalLoadCount++;

                outstandingGlobalLoads.push_back(destReg);
            }
        } else if (isGlobalMemStore(inst))
            globalStoreCount++;
        else if (isDSRead(inst)) {
            // Track destination registers - remove any previous load to same register
            for (const auto& destReg : inst.getDestRegs()) {
                // Check if this register already has an outstanding load
                auto oldSize = outstandingDSLoads.size();
                outstandingDSLoads.erase(
                    std::remove_if(
                        outstandingDSLoads.begin(), outstandingDSLoads.end(),
                        [&destReg](const StinkyRegister& r) { return r.isOverlap(destReg); }),
                    outstandingDSLoads.end());

                // Only increment count if this is a new outstanding load (not a replacement)
                if (outstandingDSLoads.size() == oldSize) dsLoadCount++;

                outstandingDSLoads.push_back(destReg);
            }
        } else if (isDSWrite(inst))
            dsStoreCount++;
        else if (isTensorLoad(inst))
            tensorLoadCount++;
        else if (isAtomic(inst))
            atomicCount++;
    }

    void applyWaitCnt(const SWaitCntData& wait) {
        auto applyCount = [](int waitValue, int& counter, std::vector<StinkyRegister>& regList) {
            if (waitValue == WAIT_COMPLETE) {
                counter = 0;
                regList.clear();
            } else if (waitValue != WAIT_IGNORE) {
                int toWait = counter - waitValue;
                if (toWait > 0) {
                    counter = std::max(0, counter - toWait);
                    // Remove the oldest 'toWait' registers
                    if (regList.size() >= (size_t)toWait)
                        regList.erase(regList.begin(), regList.begin() + toWait);
                    else
                        regList.clear();
                }
            }
        };

        applyCount(wait.vlcnt, globalLoadCount, outstandingGlobalLoads);
        applyCount(wait.vscnt, globalStoreCount, outstandingGlobalLoads);  // Store uses same list
        applyCount(wait.dlcnt, dsLoadCount, outstandingDSLoads);
        applyCount(wait.dscnt, dsStoreCount, outstandingDSLoads);  // Store uses same list
    }

    void applyTensorWaitCnt(const SWaitTensorCntData& wait) {
        if (wait.tlcnt == WAIT_COMPLETE)
            tensorLoadCount = 0;
        else if (wait.tlcnt != WAIT_IGNORE)
            tensorLoadCount = std::max(0, tensorLoadCount - wait.tlcnt);
    }

    bool isAtomic(const StinkyInstruction& inst) const {
        // Check if instruction is atomic (implementation depends on ISA)
        return false;  // Placeholder
    }

    // Merge state from another block (for control flow join points)
    // Strategy:
    // 1. If counts match and lists differ: prefer the incoming state (loop state over preloop)
    // 2. If counts differ: clear lists (conservative for diamond CFG)
    void mergeFrom(const MemoryOperationState& other) {
        // Save old state before merging
        int oldDSLoadCount = dsLoadCount;
        int oldGlobalLoadCount = globalLoadCount;

        // Take max counts (will be adjusted if we take union of lists)
        globalStoreCount = std::max(globalStoreCount, other.globalStoreCount);
        dsStoreCount = std::max(dsStoreCount, other.dsStoreCount);
        tensorLoadCount = std::max(tensorLoadCount, other.tensorLoadCount);
        atomicCount = std::max(atomicCount, other.atomicCount);

        // Register list merge strategy:
        // - If lists are identical: keep them (already done, no action needed)
        // - If lists differ but counts match: prefer OTHER's list (loop over preloop)
        // - If counts differ: clear lists (conservative for diamond CFG)

        if (outstandingDSLoads != other.outstandingDSLoads) {
            // Check if lists contain the same registers (just reordered) or different registers
            bool sameRegisters = true;
            if (outstandingDSLoads.size() == other.outstandingDSLoads.size()) {
                for (const auto& reg : outstandingDSLoads) {
                    bool found = false;
                    for (const auto& otherReg : other.outstandingDSLoads) {
                        if (reg.isOverlap(otherReg)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        sameRegisters = false;
                        break;
                    }
                }
            } else {
                sameRegisters = false;
            }

            if (sameRegisters && oldDSLoadCount == other.dsLoadCount) {
                // Same registers, just reordered (preloop + loop case)
                // Create canonical ordering by sorting
                std::vector<StinkyRegister> merged = other.outstandingDSLoads;
                std::sort(merged.begin(), merged.end());
                outstandingDSLoads = merged;
                dsLoadCount = std::max(dsLoadCount, other.dsLoadCount);
            } else {
                // Different registers - take UNION for conservative tracking
                // This preserves register information across multi-predecessor merges
                std::vector<StinkyRegister> unionRegs = outstandingDSLoads;
                for (const auto& reg : other.outstandingDSLoads) {
                    bool found = false;
                    for (const auto& existing : outstandingDSLoads) {
                        if (reg.isOverlap(existing)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) unionRegs.push_back(reg);
                }
                std::sort(unionRegs.begin(), unionRegs.end());
                outstandingDSLoads = unionRegs;
                // Set count to union size (conservative)
                dsLoadCount = unionRegs.size();
            }
        } else {
            // Lists are identical, just take max count
            dsLoadCount = std::max(dsLoadCount, other.dsLoadCount);
        }

        if (outstandingGlobalLoads != other.outstandingGlobalLoads) {
            // Check if lists contain the same registers (just reordered) or different registers
            bool sameRegisters = true;
            if (outstandingGlobalLoads.size() == other.outstandingGlobalLoads.size()) {
                for (const auto& reg : outstandingGlobalLoads) {
                    bool found = false;
                    for (const auto& otherReg : other.outstandingGlobalLoads) {
                        if (reg.isOverlap(otherReg)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        sameRegisters = false;
                        break;
                    }
                }
            } else {
                sameRegisters = false;
            }

            if (sameRegisters && oldGlobalLoadCount == other.globalLoadCount) {
                // Same registers, just reordered (preloop + loop case)
                // Create canonical ordering by sorting
                std::vector<StinkyRegister> merged = other.outstandingGlobalLoads;
                std::sort(merged.begin(), merged.end());
                outstandingGlobalLoads = merged;
                globalLoadCount = std::max(globalLoadCount, other.globalLoadCount);
            } else {
                // Different registers - take UNION for conservative tracking
                // This preserves register information across multi-predecessor merges
                std::vector<StinkyRegister> unionRegs = outstandingGlobalLoads;
                for (const auto& reg : other.outstandingGlobalLoads) {
                    bool found = false;
                    for (const auto& existing : outstandingGlobalLoads) {
                        if (reg.isOverlap(existing)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) unionRegs.push_back(reg);
                }
                std::sort(unionRegs.begin(), unionRegs.end());
                outstandingGlobalLoads = unionRegs;
                // Set count to union size (conservative)
                globalLoadCount = unionRegs.size();
            }
        } else {
            // Lists are identical, just take max count
            globalLoadCount = std::max(globalLoadCount, other.globalLoadCount);
        }
    }

    // Check if state is empty (no outstanding operations)
    bool isEmpty() const {
        return globalLoadCount == 0 && globalStoreCount == 0 && dsLoadCount == 0 &&
               dsStoreCount == 0 && tensorLoadCount == 0 && atomicCount == 0;
    }
};

/**
 * @brief Stores the wait state at basic block boundaries
 *
 * For multi-path analysis, we maintain separate exit states for each
 * incoming path to preserve precision across the CFG.
 */
struct BasicBlockWaitState {
    MemoryOperationState entryState;               // Merged for intra-block analysis
    std::vector<MemoryOperationState> exitStates;  // One per incoming path
    bool processed = false;
};

/**
 * @brief Wait count requirement
 */
struct WaitCntRequirement {
    int vlcnt = WAIT_IGNORE;
    int vscnt = WAIT_IGNORE;
    int dlcnt = WAIT_IGNORE;
    int dscnt = WAIT_IGNORE;

    bool isValid() const {
        return vlcnt != WAIT_IGNORE || vscnt != WAIT_IGNORE || dlcnt != WAIT_IGNORE ||
               dscnt != WAIT_IGNORE;
    }

    void merge(const WaitCntRequirement& other) {
        auto mergeCount = [](int& target, int value) {
            if (value != WAIT_IGNORE) {
                if (target == WAIT_IGNORE)
                    target = value;
                else
                    target = std::min(target, value);
            }
        };

        mergeCount(vlcnt, other.vlcnt);
        mergeCount(vscnt, other.vscnt);
        mergeCount(dlcnt, other.dlcnt);
        mergeCount(dscnt, other.dscnt);
    }
};

/**
 * @brief Configurable WaitCnt inserter with policy-based design
 */
class ConfigurableWaitCntInserter {
   public:
    enum class DependencyType { LOAD_TO_USE, STORE_TO_LOAD, STORE_TO_STORE };

    struct MemoryDependency {
        IRList::iterator memOp;
        IRList::iterator consumer;
        StinkyRegister reg;
        WaitCntRequirement waitReq;
        DependencyType type;
    };

    ConfigurableWaitCntInserter(
        BasicBlock& bb, AsmIRBuilder& irBuilder, GfxArchID arch,
        const WaitCntConfig& config = WaitCntConfig::standard(),
        const std::vector<MemoryOperationState>* predecessorStates = nullptr,
        bool isLoopBlock = false)
        : bb_(&bb),
          irBuilder_(irBuilder),
          arch_(arch),
          config_(config),
          predecessorStates_(predecessorStates ? *predecessorStates
                                               : std::vector<MemoryOperationState>()),
          originalPredecessorStates_(predecessorStates_)  // Save a copy
          ,
          currentState_()  // Will be initialized in Phase 0
          ,
          isLoopBlock_(isLoopBlock) {}

    /**
     * @brief Main entry point - insert all required waitcnt
     * @return Exit states (one per incoming path) for precise multi-path tracking
     */
    std::vector<MemoryOperationState> insertWaitCounts() {
        // Phase 0: Handle cross-block dependencies from predecessor states
        // Always call this to initialize currentState_ from predecessorStates_
        insertCrossBlockDependencyWaitCounts();

        // Phase 1: Insert configurable waitcnt before barriers
        insertBarrierWaitCounts();

        // Phase 2: Build dependency map based on policy
        if (config_.dependencyPolicy.trackLoadDependencies ||
            config_.dependencyPolicy.trackStoreDependencies) {
            buildMemoryDependencies();
        }

        // Phase 3: Insert waitcnt for dependencies
        if (!dependencies_.empty()) {
            insertMemoryDependencyWaitCounts();
        }

        // Phase 4: Compute and return exit states (one per path)
        return computeExitStates();
    }

    /**
     * @brief Compute wait requirement for a single predecessor path
     *
     * Given a state from one predecessor and an instruction, compute the minimum
     * wait counts needed before executing that instruction.
     */
    WaitCntRequirement computeWaitRequirementForPath(const MemoryOperationState& predState,
                                                     const StinkyInstruction& inst) {
        WaitCntRequirement pathReq;

        // Handle DS loads for this path
        if (predState.dsLoadCount > 0) {
            if (!predState.outstandingDSLoads.empty()) {
                // We have register tracking - compute precise dlcnt
                int latestNeededIndex = -1;
                for (const auto& srcReg : inst.getSrcRegs()) {
                    if (srcReg.dataType != StinkyRegister::Type::Register) continue;
                    if (srcReg.reg.type != RegType::V && srcReg.reg.type != RegType::A) continue;

                    // Check if this source register overlaps with any outstanding load
                    for (size_t i = 0; i < predState.outstandingDSLoads.size(); ++i) {
                        if (srcReg.isOverlap(predState.outstandingDSLoads[i])) {
                            latestNeededIndex = std::max(latestNeededIndex, (int)i);
                        }
                    }
                }

                if (latestNeededIndex >= 0) {
                    // We need to wait for loads up to and including latestNeededIndex
                    // dlcnt = how many loads can remain outstanding
                    int remaining = predState.outstandingDSLoads.size() - (latestNeededIndex + 1);
                    pathReq.dlcnt = std::max(0, remaining);
                }
            } else {
                // Conservative state (count without register list)
                // Fall back to wait-all: dlcnt=0
                pathReq.dlcnt = 0;
            }
        }

        // Handle global loads for this path
        if (predState.globalLoadCount > 0) {
            if (!predState.outstandingGlobalLoads.empty()) {
                // We have register tracking - compute precise vlcnt
                int latestNeededIndex = -1;
                for (const auto& srcReg : inst.getSrcRegs()) {
                    if (srcReg.dataType != StinkyRegister::Type::Register) continue;
                    if (srcReg.reg.type != RegType::V && srcReg.reg.type != RegType::A) continue;

                    // Check if this source register overlaps with any outstanding load
                    for (size_t i = 0; i < predState.outstandingGlobalLoads.size(); ++i) {
                        if (srcReg.isOverlap(predState.outstandingGlobalLoads[i])) {
                            latestNeededIndex = std::max(latestNeededIndex, (int)i);
                        }
                    }
                }

                if (latestNeededIndex >= 0) {
                    // We need to wait for loads up to and including latestNeededIndex
                    // vlcnt = how many loads can remain outstanding
                    int remaining =
                        predState.outstandingGlobalLoads.size() - (latestNeededIndex + 1);
                    pathReq.vlcnt = std::max(0, remaining);
                }
            } else {
                // Conservative state (count without register list)
                // Fall back to wait-all: vlcnt=0
                pathReq.vlcnt = 0;
            }
        }

        // Handle stores (always conservative)
        if (predState.dsStoreCount > 0) pathReq.dscnt = 0;  // Wait for all DS stores

        if (predState.globalStoreCount > 0) pathReq.vscnt = 0;  // Wait for all global stores

        return pathReq;
    }

    /**
     * @brief Merge all predecessor states into currentState_
     */
    void mergePredecessorStates() {
        currentState_ = MemoryOperationState();
        bool first = true;
        for (const auto& predState : predecessorStates_) {
            if (first) {
                currentState_ = predState;
                first = false;
            } else {
                currentState_.mergeFrom(predState);
            }
        }
    }

    /**
     * @brief Insert waitcnts for cross-block dependencies using multi-path analysis
     *
     * For GPU performance, we need minimal waiting. Multi-path analysis:
     * 1. Keep predecessor states separate (don't merge early)
     * 2. For each instruction, compute wait requirement from EACH path
     * 3. Take minimum (most restrictive) across all paths
     * 4. After inserting waits, merge predecessor states to create currentState_
     *
     * This ensures we only wait as much as the most restrictive path requires.
     */
    void insertCrossBlockDependencyWaitCounts() {
        // Multi-path analysis: keep predecessor states separate
        if (predecessorStates_.empty()) {
            // No predecessors - start with empty state
            currentState_ = MemoryOperationState();
            return;
        }

        // Scan first few instructions to find register uses
        // After first wait, continue checking against currentState_
        int scanned = 0;
        const int MAX_SCAN = 10;  // Look at first 10 instructions
        bool firstWaitInserted = false;

        for (auto it = bb_->begin(); it != bb_->end() && scanned < MAX_SCAN; ++it, ++scanned) {
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;

            // Skip if this is already a waitcnt
            if (isWaitCnt(inst)) continue;

            // Check if this instruction uses any registers
            if (inst.getSrcRegs().empty()) continue;

            // Check if ANY path (or currentState after first wait) has outstanding operations
            bool hasOutstandingOps = false;
            if (!firstWaitInserted) {
                // Check all predecessor paths
                for (const auto& predState : predecessorStates_) {
                    if (predState.dsLoadCount > 0 || predState.dsStoreCount > 0 ||
                        predState.globalLoadCount > 0 || predState.globalStoreCount > 0) {
                        hasOutstandingOps = true;
                        break;
                    }
                }
            } else {
                // Check currentState_
                if (currentState_.dsLoadCount > 0 || currentState_.dsStoreCount > 0 ||
                    currentState_.globalLoadCount > 0 || currentState_.globalStoreCount > 0) {
                    hasOutstandingOps = true;
                }
            }

            if (!hasOutstandingOps) continue;

            // Check if any source registers are VGPRs or AGPRs
            bool usesVGPR = false;
            for (const auto& reg : inst.getSrcRegs()) {
                if (reg.dataType == StinkyRegister::Type::Register &&
                    (reg.reg.type == RegType::V || reg.reg.type == RegType::A)) {
                    usesVGPR = true;
                    break;
                }
            }

            if (!usesVGPR) continue;

            // Compute wait requirement
            WaitCntRequirement finalReq;
            if (!firstWaitInserted) {
                // Multi-path analysis: compute for EACH predecessor path
                // and take the minimum (most restrictive) to minimize GPU stalls
                for (const auto& predState : predecessorStates_) {
                    WaitCntRequirement pathReq = computeWaitRequirementForPath(predState, inst);
                    finalReq.merge(pathReq);  // merge takes minimum
                }
            } else {
                // Check against currentState_
                finalReq = computeWaitRequirementForPath(currentState_, inst);
            }

            // Insert waitcnt if needed
            if (finalReq.isValid()) {
                insertWaitCntInstruction(&inst, finalReq);

                if (!firstWaitInserted) {
                    // CRITICAL: Apply wait to EACH predecessor path BEFORE merging
                    // This preserves register precision through the merge
                    SWaitCntData waitData(finalReq.vlcnt, finalReq.vscnt, finalReq.dlcnt,
                                          finalReq.dscnt, WAIT_IGNORE);

                    for (auto& predState : predecessorStates_) {
                        predState.applyWaitCnt(waitData);
                    }

                    // Now merge the post-wait states
                    mergePredecessorStates();
                    firstWaitInserted = true;
                    predecessorStates_.clear();
                } else {
                    // Apply wait to currentState_
                    SWaitCntData waitData(finalReq.vlcnt, finalReq.vscnt, finalReq.dlcnt,
                                          finalReq.dscnt, WAIT_IGNORE);
                    currentState_.applyWaitCnt(waitData);
                }
            }
        }

        // If we didn't insert any wait, still need to merge predecessor states
        if (!firstWaitInserted) {
            mergePredecessorStates();
        }
    }

    /**
     * @brief Compute exit states - one per incoming path for precise tracking
     *
     * For multi-predecessor blocks, we maintain separate exit states for each
     * incoming path rather than merging them. This preserves per-path precision
     * and allows successor blocks to perform accurate multi-path analysis.
     */
    std::vector<MemoryOperationState> computeExitStates() {
        std::vector<MemoryOperationState> exitStates;

        // For multi-predecessor blocks WITHOUT back-edges (diamond CFG), maintain per-path states
        // For loops (blocks with back-edges), use merged currentState_ for convergence
        bool usePerPathTracking = !isLoopBlock_ && originalPredecessorStates_.size() >= 2;

        if (usePerPathTracking) {
            // Find all waits inserted in this block (Phase 0, 1, 2, 3)
            std::vector<const SWaitCntData*> insertedWaits;
            for (auto it = bb_->begin(); it != bb_->end(); ++it) {
                auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!_instPtr) continue;
                StinkyInstruction& inst = *_instPtr;
                if (isWaitCnt(inst)) {
                    const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                    if (wait) insertedWaits.push_back(wait);
                }
            }

            // For each predecessor path, compute its exit state
            for (const auto& predState : originalPredecessorStates_) {
                MemoryOperationState pathExit = predState;

                // Apply all new memory operations in this block
                for (auto it = bb_->begin(); it != bb_->end(); ++it) {
                    auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if (!_instPtr) continue;
                    StinkyInstruction& inst = *_instPtr;

                    bool isMemOp = (isGlobalMemLoad(inst) || isGlobalMemStore(inst) ||
                                    isDSRead(inst) || isDSWrite(inst) || isTensorLoad(inst));
                    if (isMemOp) {
                        pathExit.incrementForInst(inst);
                    }
                }

                // Apply all inserted waits to this path
                for (const auto* wait : insertedWaits) {
                    pathExit.applyWaitCnt(*wait);
                }

                exitStates.push_back(pathExit);
            }
        } else {
            // No predecessors - compute single exit state from currentState_
            MemoryOperationState exitState = currentState_;

            bool seenMemoryOp = false;
            for (auto it = bb_->begin(); it != bb_->end(); ++it) {
                auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!_instPtr) continue;
                StinkyInstruction& inst = *_instPtr;

                bool isMemOp = (isGlobalMemLoad(inst) || isGlobalMemStore(inst) || isDSRead(inst) ||
                                isDSWrite(inst) || isTensorLoad(inst));
                if (isMemOp) {
                    seenMemoryOp = true;
                    exitState.incrementForInst(inst);
                }

                if (isWaitCnt(inst)) {
                    const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                    if (wait && seenMemoryOp) {
                        exitState.applyWaitCnt(*wait);
                    }
                }

                const SWaitTensorCntData* tensorWait = inst.getModifier<SWaitTensorCntData>();
                if (tensorWait && seenMemoryOp) {
                    exitState.applyTensorWaitCnt(*tensorWait);
                }
            }

            exitStates.push_back(exitState);
        }

        return exitStates;
    }

    /**
     * @brief Get configuration being used
     */
    const WaitCntConfig& getConfig() const {
        return config_;
    }

    /**
     * @brief Get collected dependencies (for debugging)
     */
    const std::vector<MemoryDependency>& getDependencies() const {
        return dependencies_;
    }

   private:
    // ================================================================
    // PHASE 1: Configurable Barrier WaitCnt
    // ================================================================

    void insertBarrierWaitCounts() {
        for (auto it = bb_->begin(); it != bb_->end(); ++it) {
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;

            if (!isBarrier(inst)) continue;

            // Analyze what needs to complete based on policy
            BarrierRequirements req = analyzeBarrierRequirements(it);

            // Insert tensor wait if needed
            if (req.needTensorWait && config_.barrierPolicy.waitTensorLoad) {
                insertTensorWaitCnt(&inst);
            }

            // Build waitcnt requirement based on policy
            WaitCntRequirement waitReq = buildBarrierWaitRequirement(req);

            if (waitReq.isValid()) {
                insertWaitCntInstruction(&inst, waitReq);
            }
        }
    }

    struct BarrierRequirements {
        bool foundDSRead = false;
        bool foundDSWrite = false;
        bool foundGlobalRead = false;
        bool foundGlobalWrite = false;
        bool foundTensorLoad = false;
        bool foundAtomics = false;

        bool needTensorWait = false;
    };

    BarrierRequirements analyzeBarrierRequirements(IRList::iterator barrierIt) {
        BarrierRequirements req;

        if (barrierIt == bb_->begin()) return req;

        // Scan backwards to find all operations
        IRList::iterator it = barrierIt;
        do {
            --it;
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;

            // Categorize instruction
            if (isTensorLoad(inst))
                req.foundTensorLoad = true;
            else if (isDSRead(inst))
                req.foundDSRead = true;
            else if (isDSWrite(inst))
                req.foundDSWrite = true;
            else if (isGlobalMemLoad(inst))
                req.foundGlobalRead = true;
            else if (isGlobalMemStore(inst))
                req.foundGlobalWrite = true;
            // Add atomic check if needed

            // Stop at previous barrier
            if (isBarrier(inst)) break;

        } while (it != bb_->begin());

        // Determine if tensor wait is needed
        req.needTensorWait = req.foundTensorLoad;

        return req;
    }

    WaitCntRequirement buildBarrierWaitRequirement(const BarrierRequirements& req) {
        WaitCntRequirement waitReq;

        // Apply policy to determine what to wait for
        if (req.foundGlobalRead && config_.barrierPolicy.waitGlobalRead) {
            waitReq.vlcnt = WAIT_COMPLETE;
        }

        if (req.foundGlobalWrite && config_.barrierPolicy.waitGlobalWrite) {
            waitReq.vscnt = WAIT_COMPLETE;
        }

        if (req.foundDSRead && config_.barrierPolicy.waitDSRead) {
            waitReq.dlcnt = WAIT_COMPLETE;
        }

        if (req.foundDSWrite && config_.barrierPolicy.waitDSWrite) {
            waitReq.dscnt = WAIT_COMPLETE;
        }

        return waitReq;
    }

    void insertTensorWaitCnt(StinkyInstruction* insertPoint) {
        StinkyInstruction* waitInst =
            irBuilder_.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch_), insertPoint);
        waitInst->addModifier<SWaitTensorCntData>(SWaitTensorCntData(WAIT_COMPLETE));
    }

    // ================================================================
    // PHASE 2: Dependency Analysis (Policy-Driven)
    // ================================================================

    void buildMemoryDependencies() {
        dependencies_.clear();

        for (auto it = bb_->begin(); it != bb_->end(); ++it) {
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;

            if (!isMemoryOperation(inst)) continue;

            // Apply policy
            if (isMemoryLoad(inst) && config_.dependencyPolicy.trackLoadDependencies) {
                buildLoadDependencies(it, inst);
            } else if (isMemoryStore(inst) && config_.dependencyPolicy.trackStoreDependencies) {
                buildStoreDependencies(it, inst);
            }
        }
    }

    void buildLoadDependencies(IRList::iterator it, const StinkyInstruction& inst) {
        for (const StinkyRegister& destReg : inst.getDestRegs()) {
            IRList::iterator useIt = findFirstRegisterUse(it, destReg);

            if (useIt != bb_->end()) {
                MemoryDependency dep;
                dep.memOp = it;
                dep.consumer = useIt;
                dep.reg = destReg;
                dep.waitReq = computeLoadWaitRequirement(it, useIt);
                dep.type = DependencyType::LOAD_TO_USE;
                dependencies_.push_back(dep);
            }
        }
    }

    void buildStoreDependencies(IRList::iterator it, const StinkyInstruction& inst) {
        IRList::iterator nextMemIt = findNextConflictingMemoryOp(it, inst);

        if (nextMemIt != bb_->end()) {
            MemoryDependency dep;
            dep.memOp = it;
            dep.consumer = nextMemIt;
            dep.reg = StinkyRegister();
            dep.waitReq = computeStoreWaitRequirement(it, nextMemIt);

            const StinkyInstruction& nextInst = getStinkyInst(nextMemIt);
            dep.type = isMemoryStore(nextInst) ? DependencyType::STORE_TO_STORE
                                               : DependencyType::STORE_TO_LOAD;

            dependencies_.push_back(dep);
        }
    }

    // ================================================================
    // PHASE 3: Insert Dependency WaitCnt
    // ================================================================

    void insertMemoryDependencyWaitCounts() {
        // Group dependencies by insertion point (use pointer as key for std::map)
        std::map<IRBase*, std::vector<MemoryDependency*>> insertionPoints;

        for (auto& dep : dependencies_) {
            insertionPoints[&*dep.consumer].push_back(&dep);
        }

        // Insert or merge waitcnt at each point
        for (auto& [insertPointPtr, deps] : insertionPoints) {
            WaitCntRequirement mergedReq;
            for (MemoryDependency* dep : deps) {
                mergedReq.merge(dep->waitReq);
            }

            if (mergedReq.isValid()) {
                // Use the iterator from the first dependency (they all point to same place)
                StinkyInstruction* insertPoint = &getStinkyInst(deps[0]->consumer);

                if (config_.dependencyPolicy.mergeAdjacentWaitCnt) {
                    insertOrMergeWaitCnt(insertPoint, mergedReq);
                } else {
                    insertWaitCntInstruction(insertPoint, mergedReq);
                }
            }
        }
    }

    void insertOrMergeWaitCnt(StinkyInstruction* insertPoint, const WaitCntRequirement& req) {
        // Try to merge with previous waitcnt
        if (IRList::iterator(insertPoint) != bb_->begin()) {
            StinkyInstruction& prevInst = *static_cast<StinkyInstruction*>(insertPoint->getPrev());

            if (isWaitCnt(prevInst)) {
                SWaitCntData* existingWait = prevInst.getModifier<SWaitCntData>();
                if (existingWait) {
                    existingWait->vlcnt = std::min(existingWait->vlcnt, req.vlcnt);
                    existingWait->vscnt = std::min(existingWait->vscnt, req.vscnt);
                    existingWait->dlcnt = std::min(existingWait->dlcnt, req.dlcnt);
                    existingWait->dscnt = std::min(existingWait->dscnt, req.dscnt);
                    return;
                }
            }
        }

        insertWaitCntInstruction(insertPoint, req);
    }

    void insertWaitCntInstruction(StinkyInstruction* insertPoint, const WaitCntRequirement& req) {
        StinkyInstruction* waitInst =
            irBuilder_.create(getMCIDByUOp(GFX::s_waitcnt, arch_), insertPoint);

        SWaitCntData waitData(req.vlcnt, req.vscnt, req.dlcnt, req.dscnt, WAIT_IGNORE);
        waitInst->addModifier<SWaitCntData>(waitData);
    }

    // ================================================================
    // Helper Functions
    // ================================================================

    bool isMemoryOperation(const StinkyInstruction& inst) const {
        return isMemoryLoad(inst) || isMemoryStore(inst);
    }

    bool isMemoryLoad(const StinkyInstruction& inst) const {
        return isGlobalMemLoad(inst) || isDSRead(inst);
    }

    bool isMemoryStore(const StinkyInstruction& inst) const {
        return isGlobalMemStore(inst) || isDSWrite(inst);
    }

    bool isGlobalMemOperation(const StinkyInstruction& inst) const {
        return isGlobalMemLoad(inst) || isGlobalMemStore(inst);
    }

    bool isDSOperation(const StinkyInstruction& inst) const {
        return isDSRead(inst) || isDSWrite(inst);
    }

    bool isWaitCnt(const StinkyInstruction& inst) const {
        return inst.getModifier<SWaitCntData>() != nullptr;
    }

    IRList::iterator findFirstRegisterUse(IRList::iterator start, const StinkyRegister& reg) {
        IRList::iterator it = start;
        ++it;

        // bool allowCrossBoundary = config_.dependencyPolicy.trackCrossBoundary;
        // if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
        //     it = properties_.loopBegin;

        while (it != start && it != bb_->end()) {
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;

            for (const StinkyRegister& srcReg : inst.getSrcRegs()) {
                if (reg.isOverlap(srcReg)) return it;
            }

            if (isWaitCnt(inst)) {
                if (loadSatisfiedBy(start, inst)) return bb_->end();
            }

            ++it;

            // if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
            //     it = properties_.loopBegin;
        }

        return bb_->end();
    }

    IRList::iterator findNextConflictingMemoryOp(IRList::iterator storeIt,
                                                 const StinkyInstruction& storeInst) {
        IRList::iterator it = storeIt;
        ++it;

        // bool allowCrossBoundary = config_.dependencyPolicy.trackCrossBoundary;
        // if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
        //     it = properties_.loopBegin;

        while (it != storeIt && it != bb_->end()) {
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;

            bool couldConflict = false;
            if (isGlobalMemStore(storeInst))
                couldConflict = isGlobalMemOperation(inst);
            else if (isDSWrite(storeInst))
                couldConflict = isDSOperation(inst);

            if (couldConflict) return it;

            if (isWaitCnt(inst)) {
                const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                if (wait && storeCompletedBy(storeInst, *wait)) return bb_->end();
            }

            ++it;

            // if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
            //     it = properties_.loopBegin;
        }

        return bb_->end();
    }

    bool loadSatisfiedBy(IRList::iterator memIt, const StinkyInstruction& waitInst) const {
        const StinkyInstruction& memInst = getStinkyInst(memIt);
        const SWaitCntData* waitData = waitInst.getModifier<SWaitCntData>();

        if (!waitData) return false;

        if (isGlobalMemLoad(memInst))
            return waitData->vlcnt == WAIT_COMPLETE;
        else if (isDSRead(memInst))
            return waitData->dlcnt == WAIT_COMPLETE;

        return false;
    }

    bool storeCompletedBy(const StinkyInstruction& storeInst, const SWaitCntData& wait) const {
        if (isGlobalMemStore(storeInst))
            return wait.vscnt == WAIT_COMPLETE;
        else if (isDSWrite(storeInst))
            return wait.dscnt == WAIT_COMPLETE;
        return false;
    }

    WaitCntRequirement computeLoadWaitRequirement(IRList::iterator memIt, IRList::iterator useIt) {
        WaitCntRequirement req;
        MemoryOperationState state = currentState_;

        const StinkyInstruction& memInst = getStinkyInst(memIt);

        IRList::iterator it = memIt;
        ++it;

        while (it != useIt && it != bb_->end()) {
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;
            state.incrementForInst(inst);

            if (isWaitCnt(inst)) {
                const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                if (wait) state.applyWaitCnt(*wait);
            }

            ++it;

            // if(config_.dependencyPolicy.trackCrossBoundary && properties_.containsLoop
            //    && it == properties_.loopEnd)
            //     it = properties_.loopBegin;
        }

        if (isGlobalMemLoad(memInst))
            req.vlcnt = std::min(state.globalLoadCount, 127);
        else if (isDSRead(memInst))
            req.dlcnt = std::min(state.dsLoadCount, 127);

        return req;
    }

    WaitCntRequirement computeStoreWaitRequirement(IRList::iterator storeIt,
                                                   IRList::iterator nextMemIt) {
        WaitCntRequirement req;
        MemoryOperationState state = currentState_;

        const StinkyInstruction& storeInst = getStinkyInst(storeIt);

        IRList::iterator it = storeIt;
        ++it;

        while (it != nextMemIt && it != bb_->end()) {
            auto* _instPtr = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!_instPtr) continue;
            StinkyInstruction& inst = *_instPtr;
            state.incrementForInst(inst);

            if (isWaitCnt(inst)) {
                const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                if (wait) state.applyWaitCnt(*wait);
            }

            ++it;

            // if(config_.dependencyPolicy.trackCrossBoundary && properties_.containsLoop
            //    && it == properties_.loopEnd)
            //     it = properties_.loopBegin;
        }

        if (isGlobalMemStore(storeInst))
            req.vscnt = std::min(state.globalStoreCount, 127);
        else if (isDSWrite(storeInst))
            req.dscnt = std::min(state.dsStoreCount, 127);

        return req;
    }

    BasicBlock* bb_;
    AsmIRBuilder& irBuilder_;
    GfxArchID arch_;
    WaitCntConfig config_;
    std::vector<MemoryDependency> dependencies_;
    std::vector<MemoryOperationState> predecessorStates_;  // States from all predecessors
    std::vector<MemoryOperationState>
        originalPredecessorStates_;      // Saved for exit state computation
    MemoryOperationState currentState_;  // Current state during analysis
    bool isLoopBlock_;                   // Whether this block has a back-edge
};

/**
 * @brief Pass implementation with configurable policy
 */
class StinkyConfigurableWaitCntPass : public StinkyInstPass {
   public:
    static char ID;

    StinkyConfigurableWaitCntPass(const WaitCntConfig& config = WaitCntConfig::standard())
        : config_(config) {}

    const char* getName() const override {
        return "StinkyConfigurableWaitCntPass";
    }

    PassID getPassID() const override {
        return &StinkyConfigurableWaitCntPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        const auto& rpo = AM.getResult<BBIndexAnalysis>(func).rpo;

        // Map to store wait states for each basic block
        std::map<BasicBlock*, BasicBlockWaitState> blockStates;

        // Detect if we have any loops (blocks with back-edges)
        bool hasLoops = false;
        for (auto* bb : rpo) {
            if (hasLoopBackEdge(bb)) {
                hasLoops = true;
            }
        }

        // Iterative dataflow analysis for convergence (needed for loops)
        const int MAX_ITERATIONS = 10;
        int iteration = 0;
        bool changed = true;

        while ((changed || iteration < 2) && iteration < MAX_ITERATIONS) {
            changed = false;
            iteration++;

            // Process each BasicBlock in RPO
            for (auto* bb : rpo) {
                if (bb->empty()) continue;

                // Compute entry state from predecessors
                MemoryOperationState entryState;
                bool hasValidPredecessor = false;
                bool hasNonBackEdgePred = false;
                int validPredCount = 0;

                // First, check if we have any non-back-edge predecessors
                for (BasicBlock* pred : bb->getPredecessors()) {
                    if (pred != bb) {
                        auto predStateIt = blockStates.find(pred);
                        if (predStateIt != blockStates.end() && predStateIt->second.processed) {
                            hasNonBackEdgePred = true;
                            break;
                        }
                    }
                }

                // Detect if this is a pure diamond CFG (multiple non-back-edge predecessors)
                bool isDiamondCFG = false;
                if (bb->getPredecessors().size() >= 2) {
                    int nonBackEdgeCount = 0;
                    for (BasicBlock* pred : bb->getPredecessors()) {
                        if (pred != bb) {
                            auto predStateIt = blockStates.find(pred);
                            if (predStateIt != blockStates.end() && predStateIt->second.processed) {
                                nonBackEdgeCount++;
                            }
                        }
                    }
                    isDiamondCFG = (nonBackEdgeCount >= 2);
                }

                // Collect ALL exit states from ALL predecessors (multi-path analysis)
                // Each predecessor may have multiple exit states (one per its incoming path)
                std::vector<MemoryOperationState> predecessorStates;

                for (BasicBlock* pred : bb->getPredecessors()) {
                    // Skip back-edges ONLY on first iteration when we have entry paths
                    // This avoids merging with not-yet-stable loop state
                    // After first iteration (iteration > 1), allow merging with back-edge
                    // But for diamond CFG (multiple non-back-edge preds), always skip back-edge
                    bool shouldSkipBackEdge =
                        (pred == bb && hasNonBackEdgePred) && (iteration == 1 || isDiamondCFG);

                    if (shouldSkipBackEdge) continue;

                    auto predStateIt = blockStates.find(pred);
                    if (predStateIt != blockStates.end() && predStateIt->second.processed) {
                        validPredCount++;
                        hasValidPredecessor = true;

                        // Collect ALL exit states from this predecessor (per-path tracking)
                        for (const auto& exitState : predStateIt->second.exitStates) {
                            predecessorStates.push_back(exitState);
                        }
                    }
                }

                // Merge predecessor states for convergence detection
                // (Inserter will use separate states for multi-path analysis)
                bool firstPred = true;
                for (const auto& predState : predecessorStates) {
                    if (firstPred) {
                        entryState = predState;
                        firstPred = false;
                    } else {
                        entryState.mergeFrom(predState);
                    }
                }

                // Get or create state for this block
                BasicBlockWaitState& bbState = blockStates[bb];

                // Check if entry state changed (for convergence detection)
                bool entryChanged = false;
                if (bbState.processed && !statesEqual(bbState.entryState, entryState)) {
                    entryChanged = true;
                    changed = true;
                }

                bbState.entryState = entryState;

                // Process the block if:
                // 1. First time processing (!bbState.processed)
                // 2. Entry state changed (entryChanged)
                bool shouldProcess = !bbState.processed || entryChanged;

                if (shouldProcess) {
                    // Clear existing waitcnts if reprocessing
                    if (bbState.processed) {
                        clearWaitCnts(*bb);
                    }

                    auto irBuilder = AsmIRBuilder(*bb, arch);

                    // Pass predecessor states separately for multi-path analysis
                    bool isLoop = hasLoopBackEdge(bb);
                    ConfigurableWaitCntInserter inserter(*bb, irBuilder, arch, config_,
                                                         &predecessorStates, isLoop);
                    std::vector<MemoryOperationState> newExitStates = inserter.insertWaitCounts();

                    // Check if ANY exit state changed (for convergence)
                    if (bbState.exitStates.size() != newExitStates.size()) {
                        changed = true;
                    } else {
                        for (size_t i = 0; i < newExitStates.size(); ++i) {
                            if (!statesEqual(bbState.exitStates[i], newExitStates[i])) {
                                changed = true;
                                break;
                            }
                        }
                    }

                    bbState.exitStates = newExitStates;
                    bbState.processed = true;
                }
            }

            // If no loops, one iteration is sufficient
            if (!hasLoops && iteration >= 1) break;
        }
        return preserveCFGAnalyses();
    }

    // Allow configuration to be changed
    void setConfig(const WaitCntConfig& config) {
        config_ = config;
    }

    const WaitCntConfig& getConfig() const {
        return config_;
    }

   private:
    /**
     * @brief Check if a basic block has a loop back-edge to itself
     */
    bool hasLoopBackEdge(const BasicBlock* bb) const {
        for (const BasicBlock* succ : bb->getSuccessors()) {
            if (succ == bb) return true;
        }
        return false;
    }

    /**
     * @brief Compare two MemoryOperationState objects for equality
     */
    bool statesEqual(const MemoryOperationState& a, const MemoryOperationState& b) const {
        // For convergence detection, must compare both counts and register lists
        // The register order matters for precise dlcnt computation
        return a.globalLoadCount == b.globalLoadCount && a.globalStoreCount == b.globalStoreCount &&
               a.dsLoadCount == b.dsLoadCount && a.dsStoreCount == b.dsStoreCount &&
               a.tensorLoadCount == b.tensorLoadCount && a.atomicCount == b.atomicCount &&
               a.outstandingDSLoads == b.outstandingDSLoads &&
               a.outstandingGlobalLoads == b.outstandingGlobalLoads;
    }

    /**
     * @brief Clear all waitcnt instructions from a basic block
     */
    void clearWaitCnts(BasicBlock& bb) const {
        for (auto it = bb.begin(); it != bb.end();) {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(*it);
            if (inst.getModifier<SWaitCntData>() != nullptr ||
                inst.getModifier<SWaitTensorCntData>() != nullptr) {
                it = bb.eraseIR(it);
            } else {
                ++it;
            }
        }
    }

    WaitCntConfig config_;
};

char StinkyConfigurableWaitCntPass::ID = 0;
}  // namespace

namespace stinkytofu {
// Factory functions for different configurations

std::unique_ptr<Pass> createStinkyUnrollWaitCntPass() {
    return std::make_unique<StinkyConfigurableWaitCntPass>(WaitCntConfig::unrollLoop());
}

std::unique_ptr<Pass> createStinkyConservativeWaitCntPass() {
    return std::make_unique<StinkyConfigurableWaitCntPass>(WaitCntConfig::conservative());
}

std::unique_ptr<Pass> createStinkyMinimalWaitCntPass() {
    return std::make_unique<StinkyConfigurableWaitCntPass>(WaitCntConfig::minimal());
}

std::unique_ptr<Pass> createStinkyCustomWaitCntPass(const WaitCntConfig& config) {
    return std::make_unique<StinkyConfigurableWaitCntPass>(config);
}
}  // namespace stinkytofu
