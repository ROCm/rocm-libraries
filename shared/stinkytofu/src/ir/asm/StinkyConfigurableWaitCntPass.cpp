/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Configurable WaitCnt Pass - Policy-Based Design
 * 
 * Allows configuring what to wait for at barriers and dependencies
 * ************************************************************************ */

#include "ir/asm/StinkyConfigurableWaitCntPass.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace
{
    using namespace stinkytofu;

    constexpr int8_t WAIT_COMPLETE = 0;
    constexpr int8_t WAIT_IGNORE   = -1;
    constexpr int    MAX_WAITCNT   = 255;

    // Note: BarrierWaitPolicy, DependencyTrackingPolicy, and WaitCntConfig
    // are now defined in StinkyConfigurableWaitCntPass.hpp

    /**
     * @brief Tracks outstanding memory operations
     */
    struct MemoryOperationState
    {
        int globalLoadCount  = 0;
        int globalStoreCount = 0;
        int dsLoadCount      = 0;
        int dsStoreCount     = 0;
        int tensorLoadCount  = 0;
        int atomicCount      = 0;

        void incrementForInst(const StinkyInstruction& inst)
        {
            if(isGlobalMemLoad(inst))
                globalLoadCount++;
            else if(isGlobalMemStore(inst))
                globalStoreCount++;
            else if(isDSRead(inst))
                dsLoadCount++;
            else if(isDSWrite(inst))
                dsStoreCount++;
            else if(isTensorLoad(inst))
                tensorLoadCount++;
            else if(isAtomic(inst))
                atomicCount++;
        }

        void applyWaitCnt(const SWaitCntData& wait)
        {
            auto applyCount = [](int8_t waitValue, int& counter) {
                if(waitValue == WAIT_COMPLETE)
                    counter = 0;
                else if(waitValue != WAIT_IGNORE)
                    counter = std::max(0, counter - waitValue);
            };

            applyCount(wait.vlcnt, globalLoadCount);
            applyCount(wait.vscnt, globalStoreCount);
            applyCount(wait.dlcnt, dsLoadCount);
            applyCount(wait.dscnt, dsStoreCount);
        }

        void applyTensorWaitCnt(const SWaitTensorCntData& wait)
        {
            if(wait.tlcnt == WAIT_COMPLETE)
                tensorLoadCount = 0;
            else if(wait.tlcnt != WAIT_IGNORE)
                tensorLoadCount = std::max(0, tensorLoadCount - wait.tlcnt);
        }

        bool isAtomic(const StinkyInstruction& inst) const
        {
            // Check if instruction is atomic (implementation depends on ISA)
            return false; // Placeholder
        }
    };

    /**
     * @brief Wait count requirement
     */
    struct WaitCntRequirement
    {
        int8_t vlcnt = WAIT_IGNORE;
        int8_t vscnt = WAIT_IGNORE;
        int8_t dlcnt = WAIT_IGNORE;
        int8_t dscnt = WAIT_IGNORE;

        bool isValid() const
        {
            return vlcnt != WAIT_IGNORE || vscnt != WAIT_IGNORE || dlcnt != WAIT_IGNORE
                   || dscnt != WAIT_IGNORE;
        }

        void merge(const WaitCntRequirement& other)
        {
            auto mergeCount = [](int8_t& target, int8_t value) {
                if(value != WAIT_IGNORE)
                {
                    if(target == WAIT_IGNORE)
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
    class ConfigurableWaitCntInserter
    {
    public:
        enum class DependencyType
        {
            LOAD_TO_USE,
            STORE_TO_LOAD,
            STORE_TO_STORE
        };

        struct MemoryDependency
        {
            IRList::iterator   memOp;
            IRList::iterator   consumer;
            StinkyRegister     reg;
            WaitCntRequirement waitReq;
            DependencyType     type;
        };

        ConfigurableWaitCntInserter(IRList&                 insts,
                                    StinkyInstIRBuilder&    irBuilder,
                                    GfxArchID               arch,
                                    const IRListProperties& props,
                                    const WaitCntConfig&    config = WaitCntConfig::standard())
            : insts_(insts)
            , irBuilder_(irBuilder)
            , arch_(arch)
            , properties_(props)
            , config_(config)
        {
        }

        /**
         * @brief Main entry point - insert all required waitcnt
         */
        void insertWaitCounts()
        {
            // Phase 1: Insert configurable waitcnt before barriers
            insertBarrierWaitCounts();

            // Phase 2: Build dependency map based on policy
            if(config_.dependencyPolicy.trackLoadDependencies
               || config_.dependencyPolicy.trackStoreDependencies)
            {
                buildMemoryDependencies();
            }

            // Phase 3: Insert waitcnt for dependencies
            if(!dependencies_.empty())
            {
                insertMemoryDependencyWaitCounts();
            }
        }

        /**
         * @brief Get configuration being used
         */
        const WaitCntConfig& getConfig() const
        {
            return config_;
        }

        /**
         * @brief Get collected dependencies (for debugging)
         */
        const std::vector<MemoryDependency>& getDependencies() const
        {
            return dependencies_;
        }

    private:
        // ================================================================
        // PHASE 1: Configurable Barrier WaitCnt
        // ================================================================

        void insertBarrierWaitCounts()
        {
            for(auto it = insts_.begin(); it != insts_.end(); ++it)
            {
                StinkyInstruction& inst = getStinkyInst(it);

                if(!isBarrier(inst))
                    continue;

                // Analyze what needs to complete based on policy
                BarrierRequirements req = analyzeBarrierRequirements(it);

                // Insert tensor wait if needed
                if(req.needTensorWait && config_.barrierPolicy.waitTensorLoad)
                {
                    insertTensorWaitCnt(it);
                }

                // Build waitcnt requirement based on policy
                WaitCntRequirement waitReq = buildBarrierWaitRequirement(req);

                if(waitReq.isValid())
                {
                    insertWaitCntInstruction(it, waitReq);
                }
            }
        }

        struct BarrierRequirements
        {
            bool foundDSRead      = false;
            bool foundDSWrite     = false;
            bool foundGlobalRead  = false;
            bool foundGlobalWrite = false;
            bool foundTensorLoad  = false;
            bool foundAtomics     = false;

            bool needTensorWait = false;
        };

        BarrierRequirements analyzeBarrierRequirements(IRList::iterator barrierIt)
        {
            BarrierRequirements req;

            if(barrierIt == insts_.begin())
                return req;

            // Scan backwards to find all operations
            IRList::iterator it = barrierIt;
            do
            {
                --it;
                StinkyInstruction& inst = getStinkyInst(it);

                // Categorize instruction
                if(isTensorLoad(inst))
                    req.foundTensorLoad = true;
                else if(isDSRead(inst))
                    req.foundDSRead = true;
                else if(isDSWrite(inst))
                    req.foundDSWrite = true;
                else if(isGlobalMemLoad(inst))
                    req.foundGlobalRead = true;
                else if(isGlobalMemStore(inst))
                    req.foundGlobalWrite = true;
                // Add atomic check if needed

                // Stop at previous barrier
                if(isBarrier(inst))
                    break;

            } while(it != insts_.begin());

            // Determine if tensor wait is needed
            req.needTensorWait = req.foundTensorLoad;

            return req;
        }

        WaitCntRequirement buildBarrierWaitRequirement(const BarrierRequirements& req)
        {
            WaitCntRequirement waitReq;

            // Apply policy to determine what to wait for
            if(req.foundGlobalRead && config_.barrierPolicy.waitGlobalRead)
            {
                waitReq.vlcnt = WAIT_COMPLETE;
            }

            if(req.foundGlobalWrite && config_.barrierPolicy.waitGlobalWrite)
            {
                waitReq.vscnt = WAIT_COMPLETE;
            }

            if(req.foundDSRead && config_.barrierPolicy.waitDSRead)
            {
                waitReq.dlcnt = WAIT_COMPLETE;
            }

            if(req.foundDSWrite && config_.barrierPolicy.waitDSWrite)
            {
                waitReq.dscnt = WAIT_COMPLETE;
            }

            return waitReq;
        }

        void insertTensorWaitCnt(IRList::iterator insertPoint)
        {
            StinkyInstruction* waitInst = irBuilder_.createStinkyInstBefore(
                insertPoint, getMCIDByUOp(GFX::s_wait_tensorcnt, arch_));
            waitInst->addModifier<SWaitTensorCntData>(SWaitTensorCntData(WAIT_COMPLETE));
        }

        // ================================================================
        // PHASE 2: Dependency Analysis (Policy-Driven)
        // ================================================================

        void buildMemoryDependencies()
        {
            dependencies_.clear();

            for(auto it = insts_.begin(); it != insts_.end(); ++it)
            {
                StinkyInstruction& inst = getStinkyInst(it);

                if(!isMemoryOperation(inst))
                    continue;

                // Apply policy
                if(isMemoryLoad(inst) && config_.dependencyPolicy.trackLoadDependencies)
                {
                    buildLoadDependencies(it, inst);
                }
                else if(isMemoryStore(inst) && config_.dependencyPolicy.trackStoreDependencies)
                {
                    buildStoreDependencies(it, inst);
                }
            }
        }

        void buildLoadDependencies(IRList::iterator it, const StinkyInstruction& inst)
        {
            for(const StinkyRegister& destReg : inst.destRegs)
            {
                IRList::iterator useIt = findFirstRegisterUse(it, destReg);

                if(useIt != insts_.end())
                {
                    MemoryDependency dep;
                    dep.memOp    = it;
                    dep.consumer = useIt;
                    dep.reg      = destReg;
                    dep.waitReq  = computeLoadWaitRequirement(it, useIt);
                    dep.type     = DependencyType::LOAD_TO_USE;
                    dependencies_.push_back(dep);
                }
            }
        }

        void buildStoreDependencies(IRList::iterator it, const StinkyInstruction& inst)
        {
            IRList::iterator nextMemIt = findNextConflictingMemoryOp(it, inst);

            if(nextMemIt != insts_.end())
            {
                MemoryDependency dep;
                dep.memOp    = it;
                dep.consumer = nextMemIt;
                dep.reg      = StinkyRegister();
                dep.waitReq  = computeStoreWaitRequirement(it, nextMemIt);

                const StinkyInstruction& nextInst = getStinkyInst(nextMemIt);
                dep.type = isMemoryStore(nextInst) ? DependencyType::STORE_TO_STORE
                                                   : DependencyType::STORE_TO_LOAD;

                dependencies_.push_back(dep);
            }
        }

        // ================================================================
        // PHASE 3: Insert Dependency WaitCnt
        // ================================================================

        void insertMemoryDependencyWaitCounts()
        {
            // Group dependencies by insertion point (use pointer as key for std::map)
            std::map<IRBase*, std::vector<MemoryDependency*>> insertionPoints;

            for(auto& dep : dependencies_)
            {
                insertionPoints[&*dep.consumer].push_back(&dep);
            }

            // Insert or merge waitcnt at each point
            for(auto& [insertPointPtr, deps] : insertionPoints)
            {
                WaitCntRequirement mergedReq;
                for(MemoryDependency* dep : deps)
                {
                    mergedReq.merge(dep->waitReq);
                }

                if(mergedReq.isValid())
                {
                    // Use the iterator from the first dependency (they all point to same place)
                    IRList::iterator insertPoint = deps[0]->consumer;

                    if(config_.dependencyPolicy.mergeAdjacentWaitCnt)
                    {
                        insertOrMergeWaitCnt(insertPoint, mergedReq);
                    }
                    else
                    {
                        insertWaitCntInstruction(insertPoint, mergedReq);
                    }
                }
            }
        }

        void insertOrMergeWaitCnt(IRList::iterator insertPoint, const WaitCntRequirement& req)
        {
            // Try to merge with previous waitcnt
            if(insertPoint != insts_.begin())
            {
                IRList::iterator prevIt = insertPoint;
                --prevIt;
                StinkyInstruction& prevInst = getStinkyInst(prevIt);

                if(isWaitCnt(prevInst))
                {
                    SWaitCntData* existingWait = prevInst.getModifier<SWaitCntData>();
                    if(existingWait)
                    {
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

        void insertWaitCntInstruction(IRList::iterator insertPoint, const WaitCntRequirement& req)
        {
            StinkyInstruction* waitInst = irBuilder_.createStinkyInstBefore(
                insertPoint, getMCIDByUOp(GFX::s_waitcnt, arch_));

            SWaitCntData waitData(req.vlcnt, req.vscnt, req.dlcnt, req.dscnt, WAIT_IGNORE);
            waitInst->addModifier<SWaitCntData>(waitData);
        }

        // ================================================================
        // Helper Functions
        // ================================================================

        bool isMemoryOperation(const StinkyInstruction& inst) const
        {
            return isMemoryLoad(inst) || isMemoryStore(inst);
        }

        bool isMemoryLoad(const StinkyInstruction& inst) const
        {
            return isGlobalMemLoad(inst) || isDSRead(inst);
        }

        bool isMemoryStore(const StinkyInstruction& inst) const
        {
            return isGlobalMemStore(inst) || isDSWrite(inst);
        }

        bool isGlobalMemOperation(const StinkyInstruction& inst) const
        {
            return isGlobalMemLoad(inst) || isGlobalMemStore(inst);
        }

        bool isDSOperation(const StinkyInstruction& inst) const
        {
            return isDSRead(inst) || isDSWrite(inst);
        }

        bool isWaitCnt(const StinkyInstruction& inst) const
        {
            return inst.getModifier<SWaitCntData>() != nullptr;
        }

        IRList::iterator findFirstRegisterUse(IRList::iterator start, const StinkyRegister& reg)
        {
            IRList::iterator it = start;
            ++it;

            bool allowCrossBoundary = config_.dependencyPolicy.trackCrossBoundary;
            if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
                it = properties_.loopBegin;

            while(it != start && it != insts_.end())
            {
                StinkyInstruction& inst = getStinkyInst(it);

                for(const StinkyRegister& srcReg : inst.srcRegs)
                {
                    if(reg.isOverlap(srcReg))
                        return it;
                }

                if(isWaitCnt(inst))
                {
                    if(loadSatisfiedBy(start, inst))
                        return insts_.end();
                }

                ++it;

                if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
                    it = properties_.loopBegin;
            }

            return insts_.end();
        }

        IRList::iterator findNextConflictingMemoryOp(IRList::iterator         storeIt,
                                                     const StinkyInstruction& storeInst)
        {
            IRList::iterator it = storeIt;
            ++it;

            bool allowCrossBoundary = config_.dependencyPolicy.trackCrossBoundary;
            if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
                it = properties_.loopBegin;

            while(it != storeIt && it != insts_.end())
            {
                StinkyInstruction& inst = getStinkyInst(it);

                bool couldConflict = false;
                if(isGlobalMemStore(storeInst))
                    couldConflict = isGlobalMemOperation(inst);
                else if(isDSWrite(storeInst))
                    couldConflict = isDSOperation(inst);

                if(couldConflict)
                    return it;

                if(isWaitCnt(inst))
                {
                    const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                    if(wait && storeCompletedBy(storeInst, *wait))
                        return insts_.end();
                }

                ++it;

                if(allowCrossBoundary && properties_.containsLoop && it == properties_.loopEnd)
                    it = properties_.loopBegin;
            }

            return insts_.end();
        }

        bool loadSatisfiedBy(IRList::iterator memIt, const StinkyInstruction& waitInst) const
        {
            const StinkyInstruction& memInst  = getStinkyInst(memIt);
            const SWaitCntData*      waitData = waitInst.getModifier<SWaitCntData>();

            if(!waitData)
                return false;

            if(isGlobalMemLoad(memInst))
                return waitData->vlcnt == WAIT_COMPLETE;
            else if(isDSRead(memInst))
                return waitData->dlcnt == WAIT_COMPLETE;

            return false;
        }

        bool storeCompletedBy(const StinkyInstruction& storeInst, const SWaitCntData& wait) const
        {
            if(isGlobalMemStore(storeInst))
                return wait.vscnt == WAIT_COMPLETE;
            else if(isDSWrite(storeInst))
                return wait.dscnt == WAIT_COMPLETE;
            return false;
        }

        WaitCntRequirement computeLoadWaitRequirement(IRList::iterator memIt,
                                                      IRList::iterator useIt)
        {
            WaitCntRequirement   req;
            MemoryOperationState state;

            const StinkyInstruction& memInst = getStinkyInst(memIt);

            IRList::iterator it = memIt;
            ++it;

            while(it != useIt && it != insts_.end())
            {
                StinkyInstruction& inst = getStinkyInst(it);
                state.incrementForInst(inst);

                if(isWaitCnt(inst))
                {
                    const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                    if(wait)
                        state.applyWaitCnt(*wait);
                }

                ++it;

                if(config_.dependencyPolicy.trackCrossBoundary && properties_.containsLoop
                   && it == properties_.loopEnd)
                    it = properties_.loopBegin;
            }

            if(isGlobalMemLoad(memInst))
                req.vlcnt = static_cast<int8_t>(std::min(state.globalLoadCount, 127));
            else if(isDSRead(memInst))
                req.dlcnt = static_cast<int8_t>(std::min(state.dsLoadCount, 127));

            return req;
        }

        WaitCntRequirement computeStoreWaitRequirement(IRList::iterator storeIt,
                                                       IRList::iterator nextMemIt)
        {
            WaitCntRequirement   req;
            MemoryOperationState state;

            const StinkyInstruction& storeInst = getStinkyInst(storeIt);

            IRList::iterator it = storeIt;
            ++it;

            while(it != nextMemIt && it != insts_.end())
            {
                StinkyInstruction& inst = getStinkyInst(it);
                state.incrementForInst(inst);

                if(isWaitCnt(inst))
                {
                    const SWaitCntData* wait = inst.getModifier<SWaitCntData>();
                    if(wait)
                        state.applyWaitCnt(*wait);
                }

                ++it;

                if(config_.dependencyPolicy.trackCrossBoundary && properties_.containsLoop
                   && it == properties_.loopEnd)
                    it = properties_.loopBegin;
            }

            if(isGlobalMemStore(storeInst))
                req.vscnt = static_cast<int8_t>(std::min(state.globalStoreCount, 127));
            else if(isDSWrite(storeInst))
                req.dscnt = static_cast<int8_t>(std::min(state.dsStoreCount, 127));

            return req;
        }

        IRList&                       insts_;
        StinkyInstIRBuilder&          irBuilder_;
        GfxArchID                     arch_;
        const IRListProperties&       properties_;
        WaitCntConfig                 config_;
        std::vector<MemoryDependency> dependencies_;
    };

    /**
     * @brief Pass implementation with configurable policy
     */
    class StinkyConfigurableWaitCntPass : public StinkyInstPass
    {
    public:
        static char ID;

        StinkyConfigurableWaitCntPass(const WaitCntConfig& config = WaitCntConfig::standard())
            : config_(config)
        {
        }

        const char* getName() const override
        {
            return "StinkyConfigurableWaitCntPass";
        }

        PassID getPassID() const override
        {
            return &StinkyConfigurableWaitCntPass::ID;
        }

        void run(IRList& insts, PassContext& passCtx) override
        {
            if(insts.empty())
                return;

            GfxArchID arch = getGfxArchID(passCtx.getKernelInfo().arch[0],
                                          passCtx.getKernelInfo().arch[1],
                                          passCtx.getKernelInfo().arch[2]);

            StinkyInstIRBuilder& irBuilder
                = passCtx.getOrCreateIRBuilder<StinkyInstIRBuilder>(insts, arch);

            ConfigurableWaitCntInserter inserter(
                insts, irBuilder, arch, passCtx.getProperties(), config_);
            inserter.insertWaitCounts();
        }

        // Allow configuration to be changed
        void setConfig(const WaitCntConfig& config)
        {
            config_ = config;
        }

        const WaitCntConfig& getConfig() const
        {
            return config_;
        }

    private:
        WaitCntConfig config_;
    };

    char StinkyConfigurableWaitCntPass::ID = 0;
}

namespace stinkytofu
{
    // Factory functions for different configurations

    std::unique_ptr<Pass> createStinkyUnrollWaitCntPass()
    {
        return std::make_unique<StinkyConfigurableWaitCntPass>(WaitCntConfig::unrollLoop());
    }

    std::unique_ptr<Pass> createStinkyConservativeWaitCntPass()
    {
        return std::make_unique<StinkyConfigurableWaitCntPass>(WaitCntConfig::conservative());
    }

    std::unique_ptr<Pass> createStinkyMinimalWaitCntPass()
    {
        return std::make_unique<StinkyConfigurableWaitCntPass>(WaitCntConfig::minimal());
    }

    std::unique_ptr<Pass> createStinkyCustomWaitCntPass(const WaitCntConfig& config)
    {
        return std::make_unique<StinkyConfigurableWaitCntPass>(config);
    }
}
