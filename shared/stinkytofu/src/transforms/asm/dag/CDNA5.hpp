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
//
// CDNA5 (Gfx1250) ready-queue for StinkyDAGSchedulerPass.
//
// StinkyDAGSchedulerPass splits each basic block into regions at non-movable side effects
// (waits, stores, branches, etc.), builds a per-region dependency DAG from physical registers,
// then drains ready nodes via this queue. CDNA5 adds WMMA-centric heuristics on top of that
// DAG: tensor/global prioritization, DS latency modeling, program-order ties, loop-aware head
// balancing (stops WMMA-at-loop-tail plus WMMA-right-after-head in the next iteration), and
// cross-region WMMA spacing.
//
#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <unordered_set>

#include "ReadyQueue.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace
{
    using namespace stinkytofu;

    // -------------------------------------------------------------------------
    // Prefix / loop analysis (free functions; no CDNA5ReadyQueue state)
    // -------------------------------------------------------------------------

    // Scheduling rule (2): simulate ds_load completion over [blockBegin, regionStart) — outstanding
    // VGPR latencies decrease by each instruction's issueCycles; each ds_load overwrites dest VGPRs
    // with that op's latencyCycles (WAW). Remaining counts seed wmmaRegisterLatencyCounters so the
    // first WMMA in a region sees preloop / in-BB loads the register DAG may not edge to that WMMA
    // (double-buffer: WMMA on X0 while in-loop ds fills X1). Caller: onInitRegion.
    static void seedWmmaDsLatencyFromPrefix(IRList::iterator                blockBegin,
                                            IRList::iterator                regionStart,
                                            std::map<int, int>&             wmmaRegisterLatencyCounters)
    {
        wmmaRegisterLatencyCounters.clear();
        std::map<int, int> pending;

        for(IRList::iterator it = blockBegin; it != regionStart; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            const int          iss = inst.issueCycles;

            for(auto pit = pending.begin(); pit != pending.end();)
            {
                pit->second -= iss;
                if(pit->second <= 0)
                    pit = pending.erase(pit);
                else
                    ++pit;
            }

            if(isDSRead(inst))
            {
                for(const StinkyRegister& dstReg : inst.getDestRegs())
                {
                    if(!dstReg.isRegister())
                        continue;
                    for(unsigned off = 0; off < dstReg.reg.num; ++off)
                        pending[dstReg.reg.idx + off] = inst.latencyCycles;
                }
            }
        }

        for(const auto& [regIdx, rem] : pending)
        {
            if(rem > 0)
                wmmaRegisterLatencyCounters[regIdx] = rem;
        }
    }

    // Scheduling rule (5) helper: walk backward in **linear IR / source order** from branchIt
    // (exclusive), skipping LABEL ops; true if the first such instruction is WMMA/SWMMA.
    //
    // This is not read from the dependency DAG. The DAG is built from def-use (edges flow
    // producer -> consumer; the scheduler drains "ready" nodes when in-degree hits 0). "Last"
    // here does **not** mean "last WMMA to execute in the iteration" or "bottom of the DAG" —
    // it means the **textual** opcode immediately before the branch in the IR list. That static
    // shape is cheap to detect and still marks loops whose **control flow** jumps from "branch
    // right after a WMMA" back to the head, which is when head WMMA can schedule back-to-back
    // with tail WMMA without a def-use arc between them. Caller: analyzeLoopTailWmmaAndHeadLabels.
    static bool lastNonLabelBeforeIsWmma(IRList::iterator bbBegin, IRList::iterator branchIt)
    {
        if(branchIt == bbBegin)
            return false;
        for(auto it = std::prev(branchIt);;)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isLabel(inst))
            {
                if(it == bbBegin)
                    return false;
                it = std::prev(it);
                continue;
            }
            return isWMMA(inst) || isSWMMA(inst);
        }
    }

    // Scheduling rule (5) helper: true if any LABEL in [blockBegin, regionStart) matches a recorded
    // loop-head name (backward-edge target with tail WMMA). Preloop prefixes fail → no deferral
    // before the loop label. Caller: onInitRegion (sets deferHeadBalanceThisRegion_).
    static bool prefixContainsAnyLoopHead(IRList::iterator                    blockBegin,
                                          IRList::iterator                    regionStart,
                                          const std::unordered_set<std::string>& headLabels)
    {
        if(headLabels.empty())
            return false;
        for(auto it = blockBegin; it != regionStart; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(!isLabel(inst))
                continue;
            auto* ld = inst.getModifier<LabelData>();
            if(ld && headLabels.count(ld->label) != 0)
                return true;
        }
        return false;
    }

    // Scheduling rule (5): scan one BB — for each backward edge, if lastNonLabelBeforeIsWmma,
    // record branch target in *outHeadLabels and set *outAnyTailWmma. Caller: onInit →
    // deferFirstHeadWmmaActive_ / loopHeadLabelsForBalance_. No hardcoded label strings.
    static void analyzeLoopTailWmmaAndHeadLabels(IRList::iterator          bbBegin,
                                                 IRList::iterator          bbEnd,
                                                 bool*                     outAnyTailWmma,
                                                 std::unordered_set<std::string>* outHeadLabels)
    {
        // Diagrams / IR-vs-DAG notes below. Steps 1–4 and **Presumptions** (tail position vs
        // head WMMA earliness): see the block comment immediately after this function.
        //
        // One basic block, linear order (bbBegin -> bbEnd):
        //
        //   bbBegin
        //      |
        //      v
        //   L_loop:                    <--- branch target (recorded in *outHeadLabels)
        //      |
        //      ... body (may include more WMMA) ...
        //      |
        //      v
        //   [WMMA]  <--- last real insn before branch (lastNonLabelBeforeIsWmma)
        //      |
        //      v
        //   branch  --->  L_loop       <--- backward edge: target is *earlier* in block
        //      |              |
        //   bbEnd          (distHead < distBr)
        //
        // If that pattern matches, set *outAnyTailWmma and insert "L_loop" into *outHeadLabels.
        // Forward branches (target below branch) and branches whose tail is not WMMA are ignored.
        //
        // IR order vs DAG (people often draw the DAG "top -> down" for dependencies; that is
        // **not** the same axis as scrolling the basic block top -> bottom):
        //
        //   Linear IR list (bbBegin ----> bbEnd)     Def-use DAG (producer above consumer)
        //   ------------------------------------     -------------------------------
        //   L_loop:                                  WMMA_h at head may only depend on
        //      |                                     e.g. ds_load / VALU from *before* WMMA_t
        //      v                                     in the same iteration — not on WMMA_t.
        //   ... body ...
        //      |
        //      v
        //   WMMA_t  <--- "last" = **previous opcode in the IR list** before the branch
        //      |         (walk backward, skip labels). Not "last in DAG topo order" and not
        //      v         "last WMMA to fire" in a simulated schedule — purely static layout.
        //   branch  ---> L_loop
        //
        //   Backward-edge test uses **indices in the IR list** (distHead < distBr), not DAG
        //   depth. Classification runs on this static shape; pickOne then applies deferral
        //   when scheduling the region after the head label.
        //
        // Why this avoids two WMMAs "stuck together" across the back-edge (ASCII):
        //
        //   iteration N (bottom of loop)              iteration N+1 (after branch)
        //   --------------------------                ---------------------------
        //          ... body ...
        //               |
        //               v
        //          +-----------+
        //          |  WMMA_t   |  last real insn before branch (tail)
        //          +-----------+
        //               |
        //               +------ branch to L_loop ----->  (jump to earlier label)
        //               v
        //          +-----------+
        //          |  WMMA_h   |  first WMMA at loop head — operands often ready from *before*
        //          +-----------+  WMMA_t; register DAG typically has NO edge WMMA_t --> WMMA_h
        //
        //   Greedy Phase A (no rule 5):     ... [ WMMA_t ][ WMMA_h ]   <- consecutive issue
        //
        //   Rule (5) after head label:      ... [ WMMA_t ][ VALU/DS/tensor/... ][ WMMA_h ]
        //                                        pickOne blocks WMMA while other* queues work
        //
        *outAnyTailWmma = false;
        outHeadLabels->clear();

        std::map<std::string, IRList::iterator> labelToFollowingInst;
        for(auto it = bbBegin; it != bbEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(!isLabel(inst))
                continue;
            auto* ld = inst.getModifier<LabelData>();
            if(!ld)
                continue;
            labelToFollowingInst[ld->label] = std::next(it);
        }

        for(auto it = bbBegin; it != bbEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(!isBranch(inst))
                continue;
            if(inst.getSrcRegs().empty()
               || inst.getSrcRegs()[0].dataType != StinkyRegister::Type::LiteralString)
                continue;
            const std::string tgt = getBranchTarget(inst);
            auto              fit = labelToFollowingInst.find(tgt);
            if(fit == labelToFollowingInst.end())
                continue;
            const long distHead = std::distance(bbBegin, fit->second);
            const long distBr   = std::distance(bbBegin, it);
            if(distHead >= distBr)
                continue; // not a backward edge within this block
            if(!lastNonLabelBeforeIsWmma(bbBegin, it))
                continue;

            *outAnyTailWmma = true;
            outHeadLabels->insert(tgt);
        }
    }

    // Rule (5) — loop tail WMMA / head WMMA: numbered pipeline
    //
    // Step 1 — Identify the pattern in the basic block (onInit → analyzeLoopTailWmmaAndHeadLabels).
    //   • Build a map: each LABEL’s name → iterator to the first instruction *after* that label.
    //   • Walk every branch. If the target label resolves to an instruction *earlier* in the same
    //     BB (distHead < distBr), it is a same-BB backward edge.
    //   • For each backward edge, call lastNonLabelBeforeIsWmma: the first real (non-LABEL)
    //     instruction immediately before the branch in **linear IR order** must be WMMA/SWMMA.
    //   • For every match, insert the branch target name into *outHeadLabels and set *outAnyTailWmma.
    //     onInit copies that into loopHeadLabelsForBalance_ and deferFirstHeadWmmaActive_.
    //   • If loopConfig.unrollGemm is false, onInit/onInitRegion return early — none of the
    //     following steps run.
    //
    // Step 2 — Decide which scheduling regions are affected (onInitRegion).
    //   • deferHeadBalanceThisRegion_ = deferFirstHeadWmmaActive_ &&
    //     prefixContainsAnyLoopHead(blockBegin, regionStart, loopHeadLabelsForBalance_).
    //   • So only regions whose linear prefix (from BB start up to regionStart) already contains
    //     a recorded loop-head label participate. Preloop code before the label is excluded.
    //
    // Step 3 — While picking ready nodes (pickOne, Phase A).
    //   • blockWmmaForLoopHeadBalance = deferHeadBalanceThisRegion_ && deferFirstHeadWmmaActive_
    //     && otherQueuesHaveWork (global / local / other still non-empty).
    //   • When true, WMMA is not taken even if the DAG says it is ready: non-WMMA work is
    //     scheduled first, so tail WMMA (end of iteration N) and head WMMA (iteration N+1)
    //     are not issued back-to-back. (ASCII inside analyzeLoopTailWmmaAndHeadLabels shows why.)
    //
    // Step 4 — Stop deferring for the rest of the BB after one head WMMA (pickOneFromWMMA).
    //   • If deferHeadBalanceThisRegion_ was set when this WMMA is issued, clear
    //     deferFirstHeadWmmaActive_. One balanced head WMMA is enough; later WMMAs in the BB
    //     are not forced through the same deferral.
    //
    // Presumptions — tail WMMA *position* vs how *early* head WMMA can issue:
    //
    //   • Head WMMA readiness is decided by the **def-use DAG** (operands from the previous
    //     iteration’s loads/VALU, etc.), not by “how far down the body” the tail WMMA sits.
    //     So the first head WMMA can become ready **very early** in dynamic order even if the
    //     tail WMMA is **late** in static IR — there is no edge tail→head WMMA, so nothing in
    //     the DAG forces head WMMA to wait for tail WMMA to finish.
    //
    //   • This heuristic does **not** measure distance (instruction count or cycles) from the
    //     loop head to the tail WMMA, and does **not** scale deferral from that distance. Step 1
    //     only checks a **boolean shape**: the real insn **immediately before** the backward
    //     branch is WMMA. If VALU or other ops appear after the last WMMA but before the branch,
    //     the pattern is **not** detected (false negative for that loop shape).
    //
    //   • How we **do** cope without a positional model: Step 3 is deliberately coarse — while
    //     deferral is active and global/local/other still have **any** ready node, Phase A skips
    //     WMMA. The **amount** of separation before the first head WMMA is therefore “whatever
    //     non-WMMA work the DAG currently exposes as ready,” not a computed gap from tail WMMA’s
    //     line number. If little non-WMMA work is ready, otherQueuesHaveWork becomes false and
    //     head WMMA may still issue soon; rules (2)(3)(4) continue to gate WMMA independently.

    // -------------------------------------------------------------------------
    // CDNA5ReadyQueue — WMMA scheduling policy (Gfx1250)
    // -------------------------------------------------------------------------
    //
    // Scheduling menu (every pick still respects the DAG: in-degree 0 only):
    //
    //  (1) Program order vs WMMA — prefer WMMA in Phase A, but not before pickable
    //      global/tensor/local/other with a smaller DAG id (preload / X0 vs X1 double-buffer).
    //  (2) DS / VGPR latency — block WMMA until modeled ds_load latency for WMMA src VGPRs
    //      has decayed; seed from the BB prefix before each region.
    //  (3) WMMA–WMMA spacing — after each WMMA pick, sibling WMMAs wait (per-node counters)
    //      until mandated gap cycles have been “spent” by issued instructions.
    //  (4) Density + cross-region gap — avgIssueInterval from region WMMA/DS/issue totals;
    //      wmmaPipelineGapCycles survives region splits so WMMAs do not pack only because
    //      micro-regions reset per-node spacing.
    //  (5) Loop tail vs head — defer first WMMA after the loop-head label until non-WMMA queues
    //      drain once. Steps 1–4 + **Presumptions** (tail position vs head WMMA earliness; what is
    //      not modeled) are in the block comment after analyzeLoopTailWmmaAndHeadLabels; diagrams
    //      are inside that function. Skipped when loopConfig.unrollGemm is false.
    //
    // Where rules are implemented (grep these function names in this file, then read the block
    // comment above each body for state variables and callees):
    //
    //   pickOne                 — main orchestration: (1)(2)(3)(4)(5); (5) Step 3 blockWmmaForLoopHeadBalance
    //   pickOneFromWMMA         — (3)(4) inter-Wmma delay + wmmaPipelineGapCycles; (5) Step 4 clear deferral
    //   updateWMMAStatus        — (2)(3)(4) time step for counters after any picked insn
    //   isWMMALatencyFree       — (2) gate before WMMA
    //   findSmallestPickableNonWmma — (1) min-id non-WMMA vs WMMA program order
    //   push                    — (1) bucket routing into ready queues
    //   onInit                  — (4) gap reset + WMMA latency snap; (5) Step 1 stores loop-head set
    //   onInitRegion            — (2) prefix DS seed; (4) WMMAIssueConfig budget; (5) Step 2 defer flag
    //   seedWmmaDsLatencyFromPrefix — (2) prefix-only DS latency → wmmaRegisterLatencyCounters
    //   analyzeLoopTailWmmaAndHeadLabels — (5) Step 1 pattern scan (uses lastNonLabelBeforeIsWmma)
    //   prefixContainsAnyLoopHead — (5) Step 2 helper (region prefix vs recorded head labels)
    //
    // Temporarily named CDNA5 for MI450-class hardware; rename when marketing settles.
    //
    class CDNA5ReadyQueue : public ReadyQueue
    {
        // --- Priority buckets (DAG ids compare smaller = earlier in source within region) ---
        DAGidPriorityQueue wmmaQueue;
        DAGidPriorityQueue globalReadQueue; // tensor_load_to_lds when distributeGlobalRead
        DAGidPriorityQueue localReadQueue; // reserved; same priority scheme as CDNA3
        DAGidPriorityQueue barrierQueue;
        DAGidPriorityQueue otherQueue; // scalars, ds_load, waits in region, etc.

        // Throttle tensor issues vs other work (mirrors CDNA3 globalReadPerMFMA idea).
        int globalReadCounter = 0;
        int globalReadPerWMMA = 1;

        // --- WMMA interlock state (decayed by updateWMMAStatus on every picked instruction) ---

        // Per other WMMA DAG node: cycles until that WMMA may be considered for priority 1.
        std::map<DAGNode*, int> wmmaNodeCounters;

        // Per VGPR index: remaining modeled latency until ds_load result is safe for WMMA src.
        std::map<int, int> wmmaRegisterLatencyCounters;

        // Region snapshot for pickOneFromWMMA delay math (totalIssuedCycles is also decremented
        // each pick so "rest" shrinks as the region is scheduled).
        WMMAIssueConfig wmmaIssueConfig;

        // Global WMMA gap: unlike wmmaNodeCounters, survives onInitRegion so waitcnt boundaries
        // do not erase spacing between consecutive WMMAs. Forced pick at end of pickOne bypasses
        // the priority-1 check that requires this to be 0.
        int wmmaPipelineGapCycles = 0;

        // After issuing WMMA, skip priority-1 WMMA once if global/local/other still have nodes,
        // so we do not string WMMAs when scalar/tensor work is still ready.
        bool lastPickedWasWMMA = false;

        // --- Loop head balancing (computed once per BB in onInit) ---
        bool deferFirstHeadWmmaActive_ = false; // true after analyze if any tail-WMMA back-edge
        std::unordered_set<std::string> loopHeadLabelsForBalance_; // branch targets to match
        bool deferHeadBalanceThisRegion_ = false; // per onInitRegion: prefix saw a head label

        void     updateWMMAStatus(DAGNode* node);
        bool     isWMMALatencyFree(DAGNode* node);
        DAGNode* pickOneFromWMMA();
        bool     findSmallestPickableNonWmma(DAGNode** outNode, int* kindOut) const;

    public:
        explicit CDNA5ReadyQueue(const PassContext& passCtx)
            : ReadyQueue(passCtx)
        {
        }

        DAGNode* pickOne() override;
        void     push(DAGNode* node) override;
        bool     empty() const override;

        void onInit(IRList::iterator regionStart, IRList::iterator regionEnd) override;

        void onInitRegion(IRList::iterator regionStart,
                          IRList::iterator regionEnd,
                          IRList::iterator blockBegin) override;
    };

    // Scheduling rules (2)(3)(4): after any picked instruction, advance modeled time — decay
    // wmmaPipelineGapCycles, wmmaNodeCounters, wmmaRegisterLatencyCounters; subtract cycles from
    // wmmaIssueConfig.totalIssuedCycles. Callers: pickOne (all paths), pickOneFromWMMA.
    void CDNA5ReadyQueue::updateWMMAStatus(DAGNode* node)
    {
        const int cycles = node->inst->issueCycles;

        // Every scheduled insn advances "time" for WMMA gap and WMMA/DS countdowns alike.
        wmmaPipelineGapCycles = std::max(0, wmmaPipelineGapCycles - cycles);

        wmmaIssueConfig.totalIssuedCycles -= cycles;

        // Decrement per-WMMA issue interval counters.
        for(auto& [n, counter] : wmmaNodeCounters)
        {
            if(counter > 0)
                counter -= cycles;
        }

        // Decrement ds_read latency per VGPR; remove when latency reaches 0 (VGPR ready).
        for(auto it = wmmaRegisterLatencyCounters.begin(); it != wmmaRegisterLatencyCounters.end();)
        {
            if(it->second > 0)
            {
                it->second -= cycles;
                if(it->second <= 0)
                    it = wmmaRegisterLatencyCounters.erase(it);
                else
                    ++it;
            }
            else
                ++it;
        }
    }

    // Scheduling rule (2): gate WMMA on wmmaRegisterLatencyCounters — any WMMA src VGPR with
    // outstanding modeled ds_load latency blocks issue. Operand walk: getSrcRegs per node->inst.
    bool CDNA5ReadyQueue::isWMMALatencyFree(DAGNode* node)
    {
        // WMMA src operands (A/B/acc bundles) share getSrcRegs in our IR; any VGPR still in the
        // DS latency map with >0 blocks issue so WMMA does not start while loads are in flight.
        for(const StinkyRegister& dstReg : node->inst->getSrcRegs())
        {
            if(!dstReg.isRegister())
                continue;

            for(unsigned off = 0; off < dstReg.reg.num; ++off)
            {
                auto it = wmmaRegisterLatencyCounters.find(dstReg.reg.idx + off);
                if(it != wmmaRegisterLatencyCounters.end() && it->second > 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    // Scheduling rules (3)(4): compute delay from wmmaIssueConfig (rest, avgIssueInterval,
    // latencyCycles); raise wmmaNodeCounters on sibling WMMAs; set wmmaPipelineGapCycles.
    // Rule (5): if deferHeadBalanceThisRegion_, first pick clears deferFirstHeadWmmaActive_.
    // Then updateWMMAStatus; resets globalReadCounter. Callers: pickOne Phase A / Phase D.
    DAGNode* CDNA5ReadyQueue::pickOneFromWMMA()
    {
        assert(!wmmaQueue.empty() && "The WMMA queue must not be empty");
        DAGNode* node = wmmaQueue.top();
        wmmaQueue.pop();

        // Push out sibling WMMAs: delay is derived from avgIssueInterval and remaining non-WMMA
        // slack (rest) in the region snapshot, similar in spirit to CDNA3 MFMA spacing.
        int  delay;
        auto rest
            = (int)((wmmaIssueConfig.totalIssuedCycles - wmmaIssueConfig.totalWmmaIssuedCycles)
                    / wmmaIssueConfig.issuedCount);
        if(wmmaIssueConfig.issuedCount <= 0
           || (wmmaIssueConfig.issuedCount > 0 && rest < node->inst->latencyCycles))
        {
            if(node->inst->latencyCycles >= wmmaIssueConfig.avgIssueInterval)
                delay = std::max(rest, 1);
            else
                delay = node->inst->latencyCycles;
        }
        else
            delay = wmmaIssueConfig.avgIssueInterval;

        for(auto& [n, counter] : wmmaNodeCounters)
        {
            if(n != node)
                counter = std::max(counter, delay);
        }
        wmmaNodeCounters.erase(node);

        wmmaPipelineGapCycles = std::max(wmmaPipelineGapCycles, delay);

        wmmaIssueConfig.issuedCount--;
        wmmaIssueConfig.totalWmmaIssuedCycles -= node->inst->issueCycles;
        // One balanced head WMMA is enough to drop loop-head deferral for the rest of the BB.
        if(deferHeadBalanceThisRegion_)
            deferFirstHeadWmmaActive_ = false;
        updateWMMAStatus(node);
        globalReadCounter = 0;
        return node;
    }

    // Scheduling rule (1): pick minimum DAG id among ready non-WMMA nodes. Queues: globalReadQueue
    // (throttled by globalReadCounter vs globalReadPerWMMA), localReadQueue, otherQueue.
    bool CDNA5ReadyQueue::findSmallestPickableNonWmma(DAGNode** outNode, int* kindOut) const
    {
        // Among currently pickable global (if throttle allows), local, and other nodes, return
        // the one with minimum DAG id — stable tie-break that matches source order when the DAG
        // leaves multiple nodes ready.
        *outNode = nullptr;
        *kindOut = -1;
        DAGNode* best = nullptr;
        int      kind = -1;

        if(!globalReadQueue.empty()
           && (globalReadCounter < globalReadPerWMMA || otherQueue.empty()))
        {
            best = globalReadQueue.top();
            kind = 0;
        }
        if(!localReadQueue.empty())
        {
            DAGNode* t = localReadQueue.top();
            if(!best || t->id < best->id)
            {
                best = t;
                kind = 1;
            }
        }
        if(!otherQueue.empty())
        {
            DAGNode* t = otherQueue.top();
            if(!best || t->id < best->id)
            {
                best = t;
                kind = 2;
            }
        }

        if(!best)
            return false;
        *outNode = best;
        *kindOut = kind;
        return true;
    }

    // Main scheduling orchestration — rules (1)–(5):
    //   Phase A: WMMA if wmmaNodeCounters, wmmaPipelineGapCycles, isWMMALatencyFree, programOrderOk
    //            (vs findSmallestPickableNonWmma), lastPickedWasWMMA / otherQueuesHaveWork, and
    //            !blockWmmaForLoopHeadBalance (rule 5).
    //   Phase B: pop non-WMMA; on ds_load, extend wmmaRegisterLatencyCounters (rule 2).
    //   Phase C: barriers. Phase D: forced WMMA via pickOneFromWMMA (skips some Phase A gates).
    DAGNode* CDNA5ReadyQueue::pickOne()
    {
        // Phase A — try WMMA first if all WMMA-specific gates pass.
        // Phase B — smallest-id non-WMMA among global/local/other.
        // Phase C — barriers when the min-id queue is empty but barriers remain ready.
        // Phase D — forced WMMA (only WMMA left in the ready set for this step).

        bool otherQueuesHaveWork
            = !globalReadQueue.empty() || !localReadQueue.empty() || !otherQueue.empty();

        DAGNode* smallestPickable = nullptr;
        int      pickKind         = -1;
        findSmallestPickableNonWmma(&smallestPickable, &pickKind);

        if(!wmmaQueue.empty())
        {
            DAGNode* top = wmmaQueue.top();
            int      c   = wmmaNodeCounters.count(top) ? wmmaNodeCounters.at(top) : 0;
            bool programOrderOk
                = smallestPickable == nullptr || top->id < smallestPickable->id;
            // Rule (5): avoid WMMA at loop tail then WMMA as first work after the head label.
            const bool blockWmmaForLoopHeadBalance
                = deferHeadBalanceThisRegion_ && deferFirstHeadWmmaActive_ && otherQueuesHaveWork;
            if(c <= 0 && wmmaPipelineGapCycles <= 0 && isWMMALatencyFree(top) && programOrderOk
               && (!lastPickedWasWMMA || !otherQueuesHaveWork) && !blockWmmaForLoopHeadBalance)
            {
                DAGNode* node     = pickOneFromWMMA();
                lastPickedWasWMMA = true;
                return node;
            }
        }

        if(smallestPickable != nullptr)
        {
            DAGNode* node = nullptr;
            if(pickKind == 0)
            {
                node = globalReadQueue.top();
                globalReadQueue.pop();
                globalReadCounter++;
            }
            else if(pickKind == 1)
            {
                node = localReadQueue.top();
                localReadQueue.pop();
            }
            else
            {
                assert(pickKind == 2);
                node = otherQueue.top();
                otherQueue.pop();
                if(isDSRead(*node->inst))
                {
                    for(const StinkyRegister& dstReg : node->inst->getDestRegs())
                    {
                        if(!dstReg.isRegister())
                            continue;
                        for(unsigned off = 0; off < dstReg.reg.num; ++off)
                            wmmaRegisterLatencyCounters[dstReg.reg.idx + off]
                                = node->inst->latencyCycles;
                    }
                }
            }
            updateWMMAStatus(node);
            lastPickedWasWMMA = false;
            return node;
        }

        if(!barrierQueue.empty())
        {
            DAGNode* barrier = barrierQueue.top();
            barrierQueue.pop();
            updateWMMAStatus(barrier);
            lastPickedWasWMMA = false;
            return barrier;
        }

        DAGNode* node     = pickOneFromWMMA();
        lastPickedWasWMMA = true;
        return node;
    }

    // Scheduling rule (1): route ready DAG nodes into wmmaQueue, globalReadQueue (tensor_load when
    // distributeGlobalRead), barrierQueue, or otherQueue. pickOne drains buckets; order here is not schedule order.
    void CDNA5ReadyQueue::push(DAGNode* node)
    {
        // Route ready nodes into buckets. Order here does not imply schedule order; pickOne
        // implements the actual priority and min-id policy.
        if(isWMMA(*node->inst))
        {
            wmmaNodeCounters[node] = 0; // pickOneFromWMMA may raise before this WMMA becomes top
            wmmaQueue.push(node);
            return;
        }

        if(getPassContext().getPassFeatureConfig().dagFeatures.distributeGlobalRead
           && isTensorLoad(*node->inst))
        {
            globalReadQueue.push(node);
            return;
        }

        if(isBarrier(*node->inst))
        {
            barrierQueue.push(node);
            return;
        }

        otherQueue.push(node);
    }

    // ReadyQueue API: true when all scheduling buckets are empty (no rule logic).
    bool CDNA5ReadyQueue::empty() const
    {
        return wmmaQueue.empty() && globalReadQueue.empty() && otherQueue.empty()
               && barrierQueue.empty() && localReadQueue.empty();
    }

    // Once per BB (StinkyDAGSchedulerPass passes bb bounds). Rule (4): reset wmmaPipelineGapCycles;
    // set wmmaIssueConfig.latency from first WMMA/SWMMA in block. Rule (5): analyzeLoopTailWmmaAndHeadLabels
    // → deferFirstHeadWmmaActive_, loopHeadLabelsForBalance_. Skipped when loopConfig.unrollGemm is false.
    void CDNA5ReadyQueue::onInit(IRList::iterator regionStart, IRList::iterator regionEnd)
    {
        // Full basic block [regionStart, regionEnd) — StinkyDAGSchedulerPass passes bb bounds.
        deferFirstHeadWmmaActive_ = false;
        loopHeadLabelsForBalance_.clear();
        deferHeadBalanceThisRegion_ = false;

        if(getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false)
            return;

        wmmaPipelineGapCycles = 0;

        analyzeLoopTailWmmaAndHeadLabels(
            regionStart, regionEnd, &deferFirstHeadWmmaActive_, &loopHeadLabelsForBalance_);
        // deferFirstHeadWmmaActive_ true => defer first post-head WMMA until non-WMMA wave
        // drains, but only in regions whose prefix contains a recorded head (see onInitRegion).

        wmmaIssueConfig.latency = 0;
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isWMMA(inst) || isSWMMA(inst))
            {
                wmmaIssueConfig.latency = inst.latencyCycles;
                break;
            }
        }

    }

    // Per scheduling region. Rule (2): seedWmmaDsLatencyFromPrefix(blockBegin, regionStart, …).
    // Rule (4): scan region into wmmaIssueConfig (totalIssuedCycles, WMMA counts, totalDSLatency,
    // avgIssueInterval). Rule (5): deferHeadBalanceThisRegion_ via prefixContainsAnyLoopHead.
    // Resets lastPickedWasWMMA. Skipped when loopConfig.unrollGemm is false.
    void CDNA5ReadyQueue::onInitRegion(IRList::iterator regionStart,
                                       IRList::iterator regionEnd,
                                       IRList::iterator blockBegin)
    {
        // Per scheduling region (between non-movable side effects). blockBegin is the BB start
        // so prefix seeding and loop-head detection see preloop / prior regions in file order.
        lastPickedWasWMMA = false;
        if(getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false)
            return;

        deferHeadBalanceThisRegion_
            = deferFirstHeadWmmaActive_
              && prefixContainsAnyLoopHead(blockBegin, regionStart, loopHeadLabelsForBalance_);

        seedWmmaDsLatencyFromPrefix(blockBegin, regionStart, wmmaRegisterLatencyCounters);

        wmmaIssueConfig.totalIssuedCycles     = 0;
        wmmaIssueConfig.totalWmmaIssuedCycles = 0;
        wmmaIssueConfig.issuedCount           = 0;
        int totalDSLatency                    = 0;
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);

            wmmaIssueConfig.totalIssuedCycles += inst.issueCycles;

            if(isDSRead(inst))
                totalDSLatency += inst.latencyCycles;

            if(isWMMA(inst) || isSWMMA(inst))
            {
                wmmaIssueConfig.issuedCount += 1;
                wmmaIssueConfig.totalWmmaIssuedCycles += inst.issueCycles;
            }
        }
        // Budget per WMMA (avgIssueInterval): spread WMMAs when the region is DS- or issue-heavy.
        //
        //   totalIssuedCycles   |========================|  all insn issueCycles in region
        //   totalDSLatency      |==============================|  sum of ds_load latencyCycles
        //                                    |
        //                         numer = max(both)   <-- whichever rail is longer wins
        //                                    |
        //                                    v
        //              totalAvgLatency = ceil( numer / issuedCount )   "room" per WMMA slot
        //
        //   wmmaIssueConfig.latency   |===|  one WMMA op's latencyCycles (floor on spacing)
        //
        //   avgIssueInterval = max(totalAvgLatency, latency)
        //   pickOneFromWMMA uses avgIssueInterval so sibling WMMAs wait longer when DS dominates.
        //
        int totalAvgLatency;
        if(wmmaIssueConfig.issuedCount <= 0)
            totalAvgLatency = wmmaIssueConfig.latency;
        else
            totalAvgLatency = (int)std::ceil(
                (float)std::max(wmmaIssueConfig.totalIssuedCycles, totalDSLatency)
                / wmmaIssueConfig.issuedCount);
        if(totalAvgLatency > wmmaIssueConfig.latency)
        {
            wmmaIssueConfig.avgIssueInterval = totalAvgLatency;
        }
        else
        {
            wmmaIssueConfig.avgIssueInterval = wmmaIssueConfig.latency;
        }
    }
}
