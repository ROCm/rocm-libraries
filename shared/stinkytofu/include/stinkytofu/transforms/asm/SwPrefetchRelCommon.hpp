// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"

namespace stinkytofu {
class BasicBlock;
class Function;
struct StinkyInstruction;

// TODO(next PR): move these gfx1250 HW constants into Gfx1250Formats.def and derive them
// from there; they are duplicated here only to keep this PR self-contained.
/// Global byte offsets for SW prefetch (128*255, then every 32*128).
inline constexpr int64_t kSwPrefetchFirstGlobalByte = int64_t(128) * 255;
inline constexpr int64_t kSwPrefetchSpacingBytes = int64_t(32) * 128;

/// klength simm5: 31 => 32 instruction cache lines (128 B each).
inline constexpr int32_t kSwPrefetchPcRelKlengthImm = 31;

/// Debug/perf-only toggle: emit `s_wait_xcnt 0` before each SW instruction prefetch
/// (relative & absolute) for gfx1250 XNACK safety. A shader must drain outstanding address
/// translations before any `s_prefetch_inst[_pc_rel]`, else the prefetch can be caught in an
/// XNACK replay group. Relative inserts one wait per (scattered) prefetch; absolute inserts one
/// wait per contiguous burst.
///
/// Intentionally NOT a ModuleOption / TensileLite parameter — purely a stinkytofu-internal knob
/// for debugging and performance A/B testing. Flip to false to measure without the wait; when
/// false every gated site collapses to today's bytes exactly (zero golden churn). Also gated by
/// opcode availability (getMCIDByUOp(GFX::s_wait_xcnt) != nullptr), so it is a no-op off gfx1250.
inline constexpr bool kSwPrefetchEmitXnackWait = true;

/// s_wait_xcnt 0 encodes as a fixed 4-byte SOPP (MC_SOPP => 0 literal bytes). Compile-time size
/// contribution of the wait, folded into the abs byte-accounting constants below.
inline constexpr int64_t kSwPrefetchXnackWaitBytes = kSwPrefetchEmitXnackWait ? int64_t(4) : 0;

/// Grid boundary P(k) = kSwPrefetchFirstGlobalByte + k * kSwPrefetchSpacingBytes.
inline constexpr int64_t swPrefetchGridOffset(int64_t k) {
    return kSwPrefetchFirstGlobalByte + k * kSwPrefetchSpacingBytes;
}

/// Sentinel: `SwPrefetchRelPhase1Accum::firstPostCpLayoutByte` when the BB has no post-CP bytes.
inline constexpr int64_t kSwPrefetchNoPerBbGridAnchor = int64_t(-1);

/// Per-BB anchored grid: `P_bb(localK) = bbAnchorGlobal + localK * kSwPrefetchSpacingBytes`
/// (4 KiB steps from the first post-CP byte in the BB). \p bbAnchorGlobal must be ≥ `P(0)` when
/// valid. The original dynamic pass uses global `swPrefetchGridOffset(k)` instead; both are
/// global layout coordinates for ISA lowering.
inline int64_t swPrefetchPerBbAnchorGridOffset(int64_t localK, int64_t bbAnchorGlobal) {
    return bbAnchorGlobal + localK * kSwPrefetchSpacingBytes;
}

/// Place software prefetch (`s_prefetch_inst_pc_rel`) at fixed global byte
/// boundaries P(k), using one forward IR walk per basic block. See
/// SwInstructionPrefetchRelStaticPass.cpp design comments for anchor, getpc
/// window, and tail-flush rules.
///
/// \p blockGlobalByteOffset  Global layout offset of this BB's first byte.
/// \p allowSwPrefetchInsertion  If false, identical walk without IR mutation.
/// \p debugPassTag  Prefix for optional debug lines (e.g. pass name).
STINKYTOFU_EXPORT void insertSwPrefetchLabels(
    BasicBlock& bb, int64_t blockGlobalByteOffset, GfxArchID archId, std::ostream* dbgOut,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols,
    bool allowSwPrefetchInsertion = true,
    const char* debugPassTag = "SwInstructionPrefetchRelStaticPass");

/// Debug-only: list P(k) grid boundaries that fall in this basic block.
STINKYTOFU_EXPORT void debugPrintSwPrefetchGrid(
    std::ostream& os, const std::string& bbLabel, int64_t blockGlobalStart, int64_t blockBytes,
    const char* debugPassTag = "SwInstructionPrefetchRelStaticPass");

/// Phase 1 (dynamic pass): builds the layout map and the per-BB post-CP byte accumulation
/// (`accumByte`); each BB's value is the max over its predecessors, following forward edges only
/// (loop back-edges excluded). No IR mutation.
struct STINKYTOFU_EXPORT SwPrefetchRelPhase1Accum {
    int64_t totalLayoutBytes = 0;
    std::unordered_map<BasicBlock*, int64_t> layoutStart;
    std::unordered_map<BasicBlock*, int64_t> blockLocalBytes;
    std::unordered_map<BasicBlock*, int64_t> blockLocalBytesPostCp;
    /// Per-BB anchor `A(bb)`: global layout offset of each BB's first post-CP byte
    /// (`max(layoutBefore, P(0))` of the first insn with post-CP bytes), or
    /// `kSwPrefetchNoPerBbGridAnchor` if the BB has none. Measured from real insn layout, so
    /// internal alignment gaps that push the first post-CP insn past `layoutStart` are honored.
    std::unordered_map<BasicBlock*, int64_t> firstPostCpLayoutByte;
    std::unordered_map<BasicBlock*, int64_t> accumByte;
    std::unordered_map<BasicBlock*, int64_t> accumExit;
    /// Global layout offset of each real instruction's first byte (PHI/LABEL omitted).
    std::unordered_map<StinkyInstruction*, int64_t> layoutGlobal;
};

/// Two read-only walks: (1) all BBs in function list order to record layout offsets, then (2) a
/// single CFG reverse-post-order pass that fills `accumByte`. A BB's entry value is the max of its
/// predecessors' exit values, taken only over forward edges — every CFG edge except a loop
/// back-edge (an edge that jumps back to a loop header). Ignoring back-edges makes the propagation
/// acyclic, so one pass suffices and every reachable BB is visited exactly once (no BB is skipped;
/// unreachable BBs default to 0). This does not drop any block's own bytes: a loop body's post-CP
/// bytes are still counted when its BBs are visited — only the back-edge's contribution is not fed
/// back around the cycle.
STINKYTOFU_EXPORT void computeSwPrefetchRelPhase1Accum(
    Function& func, const std::unordered_map<std::string, int64_t>* asmSetSymbols,
    SwPrefetchRelPhase1Accum& out, std::ostream* dbgOut = nullptr,
    const char* debugPassTag = "SwInstructionPrefetchRelDynamicPass",
    bool phase2UsesPerBbAnchorGrid = false);

/// Walk the fixed prefetch grid `P(k) = 32640 + k*4096` across this BB and emit
/// `s_prefetch_inst_pc_rel` at a grid point only when two conditions both hold: (1) the point is
/// past the CP window (`P >= 32640`), and (2) it lies within this BB's accumulated post-CP
/// execution range (so each point is prefetched once, along the path that reaches it).
/// \p bbEntryAccum is Phase-1 `accumByte[bb]` (post-CP bytes accumulated before this BB).
/// \p kNextIn is the first grid index `k` not yet consumed by earlier BBs; the walk runs BB-by-BB
/// in layout order and carries `k` forward, so pass 0 only to start a fresh sweep.
/// Returns number of prefetches inserted in this BB.
STINKYTOFU_EXPORT int insertSwPrefetchLabelsDynamic(
    BasicBlock& bb, int64_t blockGlobalByteOffset, int64_t bbEntryAccum, int64_t kNextIn,
    GfxArchID archId, std::ostream* dbgOut,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols,
    bool allowSwPrefetchInsertion = true,
    const char* debugPassTag = "SwInstructionPrefetchRelDynamicPass");

/// Alternate Phase-2 pipeline: same two-condition insert test (with **closed-left** match when
/// `P == bbGridAnchorGlobal == layoutBefore` so `P_bb(0)==A==layoutStart` inserts **before** the
/// first insn at the anchor), opcode, and getpc rules as `insertSwPrefetchLabelsDynamic`, but
/// grid targets are **`bbGridAnchorGlobal + localK * kSwPrefetchSpacingBytes`** (anchor from
/// `SwPrefetchRelPhase1Accum::firstPostCpLayoutByte`), not **`32640 + k * step`**. Phase 1 /
/// `accumByte` unchanged. If `bbGridAnchorGlobal == kSwPrefetchNoPerBbGridAnchor`, inserts nothing.
STINKYTOFU_EXPORT int insertSwPrefetchLabelsDynamicPerBbAnchor(
    BasicBlock& bb, int64_t blockGlobalByteOffset, int64_t bbEntryAccum, int64_t bbGridAnchorGlobal,
    int64_t kLocalNextIn, GfxArchID archId, std::ostream* dbgOut,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols,
    bool allowSwPrefetchInsertion = true,
    const char* debugPassTag = "SwInstructionPrefetchRelDynamicPassPerBbAnchor");

}  // namespace stinkytofu
