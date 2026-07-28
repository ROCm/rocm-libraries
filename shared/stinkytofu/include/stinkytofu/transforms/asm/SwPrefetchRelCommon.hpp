/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

/// Global byte offsets for SW prefetch (128*255, then every 32*128).
inline constexpr int64_t kSwPrefetchFirstGlobalByte = int64_t(128) * 255;
inline constexpr int64_t kSwPrefetchSpacingBytes = int64_t(32) * 128;

/// klength simm5: 31 => 32 instruction cache lines (128 B each).
inline constexpr int32_t kSwPrefetchPcRelKlengthImm = 31;

/// Grid boundary P(k) = kSwPrefetchFirstGlobalByte + k * kSwPrefetchSpacingBytes.
inline constexpr int64_t swPrefetchGridOffset(int64_t k) {
    return kSwPrefetchFirstGlobalByte + k * kSwPrefetchSpacingBytes;
}

/// Sentinel: `SwPrefetchRelPhase1Accum::firstPostCpLayoutByte` when the BB has no post-CP bytes.
inline constexpr int64_t kSwPrefetchNoPerBbGridAnchor = int64_t(-1);

/// Per-BB anchored grid: `P_bb(localK) = bbAnchorGlobal + localK * kSwPrefetchSpacingBytes`
/// (4 KiB steps from the first post-CP byte in the BB). \p bbAnchorGlobal must be ≥ `P(0)` when
/// valid; see §15 of SwInstructionPrefetchRelDynamicPass-Design.md. Original dynamic pass uses
/// global `swPrefetchGridOffset(k)` instead; both are global layout coordinates for ISA lowering.
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

/// Phase 1 (dynamic pass): layout map + CFG post-CP accum (front-edge Phi-max). No IR mutation.
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

/// Walk all BBs in function list order (layout), then CFG RPO for `accumByte` (post-32640,
/// front-edge predecessors only; loop back-edges excluded).
STINKYTOFU_EXPORT void computeSwPrefetchRelPhase1Accum(
    Function& func, const std::unordered_map<std::string, int64_t>* asmSetSymbols,
    SwPrefetchRelPhase1Accum& out, std::ostream* dbgOut = nullptr,
    const char* debugPassTag = "SwInstructionPrefetchRelDynamicPass",
    bool phase2UsesPerBbAnchorGrid = false);

/// CFG-gated grid walk (dual gate §2.3): emit `s_prefetch_inst_pc_rel` when `cfgGate` holds.
/// \p bbEntryAccum is Phase-1 `accumByte[bb]`. \p kNextIn is usually 0 (per-BB sweep §4.3).
/// Returns number of prefetches inserted in this BB.
STINKYTOFU_EXPORT int insertSwPrefetchLabelsDynamic(
    BasicBlock& bb, int64_t blockGlobalByteOffset, int64_t bbEntryAccum, int64_t kNextIn,
    GfxArchID archId, std::ostream* dbgOut,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols,
    bool allowSwPrefetchInsertion = true,
    const char* debugPassTag = "SwInstructionPrefetchRelDynamicPass");

/// Alternate Phase-2 pipeline: same CFG dual gate (with **closed-left** match when
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
