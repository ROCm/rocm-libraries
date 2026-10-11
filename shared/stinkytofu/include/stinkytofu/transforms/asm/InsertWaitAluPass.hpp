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
#pragma once

#include <cstdint>
#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class ModulePass;
class PassContext;
class Function;
struct StinkyInstruction;

/// Function metadata for drivers that cut a region out of a kernel: 1 when the
/// function starts at the kernel's entry point, 0 when it starts mid-kernel
/// (e.g. stinkytofu-opt --from-label past the first instruction). Absent means
/// the function is the kernel entry. RemoveWaitAlu and InsertWaitAlu use it to
/// leave SCHED_MODE alone in a region that does not own the kernel's entry.
inline constexpr const char* kFuncStartsAtKernelEntryMetaKey = "stinkytofu.startsAtKernelEntry";

/// Set by RemoveWaitAlu when the input contains a pre-existing s_wait_alu +
/// waitcnt bundle. Under per-pipe VA_VDST semantics that placement is part of
/// the input schedule: removing and reconstructing waits cannot prove an
/// equivalent profitable placement. InsertWaitAlu honors the marker so the
/// remove/insert pair is a conservative function-level no-op.
inline constexpr const char* kPreserveWaitAluScheduleMetaKey = "stinkytofu.preserveWaitAluSchedule";

/// True when the selected part of a function contains a regenerated
/// VA_VDST/VM_VSRC wait immediately bundled with a waitcnt.
STINKYTOFU_EXPORT bool hasRegeneratedFieldWaitBundle(Function& func, PassContext& passCtx);

/// Where InsertWaitAlu writes setreg(SCHED_MODE.DEP_MODE) = 2.
enum class SchedModeLifecycle : uint8_t {
    /// At the kernel entry label(s), but not in a function that starts
    /// mid-kernel or already writes DEP_MODE itself.
    Auto,
    /// At the entry label(s), or the function entry, unconditionally.
    Always,
    /// Never; the input owns the mode.
    Never,
};

/// Pass policy.
///
/// Baseline is the per-pipe / per-FIFO count alone; each refinement is enabled per arch.
struct InsertWaitAluOptions {
    /// Stamp VALU source operands as well as dests, for the src-operand WAR hazard.
    bool enableESM2TrackValuVsrc = false;
    /// Count a CSMACC producer's followers across the VA order, and retire it at the shared floor.
    bool sharedOrderCountFollowers = false;
    /// Count an XDL producer's followers from the nearest anchor rather than its own ticket.
    bool xdlCountFromNextWmma = false;
    SchedModeLifecycle schedModeLifecycle = SchedModeLifecycle::Auto;
    /// Skip a va_vdst / vm_vsrc wait once HWModel::WaitHide's count of later matrix ops
    /// or memory reads has issued. That is a timing assumption the ISA does not make:
    /// waits are mandatory between counter-managed instructions (per the ISA).
    bool waitHide = false;
};

/// Insert s_wait_alu instructions for SCHED_MODE 2 (VA_VDST + VM_VSRC).
///
/// Function pass: full scoreboard analysis when run on the entry, conservative
/// entry drain when run on a callable function. Used by stinkytofu-opt single-pass
/// mode and unit tests.
/// The options the gfx1250 pipeline runs InsertWaitAlu with. A scheduler querying
/// WaitAluTracker must use the same ones to predict the waits the pass will emit.
///
/// By default va_vdst is counted per pipe: a wait proves a VALU write complete only
/// once enough later ops issued on its own pipe, as asm_verify checks.
/// assumeInOrderVaVdstAcrossPipes enables sharedOrderCountFollowers and
/// xdlCountFromNextWmma, which assume VA_VDST retires in issue order across pipes, so
/// a wait on a later WMMA also retires an earlier VALU. Do not enable it on gfx1250:
/// gfx1250_va_vdst_cross_pipe shows the counter does not provide that ordering.
inline InsertWaitAluOptions gfx1250InsertWaitAluOptions(bool enableESM2TrackValuVsrc,
                                                        bool assumeInOrderVaVdstAcrossPipes) {
    return {enableESM2TrackValuVsrc, /*sharedOrderCountFollowers=*/assumeInOrderVaVdstAcrossPipes,
            /*xdlCountFromNextWmma=*/assumeInOrderVaVdstAcrossPipes};
}

/// The s_wait_alu InsertWaitAlu would emit before an instruction; -1 = no wait on that
/// counter.
struct WaitAluNeed {
    int vaVdst = -1;
    int vmVsrc = -1;
    bool any() const {
        return vaVdst >= 0 || vmVsrc >= 0;
    }
};

/// InsertWaitAlu's scoreboard stepped one instruction at a time, so a scheduler can ask
/// what s_wait_alu a candidate would need at the current point of its schedule. Uses the
/// pass's own walk; it starts from an empty state, where the pass seeds a BB from its
/// predecessors.
class STINKYTOFU_EXPORT WaitAluTracker {
   public:
    WaitAluTracker(const PassContext& passCtx, InsertWaitAluOptions opts);
    ~WaitAluTracker();
    WaitAluTracker(const WaitAluTracker&) = delete;
    WaitAluTracker& operator=(const WaitAluTracker&) = delete;

    /// The wait `inst` would need if it were issued next.
    WaitAluNeed query(const StinkyInstruction& inst) const;
    /// Issue `inst`: apply the wait it needs, then record what it produces.
    void commit(const StinkyInstruction& inst);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertWaitAluPass(InsertWaitAluOptions opts = {});

/// Whole-kernel driver: full analysis on the entry function, then the conservative
/// call-boundary drain on every callee. Reserves a seam for future caller<->callee
/// analysis.
STINKYTOFU_EXPORT std::unique_ptr<ModulePass> createInsertWaitAluModulePass(
    InsertWaitAluOptions opts = {});

}  // namespace stinkytofu
