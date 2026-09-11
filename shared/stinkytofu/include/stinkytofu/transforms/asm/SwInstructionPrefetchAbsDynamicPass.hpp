// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsStaticPass.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// SW instruction prefetch — absolute-address, dynamic (large-kernel) policy.
///
/// Background: gfx1250 has no hardware instruction prefetch. The command processor
/// (CP) preloads only the first 32640 bytes of a kernel; past that, fetching an
/// uncached 128-byte instruction line costs ~1000 cycles. These passes insert
/// `s_prefetch_inst` hints so fetch runs ahead of the program counter. Hints only —
/// no change to results.
///
/// Two axes name the passes:
///   - relative vs absolute: HOW a hint addresses code. "relative" uses a PC-relative
///     opcode (no extra registers); "absolute" (this family) forms an absolute address
///     once via `s_getpc` + a label into a reserved SGPR base, then prefetches
///     label+offset.
///   - static vs dynamic: WHERE hints go. "static" places them at fixed kernel-layout
///     byte offsets; "dynamic" (this pass) places them at the hot blocks the kernel
///     actually branches to at run time, so it still works when the layout is too
///     large to grid-cover.
///
/// Coverage split by kernel size (totalLayoutBytes), one pass per range:
///   CP preload : [0, 32640]      handled by hardware; no pass needed.
///   AbsStatic  : (32640, 65536]  one entry burst, N<=8 hints (abs-static pass).
///   AbsDynamic : (65536, +inf)   THIS pass: a run-time-targeted GW branch prefetch chain.
///
/// What this pass does (TensileLite-specific): it recognizes TensileLite's global-write
/// label names (label_GW_B0_MB / label_GW_B0_GSU1 / label_GW_B1_GSU1,
/// label_TailLoopBegin*, label_OptNLL_End) and prefetches the block the kernel will
/// jump to, selected at run time by the GSU and beta values — so it is not a
/// general-purpose pass. When debug output is enabled it first logs the intended
/// prefetch targets to the debug file (read-only analysis, no IR change); it emits
/// hints only when the conditions below hold. When it emits, it places a small GW branch
/// prefetch chain (getpc + label + `s_prefetch_inst`) right after `label_MultiGemmEnd`,
/// one branch per GSU/beta case, each branch prefetching its target block.
///
/// Optional near-boundary "cover" (enabled in production): an unconditional burst
/// placed just past the CP window that prefetches the once-through fast path (the code
/// between the CP boundary and the first tail/epilogue block), anchored at
/// `label_SW_PrefetchAbs_CpBoundary`. It is independent of the GSU/beta condition, so
/// it also runs on GSU0 / no-beta kernels.
///
/// Prefetch budget (at most 8 hints per launch — one I-cache window):
///   - coverN : hints spent on the near-boundary cover (0 when the fast path already
///              fits the CP window, in which case the cover emits nothing).
///   - armN   : hints spent per GW branch prefetch block (one branch per GSU/beta case).
///   - coverN + armN <= 8, and each branch keeps armN >= armFloor (=4), so coverN is
///     clamped to {0..4}.
///
/// When is nothing emitted?
///   - `totalLayoutBytes <= 32640`        : whole kernel fits the CP window.
///   - `32640 < totalLayoutBytes <= 65536`: handled by the abs-static pass instead.
///   - `baseSgpr < 0`                     : no reserved SGPR base.
///   - Stream-K kernels                   : both the branch chain and the cover bail.
/// The GSU/beta branch chain is additionally skipped for GSU0 / no-beta kernels, but the
/// near-boundary cover (when enabled) still emits for those. The read-only debug dump
/// (when enabled) runs for all post-CP kernels regardless.
///
/// Mutually exclusive with the PC-relative prefetch passes — do not run together.
/// Debug output: `sw_prefetch_abs_dynamic_pass.txt`.

/// \p baseSgpr  Low index of the reserved 64-bit SGPR pair. Pass -1 to no-op.
/// \p cpBoundaryCover  Enable the near-boundary cover (the unconditional dynamic-width
///                     burst prepended to the branch chain). Default off (staged rollout).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    int baseSgpr, const std::string& debugOutputPath = {}, bool cpBoundaryCover = false);

/// Overload that reads base SGPR and debug path from \p module options:
/// `SwInstructionPrefetchAbsBaseSgpr` and `StinkyTofuCostOutputDir`.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    StinkyAsmModule& module);

}  // namespace stinkytofu
