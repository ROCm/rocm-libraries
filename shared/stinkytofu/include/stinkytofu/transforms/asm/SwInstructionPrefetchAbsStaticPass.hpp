// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// SW instruction prefetch — absolute-address, static (mid-size kernel) policy.
///
/// Background: gfx1250 has no hardware instruction prefetch; the CP preloads only the
/// first 32640 bytes, so larger kernels stall on I-cache misses. These passes insert
/// `s_prefetch_inst` hints (pure hints; no result change).
///   - relative vs absolute: relative uses a PC-relative opcode (no extra SGPRs);
///     absolute (this family) forms an absolute address once via `s_getpc` + a label
///     into a reserved SGPR base, then prefetches label+offset.
///   - static vs dynamic: static (this pass) places hints at fixed kernel-layout byte
///     offsets; dynamic places them at run-time-selected hot blocks.
///
/// Coverage split by kernel size (totalLayoutBytes), one pass per range:
///   CP preload : [0, 32640]      handled by hardware; no pass needed.
///   AbsStatic  : (32640, 65536]  THIS pass: one entry burst, N<=8 hints.
///   AbsDynamic : (65536, +inf)   run-time-targeted ladder (abs-dynamic pass).
///
/// **Regime:** `32640 < totalLayoutBytes <= 65536` (kernel fits in ~64 KiB
/// I-cache; no replacement modeling needed). No-op outside this range.
///
/// **Shape (single-label + koffset):** one unconditional burst in the entry
/// basic block, before any branch:
///   label_Do_SW_PrefetchAbs_entry
///   s_getpc_b64     s[base:base+1]
///   s_add_i32       s[base+2], label_SW_PrefetchAbs_0, 4   ; PC-rel offset (+4 getpc correction)
///   s_add_u32       s[base],   s[base],   s[base+2]
///   s_addc_u32      s[base+1], s[base+1], 0
///   s_prefetch_inst s[base:base+1], 0,    null, 0x1f       ; P(0)
///   s_prefetch_inst s[base:base+1], 4096, null, 0x1f       ; P(1)
///   ...
///
/// One target label `label_SW_PrefetchAbs_0` is inserted at the first real
/// instruction at or after `P(0) = 32640`. koffsets 0, 4096, 8192, ... cover
/// subsequent 4 KiB grid steps from the same base. This is cheaper than
/// per-k labels (one getpc+add chain) and correct when the post-CP layout is
/// contiguous (no alignment gaps that desync koffsets from physical anchors).
///
/// Address materialization uses a bare label + temp SGPR (the rocisa long-branch
/// idiom), NOT an `@pc` relocation variant (`@pc` is invalid; assembler rejects it).
/// klength uses the simm5 immediate (`0x1f`), slength = null — no length SGPR.
/// Minimum-SGPR alternative (2 SGPRs): `@rel32@lo+4` / `@rel32@hi+12` on the adds.
///
/// **Requires:** `SwInstructionPrefetchAbsBaseSgpr` >= 0 — **3** reserved SGPRs: even-aligned
/// pair `s[base:base+1]` + scratch `s[base+2]`. Reserved through the prolog and freed at
/// label_MultiGemmEnd (body reuses → net ~0 pressure). No-op when -1.
///
/// Mutually exclusive with `SwInstructionPrefetchRelStaticPass` /
/// `SwInstructionPrefetchRelDynamicPass` — do not run together.
/// Debug output: `sw_prefetch_abs_static_pass.txt`.

/// \p baseSgpr  Low index of the reserved 3-SGPR block (even-aligned pair
///              s[base:base+1] + scratch s[base+2]). Pass -1 to no-op.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsStaticPass(
    int baseSgpr, const std::string& debugOutputPath = {});

/// Overload that reads base SGPR and debug path from \p module options:
/// `SwInstructionPrefetchAbsBaseSgpr` and `StinkyTofuCostOutputDir`.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsStaticPass(
    StinkyAsmModule& module);

/// I-cache size threshold for the static policy gate (64 KiB).
inline constexpr int64_t kSwPrefetchAbsStaticIcacheSizeBytes = int64_t(65536);

/// Label names emitted by this pass.
inline constexpr std::string_view kSwPrefetchAbsSiteLabel = "label_Do_SW_PrefetchAbs_entry";
inline constexpr std::string_view kSwPrefetchAbsTargetLabelBase = "label_SW_PrefetchAbs_";

}  // namespace stinkytofu
