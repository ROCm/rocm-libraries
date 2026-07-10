/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_helpers.asm.h -- C99 port of rocke.helpers.asm.
 *
 * Typed AMDGPU inline-asm helpers (NEW, additive -- golden-safe). This module
 * wraps `rocke_b_inline_asm` / `rocke_b_inline_asm_multi` with named, typed helpers
 * for machine instructions whose operand *register classes* must be pinned
 * explicitly. The motivating case is the dense, unscaled
 * `v_mfma_f32_16x16x128_f8f6f4` with AGPR srcA/srcB + VGPR accumulator (the
 * aiter staging layout): the only LLVM intrinsic for K=128 fp8 is the *scaled*
 * one (`llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`), and even pinning its
 * scales to 0 leaves srcA/srcB placement to the register allocator. The `a`
 * inline-asm constraint forces them into the AGPR file.
 *
 * Scope of THIS file: only `mfma_f8f6f4_agpr` and `mfma_f8f6f4_agpr_cluster`
 * are ported (the trivial `s_nop` helper in helpers/asm.py is out of scope for
 * this phase).
 *
 * AMDGPU inline-asm constraint cheatsheet:
 *   `v` = VGPR input, `a` = AGPR input, `s` = SGPR input;
 *   `=v` / `=a` / `=s` = outputs; a digit (`0`) ties an input to that output
 *   (read+write in place, required for the MFMA accumulator).
 *
 * Nothing here changes any existing op/atom: it adds new helper functions that
 * only the mega-kernel selects, so the golden shared-kernel digest is
 * unaffected.
 */
#ifndef ROCKE_HELPER_HELPERS_ASM_H
#define ROCKE_HELPER_HELPERS_ASM_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* gfx950 16x16x128 f8f6f4 MFMA result/source-hazard wait-states (Python
 * `_MFMA_F8F6F4_HAZARD_NOP`). Baked-in `s_nop 8` is the proven minimum for
 * chained AGPR-source correctness on an inline-asm MFMA (the GCNHazardRecognizer
 * is opaque to asm blobs, so it inserts no waits of its own). */
#define ROCKE_MFMA_F8F6F4_HAZARD_NOP 8

/* Dense **unscaled** `v_mfma_f32_16x16x128_f8f6f4` with AGPR srcA/srcB.
 *
 * Emits (per ASM_HELPERS_PLAN section 2, plus the hazard `s_nop` proven
 * required for inline-asm MFMAs):
 *
 *     %acc = call <4 x float> asm sideeffect
 *            "v_mfma_f32_16x16x128_f8f6f4 $0, $2, $3, $1\n\ts_nop 8",
 *            "=v,0,a,a"(<4 x float> %c, <8 x i32> %a8, <8 x i32> %b8)
 *
 * Operand / `$`-mapping ($0 = output, then inputs in listed order):
 *   - $0 `=v` : output accumulator, VGPR
 *   - $1 `0`  : tied input accumulator (ties to $0 -- read+write in place)
 *   - $2 `a`  : srcA, AGPR, <8 x i32>
 *   - $3 `a`  : srcB, AGPR, <8 x i32>
 *
 * Args:
 *   a, bb: the A and B operand tiles. Bitcast to <8 x i32> internally if not
 *     already that type (the asm constraint type must match exactly).
 *   acc: the incoming <4 x float> accumulator (tied; returned updated).
 *   convergent: mark the asm convergent (Python default True).
 *   hazard_nop: number for the trailing `s_nop` (Python default 8; pass
 *     ROCKE_MFMA_F8F6F4_HAZARD_NOP for the default, or 0 to suppress it).
 *
 * Returns the updated <4 x float> accumulator Value, or NULL on a dead/failed
 * builder (sticky-error model). */
rocke_value_t* rocke_mfma_f8f6f4_agpr(rocke_ir_builder_t* b,
                                      rocke_value_t* a,
                                      rocke_value_t* bb,
                                      rocke_value_t* acc,
                                      bool convergent,
                                      int hazard_nop);

/* Cluster of N back-to-back **unscaled** `v_mfma_f32_16x16x128_f8f6f4` MFMAs
 * (AGPR srcA/srcB + VGPR acc) emitted as ONE inline-asm block.
 *
 * Instead of one `sideeffect` asm per MFMA (which fragments the schedule), the
 * whole MFMA burst of one K-group is a SINGLE asm block whose internal
 * instruction stream is the hand-schedule. The N MFMAs write N DISTINCT
 * accumulators and read N DISTINCT source pairs, so a single trailing `s_nop`
 * (`tail_nop`) after the last MFMA covers the result-latency hazard for the
 * whole cluster.
 *
 * Operand order (after the N outputs): N tied accs, then 2N sources.
 *   $0..$(n-1)      : outputs        (=v)
 *   $n..$(2n-1)     : tied accs      (0,1,..,n-1)  -> $(n+i) is acc i
 *   $2n..$(4n-1)    : sources (a)    -> srcA_i = $(2n+2i), srcB_i = $(2n+2i+1)
 *
 * Args:
 *   accs: array of N incoming <4 x float> accumulators (tied in/out).
 *   srcs_a, srcs_b: parallel arrays of N A/B operand tiles; each is bitcast to
 *     <8 x i32> (the AGPR-constrained MFMA source type) as needed. (The Python
 *     takes a list of (a, bb) pairs; here it is split into two parallel arrays
 *     for a plain C ABI.)
 *   n: cluster size (>= 1).
 *   tail_nop: `s_nop` count after the last MFMA (Python default 8).
 *   inter_nop: `s_nop` between consecutive MFMAs (Python default 0).
 *   convergent: mark the asm convergent (Python default True).
 *   out_accs: caller-provided array of length >= n; on success receives the N
 *     updated <4 x float> accumulators (same order). May be NULL to discard.
 *
 * Returns 0 on success, -1 on a dead/failed builder or invalid arguments (the
 * builder's sticky error is set on the error path). */
int rocke_mfma_f8f6f4_agpr_cluster(rocke_ir_builder_t* b,
                                   rocke_value_t* const* accs,
                                   rocke_value_t* const* srcs_a,
                                   rocke_value_t* const* srcs_b,
                                   int n,
                                   int tail_nop,
                                   int inter_nop,
                                   bool convergent,
                                   rocke_value_t** out_accs);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_HELPERS_ASM_H */
