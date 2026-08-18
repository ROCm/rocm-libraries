// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_fp8_moe_fused_mega_fp8_select_dispatch.c
 *
 * PER-EXPERT REBASING + MFMA/CADENCE DISPATCH chunk of the C99 port of
 * rocke/instances/common/moe_fused_mega_fp8.py.
 *
 * This translation unit implements the small shared closures/emitters that the
 * rest of the build driver invokes everywhere:
 *
 *   _emit_loop_cadence_hint  (Python lines 203-214)
 *   _emit_sgb_gateup_dtla    (Python lines 225-239)
 *   _emit_sgb_down_group     (Python lines 242-254)
 *   _emit_mfma               (Python lines 257-267)
 *   _emit_mfma_down          (Python lines 270-282)
 *   _elem_bytes_b            (Python lines 1710-1713)
 *   _b_base                  (Python lines 1724-1728)
 *   _scale_base              (Python lines 1733-1738)
 *   _select_item             (Python lines 1749-1772)
 *
 * The builder-call sequence is byte-identical to the Python body; every other
 * phase is implemented in its own peer .c file and reached through the shared
 * private header. This file edits no headers and defines no other functions.
 */

#include <string.h>

#include "rocke/helper_helpers.asm.h" /* rocke_mfma_f8f6f4_agpr */
#include "rocke/instance_moe_fused_mega_fp8_internal.h"

/* sched_group_barrier mask bits (Python module-level _SGB_*; lines 218-222). */
#define ROCKE_MOE_FP8_SGB_MFMA 0x008
#define ROCKE_MOE_FP8_SGB_VMEM_READ 0x020
#define ROCKE_MOE_FP8_SGB_VMEM_WRITE 0x040
#define ROCKE_MOE_FP8_SGB_DS_READ 0x100
#define ROCKE_MOE_FP8_SGB_DS_WRITE 0x200

/* ----------------------------------------------------------------------- *
 * effective-cadence helper.
 *
 * Python pattern: ``cadence if cadence is not None else _SCHED_CADENCE``.
 *
 * In C: the explicit per-call `cadence` (NULL => unset) overrides; otherwise
 * fall back to ctx->cadence (the spec.sched_cadence override, NULL => unset),
 * which itself falls back to ctx->levers.sched_cadence (the import-time env
 * default, always non-NULL when defaulted). This mirrors the chain
 * documented on the internal header: per-call -> spec -> levers env default.
 */
static const char* moe_fp8_effective_cadence(const rocke_moe_fp8_build_ctx_t* ctx,
                                             const char* cadence)
{
    if(cadence != NULL)
        return cadence;
    if(ctx->cadence != NULL)
        return ctx->cadence;
    return ctx->levers.sched_cadence;
}

static bool moe_fp8_cadence_is(const char* eff, const char* want)
{
    return eff != NULL && strcmp(eff, want) == 0;
}

/* ======================================================================= *
 * scheduling-cadence emitters (Python lines 203-254)
 * ======================================================================= */

/* _emit_loop_cadence_hint: emit b.iglp_opt(1) iff effective cadence == "iglp1". */
void rocke_moe_fp8_emit_loop_cadence_hint(rocke_moe_fp8_build_ctx_t* ctx, const char* cadence)
{
    const char* eff = moe_fp8_effective_cadence(ctx, cadence);
    if(moe_fp8_cadence_is(eff, "iglp1"))
        rocke_b_iglp_opt(ctx->b, 1);
}

/* _emit_sgb_gateup_dtla: compv4-style cadence for the DTLA gate/up loop body
 * (no-op unless effective cadence == "sgb"). */
void rocke_moe_fp8_emit_sgb_gateup_dtla(rocke_moe_fp8_build_ctx_t* ctx,
                                        int n_mfma,
                                        const char* cadence)
{
    const char* eff = moe_fp8_effective_cadence(ctx, cadence);
    if(!moe_fp8_cadence_is(eff, "sgb"))
        return;
    rocke_b_sched_group_barrier(ctx->b, ROCKE_MOE_FP8_SGB_VMEM_READ, 1, 0);
    rocke_b_sched_group_barrier(ctx->b, ROCKE_MOE_FP8_SGB_DS_READ, n_mfma, 0);
    rocke_b_sched_group_barrier(ctx->b, ROCKE_MOE_FP8_SGB_MFMA, n_mfma, 0);
}

/* _emit_sgb_down_group: compv4-style VMEM<->MFMA cadence for the down loop
 * per-group body (no-op unless effective cadence == "sgb"). */
void rocke_moe_fp8_emit_sgb_down_group(rocke_moe_fp8_build_ctx_t* ctx,
                                       int n_mfma,
                                       const char* cadence)
{
    const char* eff = moe_fp8_effective_cadence(ctx, cadence);
    if(!moe_fp8_cadence_is(eff, "sgb"))
        return;
    rocke_b_sched_group_barrier(ctx->b, ROCKE_MOE_FP8_SGB_VMEM_READ, 1, 0);
    rocke_b_sched_group_barrier(ctx->b, ROCKE_MOE_FP8_SGB_MFMA, n_mfma, 0);
}

/* ======================================================================= *
 * MFMA dispatchers (Python lines 257-282)
 * ======================================================================= */

/* Atom-shape predicate matching Python `atom.k == 128 and atom.dtype_in ==
 * "fp8e4m3"` (the K=128 fp8 hero atom). */
static bool moe_fp8_is_hero_fp8_atom(const rocke_mfma_atom_t* atom)
{
    return atom != NULL && atom->k == 128 && atom->dtype_in != NULL
           && strcmp(atom->dtype_in, "fp8e4m3") == 0;
}

/* _emit_mfma: one K=128 fp8 MFMA, via mfma_f8f6f4_agpr iff
 * levers.use_asm_agpr_mfma && atom.k==128 && fp8e4m3, else atom.emit. */
rocke_value_t* rocke_moe_fp8_emit_mfma(rocke_moe_fp8_build_ctx_t* ctx,
                                       rocke_value_t* a,
                                       rocke_value_t* bb,
                                       rocke_value_t* acc)
{
    const rocke_mfma_atom_t* atom = ctx->atom;
    if(ctx->levers.use_asm_agpr_mfma && moe_fp8_is_hero_fp8_atom(atom))
    {
        /* Python: mfma_f8f6f4_agpr(b, a, bb, acc, hazard_nop=_ASM_MFMA_HAZARD_NOP).
         * The helper's convergent default is True (Python keyword default). */
        return rocke_mfma_f8f6f4_agpr(ctx->b,
                                      a,
                                      bb,
                                      acc,
                                      /*convergent=*/true,
                                      ctx->levers.asm_mfma_hazard_nop);
    }
    /* atom.emit(b, a, bb, acc) -> b.mma(atom.name, a, bb, acc) (NULL extra). */
    return rocke_b_mma(ctx->b, atom->name, a, bb, acc, NULL, 0);
}

/* _emit_mfma_down: down-loop MFMA; routes through the AGPR-source helper iff
 * levers.use_asm_agpr_mfma_down (else delegates to _emit_mfma). */
rocke_value_t* rocke_moe_fp8_emit_mfma_down(rocke_moe_fp8_build_ctx_t* ctx,
                                            rocke_value_t* a,
                                            rocke_value_t* bb,
                                            rocke_value_t* acc)
{
    const rocke_mfma_atom_t* atom = ctx->atom;
    if(ctx->levers.use_asm_agpr_mfma_down && moe_fp8_is_hero_fp8_atom(atom))
    {
        return rocke_mfma_f8f6f4_agpr(ctx->b,
                                      a,
                                      bb,
                                      acc,
                                      /*convergent=*/true,
                                      ctx->levers.asm_mfma_hazard_nop);
    }
    return rocke_moe_fp8_emit_mfma(ctx, a, bb, acc);
}

/* ======================================================================= *
 * per-expert pointer rebasing closures (Python lines 1710-1772)
 * ======================================================================= */

/* Integration (link fix, no emit change): the per-expert pointer rebasing
 * closures rocke_moe_fp8_elem_bytes_b / _b_base / _scale_base / _select_item have
 * their single canonical (external) definition in the glue peer part-file
 *   instance_moe_fused_mega_fp8_moe_fused_mega_fp8_glue.c
 * and are declared in the shared instance_moe_fused_mega_fp8_internal.h (included
 * above). The emitter also placed byte-identical copies here; keeping both would
 * trigger multiple-definition link errors, so these copies are omitted and the
 * call sites in this TU bind to the glue definitions via the shared header. */
