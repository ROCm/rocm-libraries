// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_helpers.asm.c -- C99 port of rocke.helpers.asm
 * (mfma_f8f6f4_agpr, mfma_f8f6f4_agpr_cluster).
 *
 * Byte-faithful translation: the builder-call sequence is reproduced in the
 * exact same order as the Python so the emitted IR (and SSA value numbering)
 * stays identical. The Python `inline_asm` returns the single result Value and
 * `inline_asm_multi` returns the list of result Values; the C builder calls
 * return the op, whose `results[]` array holds those Values in declaration
 * order.
 *
 * The Python `result_name_hint` ("mfma" / "mfmacl") is an emission cosmetic the
 * C `rocke_b_inline_asm` does not expose (it names asm results "asm"); this is an
 * accepted, documented divergence in the C port -- it does not affect the asm
 * template, constraints, operands, or result types, which are what the lowerers
 * consume.
 */

#include "rocke/helper_helpers.asm.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live           */
#include "rocke/strbuf.h" /* growable template assembly          */

/* The f8f6f4 MFMA srcA/srcB operand type the AMDGPU backend accepts under the
 * inline-asm `a` (AGPR) constraint: a 256-bit <8 x i32> (8 AGPRs). Python
 * module constant `_MFMA_F8F6F4_SRC_TY = VectorType(I32, 8)`. Built per-call
 * via the builder's arena (the C type constructors are builder-scoped). */
static const rocke_type_t* mfma_f8f6f4_src_ty(rocke_ir_builder_t* b)
{
    return rocke_vector_type(b, rocke_i32(), 8);
}

/* Python `_as_src_v8i32`: bitcast a 256-bit operand to <8 x i32> (no-op if
 * already that type).
 *
 *     def _as_src_v8i32(b, v):
 *         if isinstance(v.type, VectorType) and v.type == _MFMA_F8F6F4_SRC_TY:
 *             return v
 *         return b.vec_bitcast(v, _MFMA_F8F6F4_SRC_TY)
 */
static rocke_value_t* as_src_v8i32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    const rocke_type_t* src_ty = mfma_f8f6f4_src_ty(b);
    if(!src_ty)
    {
        return NULL;
    }
    if(v != NULL && v->type != NULL && v->type->kind == ROCKE_TYPE_VECTOR
       && rocke_type_eq(v->type, src_ty))
    {
        return v;
    }
    return rocke_b_vec_bitcast(b, v, src_ty);
}

rocke_value_t* rocke_mfma_f8f6f4_agpr(rocke_ir_builder_t* b,
                                      rocke_value_t* a,
                                      rocke_value_t* bb,
                                      rocke_value_t* acc,
                                      bool convergent,
                                      int hazard_nop)
{
    rocke_strbuf_t tmpl;
    rocke_value_t* operands[3];
    const rocke_type_t* result_types[1];
    rocke_inline_asm_opts_t opts;
    rocke_op_t* op;
    rocke_value_t* result = NULL;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(acc == NULL || acc->type == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "mfma_f8f6f4_agpr: acc must be a typed Value");
    }

    /* The backend accepts only <8 x i32> for the `a`-constrained MFMA sources;
     * bitcast the native fp8 fragment (<32 x fp8e4m3>) if needed. Python emits
     * the A bitcast before the B bitcast. */
    a = as_src_v8i32(b, a);
    bb = as_src_v8i32(b, bb);

    /* NOTE: the AMDGPU assembler treats `;` as a COMMENT, so additional
     * statements MUST be separated by a NEWLINE (\n), not `;`. */
    if(rocke_strbuf_init(&tmpl, 64) != 0)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "mfma_f8f6f4_agpr: OOM");
    }
    rocke_strbuf_append(&tmpl, "v_mfma_f32_16x16x128_f8f6f4 $0, $2, $3, $1");
    if(hazard_nop)
    {
        rocke_strbuf_appendf(&tmpl, "\n\ts_nop %d", (int)hazard_nop);
    }
    if(tmpl.oom)
    {
        rocke_strbuf_free(&tmpl);
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "mfma_f8f6f4_agpr: OOM");
    }

    /* Python: b.inline_asm(template, "=v,0,a,a", [acc, a, bb],
     *                       result_type=acc.type, sideeffect=True,
     *                       convergent=convergent, result_name_hint="mfma")
     * Operand list order: [acc, a, bb]. */
    operands[0] = acc;
    operands[1] = a;
    operands[2] = bb;
    result_types[0] = acc->type;

    opts.sideeffect = true;
    opts.sideeffect_set = true;
    opts.convergent = convergent;
    opts.convergent_set = true;

    op = rocke_b_inline_asm(
        b, rocke_strbuf_cstr(&tmpl), "=v,0,a,a", operands, 3, result_types, 1, &opts);
    rocke_strbuf_free(&tmpl);

    if(op != NULL && op->num_results >= 1)
    {
        result = op->results[0];
    }
    return result;
}

int rocke_mfma_f8f6f4_agpr_cluster(rocke_ir_builder_t* b,
                                   rocke_value_t* const* accs,
                                   rocke_value_t* const* srcs_a,
                                   rocke_value_t* const* srcs_b,
                                   int n,
                                   int tail_nop,
                                   int inter_nop,
                                   bool convergent,
                                   rocke_value_t** out_accs)
{
    rocke_strbuf_t tmpl;
    rocke_strbuf_t cons;
    rocke_value_t** src_a = NULL;
    rocke_value_t** src_b = NULL;
    rocke_value_t** operands = NULL;
    const rocke_type_t** result_types = NULL;
    rocke_inline_asm_opts_t opts;
    rocke_op_t* op;
    int i;
    int rc = -1;

    if(!rocke_i_live(b))
    {
        return -1;
    }
    /* Python: assert n == len(srcs) and n >= 1 */
    if(n < 1 || accs == NULL || srcs_a == NULL || srcs_b == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "mfma_f8f6f4_agpr_cluster: requires n >= 1 and non-null "
                        "accs/srcs");
        return -1;
    }
    for(i = 0; i < n; ++i)
    {
        if(accs[i] == NULL || accs[i]->type == NULL)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "mfma_f8f6f4_agpr_cluster: acc[%d] must be a typed "
                            "Value",
                            i);
            return -1;
        }
    }

    /* Scratch arrays. Use the builder arena so they live as long as needed and
     * are reclaimed with the builder (no manual free path to leak on error). */
    src_a = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(rocke_value_t*));
    src_b = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(rocke_value_t*));
    operands
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)(3 * n) * sizeof(rocke_value_t*));
    result_types = (const rocke_type_t**)rocke_arena_alloc(&b->arena,
                                                           (size_t)n * sizeof(const rocke_type_t*));
    if(!src_a || !src_b || !operands || !result_types)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "mfma_f8f6f4_agpr_cluster: OOM");
        return -1;
    }

    /* Python: bitcast all sources to <8 x i32>, in (a, bb) order per pair. */
    for(i = 0; i < n; ++i)
    {
        src_a[i] = as_src_v8i32(b, srcs_a[i]);
        src_b[i] = as_src_v8i32(b, srcs_b[i]);
    }

    /* Build the template (one v_mfma line per MFMA, optional inter/tail nops). */
    if(rocke_strbuf_init(&tmpl, 128) != 0 || rocke_strbuf_init(&cons, 64) != 0)
    {
        rocke_strbuf_free(&tmpl);
        rocke_i_set_err(b, ROCKE_ERR_OOM, "mfma_f8f6f4_agpr_cluster: OOM");
        return -1;
    }
    /*   $0..$(n-1)   : outputs
     *   $(n+i)       : tied acc i
     *   $(2n+2i)     : srcA_i ; $(2n+2i+1) : srcB_i
     * Lines joined with "\n\t" (Python "\n\t".join(lines)). */
    for(i = 0; i < n; ++i)
    {
        int vdst = i;
        int vsrc = n + i;
        int srcA = 2 * n + 2 * i;
        int srcB = 2 * n + 2 * i + 1;
        if(i > 0)
        {
            rocke_strbuf_append(&tmpl, "\n\t");
        }
        rocke_strbuf_appendf(
            &tmpl, "v_mfma_f32_16x16x128_f8f6f4 $%d, $%d, $%d, $%d", vdst, srcA, srcB, vsrc);
        if(inter_nop && (i + 1 < n))
        {
            rocke_strbuf_appendf(&tmpl, "\n\ts_nop %d", (int)inter_nop);
        }
    }
    if(tail_nop)
    {
        if(n > 0)
        {
            rocke_strbuf_append(&tmpl, "\n\t");
        }
        rocke_strbuf_appendf(&tmpl, "s_nop %d", (int)tail_nop);
    }

    /* Constraints: "=v"*n , then tied "0..n-1", then "a"*(2n). Python:
     *   out_constraints  = ",".join(["=v"] * n)
     *   tied_constraints = ",".join(str(i) for i in range(n))
     *   src_constraints  = ",".join(["a"] * (2 * n))
     *   constraints = f"{out},{tied},{src}" */
    for(i = 0; i < n; ++i)
    {
        rocke_strbuf_append(&cons, (i == 0) ? "=v" : ",=v");
    }
    for(i = 0; i < n; ++i)
    {
        rocke_strbuf_appendf(&cons, ",%d", i);
    }
    for(i = 0; i < 2 * n; ++i)
    {
        rocke_strbuf_append(&cons, ",a");
    }

    if(tmpl.oom || cons.oom)
    {
        rocke_strbuf_free(&tmpl);
        rocke_strbuf_free(&cons);
        rocke_i_set_err(b, ROCKE_ERR_OOM, "mfma_f8f6f4_agpr_cluster: OOM");
        return -1;
    }

    /* operands = list(accs) + [a0,b0,a1,b1,...]; result_types = [acc.type]*n */
    for(i = 0; i < n; ++i)
    {
        operands[i] = accs[i];
        result_types[i] = accs[i]->type;
    }
    for(i = 0; i < n; ++i)
    {
        operands[n + 2 * i] = src_a[i];
        operands[n + 2 * i + 1] = src_b[i];
    }

    opts.sideeffect = true;
    opts.sideeffect_set = true;
    opts.convergent = convergent;
    opts.convergent_set = true;

    op = rocke_b_inline_asm_multi(b,
                                  rocke_strbuf_cstr(&tmpl),
                                  rocke_strbuf_cstr(&cons),
                                  operands,
                                  3 * n,
                                  result_types,
                                  n,
                                  &opts);
    rocke_strbuf_free(&tmpl);
    rocke_strbuf_free(&cons);

    if(op != NULL && op->num_results >= n)
    {
        if(out_accs != NULL)
        {
            for(i = 0; i < n; ++i)
            {
                out_accs[i] = op->results[i];
            }
        }
        rc = 0;
    }
    return rc;
}
