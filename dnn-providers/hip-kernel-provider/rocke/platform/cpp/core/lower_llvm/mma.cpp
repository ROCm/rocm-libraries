// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_lower_llvm_mma.c -- BUCKET 4 of the C99 port of
 * rocke.core.lower_llvm.
 *
 * Owns the tile.mma routing and every MFMA atom handler:
 *   - f16 / bf16 16x16x* and 32x32x*
 *   - fp8 / bf8 via the shared rocke_ll_lower_mfma_fp8_bf8 body (defined here,
 *     called via the internal header by other buckets too)
 *   - the K=128 f8f6f4 ops: MX-scaled, the unscaled fp8 hero atom, and unscaled all-format
 *   - register_p_from_qk_c register-fragment reshape
 *   - WMMA routing: real RDNA3/RDNA4 emission (Gfx11/Gfx12RdnaBackend.emit_wmma)
 *     plus faithful CDNA rejection (CDNA targets reject WMMA, mirroring the
 *     Python ISABackend.emit_wmma NotImplementedError)
 *
 * The ISA-named MFMA handlers are NOT distinct opcodes in the frozen ir.h
 * (only ROCKE_OP_TILE_MMA and ROCKE_OP_TILE_REGISTER_P_FROM_QK_C exist). They are
 * reached from _op_tile_mma's op_id routing, which mirrors the Python CDNA
 * ISABackend.emit_mma rebuilding ``tile.<op_id>`` and re-dispatching.
 */
#include "rocke/mfma_scale_internal.h"
#include "rocke/tf32_internal.h"
#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/lower_llvm.h"
#include "rocke/lower_llvm_internal.h"
#include "rocke/mma_operand_internal.h"
#include "rocke/wmma_scale_internal.h"

namespace ckc
{

/* ------------------------------------------------------------ forward decls */

/* per-op handlers (file-static; reached via the op_id router, not the table) */
static void _op_tile_wmma_f32_16x16x16_f16(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_wmma_f32_16x16x16_bf16(rocke_lower_t* L, const rocke_op_t* op);
static void _emit_wmma(rocke_lower_t* L, const rocke_op_t* op, const char* op_id);
static void _emit_wmma_scale(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mma(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x32_fp8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x32_bf8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_32x32x16_fp8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_32x32x16_bf8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_scale_f32_16x16x128_f8f6f4(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x128_fp8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x128_fp4fp6fp8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f64_4x4x4_f64(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_register_p_from_qk_c(rocke_lower_t* L, const rocke_op_t* op);

/* Table-driven dense MFMA dispatch (definition below the spec table). Resolves
 * the plain / scalar / bf16-bitcast atoms; returns true if op_id matched. */
static bool _try_emit_mfma_table(rocke_lower_t* L, const rocke_op_t* op, const char* op_id);

/* ------------------------------------------------------------ small helpers */

/* The single-result name (Python op.result.name). The MMA ops always have
 * exactly one result; guard defensively to keep the lowerer sticky-safe. */
static const char* mma_result_name(rocke_lower_t* L, const rocke_op_t* op)
{
    if(op->num_results != 1)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "%s: expected exactly one result, got %d",
                      op->name ? op->name : "tile.mma",
                      op->num_results);
    }
    return op->results[0]->name;
}

/* The register type `vec<elem x count>`, built in the lowerer's arena. */
static const rocke_type_t*
    mma_register_vector(rocke_lower_t* L, const rocke_type_t* elem, int count)
{
    const rocke_type_t* t = rocke_i_vector_type_in_arena(&L->arena, elem, count);
    if(!t)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "MMA register type: OOM");
    return t;
}

/* MMA operand `role` (A or B) of `op` as the instruction's register type
 * (Python _Lowerer._mma_operand): unchanged, or reinterpreted with one bitcast,
 * as rocke_mma_operand_needs_reinterpret decides; that rule also throws for an
 * operand it refuses. */
static const char* mma_operand(rocke_lower_t* L,
                               const rocke_op_t* op,
                               const char* op_id,
                               rocke_mma_role_t role,
                               const rocke_type_t* register_type)
{
    const rocke_value_t* value = rocke_mma_operand_value(op, role);
    if(rocke_mma_operand_needs_reinterpret(
           op_id, role, value->type, register_type, L->backend->gfx))
        return rocke_ll_reinterpret(L, value, register_type);
    return rocke_ll_operand(L, value);
}

/* ====================================================================== */
/* Emulated atoms: BF16x3 expansion (Python _expand_emulated_mma)          */
/* ====================================================================== */

/* The catalog atom for op_id on the arch being lowered, when it needs
 * expansion; NULL when it is natively encoded (every atom predating XF32) or
 * absent. Per-arch, not a bare-op_id table: the same logical atom may be native
 * on one target and emulated on another, so the arch being lowered is the one
 * asked. Mirrors Python _emulated_mma_atom. */
static const rocke_mma_op_t* _emulated_mma_atom(rocke_lower_t* L, const char* op_id)
{
    const rocke_arch_target_t* arch;
    const rocke_mma_op_t* atom;
    if(!L->backend || !L->backend->gfx)
    {
        return NULL;
    }
    arch = rocke_arch_target_from_gfx(L->backend->gfx);
    if(!arch)
    {
        return NULL;
    }
    atom = rocke_mma_catalog_by_op_id(&arch->mma, op_id);
    if(!atom || rocke_arch_mma_op_is_native(atom))
    {
        return NULL;
    }
    return atom;
}

/* The fp32 view of a TF32 fragment: its I32 carrier bits, reinterpreted (Python
 * _tf32_payload_as_f32). A bitcast and nothing else -- never a rounding -- so the
 * split sees every payload bit the caller put there. That is the difference from
 * a native XF32 instruction, which ignores the low 13 bits of each operand. */
static const char*
    _tf32_payload_as_f32(rocke_lower_t* L, const rocke_value_t* v, const char* hint, int n)
{
    char name_hint[32];
    char i32v[24];
    char f32v[24];
    const char* cast;

    snprintf(name_hint, sizeof(name_hint), "%s_f32", hint);
    cast = rocke_ll_fresh(L, name_hint);
    snprintf(i32v, sizeof(i32v), "<%d x i32>", n);
    snprintf(f32v, sizeof(f32v), "<%d x float>", n);
    rocke_ll_emitf(L, "  %s = bitcast %s %s to %s", cast, i32v, rocke_ll_operand(L, v), f32v);
    return cast;
}

/* bf16 as the i16 lanes the TF32 split saturates on (Python _BF16_MAX_FINITE and
 * _BF16_SIGN_BIT): the largest finite bf16 has magnitude pattern 0x7f7f, and the
 * sign bit is LLVM's signed i16 literal for 0x8000. */
static const int kBf16MaxFinite = 0x7F7F;
static const int kBf16SignBit = -0x8000;

/* `<n x i16>` constant with every lane `value` (the `lanes` helper in Python
 * _split_f32_to_bf16). An explicit list rather than `splat (...)`, which older
 * LLVM flavors do not parse. Arena-owned, sized to the lane count. */
static const char* _i16_lanes(rocke_lower_t* L, int value, int n)
{
    char lane[16];
    const int lane_len = snprintf(lane, sizeof(lane), "i16 %d", value);
    const size_t cap = (size_t)n * (size_t)(lane_len + 2) + 3;
    char* out = static_cast<char*>(rocke_arena_alloc(&L->arena, cap));
    char* p = out;
    int i;

    *p++ = '<';
    for(i = 0; i < n; ++i)
    {
        if(i)
        {
            *p++ = ',';
            *p++ = ' ';
        }
        memcpy(p, lane, (size_t)lane_len);
        p += lane_len;
    }
    *p++ = '>';
    *p = '\0';
    return out;
}

/* Dekker-style split of an fp32 fragment into two bf16 fragments (Python
 * _split_f32_to_bf16). ``src`` names the fp32 vector of n elements. ``big`` is it
 * rounded to bf16 (and saturated, below); ``small`` is what that rounding threw
 * away, itself rounded to bf16.
 *
 * The narrowings are `fptrunc ... to bfloat`, which is round-to-nearest-even --
 * LLVM has no other fp32->bf16 rounding -- and the AMDGPU backend selects the
 * packed v_cvt_pk_bf16_f32 for the vector form on gfx950. That is CK's shipping
 * packed path; CK's *scalar* fallback truncates instead and is deliberately not
 * the parity target.
 *
 * SATURATION. Rounding to bf16 overflows to infinity for every payload of
 * magnitude 0x7f7f8000 or more: the top 2^-9 of the finite fp32 range, which
 * holds TF32's largest finite values. Left alone, ``big`` becomes +-inf and
 * ``small`` the opposite infinity, and the `inf * 0` that results in the cross
 * terms turns a finite product into NaN. ``big`` is therefore capped at the
 * largest finite bf16, 0x7f7f, keeping the sign. ``small`` is still taken from
 * the original payload, so it holds what the cap threw away -- finite, and within
 * bf16's own precision of it -- and the three terms sum to the same product as
 * anywhere else. Payloads below the threshold are untouched, bit for bit.
 *
 * A non-finite payload goes through the same cap, which makes ``big`` finite for
 * it too. ``small`` is computed from the payload itself, so it stays +-inf (or
 * NaN), and small*big -- the first term issued -- is then non-finite whatever the
 * other operand holds: an infinity or NaN in an operand makes every output
 * element it feeds non-finite. `Inf * 0` is NaN as IEEE says, and Inf against an
 * ordinary finite partner stays Inf; a partner whose own residual is large enough
 * to overflow against the cap can turn it into NaN. */
static void _split_f32_to_bf16(rocke_lower_t* L,
                               const char* src,
                               const char* hint,
                               int n,
                               const rocke_type_t* bf16_vec,
                               rocke_value_t* out_big,
                               rocke_value_t* out_small)
{
    char f32v[24];
    char bf16v[24];
    char i16v[24];
    char name_hint[32];
    const char* big_rne;
    const char* bits;
    const char* sign;
    const char* cap;
    const char* over;
    const char* sat;
    const char* big;
    const char* big_f32;
    const char* diff;
    const char* small;
    const char* sign_lanes;
    const char* max_lanes;

    snprintf(f32v, sizeof(f32v), "<%d x float>", n);
    snprintf(bf16v, sizeof(bf16v), "<%d x bfloat>", n);
    snprintf(i16v, sizeof(i16v), "<%d x i16>", n);

    /* Fresh names are allocated in emission order, so the temp counter matches
     * the Python lowerer's and the two engines agree byte for byte. */
    snprintf(name_hint, sizeof(name_hint), "%s_big_rne", hint);
    big_rne = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_big_bits", hint);
    bits = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_big_sign", hint);
    sign = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_big_cap", hint);
    cap = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_big_over", hint);
    over = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_big_sat", hint);
    sat = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_big", hint);
    big = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_big_f32", hint);
    big_f32 = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_diff", hint);
    diff = rocke_ll_fresh(L, name_hint);
    snprintf(name_hint, sizeof(name_hint), "%s_small", hint);
    small = rocke_ll_fresh(L, name_hint);

    sign_lanes = _i16_lanes(L, kBf16SignBit, n);
    max_lanes = _i16_lanes(L, kBf16MaxFinite, n);

    rocke_ll_emitf(L, "  %s = fptrunc %s %s to %s", big_rne, f32v, src, bf16v);
    /* The bf16 bit patterns are ordered like their magnitudes, so "past the
     * largest finite value" is an unsigned compare against that pattern with the
     * value's own sign bit attached: it catches +-inf and every NaN (all 0x7f80
     * and up in magnitude) and nothing finite. */
    rocke_ll_emitf(L, "  %s = bitcast %s %s to %s", bits, bf16v, big_rne, i16v);
    rocke_ll_emitf(L, "  %s = and %s %s, %s", sign, i16v, bits, sign_lanes);
    rocke_ll_emitf(L, "  %s = or %s %s, %s", cap, i16v, sign, max_lanes);
    rocke_ll_emitf(L, "  %s = icmp ugt %s %s, %s", over, i16v, bits, cap);
    rocke_ll_emitf(
        L, "  %s = select <%d x i1> %s, %s %s, %s %s", sat, n, over, i16v, cap, i16v, bits);
    rocke_ll_emitf(L, "  %s = bitcast %s %s to %s", big, i16v, sat, bf16v);
    rocke_ll_emitf(L, "  %s = fpext %s %s to %s", big_f32, bf16v, big, f32v);
    rocke_ll_emitf(L, "  %s = fsub %s %s, %s", diff, f32v, src, big_f32);
    rocke_ll_emitf(L, "  %s = fptrunc %s %s to %s", small, f32v, diff, bf16v);

    out_big->name = big;
    out_big->type = bf16_vec;
    out_big->op = NULL;
    out_small->name = small;
    out_small->type = bf16_vec;
    out_small->op = NULL;
}

/* Expand an emulated tile.mma in place; return whether it was one. Mirrors
 * Python _expand_emulated_mma.
 *
 * BF16x3: reinterpret both TF32 operands as fp32, split them, then issue three
 * BF16 MMAs threading a single fp32 accumulator. The small*small term is dropped:
 * it is below 2^-16 of the product, the same order as the split's own
 * representation error, so a fourth MMA would buy back only part of the error.
 *
 * The term ORDER is load-bearing. fp32 accumulation is not associative, so
 * issuing these three in any other order produces different result bits and
 * breaks parity with CK. */
static bool _expand_emulated_mma(rocke_lower_t* L, const rocke_op_t* op, const char* op_id)
{
    const rocke_mma_op_t* atom = _emulated_mma_atom(L, op_id);
    const rocke_value_t *a, *b, *c;
    rocke_value_t a_big, a_small, b_big, b_small;
    rocke_value_t acc_tmp[2];
    rocke_type_t bf16_vec;
    char bf16_vec_name[24];
    const rocke_value_t* acc;
    const char* physical;
    const char* a_f32;
    const char* b_f32;
    int n;
    int i;

    if(atom == NULL)
    {
        return false;
    }
    /* rocke_ll_fail is [[noreturn]]; nothing after a fail call runs. */
    if(atom->emulation_recipe != ROCKE_MMA_EMULATION_BF16X3)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "tile.mma %s: no lowering for emulation recipe %d",
                      atom->op_id,
                      (int)atom->emulation_recipe);
    }
    if(op->num_operands != 3 || op->num_results != 1)
    {
        rocke_ll_fail(
            L, ROCKE_ERR_VALUE, "tile.mma %s expects 3 operands and 1 result", atom->op_id);
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    for(i = 0; i < 2; ++i)
    {
        const rocke_value_t* v = (i == 0) ? a : b;
        const char* role = (i == 0) ? "a" : "b";
        const int frag_len = (i == 0) ? atom->a_frag_len : atom->b_frag_len;
        if(!v->type || v->type->kind != ROCKE_TYPE_VECTOR || !v->type->elem
           || v->type->elem->scalar != ROCKE_SCALAR_TF32)
        {
            rocke_ll_fail(L,
                          ROCKE_ERR_VALUE,
                          "tile.mma %s: BF16x3 emulation needs a TF32 vector for "
                          "operand %s, got %s; wrap fp32 payload bits with "
                          "bitcast(v, TF32), or round them first with "
                          "cvt_f32_to_tf32(v)",
                          atom->op_id,
                          role,
                          (v->type && v->type->name) ? v->type->name : "(none)");
        }
        if(v->type->count != frag_len)
        {
            rocke_ll_fail(L,
                          ROCKE_ERR_VALUE,
                          "tile.mma %s: operand %s must be a vec<tf32x%d>, got %s",
                          atom->op_id,
                          role,
                          frag_len,
                          v->type->name);
        }
    }
    n = a->type->count;

    snprintf(bf16_vec_name, sizeof(bf16_vec_name), "vec<bf16x%d>", n);
    memset(&bf16_vec, 0, sizeof(bf16_vec));
    bf16_vec.kind = ROCKE_TYPE_VECTOR;
    bf16_vec.name = bf16_vec_name;
    bf16_vec.elem = rocke_scalar_by_name("bf16");
    bf16_vec.count = n;

    a_f32 = _tf32_payload_as_f32(L, a, "xf32_a", n);
    _split_f32_to_bf16(L, a_f32, "xf32_a", n, &bf16_vec, &a_big, &a_small);
    b_f32 = _tf32_payload_as_f32(L, b, "xf32_b", n);
    _split_f32_to_bf16(L, b_f32, "xf32_b", n, &bf16_vec, &b_big, &b_small);
    if(!rocke_ll_live(L))
    {
        return true;
    }

    physical = rocke_arch_mma_op_physical_op_id(atom);
    {
        const rocke_value_t* terms[3][2]
            = {{&a_small, &b_big}, {&a_big, &b_small}, {&a_big, &b_big}};
        acc = c;
        for(i = 0; i < 3; ++i)
        {
            /* The last term must land in the caller's SSA value; the earlier two
             * thread through temporaries. */
            const bool last = (i == 2);
            rocke_value_t* out = last ? NULL : &acc_tmp[i];
            rocke_value_t* result;
            rocke_op_t sub;
            rocke_value_t* operands[3];
            rocke_value_t* results[1];

            if(!last)
            {
                out->name = rocke_ll_fresh(L, "xf32_acc");
                out->type = c->type;
                out->op = NULL;
                result = out;
            }
            else
            {
                result = op->results[0];
            }

            operands[0] = const_cast<rocke_value_t*>(terms[i][0]);
            operands[1] = const_cast<rocke_value_t*>(terms[i][1]);
            operands[2] = const_cast<rocke_value_t*>(acc);
            results[0] = result;

            memset(&sub, 0, sizeof(sub));
            sub.opcode = op->opcode;
            sub.name = op->name;
            sub.operands = operands;
            sub.num_operands = 3;
            sub.results = results;
            sub.num_results = 1;
            sub.loc = op->loc;

            /* The physical atom is native by construction -- the catalog
             * refuses an expansion chain -- so the dense table resolves it and
             * the text is identical to a hand-written physical tile.mma.
             *
             * Python reaches the same emission through backend.emit_mma, which
             * also covers WMMA. Every emulated row shipped today is MFMA, so
             * the two engines cannot disagree on bytes; a future WMMA-arch row
             * would land in the fail below rather than emit something wrong. */
            if(!_try_emit_mfma_table(L, &sub, physical))
            {
                rocke_ll_fail(L,
                              ROCKE_ERR_NOTIMPL,
                              "tile.mma %s: physical atom '%s' has no dense MFMA "
                              "emission",
                              atom->op_id,
                              physical);
            }
            if(!rocke_ll_live(L))
            {
                return true;
            }
            acc = result;
        }
    }
    return true;
}

/* ====================================================================== */
/* tile.mma routing (Python _op_tile_mma -> ISABackend.emit_mma)          */
/* ====================================================================== */

/* op_id -> handler. Mirrors the Python CDNA emit_mma which rebuilds the legacy
 * ``tile.<op_id>`` op and re-dispatches it to the matching _op_tile_<op_id>
 * method. We dispatch directly to the file-static handler to keep the emitted
 * text byte-identical. */
static void _op_tile_mma(rocke_lower_t* L, const rocke_op_t* op)
{
    const char* op_id;
    if(!rocke_ll_live(L))
    {
        return;
    }
    op_id = rocke_attr_get_str(&op->attrs, "op_id");
    if(!op_id)
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.mma: missing op_id attribute");
    }

    if(const auto* atom = rocke_gfx950_scaled_mfma(op_id))
    {
        if(!L->backend || strcmp(L->backend->gfx, "gfx950") != 0)
            rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "scaled MFMA requires gfx950");
        if(const char* error = rocke_scaled_mfma_error(op_id, op->operands, op->num_operands))
            rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s", error);
        if(op->num_results != 1 || strcmp(op->results[0]->type->name, "vec<f32x16>") != 0)
            rocke_ll_fail(L, ROCKE_ERR_VALUE, "scaled MFMA result must be vec<f32x16>");
        rocke_ll_need(L, "mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32");
        rocke_ll_emitf(
            L,
            "  %s = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32("
            "<8 x i32> %s, <8 x i32> %s, <16 x float> %s, i32 %d, i32 %d, i32 0, i32 %s, i32 0, "
            "i32 %s)",
            mma_result_name(L, op),
            rocke_ll_operand(L, op->operands[0]),
            rocke_ll_operand(L, op->operands[1]),
            rocke_ll_operand(L, op->operands[2]),
            rocke_mfma_matrix_format(atom->a_dtype),
            rocke_mfma_matrix_format(atom->b_dtype),
            rocke_ll_operand(L, op->operands[3]),
            rocke_ll_operand(L, op->operands[4]));
        return;
    }
    char tf32_error[ROCKE_ERR_MSG_CAP];
    if(rocke_tf32_mma_target_error(op_id, L->backend->gfx, tf32_error, sizeof tf32_error))
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s", tf32_error);

    /* An emulated atom is expanded here, before ISA dispatch, so the expansion
     * is written once rather than per ISA family. Native atoms -- every atom
     * that predates XF32 -- fall straight through, so their emission is
     * unchanged. */
    if(_expand_emulated_mma(L, op, op_id))
    {
        return;
    }

    /* f16 / bf16 / f32 / f64 dense atoms resolve from the table; the K=128 f8f6f4
     * ops and the fp8-bf8 atoms keep their dedicated bodies below. */
    if(_try_emit_mfma_table(L, op, op_id))
    {
        return;
    }

    /* f64 4x4x4: scalar intrinsic around the <1 x double> accumulator */
    if(strcmp(op_id, "mfma_f64_4x4x4_f64") == 0)
    {
        _op_tile_mfma_f64_4x4x4_f64(L, op);
    }
    /* fp8 / bf8 */
    else if(strcmp(op_id, "mfma_f32_16x16x32_fp8") == 0)
    {
        _op_tile_mfma_f32_16x16x32_fp8(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_16x16x32_bf8") == 0)
    {
        _op_tile_mfma_f32_16x16x32_bf8(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_32x32x16_fp8") == 0)
    {
        _op_tile_mfma_f32_32x32x16_fp8(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_32x32x16_bf8") == 0)
    {
        _op_tile_mfma_f32_32x32x16_bf8(L, op);
        /* K=128 f8f6f4: MX-scaled, unscaled hero, unscaled all-format */
    }
    else if(strcmp(op_id, "mfma_scale_f32_16x16x128_f8f6f4") == 0)
    {
        _op_tile_mfma_scale_f32_16x16x128_f8f6f4(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_16x16x128_fp4fp6fp8") == 0)
    {
        _op_tile_mfma_f32_16x16x128_fp4fp6fp8(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_16x16x128_fp8") == 0)
    {
        _op_tile_mfma_f32_16x16x128_fp8(L, op);
        /* WMMA op_ids. On an RDNA backend (gfx11/gfx12) these emit a real WMMA
         * call (Python Gfx11/Gfx12RdnaBackend.emit_wmma); on a CDNA backend they
         * reject (Python ISABackend.emit_wmma raises NotImplementedError). The
         * gfx12-specific op_ids ("wmma_gfx12_*") can only occur on RDNA4. */
    }
    else if(rocke_gfx1250_scaled_wmma(op_id))
    {
        _emit_wmma_scale(L, op);
    }
    else if(strncmp(op_id, "wmma_", 5) == 0)
    {
        if(L->backend && L->backend->kind == ROCKE_LL_ISA_RDNA)
        {
            _emit_wmma(L, op, op_id);
        }
        else if(strcmp(op_id, "wmma_f32_16x16x16_f16") == 0)
        {
            _op_tile_wmma_f32_16x16x16_f16(L, op);
        }
        else if(strcmp(op_id, "wmma_f32_16x16x16_bf16") == 0)
        {
            _op_tile_wmma_f32_16x16x16_bf16(L, op);
        }
        else
        {
            rocke_ll_fail(L,
                          ROCKE_ERR_NOTIMPL,
                          "WMMA op 'tile.%s' not available on %s "
                          "(WMMA is an RDNA/gfx11 instruction; this is a "
                          "CDNA/MFMA target)",
                          op_id,
                          L->backend ? L->backend->gfx : "(cdna)");
        }
    }
    else
    {
        rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "tile.mma: unsupported op_id '%s'", op_id);
    }
}

/* Emit the scaled-WMMA intrinsic using the resolved operand contract.
 * The contract supplies packed carrier types, matrix and scale format selectors,
 * and the declaration identity. This instruction family requires LLVM 23. */
static void _emit_wmma_scale(rocke_lower_t* L, const rocke_op_t* op)
{
    const char* intrinsic;
    const char* decl_key;
    const char* scale_ty;
    const char* op_name;

    if(!rocke_ll_live(L))
    {
        return;
    }
    const rocke_mma_op_t* atom = rocke_gfx1250_scaled_wmma_from_op(op);
    if(!atom)
    {
        rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "unsupported scaled WMMA op '%s'", op->name);
    }
    const rocke_scaled_wmma_op_t contract = rocke_scaled_wmma_contract(atom);
    const rocke_scaled_wmma_op_t* spec = &contract;
    char concrete_name[160];
    snprintf(concrete_name, sizeof(concrete_name), "tile.%s", spec->op_id);
    op_name = concrete_name;
    /* Admitted on any target whose MMA catalog lists the atom (Python
     * ISABackend.emit_scaled_wmma), never by the target's name. */
    const char* gfx = (L->backend && L->backend->gfx) ? L->backend->gfx : NULL;
    const rocke_arch_target_t* target = gfx ? rocke_arch_target_from_gfx(gfx) : NULL;
    if(!target || !rocke_mma_catalog_by_op_id(&target->mma, spec->op_id))
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "%s is not available on %s: its MMA catalog does not list it",
                      op_name,
                      gfx ? gfx : "(unknown)");
    }
    if(L->flavor != ROCKE_LLVM_FLAVOR_LLVM23)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "%s requires llvm23 (ROCm 7.13+), got %s",
                      op_name,
                      rocke_llvm_flavor_name(L->flavor));
    }
    if(op->num_operands != 5)
    {
        rocke_ll_fail(
            L, ROCKE_ERR_VALUE, "%s expects 5 operands, got %d", op_name, op->num_operands);
    }

    char packed_type[8];
    snprintf(packed_type, sizeof(packed_type), "i%d", rocke_scale_word_bits(&spec->scales));
    scale_ty = packed_type;
    if(strcmp(op->operands[3]->type->name, scale_ty) != 0
       || strcmp(op->operands[4]->type->name, scale_ty) != 0)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "%s expects %s scale operands, got %s/%s",
                      op_name,
                      scale_ty,
                      op->operands[3]->type->name,
                      op->operands[4]->type->name);
    }

    // Physical declaration identity comes from the resolved backend contract.
    decl_key = spec->declaration_key;
    intrinsic = spec->intrinsic;
    rocke_ll_need(L, decl_key);
    rocke_ll_emitf(L,
                   "  %s = call <8 x float> @%s("
                   "i32 %d, <%d x i32> %s, i32 %d, <%d x i32> %s, "
                   "i16 0, <8 x float> %s, i32 0, i32 %d, %s %s, "
                   "i32 0, i32 %d, %s %s, i1 false, i1 false)",
                   mma_result_name(L, op),
                   intrinsic,
                   spec->matrix_formats[0],
                   spec->matrix_words[0],
                   rocke_ll_operand(L, op->operands[0]),
                   spec->matrix_formats[1],
                   spec->matrix_words[1],
                   rocke_ll_operand(L, op->operands[1]),
                   rocke_ll_operand(L, op->operands[2]),
                   spec->scale_formats[0],
                   scale_ty,
                   rocke_ll_operand(L, op->operands[3]),
                   spec->scale_formats[1],
                   scale_ty,
                   rocke_ll_operand(L, op->operands[4]));
}

/* ====================================================================== */
/* WMMA on CDNA: faithful rejection (Python ISABackend.emit_wmma raises)   */
/* ====================================================================== */

static void _op_tile_wmma_f32_16x16x16_f16(rocke_lower_t* L, const rocke_op_t* op)
{
    (void)op;
    /* CDNA/MFMA targets reject WMMA (an RDNA/gfx11 instruction). The FROZEN
     * ir.h exposes no WMMA opcodes and the internal header notes RDNA WMMA
     * emission is out of scope, so this is a faithful NotImplementedError. */
    rocke_ll_fail(L,
                  ROCKE_ERR_NOTIMPL,
                  "WMMA op 'tile.wmma_f32_16x16x16_f16' not available on %s "
                  "(WMMA is an RDNA/gfx11 instruction; this is a CDNA/MFMA target)",
                  L->backend ? L->backend->gfx : "(cdna)");
}

static void _op_tile_wmma_f32_16x16x16_bf16(rocke_lower_t* L, const rocke_op_t* op)
{
    (void)op;
    rocke_ll_fail(L,
                  ROCKE_ERR_NOTIMPL,
                  "WMMA op 'tile.wmma_f32_16x16x16_bf16' not available on %s "
                  "(WMMA is an RDNA/gfx11 instruction; this is a CDNA/MFMA target)",
                  L->backend ? L->backend->gfx : "(cdna)");
}

/* ====================================================================== */
/* RDNA WMMA emission (Python Gfx11RdnaBackend / Gfx12RdnaBackend          */
/* .emit_wmma). The legacy op name is "tile.<op_id>"; the gfx12 op_ids     */
/* ("wmma_gfx12_*") resolve against the RDNA4 table (8-wide fragments),    */
/* the rest against the RDNA3/3.5 table (16-wide). bf16 operands are       */
/* bitcast to <W x i16> before the call (call_elt != ssa_elt).            */
/* ====================================================================== */

/* Local float-WMMA spec table, a faithful copy of the Python backend tables
 * (_RDNA_WMMA for RDNA3/3.5 + _RDNA_GFX12_WMMA for RDNA4). Held here rather than
 * via rocke/isa_backend.h to avoid the header's own `rocke_isa_backend` struct
 * colliding with the lowerer's same-named backend struct. The op_ids are
 * disjoint between the two families, so one flat table resolves both: gfx12
 * op_ids carry the "wmma_gfx12_" prefix (8-wide fragments), the rest are
 * RDNA3/3.5 (16-wide). */
/* The call shape a float-WMMA intrinsic takes. RDNA3/3.5 and RDNA4 share the
 * plain 3-operand (A, B, C) form; gfx1250 does not, so the shape has to travel
 * with the spec rather than be inferred from the fragment width. */
typedef enum _wmma_shape
{
    /* (<W x elt> A, <W x elt> B, <8 x float> C) -- gfx11 / gfx12 */
    WMMA_SHAPE_AB_C = 0,
    /* (i1 negA, <16 x elt> A, i1 negB, <16 x elt> B, i16 fmt, <8 x float> C,
     * i1 reuseA, i1 reuseB) -- gfx1250 K=32 f16/bf16 */
    WMMA_SHAPE_GFX1250,
    /* (<8 x i32> A, <8 x i32> B, i16 fmt, <8 x float> C, i1, i1) -- gfx1250
     * K=64 fp8/bf8, where a lane's fragment is 32 packed bytes */
    WMMA_SHAPE_GFX1250_FP8
} _wmma_shape_t;

typedef struct _wmma_spec
{
    const char* op_id; /* the tile.mma op_id (no "tile." prefix)        */
    const char* decl_key; /* _need() key                                   */
    const char* intrinsic; /* fully-mangled @llvm.amdgcn.wmma....           */
    const char* ssa_elt; /* SSA operand element type                      */
    const char* call_elt; /* call-site operand element type                */
    int frag_width; /* A/B operand vector width (16 RDNA3/3.5, 8 RDNA4) */
    _wmma_shape_t shape; /* which intrinsic signature to emit             */
} _wmma_spec_t;

static const _wmma_spec_t WMMA_SPECS[] = {
    /* _RDNA_WMMA (RDNA3/3.5, frag_width 16) */
    {"wmma_f32_16x16x16_f16",
     "wmma.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16",
     "half",
     "half",
     16,
     WMMA_SHAPE_AB_C},
    {"wmma_f32_16x16x16_bf16",
     "wmma.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16",
     "bfloat",
     "i16",
     16,
     WMMA_SHAPE_AB_C},
    /* _RDNA_GFX12_WMMA (RDNA4, frag_width 8) */
    {"wmma_gfx12_f32_16x16x16_f16",
     "wmma.gfx12.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16",
     "half",
     "half",
     8,
     WMMA_SHAPE_AB_C},
    {"wmma_gfx12_f32_16x16x16_bf16",
     "wmma.gfx12.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16",
     "bfloat",
     "i16",
     8,
     WMMA_SHAPE_AB_C},
    /* _GFX1250_WMMA (K=32, frag_width 16). Note bf16 keeps ssa_elt ==
     * call_elt: the gfx1250 intrinsic takes <16 x bfloat> directly, so unlike
     * gfx11/gfx12 there is NO i16 bitcast. */
    {"wmma_gfx1250_f32_16x16x32_f16",
     "wmma.gfx1250.f32.16x16x32.f16",
     "llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16",
     "half",
     "half",
     16,
     WMMA_SHAPE_GFX1250},
    {"wmma_gfx1250_f32_16x16x32_bf16",
     "wmma.gfx1250.f32.16x16x32.bf16",
     "llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16",
     "bfloat",
     "bfloat",
     16,
     WMMA_SHAPE_GFX1250},
    /* _GFX1250_WMMA_FP8 (K=64). A/B are passed as <8 x i32> (mma_operand). */
    {"wmma_gfx1250_f32_16x16x64_fp8_fp8",
     "wmma.gfx1250.f32.16x16x64.fp8.fp8",
     "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32",
     "i32",
     "i32",
     8,
     WMMA_SHAPE_GFX1250_FP8},
    {"wmma_gfx1250_f32_16x16x64_fp8_bf8",
     "wmma.gfx1250.f32.16x16x64.fp8.bf8",
     "llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v8f32.v8i32",
     "i32",
     "i32",
     8,
     WMMA_SHAPE_GFX1250_FP8},
    {"wmma_gfx1250_f32_16x16x64_bf8_fp8",
     "wmma.gfx1250.f32.16x16x64.bf8.fp8",
     "llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v8f32.v8i32",
     "i32",
     "i32",
     8,
     WMMA_SHAPE_GFX1250_FP8},
    {"wmma_gfx1250_f32_16x16x64_bf8_bf8",
     "wmma.gfx1250.f32.16x16x64.bf8.bf8",
     "llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v8f32.v8i32",
     "i32",
     "i32",
     8,
     WMMA_SHAPE_GFX1250_FP8},
};
static const int WMMA_SPECS_N = (int)(sizeof(WMMA_SPECS) / sizeof(WMMA_SPECS[0]));

/* Integer WMMA spec table (Python _RDNA_WMMA_INT). Integer WMMA differs from
 * the float path: operands/accumulator are i32 vectors (A/B packed, C/D the i32
 * accumulator), and the intrinsic signature carries i1 signedness flags before
 * each matrix operand and a trailing i1 clamp. A and B are passed as <N x i32>;
 * packed bytes of the same width (for example <16 x i8>) are reinterpreted with
 * one bitcast (mma_operand). Quantized data is signed and within i32
 * range, so the flags are (signedA=1, signedB=1, clamp=0). */
typedef struct _wmma_int_spec
{
    const char* op_id; /* the tile.mma op_id (no "tile." prefix)        */
    const char* decl_key; /* _need() key                                   */
    const char* intrinsic; /* fully-mangled @llvm.amdgcn.wmma....           */
    int op_vec; /* A/B operand vector width                      */
    int acc_vec; /* accumulator/result vector width               */
} _wmma_int_spec_t;

static const _wmma_int_spec_t WMMA_INT_SPECS[] = {
    {"wmma_i32_16x16x16_iu8",
     "wmma.i32.16x16x16.iu8",
     "llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32",
     4,
     8},
    {"wmma_i32_16x16x16_iu4",
     "wmma.i32.16x16x16.iu4",
     "llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32",
     2,
     8},
};
static const int WMMA_INT_SPECS_N = (int)(sizeof(WMMA_INT_SPECS) / sizeof(WMMA_INT_SPECS[0]));

/* Emit an integer WMMA (iu8/iu4) call (Python _emit_wmma_int). The signature is
 * (i1 signedA, <N x i32> A, i1 signedB, <N x i32> B, <8 x i32> C, i1 clamp) with
 * an <8 x i32> result. Both signedness flags are 1 (signed quant data); clamp
 * is 0 (values stay within i32 range). */
static void _emit_wmma_int(rocke_lower_t* L, const rocke_op_t* op, const _wmma_int_spec_t* spec)
{
    const rocke_value_t* c = op->operands[2];
    rocke_ll_need(L, spec->decl_key);
    const rocke_type_t* register_type = mma_register_vector(L, rocke_i32(), spec->op_vec);
    const char* a_arg = mma_operand(L, op, spec->op_id, ROCKE_MMA_ROLE_A, register_type);
    const char* b_arg = mma_operand(L, op, spec->op_id, ROCKE_MMA_ROLE_B, register_type);
    rocke_ll_emitf(L,
                   "  %s = call <%d x i32> @%s("
                   "i1 1, <%d x i32> %s, "
                   "i1 1, <%d x i32> %s, "
                   "<%d x i32> %s, i1 0)",
                   mma_result_name(L, op),
                   spec->acc_vec,
                   spec->intrinsic,
                   spec->op_vec,
                   a_arg,
                   spec->op_vec,
                   b_arg,
                   spec->acc_vec,
                   rocke_ll_operand(L, c));
}

static void _emit_wmma(rocke_lower_t* L, const rocke_op_t* op, const char* op_id)
{
    const _wmma_spec_t* spec = NULL;
    const rocke_value_t *a, *b, *c;
    const char *a_arg, *b_arg;
    int w, i;

    if(!rocke_ll_live(L))
    {
        return;
    }
    if(op->num_operands != 3)
    {
        rocke_ll_fail(
            L, ROCKE_ERR_VALUE, "%s expects 3 operands", op->name ? op->name : "tile.mma");
    }

    /* Integer WMMA (iu8/iu4) is checked first, mirroring
     * Gfx11RdnaBackend.emit_wmma (int_spec lookup precedes the float table). */
    for(i = 0; i < WMMA_INT_SPECS_N; i++)
    {
        if(strcmp(WMMA_INT_SPECS[i].op_id, op_id) == 0)
        {
            _emit_wmma_int(L, op, &WMMA_INT_SPECS[i]);
            return;
        }
    }

    for(i = 0; i < WMMA_SPECS_N; i++)
    {
        if(strcmp(WMMA_SPECS[i].op_id, op_id) == 0)
        {
            spec = &WMMA_SPECS[i];
            break;
        }
    }
    if(spec == NULL)
    {
        /* WMMA_SPECS / WMMA_INT_SPECS above cover exactly the Python
         * _RDNA_WMMA (gfx11), _RDNA_GFX12_WMMA (gfx12), _RDNA_WMMA_INT and
         * _GFX1250_WMMA / _GFX1250_WMMA_FP8 op_ids -- the full set reachable
         * on every WMMA backend rocke_ll_backend_for resolves. */
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "unsupported RDNA WMMA op 'tile.%s' for %s",
                      op_id,
                      L->backend ? L->backend->gfx : "(rdna)");
    }

    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    w = spec->frag_width;

    rocke_ll_need(L, spec->decl_key);
    if(spec->shape == WMMA_SHAPE_GFX1250_FP8)
    {
        /* A and B are passed as the intrinsic's <8 x i32> (32 low-bit bytes per
         * lane); a <32 x i8> fragment is reinterpreted with one bitcast. */
        const rocke_type_t* register_type = mma_register_vector(L, rocke_i32(), w);
        a_arg = mma_operand(L, op, op_id, ROCKE_MMA_ROLE_A, register_type);
        b_arg = mma_operand(L, op, op_id, ROCKE_MMA_ROLE_B, register_type);
    }
    else
    {
        a_arg = rocke_ll_operand(L, a);
        b_arg = rocke_ll_operand(L, b);
    }

    if(strcmp(spec->call_elt, spec->ssa_elt) != 0)
    {
        /* bf16 (and any future type whose SSA element differs from the
         * intrinsic's operand element): bitcast <W x ssa_elt> -> <W x call_elt>. */
        const char* a_cast = rocke_ll_fresh(L, "wmma_a");
        const char* b_cast = rocke_ll_fresh(L, "wmma_b");
        rocke_ll_emitf(L,
                       "  %s = bitcast <%d x %s> %s to <%d x %s>",
                       a_cast,
                       w,
                       spec->ssa_elt,
                       a_arg,
                       w,
                       spec->call_elt);
        rocke_ll_emitf(L,
                       "  %s = bitcast <%d x %s> %s to <%d x %s>",
                       b_cast,
                       w,
                       spec->ssa_elt,
                       b_arg,
                       w,
                       spec->call_elt);
        a_arg = a_cast;
        b_arg = b_cast;
    }

    if(spec->shape == WMMA_SHAPE_GFX1250)
    {
        /* gfx1250 8-operand form: (i1 negA, A, i1 negB, B, i16 fmt, C,
         * i1 reuseA, i1 reuseB). The negate / format / reuse immediates are
         * pinned to the plain unscaled MMA. */
        rocke_ll_emitf(L,
                       "  %s = call <8 x float> @%s("
                       "i1 false, <%d x %s> %s, "
                       "i1 false, <%d x %s> %s, "
                       "i16 0, <8 x float> %s, "
                       "i1 false, i1 false)",
                       mma_result_name(L, op),
                       spec->intrinsic,
                       w,
                       spec->call_elt,
                       a_arg,
                       w,
                       spec->call_elt,
                       b_arg,
                       rocke_ll_operand(L, c));
        return;
    }
    if(spec->shape == WMMA_SHAPE_GFX1250_FP8)
    {
        /* gfx1250 K=64 6-operand form: (A, B, i16 fmt, C, i1, i1), format and
         * reuse immediates pinned to 0 (plain unscaled MMA). */
        rocke_ll_emitf(L,
                       "  %s = call <8 x float> @%s("
                       "<%d x %s> %s, <%d x %s> %s, "
                       "i16 0, <8 x float> %s, "
                       "i1 false, i1 false)",
                       mma_result_name(L, op),
                       spec->intrinsic,
                       w,
                       spec->call_elt,
                       a_arg,
                       w,
                       spec->call_elt,
                       b_arg,
                       rocke_ll_operand(L, c));
        return;
    }
    rocke_ll_emitf(L,
                   "  %s = call <8 x float> @%s("
                   "<%d x %s> %s, <%d x %s> %s, <8 x float> %s)",
                   mma_result_name(L, op),
                   spec->intrinsic,
                   w,
                   spec->call_elt,
                   a_arg,
                   w,
                   spec->call_elt,
                   b_arg,
                   rocke_ll_operand(L, c));
}

/* ====================================================================== */
/* f16 / bf16 / f32 / f64 dense MFMA atoms (table-driven)                 */
/* ====================================================================== */

/* The plain / scalar / bf16-bitcast MFMA atoms are structural twins: each one
 * guards (live + exactly 3 operands), tracks one intrinsic decl key, optionally
 * prepends a fixed two-line operand bitcast (the bf16 `_1k` atoms widen
 * <4 x bfloat> -> <4 x i16>), then emits a single MFMA call. They differ only
 * by four literal strings (decl key, intrinsic, A/B element-vector spelling,
 * accumulator/result vector spelling) plus the optional bitcast target. This
 * mirrors the WMMA_SPECS / _emit_wmma idiom already used in this file. */
typedef struct _mfma_spec
{
    const char* op_id; /* the tile.mma op_id (no "tile." prefix)         */
    const char* decl_key; /* _need() key                                    */
    const char* intrinsic; /* fully-mangled @llvm.amdgcn.mfma....            */
    const char* ab_ty; /* A/B SSA operand type spelling                  */
    const char* acc_ty; /* accumulator/result vector spelling             */
    const char* bitcast_to; /* operand bitcast target (NULL = no bitcast)     */
    /* The A/B register type when A and B pass through mma_operand (a value of
     * another shape is reinterpreted); NULL prints them as is, as Python does
     * for every atom but the f32 and f64 scalar ones. */
    const rocke_type_t* (*ab_register)(void);
} _mfma_spec_t;

static const _mfma_spec_t MFMA_SPECS[] = {
    {"mfma_f32_32x32x4_xf32",
     "mfma.f32.32x32x4.xf32",
     "llvm.amdgcn.mfma.f32.32x32x4.xf32",
     "<2 x i32>",
     "<16 x float>",
     "<2 x float>"},
    {"mfma_f32_16x16x8_xf32",
     "mfma.f32.16x16x8.xf32",
     "llvm.amdgcn.mfma.f32.16x16x8.xf32",
     "<2 x i32>",
     "<4 x float>",
     "<2 x float>"},
    {"mfma_f32_16x16x16_f16",
     "mfma.f32.16x16x16f16",
     "llvm.amdgcn.mfma.f32.16x16x16f16",
     "<4 x half>",
     "<4 x float>",
     NULL},
    {"mfma_f32_16x16x32_f16",
     "mfma.f32.16x16x32.f16",
     "llvm.amdgcn.mfma.f32.16x16x32.f16",
     "<8 x half>",
     "<4 x float>",
     NULL},
    /* bf16 `_1k`: bitcast <4 x bfloat> -> <4 x i16> before the call. */
    {"mfma_f32_16x16x16_bf16",
     "mfma.f32.16x16x16bf16.1k",
     "llvm.amdgcn.mfma.f32.16x16x16bf16.1k",
     "<4 x bfloat>",
     "<4 x float>",
     "<4 x i16>"},
    {"mfma_f32_16x16x32_bf16",
     "mfma.f32.16x16x32.bf16",
     "llvm.amdgcn.mfma.f32.16x16x32.bf16",
     "<8 x bfloat>",
     "<4 x float>",
     NULL},
    {"mfma_f32_32x32x8_f16",
     "mfma.f32.32x32x8f16",
     "llvm.amdgcn.mfma.f32.32x32x8f16",
     "<4 x half>",
     "<16 x float>",
     NULL},
    /* bf16 `_1k`: bitcast <4 x bfloat> -> <4 x i16> exactly like 16x16x16. */
    {"mfma_f32_32x32x8_bf16",
     "mfma.f32.32x32x8bf16.1k",
     "llvm.amdgcn.mfma.f32.32x32x8bf16.1k",
     "<4 x bfloat>",
     "<16 x float>",
     "<4 x i16>"},
    /* fp64 scalar atom: A/B are single doubles per lane. */
    {"mfma_f64_16x16x4_f64",
     "mfma.f64.16x16x4f64",
     "llvm.amdgcn.mfma.f64.16x16x4f64",
     "double",
     "<4 x double>",
     NULL,
     rocke_f64},
    /* fp32 (TF32-class) scalar atoms: A/B are single floats per lane. */
    {"mfma_f32_16x16x4_f32",
     "mfma.f32.16x16x4f32",
     "llvm.amdgcn.mfma.f32.16x16x4f32",
     "float",
     "<4 x float>",
     NULL,
     rocke_f32},
    {"mfma_f32_32x32x2_f32",
     "mfma.f32.32x32x2f32",
     "llvm.amdgcn.mfma.f32.32x32x2f32",
     "float",
     "<16 x float>",
     NULL,
     rocke_f32},
    {"mfma_f32_32x32x16_f16",
     "mfma.f32.32x32x16.f16",
     "llvm.amdgcn.mfma.f32.32x32x16.f16",
     "<8 x half>",
     "<16 x float>",
     NULL},
    {"mfma_f32_32x32x16_bf16",
     "mfma.f32.32x32x16.bf16",
     "llvm.amdgcn.mfma.f32.32x32x16.bf16",
     "<8 x bfloat>",
     "<16 x float>",
     NULL},
    {"mfma_f32_4x4x4_f16",
     "mfma.f32.4x4x4f16",
     "llvm.amdgcn.mfma.f32.4x4x4f16",
     "<4 x half>",
     "<4 x float>",
     NULL},
    /* bf16 `_1k`: bitcast <4 x bfloat> -> <4 x i16> exactly like 16x16x16. */
    {"mfma_f32_4x4x4_bf16",
     "mfma.f32.4x4x4bf16.1k",
     "llvm.amdgcn.mfma.f32.4x4x4bf16.1k",
     "<4 x bfloat>",
     "<4 x float>",
     "<4 x i16>"},
};
static const int MFMA_SPECS_N = (int)(sizeof(MFMA_SPECS) / sizeof(MFMA_SPECS[0]));

/* Emit one plain/scalar/bf16-bitcast MFMA call from its spec. Reproduces the
 * exact per-atom fragment order: guard, _need, optional fixed two-line operand
 * bitcast, then the single MFMA call. Subexpressions are hoisted to temporaries
 * left-to-right so the emitted text is independent of argument-evaluation order
 * (the bitcast path allocates fresh SSA names, which is order-sensitive). */
static void _emit_mfma(rocke_lower_t* L, const rocke_op_t* op, const _mfma_spec_t* spec)
{
    const rocke_value_t *a, *b, *c;
    const char *a_arg, *b_arg;
    const char* call_ty;

    if(!rocke_ll_live(L) || op->num_operands != 3)
    {
        if(rocke_ll_live(L))
        {
            rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    rocke_ll_need(L, spec->decl_key);

    if(spec->bitcast_to != NULL)
    {
        /* bf16 `_1k`: bitcast the A/B operands to the integer vector the
         * intrinsic expects. Fresh names are allocated A-then-B, matching the
         * original per-atom order, before either bitcast line is emitted. */
        const char* a_cast = rocke_ll_fresh(L, "mfma_a_i16");
        const char* b_cast = rocke_ll_fresh(L, "mfma_b_i16");
        rocke_ll_emitf(L,
                       "  %s = bitcast %s %s to %s",
                       a_cast,
                       spec->ab_ty,
                       rocke_ll_operand(L, a),
                       spec->bitcast_to);
        rocke_ll_emitf(L,
                       "  %s = bitcast %s %s to %s",
                       b_cast,
                       spec->ab_ty,
                       rocke_ll_operand(L, b),
                       spec->bitcast_to);
        a_arg = a_cast;
        b_arg = b_cast;
        call_ty = spec->bitcast_to;
    }
    else if(spec->ab_register != NULL)
    {
        const rocke_type_t* register_type = spec->ab_register();
        a_arg = mma_operand(L, op, spec->op_id, ROCKE_MMA_ROLE_A, register_type);
        b_arg = mma_operand(L, op, spec->op_id, ROCKE_MMA_ROLE_B, register_type);
        call_ty = spec->ab_ty;
    }
    else
    {
        a_arg = rocke_ll_operand(L, a);
        b_arg = rocke_ll_operand(L, b);
        call_ty = spec->ab_ty;
    }

    rocke_ll_emitf(L,
                   "  %s = call %s @%s("
                   "%s %s, %s %s, %s %s, i32 0, i32 0, i32 0)",
                   mma_result_name(L, op),
                   spec->acc_ty,
                   spec->intrinsic,
                   call_ty,
                   a_arg,
                   call_ty,
                   b_arg,
                   spec->acc_ty,
                   rocke_ll_operand(L, c));
}

/* Resolve op_id against the dense MFMA table; emit and return true on a hit. */
static bool _try_emit_mfma_table(rocke_lower_t* L, const rocke_op_t* op, const char* op_id)
{
    int i;
    for(i = 0; i < MFMA_SPECS_N; i++)
    {
        if(strcmp(MFMA_SPECS[i].op_id, op_id) == 0)
        {
            _emit_mfma(L, op, &MFMA_SPECS[i]);
            return true;
        }
    }
    return false;
}

/* ====================================================================== */
/* Shared FP8 / BF8 MFMA body (Python _lower_mfma_fp8_bf8)               */
/* ====================================================================== */

void rocke_ll_lower_mfma_fp8_bf8(
    rocke_lower_t* L, const rocke_op_t* op, const char* dtype, int out_vec, const char* intrinsic)
{
    const rocke_value_t *a, *b, *c;
    const char* ab_ty;
    const char *a_cast, *b_cast;
    char key[64];
    char a_hint[32], b_hint[32];

    if(!rocke_ll_live(L))
    {
        return;
    }
    if(op->num_operands != 3)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 3 operands", op->name);
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];

    /* _need(f"mfma.f32.{intrinsic}") */
    snprintf(key, sizeof(key), "mfma.f32.%s", intrinsic);
    rocke_ll_need(L, key);

    /* LLVM 21+ (llvm22 / llvm23) packs the 64-bit-per-lane A/B operand as scalar
     * i64; LLVM 20 uses <2 x i32>. Same bits, different lane packing. */
    ab_ty = rocke_ll_flavor_is_modern(L->flavor) ? "i64" : "<2 x i32>";

    snprintf(a_hint, sizeof(a_hint), "mfma_a_%s", dtype ? dtype : "f8");
    snprintf(b_hint, sizeof(b_hint), "mfma_b_%s", dtype ? dtype : "f8");
    a_cast = rocke_ll_fresh(L, a_hint);
    b_cast = rocke_ll_fresh(L, b_hint);

    rocke_ll_emitf(L, "  %s = bitcast <8 x i8> %s to %s", a_cast, rocke_ll_operand(L, a), ab_ty);
    rocke_ll_emitf(L, "  %s = bitcast <8 x i8> %s to %s", b_cast, rocke_ll_operand(L, b), ab_ty);
    rocke_ll_emitf(L,
                   "  %s = call <%d x float> @llvm.amdgcn.mfma.f32.%s("
                   "%s %s, %s %s, <%d x float> %s, i32 0, i32 0, i32 0)",
                   mma_result_name(L, op),
                   out_vec,
                   intrinsic,
                   ab_ty,
                   a_cast,
                   ab_ty,
                   b_cast,
                   out_vec,
                   rocke_ll_operand(L, c));
}

static void _op_tile_mfma_f64_4x4x4_f64(rocke_lower_t* L, const rocke_op_t* op)
{
    const char *a_arg, *b_arg, *c_elem, *d_elem;
    if(!rocke_ll_live(L) || op->num_operands != 3)
    {
        if(rocke_ll_live(L))
        {
            rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    rocke_ll_need(L, "mfma.f64.4x4x4f64");
    a_arg = mma_operand(L, op, "mfma_f64_4x4x4_f64", ROCKE_MMA_ROLE_A, rocke_f64());
    b_arg = mma_operand(L, op, "mfma_f64_4x4x4_f64", ROCKE_MMA_ROLE_B, rocke_f64());
    c_elem = rocke_ll_fresh(L, "mfma_c");
    d_elem = rocke_ll_fresh(L, "mfma_d");
    rocke_ll_emitf(L,
                   "  %s = extractelement <1 x double> %s, i32 0",
                   c_elem,
                   rocke_ll_operand(L, op->operands[2]));
    rocke_ll_emitf(L,
                   "  %s = call double @llvm.amdgcn.mfma.f64.4x4x4f64("
                   "double %s, double %s, double %s, i32 0, i32 0, i32 0)",
                   d_elem,
                   a_arg,
                   b_arg,
                   c_elem);
    rocke_ll_emitf(L,
                   "  %s = insertelement <1 x double> undef, double %s, i32 0",
                   mma_result_name(L, op),
                   d_elem);
}

static void _op_tile_mfma_f32_16x16x32_fp8(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_lower_mfma_fp8_bf8(L, op, "fp8", 4, "16x16x32.fp8.fp8");
}

static void _op_tile_mfma_f32_16x16x32_bf8(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_lower_mfma_fp8_bf8(L, op, "bf8", 4, "16x16x32.bf8.bf8");
}

static void _op_tile_mfma_f32_32x32x16_fp8(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_lower_mfma_fp8_bf8(L, op, "fp8", 16, "32x32x16.fp8.fp8");
}

static void _op_tile_mfma_f32_32x32x16_bf8(rocke_lower_t* L, const rocke_op_t* op)
{
    rocke_ll_lower_mfma_fp8_bf8(L, op, "bf8", 16, "32x32x16.bf8.bf8");
}

/* ====================================================================== */
/* K=128 f8f6f4 ops: MX-scaled, unscaled hero, unscaled all-format        */
/* ====================================================================== */

/* Declaration key of the K=128 f8f6f4 MFMA (A and B as <8 x i32>): the intrinsic
 * name without the "llvm.amdgcn." prefix. Mirrors Python _F8F6F4_16X16X128_KEY. */
#define ROCKE_F8F6F4_16X16X128_KEY "mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32"

/* Shared emitter for the MX op and the hero atom (Python
 * _emit_f8f6f4_16x16x128). A and B are bit-cast to <8 x i32> (256 bits per
 * lane). The call passes the format selectors cbsz and blgp and opsel 0. The
 * scale operand texts are the MX scale values, or "0" for the hero atom. The
 * backend selects the unscaled instruction only when both scale operands are
 * literal 0. */
static void emit_f8f6f4_16x16x128(rocke_lower_t* L,
                                  const rocke_op_t* op,
                                  const rocke_value_t* a,
                                  const rocke_value_t* b,
                                  const rocke_value_t* c,
                                  const char* a_scale,
                                  const char* b_scale,
                                  const char* a_name,
                                  const char* b_name,
                                  int cbsz,
                                  int blgp)
{
    const char *a_packed, *b_packed;
    const char *a_ty, *b_ty;

    rocke_ll_need(L, ROCKE_F8F6F4_16X16X128_KEY);
    a_packed = rocke_ll_fresh(L, a_name);
    b_packed = rocke_ll_fresh(L, b_name);
    a_ty = rocke_ll_llvm_type(L, a->type);
    b_ty = rocke_ll_llvm_type(L, b->type);
    if(strcmp(a_ty, "<8 x i32>") != 0)
    {
        rocke_ll_emitf(
            L, "  %s = bitcast %s %s to <8 x i32>", a_packed, a_ty, rocke_ll_operand(L, a));
    }
    else
    {
        a_packed = rocke_ll_operand(L, a);
    }
    if(strcmp(b_ty, "<8 x i32>") != 0)
    {
        rocke_ll_emitf(
            L, "  %s = bitcast %s %s to <8 x i32>", b_packed, b_ty, rocke_ll_operand(L, b));
    }
    else
    {
        b_packed = rocke_ll_operand(L, b);
    }
    rocke_ll_emitf(L,
                   "  %s = call <4 x float> "
                   "@llvm.amdgcn." ROCKE_F8F6F4_16X16X128_KEY "("
                   "<8 x i32> %s, <8 x i32> %s, <4 x float> %s, "
                   "i32 %d, i32 %d, i32 0, i32 %s, i32 0, i32 %s)",
                   mma_result_name(L, op),
                   a_packed,
                   b_packed,
                   rocke_ll_operand(L, c),
                   cbsz,
                   blgp,
                   a_scale,
                   b_scale);
}

/* Check the formats and operand widths of op and return cbsz and blgp (Python
 * _f8f6f4_16x16x128_selectors). The checks run here as well as in the builder,
 * because serialized IR reaches the lowerer without passing through the builder. */
static void f8f6f4_16x16x128_selectors(rocke_lower_t* L,
                                       const rocke_op_t* op,
                                       const rocke_value_t* a,
                                       const rocke_value_t* b,
                                       int* cbsz,
                                       int* blgp)
{
    char msg[320];
    if(rocke_f8f6f4_16x16x128_check(rocke_attr_get_str(&op->attrs, "a_format"),
                                    rocke_attr_get_str(&op->attrs, "b_format"),
                                    a->type,
                                    b->type,
                                    msg,
                                    sizeof(msg),
                                    cbsz,
                                    blgp))
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s", msg);
    }
}

/* MX op: the f8f6f4 MFMA with in-instruction E8M0 scales. The instruction reads
 * one E8M0 byte from each scale and multiplies by 2^(e - 127); e = 0 gives
 * 2^-127, not 1.0. Both scale operands at literal 0 make the backend select the
 * unscaled instruction instead. The optional attributes a_format and b_format
 * name the data type (default fp8e4m3) and must be equal in this version. */
static void _op_tile_mfma_scale_f32_16x16x128_f8f6f4(rocke_lower_t* L, const rocke_op_t* op)
{
    int cbsz = 0, blgp = 0;
    if(!rocke_ll_live(L))
    {
        return;
    }
    if(op->num_operands != 5)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 5 operands", op->name);
    }
    f8f6f4_16x16x128_selectors(L, op, op->operands[0], op->operands[1], &cbsz, &blgp);
    emit_f8f6f4_16x16x128(L,
                          op,
                          op->operands[0],
                          op->operands[1],
                          op->operands[2],
                          rocke_ll_operand(L, op->operands[3]),
                          rocke_ll_operand(L, op->operands[4]),
                          "mxa",
                          "mxb",
                          cbsz,
                          blgp);
}

/* Unscaled K=128 MFMA for fp8, bf8, fp6 and fp4. gfx950 has no unscaled
 * intrinsic for this instruction. The op calls the scaled intrinsic with both
 * scale operands at literal 0, and the backend then selects
 * v_mfma_f32_16x16x128_f8f6f4. The literal 0 marks "no scale". As an E8M0 value,
 * 0 would mean 2^-127, so a loaded or computed 0 must never replace the literal
 * here. The attributes a_format and b_format work as for the MX op. */
static void _op_tile_mfma_f32_16x16x128_fp4fp6fp8(rocke_lower_t* L, const rocke_op_t* op)
{
    int cbsz = 0, blgp = 0;
    if(!rocke_ll_live(L))
    {
        return;
    }
    if(op->num_operands != 3)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 3 operands", op->name);
    }
    f8f6f4_16x16x128_selectors(L, op, op->operands[0], op->operands[1], &cbsz, &blgp);
    emit_f8f6f4_16x16x128(L,
                          op,
                          op->operands[0],
                          op->operands[1],
                          op->operands[2],
                          "0",
                          "0",
                          "a_f8f6f4",
                          "b_f8f6f4",
                          cbsz,
                          blgp);
}

static void _op_tile_mfma_f32_16x16x128_fp8(rocke_lower_t* L, const rocke_op_t* op)
{
    /* Unscaled fp8 16x16x128 hero atom: gfx950 has no unscaled intrinsic for the
     * K=128 f8f6f4 MFMA, so the atom calls the scaled intrinsic with both scale
     * operands at literal 0. The backend then selects the unscaled instruction
     * v_mfma_f32_16x16x128_f8f6f4 (LLVM TableGen pattern
     * UnscaledMFMAOptimizationPat; Composable Kernel relies on the same rule).
     * The literal 0 marks "no scale". As an E8M0 value, 0 would mean 2^-127, so
     * a loaded or computed 0 must never replace the literal here. */
    if(!rocke_ll_live(L))
    {
        return;
    }
    if(op->num_operands != 3)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 3 operands", op->name);
    }
    emit_f8f6f4_16x16x128(L,
                          op,
                          op->operands[0],
                          op->operands[1],
                          op->operands[2],
                          "0",
                          "0",
                          "a_fp8_128",
                          "b_fp8_128",
                          0,
                          0);
}

/* ====================================================================== */
/* register_p_from_qk_c (Python _op_tile_register_p_from_qk_c, P13)       */
/* ====================================================================== */

static void _op_tile_register_p_from_qk_c(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* qk_c;
    const char* target;
    const char* target_llvm;
    const char* elems[8];
    const char* prev;
    int i;

    if(!rocke_ll_live(L))
    {
        return;
    }
    if(op->num_operands != 1)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 1 operand", op->name);
    }
    qk_c = op->operands[0];
    target = rocke_attr_get_str(&op->attrs, "target_dtype");
    if(!target)
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "register_p_from_qk_c: missing target_dtype attribute");
    }
    if(strcmp(target, "f16") == 0)
    {
        target_llvm = "half";
    }
    else if(strcmp(target, "bf16") == 0)
    {
        target_llvm = "bfloat";
    }
    else
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "register_p_from_qk_c: bad target_dtype '%s'", target);
    }

    /* Extract 16 f32 cells, fptrunc the first 8 in canonical order. */
    for(i = 0; i < 8; i++)
    {
        char ehint[8], thint[8];
        const char *e, *t;
        snprintf(ehint, sizeof(ehint), "pe%d", i);
        snprintf(thint, sizeof(thint), "pt%d", i);
        e = rocke_ll_fresh(L, ehint);
        rocke_ll_emitf(
            L, "  %s = extractelement <16 x float> %s, i32 %d", e, rocke_ll_operand(L, qk_c), i);
        t = rocke_ll_fresh(L, thint);
        rocke_ll_emitf(L, "  %s = fptrunc float %s to %s", t, e, target_llvm);
        elems[i] = t;
    }

    /* Pack into <8 x dtype>; the last insertelement is the op result. */
    prev = "undef";
    for(i = 0; i < 8; i++)
    {
        const char* name;
        if(i == 7)
        {
            name = mma_result_name(L, op);
        }
        else
        {
            char phint[8];
            snprintf(phint, sizeof(phint), "pp%d", i);
            name = rocke_ll_fresh(L, phint);
        }
        rocke_ll_emitf(L,
                       "  %s = insertelement <8 x %s> %s, %s %s, i32 %d",
                       name,
                       target_llvm,
                       prev,
                       target_llvm,
                       elems[i],
                       i);
        prev = name;
    }
}

/* ====================================================================== */
/* Bucket registration hook                                               */
/* ====================================================================== */

void rocke_ll_register_mma(void)
{
    /* The frozen ir.h only exposes ROCKE_OP_TILE_MMA and
     * ROCKE_OP_TILE_REGISTER_P_FROM_QK_C as opcodes; the ISA-named MFMA / WMMA
     * handlers are not distinct opcodes and are reached through _op_tile_mma's
     * op_id router (mirroring the Python CDNA emit_mma re-dispatch). */
    rocke_ll_set_handler(ROCKE_OP_TILE_MMA, _op_tile_mma);
    rocke_ll_set_handler(ROCKE_OP_TILE_REGISTER_P_FROM_QK_C, _op_tile_register_p_from_qk_c);
}

} /* namespace ckc */
