// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_lower_llvm_crosslane.c.c -- BUCKET 5 of the C99 port of
 * rocke.core.lower_llvm.
 *
 * Cross-lane / DPP / permute / swizzle / ballot ops, plus the general and
 * multi-output inline_asm lowering. Faithful translation of the Python
 * _Lowerer._op_tile_* handlers (lines ~2320-2700 of lower_llvm.py).
 *
 * This TU owns rocke_ll_emit_wave_ballot (the shared ballot emitter, declared in
 * the internal header and used by wave_all / wave_any) and registers every
 * handler in this bucket via rocke_ll_register_crosslane().
 *
 * Shared plumbing (rocke_ll_emit, rocke_ll_fresh, rocke_ll_operand, rocke_ll_need,
 * rocke_ll_llvm_type, rocke_ll_smem_global_name, rocke_ll_smem_storage_type, ...)
 * lives in bucket 0 and is reached through rocke/lower_llvm_internal.h.
 */

#include "rocke/lower_llvm_internal.h"

#include <string.h>

namespace ckc
{

/* ====================================================================== */
/* Small local helpers                                                    */
/* ====================================================================== */

/* The single-result name (Python op.result.name). The IR contract guarantees
 * a result exists for these ops; mirror Python by reading results[0]. */
static const char* ll_result_name(const rocke_op_t* op)
{
    return op->results[0]->name;
}

/* RAII guard for a heap-backed strbuf: frees the buffer on scope exit, so an
 * exception raised by a nested emit/type helper (rocke_ll_fail -> throw) while a
 * strbuf is live cannot leak its heap allocation. Codegen-neutral: it only
 * affects the throw/unwind path, never the bytes emitted on success. */
struct ll_strbuf_guard
{
    rocke_strbuf_t sb;
    bool ok;
    explicit ll_strbuf_guard(size_t cap)
    {
        ok = (rocke_strbuf_init(&sb, cap) == 0);
    }
    ~ll_strbuf_guard()
    {
        rocke_strbuf_free(&sb);
    }
    ll_strbuf_guard(const ll_strbuf_guard&) = delete;
    ll_strbuf_guard& operator=(const ll_strbuf_guard&) = delete;
};

/* ====================================================================== */
/* Shared ballot emit (Python _emit_wave_ballot) -- OWNED BY THIS BUCKET  */
/* ====================================================================== */

void rocke_ll_emit_wave_ballot(rocke_lower_t* L, const rocke_value_t* pred, const char* result_name)
{
    const char* cmp_name;
    if(!rocke_ll_live(L))
    {
        return;
    }
    rocke_ll_need(L, "ballot.i64");
    cmp_name = rocke_ll_fresh(L, "ballot_pred");
    rocke_ll_emitf(L, "  %s = icmp ne i32 %s, 0", cmp_name, rocke_ll_operand(L, pred));
    rocke_ll_emitf(L, "  %s = call i64 @llvm.amdgcn.ballot.i64(i1 %s)", result_name, cmp_name);
}

/* ====================================================================== */
/* readfirstlane / pin_sgpr / lane_id                                     */
/* ====================================================================== */

static void _op_tile_readfirstlane(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const char* tyname = v->type->name;
    const char* intrinsic_key;
    const char* ty;

    if(strcmp(tyname, "i32") == 0)
    {
        intrinsic_key = "readfirstlane.i32";
        ty = "i32";
    }
    else if(strcmp(tyname, "i64") == 0)
    {
        intrinsic_key = "readfirstlane.i64";
        ty = "i64";
    }
    else
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.readfirstlane: unsupported type '%s'", tyname);
    }
    rocke_ll_need(L, intrinsic_key);
    rocke_ll_emitf(L,
                   "  %s = call %s @llvm.amdgcn.readfirstlane.%s(%s %s)",
                   ll_result_name(op),
                   ty,
                   ty,
                   ty,
                   rocke_ll_operand(L, v));
}

static void _op_tile_pin_sgpr(rocke_lower_t* L, const rocke_op_t* op)
{
    /* No-op inline asm whose output (SGPR class) is tied to its input
     * (constraint "0" matches operand 0). Mirrors HIP's
     * ``asm volatile("" : "+s"(x))``. NOT marked sideeffect (see Python
     * docstring: a sideeffect-tagged SGPR-class asm corrupts AMDGPU's
     * divergence analysis). */
    const rocke_value_t* v = op->operands[0];
    const char* tyname = v->type->name;
    const char* ty;

    if(strcmp(tyname, "i32") == 0)
    {
        ty = "i32";
    }
    else if(strcmp(tyname, "i64") == 0)
    {
        ty = "i64";
    }
    else
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.pin_sgpr: unsupported type '%s'", tyname);
    }
    rocke_ll_emitf(L,
                   "  %s = call %s asm \"\", \"=s,0\"(%s %s)",
                   ll_result_name(op),
                   ty,
                   ty,
                   rocke_ll_operand(L, v));
}

static void _op_tile_lane_id(rocke_lower_t* L, const rocke_op_t* op)
{
    /* lane = mbcnt.hi(-1, mbcnt.lo(-1, 0)). */
    const char* lo;
    rocke_ll_need(L, "mbcnt.lo");
    rocke_ll_need(L, "mbcnt.hi");
    lo = rocke_ll_fresh(L, "mbcnt.lo");
    rocke_ll_emitf(L, "  %s = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)", lo);
    rocke_ll_emitf(
        L, "  %s = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %s)", ll_result_name(op), lo);
}

/* ====================================================================== */
/* wave ballot / all / any                                                */
/* ====================================================================== */

static void _op_tile_wave_ballot(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* pred = op->operands[0];
    rocke_ll_emit_wave_ballot(L, pred, ll_result_name(op));
}

static void _op_tile_wave_all(rocke_lower_t* L, const rocke_op_t* op)
{
    /* wave_all(p) = (ballot(p) == -1). */
    const rocke_value_t* pred = op->operands[0];
    const char* b_name;
    const char* eq_name;
    rocke_ll_need(L, "ballot.i64");
    b_name = rocke_ll_fresh(L, "ballot");
    rocke_ll_emit_wave_ballot(L, pred, b_name);
    eq_name = rocke_ll_fresh(L, "all_eq");
    rocke_ll_emitf(L, "  %s = icmp eq i64 %s, -1", eq_name, b_name);
    rocke_ll_emitf(L, "  %s = zext i1 %s to i32", ll_result_name(op), eq_name);
}

static void _op_tile_wave_any(rocke_lower_t* L, const rocke_op_t* op)
{
    /* wave_any(p) = (ballot(p) != 0). */
    const rocke_value_t* pred = op->operands[0];
    const char* b_name;
    const char* ne_name;
    b_name = rocke_ll_fresh(L, "ballot");
    rocke_ll_emit_wave_ballot(L, pred, b_name);
    ne_name = rocke_ll_fresh(L, "any_nz");
    rocke_ll_emitf(L, "  %s = icmp ne i64 %s, 0", ne_name, b_name);
    rocke_ll_emitf(L, "  %s = zext i1 %s to i32", ll_result_name(op), ne_name);
}

/* ====================================================================== */
/* ds_bpermute / ds_bpermute_b64 / ds_swizzle_xor                         */
/* ====================================================================== */

static void _op_tile_ds_bpermute(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* addr = op->operands[0];
    const rocke_value_t* data = op->operands[1];
    const char* addr_s;
    const char* data_s;
    rocke_ll_need(L, "ds.bpermute");
    addr_s = rocke_ll_operand(L, addr);
    data_s = rocke_ll_operand(L, data);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.ds.bpermute(i32 %s, i32 %s)",
                   ll_result_name(op),
                   addr_s,
                   data_s);
}

static void _op_tile_ds_bpermute_b64(rocke_lower_t* L, const rocke_op_t* op)
{
    /* Packed-i64 ds_bpermute: two 32-bit ds_bpermute calls on the high/low
     * halves of an i64 payload, recombined. */
    const rocke_value_t* addr = op->operands[0];
    const rocke_value_t* data = op->operands[1];
    const char* addr_s;
    const char* data_s;
    const char* lo32;
    const char* hi32;
    const char* sh;
    const char* plo;
    const char* phi;
    const char* wide_lo;
    const char* wide_hi;
    const char* shifted;

    rocke_ll_need(L, "ds.bpermute");
    lo32 = rocke_ll_fresh(L, "bp64.lo");
    hi32 = rocke_ll_fresh(L, "bp64.hi");
    sh = rocke_ll_fresh(L, "bp64.sh");
    plo = rocke_ll_fresh(L, "bp64.plo");
    phi = rocke_ll_fresh(L, "bp64.phi");
    wide_lo = rocke_ll_fresh(L, "bp64.wlo");
    wide_hi = rocke_ll_fresh(L, "bp64.whi");
    shifted = rocke_ll_fresh(L, "bp64.sh2");

    addr_s = rocke_ll_operand(L, addr);
    data_s = rocke_ll_operand(L, data);

    rocke_ll_emitf(L, "  %s = trunc i64 %s to i32", lo32, data_s);
    rocke_ll_emitf(L, "  %s = lshr i64 %s, 32", sh, data_s);
    rocke_ll_emitf(L, "  %s = trunc i64 %s to i32", hi32, sh);
    rocke_ll_emitf(
        L, "  %s = call i32 @llvm.amdgcn.ds.bpermute(i32 %s, i32 %s)", plo, addr_s, lo32);
    rocke_ll_emitf(
        L, "  %s = call i32 @llvm.amdgcn.ds.bpermute(i32 %s, i32 %s)", phi, addr_s, hi32);
    rocke_ll_emitf(L, "  %s = zext i32 %s to i64", wide_lo, plo);
    rocke_ll_emitf(L, "  %s = zext i32 %s to i64", wide_hi, phi);
    rocke_ll_emitf(L, "  %s = shl i64 %s, 32", shifted, wide_hi);
    rocke_ll_emitf(L, "  %s = or i64 %s, %s", ll_result_name(op), wide_lo, shifted);
}

static void _op_tile_ds_swizzle_xor(rocke_lower_t* L, const rocke_op_t* op)
{
    /* ds_swizzle_b32 XOR butterfly via SWAP-mode encoding:
     *   offset = (xor_mask << 10) | 0x1F  */
    int64_t xor_mask = 0;
    int64_t offset;
    const rocke_value_t* data;
    if(!rocke_attr_get_int(&op->attrs, "xor_mask", &xor_mask))
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.ds_swizzle_xor: missing 'xor_mask'");
    }
    offset = (xor_mask << 10) | 0x1F;
    data = op->operands[0];
    rocke_ll_need(L, "ds.swizzle");
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.ds.swizzle(i32 %s, i32 %lld)",
                   ll_result_name(op),
                   rocke_ll_operand(L, data),
                   (long long)offset);
}

/* ====================================================================== */
/* mov_dpp / permlane32_swap / perm_b32 / permlanex16 / byte_perm         */
/* ====================================================================== */

static void _op_tile_mov_dpp(rocke_lower_t* L, const rocke_op_t* op)
{
    /* v_mov_b32_dpp row-shift. DPP control word:
     *   row_shr -> 0x110 | (shift & 0xF)
     *   row_shl -> 0x100 | (shift & 0xF)
     * Passes src as `old` so unfilled lanes retain their input. */
    const rocke_value_t* data = op->operands[0];
    bool bound_ctrl;
    int64_t shift = 0;
    int dpp_ctrl;
    const char* data_s;

    rocke_ll_need(L, "update.dpp.i32");
    bound_ctrl = rocke_attr_get_bool(&op->attrs, "bound_ctrl", false);

    if(rocke_attr_get_int(&op->attrs, "row_shr", &shift))
    {
        dpp_ctrl = 0x110 | ((int)shift & 0xF);
    }
    else if(rocke_attr_get_int(&op->attrs, "row_shl", &shift))
    {
        dpp_ctrl = 0x100 | ((int)shift & 0xF);
    }
    else
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.mov_dpp: missing 'row_shr'/'row_shl'");
    }

    data_s = rocke_ll_operand(L, data);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.update.dpp.i32("
                   "i32 %s, i32 %s, i32 %d, i32 15, i32 15, i1 %s)",
                   ll_result_name(op),
                   data_s,
                   data_s,
                   dpp_ctrl,
                   bound_ctrl ? "true" : "false");
}

static void _op_tile_permlane32_swap(rocke_lower_t* L, const rocke_op_t* op)
{
    /* v_permlane32_swap_b32 -- wave64 half-swap in 1 VALU op. */
    const rocke_value_t* lo = op->operands[0];
    const rocke_value_t* hi = op->operands[1];
    const char* tmp;
    const char* lo_s;
    const char* hi_s;
    rocke_ll_need(L, "permlane32.swap");
    tmp = rocke_ll_fresh(L, "psw.tmp");
    lo_s = rocke_ll_operand(L, lo);
    hi_s = rocke_ll_operand(L, hi);
    rocke_ll_emitf(L,
                   "  %s = call { i32, i32 } @llvm.amdgcn.permlane32.swap("
                   "i32 %s, i32 %s, i1 false, i1 false)",
                   tmp,
                   lo_s,
                   hi_s);
    rocke_ll_emitf(L, "  %s = extractvalue { i32, i32 } %s, 0", op->results[0]->name, tmp);
    rocke_ll_emitf(L, "  %s = extractvalue { i32, i32 } %s, 1", op->results[1]->name, tmp);
}

static void _op_tile_perm_b32(rocke_lower_t* L, const rocke_op_t* op)
{
    /* v_perm_b32 -- in-lane byte select across two VGPRs. */
    const rocke_value_t* src0 = op->operands[0];
    const rocke_value_t* src1 = op->operands[1];
    const rocke_value_t* sel = op->operands[2];
    const char* src0_s;
    const char* src1_s;
    const char* sel_s;
    rocke_ll_need(L, "amdgcn.perm");
    src0_s = rocke_ll_operand(L, src0);
    src1_s = rocke_ll_operand(L, src1);
    sel_s = rocke_ll_operand(L, sel);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.perm(i32 %s, i32 %s, i32 %s)",
                   ll_result_name(op),
                   src0_s,
                   src1_s,
                   sel_s);
}

static void _op_tile_permlanex16(rocke_lower_t* L, const rocke_op_t* op)
{
    /* v_permlanex16_b32 swap with the lane^16 partner. */
    const rocke_value_t* v = op->operands[0];
    const char* src;
    rocke_ll_need(L, "amdgcn.permlanex16");
    src = rocke_ll_operand(L, v);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.permlanex16("
                   "i32 %s, i32 %s, i32 1985229328, i32 -19088744, "
                   "i1 false, i1 true)",
                   ll_result_name(op),
                   src,
                   src);
}

static void _op_tile_byte_perm(rocke_lower_t* L, const rocke_op_t* op)
{
    /* v_perm_b32 byte shuffle via llvm.amdgcn.perm. */
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    int64_t sel = 0;
    uint32_t sel_u;
    const char* a_s;
    const char* b_s;
    if(!rocke_attr_get_int(&op->attrs, "sel", &sel))
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.byte_perm: missing 'sel'");
    }
    sel_u = (uint32_t)((uint64_t)sel & 0xFFFFFFFFu);
    rocke_ll_need(L, "amdgcn.perm");
    a_s = rocke_ll_operand(L, a);
    b_s = rocke_ll_operand(L, b);
    rocke_ll_emitf(L,
                   "  %s = call i32 @llvm.amdgcn.perm(i32 %s, i32 %s, i32 %u)",
                   ll_result_name(op),
                   a_s,
                   b_s,
                   sel_u);
}

/* ====================================================================== */
/* ds_read_tr16_b64 / ds_read_tr16_b128 / ds_read_tr_b8                   */
/* ====================================================================== */

/* Shared body for the b16 transpose reads (b64 = <4 x i16>, b128 = <8 x i16>).
 * Mirrors the two near-identical Python handlers. */
static void ll_ds_read_tr16(rocke_lower_t* L,
                            const rocke_op_t* op,
                            const char* base_hint,
                            const char* raw_hint,
                            int vec,
                            const char* intrin_key,
                            const char* intrin_name)
{
    const rocke_value_t* smem = op->operands[0];
    const char* gname;
    const rocke_type_t* stype = NULL;
    const char* agg_ty;
    const char* base;
    const char* raw;
    const char* elem_ty;
    rocke_strbuf_t gep;
    int i;

    gname = rocke_ll_smem_global_name(L, smem, &stype);
    if(!rocke_ll_live(L))
    {
        return;
    }
    agg_ty = rocke_ll_smem_storage_type(L, stype);
    base = rocke_ll_fresh(L, base_hint);

    /* getelementptr inbounds <agg>, ptr addrspace(3) <g>, i32 0, i32 <idx>... */
    if(rocke_strbuf_init(&gep, 64) != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "ds_read_tr16: strbuf OOM");
    }
    rocke_strbuf_appendf(&gep,
                         "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, "
                         "i32 0",
                         base,
                         agg_ty,
                         gname);
    for(i = 1; i < op->num_operands; ++i)
    {
        rocke_strbuf_appendf(&gep, ", i32 %s", rocke_ll_operand(L, op->operands[i]));
    }
    if(gep.oom)
    {
        rocke_strbuf_free(&gep);
        rocke_ll_fail(L, ROCKE_ERR_OOM, "ds_read_tr16: strbuf OOM");
    }
    rocke_ll_emit(L, rocke_strbuf_cstr(&gep));
    rocke_strbuf_free(&gep);

    rocke_ll_need(L, intrin_key);
    raw = rocke_ll_fresh(L, raw_hint);
    rocke_ll_emitf(L,
                   "  %s = call <%d x i16> @llvm.amdgcn.%s(ptr addrspace(3) %s)",
                   raw,
                   vec,
                   intrin_name,
                   base);

    elem_ty = rocke_ll_llvm_type(L, op->results[0]->type->elem);
    rocke_ll_emitf(
        L, "  %s = bitcast <%d x i16> %s to <%d x %s>", ll_result_name(op), vec, raw, vec, elem_ty);
}

static void _op_tile_ds_read_tr16_b64(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_ds_read_tr16(L, op, "tr.base", "tr.raw", 4, "ds.read.tr16.b64", "ds.read.tr16.b64");
}

static void _op_tile_ds_read_tr16_b128(rocke_lower_t* L, const rocke_op_t* op)
{
    ll_ds_read_tr16(L, op, "trw.base", "trw.raw", 8, "ds.read.tr16.b128", "ds.read.tr16.b128");
}

static void _op_tile_ds_read_tr_b8(rocke_lower_t* L, const rocke_op_t* op)
{
    /* ds_read_b64_tr_b8 -- 8-bit transpose read via inline asm. Returns 64
     * bits as <2 x i32>, bitcast to <8 x i8>. */
    const rocke_value_t* smem = op->operands[0];
    const char* gname;
    const rocke_type_t* stype = NULL;
    const char* agg_ty;
    const char* base;
    const char* addr;
    const char* raw;
    rocke_strbuf_t gep;
    int i;

    gname = rocke_ll_smem_global_name(L, smem, &stype);
    if(!rocke_ll_live(L))
    {
        return;
    }
    agg_ty = rocke_ll_smem_storage_type(L, stype);
    base = rocke_ll_fresh(L, "tr8.base");

    if(rocke_strbuf_init(&gep, 64) != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "ds_read_tr_b8: strbuf OOM");
    }
    rocke_strbuf_appendf(&gep,
                         "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, "
                         "i32 0",
                         base,
                         agg_ty,
                         gname);
    for(i = 1; i < op->num_operands; ++i)
    {
        rocke_strbuf_appendf(&gep, ", i32 %s", rocke_ll_operand(L, op->operands[i]));
    }
    if(gep.oom)
    {
        rocke_strbuf_free(&gep);
        rocke_ll_fail(L, ROCKE_ERR_OOM, "ds_read_tr_b8: strbuf OOM");
    }
    rocke_ll_emit(L, rocke_strbuf_cstr(&gep));
    rocke_strbuf_free(&gep);

    addr = rocke_ll_fresh(L, "tr8.addr");
    raw = rocke_ll_fresh(L, "tr8.raw");
    rocke_ll_emitf(L, "  %s = ptrtoint ptr addrspace(3) %s to i32", addr, base);
    rocke_ll_emitf(L,
                   "  %s = call <2 x i32> asm sideeffect "
                   "\"ds_read_b64_tr_b8 $0, $1\", \"=v,v\"(i32 %s)",
                   raw,
                   addr);
    rocke_ll_emitf(L, "  %s = bitcast <2 x i32> %s to <8 x i8>", ll_result_name(op), raw);
}

/* ====================================================================== */
/* general + multi-output inline_asm                                      */
/* ====================================================================== */

static void _op_tile_inline_asm(rocke_lower_t* L, const rocke_op_t* op)
{
    /* General AMDGPU inline-asm lowering. Renders
     *   <ret> = call <ty> asm sideeffect "<tmpl>", "<cons>"(<typed operands>)
     * (void form when no result). `convergent` is accepted at the IR level but
     * NOT rendered as a flag (comgr rejects it on an asm call expr). */
    const char* raw_template;
    const char* template_esc;
    const char* constraints;
    bool sideeffect;
    const char* flag_str;
    const char* arglist;
    const char* asm_str;
    int i;

    raw_template = rocke_attr_get_str(&op->attrs, "template");
    if(raw_template == NULL)
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.inline_asm: missing 'template'");
    }
    constraints = rocke_attr_get_str(&op->attrs, "constraints");
    if(constraints == NULL)
    {
        rocke_ll_fail(L, ROCKE_ERR_KEY, "tile.inline_asm: missing 'constraints'");
    }
    template_esc = rocke_ll_escape_asm_string(L, raw_template);

    sideeffect = rocke_attr_get_bool(&op->attrs, "sideeffect", true);
    flag_str = sideeffect ? " sideeffect" : "";

    /* Build the typed operand list. The strbufs are RAII-guarded so a throw
     * from any nested emit/type helper unwinds without leaking their heap. */
    ll_strbuf_guard args(64);
    if(!args.ok)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "tile.inline_asm: strbuf OOM");
    }
    for(i = 0; i < op->num_operands; ++i)
    {
        if(i != 0)
        {
            rocke_strbuf_append(&args.sb, ", ");
        }
        rocke_strbuf_append(&args.sb, rocke_ll_operand_with_type(L, op->operands[i]));
    }
    if(args.sb.oom)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "tile.inline_asm: strbuf OOM");
    }
    arglist = rocke_strbuf_cstr(&args.sb);

    /* Build the asm expression: asm<flags> "<tmpl>", "<cons>"(<args>) */
    ll_strbuf_guard asm_expr(64);
    if(!asm_expr.ok)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "tile.inline_asm: strbuf OOM");
    }
    rocke_strbuf_appendf(
        &asm_expr.sb, "asm%s \"%s\", \"%s\"(%s)", flag_str, template_esc, constraints, arglist);
    if(asm_expr.sb.oom)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "tile.inline_asm: strbuf OOM");
    }
    asm_str = rocke_strbuf_cstr(&asm_expr.sb);

    if(op->num_results > 1)
    {
        /* Multi-output: LLVM returns a literal struct; unpack with
         * extractvalue. */
        const char* tmp;
        const char* st;
        ll_strbuf_guard struct_ty(32);
        if(!struct_ty.ok)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "tile.inline_asm: strbuf OOM");
        }
        rocke_strbuf_append(&struct_ty.sb, "{ ");
        for(i = 0; i < op->num_results; ++i)
        {
            if(i != 0)
            {
                rocke_strbuf_append(&struct_ty.sb, ", ");
            }
            rocke_strbuf_append(&struct_ty.sb, rocke_ll_llvm_type(L, op->results[i]->type));
        }
        rocke_strbuf_append(&struct_ty.sb, " }");
        if(struct_ty.sb.oom)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "tile.inline_asm: strbuf OOM");
        }
        st = rocke_strbuf_cstr(&struct_ty.sb);
        tmp = rocke_ll_fresh(L, "asmcl");
        rocke_ll_emitf(L, "  %s = call %s %s", tmp, st, asm_str);
        for(i = 0; i < op->num_results; ++i)
        {
            rocke_ll_emitf(L, "  %s = extractvalue %s %s, %d", op->results[i]->name, st, tmp, i);
        }
    }
    else if(op->num_results == 1)
    {
        const char* ret_ty = rocke_ll_llvm_type(L, op->results[0]->type);
        rocke_ll_emitf(L, "  %s = call %s %s", ll_result_name(op), ret_ty, asm_str);
    }
    else
    {
        rocke_ll_emitf(L, "  call void %s", asm_str);
    }
    /* args / asm_expr freed by their RAII guards on scope exit. */
}

/* ====================================================================== */
/* Registration hook                                                      */
/* ====================================================================== */

void rocke_ll_register_crosslane(void)
{
    rocke_ll_set_handler(ROCKE_OP_TILE_READFIRSTLANE, _op_tile_readfirstlane);
    rocke_ll_set_handler(ROCKE_OP_TILE_PIN_SGPR, _op_tile_pin_sgpr);
    rocke_ll_set_handler(ROCKE_OP_TILE_LANE_ID, _op_tile_lane_id);
    rocke_ll_set_handler(ROCKE_OP_TILE_WAVE_BALLOT, _op_tile_wave_ballot);
    rocke_ll_set_handler(ROCKE_OP_TILE_WAVE_ALL, _op_tile_wave_all);
    rocke_ll_set_handler(ROCKE_OP_TILE_WAVE_ANY, _op_tile_wave_any);
    rocke_ll_set_handler(ROCKE_OP_TILE_DS_BPERMUTE, _op_tile_ds_bpermute);
    rocke_ll_set_handler(ROCKE_OP_TILE_DS_BPERMUTE_B64, _op_tile_ds_bpermute_b64);
    rocke_ll_set_handler(ROCKE_OP_TILE_DS_SWIZZLE_XOR, _op_tile_ds_swizzle_xor);
    rocke_ll_set_handler(ROCKE_OP_TILE_MOV_DPP, _op_tile_mov_dpp);
    rocke_ll_set_handler(ROCKE_OP_TILE_PERMLANE32_SWAP, _op_tile_permlane32_swap);
    rocke_ll_set_handler(ROCKE_OP_TILE_PERM_B32, _op_tile_perm_b32);
    rocke_ll_set_handler(ROCKE_OP_TILE_PERMLANEX16, _op_tile_permlanex16);
    rocke_ll_set_handler(ROCKE_OP_TILE_BYTE_PERM, _op_tile_byte_perm);
    rocke_ll_set_handler(ROCKE_OP_TILE_DS_READ_TR16_B64, _op_tile_ds_read_tr16_b64);
    rocke_ll_set_handler(ROCKE_OP_TILE_DS_READ_TR16_B128, _op_tile_ds_read_tr16_b128);
    rocke_ll_set_handler(ROCKE_OP_TILE_DS_READ_TR_B8, _op_tile_ds_read_tr_b8);
    rocke_ll_set_handler(ROCKE_OP_TILE_INLINE_ASM, _op_tile_inline_asm);
}

} /* namespace ckc */
