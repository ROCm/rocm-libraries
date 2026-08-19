// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_hip_lower_hip_mma.c -- C99 port of rocke.core.lower_hip, BUCKET 3:
 *   mma / cross-lane (dpp/permute/bpermute/swizzle) / barriers & scheduling /
 *   vector elementwise / control-flow ops.
 *
 * Each `_op_*` Python method becomes a static `rocke_h_op_*` handler with the
 * (lw, op) signature from lower_hip_internal.h. Shared helpers
 * (rocke_h_emit / rocke_h_emitf / rocke_h_name / rocke_h_type_to_hip /
 * rocke_h_hip_scalar / rocke_h_require_wmma_arch / rocke_h_require_ds_read_tr /
 * rocke_h_lower_region / rocke_h_push_indent / rocke_h_pop_indent / rocke_attr_get_* /
 * rocke_vector_type) are DEFINED elsewhere (bucket 0); only called here.
 *
 * The simple handlers reproduce every _emit() format string byte-for-byte.
 * The complex matrix-engine handlers are fully ported too:
 *   - tile.mma re-dispatches into the concrete MFMA/WMMA atoms (rocke_h_op_tile_mma),
 *     and the register-permutation shim (register_p_from_qk_c) emits its 8-lane
 *     copy loop;
 *   - the IR side-table machinery is wired: the ds_read transpose loads
 *     (ds_read_tr16_b64/b128) resolve their LDS storage via rocke_h_smem_storage,
 *     and scf.yield walks the enclosing scf.for via h_find_enclosing_for.
 * The remaining ROCKE_ERR_NOTIMPL handlers here are NOT stubs but FAITHFUL parity
 * with Python's own NotImplementedError: tile.inline_asm and tile.ds_read_tr_b8
 * have no _op_ method in lower_hip.py (the HIP source backend deliberately does
 * not lower raw inline-asm payloads nor the 8-bit transpose read), so Python's
 * getattr-dispatch raises immediately and we reproduce that same rejection.
 * Every opcode is registered in the dispatch table; none is dropped.
 */
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_hip_internal.h"

#include <stdio.h> /* snprintf */
#include <stdlib.h> /* atoi     */

namespace ckc
{

/* Convenience: the single result Value of `op` (Python op.result). */
static const rocke_value_t* h_res(const rocke_op_t* op)
{
    return op->results[0];
}

/* Python idiom `n = t.count if isinstance(t, VectorType) else 1`. */
static int h_vcount(const rocke_type_t* t)
{
    return (t && t->kind == ROCKE_TYPE_VECTOR) ? t->count : 1;
}

/* Python idiom `t.elem.name if isinstance(t, VectorType) else t.name`, mapped
 * to the HIP scalar spelling via _HIP_TYPE (rocke_h_hip_scalar). */
static const char* h_elem_scalar(const rocke_type_t* t)
{
    const char* nm = (t && t->kind == ROCKE_TYPE_VECTOR) ? t->elem->name : t->name;
    return rocke_h_hip_scalar(nm);
}

/* ============================== mma ====================================== */

/* def _op_tile_mma(self, op):
 *     op_id = op.attrs["op_id"]
 *     legacy = Op(name=f"tile.{op_id}", operands=list(op.operands),
 *                 results=list(op.results),
 *                 attrs={k: v for k, v in op.attrs.items() if k != "op_id"},
 *                 loc=op.loc)
 *     self.lower_op(legacy)
 *
 * The HIP source path has no ISA backend: the neutral tile.mma op carries the
 * concrete atom name in attrs["op_id"], which we re-form into "tile.<op_id>"
 * and re-dispatch through the regular per-op handler (the mfma_* / wmma_* family
 * + the MX/fp4/fp6 scaled shims, all defined as concrete ROCKE_OP_TILE_* handlers
 * in this and other buckets). The concrete handlers read only op.operands /
 * op.result (and the WMMA gate keys off the op_id *string* it is passed, not
 * op.attrs), so a synthetic op aliasing the same operands/results/regions and
 * reusing the original attrs map reproduces the Python emission exactly. */
static rocke_status_t rocke_h_op_tile_mma(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const char* op_id;
    char dotted[160];
    rocke_opcode_t legacy_opcode;
    rocke_h_handler_fn fn;
    rocke_op_t legacy;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    op_id = rocke_attr_get_str(&op->attrs, "op_id");
    if(!op_id)
    {
        return rocke_h_fail(lw, ROCKE_ERR_KEY, "tile.mma: missing 'op_id' attr");
    }
    snprintf(dotted, sizeof(dotted), "tile.%s", op_id);
    legacy_opcode = rocke_opcode_from_name(dotted);
    fn = rocke_h_dispatch(legacy_opcode);
    if(!fn)
    {
        /* Python: lower_op(Op("tile.<op_id>", ...)) raises NotImplementedError
         * when no _op_tile_<op_id> handler exists. */
        return rocke_h_fail(lw, ROCKE_ERR_NOTIMPL, "no HIP lowering for op '%s'", dotted);
    }
    /* Build the synthetic "tile.<op_id>" op: same operands/results/regions, the
     * original attrs map (op_id is simply ignored by the concrete handlers). */
    legacy = *op;
    legacy.opcode = legacy_opcode;
    legacy.name = rocke_arena_strdup(&lw->b->arena, dotted);
    return fn(lw, &legacy);
}

/* def _op_tile_register_p_from_qk_c(self, op):
 *     (qk_c,) = op.operands
 *     target = op.attrs["target_dtype"]
 *     res_t = _type_to_hip(op.result.type)
 *     nice = _name(op.result)
 *     self._emit(f"{res_t} {nice};")
 *     for i in range(8):
 *         self._emit(f"{nice}[{i}] = ({target}){_name(qk_c)}[{i}];") */
static rocke_status_t rocke_h_op_tile_register_p_from_qk_c(rocke_h_lowerer_t* lw,
                                                           const rocke_op_t* op)
{
    const rocke_value_t* qk_c = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* target = rocke_attr_get_str(&op->attrs, "target_dtype");
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* qn = rocke_h_name(lw, qk_c);
    int i;
    if(!target)
    {
        return rocke_h_fail(lw, ROCKE_ERR_KEY, "register_p_from_qk_c: missing 'target_dtype' attr");
    }
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < 8; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = (%s)%s[%d];", nice, i, target, qn, i);
    }
    return lw->status;
}

/* tile.inline_asm has NO _op_tile_inline_asm method in lower_hip.py: the HIP
 * source backend does not lower raw inline-asm payloads. Python's lower_op
 * getattr-dispatch therefore returns None and raises
 *   NotImplementedError(f"no HIP lowering for op {op.name!r}")
 * Faithful parity: set the sticky ROCKE_ERR_NOTIMPL with the same message shape
 * the central dispatcher (rocke_h_lower_op) uses for an unhandled op. */
static rocke_status_t rocke_h_op_tile_inline_asm(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    return rocke_h_fail(lw,
                        ROCKE_ERR_NOTIMPL,
                        "no HIP lowering for op '%s'",
                        op->name ? op->name : rocke_opcode_name(op->opcode));
}

/* ============================== cross-lane =============================== */

/* def _op_tile_readfirstlane(self, op):
 *     (v,) = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = __builtin_amdgcn_readfirstlane({_name(v)});") */
static rocke_status_t rocke_h_op_tile_readfirstlane(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* ty = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw,
                  "%s %s = __builtin_amdgcn_readfirstlane(%s);",
                  ty,
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_tile_pin_sgpr(self, op):
 *     (v,) = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = {_name(v)};")
 *     self._emit(f'asm volatile("" : "+s"({_name(op.result)}));') */
static rocke_status_t rocke_h_op_tile_pin_sgpr(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* ty = rocke_h_type_to_hip(lw, r->type);
    const char* rn = rocke_h_name(lw, r);
    rocke_h_emitf(lw, "%s %s = %s;", ty, rn, rocke_h_name(lw, v));
    rocke_h_emitf(lw, "asm volatile(\"\" : \"+s\"(%s));", rn);
    return lw->status;
}

/* def _op_tile_lane_id(self, op):
 *     self._emit(f"int {_name(op.result)} = "
 *                f"__builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));") */
static rocke_status_t rocke_h_op_tile_lane_id(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "int %s = __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));",
                  rocke_h_name(lw, r));
    return lw->status;
}

/* def _op_tile_wave_ballot(self, op):
 *     (pred,) = op.operands
 *     self._emit(f"int64_t {_name(op.result)} = __ballot({_name(pred)});") */
static rocke_status_t rocke_h_op_tile_wave_ballot(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* p = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw, "int64_t %s = __ballot(%s);", rocke_h_name(lw, r), rocke_h_name(lw, p));
    return lw->status;
}

/* def _op_tile_wave_all(self, op):
 *     self._emit(f"int32_t {_name(op.result)} = __all({_name(pred)});") */
static rocke_status_t rocke_h_op_tile_wave_all(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* p = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw, "int32_t %s = __all(%s);", rocke_h_name(lw, r), rocke_h_name(lw, p));
    return lw->status;
}

/* def _op_tile_wave_any(self, op):
 *     self._emit(f"int32_t {_name(op.result)} = __any({_name(pred)});") */
static rocke_status_t rocke_h_op_tile_wave_any(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* p = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw, "int32_t %s = __any(%s);", rocke_h_name(lw, r), rocke_h_name(lw, p));
    return lw->status;
}

/* def _op_tile_ds_bpermute(self, op):
 *     addr, data = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"__builtin_amdgcn_ds_bpermute({_name(addr)}, {_name(data)});") */
static rocke_status_t rocke_h_op_tile_ds_bpermute(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* addr = op->operands[0];
    const rocke_value_t* data = op->operands[1];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "int %s = __builtin_amdgcn_ds_bpermute(%s, %s);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, addr),
                  rocke_h_name(lw, data));
    return lw->status;
}

/* def _op_tile_ds_bpermute_b64(self, op): two 32-bit ds_bpermute + recombine. */
static rocke_status_t rocke_h_op_tile_ds_bpermute_b64(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* addr = op->operands[0];
    const rocke_value_t* data = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, addr);
    const char* dn = rocke_h_name(lw, data);
    rocke_h_emitf(lw, "int %s_lo = (int)((uint64_t)%s & 0xffffffffu);", nice, dn);
    rocke_h_emitf(lw, "int %s_hi = (int)((uint64_t)%s >> 32);", nice, dn);
    rocke_h_emitf(lw, "int %s_plo = __builtin_amdgcn_ds_bpermute(%s, %s_lo);", nice, an, nice);
    rocke_h_emitf(lw, "int %s_phi = __builtin_amdgcn_ds_bpermute(%s, %s_hi);", nice, an, nice);
    rocke_h_emitf(
        lw, "int64_t %s = ((int64_t)(uint32_t)%s_phi << 32) | (uint32_t)%s_plo;", nice, nice, nice);
    return lw->status;
}

/* def _op_tile_ds_swizzle_xor(self, op):
 *     offset = (xor_mask << 10) | 0x1F */
static rocke_status_t rocke_h_op_tile_ds_swizzle_xor(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* data = op->operands[0];
    const rocke_value_t* r = h_res(op);
    int64_t xor_mask = 0;
    int offset;
    rocke_attr_get_int(&op->attrs, "xor_mask", &xor_mask);
    offset = (int)(((unsigned)xor_mask << 10) | 0x1Fu);
    rocke_h_emitf(lw,
                  "int %s = __builtin_amdgcn_ds_swizzle(%s, %d);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, data),
                  offset);
    return lw->status;
}

/* def _op_tile_mov_dpp(self, op): row_shr/row_shl -> dpp_ctrl, update_dpp. */
static rocke_status_t rocke_h_op_tile_mov_dpp(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* data = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* dn = rocke_h_name(lw, data);
    bool bound_ctrl = rocke_attr_get_bool(&op->attrs, "bound_ctrl", false);
    int64_t shift = 0;
    int dpp_ctrl;
    if(rocke_attr_get_int(&op->attrs, "row_shr", &shift))
    {
        dpp_ctrl = 0x110 | ((int)shift & 0xF);
    }
    else
    {
        rocke_attr_get_int(&op->attrs, "row_shl", &shift);
        dpp_ctrl = 0x100 | ((int)shift & 0xF);
    }
    rocke_h_emitf(lw,
                  "int %s = __builtin_amdgcn_update_dpp(%s, %s, %d, 15, 15, %d);",
                  rocke_h_name(lw, r),
                  dn,
                  dn,
                  dpp_ctrl,
                  bound_ctrl ? 1 : 0);
    return lw->status;
}

/* def _op_tile_permlane32_swap(self, op): inline-asm v_permlane32_swap_b32. */
static rocke_status_t rocke_h_op_tile_permlane32_swap(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* lo_in = op->operands[0];
    const rocke_value_t* hi_in = op->operands[1];
    const rocke_value_t* r0 = op->results[0];
    const rocke_value_t* r1 = op->results[1];
    const char* r0n = rocke_h_name(lw, r0);
    const char* r1n = rocke_h_name(lw, r1);
    rocke_h_emitf(lw, "int %s = %s;", r0n, rocke_h_name(lw, lo_in));
    rocke_h_emitf(lw, "int %s = %s;", r1n, rocke_h_name(lw, hi_in));
    rocke_h_emitf(lw,
                  "asm volatile(\"v_permlane32_swap_b32 %%0, %%1\" : "
                  "\"+v\"(%s), \"+v\"(%s));",
                  r0n,
                  r1n);
    return lw->status;
}

/* def _op_tile_permlanex16(self, op): lane^16 swap via __builtin_amdgcn_permlanex16. */
static rocke_status_t rocke_h_op_tile_permlanex16(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* vn = rocke_h_name(lw, v);
    rocke_h_emitf(lw,
                  "int %s = __builtin_amdgcn_permlanex16("
                  "%s, %s, 0x76543210u, 0xfedcba98u, false, true);",
                  rocke_h_name(lw, r),
                  vn,
                  vn);
    return lw->status;
}

/* def _op_tile_byte_perm(self, op):
 *     sel = int(op.attrs["sel"]) & 0xFFFFFFFF
 *     self._emit(f"int {_name(op.result)} = __builtin_amdgcn_perm("
 *                f"{_name(a)}, {_name(b)}, {sel}u);") */
static rocke_status_t rocke_h_op_tile_byte_perm(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    int64_t sel = 0;
    rocke_attr_get_int(&op->attrs, "sel", &sel);
    rocke_h_emitf(lw,
                  "int %s = __builtin_amdgcn_perm(%s, %s, %uu);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, a),
                  rocke_h_name(lw, b),
                  (unsigned)(sel & 0xFFFFFFFF));
    return lw->status;
}

/* def _op_tile_perm_b32(self, op):
 *     src0, src1, sel = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"__builtin_amdgcn_perm({_name(src0)}, {_name(src1)}, {_name(sel)});") */
static rocke_status_t rocke_h_op_tile_perm_b32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* src0 = op->operands[0];
    const rocke_value_t* src1 = op->operands[1];
    const rocke_value_t* sel = op->operands[2];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "int %s = __builtin_amdgcn_perm(%s, %s, %s);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, src0),
                  rocke_h_name(lw, src1),
                  rocke_h_name(lw, sel));
    return lw->status;
}

/* Join the names of `vals[0..n)` with "][" (Python "][".join(...)). Returns ""
 * for n==0. Mirrors mem_idx_join in the memory bucket. */
static const char* h_idx_join(rocke_h_lowerer_t* lw, rocke_value_t* const* vals, int n)
{
    rocke_strbuf_t sb;
    const char* out;
    int i;
    if(n <= 0)
    {
        return "";
    }
    rocke_strbuf_init(&sb, 0);
    for(i = 0; i < n; i++)
    {
        if(i > 0)
        {
            rocke_strbuf_append(&sb, "][");
        }
        rocke_strbuf_append(&sb, rocke_h_name(lw, vals[i]));
    }
    out = rocke_arena_strdup(&lw->b->arena, rocke_strbuf_cstr(&sb));
    rocke_strbuf_free(&sb);
    return out ? out : "";
}

/* Python idiom `{"f16": "f16x", "bf16": "bf16x"}.get(elem, "f16x")`. */
static const char* h_tr_vec_prefix(const char* elem)
{
    if(elem && !__builtin_strcmp(elem, "bf16"))
    {
        return "bf16x";
    }
    return "f16x";
}

/* Shared port of _op_tile_ds_read_tr16_b64 / _b128. The two variants differ
 * only in the raw type / shim name / lane count / memcpy byte count.
 *   self._require_ds_read_tr(op_id)
 *   smem = op.operands[0]; indices = op.operands[1:]
 *   storage = smem.op.attrs.get("_storage")  -- RuntimeError on miss
 *   idx_str = "][".join(_name(i) for i in indices)
 *   elem = op.attrs.get("elem_type", "f16")
 *   vec_prefix = {"f16":"f16x","bf16":"bf16x"}.get(elem, "f16x")
 *   nice = _name(op.result); raw_tmp = f"_trraw_{nice.lstrip('%')}"
 *   self._emit(f"{raw_ty} {raw_tmp} = {shim}("
 *              f"(const __attribute__((address_space(3))) void*)&{storage}[{idx_str}]);")
 *   self._emit(f"{vec_prefix}{lanes} {nice}; __builtin_memcpy(&{nice}, &{raw_tmp}, {bytes});")
 * (_name already strips the leading '%', so nice.lstrip('%') == nice.) */
static rocke_status_t h_ds_read_tr16(rocke_h_lowerer_t* lw,
                                     const rocke_op_t* op,
                                     const char* op_id,
                                     const char* raw_ty,
                                     const char* shim,
                                     int lanes,
                                     int bytes)
{
    const rocke_value_t* smem;
    const char *storage, *idx_str, *elem, *vec_prefix, *nice, *raw_tmp;
    char raw_buf[160];
    if(rocke_h_require_ds_read_tr(lw, op_id) != ROCKE_OK)
    {
        return lw->status;
    }
    smem = op->operands[0];
    storage = rocke_h_smem_storage(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "%s before smem_alloc was lowered", op_id);
    }
    idx_str = h_idx_join(lw, &op->operands[1], op->num_operands - 1);
    elem = rocke_attr_get_str(&op->attrs, "elem_type");
    if(!elem)
    {
        elem = "f16";
    }
    vec_prefix = h_tr_vec_prefix(elem);
    nice = rocke_h_name(lw, op->results[0]);
    snprintf(raw_buf, sizeof(raw_buf), "_trraw_%s", nice);
    raw_tmp = raw_buf;
    rocke_h_emitf(lw,
                  "%s %s = %s("
                  "(const __attribute__((address_space(3))) void*)&%s[%s]);",
                  raw_ty,
                  raw_tmp,
                  shim,
                  storage,
                  idx_str);
    rocke_h_emitf(lw,
                  "%s%d %s; __builtin_memcpy(&%s, &%s, %d);",
                  vec_prefix,
                  lanes,
                  nice,
                  nice,
                  raw_tmp,
                  bytes);
    return lw->status;
}

static rocke_status_t rocke_h_op_tile_ds_read_tr16_b64(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_ds_read_tr16(
        lw, op, "ds_read_tr16_b64", "i16x4_raw", "_llvm_amdgcn_ds_read_tr16_b64", 4, 8);
}

static rocke_status_t rocke_h_op_tile_ds_read_tr16_b128(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_ds_read_tr16(
        lw, op, "ds_read_tr16_b128", "i16x8_raw", "_llvm_amdgcn_ds_read_tr16_b128", 8, 16);
}

/* tile.ds_read_tr_b8 has NO _op_tile_ds_read_tr_b8 method in lower_hip.py: the
 * HIP source backend has no 8-bit transpose-read lowering (the IRBuilder exposes
 * ds_read_tr_b8 but only the LLVM path lowers it). Python's getattr-dispatch
 * returns None and raises NotImplementedError immediately -- there is NO arch
 * gate on this path (the gate runs only inside the b64/b128 methods that
 * actually exist). Faithful parity: fail straight to ROCKE_ERR_NOTIMPL with the
 * unhandled-op message shape; do not invoke rocke_h_require_ds_read_tr. */
static rocke_status_t rocke_h_op_tile_ds_read_tr_b8(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    return rocke_h_fail(lw,
                        ROCKE_ERR_NOTIMPL,
                        "no HIP lowering for op '%s'",
                        op->name ? op->name : rocke_opcode_name(op->opcode));
}

/* ============================== barriers / scheduling ==================== */

/* def _op_tile_sync(self, op): self._emit("__syncthreads();") */
static rocke_status_t rocke_h_op_tile_sync(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    (void)op;
    rocke_h_emit(lw, "__syncthreads();");
    return lw->status;
}

/* def _op_tile_sync_half_block(self, op):
 *     self._emit(f"if ({_name(sel)}) {{ __builtin_amdgcn_s_barrier(); }}") */
static rocke_status_t rocke_h_op_tile_sync_half_block(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* sel = op->operands[0];
    rocke_h_emitf(lw, "if (%s) { __builtin_amdgcn_s_barrier(); }", rocke_h_name(lw, sel));
    return lw->status;
}

/* def _op_tile_sync_lds_only(self, op):
 *     mask = self._encode_waitcnt(vmcnt=-1, expcnt=-1, lgkmcnt=0)
 *     self._emit(f"__builtin_amdgcn_s_waitcnt({mask});")
 *     self._emit("__syncthreads();") */
static rocke_status_t rocke_h_op_tile_sync_lds_only(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    int mask;
    (void)op;
    mask = rocke_h_encode_waitcnt(lw, -1, -1, 0);
    rocke_h_emitf(lw, "__builtin_amdgcn_s_waitcnt(%d);", mask);
    rocke_h_emit(lw, "__syncthreads();");
    return lw->status;
}

/* def _op_tile_s_barrier_bare(self, op):
 *     self._emit("__builtin_amdgcn_s_barrier();") */
static rocke_status_t rocke_h_op_tile_s_barrier_bare(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    (void)op;
    rocke_h_emit(lw, "__builtin_amdgcn_s_barrier();");
    return lw->status;
}

/* def _op_tile_s_waitcnt(self, op):
 *     vm/lk/ec from attrs (default -1); mask = _encode_waitcnt(vm, ec, lk). */
static rocke_status_t rocke_h_op_tile_s_waitcnt(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    int64_t vm = -1, lk = -1, ec = -1;
    int mask;
    rocke_attr_get_int(&op->attrs, "vmcnt", &vm);
    rocke_attr_get_int(&op->attrs, "lgkmcnt", &lk);
    rocke_attr_get_int(&op->attrs, "expcnt", &ec);
    mask = rocke_h_encode_waitcnt(lw, (int)vm, (int)ec, (int)lk);
    rocke_h_emitf(lw, "__builtin_amdgcn_s_waitcnt(%d);", mask);
    return lw->status;
}

/* def _op_tile_iglp_opt(self, op):
 *     self._emit(f"__builtin_amdgcn_iglp_opt({int(op.attrs.get('level', 0))});") */
static rocke_status_t rocke_h_op_tile_iglp_opt(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    int64_t level = 0;
    rocke_attr_get_int(&op->attrs, "level", &level);
    rocke_h_emitf(lw, "__builtin_amdgcn_iglp_opt(%d);", (int)level);
    return lw->status;
}

/* def _op_tile_sched_barrier(self, op):
 *     self._emit(f"__builtin_amdgcn_sched_barrier({int(op.attrs.get('mask', 0))});") */
static rocke_status_t rocke_h_op_tile_sched_barrier(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    int64_t mask = 0;
    rocke_attr_get_int(&op->attrs, "mask", &mask);
    rocke_h_emitf(lw, "__builtin_amdgcn_sched_barrier(%d);", (int)mask);
    return lw->status;
}

/* def _op_tile_sched_group_barrier(self, op):
 *     m=attrs["mask"], c=attrs["count"], g=attrs.get("group",0)
 *     self._emit(f"__builtin_amdgcn_sched_group_barrier({m}, {c}, {g});") */
static rocke_status_t rocke_h_op_tile_sched_group_barrier(rocke_h_lowerer_t* lw,
                                                          const rocke_op_t* op)
{
    int64_t m = 0, c = 0, g = 0;
    rocke_attr_get_int(&op->attrs, "mask", &m);
    rocke_attr_get_int(&op->attrs, "count", &c);
    rocke_attr_get_int(&op->attrs, "group", &g);
    rocke_h_emitf(lw, "__builtin_amdgcn_sched_group_barrier(%d, %d, %d);", (int)m, (int)c, (int)g);
    return lw->status;
}

/* def _op_tile_s_setprio(self, op):
 *     self._emit(f"__builtin_amdgcn_s_setprio({int(op.attrs['level'])});") */
static rocke_status_t rocke_h_op_tile_s_setprio(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    int64_t level = 0;
    rocke_attr_get_int(&op->attrs, "level", &level);
    rocke_h_emitf(lw, "__builtin_amdgcn_s_setprio(%d);", (int)level);
    return lw->status;
}

/* ============================== vector =================================== */

/* def _op_vector_bitcast(self, op): memcpy into the result type. */
static rocke_status_t rocke_h_op_vector_bitcast(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* tgt = rocke_h_type_to_hip(lw, r->type);
    const char* rn = rocke_h_name(lw, r);
    rocke_h_emitf(lw,
                  "%s %s; __builtin_memcpy(&%s, &%s, sizeof(%s));",
                  tgt,
                  rn,
                  rn,
                  rocke_h_name(lw, v),
                  tgt);
    return lw->status;
}

/* def _op_vector_extract(self, op):
 *     elem_t = v.type.elem if VectorType else v.type
 *     self._emit(f"{_HIP_TYPE[elem_t.name]} {_name(op.result)} = {_name(v)}[{i}];") */
static rocke_status_t rocke_h_op_vector_extract(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    int64_t i = 0;
    rocke_attr_get_int(&op->attrs, "index", &i);
    rocke_h_emitf(lw,
                  "%s %s = %s[%d];",
                  h_elem_scalar(v->type),
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, v),
                  (int)i);
    return lw->status;
}

/* def _op_vector_insert(self, op): res = v; res[i] = scalar. */
static rocke_status_t rocke_h_op_vector_insert(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* scalar = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    int64_t i = 0;
    rocke_attr_get_int(&op->attrs, "index", &i);
    rocke_h_emitf(lw, "%s %s = %s;", res_t, nice, rocke_h_name(lw, v));
    rocke_h_emitf(lw, "%s[%d] = %s;", nice, (int)i, rocke_h_name(lw, scalar));
    return lw->status;
}

/* def _op_vector_pack(self, op): res[i] = comp_i for each operand. */
static rocke_status_t rocke_h_op_vector_pack(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    int i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < op->num_operands; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = %s;", nice, i, rocke_h_name(lw, op->operands[i]));
    }
    return lw->status;
}

/* def _op_vector_concat(self, op): copy a then b into the result vector. */
static rocke_status_t rocke_h_op_vector_concat(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    int n_a = h_vcount(a->type), n_b = h_vcount(b->type), i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n_a; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = %s[%d];", nice, i, an, i);
    }
    for(i = 0; i < n_b; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = %s[%d];", nice, n_a + i, bn, i);
    }
    return lw->status;
}

/* def _op_vector_splat(self, op): res[i] = scalar for n lanes (n from attr). */
static rocke_status_t rocke_h_op_vector_splat(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* scalar = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* sn = rocke_h_name(lw, scalar);
    int64_t n = 1;
    int i;
    rocke_attr_get_int(&op->attrs, "vec", &n);
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < (int)n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = %s;", nice, i, sn);
    }
    return lw->status;
}

/* def _op_vector_select(self, op): per-lane ternary; mask may be scalar. */
static rocke_status_t rocke_h_op_vector_select(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* mask = op->operands[0];
    const rocke_value_t* lhs = op->operands[1];
    const rocke_value_t* rhs = op->operands[2];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* mn = rocke_h_name(lw, mask);
    const char* ln = rocke_h_name(lw, lhs);
    const char* rn = rocke_h_name(lw, rhs);
    int n = h_vcount(r->type), i;
    bool scalar_mask = (mask->type == NULL) || (mask->type->kind != ROCKE_TYPE_VECTOR);
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        if(scalar_mask)
        {
            rocke_h_emitf(lw, "%s[%d] = %s ? %s[%d] : %s[%d];", nice, i, mn, ln, i, rn, i);
        }
        else
        {
            rocke_h_emitf(lw, "%s[%d] = %s[%d] ? %s[%d] : %s[%d];", nice, i, mn, i, ln, i, rn, i);
        }
    }
    return lw->status;
}

/* def _op_vector_sum(self, op): scalar accumulate v[0] + ... + v[n-1]. */
static rocke_status_t rocke_h_op_vector_sum(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    int n = h_vcount(v->type), i;
    rocke_h_emitf(lw, "%s %s = %s[0];", h_elem_scalar(v->type), nice, vn);
    for(i = 1; i < n; i++)
    {
        rocke_h_emitf(lw, "%s = %s + %s[%d];", nice, nice, vn, i);
    }
    return lw->status;
}

/* def _op_vector_reduce_max(self, op): scalar running max over the lanes. */
static rocke_status_t rocke_h_op_vector_reduce_max(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    int n = h_vcount(v->type), i;
    rocke_h_emitf(lw, "%s %s = %s[0];", h_elem_scalar(v->type), nice, vn);
    for(i = 1; i < n; i++)
    {
        rocke_h_emitf(lw, "%s = (%s[%d] > %s) ? %s[%d] : %s;", nice, vn, i, nice, vn, i, nice);
    }
    return lw->status;
}

/* Shared per-lane binary op emitter: res[i] = a[i] <op> b[i]. */
static rocke_status_t h_vec_binop(rocke_h_lowerer_t* lw, const rocke_op_t* op, const char* o)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    int n = h_vcount(r->type), i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = %s[%d] %s %s[%d];", nice, i, an, i, o, bn, i);
    }
    return lw->status;
}

static rocke_status_t rocke_h_op_vector_add(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_binop(lw, op, "+");
}
static rocke_status_t rocke_h_op_vector_sub(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_binop(lw, op, "-");
}
static rocke_status_t rocke_h_op_vector_mul(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_binop(lw, op, "*");
}
static rocke_status_t rocke_h_op_vector_and(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_binop(lw, op, "&");
}
static rocke_status_t rocke_h_op_vector_or(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_binop(lw, op, "|");
}
static rocke_status_t rocke_h_op_vector_shl(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_binop(lw, op, "<<");
}

/* def _op_vector_lshr(self, op): res[i] = ((uint32_t)a[i]) >> b[i]. */
static rocke_status_t rocke_h_op_vector_lshr(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    int n = h_vcount(r->type), i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = ((uint32_t)%s[%d]) >> %s[%d];", nice, i, an, i, bn, i);
    }
    return lw->status;
}

/* Shared per-lane min/max emitter (op selects the comparator). */
static rocke_status_t h_vec_minmax(rocke_h_lowerer_t* lw, const rocke_op_t* op, const char* cmp)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    int n = h_vcount(r->type), i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw,
                      "%s[%d] = (%s[%d] %s %s[%d]) ? %s[%d] : %s[%d];",
                      nice,
                      i,
                      an,
                      i,
                      cmp,
                      bn,
                      i,
                      an,
                      i,
                      bn,
                      i);
    }
    return lw->status;
}

static rocke_status_t rocke_h_op_vector_smax(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_minmax(lw, op, ">");
}
static rocke_status_t rocke_h_op_vector_smin(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_minmax(lw, op, "<");
}
/* vector.max shares the smax ">" form in the Python lowerer. */
static rocke_status_t rocke_h_op_vector_max(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_minmax(lw, op, ">");
}

/* def _op_vector_cmp(self, op): per-lane comparison via pred attr. */
static rocke_status_t rocke_h_op_vector_cmp(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    const char* pred = rocke_attr_get_str(&op->attrs, "pred");
    const char* cop = "<";
    int n = h_vcount(r->type), i;
    if(!pred)
    {
        pred = "lt";
    }
    if(!__builtin_strcmp(pred, "lt"))
        cop = "<";
    else if(!__builtin_strcmp(pred, "le"))
        cop = "<=";
    else if(!__builtin_strcmp(pred, "gt"))
        cop = ">";
    else if(!__builtin_strcmp(pred, "ge"))
        cop = ">=";
    else if(!__builtin_strcmp(pred, "eq"))
        cop = "==";
    else if(!__builtin_strcmp(pred, "ne"))
        cop = "!=";
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = %s[%d] %s %s[%d];", nice, i, an, i, cop, bn, i);
    }
    return lw->status;
}

/* Shared per-lane element cast emitter: res[i] = (elem_cpp)v[i].
 * Used by vector.trunc and vector.sext (result.type.elem spelling). */
static rocke_status_t h_vec_elem_cast(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* elem_cpp = rocke_h_type_to_hip(lw, r->type->elem);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    int n = h_vcount(r->type), i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = (%s)%s[%d];", nice, i, elem_cpp, vn, i);
    }
    return lw->status;
}

static rocke_status_t rocke_h_op_vector_trunc(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_elem_cast(lw, op);
}
static rocke_status_t rocke_h_op_vector_sext(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    return h_vec_elem_cast(lw, op);
}

/* def _op_vector_fma(self, op): res[i] = fmaf((float)a[i],(float)b[i],(float)c[i]). */
static rocke_status_t rocke_h_op_vector_fma(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    const rocke_value_t* c = op->operands[2];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* an = rocke_h_name(lw, a);
    const char* bn = rocke_h_name(lw, b);
    const char* cn = rocke_h_name(lw, c);
    int n = h_vcount(r->type), i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw,
                      "%s[%d] = fmaf((float)%s[%d], (float)%s[%d], (float)%s[%d]);",
                      nice,
                      i,
                      an,
                      i,
                      bn,
                      i,
                      cn,
                      i);
    }
    return lw->status;
}

/* def _op_vector_trunc_f32_to_f16(self, op): per-lane (fp16) cast (legacy). */
static rocke_status_t rocke_h_op_vector_trunc_f32_to_f16(rocke_h_lowerer_t* lw,
                                                         const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    int n = h_vcount(v->type), i;
    rocke_h_emitf(lw, "f16x%d %s;", n, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = (fp16)%s[%d];", nice, i, vn, i);
    }
    return lw->status;
}

/* def _op_vector_trunc_f32_to(self, op): per-lane (target) cast; target attr. */
static rocke_status_t rocke_h_op_vector_trunc_f32_to(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    const char* target = rocke_attr_get_str(&op->attrs, "target");
    const char* elem_cpp;
    int n = h_vcount(v->type), i;
    if(!target)
    {
        target = "f16";
    }
    elem_cpp = rocke_h_hip_scalar(target);
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = (%s)%s[%d];", nice, i, elem_cpp, vn, i);
    }
    return lw->status;
}

/* ============================== control flow ============================ */

/* def _op_cf_return(self, op): self._emit("return;") */
static rocke_status_t rocke_h_op_cf_return(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    (void)op;
    rocke_h_emit(lw, "return;");
    return lw->status;
}

/* def _op_scf_if(self, op):
 *     self._emit(f"if({_name(cond)}) {{")
 *     push; lower_region(op.regions[0]); pop
 *     self._emit("}") */
static rocke_status_t rocke_h_op_scf_if(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* cond = op->operands[0];
    rocke_h_emitf(lw, "if(%s) {", rocke_h_name(lw, cond));
    rocke_h_push_indent(lw);
    rocke_h_lower_region(lw, op->regions[0]);
    rocke_h_pop_indent(lw);
    rocke_h_emit(lw, "}");
    return lw->status;
}

/* Python local `cpp_type_for(type_name)` inside _op_scf_for:
 *     if type_name.startswith("vec<f32x"): return f"f32x{int(inner)}"
 *     if type_name.startswith("vec<f16x"): return f"f16x{int(inner)}"
 *     return _HIP_TYPE.get(type_name, "auto")
 * The IR vector type name is "vec<<elem>x<n>>", so the inner is the n digits
 * between "vec<f32x"/"vec<f16x" and the trailing ">". Returns an arena string. */
static const char* h_for_cpp_type(rocke_h_lowerer_t* lw, const char* type_name)
{
    const char* hip;
    if(type_name && !__builtin_strncmp(type_name, "vec<f32x", 8))
    {
        int n = atoi(type_name + 8);
        return rocke_arena_printf(&lw->b->arena, "f32x%d", n);
    }
    if(type_name && !__builtin_strncmp(type_name, "vec<f16x", 8))
    {
        int n = atoi(type_name + 8);
        return rocke_arena_printf(&lw->b->arena, "f16x%d", n);
    }
    hip = type_name ? rocke_h_hip_scalar(type_name) : NULL;
    return hip ? hip : "auto";
}

/* def _op_scf_for(self, op):
 *     num_iter = op.attrs.get("num_iter_args", 0)
 *     lower, upper, step = op.operands[0:3]
 *     iter_inits = op.operands[3 : 3 + num_iter]
 *     iter_meta = op.attrs.get("iter_args", [])
 *     iv_name = op.attrs["iv"][1:]
 *     iv_ty = _HIP_TYPE[op.attrs["iv_type"]]
 *     for meta, result in zip(iter_meta, op.results):
 *         self._emit(f"{cpp_type_for(meta['type'])} {_name(result)};")
 *     self._emit("{"); push
 *     for meta, init in zip(iter_meta, iter_inits):
 *         self._emit(f"{cpp_type_for(meta['type'])} {meta['name'][1:]} = {_name(init)};")
 *     self._emit(f"for({iv_ty} {iv_name} = {_name(lower)}; {iv_name} < {_name(upper)};"
 *                f" {iv_name} += {_name(step)}) {{")
 *     push; lower_region(op.regions[0]); pop; self._emit("}")
 *     for meta, result in zip(iter_meta, op.results):
 *         self._emit(f"{_name(result)} = {meta['name'][1:]};")
 *     pop; self._emit("}")
 * (meta['name'] / iv carry a leading '%' that [1:] strips; the C builder stores
 *  the iter_args list as a ROCKE_ATTR_LIST of small {name,type} attr maps.) */
static rocke_status_t rocke_h_op_scf_for(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* lower = op->operands[0];
    const rocke_value_t* upper = op->operands[1];
    const rocke_value_t* step = op->operands[2];
    const rocke_attr_value_t* iter_meta;
    const char *iv_full, *iv_name, *iv_type, *iv_ty;
    const char *lo_n, *up_n, *st_n;
    int64_t num_iter = 0;
    int meta_n = 0, i;
    struct rocke_attr_map** items = NULL;

    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    rocke_attr_get_int(&op->attrs, "num_iter_args", &num_iter);

    iter_meta = rocke_attr_get(&op->attrs, "iter_args");
    if(iter_meta && iter_meta->kind == ROCKE_ATTR_LIST)
    {
        items = iter_meta->u.list.items;
        meta_n = iter_meta->u.list.count;
    }

    iv_full = rocke_attr_get_str(&op->attrs, "iv");
    if(!iv_full)
    {
        return rocke_h_fail(lw, ROCKE_ERR_KEY, "scf.for: missing 'iv' attr");
    }
    iv_name = (iv_full[0] == '%') ? iv_full + 1 : iv_full; /* [1:] */
    iv_type = rocke_attr_get_str(&op->attrs, "iv_type");
    iv_ty = iv_type ? rocke_h_hip_scalar(iv_type) : NULL;
    if(!iv_ty)
    {
        /* Python _HIP_TYPE[...] KeyError parity. */
        return rocke_h_fail(lw,
                            ROCKE_ERR_KEY,
                            "scf.for: no HIP type for iv_type '%s'",
                            iv_type ? iv_type : "<none>");
    }

    /* Declare for-op results in the enclosing scope (zip stops at min length). */
    for(i = 0; i < meta_n && i < op->num_results; i++)
    {
        const char* mtype = rocke_attr_get_str(items[i], "type");
        rocke_h_emitf(lw, "%s %s;", h_for_cpp_type(lw, mtype), rocke_h_name(lw, op->results[i]));
    }

    /* Inner C++ block: iter_args, the loop, and assignment back to results. */
    rocke_h_emit(lw, "{");
    rocke_h_push_indent(lw);
    /* zip(iter_meta, iter_inits): iter_inits = operands[3 : 3 + num_iter]. */
    for(i = 0; i < meta_n && i < (int)num_iter; i++)
    {
        const char* mtype = rocke_attr_get_str(items[i], "type");
        const char* mname = rocke_attr_get_str(items[i], "name");
        const char* mnm = (mname && mname[0] == '%') ? mname + 1 : (mname ? mname : "");
        rocke_h_emitf(lw,
                      "%s %s = %s;",
                      h_for_cpp_type(lw, mtype),
                      mnm,
                      rocke_h_name(lw, op->operands[3 + i]));
    }
    lo_n = rocke_h_name(lw, lower);
    up_n = rocke_h_name(lw, upper);
    st_n = rocke_h_name(lw, step);
    rocke_h_emitf(lw,
                  "for(%s %s = %s; %s < %s; %s += %s) {",
                  iv_ty,
                  iv_name,
                  lo_n,
                  iv_name,
                  up_n,
                  iv_name,
                  st_n);
    rocke_h_push_indent(lw);
    rocke_h_lower_region(lw, op->regions[0]);
    rocke_h_pop_indent(lw);
    rocke_h_emit(lw, "}");
    for(i = 0; i < meta_n && i < op->num_results; i++)
    {
        const char* mname = rocke_attr_get_str(items[i], "name");
        const char* mnm = (mname && mname[0] == '%') ? mname + 1 : (mname ? mname : "");
        rocke_h_emitf(lw, "%s = %s;", rocke_h_name(lw, op->results[i]), mnm);
    }
    rocke_h_pop_indent(lw);
    rocke_h_emit(lw, "}");
    return lw->status;
}

/* def _op_scf_yield(self, op):
 *     parent_for = _find_enclosing_for(self.kernel.body, op)
 *     if parent_for is None: raise RuntimeError("scf.yield without enclosing scf.for")
 *     meta = parent_for.attrs.get("iter_args", [])
 *     if len(op.operands) != len(meta):
 *         raise RuntimeError(f"scf.yield: {len(op.operands)} values vs {len(meta)} iter_args")
 *     for m, v in zip(meta, op.operands):
 *         self._emit(f"{m['name'][1:]} = {_name(v)};")
 *
 * def _find_enclosing_for(region, target):
 *     for op in region.ops:
 *         if op.name == "scf.for":
 *             for r in op.regions:
 *                 if target in r.ops: return op
 *         ... (recurse into nested regions) */
static const rocke_op_t* h_find_enclosing_for(const rocke_region_t* region,
                                              const rocke_op_t* target)
{
    int i, r, k;
    if(!region)
    {
        return NULL;
    }
    for(i = 0; i < region->num_ops; i++)
    {
        const rocke_op_t* op = region->ops[i];
        if(op->opcode == ROCKE_OP_SCF_FOR)
        {
            for(r = 0; r < op->num_regions; r++)
            {
                const rocke_region_t* reg = op->regions[r];
                const rocke_op_t* found;
                for(k = 0; k < reg->num_ops; k++)
                {
                    if(reg->ops[k] == target)
                    {
                        return op;
                    }
                }
                found = h_find_enclosing_for(reg, target);
                if(found)
                {
                    return found;
                }
            }
        }
        else
        {
            for(r = 0; r < op->num_regions; r++)
            {
                const rocke_op_t* found = h_find_enclosing_for(op->regions[r], target);
                if(found)
                {
                    return found;
                }
            }
        }
    }
    return NULL;
}

static rocke_status_t rocke_h_op_scf_yield(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_op_t* parent_for;
    const rocke_attr_value_t* meta;
    struct rocke_attr_map** items = NULL;
    int meta_n = 0, i;

    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    parent_for = h_find_enclosing_for(lw->kernel->body, op);
    if(!parent_for)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "scf.yield without enclosing scf.for");
    }
    meta = rocke_attr_get(&parent_for->attrs, "iter_args");
    if(meta && meta->kind == ROCKE_ATTR_LIST)
    {
        items = meta->u.list.items;
        meta_n = meta->u.list.count;
    }
    if(op->num_operands != meta_n)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "scf.yield: %d values vs %d iter_args", op->num_operands, meta_n);
    }
    for(i = 0; i < meta_n; i++)
    {
        const char* mname = rocke_attr_get_str(items[i], "name");
        const char* mnm = (mname && mname[0] == '%') ? mname + 1 : (mname ? mname : "");
        rocke_h_emitf(lw, "%s = %s;", mnm, rocke_h_name(lw, op->operands[i]));
    }
    return lw->status;
}

/* ============================== registration table ====================== */

const rocke_h_handler_entry_t* rocke_h_handlers_mma(void)
{
    static const rocke_h_handler_entry_t table[] = {
        /* mma */
        {ROCKE_OP_TILE_MMA, rocke_h_op_tile_mma},
        {ROCKE_OP_TILE_REGISTER_P_FROM_QK_C, rocke_h_op_tile_register_p_from_qk_c},
        {ROCKE_OP_TILE_INLINE_ASM, rocke_h_op_tile_inline_asm},
        /* cross-lane / dpp / permute */
        {ROCKE_OP_TILE_READFIRSTLANE, rocke_h_op_tile_readfirstlane},
        {ROCKE_OP_TILE_PIN_SGPR, rocke_h_op_tile_pin_sgpr},
        {ROCKE_OP_TILE_LANE_ID, rocke_h_op_tile_lane_id},
        {ROCKE_OP_TILE_WAVE_ALL, rocke_h_op_tile_wave_all},
        {ROCKE_OP_TILE_WAVE_ANY, rocke_h_op_tile_wave_any},
        {ROCKE_OP_TILE_WAVE_BALLOT, rocke_h_op_tile_wave_ballot},
        {ROCKE_OP_TILE_DS_BPERMUTE, rocke_h_op_tile_ds_bpermute},
        {ROCKE_OP_TILE_DS_BPERMUTE_B64, rocke_h_op_tile_ds_bpermute_b64},
        {ROCKE_OP_TILE_DS_SWIZZLE_XOR, rocke_h_op_tile_ds_swizzle_xor},
        {ROCKE_OP_TILE_MOV_DPP, rocke_h_op_tile_mov_dpp},
        {ROCKE_OP_TILE_PERMLANE32_SWAP, rocke_h_op_tile_permlane32_swap},
        {ROCKE_OP_TILE_PERM_B32, rocke_h_op_tile_perm_b32},
        {ROCKE_OP_TILE_PERMLANEX16, rocke_h_op_tile_permlanex16},
        {ROCKE_OP_TILE_BYTE_PERM, rocke_h_op_tile_byte_perm},
        {ROCKE_OP_TILE_DS_READ_TR16_B64, rocke_h_op_tile_ds_read_tr16_b64},
        {ROCKE_OP_TILE_DS_READ_TR16_B128, rocke_h_op_tile_ds_read_tr16_b128},
        {ROCKE_OP_TILE_DS_READ_TR_B8, rocke_h_op_tile_ds_read_tr_b8},
        /* barriers / scheduling */
        {ROCKE_OP_TILE_SYNC, rocke_h_op_tile_sync},
        {ROCKE_OP_TILE_SYNC_HALF_BLOCK, rocke_h_op_tile_sync_half_block},
        {ROCKE_OP_TILE_SYNC_LDS_ONLY, rocke_h_op_tile_sync_lds_only},
        {ROCKE_OP_TILE_S_BARRIER_BARE, rocke_h_op_tile_s_barrier_bare},
        {ROCKE_OP_TILE_S_WAITCNT, rocke_h_op_tile_s_waitcnt},
        {ROCKE_OP_TILE_S_SETPRIO, rocke_h_op_tile_s_setprio},
        {ROCKE_OP_TILE_IGLP_OPT, rocke_h_op_tile_iglp_opt},
        {ROCKE_OP_TILE_SCHED_BARRIER, rocke_h_op_tile_sched_barrier},
        {ROCKE_OP_TILE_SCHED_GROUP_BARRIER, rocke_h_op_tile_sched_group_barrier},
        /* vector */
        {ROCKE_OP_VECTOR_ADD, rocke_h_op_vector_add},
        {ROCKE_OP_VECTOR_SUB, rocke_h_op_vector_sub},
        {ROCKE_OP_VECTOR_MUL, rocke_h_op_vector_mul},
        {ROCKE_OP_VECTOR_AND, rocke_h_op_vector_and},
        {ROCKE_OP_VECTOR_OR, rocke_h_op_vector_or},
        {ROCKE_OP_VECTOR_SHL, rocke_h_op_vector_shl},
        {ROCKE_OP_VECTOR_LSHR, rocke_h_op_vector_lshr},
        {ROCKE_OP_VECTOR_SMAX, rocke_h_op_vector_smax},
        {ROCKE_OP_VECTOR_SMIN, rocke_h_op_vector_smin},
        {ROCKE_OP_VECTOR_MAX, rocke_h_op_vector_max},
        {ROCKE_OP_VECTOR_FMA, rocke_h_op_vector_fma},
        {ROCKE_OP_VECTOR_SUM, rocke_h_op_vector_sum},
        {ROCKE_OP_VECTOR_REDUCE_MAX, rocke_h_op_vector_reduce_max},
        {ROCKE_OP_VECTOR_SPLAT, rocke_h_op_vector_splat},
        {ROCKE_OP_VECTOR_SELECT, rocke_h_op_vector_select},
        {ROCKE_OP_VECTOR_CMP, rocke_h_op_vector_cmp},
        {ROCKE_OP_VECTOR_TRUNC, rocke_h_op_vector_trunc},
        {ROCKE_OP_VECTOR_SEXT, rocke_h_op_vector_sext},
        {ROCKE_OP_VECTOR_TRUNC_F32_TO_F16, rocke_h_op_vector_trunc_f32_to_f16},
        {ROCKE_OP_VECTOR_TRUNC_F32_TO, rocke_h_op_vector_trunc_f32_to},
        {ROCKE_OP_VECTOR_BITCAST, rocke_h_op_vector_bitcast},
        {ROCKE_OP_VECTOR_EXTRACT, rocke_h_op_vector_extract},
        {ROCKE_OP_VECTOR_INSERT, rocke_h_op_vector_insert},
        {ROCKE_OP_VECTOR_PACK, rocke_h_op_vector_pack},
        {ROCKE_OP_VECTOR_CONCAT, rocke_h_op_vector_concat},
        /* control flow */
        {ROCKE_OP_SCF_FOR, rocke_h_op_scf_for},
        {ROCKE_OP_SCF_IF, rocke_h_op_scf_if},
        {ROCKE_OP_SCF_YIELD, rocke_h_op_scf_yield},
        {ROCKE_OP_CF_RETURN, rocke_h_op_cf_return},
        {ROCKE_OP_INVALID, NULL}, /* terminator */
    };
    return table;
}

} /* namespace ckc */
