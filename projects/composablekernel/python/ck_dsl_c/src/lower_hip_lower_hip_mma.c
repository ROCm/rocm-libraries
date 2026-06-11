/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * lower_hip_lower_hip_mma.c -- C99 port of ck_dsl.core.lower_hip, BUCKET 3:
 *   mma / cross-lane (dpp/permute/bpermute/swizzle) / barriers & scheduling /
 *   vector elementwise / control-flow ops.
 *
 * Each `_op_*` Python method becomes a static `ckc_h_op_*` handler with the
 * (lw, op) signature from lower_hip_internal.h. Shared helpers
 * (ckc_h_emit / ckc_h_emitf / ckc_h_name / ckc_h_type_to_hip /
 * ckc_h_hip_scalar / ckc_h_require_wmma_arch / ckc_h_require_ds_read_tr /
 * ckc_h_lower_region / ckc_h_push_indent / ckc_h_pop_indent / ckc_attr_get_* /
 * ckc_vector_type) are DEFINED elsewhere (bucket 0); only called here.
 *
 * The simple handlers reproduce every _emit() format string byte-for-byte.
 * The complex matrix-engine handlers (tile.mma re-dispatch into the concrete
 * MFMA/WMMA atoms, register-permutation shim) and the few handlers that need
 * IR side-table machinery not yet wired in C (ds_read transpose loads,
 * scf.yield's enclosing-for walk, inline asm payloads) are REGISTERED with a
 * marked TODO stub so no opcode is dropped from the dispatch table.
 */
#include "ckc/ir.h"
#include "ckc/lower_hip.h"
#include "ckc/lower_hip_internal.h"

/* Convenience: the single result Value of `op` (Python op.result). */
static const ckc_value_t *h_res(const ckc_op_t *op) {
    return op->results[0];
}

/* Python idiom `n = t.count if isinstance(t, VectorType) else 1`. */
static int h_vcount(const ckc_type_t *t) {
    return (t && t->kind == CKC_TYPE_VECTOR) ? t->count : 1;
}

/* Python idiom `t.elem.name if isinstance(t, VectorType) else t.name`, mapped
 * to the HIP scalar spelling via _HIP_TYPE (ckc_h_hip_scalar). */
static const char *h_elem_scalar(const ckc_type_t *t) {
    const char *nm = (t && t->kind == CKC_TYPE_VECTOR) ? t->elem->name : t->name;
    return ckc_h_hip_scalar(nm);
}

/* Marked placeholder for a deferred-port handler: emits a comment line that
 * documents the op so the generated source still parses, then returns the
 * (still-OK) sticky status. */
static ckc_status_t h_stub(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    ckc_h_emitf(lw, "// TODO(port): faithful lowering of %s deferred",
                op->name ? op->name : "<op>");
    return lw->status;
}

/* ============================== mma ====================================== */

/* def _op_tile_mma(self, op): re-dispatches tile.<op_id> to the concrete MFMA/
 * WMMA handler. The concrete matrix-engine intrinsic emission (the mfma_* /
 * wmma_* family + the MX/fp4/fp6 scaled shims) is a large, arch-gated port that
 * is deferred. Registered as a stub so the opcode resolves. */
static ckc_status_t ckc_h_op_tile_mma(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    /* TODO(port): faithful MFMA/WMMA lowering deferred -- re-dispatch of the
     * concrete tile.<op_id> matrix atom (mfma_f32_*, wmma_*, scaled MX shims). */
    return h_stub(lw, op);
}

/* def _op_tile_register_p_from_qk_c(self, op): P13 register-permutation shim. */
static ckc_status_t ckc_h_op_tile_register_p_from_qk_c(ckc_h_lowerer_t *lw,
                                                       const ckc_op_t *op) {
    /* TODO(port): faithful register_p_from_qk_c lowering deferred. */
    return h_stub(lw, op);
}

/* def _op_tile_inline_asm(self, op): raw inline-asm payload passthrough. */
static ckc_status_t ckc_h_op_tile_inline_asm(ckc_h_lowerer_t *lw,
                                             const ckc_op_t *op) {
    /* TODO(port): faithful inline_asm lowering deferred. */
    return h_stub(lw, op);
}

/* ============================== cross-lane =============================== */

/* def _op_tile_readfirstlane(self, op):
 *     (v,) = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = __builtin_amdgcn_readfirstlane({_name(v)});") */
static ckc_status_t ckc_h_op_tile_readfirstlane(ckc_h_lowerer_t *lw,
                                                const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *ty = ckc_h_type_to_hip(lw, r->type);
    ckc_h_emitf(lw, "%s %s = __builtin_amdgcn_readfirstlane(%s);",
                ty, ckc_h_name(lw, r), ckc_h_name(lw, v));
    return lw->status;
}

/* def _op_tile_pin_sgpr(self, op):
 *     (v,) = op.operands
 *     ty = _type_to_hip(op.result.type)
 *     self._emit(f"{ty} {_name(op.result)} = {_name(v)};")
 *     self._emit(f'asm volatile("" : "+s"({_name(op.result)}));') */
static ckc_status_t ckc_h_op_tile_pin_sgpr(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *ty = ckc_h_type_to_hip(lw, r->type);
    const char *rn = ckc_h_name(lw, r);
    ckc_h_emitf(lw, "%s %s = %s;", ty, rn, ckc_h_name(lw, v));
    ckc_h_emitf(lw, "asm volatile(\"\" : \"+s\"(%s));", rn);
    return lw->status;
}

/* def _op_tile_lane_id(self, op):
 *     self._emit(f"int {_name(op.result)} = "
 *                f"__builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));") */
static ckc_status_t ckc_h_op_tile_lane_id(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *r = h_res(op);
    ckc_h_emitf(lw,
                "int %s = __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));",
                ckc_h_name(lw, r));
    return lw->status;
}

/* def _op_tile_wave_ballot(self, op):
 *     (pred,) = op.operands
 *     self._emit(f"int64_t {_name(op.result)} = __ballot({_name(pred)});") */
static ckc_status_t ckc_h_op_tile_wave_ballot(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *p = op->operands[0];
    const ckc_value_t *r = h_res(op);
    ckc_h_emitf(lw, "int64_t %s = __ballot(%s);", ckc_h_name(lw, r), ckc_h_name(lw, p));
    return lw->status;
}

/* def _op_tile_wave_all(self, op):
 *     self._emit(f"int32_t {_name(op.result)} = __all({_name(pred)});") */
static ckc_status_t ckc_h_op_tile_wave_all(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *p = op->operands[0];
    const ckc_value_t *r = h_res(op);
    ckc_h_emitf(lw, "int32_t %s = __all(%s);", ckc_h_name(lw, r), ckc_h_name(lw, p));
    return lw->status;
}

/* def _op_tile_wave_any(self, op):
 *     self._emit(f"int32_t {_name(op.result)} = __any({_name(pred)});") */
static ckc_status_t ckc_h_op_tile_wave_any(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *p = op->operands[0];
    const ckc_value_t *r = h_res(op);
    ckc_h_emitf(lw, "int32_t %s = __any(%s);", ckc_h_name(lw, r), ckc_h_name(lw, p));
    return lw->status;
}

/* def _op_tile_ds_bpermute(self, op):
 *     addr, data = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"__builtin_amdgcn_ds_bpermute({_name(addr)}, {_name(data)});") */
static ckc_status_t ckc_h_op_tile_ds_bpermute(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *addr = op->operands[0];
    const ckc_value_t *data = op->operands[1];
    const ckc_value_t *r = h_res(op);
    ckc_h_emitf(lw, "int %s = __builtin_amdgcn_ds_bpermute(%s, %s);",
                ckc_h_name(lw, r), ckc_h_name(lw, addr), ckc_h_name(lw, data));
    return lw->status;
}

/* def _op_tile_ds_bpermute_b64(self, op): two 32-bit ds_bpermute + recombine. */
static ckc_status_t ckc_h_op_tile_ds_bpermute_b64(ckc_h_lowerer_t *lw,
                                                  const ckc_op_t *op) {
    const ckc_value_t *addr = op->operands[0];
    const ckc_value_t *data = op->operands[1];
    const ckc_value_t *r = h_res(op);
    const char *nice = ckc_h_name(lw, r);
    const char *an = ckc_h_name(lw, addr);
    const char *dn = ckc_h_name(lw, data);
    ckc_h_emitf(lw, "int %s_lo = (int)((uint64_t)%s & 0xffffffffu);", nice, dn);
    ckc_h_emitf(lw, "int %s_hi = (int)((uint64_t)%s >> 32);", nice, dn);
    ckc_h_emitf(lw, "int %s_plo = __builtin_amdgcn_ds_bpermute(%s, %s_lo);",
                nice, an, nice);
    ckc_h_emitf(lw, "int %s_phi = __builtin_amdgcn_ds_bpermute(%s, %s_hi);",
                nice, an, nice);
    ckc_h_emitf(lw,
                "int64_t %s = ((int64_t)(uint32_t)%s_phi << 32) | (uint32_t)%s_plo;",
                nice, nice, nice);
    return lw->status;
}

/* def _op_tile_ds_swizzle_xor(self, op):
 *     offset = (xor_mask << 10) | 0x1F */
static ckc_status_t ckc_h_op_tile_ds_swizzle_xor(ckc_h_lowerer_t *lw,
                                                 const ckc_op_t *op) {
    const ckc_value_t *data = op->operands[0];
    const ckc_value_t *r = h_res(op);
    int64_t xor_mask = 0;
    int offset;
    ckc_attr_get_int(&op->attrs, "xor_mask", &xor_mask);
    offset = (int)(((unsigned)xor_mask << 10) | 0x1Fu);
    ckc_h_emitf(lw, "int %s = __builtin_amdgcn_ds_swizzle(%s, %d);",
                ckc_h_name(lw, r), ckc_h_name(lw, data), offset);
    return lw->status;
}

/* def _op_tile_mov_dpp(self, op): row_shr/row_shl -> dpp_ctrl, update_dpp. */
static ckc_status_t ckc_h_op_tile_mov_dpp(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *data = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *dn = ckc_h_name(lw, data);
    bool bound_ctrl = ckc_attr_get_bool(&op->attrs, "bound_ctrl", false);
    int64_t shift = 0;
    int dpp_ctrl;
    if (ckc_attr_get_int(&op->attrs, "row_shr", &shift)) {
        dpp_ctrl = 0x110 | ((int)shift & 0xF);
    } else {
        ckc_attr_get_int(&op->attrs, "row_shl", &shift);
        dpp_ctrl = 0x100 | ((int)shift & 0xF);
    }
    ckc_h_emitf(lw,
                "int %s = __builtin_amdgcn_update_dpp(%s, %s, %d, 15, 15, %d);",
                ckc_h_name(lw, r), dn, dn, dpp_ctrl, bound_ctrl ? 1 : 0);
    return lw->status;
}

/* def _op_tile_permlane32_swap(self, op): inline-asm v_permlane32_swap_b32. */
static ckc_status_t ckc_h_op_tile_permlane32_swap(ckc_h_lowerer_t *lw,
                                                  const ckc_op_t *op) {
    const ckc_value_t *lo_in = op->operands[0];
    const ckc_value_t *hi_in = op->operands[1];
    const ckc_value_t *r0 = op->results[0];
    const ckc_value_t *r1 = op->results[1];
    const char *r0n = ckc_h_name(lw, r0);
    const char *r1n = ckc_h_name(lw, r1);
    ckc_h_emitf(lw, "int %s = %s;", r0n, ckc_h_name(lw, lo_in));
    ckc_h_emitf(lw, "int %s = %s;", r1n, ckc_h_name(lw, hi_in));
    ckc_h_emitf(lw,
                "asm volatile(\"v_permlane32_swap_b32 %%0, %%1\" : "
                "\"+v\"(%s), \"+v\"(%s));",
                r0n, r1n);
    return lw->status;
}

/* def _op_tile_permlanex16(self, op): lane^16 swap via __builtin_amdgcn_permlanex16. */
static ckc_status_t ckc_h_op_tile_permlanex16(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *vn = ckc_h_name(lw, v);
    ckc_h_emitf(lw,
                "int %s = __builtin_amdgcn_permlanex16("
                "%s, %s, 0x76543210u, 0xfedcba98u, false, true);",
                ckc_h_name(lw, r), vn, vn);
    return lw->status;
}

/* def _op_tile_byte_perm(self, op):
 *     sel = int(op.attrs["sel"]) & 0xFFFFFFFF
 *     self._emit(f"int {_name(op.result)} = __builtin_amdgcn_perm("
 *                f"{_name(a)}, {_name(b)}, {sel}u);") */
static ckc_status_t ckc_h_op_tile_byte_perm(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *r = h_res(op);
    int64_t sel = 0;
    ckc_attr_get_int(&op->attrs, "sel", &sel);
    ckc_h_emitf(lw, "int %s = __builtin_amdgcn_perm(%s, %s, %uu);",
                ckc_h_name(lw, r), ckc_h_name(lw, a), ckc_h_name(lw, b),
                (unsigned)(sel & 0xFFFFFFFF));
    return lw->status;
}

/* def _op_tile_perm_b32(self, op):
 *     src0, src1, sel = op.operands
 *     self._emit(f"int {_name(op.result)} = "
 *                f"__builtin_amdgcn_perm({_name(src0)}, {_name(src1)}, {_name(sel)});") */
static ckc_status_t ckc_h_op_tile_perm_b32(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *src0 = op->operands[0];
    const ckc_value_t *src1 = op->operands[1];
    const ckc_value_t *sel = op->operands[2];
    const ckc_value_t *r = h_res(op);
    ckc_h_emitf(lw, "int %s = __builtin_amdgcn_perm(%s, %s, %s);",
                ckc_h_name(lw, r), ckc_h_name(lw, src0), ckc_h_name(lw, src1),
                ckc_h_name(lw, sel));
    return lw->status;
}

/* The transpose LDS reads need the smem _storage side table + the arch gate.
 * The arch gate is exported; the storage resolution is deferred. Keep the gate
 * (so an unsupported target still errors) then emit the TODO stub. */
static ckc_status_t ckc_h_op_tile_ds_read_tr16_b64(ckc_h_lowerer_t *lw,
                                                   const ckc_op_t *op) {
    if (ckc_h_require_ds_read_tr(lw, "ds_read_tr16_b64") != CKC_OK) {
        return lw->status;
    }
    return h_stub(lw, op);
}

static ckc_status_t ckc_h_op_tile_ds_read_tr16_b128(ckc_h_lowerer_t *lw,
                                                    const ckc_op_t *op) {
    if (ckc_h_require_ds_read_tr(lw, "ds_read_tr16_b128") != CKC_OK) {
        return lw->status;
    }
    return h_stub(lw, op);
}

static ckc_status_t ckc_h_op_tile_ds_read_tr_b8(ckc_h_lowerer_t *lw,
                                                const ckc_op_t *op) {
    if (ckc_h_require_ds_read_tr(lw, "ds_read_tr_b8") != CKC_OK) {
        return lw->status;
    }
    return h_stub(lw, op);
}

/* ============================== barriers / scheduling ==================== */

/* def _op_tile_sync(self, op): self._emit("__syncthreads();") */
static ckc_status_t ckc_h_op_tile_sync(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    (void)op;
    ckc_h_emit(lw, "__syncthreads();");
    return lw->status;
}

/* def _op_tile_sync_half_block(self, op):
 *     self._emit(f"if ({_name(sel)}) {{ __builtin_amdgcn_s_barrier(); }}") */
static ckc_status_t ckc_h_op_tile_sync_half_block(ckc_h_lowerer_t *lw,
                                                  const ckc_op_t *op) {
    const ckc_value_t *sel = op->operands[0];
    ckc_h_emitf(lw, "if (%s) { __builtin_amdgcn_s_barrier(); }", ckc_h_name(lw, sel));
    return lw->status;
}

/* def _op_tile_sync_lds_only(self, op):
 *     mask = self._encode_waitcnt(vmcnt=-1, expcnt=-1, lgkmcnt=0)
 *     self._emit(f"__builtin_amdgcn_s_waitcnt({mask});")
 *     self._emit("__syncthreads();") */
static ckc_status_t ckc_h_op_tile_sync_lds_only(ckc_h_lowerer_t *lw,
                                                const ckc_op_t *op) {
    int mask;
    (void)op;
    mask = ckc_h_encode_waitcnt(lw, -1, -1, 0);
    ckc_h_emitf(lw, "__builtin_amdgcn_s_waitcnt(%d);", mask);
    ckc_h_emit(lw, "__syncthreads();");
    return lw->status;
}

/* def _op_tile_s_barrier_bare(self, op):
 *     self._emit("__builtin_amdgcn_s_barrier();") */
static ckc_status_t ckc_h_op_tile_s_barrier_bare(ckc_h_lowerer_t *lw,
                                                 const ckc_op_t *op) {
    (void)op;
    ckc_h_emit(lw, "__builtin_amdgcn_s_barrier();");
    return lw->status;
}

/* def _op_tile_s_waitcnt(self, op):
 *     vm/lk/ec from attrs (default -1); mask = _encode_waitcnt(vm, ec, lk). */
static ckc_status_t ckc_h_op_tile_s_waitcnt(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    int64_t vm = -1, lk = -1, ec = -1;
    int mask;
    ckc_attr_get_int(&op->attrs, "vmcnt", &vm);
    ckc_attr_get_int(&op->attrs, "lgkmcnt", &lk);
    ckc_attr_get_int(&op->attrs, "expcnt", &ec);
    mask = ckc_h_encode_waitcnt(lw, (int)vm, (int)ec, (int)lk);
    ckc_h_emitf(lw, "__builtin_amdgcn_s_waitcnt(%d);", mask);
    return lw->status;
}

/* def _op_tile_iglp_opt(self, op):
 *     self._emit(f"__builtin_amdgcn_iglp_opt({int(op.attrs.get('level', 0))});") */
static ckc_status_t ckc_h_op_tile_iglp_opt(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    int64_t level = 0;
    ckc_attr_get_int(&op->attrs, "level", &level);
    ckc_h_emitf(lw, "__builtin_amdgcn_iglp_opt(%d);", (int)level);
    return lw->status;
}

/* def _op_tile_sched_barrier(self, op):
 *     self._emit(f"__builtin_amdgcn_sched_barrier({int(op.attrs.get('mask', 0))});") */
static ckc_status_t ckc_h_op_tile_sched_barrier(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    int64_t mask = 0;
    ckc_attr_get_int(&op->attrs, "mask", &mask);
    ckc_h_emitf(lw, "__builtin_amdgcn_sched_barrier(%d);", (int)mask);
    return lw->status;
}

/* def _op_tile_sched_group_barrier(self, op):
 *     m=attrs["mask"], c=attrs["count"], g=attrs.get("group",0)
 *     self._emit(f"__builtin_amdgcn_sched_group_barrier({m}, {c}, {g});") */
static ckc_status_t ckc_h_op_tile_sched_group_barrier(ckc_h_lowerer_t *lw,
                                                      const ckc_op_t *op) {
    int64_t m = 0, c = 0, g = 0;
    ckc_attr_get_int(&op->attrs, "mask", &m);
    ckc_attr_get_int(&op->attrs, "count", &c);
    ckc_attr_get_int(&op->attrs, "group", &g);
    ckc_h_emitf(lw, "__builtin_amdgcn_sched_group_barrier(%d, %d, %d);",
                (int)m, (int)c, (int)g);
    return lw->status;
}

/* def _op_tile_s_setprio(self, op):
 *     self._emit(f"__builtin_amdgcn_s_setprio({int(op.attrs['level'])});") */
static ckc_status_t ckc_h_op_tile_s_setprio(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    int64_t level = 0;
    ckc_attr_get_int(&op->attrs, "level", &level);
    ckc_h_emitf(lw, "__builtin_amdgcn_s_setprio(%d);", (int)level);
    return lw->status;
}

/* ============================== vector =================================== */

/* def _op_vector_bitcast(self, op): memcpy into the result type. */
static ckc_status_t ckc_h_op_vector_bitcast(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *tgt = ckc_h_type_to_hip(lw, r->type);
    const char *rn = ckc_h_name(lw, r);
    ckc_h_emitf(lw, "%s %s; __builtin_memcpy(&%s, &%s, sizeof(%s));",
                tgt, rn, rn, ckc_h_name(lw, v), tgt);
    return lw->status;
}

/* def _op_vector_extract(self, op):
 *     elem_t = v.type.elem if VectorType else v.type
 *     self._emit(f"{_HIP_TYPE[elem_t.name]} {_name(op.result)} = {_name(v)}[{i}];") */
static ckc_status_t ckc_h_op_vector_extract(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    int64_t i = 0;
    ckc_attr_get_int(&op->attrs, "index", &i);
    ckc_h_emitf(lw, "%s %s = %s[%d];", h_elem_scalar(v->type),
                ckc_h_name(lw, r), ckc_h_name(lw, v), (int)i);
    return lw->status;
}

/* def _op_vector_insert(self, op): res = v; res[i] = scalar. */
static ckc_status_t ckc_h_op_vector_insert(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *scalar = op->operands[1];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    int64_t i = 0;
    ckc_attr_get_int(&op->attrs, "index", &i);
    ckc_h_emitf(lw, "%s %s = %s;", res_t, nice, ckc_h_name(lw, v));
    ckc_h_emitf(lw, "%s[%d] = %s;", nice, (int)i, ckc_h_name(lw, scalar));
    return lw->status;
}

/* def _op_vector_pack(self, op): res[i] = comp_i for each operand. */
static ckc_status_t ckc_h_op_vector_pack(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    int i;
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < op->num_operands; i++) {
        ckc_h_emitf(lw, "%s[%d] = %s;", nice, i, ckc_h_name(lw, op->operands[i]));
    }
    return lw->status;
}

/* def _op_vector_concat(self, op): copy a then b into the result vector. */
static ckc_status_t ckc_h_op_vector_concat(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *an = ckc_h_name(lw, a);
    const char *bn = ckc_h_name(lw, b);
    int n_a = h_vcount(a->type), n_b = h_vcount(b->type), i;
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n_a; i++) {
        ckc_h_emitf(lw, "%s[%d] = %s[%d];", nice, i, an, i);
    }
    for (i = 0; i < n_b; i++) {
        ckc_h_emitf(lw, "%s[%d] = %s[%d];", nice, n_a + i, bn, i);
    }
    return lw->status;
}

/* def _op_vector_splat(self, op): res[i] = scalar for n lanes (n from attr). */
static ckc_status_t ckc_h_op_vector_splat(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *scalar = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *sn = ckc_h_name(lw, scalar);
    int64_t n = 1;
    int i;
    ckc_attr_get_int(&op->attrs, "vec", &n);
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < (int)n; i++) {
        ckc_h_emitf(lw, "%s[%d] = %s;", nice, i, sn);
    }
    return lw->status;
}

/* def _op_vector_select(self, op): per-lane ternary; mask may be scalar. */
static ckc_status_t ckc_h_op_vector_select(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *mask = op->operands[0];
    const ckc_value_t *lhs = op->operands[1];
    const ckc_value_t *rhs = op->operands[2];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *mn = ckc_h_name(lw, mask);
    const char *ln = ckc_h_name(lw, lhs);
    const char *rn = ckc_h_name(lw, rhs);
    int n = h_vcount(r->type), i;
    bool scalar_mask = (mask->type == NULL) || (mask->type->kind != CKC_TYPE_VECTOR);
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        if (scalar_mask) {
            ckc_h_emitf(lw, "%s[%d] = %s ? %s[%d] : %s[%d];", nice, i, mn, ln, i, rn, i);
        } else {
            ckc_h_emitf(lw, "%s[%d] = %s[%d] ? %s[%d] : %s[%d];",
                        nice, i, mn, i, ln, i, rn, i);
        }
    }
    return lw->status;
}

/* def _op_vector_sum(self, op): scalar accumulate v[0] + ... + v[n-1]. */
static ckc_status_t ckc_h_op_vector_sum(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *nice = ckc_h_name(lw, r);
    const char *vn = ckc_h_name(lw, v);
    int n = h_vcount(v->type), i;
    ckc_h_emitf(lw, "%s %s = %s[0];", h_elem_scalar(v->type), nice, vn);
    for (i = 1; i < n; i++) {
        ckc_h_emitf(lw, "%s = %s + %s[%d];", nice, nice, vn, i);
    }
    return lw->status;
}

/* def _op_vector_reduce_max(self, op): scalar running max over the lanes. */
static ckc_status_t ckc_h_op_vector_reduce_max(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *nice = ckc_h_name(lw, r);
    const char *vn = ckc_h_name(lw, v);
    int n = h_vcount(v->type), i;
    ckc_h_emitf(lw, "%s %s = %s[0];", h_elem_scalar(v->type), nice, vn);
    for (i = 1; i < n; i++) {
        ckc_h_emitf(lw, "%s = (%s[%d] > %s) ? %s[%d] : %s;", nice, vn, i, nice, vn, i, nice);
    }
    return lw->status;
}

/* Shared per-lane binary op emitter: res[i] = a[i] <op> b[i]. */
static ckc_status_t h_vec_binop(ckc_h_lowerer_t *lw, const ckc_op_t *op, const char *o) {
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *an = ckc_h_name(lw, a);
    const char *bn = ckc_h_name(lw, b);
    int n = h_vcount(r->type), i;
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = %s[%d] %s %s[%d];", nice, i, an, i, o, bn, i);
    }
    return lw->status;
}

static ckc_status_t ckc_h_op_vector_add(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_binop(lw, op, "+");
}
static ckc_status_t ckc_h_op_vector_sub(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_binop(lw, op, "-");
}
static ckc_status_t ckc_h_op_vector_mul(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_binop(lw, op, "*");
}
static ckc_status_t ckc_h_op_vector_and(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_binop(lw, op, "&");
}
static ckc_status_t ckc_h_op_vector_or(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_binop(lw, op, "|");
}
static ckc_status_t ckc_h_op_vector_shl(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_binop(lw, op, "<<");
}

/* def _op_vector_lshr(self, op): res[i] = ((uint32_t)a[i]) >> b[i]. */
static ckc_status_t ckc_h_op_vector_lshr(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *an = ckc_h_name(lw, a);
    const char *bn = ckc_h_name(lw, b);
    int n = h_vcount(r->type), i;
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = ((uint32_t)%s[%d]) >> %s[%d];", nice, i, an, i, bn, i);
    }
    return lw->status;
}

/* Shared per-lane min/max emitter (op selects the comparator). */
static ckc_status_t h_vec_minmax(ckc_h_lowerer_t *lw, const ckc_op_t *op, const char *cmp) {
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *an = ckc_h_name(lw, a);
    const char *bn = ckc_h_name(lw, b);
    int n = h_vcount(r->type), i;
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = (%s[%d] %s %s[%d]) ? %s[%d] : %s[%d];",
                    nice, i, an, i, cmp, bn, i, an, i, bn, i);
    }
    return lw->status;
}

static ckc_status_t ckc_h_op_vector_smax(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_minmax(lw, op, ">");
}
static ckc_status_t ckc_h_op_vector_smin(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_minmax(lw, op, "<");
}
/* vector.max shares the smax ">" form in the Python lowerer. */
static ckc_status_t ckc_h_op_vector_max(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_minmax(lw, op, ">");
}

/* def _op_vector_cmp(self, op): per-lane comparison via pred attr. */
static ckc_status_t ckc_h_op_vector_cmp(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *an = ckc_h_name(lw, a);
    const char *bn = ckc_h_name(lw, b);
    const char *pred = ckc_attr_get_str(&op->attrs, "pred");
    const char *cop = "<";
    int n = h_vcount(r->type), i;
    if (!pred) {
        pred = "lt";
    }
    if (!__builtin_strcmp(pred, "lt")) cop = "<";
    else if (!__builtin_strcmp(pred, "le")) cop = "<=";
    else if (!__builtin_strcmp(pred, "gt")) cop = ">";
    else if (!__builtin_strcmp(pred, "ge")) cop = ">=";
    else if (!__builtin_strcmp(pred, "eq")) cop = "==";
    else if (!__builtin_strcmp(pred, "ne")) cop = "!=";
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = %s[%d] %s %s[%d];", nice, i, an, i, cop, bn, i);
    }
    return lw->status;
}

/* Shared per-lane element cast emitter: res[i] = (elem_cpp)v[i].
 * Used by vector.trunc and vector.sext (result.type.elem spelling). */
static ckc_status_t h_vec_elem_cast(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *elem_cpp = ckc_h_type_to_hip(lw, r->type->elem);
    const char *nice = ckc_h_name(lw, r);
    const char *vn = ckc_h_name(lw, v);
    int n = h_vcount(r->type), i;
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = (%s)%s[%d];", nice, i, elem_cpp, vn, i);
    }
    return lw->status;
}

static ckc_status_t ckc_h_op_vector_trunc(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_elem_cast(lw, op);
}
static ckc_status_t ckc_h_op_vector_sext(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    return h_vec_elem_cast(lw, op);
}

/* def _op_vector_fma(self, op): res[i] = fmaf((float)a[i],(float)b[i],(float)c[i]). */
static ckc_status_t ckc_h_op_vector_fma(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *a = op->operands[0];
    const ckc_value_t *b = op->operands[1];
    const ckc_value_t *c = op->operands[2];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *an = ckc_h_name(lw, a);
    const char *bn = ckc_h_name(lw, b);
    const char *cn = ckc_h_name(lw, c);
    int n = h_vcount(r->type), i;
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = fmaf((float)%s[%d], (float)%s[%d], (float)%s[%d]);",
                    nice, i, an, i, bn, i, cn, i);
    }
    return lw->status;
}

/* def _op_vector_trunc_f32_to_f16(self, op): per-lane (fp16) cast (legacy). */
static ckc_status_t ckc_h_op_vector_trunc_f32_to_f16(ckc_h_lowerer_t *lw,
                                                     const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *nice = ckc_h_name(lw, r);
    const char *vn = ckc_h_name(lw, v);
    int n = h_vcount(v->type), i;
    ckc_h_emitf(lw, "f16x%d %s;", n, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = (fp16)%s[%d];", nice, i, vn, i);
    }
    return lw->status;
}

/* def _op_vector_trunc_f32_to(self, op): per-lane (target) cast; target attr. */
static ckc_status_t ckc_h_op_vector_trunc_f32_to(ckc_h_lowerer_t *lw,
                                                 const ckc_op_t *op) {
    const ckc_value_t *v = op->operands[0];
    const ckc_value_t *r = h_res(op);
    const char *res_t = ckc_h_type_to_hip(lw, r->type);
    const char *nice = ckc_h_name(lw, r);
    const char *vn = ckc_h_name(lw, v);
    const char *target = ckc_attr_get_str(&op->attrs, "target");
    const char *elem_cpp;
    int n = h_vcount(v->type), i;
    if (!target) {
        target = "f16";
    }
    elem_cpp = ckc_h_hip_scalar(target);
    ckc_h_emitf(lw, "%s %s;", res_t, nice);
    for (i = 0; i < n; i++) {
        ckc_h_emitf(lw, "%s[%d] = (%s)%s[%d];", nice, i, elem_cpp, vn, i);
    }
    return lw->status;
}

/* ============================== control flow ============================ */

/* def _op_cf_return(self, op): self._emit("return;") */
static ckc_status_t ckc_h_op_cf_return(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    (void)op;
    ckc_h_emit(lw, "return;");
    return lw->status;
}

/* def _op_scf_if(self, op):
 *     self._emit(f"if({_name(cond)}) {{")
 *     push; lower_region(op.regions[0]); pop
 *     self._emit("}") */
static ckc_status_t ckc_h_op_scf_if(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    const ckc_value_t *cond = op->operands[0];
    ckc_h_emitf(lw, "if(%s) {", ckc_h_name(lw, cond));
    ckc_h_push_indent(lw);
    ckc_h_lower_region(lw, op->regions[0]);
    ckc_h_pop_indent(lw);
    ckc_h_emit(lw, "}");
    return lw->status;
}

/* scf.for / scf.yield carry per-iter-arg metadata (a list of {name,type} dicts)
 * in op.attrs; that structured-attr shape and the enclosing-for walk used by
 * scf.yield are not yet modeled by the C attr map. Register both so the
 * dispatch table is complete; faithful body is deferred. */
static ckc_status_t ckc_h_op_scf_for(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    /* TODO(port): faithful scf.for lowering deferred (iter_args metadata +
     * nested region emission). */
    return h_stub(lw, op);
}

static ckc_status_t ckc_h_op_scf_yield(ckc_h_lowerer_t *lw, const ckc_op_t *op) {
    /* TODO(port): faithful scf.yield lowering deferred (enclosing-for walk). */
    return h_stub(lw, op);
}

/* ============================== registration table ====================== */

const ckc_h_handler_entry_t *ckc_h_handlers_mma(void) {
    static const ckc_h_handler_entry_t table[] = {
        /* mma */
        { CKC_OP_TILE_MMA,                  ckc_h_op_tile_mma },
        { CKC_OP_TILE_REGISTER_P_FROM_QK_C, ckc_h_op_tile_register_p_from_qk_c },
        { CKC_OP_TILE_INLINE_ASM,           ckc_h_op_tile_inline_asm },
        /* cross-lane / dpp / permute */
        { CKC_OP_TILE_READFIRSTLANE,        ckc_h_op_tile_readfirstlane },
        { CKC_OP_TILE_PIN_SGPR,             ckc_h_op_tile_pin_sgpr },
        { CKC_OP_TILE_LANE_ID,              ckc_h_op_tile_lane_id },
        { CKC_OP_TILE_WAVE_ALL,             ckc_h_op_tile_wave_all },
        { CKC_OP_TILE_WAVE_ANY,             ckc_h_op_tile_wave_any },
        { CKC_OP_TILE_WAVE_BALLOT,          ckc_h_op_tile_wave_ballot },
        { CKC_OP_TILE_DS_BPERMUTE,          ckc_h_op_tile_ds_bpermute },
        { CKC_OP_TILE_DS_BPERMUTE_B64,      ckc_h_op_tile_ds_bpermute_b64 },
        { CKC_OP_TILE_DS_SWIZZLE_XOR,       ckc_h_op_tile_ds_swizzle_xor },
        { CKC_OP_TILE_MOV_DPP,              ckc_h_op_tile_mov_dpp },
        { CKC_OP_TILE_PERMLANE32_SWAP,      ckc_h_op_tile_permlane32_swap },
        { CKC_OP_TILE_PERM_B32,             ckc_h_op_tile_perm_b32 },
        { CKC_OP_TILE_PERMLANEX16,          ckc_h_op_tile_permlanex16 },
        { CKC_OP_TILE_BYTE_PERM,            ckc_h_op_tile_byte_perm },
        { CKC_OP_TILE_DS_READ_TR16_B64,     ckc_h_op_tile_ds_read_tr16_b64 },
        { CKC_OP_TILE_DS_READ_TR16_B128,    ckc_h_op_tile_ds_read_tr16_b128 },
        { CKC_OP_TILE_DS_READ_TR_B8,        ckc_h_op_tile_ds_read_tr_b8 },
        /* barriers / scheduling */
        { CKC_OP_TILE_SYNC,                 ckc_h_op_tile_sync },
        { CKC_OP_TILE_SYNC_HALF_BLOCK,      ckc_h_op_tile_sync_half_block },
        { CKC_OP_TILE_SYNC_LDS_ONLY,        ckc_h_op_tile_sync_lds_only },
        { CKC_OP_TILE_S_BARRIER_BARE,       ckc_h_op_tile_s_barrier_bare },
        { CKC_OP_TILE_S_WAITCNT,            ckc_h_op_tile_s_waitcnt },
        { CKC_OP_TILE_S_SETPRIO,            ckc_h_op_tile_s_setprio },
        { CKC_OP_TILE_IGLP_OPT,             ckc_h_op_tile_iglp_opt },
        { CKC_OP_TILE_SCHED_BARRIER,        ckc_h_op_tile_sched_barrier },
        { CKC_OP_TILE_SCHED_GROUP_BARRIER,  ckc_h_op_tile_sched_group_barrier },
        /* vector */
        { CKC_OP_VECTOR_ADD,                ckc_h_op_vector_add },
        { CKC_OP_VECTOR_SUB,                ckc_h_op_vector_sub },
        { CKC_OP_VECTOR_MUL,                ckc_h_op_vector_mul },
        { CKC_OP_VECTOR_AND,                ckc_h_op_vector_and },
        { CKC_OP_VECTOR_OR,                 ckc_h_op_vector_or },
        { CKC_OP_VECTOR_SHL,                ckc_h_op_vector_shl },
        { CKC_OP_VECTOR_LSHR,               ckc_h_op_vector_lshr },
        { CKC_OP_VECTOR_SMAX,               ckc_h_op_vector_smax },
        { CKC_OP_VECTOR_SMIN,               ckc_h_op_vector_smin },
        { CKC_OP_VECTOR_MAX,                ckc_h_op_vector_max },
        { CKC_OP_VECTOR_FMA,                ckc_h_op_vector_fma },
        { CKC_OP_VECTOR_SUM,                ckc_h_op_vector_sum },
        { CKC_OP_VECTOR_REDUCE_MAX,         ckc_h_op_vector_reduce_max },
        { CKC_OP_VECTOR_SPLAT,              ckc_h_op_vector_splat },
        { CKC_OP_VECTOR_SELECT,             ckc_h_op_vector_select },
        { CKC_OP_VECTOR_CMP,                ckc_h_op_vector_cmp },
        { CKC_OP_VECTOR_TRUNC,              ckc_h_op_vector_trunc },
        { CKC_OP_VECTOR_SEXT,               ckc_h_op_vector_sext },
        { CKC_OP_VECTOR_TRUNC_F32_TO_F16,   ckc_h_op_vector_trunc_f32_to_f16 },
        { CKC_OP_VECTOR_TRUNC_F32_TO,       ckc_h_op_vector_trunc_f32_to },
        { CKC_OP_VECTOR_BITCAST,            ckc_h_op_vector_bitcast },
        { CKC_OP_VECTOR_EXTRACT,            ckc_h_op_vector_extract },
        { CKC_OP_VECTOR_INSERT,             ckc_h_op_vector_insert },
        { CKC_OP_VECTOR_PACK,               ckc_h_op_vector_pack },
        { CKC_OP_VECTOR_CONCAT,             ckc_h_op_vector_concat },
        /* control flow */
        { CKC_OP_SCF_FOR,                   ckc_h_op_scf_for },
        { CKC_OP_SCF_IF,                    ckc_h_op_scf_if },
        { CKC_OP_SCF_YIELD,                 ckc_h_op_scf_yield },
        { CKC_OP_CF_RETURN,                 ckc_h_op_cf_return },
        { CKC_OP_INVALID,                   NULL }, /* terminator */
    };
    return table;
}
