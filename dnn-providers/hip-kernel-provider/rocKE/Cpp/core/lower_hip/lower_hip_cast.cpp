// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_hip_lower_hip_cast.c -- C99 port of rocke.core.lower_hip, BUCKET 1:
 *   casts / conversions (int width casts, bitcast, rint, fp8/bf8/i8 packed and
 *   scalar conversions) + gpu thread/block id handlers.
 *
 * Each `_op_*` Python method becomes a static `rocke_h_op_*` handler with the
 * (lw, op) signature from lower_hip_internal.h. Shared helpers (rocke_h_emit /
 * rocke_h_emitf / rocke_h_name / rocke_h_type_to_hip / ...) are DEFINED in bucket 0
 * (lower_hip_core.c) and only called here. The registration table is exported
 * via rocke_h_handlers_cast(), which bucket 0 stitches into the dispatch table.
 *
 * Output text is byte-identical to the Python lowerer: every _emit() format
 * string is reproduced exactly (the Python self._emit adds the indent prefix;
 * here rocke_h_emit/rocke_h_emitf does the same).
 */
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_hip_internal.h"

namespace ckc
{

/* Convenience: the single result Value of `op` (Python op.result). Every handler
 * in this bucket produces exactly one result, mirroring the Python @property
 * which asserts a single result. */
static const rocke_value_t* h_res(const rocke_op_t* op)
{
    return op->results[0];
}

/* ----------------------------- arith: int width casts ---------------------- */

/* def _op_arith_zext(self, op):
 *     (v,) = op.operands
 *     self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
 *                f"({_type_to_hip(op.result.type)}){_name(v)};") */
static rocke_status_t rocke_h_op_arith_zext(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* t = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw, "%s %s = (%s)%s;", t, rocke_h_name(lw, r), t, rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_sext(self, op): -- identical lowering to zext in HIP. */
static rocke_status_t rocke_h_op_arith_sext(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* t = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw, "%s %s = (%s)%s;", t, rocke_h_name(lw, r), t, rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_trunc(self, op): -- C-style cast to the (narrower) target. */
static rocke_status_t rocke_h_op_arith_trunc(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* t = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw, "%s %s = (%s)%s;", t, rocke_h_name(lw, r), t, rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_trunc_f32_to_f16(self, op):
 *     (v,) = op.operands
 *     self._emit(f"fp16 {_name(op.result)} = (fp16){_name(v)};") */
static rocke_status_t rocke_h_op_arith_trunc_f32_to_f16(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw, "fp16 %s = (fp16)%s;", rocke_h_name(lw, r), rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_rint_f32(self, op):
 *     (v,) = op.operands
 *     self._emit(f"float {_name(op.result)} = rintf({_name(v)});") */
static rocke_status_t rocke_h_op_arith_rint_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw, "float %s = rintf(%s);", rocke_h_name(lw, r), rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_bitcast(self, op):
 *     (v,) = op.operands
 *     tgt = _type_to_hip(op.result.type)
 *     self._emit(f"{tgt} {_name(op.result)}; "
 *                f"__builtin_memcpy(&{_name(op.result)}, &{_name(v)}, sizeof({tgt}));") */
static rocke_status_t rocke_h_op_arith_bitcast(rocke_h_lowerer_t* lw, const rocke_op_t* op)
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

/* ----------------------------- arith: float casts -------------------------- */

/* def _op_arith_cast_to_f32(self, op):
 *     (v,) = op.operands
 *     self._emit(f"float {_name(op.result)} = (float){_name(v)};") */
static rocke_status_t rocke_h_op_arith_cast_to_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw, "float %s = (float)%s;", rocke_h_name(lw, r), rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_cast_f32_to(self, op):
 *     (v,) = op.operands
 *     cpp_t = _type_to_hip(op.result.type)
 *     self._emit(f"{cpp_t} {_name(op.result)} = ({cpp_t}){_name(v)};") */
static rocke_status_t rocke_h_op_arith_cast_f32_to(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* cpp_t = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw, "%s %s = (%s)%s;", cpp_t, rocke_h_name(lw, r), cpp_t, rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_sitofp_f32(self, op):
 *     (v,) = op.operands
 *     self._emit(f"float {_name(op.result)} = (float){_name(v)};") */
static rocke_status_t rocke_h_op_arith_sitofp_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw, "float %s = (float)%s;", rocke_h_name(lw, r), rocke_h_name(lw, v));
    return lw->status;
}

/* ----------------------------- arith: fp8/bf8 scalar cvt ------------------- */

/* def _op_arith_cvt_fp8_to_f32(self, op):
 *     (v,) = op.operands
 *     self._emit(f"float {_name(op.result)} = __builtin_amdgcn_cvt_f32_fp8("
 *                f"(unsigned int)(unsigned char){_name(v)}, 0);") */
static rocke_status_t rocke_h_op_arith_cvt_fp8_to_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "float %s = __builtin_amdgcn_cvt_f32_fp8("
                  "(unsigned int)(unsigned char)%s, 0);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_cvt_bf8_to_f32(self, op): -- e5m2 sibling. */
static rocke_status_t rocke_h_op_arith_cvt_bf8_to_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    rocke_h_emitf(lw,
                  "float %s = __builtin_amdgcn_cvt_f32_bf8("
                  "(unsigned int)(unsigned char)%s, 0);",
                  rocke_h_name(lw, r),
                  rocke_h_name(lw, v));
    return lw->status;
}

/* def _op_arith_cvt_f32_to_fp8(self, op):
 *     (v,) = op.operands
 *     tmp = f"{_name(op.result)}_pk"
 *     self._emit(f"unsigned int {tmp} = __builtin_amdgcn_cvt_pk_fp8_f32("
 *                f"{_name(v)}, 0.0f, 0u, false);")
 *     self._emit(f"fp8e4m3 {_name(op.result)} = (fp8e4m3)({tmp} & 0xffu);") */
static rocke_status_t rocke_h_op_arith_cvt_f32_to_fp8(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* rn = rocke_h_name(lw, r);
    rocke_h_emitf(lw,
                  "unsigned int %s_pk = __builtin_amdgcn_cvt_pk_fp8_f32("
                  "%s, 0.0f, 0u, false);",
                  rn,
                  rocke_h_name(lw, v));
    rocke_h_emitf(lw, "fp8e4m3 %s = (fp8e4m3)(%s_pk & 0xffu);", rn, rn);
    return lw->status;
}

/* def _op_arith_cvt_f32_to_bf8(self, op): -- e5m2 sibling. */
static rocke_status_t rocke_h_op_arith_cvt_f32_to_bf8(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* rn = rocke_h_name(lw, r);
    rocke_h_emitf(lw,
                  "unsigned int %s_pk = __builtin_amdgcn_cvt_pk_bf8_f32("
                  "%s, 0.0f, 0u, false);",
                  rn,
                  rocke_h_name(lw, v));
    rocke_h_emitf(lw, "bf8e5m2 %s = (bf8e5m2)(%s_pk & 0xffu);", rn, rn);
    return lw->status;
}

/* def _op_arith_cvt_f32_to_i8_sat(self, op):
 *     (v,) = op.operands
 *     rounded = f"{_name(op.result)}_r"
 *     as_i32 = f"{_name(op.result)}_i"
 *     self._emit(f"float {rounded} = rintf({_name(v)});")
 *     self._emit(f"int {as_i32} = (int){rounded};")
 *     self._emit(f"int8_t {_name(op.result)} = (int8_t)({as_i32} < -128 ? -128 : "
 *                f"({as_i32} > 127 ? 127 : {as_i32}));") */
static rocke_status_t rocke_h_op_arith_cvt_f32_to_i8_sat(rocke_h_lowerer_t* lw,
                                                         const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* rn = rocke_h_name(lw, r);
    rocke_h_emitf(lw, "float %s_r = rintf(%s);", rn, rocke_h_name(lw, v));
    rocke_h_emitf(lw, "int %s_i = (int)%s_r;", rn, rn);
    rocke_h_emitf(lw,
                  "int8_t %s = (int8_t)(%s_i < -128 ? -128 : "
                  "(%s_i > 127 ? 127 : %s_i));",
                  rn,
                  rn,
                  rn,
                  rn);
    return lw->status;
}

/* ----------------------------- arith: packed fp8/bf8 cvt ------------------- */

/* def _op_arith_cvt_pk_f32_fp8x4(self, op):
 *     packed/lo/hi temporaries, two __builtin_amdgcn_cvt_pk_f32_fp8 calls. */
static rocke_status_t rocke_h_op_arith_cvt_pk_f32_fp8x4(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const rocke_type_t* pair_ty = rocke_vector_type(lw->b, r->type->elem, 2);
    const char* pair_t = rocke_h_type_to_hip(lw, pair_ty);
    rocke_h_emitf(
        lw, "unsigned int %s_p; __builtin_memcpy(&%s_p, &%s, sizeof(%s_p));", nice, nice, vn, nice);
    rocke_h_emitf(
        lw, "%s %s_lo = __builtin_amdgcn_cvt_pk_f32_fp8(%s_p, false);", pair_t, nice, nice);
    rocke_h_emitf(
        lw, "%s %s_hi = __builtin_amdgcn_cvt_pk_f32_fp8(%s_p, true);", pair_t, nice, nice);
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    rocke_h_emitf(lw, "%s[0] = %s_lo.x;", nice, nice);
    rocke_h_emitf(lw, "%s[1] = %s_lo.y;", nice, nice);
    rocke_h_emitf(lw, "%s[2] = %s_hi.x;", nice, nice);
    rocke_h_emitf(lw, "%s[3] = %s_hi.y;", nice, nice);
    return lw->status;
}

/* def _op_arith_cvt_pk_f32_bf8x4(self, op): -- e5m2 sibling. */
static rocke_status_t rocke_h_op_arith_cvt_pk_f32_bf8x4(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    const rocke_type_t* pair_ty = rocke_vector_type(lw->b, r->type->elem, 2);
    const char* pair_t = rocke_h_type_to_hip(lw, pair_ty);
    rocke_h_emitf(
        lw, "unsigned int %s_p; __builtin_memcpy(&%s_p, &%s, sizeof(%s_p));", nice, nice, vn, nice);
    rocke_h_emitf(
        lw, "%s %s_lo = __builtin_amdgcn_cvt_pk_f32_bf8(%s_p, false);", pair_t, nice, nice);
    rocke_h_emitf(
        lw, "%s %s_hi = __builtin_amdgcn_cvt_pk_f32_bf8(%s_p, true);", pair_t, nice, nice);
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    rocke_h_emitf(lw, "%s[0] = %s_lo.x;", nice, nice);
    rocke_h_emitf(lw, "%s[1] = %s_lo.y;", nice, nice);
    rocke_h_emitf(lw, "%s[2] = %s_hi.x;", nice, nice);
    rocke_h_emitf(lw, "%s[3] = %s_hi.y;", nice, nice);
    return lw->status;
}

/* def _op_arith_cvt_pk_fp8_f32x4(self, op):
 *     two __builtin_amdgcn_cvt_pk_fp8_f32 calls (byte0/1 then byte2/3),
 *     memcpy the i32 into the result vector. */
static rocke_status_t rocke_h_op_arith_cvt_pk_fp8_f32x4(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw,
                  "unsigned int %s_lo = __builtin_amdgcn_cvt_pk_fp8_f32("
                  "%s[0], %s[1], 0u, false);",
                  nice,
                  vn,
                  vn);
    rocke_h_emitf(lw,
                  "unsigned int %s_p = __builtin_amdgcn_cvt_pk_fp8_f32("
                  "%s[2], %s[3], %s_lo, true);",
                  nice,
                  vn,
                  vn,
                  nice);
    rocke_h_emitf(
        lw, "%s %s; __builtin_memcpy(&%s, &%s_p, sizeof(%s));", res_t, nice, nice, nice, nice);
    return lw->status;
}

/* def _op_arith_cvt_pk_bf8_f32x4(self, op): -- e5m2 sibling. */
static rocke_status_t rocke_h_op_arith_cvt_pk_bf8_f32x4(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    rocke_h_emitf(lw,
                  "unsigned int %s_lo = __builtin_amdgcn_cvt_pk_bf8_f32("
                  "%s[0], %s[1], 0u, false);",
                  nice,
                  vn,
                  vn);
    rocke_h_emitf(lw,
                  "unsigned int %s_p = __builtin_amdgcn_cvt_pk_bf8_f32("
                  "%s[2], %s[3], %s_lo, true);",
                  nice,
                  vn,
                  vn,
                  nice);
    rocke_h_emitf(
        lw, "%s %s; __builtin_memcpy(&%s, &%s_p, sizeof(%s));", res_t, nice, nice, nice, nice);
    return lw->status;
}

/* def _op_arith_cvt_pk_i8_f32x4(self, op):
 *     per-element rintf + clamp + cast loop over 4 lanes. */
static rocke_status_t rocke_h_op_arith_cvt_pk_i8_f32x4(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* v = op->operands[0];
    const rocke_value_t* r = h_res(op);
    const char* nice = rocke_h_name(lw, r);
    const char* vn = rocke_h_name(lw, v);
    const char* res_t = rocke_h_type_to_hip(lw, r->type);
    int i;
    rocke_h_emitf(lw, "%s %s;", res_t, nice);
    for(i = 0; i < 4; i++)
    {
        rocke_h_emitf(lw, "float %s_r%d = rintf(%s[%d]);", nice, i, vn, i);
        rocke_h_emitf(lw, "int %s_i%d = (int)%s_r%d;", nice, i, nice, i);
        rocke_h_emitf(lw,
                      "%s_i%d = (%s_i%d < -128) ? -128 : ((%s_i%d > 127) ? 127 : %s_i%d);",
                      nice,
                      i,
                      nice,
                      i,
                      nice,
                      i,
                      nice,
                      i);
        rocke_h_emitf(lw, "%s[%d] = (int8_t)%s_i%d;", nice, i, nice, i);
    }
    return lw->status;
}

/* ----------------------------- gpu ids ------------------------------------- */

/* def _op_gpu_thread_id(self, op):
 *     axis = op.attrs.get("axis", "x")
 *     self._emit(f"int {_name(op.result)} = (int)threadIdx.{axis};") */
static rocke_status_t rocke_h_op_gpu_thread_id(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* r = h_res(op);
    const char* axis = rocke_attr_get_str(&op->attrs, "axis");
    if(!axis)
    {
        axis = "x";
    }
    rocke_h_emitf(lw, "int %s = (int)threadIdx.%s;", rocke_h_name(lw, r), axis);
    return lw->status;
}

/* def _op_gpu_block_id(self, op):
 *     axis = op.attrs.get("axis", "x")
 *     self._emit(f"int {_name(op.result)} = (int)blockIdx.{axis};") */
static rocke_status_t rocke_h_op_gpu_block_id(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* r = h_res(op);
    const char* axis = rocke_attr_get_str(&op->attrs, "axis");
    if(!axis)
    {
        axis = "x";
    }
    rocke_h_emitf(lw, "int %s = (int)blockIdx.%s;", rocke_h_name(lw, r), axis);
    return lw->status;
}

/* ----------------------------- registration table -------------------------- */

const rocke_h_handler_entry_t* rocke_h_handlers_cast(void)
{
    static const rocke_h_handler_entry_t table[] = {
        {ROCKE_OP_ARITH_ZEXT, rocke_h_op_arith_zext},
        {ROCKE_OP_ARITH_SEXT, rocke_h_op_arith_sext},
        {ROCKE_OP_ARITH_TRUNC, rocke_h_op_arith_trunc},
        {ROCKE_OP_ARITH_TRUNC_F32_TO_F16, rocke_h_op_arith_trunc_f32_to_f16},
        {ROCKE_OP_ARITH_RINT_F32, rocke_h_op_arith_rint_f32},
        {ROCKE_OP_ARITH_BITCAST, rocke_h_op_arith_bitcast},
        {ROCKE_OP_ARITH_CAST_TO_F32, rocke_h_op_arith_cast_to_f32},
        {ROCKE_OP_ARITH_CAST_F32_TO, rocke_h_op_arith_cast_f32_to},
        {ROCKE_OP_ARITH_SITOFP_F32, rocke_h_op_arith_sitofp_f32},
        {ROCKE_OP_ARITH_CVT_FP8_TO_F32, rocke_h_op_arith_cvt_fp8_to_f32},
        {ROCKE_OP_ARITH_CVT_BF8_TO_F32, rocke_h_op_arith_cvt_bf8_to_f32},
        {ROCKE_OP_ARITH_CVT_F32_TO_FP8, rocke_h_op_arith_cvt_f32_to_fp8},
        {ROCKE_OP_ARITH_CVT_F32_TO_BF8, rocke_h_op_arith_cvt_f32_to_bf8},
        {ROCKE_OP_ARITH_CVT_F32_TO_I8_SAT, rocke_h_op_arith_cvt_f32_to_i8_sat},
        {ROCKE_OP_ARITH_CVT_PK_F32_FP8X4, rocke_h_op_arith_cvt_pk_f32_fp8x4},
        {ROCKE_OP_ARITH_CVT_PK_F32_BF8X4, rocke_h_op_arith_cvt_pk_f32_bf8x4},
        {ROCKE_OP_ARITH_CVT_PK_FP8_F32X4, rocke_h_op_arith_cvt_pk_fp8_f32x4},
        {ROCKE_OP_ARITH_CVT_PK_BF8_F32X4, rocke_h_op_arith_cvt_pk_bf8_f32x4},
        {ROCKE_OP_ARITH_CVT_PK_I8_F32X4, rocke_h_op_arith_cvt_pk_i8_f32x4},
        {ROCKE_OP_GPU_THREAD_ID, rocke_h_op_gpu_thread_id},
        {ROCKE_OP_GPU_BLOCK_ID, rocke_h_op_gpu_block_id},
        {ROCKE_OP_INVALID, NULL}, /* terminator */
    };
    return table;
}

} /* namespace ckc */
