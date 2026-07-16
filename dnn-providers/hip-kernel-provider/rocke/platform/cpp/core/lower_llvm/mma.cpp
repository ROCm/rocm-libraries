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
 *   - scaled f8f6f4, fp4, fp6, and the unscaled fp8-128 hero atom
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
#include <stdio.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
#include "rocke/lower_llvm_internal.h"

namespace ckc
{

/* ------------------------------------------------------------ forward decls */

/* per-op handlers (file-static; reached via the op_id router, not the table) */
static void _op_tile_wmma_f32_16x16x16_f16(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_wmma_f32_16x16x16_bf16(rocke_lower_t* L, const rocke_op_t* op);
static void _emit_wmma(rocke_lower_t* L, const rocke_op_t* op, const char* op_id);
static void _op_tile_mma(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x32_fp8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x32_bf8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_32x32x16_fp8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_32x32x16_bf8(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_scale_f32_16x16x128_f8f6f4(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x128_fp4(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x96_fp6(rocke_lower_t* L, const rocke_op_t* op);
static void _op_tile_mfma_f32_16x16x128_fp8(rocke_lower_t* L, const rocke_op_t* op);
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

    /* f16 / bf16 / f32 dense atoms resolve from the table; the scaled / fp4 /
     * fp6 / fp8-bf8 / hero atoms keep their dedicated bodies below. */
    if(_try_emit_mfma_table(L, op, op_id))
    {
        return;
    }

    /* fp8 / bf8 */
    if(strcmp(op_id, "mfma_f32_16x16x32_fp8") == 0)
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
        /* MX-scaled / fp4 / fp6 / unscaled hero */
    }
    else if(strcmp(op_id, "mfma_scale_f32_16x16x128_f8f6f4") == 0)
    {
        _op_tile_mfma_scale_f32_16x16x128_f8f6f4(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_16x16x128_fp4") == 0)
    {
        _op_tile_mfma_f32_16x16x128_fp4(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_16x16x96_fp6") == 0)
    {
        _op_tile_mfma_f32_16x16x96_fp6(L, op);
    }
    else if(strcmp(op_id, "mfma_f32_16x16x128_fp8") == 0)
    {
        _op_tile_mfma_f32_16x16x128_fp8(L, op);
        /* WMMA op_ids. On an RDNA backend (gfx11/gfx12) these emit a real WMMA
         * call (Python Gfx11/Gfx12RdnaBackend.emit_wmma); on a CDNA backend they
         * reject (Python ISABackend.emit_wmma raises NotImplementedError). The
         * gfx12-specific op_ids ("wmma_gfx12_*") can only occur on RDNA4. */
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
typedef struct _wmma_spec
{
    const char* op_id; /* the tile.mma op_id (no "tile." prefix)        */
    const char* decl_key; /* _need() key                                   */
    const char* intrinsic; /* fully-mangled @llvm.amdgcn.wmma....           */
    const char* ssa_elt; /* SSA operand element type                      */
    const char* call_elt; /* call-site operand element type                */
    int frag_width; /* A/B operand vector width (16 RDNA3/3.5, 8 RDNA4) */
} _wmma_spec_t;

static const _wmma_spec_t WMMA_SPECS[] = {
    /* _RDNA_WMMA (RDNA3/3.5, frag_width 16) */
    {"wmma_f32_16x16x16_f16",
     "wmma.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16",
     "half",
     "half",
     16},
    {"wmma_f32_16x16x16_bf16",
     "wmma.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16",
     "bfloat",
     "i16",
     16},
    /* _RDNA_GFX12_WMMA (RDNA4, frag_width 8) */
    {"wmma_gfx12_f32_16x16x16_f16",
     "wmma.gfx12.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16",
     "half",
     "half",
     8},
    {"wmma_gfx12_f32_16x16x16_bf16",
     "wmma.gfx12.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16",
     "bfloat",
     "i16",
     8},
};
static const int WMMA_SPECS_N = (int)(sizeof(WMMA_SPECS) / sizeof(WMMA_SPECS[0]));

/* Integer WMMA spec table (Python _RDNA_WMMA_INT). Integer WMMA differs from
 * the float path: operands/accumulator are i32 vectors (A/B packed, C/D the i32
 * accumulator), and the intrinsic signature carries i1 signedness flags before
 * each matrix operand and a trailing i1 clamp. Operands arrive in SSA already
 * as <N x i32> so no bitcast is needed. Quantized data is signed and within i32
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
 * is 0 (values stay within i32 range). No operand bitcast (already <N x i32>). */
static void _emit_wmma_int(rocke_lower_t* L, const rocke_op_t* op, const _wmma_int_spec_t* spec)
{
    const rocke_value_t *a, *b, *c;
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    rocke_ll_need(L, spec->decl_key);
    rocke_ll_emitf(L,
                   "  %s = call <%d x i32> @%s("
                   "i1 1, <%d x i32> %s, "
                   "i1 1, <%d x i32> %s, "
                   "<%d x i32> %s, i1 0)",
                   mma_result_name(L, op),
                   spec->acc_vec,
                   spec->intrinsic,
                   spec->op_vec,
                   rocke_ll_operand(L, a),
                   spec->op_vec,
                   rocke_ll_operand(L, b),
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
        /* NAMED GAP (not a port stub): the WMMA_SPECS / WMMA_INT_SPECS tables
         * above cover exactly the Python _RDNA_WMMA (gfx11), _RDNA_GFX12_WMMA
         * (gfx12) and _RDNA_WMMA_INT op_ids -- i.e. the full set reachable on
         * the only RDNA backends rocke_ll_backend_for resolves (gfx1151 / gfx1201
         * / gfx11-generic). The Python isa backend additionally has the
         * _GFX1250_WMMA / _GFX1250_WMMA_FP8 families (16x16x32 f16/bf16,
         * 16x16x64 fp8/bf8), but there is NO gfx1250 entry in
         * rocke_ll_backend_for, so a gfx1250 build is rejected up front with
         * ROCKE_ERR_KEY ("unknown arch backend") and those op_ids never reach
         * here. Wiring them is blocked on porting the gfx1250 ISA backend
         * (split wait-counters + 57-bit SRD word3) into the C lowerer's
         * backend table first; until then this is an unreachable-but-faithful
         * rejection for an unsupported RDNA WMMA op_id. */
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
    a_arg = rocke_ll_operand(L, a);
    b_arg = rocke_ll_operand(L, b);

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
/* f16 / bf16 / f32 dense MFMA atoms (table-driven)                       */
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
} _mfma_spec_t;

static const _mfma_spec_t MFMA_SPECS[] = {
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
    /* fp32 (TF32-class) scalar atoms: A/B are single floats per lane. */
    {"mfma_f32_16x16x4_f32",
     "mfma.f32.16x16x4f32",
     "llvm.amdgcn.mfma.f32.16x16x4f32",
     "float",
     "<4 x float>",
     NULL},
    {"mfma_f32_32x32x2_f32",
     "mfma.f32.32x32x2f32",
     "llvm.amdgcn.mfma.f32.32x32x2f32",
     "float",
     "<16 x float>",
     NULL},
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

    /* LLVM 22 packs the 64-bit-per-lane A/B operand as scalar i64; LLVM 20
     * uses <2 x i32>. Same bits, different lane packing. */
    ab_ty = (L->flavor == ROCKE_LLVM_FLAVOR_LLVM22) ? "i64" : "<2 x i32>";

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
/* MX-scaled f8f6f4 / fp4 / fp6 / unscaled hero atoms                     */
/* ====================================================================== */

static void _op_tile_mfma_scale_f32_16x16x128_f8f6f4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t *a, *b, *c, *a_scale, *b_scale;
    const char *a_packed, *b_packed;
    const char *a_ty, *b_ty;

    if(!rocke_ll_live(L))
    {
        return;
    }
    if(op->num_operands != 5)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "%s expects 5 operands", op->name);
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    a_scale = op->operands[3];
    b_scale = op->operands[4];
    rocke_ll_need(L, "mfma.scale.f32.16x16x128.f8f6f4");

    /* Normalise A / B to <8 x i32> (accept either packed or byte-vector). */
    a_packed = rocke_ll_fresh(L, "mxa");
    b_packed = rocke_ll_fresh(L, "mxb");
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
                   "@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
                   "<8 x i32> %s, <8 x i32> %s, <4 x float> %s, "
                   "i32 0, i32 0, i32 0, i32 0, i32 %s, i32 0, i32 %s, i32 0)",
                   mma_result_name(L, op),
                   a_packed,
                   b_packed,
                   rocke_ll_operand(L, c),
                   rocke_ll_operand(L, a_scale),
                   rocke_ll_operand(L, b_scale));
}

static void _op_tile_mfma_f32_16x16x128_fp4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t *a, *b, *c;
    const char *a_cast, *b_cast;
    const char *a_ty, *b_ty;

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
    rocke_ll_need(L, "mfma.f32.16x16x128.fp4");
    /* fp4 mantissa packs 16 nibbles into i64 per lane; normalise to i64. */
    a_cast = rocke_ll_fresh(L, "a_fp4");
    b_cast = rocke_ll_fresh(L, "b_fp4");
    a_ty = rocke_ll_llvm_type(L, a->type);
    b_ty = rocke_ll_llvm_type(L, b->type);
    if(strcmp(a_ty, "i64") != 0)
    {
        rocke_ll_emitf(L, "  %s = bitcast %s %s to i64", a_cast, a_ty, rocke_ll_operand(L, a));
    }
    else
    {
        a_cast = rocke_ll_operand(L, a);
    }
    if(strcmp(b_ty, "i64") != 0)
    {
        rocke_ll_emitf(L, "  %s = bitcast %s %s to i64", b_cast, b_ty, rocke_ll_operand(L, b));
    }
    else
    {
        b_cast = rocke_ll_operand(L, b);
    }
    rocke_ll_emitf(L,
                   "  %s = call <4 x float> "
                   "@llvm.amdgcn.mfma.f32.16x16x128.fp4(i64 %s, i64 %s, "
                   "<4 x float> %s, i32 0, i32 0, i32 0)",
                   mma_result_name(L, op),
                   a_cast,
                   b_cast,
                   rocke_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x96_fp6(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t *a, *b, *c;
    const char *a_cast, *b_cast;
    const char *a_ty, *b_ty;

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
    rocke_ll_need(L, "mfma.f32.16x16x96.fp6");
    a_cast = rocke_ll_fresh(L, "a_fp6");
    b_cast = rocke_ll_fresh(L, "b_fp6");
    a_ty = rocke_ll_llvm_type(L, a->type);
    b_ty = rocke_ll_llvm_type(L, b->type);
    if(strcmp(a_ty, "<3 x i32>") != 0)
    {
        rocke_ll_emitf(
            L, "  %s = bitcast %s %s to <3 x i32>", a_cast, a_ty, rocke_ll_operand(L, a));
    }
    else
    {
        a_cast = rocke_ll_operand(L, a);
    }
    if(strcmp(b_ty, "<3 x i32>") != 0)
    {
        rocke_ll_emitf(
            L, "  %s = bitcast %s %s to <3 x i32>", b_cast, b_ty, rocke_ll_operand(L, b));
    }
    else
    {
        b_cast = rocke_ll_operand(L, b);
    }
    rocke_ll_emitf(L,
                   "  %s = call <4 x float> "
                   "@llvm.amdgcn.mfma.f32.16x16x96.fp6(<3 x i32> %s, "
                   "<3 x i32> %s, <4 x float> %s, i32 0, i32 0, i32 0)",
                   mma_result_name(L, op),
                   a_cast,
                   b_cast,
                   rocke_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x128_fp8(rocke_lower_t* L, const rocke_op_t* op)
{
    /* UNSCALED fp8 16x16x128 hero atom (L6): reuse the f8f6f4 scaled intrinsic
     * with both E8M0 scales pinned to 0 (2^0 == 1.0) so it is numerically a
     * plain unscaled fp8 MFMA. Uses a dedicated decl key for the 9-arg LLVM22
     * signature -- it does NOT touch the 11-arg MX-scaled decl. */
    const rocke_value_t *a, *b, *c;
    const char *a_packed, *b_packed;
    const char *a_ty, *b_ty;

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
    rocke_ll_need(L, "mfma.f32.16x16x128.fp8.hero");
    a_packed = rocke_ll_fresh(L, "a_fp8_128");
    b_packed = rocke_ll_fresh(L, "b_fp8_128");
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
                   "@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
                   "<8 x i32> %s, <8 x i32> %s, <4 x float> %s, "
                   "i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)",
                   mma_result_name(L, op),
                   a_packed,
                   b_packed,
                   rocke_ll_operand(L, c));
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
