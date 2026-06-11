/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * lower_llvm_lower_llvm_mma.c -- BUCKET 4 of the C99 port of
 * ck_dsl.core.lower_llvm.
 *
 * Owns the tile.mma routing and every MFMA atom handler:
 *   - f16 / bf16 16x16x* and 32x32x*
 *   - fp8 / bf8 via the shared ckc_ll_lower_mfma_fp8_bf8 body (defined here,
 *     called via the internal header by other buckets too)
 *   - scaled f8f6f4, fp4, fp6, and the unscaled fp8-128 hero atom
 *   - register_p_from_qk_c register-fragment reshape
 *   - WMMA-routing stubs (CDNA targets reject WMMA, mirroring the Python
 *     ISABackend.emit_wmma NotImplementedError)
 *
 * The ISA-named MFMA handlers are NOT distinct opcodes in the frozen ir.h
 * (only CKC_OP_TILE_MMA and CKC_OP_TILE_REGISTER_P_FROM_QK_C exist). They are
 * reached from _op_tile_mma's op_id routing, which mirrors the Python CDNA
 * ISABackend.emit_mma rebuilding ``tile.<op_id>`` and re-dispatching.
 */
#include <stdio.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/lower_llvm_internal.h"

/* ------------------------------------------------------------ forward decls */

/* per-op handlers (file-static; reached via the op_id router, not the table) */
static void _op_tile_wmma_f32_16x16x16_f16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_wmma_f32_16x16x16_bf16(ckc_lower_t *L, const ckc_op_t *op);
static void _emit_wmma(ckc_lower_t *L, const ckc_op_t *op, const char *op_id);
static void _op_tile_mma(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x16_f16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x32_f16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x16_bf16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x32_bf16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_32x32x8_f16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_32x32x16_f16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_32x32x16_bf16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x32_fp8(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x32_bf8(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_32x32x16_fp8(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_32x32x16_bf8(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_4x4x4_f16(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_scale_f32_16x16x128_f8f6f4(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x128_fp4(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x96_fp6(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_mfma_f32_16x16x128_fp8(ckc_lower_t *L, const ckc_op_t *op);
static void _op_tile_register_p_from_qk_c(ckc_lower_t *L, const ckc_op_t *op);

/* ------------------------------------------------------------ small helpers */

/* The single-result name (Python op.result.name). The MMA ops always have
 * exactly one result; guard defensively to keep the lowerer sticky-safe. */
static const char *mma_result_name(ckc_lower_t *L, const ckc_op_t *op) {
    if (op->num_results != 1) {
        ckc_ll_fail(L, CKC_ERR_VALUE,
                    "%s: expected exactly one result, got %d",
                    op->name ? op->name : "tile.mma", op->num_results);
        return "%__err";
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
static void _op_tile_mma(ckc_lower_t *L, const ckc_op_t *op) {
    const char *op_id;
    if (!ckc_ll_live(L)) {
        return;
    }
    op_id = ckc_attr_get_str(&op->attrs, "op_id");
    if (!op_id) {
        ckc_ll_fail(L, CKC_ERR_KEY, "tile.mma: missing op_id attribute");
        return;
    }

    /* f16 / bf16 dense */
    if (strcmp(op_id, "mfma_f32_16x16x16_f16") == 0) {
        _op_tile_mfma_f32_16x16x16_f16(L, op);
    } else if (strcmp(op_id, "mfma_f32_16x16x32_f16") == 0) {
        _op_tile_mfma_f32_16x16x32_f16(L, op);
    } else if (strcmp(op_id, "mfma_f32_16x16x16_bf16") == 0) {
        _op_tile_mfma_f32_16x16x16_bf16(L, op);
    } else if (strcmp(op_id, "mfma_f32_16x16x32_bf16") == 0) {
        _op_tile_mfma_f32_16x16x32_bf16(L, op);
    } else if (strcmp(op_id, "mfma_f32_32x32x8_f16") == 0) {
        _op_tile_mfma_f32_32x32x8_f16(L, op);
    } else if (strcmp(op_id, "mfma_f32_32x32x16_f16") == 0) {
        _op_tile_mfma_f32_32x32x16_f16(L, op);
    } else if (strcmp(op_id, "mfma_f32_32x32x16_bf16") == 0) {
        _op_tile_mfma_f32_32x32x16_bf16(L, op);
    } else if (strcmp(op_id, "mfma_f32_4x4x4_f16") == 0) {
        _op_tile_mfma_f32_4x4x4_f16(L, op);
    /* fp8 / bf8 */
    } else if (strcmp(op_id, "mfma_f32_16x16x32_fp8") == 0) {
        _op_tile_mfma_f32_16x16x32_fp8(L, op);
    } else if (strcmp(op_id, "mfma_f32_16x16x32_bf8") == 0) {
        _op_tile_mfma_f32_16x16x32_bf8(L, op);
    } else if (strcmp(op_id, "mfma_f32_32x32x16_fp8") == 0) {
        _op_tile_mfma_f32_32x32x16_fp8(L, op);
    } else if (strcmp(op_id, "mfma_f32_32x32x16_bf8") == 0) {
        _op_tile_mfma_f32_32x32x16_bf8(L, op);
    /* MX-scaled / fp4 / fp6 / unscaled hero */
    } else if (strcmp(op_id, "mfma_scale_f32_16x16x128_f8f6f4") == 0) {
        _op_tile_mfma_scale_f32_16x16x128_f8f6f4(L, op);
    } else if (strcmp(op_id, "mfma_f32_16x16x128_fp4") == 0) {
        _op_tile_mfma_f32_16x16x128_fp4(L, op);
    } else if (strcmp(op_id, "mfma_f32_16x16x96_fp6") == 0) {
        _op_tile_mfma_f32_16x16x96_fp6(L, op);
    } else if (strcmp(op_id, "mfma_f32_16x16x128_fp8") == 0) {
        _op_tile_mfma_f32_16x16x128_fp8(L, op);
    /* WMMA op_ids. On an RDNA backend (gfx11/gfx12) these emit a real WMMA
     * call (Python Gfx11/Gfx12RdnaBackend.emit_wmma); on a CDNA backend they
     * reject (Python ISABackend.emit_wmma raises NotImplementedError). The
     * gfx12-specific op_ids ("wmma_gfx12_*") can only occur on RDNA4. */
    } else if (strncmp(op_id, "wmma_", 5) == 0) {
        if (L->backend && L->backend->kind == CKC_LL_ISA_RDNA) {
            _emit_wmma(L, op, op_id);
        } else if (strcmp(op_id, "wmma_f32_16x16x16_f16") == 0) {
            _op_tile_wmma_f32_16x16x16_f16(L, op);
        } else if (strcmp(op_id, "wmma_f32_16x16x16_bf16") == 0) {
            _op_tile_wmma_f32_16x16x16_bf16(L, op);
        } else {
            ckc_ll_fail(L, CKC_ERR_NOTIMPL,
                        "WMMA op 'tile.%s' not available on %s "
                        "(WMMA is an RDNA/gfx11 instruction; this is a "
                        "CDNA/MFMA target)",
                        op_id, L->backend ? L->backend->gfx : "(cdna)");
        }
    } else {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL,
                    "tile.mma: unsupported op_id '%s'", op_id);
    }
}

/* ====================================================================== */
/* WMMA stubs (Python ISABackend.emit_wmma raises on CDNA targets)        */
/* ====================================================================== */

static void _op_tile_wmma_f32_16x16x16_f16(ckc_lower_t *L, const ckc_op_t *op) {
    (void)op;
    /* CDNA/MFMA targets reject WMMA (an RDNA/gfx11 instruction). The FROZEN
     * ir.h exposes no WMMA opcodes and the internal header notes RDNA WMMA
     * emission is out of scope, so this is a faithful NotImplementedError. */
    ckc_ll_fail(L, CKC_ERR_NOTIMPL,
                "WMMA op 'tile.wmma_f32_16x16x16_f16' not available on %s "
                "(WMMA is an RDNA/gfx11 instruction; this is a CDNA/MFMA target)",
                L->backend ? L->backend->gfx : "(cdna)");
}

static void _op_tile_wmma_f32_16x16x16_bf16(ckc_lower_t *L, const ckc_op_t *op) {
    (void)op;
    ckc_ll_fail(L, CKC_ERR_NOTIMPL,
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
 * via ckc/isa_backend.h to avoid the header's own `ckc_isa_backend` struct
 * colliding with the lowerer's same-named backend struct. The op_ids are
 * disjoint between the two families, so one flat table resolves both: gfx12
 * op_ids carry the "wmma_gfx12_" prefix (8-wide fragments), the rest are
 * RDNA3/3.5 (16-wide). */
typedef struct _wmma_spec {
    const char *op_id;     /* the tile.mma op_id (no "tile." prefix)        */
    const char *decl_key;  /* _need() key                                   */
    const char *intrinsic; /* fully-mangled @llvm.amdgcn.wmma....           */
    const char *ssa_elt;   /* SSA operand element type                      */
    const char *call_elt;  /* call-site operand element type                */
    int frag_width;        /* A/B operand vector width (16 RDNA3/3.5, 8 RDNA4) */
} _wmma_spec_t;

static const _wmma_spec_t WMMA_SPECS[] = {
    /* _RDNA_WMMA (RDNA3/3.5, frag_width 16) */
    {"wmma_f32_16x16x16_f16", "wmma.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16", "half", "half", 16},
    {"wmma_f32_16x16x16_bf16", "wmma.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16", "bfloat", "i16", 16},
    /* _RDNA_GFX12_WMMA (RDNA4, frag_width 8) */
    {"wmma_gfx12_f32_16x16x16_f16", "wmma.gfx12.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16", "half", "half", 8},
    {"wmma_gfx12_f32_16x16x16_bf16", "wmma.gfx12.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16", "bfloat", "i16", 8},
};
static const int WMMA_SPECS_N = (int)(sizeof(WMMA_SPECS) / sizeof(WMMA_SPECS[0]));

/* Integer WMMA spec table (Python _RDNA_WMMA_INT). Integer WMMA differs from
 * the float path: operands/accumulator are i32 vectors (A/B packed, C/D the i32
 * accumulator), and the intrinsic signature carries i1 signedness flags before
 * each matrix operand and a trailing i1 clamp. Operands arrive in SSA already
 * as <N x i32> so no bitcast is needed. Quantized data is signed and within i32
 * range, so the flags are (signedA=1, signedB=1, clamp=0). */
typedef struct _wmma_int_spec {
    const char *op_id;     /* the tile.mma op_id (no "tile." prefix)        */
    const char *decl_key;  /* _need() key                                   */
    const char *intrinsic; /* fully-mangled @llvm.amdgcn.wmma....           */
    int op_vec;            /* A/B operand vector width                      */
    int acc_vec;           /* accumulator/result vector width               */
} _wmma_int_spec_t;

static const _wmma_int_spec_t WMMA_INT_SPECS[] = {
    {"wmma_i32_16x16x16_iu8", "wmma.i32.16x16x16.iu8",
     "llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32", 4, 8},
    {"wmma_i32_16x16x16_iu4", "wmma.i32.16x16x16.iu4",
     "llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32", 2, 8},
};
static const int WMMA_INT_SPECS_N =
    (int)(sizeof(WMMA_INT_SPECS) / sizeof(WMMA_INT_SPECS[0]));

/* Emit an integer WMMA (iu8/iu4) call (Python _emit_wmma_int). The signature is
 * (i1 signedA, <N x i32> A, i1 signedB, <N x i32> B, <8 x i32> C, i1 clamp) with
 * an <8 x i32> result. Both signedness flags are 1 (signed quant data); clamp
 * is 0 (values stay within i32 range). No operand bitcast (already <N x i32>). */
static void _emit_wmma_int(ckc_lower_t *L, const ckc_op_t *op,
                           const _wmma_int_spec_t *spec) {
    const ckc_value_t *a, *b, *c;
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, spec->decl_key);
    ckc_ll_emitf(L,
        "  %s = call <%d x i32> @%s("
        "i1 1, <%d x i32> %s, "
        "i1 1, <%d x i32> %s, "
        "<%d x i32> %s, i1 0)",
        mma_result_name(L, op), spec->acc_vec, spec->intrinsic,
        spec->op_vec, ckc_ll_operand(L, a),
        spec->op_vec, ckc_ll_operand(L, b),
        spec->acc_vec, ckc_ll_operand(L, c));
}

static void _emit_wmma(ckc_lower_t *L, const ckc_op_t *op, const char *op_id) {
    const _wmma_spec_t *spec = NULL;
    const ckc_value_t *a, *b, *c;
    const char *a_arg, *b_arg;
    int w, i;

    if (!ckc_ll_live(L)) {
        return;
    }
    if (op->num_operands != 3) {
        ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands",
                    op->name ? op->name : "tile.mma");
        return;
    }

    /* Integer WMMA (iu8/iu4) is checked first, mirroring
     * Gfx11RdnaBackend.emit_wmma (int_spec lookup precedes the float table). */
    for (i = 0; i < WMMA_INT_SPECS_N; i++) {
        if (strcmp(WMMA_INT_SPECS[i].op_id, op_id) == 0) {
            _emit_wmma_int(L, op, &WMMA_INT_SPECS[i]);
            return;
        }
    }

    for (i = 0; i < WMMA_SPECS_N; i++) {
        if (strcmp(WMMA_SPECS[i].op_id, op_id) == 0) {
            spec = &WMMA_SPECS[i];
            break;
        }
    }
    if (spec == NULL) {
        ckc_ll_fail(L, CKC_ERR_NOTIMPL,
                    "WMMA op 'tile.%s' not yet wired for %s", op_id,
                    L->backend ? L->backend->gfx : "(rdna)");
        return;
    }

    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    w = spec->frag_width;

    ckc_ll_need(L, spec->decl_key);
    a_arg = ckc_ll_operand(L, a);
    b_arg = ckc_ll_operand(L, b);

    if (strcmp(spec->call_elt, spec->ssa_elt) != 0) {
        /* bf16 (and any future type whose SSA element differs from the
         * intrinsic's operand element): bitcast <W x ssa_elt> -> <W x call_elt>. */
        const char *a_cast = ckc_ll_fresh(L, "wmma_a");
        const char *b_cast = ckc_ll_fresh(L, "wmma_b");
        ckc_ll_emitf(L, "  %s = bitcast <%d x %s> %s to <%d x %s>",
                     a_cast, w, spec->ssa_elt, a_arg, w, spec->call_elt);
        ckc_ll_emitf(L, "  %s = bitcast <%d x %s> %s to <%d x %s>",
                     b_cast, w, spec->ssa_elt, b_arg, w, spec->call_elt);
        a_arg = a_cast;
        b_arg = b_cast;
    }

    ckc_ll_emitf(L,
        "  %s = call <8 x float> @%s("
        "<%d x %s> %s, <%d x %s> %s, <8 x float> %s)",
        mma_result_name(L, op), spec->intrinsic,
        w, spec->call_elt, a_arg, w, spec->call_elt, b_arg,
        ckc_ll_operand(L, c));
}

/* ====================================================================== */
/* f16 / bf16 dense MFMA atoms                                            */
/* ====================================================================== */

static void _op_tile_mfma_f32_16x16x16_f16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.16x16x16f16");
    ckc_ll_emitf(L,
        "  %s = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16("
        "<4 x half> %s, <4 x half> %s, <4 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op),
        ckc_ll_operand(L, a), ckc_ll_operand(L, b), ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x32_f16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.16x16x32.f16");
    ckc_ll_emitf(L,
        "  %s = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16("
        "<8 x half> %s, <8 x half> %s, <4 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op),
        ckc_ll_operand(L, a), ckc_ll_operand(L, b), ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x16_bf16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    const char *a_cast, *b_cast;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.16x16x16bf16.1k");
    /* bitcast <4 x bfloat> -> <4 x i16> for the `_1k` intrinsic. */
    a_cast = ckc_ll_fresh(L, "mfma_a_i16");
    b_cast = ckc_ll_fresh(L, "mfma_b_i16");
    ckc_ll_emitf(L, "  %s = bitcast <4 x bfloat> %s to <4 x i16>",
                 a_cast, ckc_ll_operand(L, a));
    ckc_ll_emitf(L, "  %s = bitcast <4 x bfloat> %s to <4 x i16>",
                 b_cast, ckc_ll_operand(L, b));
    ckc_ll_emitf(L,
        "  %s = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k("
        "<4 x i16> %s, <4 x i16> %s, <4 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op), a_cast, b_cast, ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x32_bf16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.16x16x32.bf16");
    ckc_ll_emitf(L,
        "  %s = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16("
        "<8 x bfloat> %s, <8 x bfloat> %s, <4 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op),
        ckc_ll_operand(L, a), ckc_ll_operand(L, b), ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_32x32x8_f16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.32x32x8f16");
    ckc_ll_emitf(L,
        "  %s = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16("
        "<4 x half> %s, <4 x half> %s, <16 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op),
        ckc_ll_operand(L, a), ckc_ll_operand(L, b), ckc_ll_operand(L, c));
}

/* NOTE: Python defines _op_tile_mfma_f32_32x32x16_f16 twice (the second
 * definition at module scope wins); both bodies are identical, so this single
 * implementation reproduces the effective behaviour byte-for-byte. */
static void _op_tile_mfma_f32_32x32x16_f16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.32x32x16.f16");
    ckc_ll_emitf(L,
        "  %s = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16("
        "<8 x half> %s, <8 x half> %s, <16 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op),
        ckc_ll_operand(L, a), ckc_ll_operand(L, b), ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_32x32x16_bf16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.32x32x16.bf16");
    ckc_ll_emitf(L,
        "  %s = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16("
        "<8 x bfloat> %s, <8 x bfloat> %s, <16 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op),
        ckc_ll_operand(L, a), ckc_ll_operand(L, b), ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_4x4x4_f16(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    if (!ckc_ll_live(L) || op->num_operands != 3) {
        if (ckc_ll_live(L)) {
            ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        }
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.4x4x4f16");
    ckc_ll_emitf(L,
        "  %s = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16("
        "<4 x half> %s, <4 x half> %s, <4 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op),
        ckc_ll_operand(L, a), ckc_ll_operand(L, b), ckc_ll_operand(L, c));
}

/* ====================================================================== */
/* Shared FP8 / BF8 MFMA body (Python _lower_mfma_fp8_bf8)               */
/* ====================================================================== */

void ckc_ll_lower_mfma_fp8_bf8(ckc_lower_t *L, const ckc_op_t *op,
                               const char *dtype, int out_vec,
                               const char *intrinsic) {
    const ckc_value_t *a, *b, *c;
    const char *ab_ty;
    const char *a_cast, *b_cast;
    char key[64];
    char a_hint[32], b_hint[32];

    if (!ckc_ll_live(L)) {
        return;
    }
    if (op->num_operands != 3) {
        ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];

    /* _need(f"mfma.f32.{intrinsic}") */
    snprintf(key, sizeof(key), "mfma.f32.%s", intrinsic);
    ckc_ll_need(L, key);

    /* LLVM 22 packs the 64-bit-per-lane A/B operand as scalar i64; LLVM 20
     * uses <2 x i32>. Same bits, different lane packing. */
    ab_ty = (L->flavor == CKC_LLVM_FLAVOR_LLVM22) ? "i64" : "<2 x i32>";

    snprintf(a_hint, sizeof(a_hint), "mfma_a_%s", dtype ? dtype : "f8");
    snprintf(b_hint, sizeof(b_hint), "mfma_b_%s", dtype ? dtype : "f8");
    a_cast = ckc_ll_fresh(L, a_hint);
    b_cast = ckc_ll_fresh(L, b_hint);

    ckc_ll_emitf(L, "  %s = bitcast <8 x i8> %s to %s",
                 a_cast, ckc_ll_operand(L, a), ab_ty);
    ckc_ll_emitf(L, "  %s = bitcast <8 x i8> %s to %s",
                 b_cast, ckc_ll_operand(L, b), ab_ty);
    ckc_ll_emitf(L,
        "  %s = call <%d x float> @llvm.amdgcn.mfma.f32.%s("
        "%s %s, %s %s, <%d x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op), out_vec, intrinsic,
        ab_ty, a_cast, ab_ty, b_cast,
        out_vec, ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x32_fp8(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_lower_mfma_fp8_bf8(L, op, "fp8", 4, "16x16x32.fp8.fp8");
}

static void _op_tile_mfma_f32_16x16x32_bf8(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_lower_mfma_fp8_bf8(L, op, "bf8", 4, "16x16x32.bf8.bf8");
}

static void _op_tile_mfma_f32_32x32x16_fp8(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_lower_mfma_fp8_bf8(L, op, "fp8", 16, "32x32x16.fp8.fp8");
}

static void _op_tile_mfma_f32_32x32x16_bf8(ckc_lower_t *L, const ckc_op_t *op) {
    ckc_ll_lower_mfma_fp8_bf8(L, op, "bf8", 16, "32x32x16.bf8.bf8");
}

/* ====================================================================== */
/* MX-scaled f8f6f4 / fp4 / fp6 / unscaled hero atoms                     */
/* ====================================================================== */

static void _op_tile_mfma_scale_f32_16x16x128_f8f6f4(ckc_lower_t *L,
                                                     const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c, *a_scale, *b_scale;
    const char *a_packed, *b_packed;
    const char *a_ty, *b_ty;

    if (!ckc_ll_live(L)) {
        return;
    }
    if (op->num_operands != 5) {
        ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 5 operands", op->name);
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    a_scale = op->operands[3];
    b_scale = op->operands[4];
    ckc_ll_need(L, "mfma.scale.f32.16x16x128.f8f6f4");

    /* Normalise A / B to <8 x i32> (accept either packed or byte-vector). */
    a_packed = ckc_ll_fresh(L, "mxa");
    b_packed = ckc_ll_fresh(L, "mxb");
    a_ty = ckc_ll_llvm_type(L, a->type);
    b_ty = ckc_ll_llvm_type(L, b->type);
    if (strcmp(a_ty, "<8 x i32>") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to <8 x i32>",
                     a_packed, a_ty, ckc_ll_operand(L, a));
    } else {
        a_packed = ckc_ll_operand(L, a);
    }
    if (strcmp(b_ty, "<8 x i32>") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to <8 x i32>",
                     b_packed, b_ty, ckc_ll_operand(L, b));
    } else {
        b_packed = ckc_ll_operand(L, b);
    }
    ckc_ll_emitf(L,
        "  %s = call <4 x float> "
        "@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
        "<8 x i32> %s, <8 x i32> %s, <4 x float> %s, "
        "i32 0, i32 0, i32 0, i32 0, i32 %s, i32 0, i32 %s, i32 0)",
        mma_result_name(L, op), a_packed, b_packed, ckc_ll_operand(L, c),
        ckc_ll_operand(L, a_scale), ckc_ll_operand(L, b_scale));
}

static void _op_tile_mfma_f32_16x16x128_fp4(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    const char *a_cast, *b_cast;
    const char *a_ty, *b_ty;

    if (!ckc_ll_live(L)) {
        return;
    }
    if (op->num_operands != 3) {
        ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.16x16x128.fp4");
    /* fp4 mantissa packs 16 nibbles into i64 per lane; normalise to i64. */
    a_cast = ckc_ll_fresh(L, "a_fp4");
    b_cast = ckc_ll_fresh(L, "b_fp4");
    a_ty = ckc_ll_llvm_type(L, a->type);
    b_ty = ckc_ll_llvm_type(L, b->type);
    if (strcmp(a_ty, "i64") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to i64",
                     a_cast, a_ty, ckc_ll_operand(L, a));
    } else {
        a_cast = ckc_ll_operand(L, a);
    }
    if (strcmp(b_ty, "i64") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to i64",
                     b_cast, b_ty, ckc_ll_operand(L, b));
    } else {
        b_cast = ckc_ll_operand(L, b);
    }
    ckc_ll_emitf(L,
        "  %s = call <4 x float> "
        "@llvm.amdgcn.mfma.f32.16x16x128.fp4(i64 %s, i64 %s, "
        "<4 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op), a_cast, b_cast, ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x96_fp6(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *a, *b, *c;
    const char *a_cast, *b_cast;
    const char *a_ty, *b_ty;

    if (!ckc_ll_live(L)) {
        return;
    }
    if (op->num_operands != 3) {
        ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.16x16x96.fp6");
    a_cast = ckc_ll_fresh(L, "a_fp6");
    b_cast = ckc_ll_fresh(L, "b_fp6");
    a_ty = ckc_ll_llvm_type(L, a->type);
    b_ty = ckc_ll_llvm_type(L, b->type);
    if (strcmp(a_ty, "<3 x i32>") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to <3 x i32>",
                     a_cast, a_ty, ckc_ll_operand(L, a));
    } else {
        a_cast = ckc_ll_operand(L, a);
    }
    if (strcmp(b_ty, "<3 x i32>") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to <3 x i32>",
                     b_cast, b_ty, ckc_ll_operand(L, b));
    } else {
        b_cast = ckc_ll_operand(L, b);
    }
    ckc_ll_emitf(L,
        "  %s = call <4 x float> "
        "@llvm.amdgcn.mfma.f32.16x16x96.fp6(<3 x i32> %s, "
        "<3 x i32> %s, <4 x float> %s, i32 0, i32 0, i32 0)",
        mma_result_name(L, op), a_cast, b_cast, ckc_ll_operand(L, c));
}

static void _op_tile_mfma_f32_16x16x128_fp8(ckc_lower_t *L, const ckc_op_t *op) {
    /* UNSCALED fp8 16x16x128 hero atom (L6): reuse the f8f6f4 scaled intrinsic
     * with both E8M0 scales pinned to 0 (2^0 == 1.0) so it is numerically a
     * plain unscaled fp8 MFMA. Uses a dedicated decl key for the 9-arg LLVM22
     * signature -- it does NOT touch the 11-arg MX-scaled decl. */
    const ckc_value_t *a, *b, *c;
    const char *a_packed, *b_packed;
    const char *a_ty, *b_ty;

    if (!ckc_ll_live(L)) {
        return;
    }
    if (op->num_operands != 3) {
        ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 3 operands", op->name);
        return;
    }
    a = op->operands[0];
    b = op->operands[1];
    c = op->operands[2];
    ckc_ll_need(L, "mfma.f32.16x16x128.fp8.hero");
    a_packed = ckc_ll_fresh(L, "a_fp8_128");
    b_packed = ckc_ll_fresh(L, "b_fp8_128");
    a_ty = ckc_ll_llvm_type(L, a->type);
    b_ty = ckc_ll_llvm_type(L, b->type);
    if (strcmp(a_ty, "<8 x i32>") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to <8 x i32>",
                     a_packed, a_ty, ckc_ll_operand(L, a));
    } else {
        a_packed = ckc_ll_operand(L, a);
    }
    if (strcmp(b_ty, "<8 x i32>") != 0) {
        ckc_ll_emitf(L, "  %s = bitcast %s %s to <8 x i32>",
                     b_packed, b_ty, ckc_ll_operand(L, b));
    } else {
        b_packed = ckc_ll_operand(L, b);
    }
    ckc_ll_emitf(L,
        "  %s = call <4 x float> "
        "@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
        "<8 x i32> %s, <8 x i32> %s, <4 x float> %s, "
        "i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)",
        mma_result_name(L, op), a_packed, b_packed, ckc_ll_operand(L, c));
}

/* ====================================================================== */
/* register_p_from_qk_c (Python _op_tile_register_p_from_qk_c, P13)       */
/* ====================================================================== */

static void _op_tile_register_p_from_qk_c(ckc_lower_t *L, const ckc_op_t *op) {
    const ckc_value_t *qk_c;
    const char *target;
    const char *target_llvm;
    const char *elems[8];
    const char *prev;
    int i;

    if (!ckc_ll_live(L)) {
        return;
    }
    if (op->num_operands != 1) {
        ckc_ll_fail(L, CKC_ERR_VALUE, "%s expects 1 operand", op->name);
        return;
    }
    qk_c = op->operands[0];
    target = ckc_attr_get_str(&op->attrs, "target_dtype");
    if (!target) {
        ckc_ll_fail(L, CKC_ERR_KEY,
                    "register_p_from_qk_c: missing target_dtype attribute");
        return;
    }
    if (strcmp(target, "f16") == 0) {
        target_llvm = "half";
    } else if (strcmp(target, "bf16") == 0) {
        target_llvm = "bfloat";
    } else {
        ckc_ll_fail(L, CKC_ERR_KEY,
                    "register_p_from_qk_c: bad target_dtype '%s'", target);
        return;
    }

    /* Extract 16 f32 cells, fptrunc the first 8 in canonical order. */
    for (i = 0; i < 8; i++) {
        char ehint[8], thint[8];
        const char *e, *t;
        snprintf(ehint, sizeof(ehint), "pe%d", i);
        snprintf(thint, sizeof(thint), "pt%d", i);
        e = ckc_ll_fresh(L, ehint);
        ckc_ll_emitf(L,
            "  %s = extractelement <16 x float> %s, i32 %d",
            e, ckc_ll_operand(L, qk_c), i);
        t = ckc_ll_fresh(L, thint);
        ckc_ll_emitf(L, "  %s = fptrunc float %s to %s", t, e, target_llvm);
        elems[i] = t;
    }

    /* Pack into <8 x dtype>; the last insertelement is the op result. */
    prev = "undef";
    for (i = 0; i < 8; i++) {
        const char *name;
        if (i == 7) {
            name = mma_result_name(L, op);
        } else {
            char phint[8];
            snprintf(phint, sizeof(phint), "pp%d", i);
            name = ckc_ll_fresh(L, phint);
        }
        ckc_ll_emitf(L,
            "  %s = insertelement <8 x %s> %s, %s %s, i32 %d",
            name, target_llvm, prev, target_llvm, elems[i], i);
        prev = name;
    }
}

/* ====================================================================== */
/* Bucket registration hook                                               */
/* ====================================================================== */

void ckc_ll_register_mma(void) {
    /* The frozen ir.h only exposes CKC_OP_TILE_MMA and
     * CKC_OP_TILE_REGISTER_P_FROM_QK_C as opcodes; the ISA-named MFMA / WMMA
     * handlers are not distinct opcodes and are reached through _op_tile_mma's
     * op_id router (mirroring the Python CDNA emit_mma re-dispatch). */
    ckc_ll_set_handler(CKC_OP_TILE_MMA, _op_tile_mma);
    ckc_ll_set_handler(CKC_OP_TILE_REGISTER_P_FROM_QK_C,
                       _op_tile_register_p_from_qk_c);
}
