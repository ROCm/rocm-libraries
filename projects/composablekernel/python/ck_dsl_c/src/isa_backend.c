/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/isa_backend.c -- C99 port of ck_dsl.core.isa.backend.
 *
 * See ckc/isa_backend.h for the Python -> C mapping. This file binds to the
 * frozen IR contract (ckc/ir.h) for the WMMA spec types and emits via
 * ckc_strbuf. It carries hardware facts only (no IR graph allocation), so it
 * does not need a builder arena.
 */
#include "ckc/isa_backend.h"

#include <string.h>

/* The shared LLVM constants are owned by lower_llvm in Python (pulled lazily to
 * avoid an import cycle). In C there is no cycle: we hold the canonical strings
 * here as static data, byte-identical to lower_llvm._TRIPLE / _DATALAYOUT. When
 * the lower_llvm port lands, these may be moved there and referenced; until
 * then this is the single C-side source. */
static const char CKC_TRIPLE[] = "amdgcn-amd-amdhsa";
static const char CKC_DATALAYOUT[] =
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32"
    "-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
    "-n32:64-S32-A5-G1-ni:7:8:9";

/* ================================ registry ============================== */

/* gfx -> backend descriptor. Adding a CDNA gfx is one row here (Python
 * BACKEND_REGISTRY). vmcnt_bits records arch.vmcnt_bits; it is a hardware fact
 * the lowerer keys future divergence on. The wired CDNA targets all share the
 * gfx9/10 split waitcnt layout and 6-bit VMCNT today (gfx942's potential
 * 4-bit compv4 partial-waitcnt is recorded as vmcnt_bits but does not change
 * the encoder selection). */
typedef struct registry_row {
    const char           *gfx;
    ckc_isa_kind_t        kind;
    int                   vmcnt_bits;
    uint32_t              word3;
    ckc_waitcnt_layout_t  waitcnt;
    int                   wave_size;
} registry_row_t;

static const registry_row_t REGISTRY[] = {
    {"gfx908",       CKC_ISA_GFX9_MFMA, 6, CKC_BUFFER_RSRC_WORD3_CDNA, CKC_WAITCNT_GFX9_10, 64},
    {"gfx90a",       CKC_ISA_GFX9_MFMA, 6, CKC_BUFFER_RSRC_WORD3_CDNA, CKC_WAITCNT_GFX9_10, 64},
    {"gfx942",       CKC_ISA_GFX9_MFMA, 6, CKC_BUFFER_RSRC_WORD3_CDNA, CKC_WAITCNT_GFX9_10, 64},
    {"gfx950",       CKC_ISA_GFX950,    6, CKC_BUFFER_RSRC_WORD3_CDNA, CKC_WAITCNT_GFX9_10, 64},
    {"gfx1151",      CKC_ISA_GFX11_RDNA, 6, CKC_BUFFER_RSRC_WORD3_RDNA, CKC_WAITCNT_GFX11, 32},
    {"gfx1201",      CKC_ISA_GFX12_RDNA, 6, CKC_BUFFER_RSRC_WORD3_RDNA, CKC_WAITCNT_GFX11, 32},
    {"gfx11-generic", CKC_ISA_GFX11_RDNA, 6, CKC_BUFFER_RSRC_WORD3_RDNA, CKC_WAITCNT_GFX11, 32},
};
static const int REGISTRY_N = (int)(sizeof(REGISTRY) / sizeof(REGISTRY[0]));

static const registry_row_t *registry_find(const char *gfx) {
    int i;
    if (gfx == NULL) {
        return NULL;
    }
    for (i = 0; i < REGISTRY_N; ++i) {
        if (strcmp(REGISTRY[i].gfx, gfx) == 0) {
            return &REGISTRY[i];
        }
    }
    return NULL;
}

ckc_isa_backend_t ckc_backend_for(const char *gfx, const char **err) {
    ckc_isa_backend_t be;
    const registry_row_t *row;

    memset(&be, 0, sizeof(be));
    be.valid = false;

    row = registry_find(gfx);
    if (row == NULL) {
        if (err != NULL) {
            /* Static message; matches the spirit of the Python KeyError
             * ("no ISA backend registered for ...; known: ..."). The exact
             * sorted-list suffix is reproducible from ckc_backend_is_known if
             * a caller wants it, so we keep this allocation-free. */
            *err = "no ISA backend registered for the given gfx target; known: "
                   "gfx11-generic, gfx1151, gfx1201, gfx908, gfx90a, gfx942, gfx950";
        }
        return be;
    }
    be.kind              = row->kind;
    be.gfx               = row->gfx; /* interned static string */
    be.vmcnt_bits        = row->vmcnt_bits;
    be.buffer_rsrc_word3 = row->word3;
    be.waitcnt_layout    = row->waitcnt;
    be.wave_size         = row->wave_size;
    be.valid             = true;
    if (err != NULL) {
        *err = NULL;
    }
    return be;
}

bool ckc_backend_is_known(const char *gfx) {
    return registry_find(gfx) != NULL;
}

/* ============================ module preamble =========================== */

const char *ckc_isa_triple(const ckc_isa_backend_t *be) {
    (void)be; /* shared across all wired targets */
    return CKC_TRIPLE;
}

const char *ckc_isa_datalayout(const ckc_isa_backend_t *be) {
    (void)be; /* shared across all wired targets */
    return CKC_DATALAYOUT;
}

int ckc_isa_module_preamble(const ckc_isa_backend_t *be, ckc_strbuf_t *out) {
    if (out == NULL) {
        return -1;
    }
    ckc_strbuf_clear(out);
    if (ckc_strbuf_appendf(out,
                           "target datalayout = \"%s\"\ntarget triple = \"%s\"",
                           ckc_isa_datalayout(be), ckc_isa_triple(be)) != 0) {
        return -1;
    }
    return 0;
}

uint32_t ckc_isa_buffer_rsrc_word3(const ckc_isa_backend_t *be) {
    if (be == NULL || !be->valid) {
        return CKC_BUFFER_RSRC_WORD3_CDNA; /* base-class default */
    }
    return be->buffer_rsrc_word3;
}

/* =============================== s_waitcnt ============================== */

/* min(max(v,0),hi) for non-negative clamp; v<0 means "no wait" => hi. */
static int waitcnt_clamp(int v, int hi) {
    if (v < 0) {
        return hi; /* "no wait" => field maximum */
    }
    if (v > hi) {
        return hi; /* clamp to field maximum (never wrap) */
    }
    return v;
}

int ckc_encode_waitcnt_gfx9_10(int vmcnt, int expcnt, int lgkmcnt) {
    int vm_b = waitcnt_clamp(vmcnt, 0x3F);
    int ec_b = waitcnt_clamp(expcnt, 0x7);
    int lk_b = waitcnt_clamp(lgkmcnt, 0xF);
    int vm_lo = vm_b & 0xF;
    int vm_hi = (vm_b >> 4) & 0x3;
    return vm_lo | (ec_b << 4) | (lk_b << 8) | (vm_hi << 14);
}

int ckc_encode_waitcnt_gfx11(int vmcnt, int expcnt, int lgkmcnt) {
    int vm_b = waitcnt_clamp(vmcnt, 0x3F);
    int ec_b = waitcnt_clamp(expcnt, 0x7);
    int lk_b = waitcnt_clamp(lgkmcnt, 0x3F);
    return (ec_b & 0x7) | ((lk_b & 0x3F) << 4) | ((vm_b & 0x3F) << 10);
}

int ckc_isa_encode_waitcnt(const ckc_isa_backend_t *be,
                           int vmcnt, int expcnt, int lgkmcnt) {
    if (be != NULL && be->valid && be->waitcnt_layout == CKC_WAITCNT_GFX11) {
        return ckc_encode_waitcnt_gfx11(vmcnt, expcnt, lgkmcnt);
    }
    return ckc_encode_waitcnt_gfx9_10(vmcnt, expcnt, lgkmcnt);
}

/* =============================== WMMA tables ============================ */

/* _RDNA_WMMA: RDNA3/3.5 float WMMA. frag_width = 16 (cross-half duplication). */
static const ckc_wmma_spec_t RDNA_WMMA[] = {
    {"tile.wmma_f32_16x16x16_f16",
     "wmma.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16",
     "half", "half", 16},
    {"tile.wmma_f32_16x16x16_bf16",
     "wmma.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16",
     "bfloat", "i16", 16},
};
static const int RDNA_WMMA_N = (int)(sizeof(RDNA_WMMA) / sizeof(RDNA_WMMA[0]));

/* _RDNA_WMMA_INT: RDNA3/3.5 integer WMMA (iu8/iu4). */
static const ckc_wmma_int_spec_t RDNA_WMMA_INT[] = {
    {"tile.wmma_i32_16x16x16_iu8",
     "wmma.i32.16x16x16.iu8",
     "llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32",
     4, 8},
    {"tile.wmma_i32_16x16x16_iu4",
     "wmma.i32.16x16x16.iu4",
     "llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32",
     2, 8},
};
static const int RDNA_WMMA_INT_N =
    (int)(sizeof(RDNA_WMMA_INT) / sizeof(RDNA_WMMA_INT[0]));

/* _RDNA_GFX12_WMMA: RDNA4 float WMMA. frag_width = 8 (no cross-half dup). */
static const ckc_wmma_spec_t RDNA_GFX12_WMMA[] = {
    {"tile.wmma_gfx12_f32_16x16x16_f16",
     "wmma.gfx12.f32.16x16x16.f16",
     "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16",
     "half", "half", 8},
    {"tile.wmma_gfx12_f32_16x16x16_bf16",
     "wmma.gfx12.f32.16x16x16.bf16",
     "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16",
     "bfloat", "i16", 8},
};
static const int RDNA_GFX12_WMMA_N =
    (int)(sizeof(RDNA_GFX12_WMMA) / sizeof(RDNA_GFX12_WMMA[0]));

const ckc_wmma_spec_t *ckc_isa_wmma_lookup(const char *op_name) {
    int i;
    if (op_name == NULL) {
        return NULL;
    }
    for (i = 0; i < RDNA_WMMA_N; ++i) {
        if (strcmp(RDNA_WMMA[i].op_name, op_name) == 0) {
            return &RDNA_WMMA[i];
        }
    }
    return NULL;
}

const ckc_wmma_int_spec_t *ckc_isa_wmma_int_lookup(const char *op_name) {
    int i;
    if (op_name == NULL) {
        return NULL;
    }
    for (i = 0; i < RDNA_WMMA_INT_N; ++i) {
        if (strcmp(RDNA_WMMA_INT[i].op_name, op_name) == 0) {
            return &RDNA_WMMA_INT[i];
        }
    }
    return NULL;
}

const ckc_wmma_spec_t *ckc_isa_wmma_gfx12_lookup(const char *op_name) {
    int i;
    if (op_name == NULL) {
        return NULL;
    }
    for (i = 0; i < RDNA_GFX12_WMMA_N; ++i) {
        if (strcmp(RDNA_GFX12_WMMA[i].op_name, op_name) == 0) {
            return &RDNA_GFX12_WMMA[i];
        }
    }
    return NULL;
}

const ckc_wmma_spec_t *ckc_isa_resolve_wmma(const ckc_isa_backend_t *be,
                                            const char *op_name) {
    if (be == NULL || !be->valid) {
        return NULL;
    }
    if (be->kind == CKC_ISA_GFX12_RDNA) {
        return ckc_isa_wmma_gfx12_lookup(op_name);
    }
    if (be->kind == CKC_ISA_GFX11_RDNA) {
        return ckc_isa_wmma_lookup(op_name);
    }
    /* CDNA kinds reject WMMA (Python emit_wmma raises NotImplementedError;
     * MFMA lowers inline in the lowerer). */
    return NULL;
}

/* ============================ WMMA call emission ======================= */

int ckc_isa_emit_wmma_call(ckc_strbuf_t *out,
                           const ckc_wmma_spec_t *spec,
                           const char *result_name,
                           const char *a_name, const char *b_name,
                           const char *c_name,
                           const char *a_cast_name, const char *b_cast_name) {
    const char *a_arg;
    const char *b_arg;
    int w;

    if (out == NULL || spec == NULL || result_name == NULL ||
        a_name == NULL || b_name == NULL || c_name == NULL) {
        return -1;
    }
    w = spec->frag_width;
    a_arg = a_name;
    b_arg = b_name;

    if (strcmp(spec->call_elt, spec->ssa_elt) != 0) {
        /* bf16: bitcast <W x ssa_elt> -> <W x call_elt> before the call. The
         * caller supplies the fresh names (lowerer._fresh("wmma_a"/"wmma_b")). */
        if (a_cast_name == NULL || b_cast_name == NULL) {
            return CKC_ERR_NOTIMPL;
        }
        if (ckc_strbuf_appendf(out,
                "  %s = bitcast <%d x %s> %s to <%d x %s>\n",
                a_cast_name, w, spec->ssa_elt, a_name, w, spec->call_elt) != 0) {
            return -1;
        }
        if (ckc_strbuf_appendf(out,
                "  %s = bitcast <%d x %s> %s to <%d x %s>\n",
                b_cast_name, w, spec->ssa_elt, b_name, w, spec->call_elt) != 0) {
            return -1;
        }
        a_arg = a_cast_name;
        b_arg = b_cast_name;
    }

    if (ckc_strbuf_appendf(out,
            "  %s = call <8 x float> @%s("
            "<%d x %s> %s, <%d x %s> %s, <8 x float> %s)\n",
            result_name, spec->intrinsic,
            w, spec->call_elt, a_arg,
            w, spec->call_elt, b_arg,
            c_name) != 0) {
        return -1;
    }
    return 0;
}

int ckc_isa_emit_wmma_int_call(ckc_strbuf_t *out,
                               const ckc_wmma_int_spec_t *spec,
                               const char *result_name,
                               const char *a_name, const char *b_name,
                               const char *c_name) {
    if (out == NULL || spec == NULL || result_name == NULL ||
        a_name == NULL || b_name == NULL || c_name == NULL) {
        return -1;
    }
    /* signedA=1, signedB=1 (our quantized data is signed), clamp=0 (values stay
     * within i32 range). Operands arrive as <op_vec x i32> in SSA, no bitcast. */
    if (ckc_strbuf_appendf(out,
            "  %s = call <%d x i32> @%s("
            "i1 1, <%d x i32> %s, "
            "i1 1, <%d x i32> %s, "
            "<%d x i32> %s, i1 0)\n",
            result_name, spec->acc_vec, spec->intrinsic,
            spec->op_vec, a_name,
            spec->op_vec, b_name,
            spec->acc_vec, c_name) != 0) {
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * emit_mma / emit_wmma full lowerer dispatch -- ported in lower_llvm.
 *
 * The Python ISABackend.emit_mma rebuilds the legacy ISA-named op
 *   Op(name="tile.<op_id>", operands=..., results=..., attrs=...-{op_id}, loc=...)
 * and dispatches it through lowerer.lower_op (CDNA MFMA) or self.emit_wmma
 * (RDNA). emit_wmma itself drives the live lowerer: lowerer._need(decl_key),
 * lowerer._operand(v), lowerer._fresh("wmma_a"), lowerer._current().emit(...).
 *
 * Because that path is lowerer-driven, it lives with the lowerer state
 * (ckc_lower_t), not behind this stateless backend-fact API. It is fully
 * ported in src/lower_llvm_lower_llvm_mma.c.c as the `_op_tile_mma` op_id
 * router (registered for CKC_OP_TILE_MMA): it reads op->attrs "op_id" via
 * ckc_attr_get_str and dispatches each CDNA atom to its file-static
 * `_op_tile_mfma_*` handler (which drive ckc_ll_need / ckc_ll_operand /
 * ckc_ll_fresh / ckc_ll_emit -- the C analogues of _need/_operand/_fresh/
 * _current().emit), reproducing the historical MFMA text byte-for-byte. The
 * Python NotImplementedError raised when a WMMA op_id reaches a CDNA target is
 * mirrored there by `_op_tile_wmma_*` handlers that ckc_ll_fail(CKC_ERR_NOTIMPL)
 * with the same message, and an unknown op_id likewise maps to CKC_ERR_NOTIMPL.
 *
 * What is self-contained and therefore lives HERE -- the byte-identical WMMA
 * call text and its bf16 bitcast prologue given SSA operand name strings -- is
 * ckc_isa_emit_wmma_call / ckc_isa_emit_wmma_int_call above, and the full
 * op.name -> spec resolution is ckc_isa_resolve_wmma / the lookup functions.
 * --------------------------------------------------------------------------- */
