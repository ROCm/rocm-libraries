/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_frag_emit.c -- C-side emitter for the attention
 * backward fragment and block-GEMM helpers. Builds, identically to the Python
 * emitter attention_bwd_frag_emit.py, a probe kernel per config that calls
 * the helpers of helper_rocke.instances.common._attention_bwd_frag.h and
 * _attention_bwd_gemm.h, lowers it via rocke_lower_kernel_to_llvm (flavor
 * AUTO) and prints the .ll so the two outputs can be byte-compared.
 *
 * argv[1] is the config index: the position in the table built by
 * build_table(), which lists the configs in the order of CONFIGS in the Python
 * emitter (`attention_bwd_frag_emit.py list` prints them). Optional argv[2]
 * selects the output mode: "ll" (default), "ir" or "verify".
 *
 * The caps answers the Python helpers take from _attention_bwd_caps (atom,
 * wave size, legal waves, asm mnemonic, tail nop, lead wait, transpose read)
 * are filled by fill_caps() below, the explicit input of the C API.
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define RT INT_MIN /* the runtime parameter ("rt") */
#define NONE (-1) /* Python None of a non-negative int knob */

/* ------------------------------------------------------------------ caps */

static const int WAVES_MFMA[] = {1, 2, 4, 8};
static const int WAVES_WMMA[] = {1, 2, 4};

static bool is_mfma(const char* arch)
{
    return strcmp(arch, "gfx942") == 0 || strcmp(arch, "gfx950") == 0;
}

static bool known_arch(const char* arch)
{
    return is_mfma(arch) || strcmp(arch, "gfx1151") == 0 || strcmp(arch, "gfx1201") == 0;
}

/* What _attention_bwd_caps answers for (arch, dtype, atom_k). */
static void
    fill_caps(const char* arch, const char* dtype, int atom_k, rocke_attn_bwd_gemm_caps_t* caps)
{
    memset(caps, 0, sizeof(*caps));
    if(!known_arch(arch))
        return;
    bool mfma = is_mfma(arch);
    caps->wave_size = mfma ? 64 : 32;
    caps->legal_waves = mfma ? WAVES_MFMA : WAVES_WMMA;
    caps->n_legal_waves = mfma ? 4 : 3;
    caps->asm_tail_nop = 15; /* AGPR_ASM_TAIL_NOP (16 and 32) */
    caps->asm_lead_wait = 2; /* ASM_MFMA_VALU_LEAD_WAIT_STATES */
    caps->has_tr_read_b64 = strcmp(arch, "gfx950") == 0;
    /* select_bwd_atom: dtype normalised (fp16 / f16 / bf16) */
    const char* dt = NULL;
    if(strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0)
        dt = "f16";
    else if(strcmp(dtype, "bf16") == 0)
        dt = "bf16";
    const char* op_id = NULL;
    char buf[64];
    if(dt != NULL)
    {
        if(mfma && (atom_k == 16 || (atom_k == 32 && strcmp(arch, "gfx950") == 0)))
        {
            snprintf(buf, sizeof(buf), "mfma_f32_16x16x%d_%s", atom_k, dt);
            op_id = buf;
        }
        else if(strcmp(arch, "gfx1151") == 0 && atom_k == 16)
        {
            snprintf(buf, sizeof(buf), "wmma_f32_16x16x16_%s", dt);
            op_id = buf;
        }
        else if(strcmp(arch, "gfx1201") == 0 && atom_k == 16)
        {
            snprintf(buf, sizeof(buf), "wmma_gfx12_f32_16x16x16_%s", dt);
            op_id = buf;
        }
    }
    if(op_id != NULL)
    {
        const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
        caps->op = t ? rocke_mma_catalog_by_op_id(&t->mma, op_id) : NULL;
    }
    /* mfma_asm_mnemonic: keyed on the raw dtype ("fp16" / "bf16") */
    if(mfma && (atom_k == 16 || atom_k == 32)
       && (strcmp(dtype, "fp16") == 0 || strcmp(dtype, "bf16") == 0)
       && !(atom_k == 32 && strcmp(arch, "gfx950") != 0))
    {
        static char mn[4][40];
        int slot = (atom_k == 32) * 2 + (strcmp(dtype, "bf16") == 0);
        snprintf(mn[slot],
                 sizeof(mn[slot]),
                 "v_mfma_f32_16x16x%d_%s",
                 atom_k,
                 strcmp(dtype, "bf16") == 0 ? "bf16" : "f16");
        caps->asm_mnemonic = mn[slot];
    }
}

/* ---------------------------------------------------------------- configs */

enum
{
    K_FACTS,
    K_GEMM,
    K_RELABEL,
    K_VIEWS
};

typedef struct
{
    int atoms[2];
    int n_atoms;
    int pairs[4][2];
    int n_pairs;
    int tr16_k[4];
    char tr16_role[4];
    int n_tr16;
} facts_t;

typedef struct
{
    int m, n, k, gr, gc, atom_k;
    const char* a; /* lds | regs_all | regs_ops */
    const char* bm; /* inner | outer | tr16 | tr16_inner */
    const char* mem; /* alloc | view | xor | dma | shift */
    int pad;
    bool asm_route;
    bool cluster_gemm;
    bool zero;
    int tail_nop, agpr_base, lead_wait; /* NONE: None */
    bool pre, extra;
    int slice_k; /* NONE: no slicing */
    int slice_buf;
    bool slice_first;
    bool b_all; /* B as registers via load_all_b */
    int src_mn0, src_k0, src_align; /* LdsOperand mn0 / k0 / align of both */
    bool tid; /* lane_and_wave(tid=param "tid") */
    int asm_steps; /* 0: none; 1: mma_agpr_asm_steps(load_k_slice); 2: bad */
    int first_s; /* slice index of `first` */
} gemm_t;

typedef struct
{
    int km, kn, d, waves, k02, k13;
    const char* cast;
    bool tr16;
    bool asm_route;
    int g1r, g1c; /* NONE: (waves, 1) */
} relabel_t;

typedef struct
{
    char role;
    int atom_k;
    const char* storage; /* k_inner | k_outer | tr16 */
    int mn0, k0, align;
} vfrag_t;

typedef struct
{
    const char* view; /* plain | f32 | dma | xor */
    int base, pitch, rows, scale;
    bool shift;
    int chunk;
    int ns[4];
    int n_ns;
    bool tr16;
    vfrag_t frag[2];
    int n_frag;
} views_t;

typedef struct
{
    int kind;
    const char* arch;
    const char* dtype;
    facts_t f;
    gemm_t g;
    relabel_t r;
    views_t v;
} cfg_t;

static cfg_t base_cfg(int kind, const char* arch, const char* dtype)
{
    cfg_t c;
    memset(&c, 0, sizeof(c));
    c.kind = kind;
    c.arch = arch;
    c.dtype = dtype;
    return c;
}

static cfg_t FACTS(const char* arch, const char* dtype, int a0, int a1)
{
    cfg_t c = base_cfg(K_FACTS, arch, dtype);
    c.f.atoms[0] = a0;
    c.f.atoms[1] = a1;
    c.f.n_atoms = a1 ? 2 : 1;
    return c;
}

static void PAIR(cfg_t* c, int ck, int ak)
{
    c->f.pairs[c->f.n_pairs][0] = ck;
    c->f.pairs[c->f.n_pairs][1] = ak;
    c->f.n_pairs++;
}

static void TR(cfg_t* c, int k, char role)
{
    c->f.tr16_k[c->f.n_tr16] = k;
    c->f.tr16_role[c->f.n_tr16] = role;
    c->f.n_tr16++;
}

static cfg_t
    GEMM(const char* arch, const char* dtype, int m, int n, int k, int gr, int gc, int atom_k)
{
    cfg_t c = base_cfg(K_GEMM, arch, dtype);
    gemm_t* g = &c.g;
    g->m = m;
    g->n = n;
    g->k = k;
    g->gr = gr;
    g->gc = gc;
    g->atom_k = atom_k;
    g->a = "lds";
    g->bm = "inner";
    g->mem = "alloc";
    g->zero = true;
    g->tail_nop = g->agpr_base = g->lead_wait = NONE;
    g->slice_k = NONE;
    g->src_align = 8;
    return c;
}

static void SLICE(cfg_t* c, int k_slice, int buffers, bool first)
{
    c->g.slice_k = k_slice;
    c->g.slice_buf = buffers;
    c->g.slice_first = first;
}

static void SRC(cfg_t* c, int mn0, int k0, int align)
{
    c->g.src_mn0 = mn0;
    c->g.src_k0 = k0;
    c->g.src_align = align;
}

static cfg_t
    RELABEL(const char* arch, const char* dtype, int km, int kn, int d, int w, int k02, int k13)
{
    cfg_t c = base_cfg(K_RELABEL, arch, dtype);
    relabel_t* r = &c.r;
    r->km = km;
    r->kn = kn;
    r->d = d;
    r->waves = w;
    r->k02 = k02;
    r->k13 = k13;
    r->cast = "generic";
    r->g1r = r->g1c = NONE;
    return c;
}

static cfg_t
    VIEWS(const char* arch, const char* dtype, const char* view, int base, int pitch, int rows)
{
    cfg_t c = base_cfg(K_VIEWS, arch, dtype);
    views_t* v = &c.v;
    v->view = view;
    v->base = base;
    v->pitch = pitch;
    v->rows = rows;
    v->scale = 1;
    v->chunk = 4;
    v->ns[0] = 1;
    v->ns[1] = 2;
    v->ns[2] = 4;
    v->ns[3] = 8;
    v->n_ns = 4;
    return c;
}

static void NS(cfg_t* c, int n0, int n1, int n2)
{
    int ns[3] = {n0, n1, n2};
    c->v.n_ns = 0;
    for(int i = 0; i < 3; ++i)
        if(ns[i])
            c->v.ns[c->v.n_ns++] = ns[i];
}

static void VFRAG(cfg_t* c, char role, int atom_k, const char* storage, int mn0, int k0, int align)
{
    vfrag_t* f = &c->v.frag[c->v.n_frag++];
    f->role = role;
    f->atom_k = atom_k;
    f->storage = storage;
    f->mn0 = mn0;
    f->k0 = k0;
    f->align = align;
}

#define MAX_CFG 128

/* Same order as CONFIGS in the Python emitter. */
static int build_table(cfg_t* t)
{
    int n = 0;
    cfg_t c;
    /* --- facts --- */
    c = FACTS("gfx942", "fp16", 16, 0);
    PAIR(&c, 16, 16);
    t[n++] = c;
    c = FACTS("gfx942", "bf16", 16, 0);
    PAIR(&c, 16, 16);
    t[n++] = c;
    c = FACTS("gfx950", "fp16", 16, 32);
    PAIR(&c, 16, 16);
    PAIR(&c, 16, 32);
    PAIR(&c, 32, 32);
    PAIR(&c, 32, 16);
    TR(&c, 16, 'a');
    TR(&c, 16, 'b');
    TR(&c, 32, 'a');
    TR(&c, 32, 'b');
    t[n++] = c;
    c = FACTS("gfx950", "bf16", 16, 32);
    PAIR(&c, 16, 32);
    PAIR(&c, 32, 32);
    TR(&c, 32, 'b');
    t[n++] = c;
    c = FACTS("gfx1151", "fp16", 16, 0);
    PAIR(&c, 16, 16);
    t[n++] = c;
    c = FACTS("gfx1201", "bf16", 16, 0);
    PAIR(&c, 16, 16);
    t[n++] = c;
    /* --- gemm --- */
    t[n++] = GEMM("gfx942", "fp16", 16, 32, 128, 1, 2, 16);
    c = GEMM("gfx942", "bf16", 32, 128, 16, 2, 1, 16);
    c.g.bm = "outer";
    c.g.asm_route = true;
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 16, 128, 64, 1, 4, 16);
    c.g.bm = "outer";
    c.g.mem = "view";
    c.g.pad = 8;
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 32, 128, 128, 1, 4, 32);
    c.g.mem = "view";
    c.g.pad = 8;
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 128, 128, 32, 4, 1, 16);
    c.g.bm = "tr16";
    c.g.mem = "view";
    c.g.asm_route = true;
    c.g.agpr_base = 128;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 32, 64, 64, 1, 4, 32);
    c.g.bm = "tr16";
    c.g.mem = "view";
    c.g.asm_route = true;
    c.g.cluster_gemm = true;
    c.g.zero = false;
    c.g.pre = true;
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 32, 32, 32, 2, 1, 16);
    c.g.a = "regs_all";
    c.g.extra = true;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 32, 64, 64, 2, 2, 32);
    c.g.a = "regs_ops";
    c.g.bm = "outer";
    c.g.extra = true;
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 64, 64, 32, 2, 2, 16);
    c.g.mem = "view";
    c.g.pad = 4;
    c.g.extra = true;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 32, 64, 64, 1, 2, 16);
    c.g.mem = "xor";
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 32, 64, 64, 1, 2, 32);
    c.g.mem = "dma";
    t[n++] = c;
    c = GEMM("gfx942", "bf16", 16, 64, 64, 1, 4, 16);
    c.g.mem = "dma";
    c.g.bm = "outer";
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 16, 32, 64, 1, 2, 16);
    c.g.mem = "shift";
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 16, 16, 64, 1, 1, 16);
    c.g.asm_route = true;
    c.g.tail_nop = 6;
    c.g.lead_wait = 0;
    c.g.zero = false;
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 16, 16, 64, 1, 1, 32);
    c.g.asm_route = true;
    c.g.cluster_gemm = true;
    c.g.lead_wait = 4;
    c.g.tail_nop = 0;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 32, 128, 64, 1, 4, 16);
    c.g.asm_route = true;
    c.g.pre = true;
    c.g.extra = true;
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 16, 64, 64, 1, 2, 16);
    c.g.bm = "outer";
    SLICE(&c, 32, 1, false);
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 32, 128, 128, 1, 4, 32);
    c.g.bm = "tr16";
    c.g.mem = "view";
    SLICE(&c, 64, 2, false);
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 32, 64, 128, 1, 4, 32);
    c.g.bm = "outer";
    SLICE(&c, 32, 2, true);
    t[n++] = c;
    c = GEMM("gfx942", "bf16", 16, 32, 64, 1, 2, 16);
    SLICE(&c, 16, 1, true);
    c.g.extra = true;
    t[n++] = c;
    t[n++] = GEMM("gfx1151", "fp16", 16, 64, 64, 1, 4, 16);
    c = GEMM("gfx1201", "bf16", 32, 32, 32, 2, 1, 16);
    c.g.bm = "outer";
    c.g.extra = true;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 16, 16, 64, 1, 1, 32);
    c.g.asm_route = true;
    c.g.cluster_gemm = true;
    t[n++] = c;
    c = GEMM("gfx942", "bf16", 32, 32, 64, 2, 1, 16);
    c.g.bm = "outer";
    c.g.b_all = true;
    c.g.extra = true;
    t[n++] = c;
    /* gfx950 8 waves: G0 (1, 8), G1 (8, 1) tr16, G4 (2, 4) sliced */
    c = GEMM("gfx950", "bf16", 32, 128, 64, 1, 8, 32);
    c.g.mem = "view";
    c.g.pad = 8;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 128, 64, 32, 8, 1, 16);
    c.g.bm = "tr16";
    c.g.mem = "view";
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 32, 128, 256, 2, 4, 32);
    c.g.bm = "outer";
    SLICE(&c, 64, 2, false);
    t[n++] = c;
    /* gfx950 16 x 192 tile */
    c = GEMM("gfx950", "fp16", 16, 192, 128, 1, 4, 32);
    c.g.mem = "view";
    c.g.pad = 8;
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 192, 128, 16, 4, 1, 16);
    c.g.bm = "tr16";
    c.g.mem = "view";
    c.g.asm_route = true;
    c.g.agpr_base = 64;
    t[n++] = c;
    /* gfx942 D 128, 16 x 128 tile, 4 waves: G1 asm pinned */
    c = GEMM("gfx942", "fp16", 128, 128, 16, 4, 1, 16);
    c.g.bm = "outer";
    c.g.asm_route = true;
    c.g.agpr_base = 128;
    t[n++] = c;
    /* LdsOperand mn0 / k0 / align */
    c = GEMM("gfx942", "fp16", 16, 32, 32, 1, 2, 16);
    SRC(&c, 16, 16, 4);
    c.g.extra = true;
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 32, 64, 64, 1, 2, 32);
    c.g.bm = "outer";
    c.g.mem = "view";
    SRC(&c, 8, 32, 2);
    SLICE(&c, 32, 1, true);
    t[n++] = c;
    /* caller tid; mma_agpr_asm_steps fed by load_k_slice */
    c = GEMM("gfx942", "bf16", 16, 32, 64, 1, 2, 16);
    c.g.tid = true;
    c.g.asm_steps = 1;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 32, 64, 64, 2, 2, 32);
    c.g.bm = "tr16";
    c.g.mem = "view";
    c.g.asm_route = true;
    c.g.zero = false;
    c.g.asm_steps = 1;
    t[n++] = c;
    /* --- relabel --- */
    t[n++] = RELABEL("gfx942", "fp16", 16, 32, 128, 2, 16, 16);
    c = RELABEL("gfx942", "bf16", 16, 64, 64, 4, 16, 16);
    c.r.cast = "rne_noguard";
    c.r.asm_route = true;
    t[n++] = c;
    c = RELABEL("gfx950", "bf16", 32, 128, 128, 4, 32, 16);
    c.r.tr16 = true;
    c.r.asm_route = true;
    t[n++] = c;
    t[n++] = RELABEL("gfx950", "fp16", 32, 64, 64, 4, 16, 32);
    c = RELABEL("gfx950", "bf16", 32, 64, 64, 2, 32, 32);
    c.r.tr16 = true;
    c.r.cast = "rne_noguard";
    t[n++] = c;
    c = RELABEL("gfx950", "fp16", 32, 128, 64, 4, 16, 32);
    c.r.tr16 = true;
    c.r.asm_route = true;
    t[n++] = c;
    t[n++] = RELABEL("gfx1151", "fp16", 16, 32, 64, 2, 16, 16);
    c = RELABEL("gfx1151", "bf16", 16, 64, 32, 4, 16, 16);
    c.r.cast = "rne_noguard";
    t[n++] = c;
    t[n++] = RELABEL("gfx1201", "bf16", 16, 32, 64, 2, 16, 16);
    /* gfx950 8 waves (32 x 128, 32 x 256), 16 x 192 asm; gfx942 16 x 128 asm */
    c = RELABEL("gfx950", "bf16", 32, 128, 128, 8, 32, 16);
    c.r.tr16 = true;
    t[n++] = c;
    c = RELABEL("gfx950", "fp16", 32, 256, 64, 8, 32, 16);
    c.r.tr16 = true;
    t[n++] = c;
    c = RELABEL("gfx950", "bf16", 16, 192, 128, 4, 32, 16);
    c.r.tr16 = true;
    c.r.asm_route = true;
    t[n++] = c;
    c = RELABEL("gfx942", "fp16", 16, 128, 128, 4, 16, 16);
    c.r.asm_route = true;
    t[n++] = c;
    /* --- views --- */
    c = VIEWS("gfx950", "fp16", "plain", 0, 72, 32);
    VFRAG(&c, 'a', 16, "k_inner", 0, 0, 8);
    t[n++] = c;
    c = VIEWS("gfx942", "bf16", "plain", 64, 36, 16);
    VFRAG(&c, 'b', 16, "k_outer", RT, 16, 8);
    VFRAG(&c, 'a', 16, "k_inner", 16, RT, 4);
    t[n++] = c;
    c = VIEWS("gfx950", "fp16", "f32", 128, 64, 16);
    c.v.scale = 2;
    NS(&c, 1, 2, 4);
    t[n++] = c;
    c = VIEWS("gfx950", "bf16", "plain", 256, 128, 32);
    c.v.shift = true;
    c.v.tr16 = true;
    t[n++] = c;
    c = VIEWS("gfx950", "bf16", "dma", 512, 128, 32);
    NS(&c, 1, 4, 8);
    c.v.tr16 = true;
    VFRAG(&c, 'b', 32, "tr16", 0, RT, 0);
    VFRAG(&c, 'a', 32, "k_inner", RT, 0, 8);
    t[n++] = c;
    c = VIEWS("gfx950", "fp16", "dma", 0, 32, 16);
    NS(&c, 1, 2, 8);
    t[n++] = c;
    c = VIEWS("gfx942", "fp16", "dma", 64, 64, 8);
    NS(&c, 8, 0, 0);
    c.v.shift = true;
    t[n++] = c;
    c = VIEWS("gfx950", "fp16", "xor", 0, 64, 32);
    c.v.chunk = 8;
    c.v.tr16 = true;
    t[n++] = c;
    c = VIEWS("gfx950", "bf16", "xor", 128, 32, 64);
    c.v.chunk = 4;
    NS(&c, 1, 4, 8);
    t[n++] = c;
    c = VIEWS("gfx942", "fp16", "xor", 64, 128, 16);
    c.v.chunk = 2;
    NS(&c, 2, 8, 0);
    t[n++] = c;
    c = VIEWS("gfx950", "fp16", "plain", 0, 64, 64);
    c.v.tr16 = true;
    VFRAG(&c, 'b', 16, "tr16", RT, 4, 0);
    VFRAG(&c, 'b', 32, "tr16", 8, 0, 0);
    t[n++] = c;
    c = VIEWS("gfx1151", "fp16", "plain", 0, 40, 16);
    VFRAG(&c, 'a', 16, "k_inner", 0, 16, 8);
    t[n++] = c;
    /* --- rejected --- */
    c = GEMM("gfx942", "fp16", 16, 32, 64, 1, 2, 16);
    c.g.bm = "tr16";
    c.g.mem = "view";
    t[n++] = c;
    t[n++] = GEMM("gfx950", "fp16", 32, 64, 64, 4, 2, 16);
    t[n++] = GEMM("gfx942", "bf16", 16, 32, 64, 1, 2, 32);
    c = GEMM("gfx1151", "fp16", 16, 32, 32, 1, 2, 16);
    c.g.asm_route = true;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 64, 64, 32, 1, 1, 16);
    c.g.asm_route = true;
    c.g.agpr_base = 250;
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 32, 64, 128, 1, 4, 32);
    SLICE(&c, 16, 1, false);
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 16, 32, 64, 1, 2, 16);
    c.g.tail_nop = 7;
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 16, 32, 64, 1, 2, 16);
    c.g.asm_route = true;
    c.g.tail_nop = 16;
    t[n++] = c;
    c = RELABEL("gfx950", "fp16", 32, 64, 64, 4, 16, 16);
    c.r.g1r = 2;
    c.r.g1c = 2;
    t[n++] = c;
    c = VIEWS("gfx950", "fp16", "xor", 0, 64, 32);
    c.v.shift = true;
    t[n++] = c;
    t[n++] = VIEWS("gfx950", "fp16", "dma", 0, 36, 32);
    c = VIEWS("gfx950", "fp16", "plain", 0, 64, 64);
    VFRAG(&c, 'b', 16, "tr16", 2, 0, 0);
    t[n++] = c;
    t[n++] = GEMM("gfx942", "fp32", 16, 32, 64, 1, 2, 16);
    c = GEMM("gfx950", "fp16", 32, 64, 64, 1, 2, 16);
    c.g.bm = "tr16_inner";
    c.g.mem = "view";
    t[n++] = c;
    c = GEMM("gfx950", "bf16", 32, 64, 64, 2, 2, 32);
    c.g.asm_steps = 2;
    t[n++] = c;
    c = GEMM("gfx950", "fp16", 32, 64, 64, 1, 2, 16);
    c.g.asm_route = true;
    c.g.lead_wait = 17;
    t[n++] = c;
    c = GEMM("gfx942", "fp16", 16, 32, 64, 1, 2, 16);
    SLICE(&c, 32, 1, true);
    c.g.first_s = 2;
    t[n++] = c;
    return n;
}

/* ------------------------------------------------------------- the probe */

typedef struct
{
    rocke_ir_builder_t* b;
    rocke_value_t* oi;
    rocke_value_t* of;
    int ni, nf;
} out_t;

static int64_t wrap32(int64_t v)
{
    v &= 0xFFFFFFFFLL;
    return v >= (1LL << 31) ? v - (1LL << 32) : v;
}

/* _Out.i of a Value */
static void out_iv(out_t* o, rocke_value_t* v)
{
    rocke_value_t* idx = rocke_b_const_i32(o->b, o->ni++);
    rocke_b_global_store(o->b, o->oi, idx, v, 1);
}

/* _Out.i of a host int */
static void out_ii(out_t* o, int64_t x)
{
    rocke_value_t* idx = rocke_b_const_i32(o->b, o->ni++);
    rocke_value_t* v = rocke_b_const_i32(o->b, wrap32(x));
    rocke_b_global_store(o->b, o->oi, idx, v, 1);
}

/* _Out.f */
static void out_f(out_t* o, rocke_value_t* v)
{
    rocke_ir_builder_t* b = o->b;
    rocke_value_t* idx = rocke_b_const_i32(b, o->nf++);
    if(v == NULL || b->status != ROCKE_OK)
        return;
    if(v->type->kind == ROCKE_TYPE_VECTOR)
        v = rocke_b_vec_extract(b, v, 0);
    v = rocke_b_cast_to_f32(b, v);
    rocke_b_global_store(b, o->of, idx, v, 1);
}

static const rocke_type_t* elem_of(const char* dtype)
{
    return strcmp(dtype, "bf16") == 0 ? rocke_bf16() : rocke_f16();
}

static int align64(int x)
{
    return (x + 63) / 64 * 64;
}

static rocke_attn_bwd_iv_t ivx(int x, rocke_value_t* rt)
{
    return x == RT ? rocke_attn_bwd_iv_val(rt) : rocke_attn_bwd_iv_int(x);
}

static void coords_vals(out_t* o, const rocke_attn_bwd_frag_coords_t* fc)
{
    for(int s = 0; s < fc->frag_len; ++s)
    {
        out_ii(o, fc->deltas[s][0]);
        out_ii(o, fc->deltas[s][1]);
    }
    int lanes[3] = {0, 17, fc->wave_size - 1};
    int cc[2 * ROCKE_ATTN_BWD_FRAG_MAX_SLOTS];
    for(int i = 0; i < 3; ++i)
    {
        rocke_attn_bwd_frag_coords_int_coords(fc, lanes[i], cc);
        for(int s = 0; s < fc->frag_len; ++s)
            out_ii(o, (int64_t)cc[2 * s] * 1000 + cc[2 * s + 1]);
    }
    out_ii(o, fc->kperm);
}

static void plan_vals(out_t* o, const rocke_attn_bwd_relabel_plan_t* p)
{
    out_ii(o, p->kind == ROCKE_ATTN_BWD_RELABEL_XLANE16);
    out_ii(o, p->c_tiles);
    out_ii(o, p->c_tile_rows);
    if(p->kind == ROCKE_ATTN_BWD_RELABEL_DIRECT)
        for(int s = 0; s < p->n_slot_src; ++s)
            out_ii(o, p->slot_src[s][0] * 100 + p->slot_src[s][1]);
    for(int m = 0; m < p->n_xlane; ++m)
    {
        out_ii(o, p->xlane[m][0]);
        out_ii(o, p->xlane[m][1]);
        out_ii(o, p->xlane[m][2]);
    }
    for(int k = 0; k < p->n_k_perm; ++k)
        out_ii(o, p->k_perm[k]);
    out_ii(o, rocke_attn_bwd_relabel_plan_needs_k_perm(p));
}

static void tr16_vals(out_t* o, const rocke_attn_bwd_tr16_read_plan_t* p)
{
    out_ii(o, p->kind == ROCKE_ATTN_BWD_TR16_STACKED);
    out_ii(o, p->g_stride);
    for(int t = 0; t < p->n_offsets; ++t)
        out_ii(o, p->k_offsets[t]);
}

static const rocke_mma_op_t* atom(const cfg_t* c, int k)
{
    rocke_attn_bwd_gemm_caps_t caps;
    fill_caps(c->arch, c->dtype, k, &caps);
    return caps.op;
}

static void probe_facts(rocke_ir_builder_t* b, out_t* o, const cfg_t* c, rocke_value_t* lane)
{
    const facts_t* f = &c->f;
    for(int ai = 0; ai < f->n_atoms; ++ai)
    {
        const rocke_mma_op_t* op = atom(c, f->atoms[ai]);
        rocke_attn_bwd_atom_facts_t fa;
        if(!rocke_attn_bwd_atom_facts(b, op, &fa))
            return;
        out_ii(o, fa.a_contig);
        out_ii(o, fa.b_contig);
        out_ii(o, fa.affine);
        out_ii(o, fa.c_is_aT);
        out_ii(o, fa.ab_same_k);
        int runs[4] = {1, 2, 4, 8};
        for(int i = 0; i < 4; ++i)
        {
            bool r;
            if(!rocke_attn_bwd_c_row_runs(b, op, runs[i], &r))
                return;
            out_ii(o, r);
        }
        bool q;
        if(!rocke_attn_bwd_c_quad_transposable(b, op, &q))
            return;
        out_ii(o, q);
        const char roles[2] = {'a', 'b'};
        for(int ri = 0; ri < 2; ++ri)
        {
            rocke_attn_bwd_frag_coords_t fc;
            if(!rocke_attn_bwd_operand_coords(b, op, roles[ri], &fc))
                return;
            coords_vals(o, &fc);
            rocke_value_t *mn, *kk;
            if(!rocke_attn_bwd_frag_coords_emit_base(b, &fc, lane, &mn, &kk))
                return;
            out_iv(o, mn);
            out_iv(o, kk);
        }
    }
    for(int pi = 0; pi < f->n_pairs; ++pi)
    {
        const rocke_mma_op_t* c_op = atom(c, f->pairs[pi][0]);
        const rocke_mma_op_t* a_op = atom(c, f->pairs[pi][1]);
        rocke_attn_bwd_relabel_plan_t plan;
        if(!rocke_attn_bwd_plan_c_to_a(b, c_op, a_op, &plan))
            return;
        plan_vals(o, &plan);
        int cap = a_op->wave_size * a_op->a_frag_len * 2;
        int* sim = (int*)calloc((size_t)cap, sizeof(int));
        if(!rocke_attn_bwd_simulate_relabel(b, &plan, c_op, a_op, sim, cap))
        {
            free(sim);
            return;
        }
        int64_t acc = 0;
        for(int li = 0; li < a_op->wave_size; ++li)
            for(int si = 0; si < a_op->a_frag_len; ++si)
            {
                int r = sim[(li * a_op->a_frag_len + si) * 2];
                int kk = sim[(li * a_op->a_frag_len + si) * 2 + 1];
                acc = (acc * 31 + li * 7 + si * 5 + r * 3 + kk) & 0x7FFFFFFF;
            }
        free(sim);
        out_ii(o, acc);
        if(rocke_attn_bwd_relabel_plan_needs_k_perm(&plan))
        {
            rocke_attn_bwd_frag_coords_t kc;
            if(!rocke_attn_bwd_kperm_b_coords(b, &plan, c_op, a_op, &kc))
                return;
            coords_vals(o, &kc);
            rocke_value_t *mn, *kk;
            if(!rocke_attn_bwd_frag_coords_emit_base(b, &kc, lane, &mn, &kk))
                return;
            out_iv(o, mn);
            out_iv(o, kk);
            if(kc.wave_size == 64)
            {
                rocke_attn_bwd_tr16_read_plan_t tp;
                if(!rocke_attn_bwd_tr16_read_plan(b, &kc, &tp))
                    return;
                tr16_vals(o, &tp);
            }
        }
    }
    for(int ti = 0; ti < f->n_tr16; ++ti)
    {
        rocke_attn_bwd_frag_coords_t fc;
        if(!rocke_attn_bwd_operand_coords(b, atom(c, f->tr16_k[ti]), f->tr16_role[ti], &fc))
            return;
        int kpl = rocke_attn_bwd_check_tr16_coords(b, &fc);
        if(kpl == 0)
            return;
        out_ii(o, kpl);
        rocke_attn_bwd_tr16_read_plan_t tp;
        if(!rocke_attn_bwd_tr16_read_plan(b, &fc, &tp))
            return;
        tr16_vals(o, &tp);
    }
    int64_t acc = 0;
    int mc[2 * 8];
    for(int kpl = 4; kpl <= 8; kpl += 4)
        for(int ln = 0; ln < 64; ++ln)
        {
            for(int t = 0; t < kpl / 4; ++t)
            {
                int kk, mn;
                rocke_attn_bwd_tr16_lane_address(ln, t, kpl, &kk, &mn);
                acc = (acc * 17 + kk * 3 + mn) & 0x7FFFFFFF;
            }
            rocke_attn_bwd_tr16_model_coords(ln, kpl, mc);
            for(int s = 0; s < kpl; ++s)
                acc = (acc * 13 + mc[2 * s] * 5 + mc[2 * s + 1]) & 0x7FFFFFFF;
        }
    out_ii(o, acc);
}

/* _mem_view */
static bool mem_view(rocke_ir_builder_t* b,
                     rocke_attn_bwd_lds_view_t* v,
                     const char* mem,
                     rocke_value_t* smem,
                     int base,
                     int pitch,
                     int rows,
                     rocke_value_t* shift)
{
    if(strcmp(mem, "xor") == 0)
        return rocke_attn_bwd_xor_lds_view_init(v, b, smem, base, pitch, rows, 1, NULL, 4);
    if(strcmp(mem, "dma") == 0)
        return rocke_attn_bwd_dma_slab_view_init(v, b, smem, base, pitch, rows, 1, NULL);
    return rocke_attn_bwd_lds_view_init(
        v, b, smem, base, pitch, rows, 1, strcmp(mem, "shift") == 0 ? shift : NULL);
}

typedef struct
{
    out_t* o;
    rocke_value_t* lane;
} pre_ctx_t;

static void pre_issue(rocke_ir_builder_t* b, void* ctx)
{
    (void)b;
    pre_ctx_t* p = (pre_ctx_t*)ctx;
    out_iv(p->o, p->lane);
}

static void out_frags(out_t* o, rocke_value_t** v, int n)
{
    for(int i = 0; i < n; ++i)
        out_f(o, v[i]);
}

static void probe_gemm(
    rocke_ir_builder_t* b, out_t* o, const cfg_t* c, rocke_value_t* lane, rocke_value_t* wave)
{
    const gemm_t* f = &c->g;
    const rocke_type_t* elem = elem_of(c->dtype);
    rocke_attn_bwd_gemm_caps_t caps;
    fill_caps(c->arch, c->dtype, f->atom_k, &caps);
    rocke_attn_bwd_block_gemm_t g;
    if(!rocke_attn_bwd_block_gemm_init(
           &g, b, c->arch, c->dtype, f->m, f->n, f->k, f->gr, f->gc, f->atom_k, &caps))
        return;
    rocke_value_t* origin[2];
    rocke_attn_bwd_block_gemm_wave_origin(b, &g, wave, origin);
    int m = f->m, n = f->n, k = f->k;
    bool inner = strcmp(f->bm, "inner") == 0 || strcmp(f->bm, "tr16_inner") == 0;
    rocke_attn_bwd_storage_t b_storage = inner ? ROCKE_ATTN_BWD_K_INNER : ROCKE_ATTN_BWD_K_OUTER;
    int b_rows = inner ? n : k, b_cols = inner ? k : n;
    rocke_attn_bwd_lds_view_t va, vb;
    rocke_attn_bwd_lds_operand_t a_src = rocke_attn_bwd_lds_operand_default();
    rocke_attn_bwd_lds_operand_t b_src = rocke_attn_bwd_lds_operand_default();
    a_src.mn0 = b_src.mn0 = f->src_mn0;
    a_src.k0 = b_src.k0 = f->src_k0;
    a_src.align = b_src.align = f->src_align;
    if(strcmp(f->mem, "alloc") == 0)
    {
        int e_mn = f->src_mn0, e_k = f->src_k0;
        int sha[2] = {m + e_mn, k + e_k};
        int shb[2] = {inner ? b_rows + e_mn : b_rows + e_k, inner ? b_cols + e_k : b_cols + e_mn};
        a_src.smem = rocke_b_smem_alloc(b, elem, sha, 2, "lds_a");
        b_src.smem = rocke_b_smem_alloc(b, elem, shb, 2, "lds_b");
        b_src.storage = b_storage;
    }
    else
    {
        rocke_value_t* shift = NULL;
        if(strcmp(f->mem, "shift") == 0)
            shift = rocke_b_param(b, "shift", rocke_i32(), NULL);
        int pa = k + f->pad, pb = b_cols + f->pad;
        int base_b = align64(m * pa);
        int total = base_b + align64(b_rows * pb);
        rocke_value_t* smem = rocke_b_smem_alloc(b, elem, &total, 1, "bwd_lds");
        if(!mem_view(b, &va, f->mem, smem, 0, pa, m, shift)
           || !mem_view(b, &vb, f->mem, smem, base_b, pb, b_rows, shift))
            return;
        out_ii(o, rocke_attn_bwd_lds_view_align_elems(&va));
        out_ii(o, rocke_attn_bwd_lds_view_align_elems(&vb));
        a_src.view = &va;
        b_src.view = &vb;
        b_src.storage = b_storage;
        b_src.reader = strncmp(f->bm, "tr16", 4) == 0 ? ROCKE_ATTN_BWD_READER_TR16
                                                      : ROCKE_ATTN_BWD_READER_PLAIN;
    }
    /* No explicit rocke_attn_bwd_lds_operand_check here: the BlockGemm entry
     * points run it (the tr16_inner rejection config relies on that). */
    int fm_n = rocke_attn_bwd_block_gemm_frags_m(&g), fn_n = rocke_attn_bwd_block_gemm_frags_n(&g);
    int ksn = rocke_attn_bwd_block_gemm_k_steps(&g);
    rocke_value_t** a_regs = (rocke_value_t**)calloc((size_t)fm_n * ksn, sizeof(rocke_value_t*));
    rocke_value_t** b_regs = (rocke_value_t**)calloc((size_t)fn_n * ksn, sizeof(rocke_value_t*));
    rocke_value_t** acc = (rocke_value_t**)calloc((size_t)fm_n * fn_n, sizeof(rocke_value_t*));
    rocke_value_t** acc0 = (rocke_value_t**)calloc((size_t)fm_n * fn_n, sizeof(rocke_value_t*));
    rocke_attn_bwd_gemm_operand_t a_op = {&a_src, {NULL, 0, 0}};
    rocke_attn_bwd_gemm_operand_t b_op = {&b_src, {NULL, 0, 0}};
    rocke_attn_bwd_mma_opts_t opts = rocke_attn_bwd_mma_opts_default();
    pre_ctx_t pc = {o, lane};
    rocke_attn_bwd_frags_t acc0_f = {acc0, fm_n, fn_n};
    if(strcmp(f->a, "regs_all") == 0)
    {
        if(!rocke_attn_bwd_block_gemm_load_all_a(b, &g, &a_src, lane, origin[0], a_regs))
            goto done;
        a_op.lds = NULL;
        a_op.regs.v = a_regs;
        a_op.regs.rows = fm_n;
        a_op.regs.cols = ksn;
    }
    else if(strcmp(f->a, "regs_ops") == 0)
    {
        rocke_attn_bwd_gemm_operand_t a0 = a_op, b0 = b_op;
        if(!rocke_attn_bwd_block_gemm_load_operands(b, &g, &a0, &b0, lane, origin, a_regs, b_regs))
            goto done;
        a_op.lds = NULL;
        a_op.regs.v = a_regs;
        a_op.regs.rows = fm_n;
        a_op.regs.cols = ksn;
        b_op.lds = NULL;
        b_op.regs.v = b_regs;
        b_op.regs.rows = fn_n;
        b_op.regs.cols = ksn;
    }
    if(f->b_all)
    {
        if(!rocke_attn_bwd_block_gemm_load_all_b(b, &g, &b_src, lane, origin[1], b_regs))
            goto done;
        b_op.lds = NULL;
        b_op.regs.v = b_regs;
        b_op.regs.rows = fn_n;
        b_op.regs.cols = ksn;
    }
    if(f->asm_route)
    {
        opts.route = ROCKE_ATTN_BWD_ROUTE_AGPR_ASM;
        opts.asm_cluster
            = f->cluster_gemm ? ROCKE_ATTN_BWD_CLUSTER_GEMM : ROCKE_ATTN_BWD_CLUSTER_K_STEP;
        if(f->agpr_base != NONE)
        {
            opts.has_asm_agpr_base = true;
            opts.asm_agpr_base = f->agpr_base;
        }
        if(f->lead_wait != NONE)
        {
            opts.has_asm_lead_wait = true;
            opts.asm_lead_wait = f->lead_wait;
        }
    }
    if(f->tail_nop != NONE)
    {
        opts.has_asm_tail_nop = true;
        opts.asm_tail_nop = f->tail_nop;
    }
    if(f->pre)
    {
        opts.pre_issue = pre_issue;
        opts.pre_issue_ctx = &pc;
    }
    if(f->zero)
        rocke_attn_bwd_block_gemm_zero_acc(b, &g, acc0);
    if(!rocke_attn_bwd_block_gemm_mma(
           b, &g, f->zero ? &acc0_f : NULL, &a_op, &b_op, lane, origin, &opts, acc))
        goto done;
    out_frags(o, acc, fm_n * fn_n);
    if(f->extra)
    {
        int fm = fm_n - 1, fn = fn_n - 1;
        rocke_value_t* fr = rocke_attn_bwd_block_gemm_mma_frag(
            b, &g, acc[fm * fn_n + fn], &a_op, &b_op, fm, fn, lane, origin);
        if(fr == NULL)
            goto done;
        out_f(o, fr);
        rocke_value_t* cc[2 * 16];
        if(!rocke_attn_bwd_block_gemm_acc_coords(b, &g, lane, origin, fm, fn, cc))
            goto done;
        for(int s = 0; s < g.op->c_frag_len; ++s)
        {
            out_iv(o, cc[2 * s]);
            out_iv(o, cc[2 * s + 1]);
        }
        out_ii(o, rocke_attn_bwd_block_gemm_threads(&g));
        out_ii(o, rocke_attn_bwd_block_gemm_mma_per_wave(&g));
    }
    if(f->slice_k != NONE)
    {
        rocke_attn_bwd_kslice_t sl = {NULL, NULL, 0};
        rocke_value_t** sa = (rocke_value_t**)calloc((size_t)ksn * fm_n, sizeof(rocke_value_t*));
        rocke_value_t** sb = (rocke_value_t**)calloc((size_t)ksn * fn_n, sizeof(rocke_value_t*));
        rocke_value_t** z = (rocke_value_t**)calloc((size_t)fm_n * fn_n, sizeof(rocke_value_t*));
        rocke_value_t** acc2 = (rocke_value_t**)calloc((size_t)fm_n * fn_n, sizeof(rocke_value_t*));
        bool ok = true;
        sl.a = sa;
        sl.b = sb;
        if(f->slice_first)
            ok = rocke_attn_bwd_block_gemm_load_k_slice(
                b, &g, &a_op, &b_op, f->first_s, f->slice_k, lane, origin, &sl);
        if(ok)
        {
            rocke_attn_bwd_block_gemm_zero_acc(b, &g, z);
            rocke_attn_bwd_frags_t zf = {z, fm_n, fn_n};
            ok = rocke_attn_bwd_block_gemm_mma_k_sliced(b,
                                                        &g,
                                                        &zf,
                                                        &a_op,
                                                        &b_op,
                                                        f->slice_k,
                                                        f->slice_buf,
                                                        lane,
                                                        origin,
                                                        f->slice_first ? &sl : NULL,
                                                        acc2);
        }
        if(ok)
            out_frags(o, acc2, fm_n * fn_n);
        free(sa);
        free(sb);
        free(z);
        free(acc2);
    }
    if(f->asm_steps)
    {
        rocke_value_t** sa = (rocke_value_t**)calloc((size_t)ksn * fm_n, sizeof(rocke_value_t*));
        rocke_value_t** sb = (rocke_value_t**)calloc((size_t)ksn * fn_n, sizeof(rocke_value_t*));
        rocke_value_t** acc3 = (rocke_value_t**)calloc((size_t)fm_n * fn_n, sizeof(rocke_value_t*));
        rocke_attn_bwd_asm_step_t* st
            = (rocke_attn_bwd_asm_step_t*)calloc((size_t)ksn, sizeof(rocke_attn_bwd_asm_step_t));
        rocke_attn_bwd_kslice_t sl = {sa, sb, 0};
        if(rocke_attn_bwd_block_gemm_load_k_slice(b, &g, &a_op, &b_op, 0, g.k, lane, origin, &sl))
        {
            int ns = rocke_attn_bwd_kslice_asm_steps(&g, &sl, st);
            if(f->asm_steps == 2)
                st[0].n_a -= 1; /* py: steps[0] = (steps[0][0][:-1], steps[0][1]) */
            rocke_attn_bwd_frags_t af = {acc, fm_n, fn_n};
            if(rocke_attn_bwd_mma_agpr_asm_steps(
                   b, &g, &af, st, ns, false, 0, false, 0, caps.asm_lead_wait, acc3))
                out_frags(o, acc3, fm_n * fn_n);
        }
        free(sa);
        free(sb);
        free(acc3);
        free(st);
    }
done:
    free(a_regs);
    free(b_regs);
    free(acc);
    free(acc0);
}

static void probe_relabel(
    rocke_ir_builder_t* b, out_t* o, const cfg_t* c, rocke_value_t* lane, rocke_value_t* wave)
{
    const relabel_t* f = &c->r;
    const rocke_type_t* elem = elem_of(c->dtype);
    int km = f->km, kn = f->kn, d = f->d, w = f->waves;
    rocke_attn_bwd_gemm_caps_t c0, c1;
    fill_caps(c->arch, c->dtype, f->k02, &c0);
    fill_caps(c->arch, c->dtype, f->k13, &c1);
    rocke_attn_bwd_block_gemm_t g0, g1;
    if(!rocke_attn_bwd_block_gemm_init(&g0, b, c->arch, c->dtype, km, kn, d, 1, w, f->k02, &c0))
        return;
    int g1r = f->g1r != NONE ? f->g1r : w, g1c = f->g1r != NONE ? f->g1c : 1;
    if(!rocke_attn_bwd_block_gemm_init(&g1, b, c->arch, c->dtype, kn, d, km, g1r, g1c, f->k13, &c1))
        return;
    rocke_value_t *o0[2], *o1[2];
    rocke_attn_bwd_block_gemm_wave_origin(b, &g0, wave, o0);
    rocke_attn_bwd_block_gemm_wave_origin(b, &g1, wave, o1);
    int base_k = align64(km * d);
    int base_do = base_k + align64(kn * d);
    int total = base_do + align64(km * d);
    rocke_value_t* smem = rocke_b_smem_alloc(b, elem, &total, 1, "bwd_lds");
    rocke_attn_bwd_lds_view_t vq, vk, vdo;
    if(!rocke_attn_bwd_lds_view_init(&vq, b, smem, 0, d, km, 1, NULL)
       || !rocke_attn_bwd_lds_view_init(&vk, b, smem, base_k, d, kn, 1, NULL)
       || !rocke_attn_bwd_lds_view_init(&vdo, b, smem, base_do, d, km, 1, NULL))
        return;
    int fm0 = rocke_attn_bwd_block_gemm_frags_m(&g0), fn0 = rocke_attn_bwd_block_gemm_frags_n(&g0);
    int fm1 = rocke_attn_bwd_block_gemm_frags_m(&g1), fn1 = rocke_attn_bwd_block_gemm_frags_n(&g1);
    int ks1 = rocke_attn_bwd_block_gemm_k_steps(&g1);
    rocke_value_t** z0 = (rocke_value_t**)calloc((size_t)fm0 * fn0, sizeof(rocke_value_t*));
    rocke_value_t** s = (rocke_value_t**)calloc((size_t)fm0 * fn0, sizeof(rocke_value_t*));
    rocke_value_t** a_p = (rocke_value_t**)calloc((size_t)fm1 * ks1, sizeof(rocke_value_t*));
    rocke_value_t** z1 = (rocke_value_t**)calloc((size_t)fm1 * fn1, sizeof(rocke_value_t*));
    rocke_value_t** dv = (rocke_value_t**)calloc((size_t)fm1 * fn1, sizeof(rocke_value_t*));
    {
        rocke_attn_bwd_block_gemm_zero_acc(b, &g0, z0);
        rocke_attn_bwd_lds_operand_t lq = rocke_attn_bwd_lds_operand_default();
        rocke_attn_bwd_lds_operand_t lk = rocke_attn_bwd_lds_operand_default();
        lq.view = &vq;
        lk.view = &vk;
        rocke_attn_bwd_gemm_operand_t aq = {&lq, {NULL, 0, 0}}, bk = {&lk, {NULL, 0, 0}};
        rocke_attn_bwd_frags_t z0f = {z0, fm0, fn0};
        rocke_attn_bwd_relabel_plan_t plan, plan2;
        rocke_attn_bwd_frag_coords_t coords;
        rocke_attn_bwd_frags_t sf = {s, fm0, fn0};
        rocke_attn_bwd_lds_operand_t ldo = rocke_attn_bwd_lds_operand_default();
        rocke_attn_bwd_gemm_operand_t ap = {NULL, {a_p, fm1, ks1}}, bdo = {&ldo, {NULL, 0, 0}};
        rocke_attn_bwd_mma_opts_t opts = rocke_attn_bwd_mma_opts_default();
        rocke_attn_bwd_frags_t z1f = {z1, fm1, fn1};
        if(!rocke_attn_bwd_block_gemm_mma(b, &g0, &z0f, &aq, &bk, lane, o0, NULL, s))
            goto done;
        if(!rocke_attn_bwd_check_relabel_grids(b, &g0, &g1, &plan))
            goto done;
        if(!rocke_attn_bwd_relabel_acc_to_a(b, &g0, &g1, &sf, lane, f->cast, a_p, &plan2))
            goto done;
        plan_vals(o, &plan2);
        out_ii(o, memcmp(&plan, &plan2, sizeof(plan)) == 0);
        if(!rocke_attn_bwd_kperm_b_operand_coords(b, &plan, &g0, &g1, &coords))
            goto done;
        coords_vals(o, &coords);
        ldo.view = &vdo;
        ldo.storage = ROCKE_ATTN_BWD_K_OUTER;
        ldo.coords = &coords;
        ldo.reader = f->tr16 ? ROCKE_ATTN_BWD_READER_TR16 : ROCKE_ATTN_BWD_READER_PLAIN;
        if(!rocke_attn_bwd_lds_operand_check(b, &ldo))
            goto done;
        if(f->asm_route)
            opts.route = ROCKE_ATTN_BWD_ROUTE_AGPR_ASM;
        else
            rocke_attn_bwd_block_gemm_zero_acc(b, &g1, z1);
        if(!rocke_attn_bwd_block_gemm_mma(
               b, &g1, f->asm_route ? NULL : &z1f, &ap, &bdo, lane, o1, &opts, dv))
            goto done;
        out_frags(o, dv, fm1 * fn1);
    }
done:
    free(z0);
    free(s);
    free(a_p);
    free(z1);
    free(dv);
}

static void probe_views(rocke_ir_builder_t* b, out_t* o, const cfg_t* c, rocke_value_t* lane)
{
    const views_t* f = &c->v;
    const rocke_type_t* elem = elem_of(c->dtype);
    rocke_value_t* r_rt = rocke_b_param(b, "row", rocke_i32(), NULL);
    rocke_value_t* c_rt = rocke_b_param(b, "col", rocke_i32(), NULL);
    rocke_value_t* shift = f->shift ? rocke_b_param(b, "shift", rocke_i32(), NULL) : NULL;
    int base = f->base, pitch = f->pitch, rows = f->rows;
    int size = align64(base + pitch * rows * f->scale) + 64;
    rocke_value_t* smem = rocke_b_smem_alloc(b, elem, &size, 1, "bwd_lds");
    rocke_attn_bwd_lds_view_t v;
    bool ok;
    if(strcmp(f->view, "xor") == 0)
        ok = rocke_attn_bwd_xor_lds_view_init(&v, b, smem, base, pitch, rows, 1, shift, f->chunk);
    else if(strcmp(f->view, "dma") == 0)
        ok = rocke_attn_bwd_dma_slab_view_init(&v, b, smem, base, pitch, rows, 1, shift);
    else
        ok = rocke_attn_bwd_lds_view_init(&v, b, smem, base, pitch, rows, f->scale, shift);
    if(!ok)
        return;
    out_ii(o, rocke_attn_bwd_lds_view_align_elems(&v));
    const rocke_type_t* vdt = f->scale == 2 ? rocke_f32() : elem;
    int pts[7][2] = {{0, 0}, {1, 8}, {rows - 1, 0}, {2, RT}, {RT, 0}, {RT, 16}, {RT, RT}};
    for(int i = 0; i < 7; ++i)
    {
        rocke_value_t* x
            = rocke_attn_bwd_lds_view_index(b, &v, ivx(pts[i][0], r_rt), ivx(pts[i][1], c_rt));
        if(x == NULL)
            return;
        out_iv(o, x);
    }
    if(strcmp(f->view, "xor") == 0)
    {
        int rs[4] = {0, 1, 5, rows - 1};
        for(int i = 0; i < 4; ++i)
            out_ii(o, rocke_attn_bwd_xor_view_row_mask(&v, rs[i]));
        rocke_value_t* x
            = rocke_attn_bwd_xor_view_emit_index(b, &v, r_rt, rocke_attn_bwd_iv_int(8));
        out_iv(o, x);
    }
    if(strcmp(f->view, "dma") == 0)
    {
        int rc[3][2] = {{0, 0}, {rows - 1, pitch - 1}, {3, 17}};
        for(int i = 0; i < 3; ++i)
        {
            int64_t off;
            if(!rocke_attn_bwd_slab_offset(b, rows, pitch, rc[i][0], rc[i][1], &off))
                return;
            out_ii(o, off);
        }
    }
    for(int ni = 0; ni < f->n_ns; ++ni)
    {
        int n = f->ns[ni];
        int rc[3][2] = {{1, 0}, {RT, 8}, {2, RT}};
        for(int i = 0; i < 3; ++i)
        {
            rocke_value_t* x = rocke_attn_bwd_lds_view_load(
                b, &v, ivx(rc[i][0], r_rt), ivx(rc[i][1], c_rt), vdt, n);
            if(x == NULL)
                return;
            out_f(o, x);
            rocke_attn_bwd_lds_view_store(b, &v, ivx(rc[i][0], r_rt), ivx(rc[i][1], c_rt), x, n);
        }
    }
    if(f->tr16)
    {
        rocke_value_t* x = rocke_attn_bwd_lds_view_tr16_load(
            b, &v, rocke_attn_bwd_iv_val(r_rt), rocke_attn_bwd_iv_int(4), elem);
        out_f(o, x);
        x = rocke_attn_bwd_lds_view_tr16_load(
            b, &v, rocke_attn_bwd_iv_int(3), rocke_attn_bwd_iv_val(c_rt), elem);
        out_f(o, x);
    }
    for(int fi = 0; fi < f->n_frag; ++fi)
    {
        const vfrag_t* fr = &f->frag[fi];
        rocke_attn_bwd_gemm_caps_t caps;
        fill_caps(c->arch, c->dtype, fr->atom_k, &caps);
        rocke_attn_bwd_frag_coords_t fc;
        if(!rocke_attn_bwd_operand_coords(b, caps.op, fr->role, &fc))
            return;
        bool tr16 = strcmp(fr->storage, "tr16") == 0;
        rocke_attn_bwd_storage_t st
            = strcmp(fr->storage, "k_outer") == 0 ? ROCKE_ATTN_BWD_K_OUTER : ROCKE_ATTN_BWD_K_INNER;
        rocke_value_t* x;
        if(tr16)
            x = rocke_attn_bwd_load_frag_lds_tr16(b,
                                                  &fc,
                                                  &v,
                                                  lane,
                                                  c->dtype,
                                                  caps.has_tr_read_b64,
                                                  ivx(fr->mn0, c_rt),
                                                  ivx(fr->k0, r_rt));
        else
            x = rocke_attn_bwd_load_frag_lds(b,
                                             &fc,
                                             NULL,
                                             &v,
                                             lane,
                                             c->dtype,
                                             ivx(fr->mn0, c_rt),
                                             ivx(fr->k0, r_rt),
                                             st,
                                             fr->align);
        if(x == NULL)
            return;
        out_f(o, x);
        if(!tr16)
        {
            int sh[2] = {64, 64};
            rocke_value_t* s2 = rocke_b_smem_alloc(b, elem, sh, 2, "lds_2d");
            rocke_value_t* y = rocke_attn_bwd_load_frag_lds(b,
                                                            &fc,
                                                            s2,
                                                            NULL,
                                                            lane,
                                                            c->dtype,
                                                            ivx(fr->mn0, c_rt),
                                                            ivx(fr->k0, r_rt),
                                                            st,
                                                            fr->align);
            if(y == NULL)
                return;
            out_f(o, y);
        }
    }
}

/* Build the probe kernel of config `c` into `b`. */
static void build(const cfg_t* c, rocke_ir_builder_t* b)
{
    out_t o;
    o.b = b;
    o.oi = rocke_b_param(b, "out_i", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    o.of = rocke_b_param(b, "out_f", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    o.ni = o.nf = 0;
    int ws = (strncmp(c->arch, "gfx11", 5) == 0 || strncmp(c->arch, "gfx12", 5) == 0) ? 32 : 64;
    rocke_value_t* tid_in = NULL;
    if(c->kind == K_GEMM && c->g.tid)
        tid_in = rocke_b_param(b, "tid", rocke_i32(), NULL);
    rocke_value_t *tid, *lane, *wave;
    rocke_attn_bwd_lane_and_wave(b, ws, tid_in, &tid, &lane, &wave);
    switch(c->kind)
    {
    case K_FACTS:
        probe_facts(b, &o, c, lane);
        break;
    case K_GEMM:
        probe_gemm(b, &o, c, lane, wave);
        break;
    case K_RELABEL:
        probe_relabel(b, &o, c, lane, wave);
        break;
    default:
        probe_views(b, &o, c, lane);
        break;
    }
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    static cfg_t table[MAX_CFG];
    int n = build_table(table);
    if(idx < 0 || idx >= n)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const cfg_t* c = &table[idx];
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "attn_bwd_frag_probe") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* A helper raises (ckc::Error) where Python raises ValueError; the guard
     * records it on the builder, as a kernel entry point would. */
    rocke_kernel_def_t* kernel = ckc::guard_builder(&b, [&]() -> rocke_kernel_def_t* {
        build(c, &b);
        return b.kernel;
    });
    if(kernel == NULL || b.status != ROCKE_OK)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }
    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, c->arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* ir_text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &ir_text);
        if(st != ROCKE_OK || !ir_text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(ir_text, stdout);
        free(ir_text);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t ndiag = 0;
        rocke_verify(kernel, &diags, &ndiag);
        for(size_t i = 0; i < ndiag; i++)
        {
            char* d = rocke_diag_to_string(&diags[i]);
            if(d)
            {
                fputs(d, stdout);
                fputc('\n', stdout);
                free(d);
            }
        }
        rocke_diags_free(diags, ndiag);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s' (expected ll|ir|verify)\n", mode);
        ret = 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
