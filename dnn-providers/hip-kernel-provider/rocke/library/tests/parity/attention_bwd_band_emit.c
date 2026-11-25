/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_band_emit.c -- C-side emitter for the attention
 * backward band helpers. Builds, identically to the Python emitter
 * attention_bwd_band_emit.py, a probe kernel that calls every entry point of
 * helper_rocke.helpers.attention_band.h and
 * helper_rocke.instances.common._attention_bwd_band.h, lowers it via
 * rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so the
 * two outputs can be byte-compared.
 *
 * argv[1] is the config index (arch, tile pair, band shape; decoded exactly as
 * _decode in the Python emitter). Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.attention_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* An operand of a band shape: the kernel parameter, a build-time int, the
 * per-sequence diagonal select (offset only) or absent. */
enum
{
    OP_RT,
    OP_INT,
    OP_BR,
    OP_NONE
};

typedef struct
{
    int mode;
    int64_t v;
} operand_t;

typedef struct
{
    int masked; /* 0: NoMaskKTail, 1: RuntimeBand */
    operand_t s_q, s_kv, off, left, right;
} shape_t;

#define RT       \
    {            \
        OP_RT, 0 \
    }
#define BR       \
    {            \
        OP_BR, 0 \
    }
#define NA         \
    {              \
        OP_NONE, 0 \
    }
#define I(x)        \
    {               \
        OP_INT, (x) \
    }

/* Same order as SHAPES in the Python emitter. */
static const shape_t SHAPES[] = {
    {0, RT, RT, NA, NA, NA},
    {1, RT, RT, BR, RT, RT},
    {1, RT, RT, BR, I(-1), I(0)},
    {1, I(1024), I(1024), I(0), I(-1), I(0)},
    {1, I(512), I(1024), I(512), I(-1), I(0)},
    {1, I(777), I(777), I(0), I(128), I(0)},
    {1, I(100), I(200), I(100), I(64), I(64)},
    {1, I(640), I(640), I(0), I(96), I(-1)},
    {0, I(1000), I(1300), NA, NA, NA},
    {1, RT, RT, BR, RT, I(0)},
    {1, RT, RT, I(0), I(32), RT},
    {1, RT, RT, I(64), RT, I(0)},
    {1, RT, RT, I(-32), I(48), RT},
    {1, I(1024), RT, BR, I(0), RT},
    {1, RT, I(2048), I(16), RT, I(0)},
};
#define N_SHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

typedef struct
{
    int bm, bn;
} tile_t;

/* Same tile lists as TILES in the Python emitter. */
static const tile_t TILES_GFX942[] = {{16, 32}, {16, 64}, {16, 128}, {32, 128}};
static const tile_t TILES_GFX950[]
    = {{16, 64}, {16, 128}, {16, 192}, {32, 128}, {32, 256}, {128, 64}};

typedef struct
{
    const char* arch;
    const tile_t* tiles;
    int n_tiles;
} arch_tiles_t;

static const arch_tiles_t ARCHES[] = {
    {"gfx942", TILES_GFX942, (int)(sizeof(TILES_GFX942) / sizeof(TILES_GFX942[0]))},
    {"gfx950", TILES_GFX950, (int)(sizeof(TILES_GFX950) / sizeof(TILES_GFX950[0]))},
};

/* Decode `idx`; returns 0 on success, -1 if unknown. */
static int decode(int idx, const char** arch, tile_t* tile, const shape_t** shape)
{
    int base = 0;
    for(size_t a = 0; a < sizeof(ARCHES) / sizeof(ARCHES[0]); ++a)
    {
        int span = ARCHES[a].n_tiles * N_SHAPES;
        if(idx >= 0 && idx < base + span)
        {
            int rel = idx - base;
            *arch = ARCHES[a].arch;
            *tile = ARCHES[a].tiles[rel / N_SHAPES];
            *shape = &SHAPES[rel % N_SHAPES];
            return 0;
        }
        base += span;
    }
    return -1;
}

static rocke_int_or_value_t operand_of(const operand_t* x, rocke_value_t* rt)
{
    return x->mode == OP_RT ? rocke_iov_value(rt) : rocke_iov_int(x->v);
}

#define MAX_RES 96

typedef struct
{
    rocke_value_t* v[MAX_RES];
    int n;
} results_t;

static void push(results_t* r, rocke_value_t* v)
{
    if(v != NULL && r->n < MAX_RES)
    {
        r->v[r->n++] = v;
    }
}

/* Build the probe kernel of config `idx` into `b`. 0 on success. */
static int build(int idx, rocke_ir_builder_t* b)
{
    const char* arch;
    tile_t tile;
    const shape_t* sh;
    if(decode(idx, &arch, &tile, &sh) != 0)
    {
        return -1;
    }
    const int km = tile.bm;
    const int kn = tile.bn;

    rocke_value_t* len_q = rocke_b_param(b, "len_q", rocke_i32(), NULL);
    rocke_value_t* len_kv = rocke_b_param(b, "len_kv", rocke_i32(), NULL);
    rocke_value_t* bottom_right = rocke_b_param(b, "bottom_right", rocke_i32(), NULL);
    rocke_value_t* left_p = rocke_b_param(b, "left", rocke_i32(), NULL);
    rocke_value_t* right_p = rocke_b_param(b, "right", rocke_i32(), NULL);
    rocke_value_t* s_kv_max = rocke_b_param(b, "s_kv_max", rocke_i32(), NULL);
    const rocke_type_t* out_ty = rocke_ptr_type(b, rocke_i32(), "global");
    rocke_value_t* out = rocke_b_param(b, "out", out_ty, NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bid = rocke_b_block_id_x(b);

    rocke_int_or_value_t q_len = operand_of(&sh->s_q, len_q);
    rocke_int_or_value_t kv_len = operand_of(&sh->s_kv, len_kv);
    rocke_attn_bwd_mask_policy_t mask;
    if(!sh->masked)
    {
        if(rocke_attn_bwd_no_mask_ktail_init(&mask, b, q_len, kv_len, km, kn) != ROCKE_OK)
        {
            return 1;
        }
    }
    else
    {
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_int_or_value_t off;
        if(sh->off.mode == OP_BR)
        {
            rocke_value_t* br = rocke_b_cmp_ne(b, bottom_right, zero);
            rocke_value_t* diff = rocke_b_sub(b, len_kv, len_q);
            off = rocke_iov_value(rocke_b_select(b, br, diff, zero));
        }
        else
        {
            off = rocke_iov_int(sh->off.v);
        }
        rocke_attn_runtime_bounds_t bounds = rocke_attn_runtime_bounds_make(q_len, kv_len);
        bounds.diag_offset = off;
        bounds.left_bound = operand_of(&sh->left, left_p);
        bounds.right_bound = operand_of(&sh->right, right_p);
        if(rocke_attn_bwd_runtime_band_init(&mask, b, &bounds, km, kn) != ROCKE_OK)
        {
            return 1;
        }
    }
    rocke_band_emitter_t* band = &mask.band;
    results_t res;
    res.n = 0;

    /* rocke.helpers.attention_band: every BandEmitter entry point. */
    rocke_row_band_t rb;
    if(rocke_band_row_band(band, tid, &rb) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, rb.row_in);
    push(&res, rb.valid);
    push(&res, rb.lo);
    push(&res, rb.hi);
    rocke_value_t* col = rocke_band_col_in(band, bid);
    push(&res, col);
    push(&res, rocke_band_cell_keep(band, &rb, bid, col));
    rocke_value_t* kt_start = NULL;
    rocke_value_t* kt_stop = NULL;
    if(rocke_band_k_tile_range(band, tid, km, kn, &kt_start, &kt_stop) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, kt_start);
    push(&res, kt_stop);
    push(&res, rocke_band_k_tile_count_ceil(band, kn));

    /* Policy entry points, as the backward kernels call them. */
    rocke_value_t* qt_start = NULL;
    rocke_value_t* n_qt = NULL;
    if(rocke_attn_bwd_mask_q_tile_range(&mask, b, bid, &qt_start, &n_qt) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, qt_start);
    push(&res, n_qt);
    push(&res, rocke_attn_bwd_mask_tile_class(&mask, b, tid, bid));
    rocke_value_t* i0 = NULL;
    rocke_value_t* i1 = NULL;
    if(rocke_attn_bwd_mask_interior_q_range(&mask, b, bid, qt_start, n_qt, &i0, &i1) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, i0);
    push(&res, i1);
    rocke_row_band_t row;
    if(rocke_attn_bwd_mask_row(&mask, b, tid, &row) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, rocke_attn_bwd_mask_keep(&mask, b, &row, bid));
    push(&res, rocke_attn_bwd_mask_cell_keep(&mask, b, tid, bid));
    rocke_value_t* lim_lo = NULL;
    rocke_value_t* lim_hi = NULL;
    if(rocke_attn_bwd_mask_row_limits(&mask, b, tid, NULL, &lim_lo, &lim_hi) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, lim_lo);
    push(&res, lim_hi);
    rocke_value_t* ok = rocke_b_cmp_lt(b, bid, tid);
    rocke_value_t* ok_lo = NULL;
    rocke_value_t* ok_hi = NULL;
    if(rocke_attn_bwd_mask_row_limits(&mask, b, tid, ok, &ok_lo, &ok_hi) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, ok_lo);
    push(&res, ok_hi);
    push(&res, rocke_attn_bwd_mask_keep_in(b, lim_lo, lim_hi, bid));
    push(&res, rocke_attn_bwd_mask_keep_in(b, ok_lo, ok_hi, bid));

    /* Free functions with the flags the policies do not pass. */
    rocke_value_t* qt0 = NULL;
    rocke_value_t* n0 = NULL;
    if(rocke_attn_bwd_band_q_tile_range(b, band, bid, kn, km, false, &qt0, &n0) != ROCKE_OK)
    {
        return 1;
    }
    push(&res, qt0);
    push(&res, n0);
    push(&res, rocke_attn_bwd_band_tile_inside_band(b, band, tid, bid, tid, bid));
    push(&res, rocke_attn_bwd_band_is_edge_tile(b, band, tid, bid, km, kn, true));
    push(&res, rocke_attn_bwd_band_is_edge_tile(b, band, tid, bid, km, kn, false));
    push(&res, rocke_attn_bwd_band_tile_class(b, band, tid, bid, km, kn, true, false));
    push(&res, rocke_attn_bwd_band_tile_class(b, band, tid, bid, km, kn, false, false));
    rocke_value_t* j0 = NULL;
    rocke_value_t* j1 = NULL;
    if(rocke_attn_bwd_band_interior_q_range(b, band, bid, qt0, n0, km, kn, false, false, &j0, &j1)
       != ROCKE_OK)
    {
        return 1;
    }
    push(&res, j0);
    push(&res, j1);
    if(rocke_attn_bwd_band_interior_q_range(b, band, bid, qt0, n0, km, kn, true, false, &j0, &j1)
       != ROCKE_OK)
    {
        return 1;
    }
    push(&res, j0);
    push(&res, j1);
    rocke_value_t* store = NULL;
    rocke_value_t* live = NULL;
    rocke_int_or_value_t max_rt = rocke_iov_value(s_kv_max);
    rocke_int_or_value_t max_int = rocke_iov_int(4096);
    if(rocke_attn_bwd_band_dkv_store_rule(
           b, tid, len_kv, ROCKE_ATTN_BWD_SEQ_BATCHED, &max_rt, &store, &live)
       != ROCKE_OK)
    {
        return 1;
    }
    push(&res, store);
    push(&res, live);
    if(rocke_attn_bwd_band_dkv_store_rule(
           b, tid, len_kv, ROCKE_ATTN_BWD_SEQ_BATCHED, &max_int, &store, &live)
       != ROCKE_OK)
    {
        return 1;
    }
    push(&res, store);
    push(&res, live);
    if(rocke_attn_bwd_band_dkv_store_rule(
           b, tid, len_kv, ROCKE_ATTN_BWD_SEQ_THD, NULL, &store, &live)
       != ROCKE_OK)
    {
        return 1;
    }
    push(&res, store);
    push(&res, live);

    /* Keep every result live: one store per value. */
    for(int i = 0; i < res.n; ++i)
    {
        rocke_value_t* v = res.v[i];
        if(rocke_type_eq(v->type, rocke_i1()))
        {
            v = rocke_b_zext(b, v, rocke_i32());
        }
        rocke_value_t* at = rocke_b_const_i32(b, i);
        rocke_b_global_store(b, out, at, v, 1);
    }
    rocke_b_ret(b);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    const char* arch;
    tile_t tile;
    const shape_t* sh;
    if(decode(idx, &arch, &tile, &sh) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "attn_bwd_band_probe") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    int rc = build(idx, &b);
    if(rc != 0 || !rocke_ir_builder_ok(&b))
    {
        const char* msg = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed for config %d: %s\n", idx, msg ? msg : "");
        rocke_ir_builder_free(&b);
        return 1;
    }
    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(text, stdout);
        }
        free(text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(t, stdout);
        }
        free(t);
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return ret;
}
