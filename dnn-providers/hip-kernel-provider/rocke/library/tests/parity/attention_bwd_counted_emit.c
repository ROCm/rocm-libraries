/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_counted_emit.c -- C-side emitter for the counted
 * in-loop waits of the attention backward q step. Builds, identically to the
 * Python emitter attention_bwd_counted_emit.py, a probe kernel that calls every
 * entry point of helper_rocke.instances.common._attention_bwd_counted.h,
 * lowers it via rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 *
 * argv[1] is the config index: the (shape, wait mode, arch) loops of
 * _configs in the Python emitter (index = (shape * 4 + mode) * 2 + arch, the
 * dtype, atomic count and stats count cycling as there), then the special
 * configs. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.instances.common._attention_bwd_counted.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define PROBE_NAME "attn_bwd_counted_probe"

typedef struct
{
    int rows, cols, threads;
} shape_t;

/* Same order as SHAPES in the Python emitter. */
static const shape_t SHAPES[] = {
    {16, 64, 128},
    {16, 128, 128},
    {32, 64, 128},
    {32, 128, 256},
    {16, 128, 256},
    {16, 32, 128},
    {32, 96, 256},
    {128, 64, 256},
    {16, 8, 64},
    {16, 64, 512},
};
#define N_SHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))
static const char* const ARCHES[] = {"gfx942", "gfx950"};
#define N_ARCHES 2
static const char* const DTYPES[] = {"f16", "bf16"};
typedef enum wmode
{
    MODE_EARLY,
    MODE_LATE,
    MODE_DEFERRED,
    MODE_TWO_STEPS,
    MODE_BAD_END,
    MODE_ZERO_BAD,
    MODE_STORE_MISMATCH
} wmode_t;
static const wmode_t MODES[] = {MODE_EARLY, MODE_LATE, MODE_DEFERRED, MODE_TWO_STEPS};
#define N_MODES 4
static const int N_ATOMICS[] = {0, 4, 8, 16, 63};
#define N_N_ATOMICS 5
static const int N_STATS[] = {8, 16};

typedef struct
{
    const char* arch;
    int dtype; /* 0: f16, 1: bf16 */
    shape_t shape;
    int n_atomics;
    int n_stats;
    wmode_t mode;
    int vec;
} config_t;

/* Same order as _SPECIAL in the Python emitter. */
static const config_t SPECIAL[] = {
    {"gfx950", 0, {16, 64, 128}, 8, 8, MODE_BAD_END, 8},
    {"gfx942", 1, {32, 128, 256}, 4, 16, MODE_BAD_END, 8},
    {"gfx950", 0, {16, 64, 128}, 64, 8, MODE_EARLY, 8},
    {"gfx942", 0, {16, 64, 128}, -1, 8, MODE_EARLY, 8},
    {"gfx950", 1, {16, 64, 128}, 8, 8, MODE_ZERO_BAD, 8},
    {"gfx942", 0, {16, 128, 256}, 8, 8, MODE_STORE_MISMATCH, 8},
    {"gfx950", 1, {32, 64, 128}, 63, 16, MODE_LATE, 8},
    {"gfx942", 0, {16, 128, 128}, 0, 8, MODE_DEFERRED, 8},
    {"gfx950", 0, {16, 64, 128}, 8, 8, MODE_EARLY, 4},
    {"gfx942", 1, {32, 128, 256}, 4, 16, MODE_TWO_STEPS, 16},
    {"gfx950", 1, {16, 64, 128}, 16, 8, MODE_DEFERRED, 2},
    {"gfx942", 0, {16, 96, 128}, 8, 8, MODE_LATE, 16},
    {"gfx950", 0, {16, 8, 64}, 0, 8, MODE_EARLY, 4},
};
#define N_SPECIAL ((int)(sizeof(SPECIAL) / sizeof(SPECIAL[0])))
#define N_MAIN (N_SHAPES * N_MODES * N_ARCHES)

/* Decode `idx`; returns 0 on success, -1 if unknown. */
static int decode(int idx, config_t* cfg)
{
    if(idx < 0 || idx >= N_MAIN + N_SPECIAL)
        return -1;
    if(idx >= N_MAIN)
    {
        *cfg = SPECIAL[idx - N_MAIN];
        return 0;
    }
    const int a = idx % N_ARCHES;
    const int m = (idx / N_ARCHES) % N_MODES;
    const int s = idx / (N_ARCHES * N_MODES);
    cfg->arch = ARCHES[a];
    cfg->dtype = (s + m) % 2;
    cfg->shape = SHAPES[s];
    cfg->n_atomics = N_ATOMICS[(s + m) % N_N_ATOMICS];
    cfg->n_stats = N_STATS[s % 2];
    cfg->mode = MODES[m];
    cfg->vec = 8;
    return 0;
}

static const char* arch_of(int idx)
{
    config_t cfg;
    return decode(idx, &cfg) == 0 ? cfg.arch : NULL;
}

/* The loader / view contexts of the probe. */
typedef struct
{
    rocke_ir_builder_t* b;
    rocke_value_t* ptr;
    rocke_value_t* stride;
    const rocke_type_t* elem;
    int vec;
} loader_ctx_t;

static rocke_value_t* load_fn(void* ctx, rocke_value_t* row, rocke_value_t* col)
{
    loader_ctx_t* c = (loader_ctx_t*)ctx;
    rocke_value_t* m = rocke_b_mul(c->b, row, c->stride);
    rocke_value_t* idx = rocke_b_add(c->b, m, col);
    return rocke_b_global_load_vN(c->b, c->ptr, idx, c->elem, c->vec, 16);
}

static void store_fn(void* ctx,
                     rocke_ir_builder_t* b,
                     rocke_value_t* row,
                     rocke_value_t* col,
                     rocke_value_t* val,
                     int vec)
{
    rocke_value_t* idx[2] = {row, col};
    rocke_b_smem_store_vN(b, (rocke_value_t*)ctx, idx, 2, val, vec);
}

/* The counted-wait event sequence of one q step (or two) per issue mode. */
static bool sequence(rocke_ir_builder_t* b, rocke_attn_bwd_counted_waits_t* w, int n, wmode_t mode)
{
    switch(mode)
    {
    case MODE_EARLY:
        rocke_attn_bwd_counted_waits_step_entry(w);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        rocke_attn_bwd_counted_waits_loads(w);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        rocke_attn_bwd_counted_waits_atomics(w, n);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        return rocke_attn_bwd_counted_waits_end_step(w, b);
    case MODE_LATE:
        rocke_attn_bwd_counted_waits_step_entry(w);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        rocke_attn_bwd_counted_waits_loads(w);
        rocke_attn_bwd_counted_waits_atomics(w, n);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        return rocke_attn_bwd_counted_waits_end_step(w, b);
    case MODE_DEFERRED:
        rocke_attn_bwd_counted_waits_step_entry(w);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        rocke_attn_bwd_counted_waits_loads(w);
        rocke_attn_bwd_counted_waits_atomics(w, n);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        return rocke_attn_bwd_counted_waits_end_step(w, b);
    case MODE_TWO_STEPS:
        return sequence(b, w, n, MODE_EARLY) && sequence(b, w, n, MODE_LATE);
    case MODE_BAD_END:
        rocke_attn_bwd_counted_waits_step_entry(w);
        rocke_attn_bwd_counted_waits_barrier(w, b);
        rocke_attn_bwd_counted_waits_atomics(w, n + 1);
        return rocke_attn_bwd_counted_waits_end_step(w, b);
    default:
        return false;
    }
}

/* Build the probe kernel of config `idx` into `b`. 0 on success. */
static int build(int idx, rocke_ir_builder_t* b)
{
    config_t cfg;
    if(decode(idx, &cfg) != 0)
        return -1;
    const int rows = cfg.shape.rows;
    const int cols = cfg.shape.cols;
    const int threads = cfg.shape.threads;
    const int n_stats = cfg.n_stats;
    const rocke_type_t* elem = cfg.dtype == 0 ? rocke_f16() : rocke_bf16();

    const rocke_type_t* ptr_elem = rocke_ptr_type(b, elem, "global");
    rocke_value_t* src_q = rocke_b_param(b, "src_q", ptr_elem, NULL);
    rocke_value_t* src_do = rocke_b_param(b, "src_do", ptr_elem, NULL);
    rocke_value_t* stride = rocke_b_param(b, "stride", rocke_i32(), NULL);
    const rocke_type_t* out_ty = rocke_ptr_type(b, rocke_f32(), "global");
    rocke_value_t* out = rocke_b_param(b, "out", out_ty, NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    const int shape2[2] = {rows, cols};
    rocke_value_t* lds_q = rocke_b_smem_alloc(b, elem, shape2, 2, "lds_q");
    rocke_value_t* lds_do = rocke_b_smem_alloc(b, elem, shape2, 2, "lds_do");

    loader_ctx_t lq = {b, src_q, stride, elem, cfg.vec};
    loader_ctx_t ld = {b, src_do, stride, elem, cfg.vec};
    rocke_attn_bwd_tile_loader_t loaders[2] = {{load_fn, &lq}, {load_fn, &ld}};

    const int n_chunks = rocke_attn_bwd_tile_chunks_count(b, rows, cols, threads, cfg.vec);
    if(n_chunks < 0)
        return 1;
    std::vector<rocke_attn_bwd_tile_chunk_t> chunks((size_t)n_chunks);
    if(!rocke_attn_bwd_tile_chunks(b, rows, cols, tid, threads, cfg.vec, chunks.data(), n_chunks))
        return 1;
    const int n_vals = 2 * n_chunks;
    std::vector<rocke_value_t*> values((size_t)n_vals);
    if(!rocke_attn_bwd_issue_tile_loads(
           b, loaders, 2, chunks.data(), n_chunks, values.data(), n_vals))
        return 1;
    rocke_attn_bwd_tile_view_t views[2] = {{store_fn, lds_q}, {store_fn, lds_do}};
    if(cfg.mode == MODE_STORE_MISMATCH)
    {
        if(!rocke_attn_bwd_store_tile(
               b, views, 2, chunks.data(), n_chunks, values.data(), n_vals - 1, cfg.vec))
            return 1;
    }
    if(!rocke_attn_bwd_store_tile(
           b, views, 2, chunks.data(), n_chunks, values.data(), n_vals, cfg.vec))
        return 1;
    const rocke_type_t* vec_ty = rocke_vector_type(b, elem, cfg.vec);
    std::vector<const rocke_type_t*> types((size_t)n_stats, rocke_f32());
    for(int i = 0; i < n_vals; ++i)
        types.push_back(vec_ty);
    if(cfg.mode == MODE_ZERO_BAD)
        types.push_back(rocke_i32());
    std::vector<rocke_value_t*> zeros(types.size());
    if(!rocke_attn_bwd_zero_like(b, types.data(), (int)types.size(), zeros.data()))
        return 1;
    rocke_attn_bwd_staged_step_t step;
    step.stats = zeros.data();
    step.n_stats = n_stats;
    step.qdo = zeros.data() + n_stats;
    step.n_qdo = (int)zeros.size() - n_stats;
    std::vector<rocke_value_t*> flat((size_t)rocke_attn_bwd_staged_step_flat_len(&step));
    rocke_attn_bwd_staged_step_flat(&step, flat.data());
    rocke_attn_bwd_staged_step_t back
        = rocke_attn_bwd_staged_step_from_flat(flat.data(), (int)flat.size(), n_stats);
    std::vector<rocke_value_t*> res;
    for(int i = 0; i < back.n_stats; ++i)
        res.push_back(back.stats[i]);
    for(int i = 0; i < back.n_qdo; ++i)
    {
        rocke_value_t* e = rocke_b_vec_extract(b, back.qdo[i], 0);
        res.push_back(rocke_b_cast_to_f32(b, e));
    }
    for(int i = 0; i < n_vals; ++i)
    {
        rocke_value_t* e = rocke_b_vec_extract(b, values[(size_t)i], 0);
        res.push_back(rocke_b_cast_to_f32(b, e));
    }
    rocke_attn_bwd_counted_waits_t waits;
    if(!rocke_attn_bwd_counted_waits_init(&waits, b, cfg.n_atomics))
        return 1;
    const bool ok = sequence(b, &waits, cfg.n_atomics, cfg.mode);
    if(ok)
    {
        for(int i = 0; i < waits.n_waits; ++i)
            res.push_back(rocke_b_const_f32(b, (double)waits.waits[i]));
    }
    rocke_attn_bwd_counted_waits_free(&waits);
    if(!ok)
        return 1;

    for(size_t i = 0; i < res.size(); ++i)
    {
        rocke_value_t* at = rocke_b_const_i32(b, (int64_t)i);
        rocke_b_global_store(b, out, at, res[i], 1);
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
    const char* arch = arch_of(idx);
    if(arch == NULL)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, PROBE_NAME) != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* A strict-mode builder error throws; the guard turns it back into the
     * sticky error, like the guarded entry points of the C++ kernels. */
    int rc = 1;
    ckc::guard_builder(&b, [&]() -> void* {
        rc = build(idx, &b);
        return &b;
    });
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
