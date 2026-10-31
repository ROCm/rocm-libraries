/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_paged_emit.c -- C-side emitter for the paged
 * address generator of the unified attention backward main kernel. Builds,
 * identically to the Python emitter attention_bwd_paged_emit.py, a probe kernel
 * that drives helper_rocke.instances.common._attention_bwd_paged.h, lowers it
 * via rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so
 * the two outputs can be byte-compared.
 *
 * argv[1] is the config index, decoded as in _configs of the Python emitter
 * (every base tile on both arches with the cache access knobs cycling, then
 * the rejects). Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_paged.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define PROBE_NAME "attn_bwd_paged_probe"

/* stride kinds */
#define STR_VALUE 0
#define STR_INT 1
#define STR_UNIT 2
#define STR_ZERO 3

/* page kinds */
enum
{
    PK_VALUE,
    PK_INT,
    PK_SUB
};
/* head / column kinds */
enum
{
    HEAD_ZERO,
    HEAD_INT,
    HEAD_VALUE
};

typedef struct
{
    const char* arch;
    int kn;
    int sub; /* 0: None */
    int page_kind;
    int guard;
    int bt_value; /* block-table stride is a runtime value */
    int head;
    int clamp;
    const char* dt;
    int sv;
    int strides_kind; /* STR_* */
    int vec;
    int col; /* 0, 8, or -1 for the runtime value */
    int ws_width;
} config_t;

static const char* const ARCHES[] = {"gfx942", "gfx950"};
static const int MULTI_KNS[] = {16, 64, 128};
static const int SUB_TILES[][2] = {{128, 16}, {128, 32}, {128, 64}, {256, 128}};
static const char* const DTYPES[] = {"f16", "bf16", "f32"};
static const int SVS[] = {8, 1};
static const int VECS[] = {8, 16, 4, 2};
static const int COLS[] = {0, 8, -1};

/* Same order as _REJECTS in the Python emitter. */
static const config_t REJECTS[] = {
    {"gfx950", 128, 24, PK_SUB, 0, 1, HEAD_ZERO, 1, "f16", 8, STR_VALUE, 8, 0, 64},
    {"gfx942", 128, 48, PK_SUB, 1, 0, HEAD_VALUE, 1, "bf16", 8, STR_INT, 8, 0, 64},
    {"gfx950", 64, 64, PK_SUB, 0, 1, HEAD_INT, 0, "f16", 8, STR_VALUE, 8, 8, 64},
    {"gfx942", 128, 32, PK_VALUE, 0, 1, HEAD_ZERO, 1, "f16", 8, STR_VALUE, 8, 0, 64},
    {"gfx950", 128, 32, PK_INT, 1, 0, HEAD_ZERO, 1, "f16", 8, STR_VALUE, 8, 0, 64},
    {"gfx942", 64, 0, PK_VALUE, 0, 1, HEAD_ZERO, 1, "f16", 4, STR_VALUE, 8, 0, 64},
    {"gfx950", 64, 0, PK_VALUE, 1, 1, HEAD_INT, 1, "i64", 8, STR_INT, 8, 0, 64},
    {"gfx942", 64, 0, PK_INT, 0, 0, HEAD_VALUE, 0, "bf16", 8, STR_INT, 8, 8, 0},
    {"gfx950", 128, 64, PK_SUB, 1, 0, HEAD_VALUE, 1, "f16", 8, STR_INT, 8, 0, 0},
};

/* Same order as _STRIDES in the Python emitter. */
static const config_t STRIDES[] = {
    {"gfx942", 64, 0, PK_VALUE, 0, 1, HEAD_INT, 1, "f16", 8, STR_UNIT, 8, 8, 64},
    {"gfx950", 128, 0, PK_INT, 1, 0, HEAD_VALUE, 0, "bf16", 1, STR_ZERO, 4, 0, 64},
    {"gfx942", 128, 32, PK_SUB, 0, 1, HEAD_ZERO, 1, "f16", 8, STR_UNIT, 2, -1, 64},
    {"gfx950", 64, 0, PK_VALUE, 1, 1, HEAD_VALUE, 0, "f32", 8, STR_ZERO, 8, 8, 64},
};

static std::vector<config_t> build_configs()
{
    typedef struct
    {
        int kn, sub, page_kind, guard, bt_value;
    } base_t;
    std::vector<base_t> bases;
    for(int ki = 0; ki < 3; ++ki)
        for(int guard = 0; guard < 2; ++guard)
            for(int pk = 0; pk < 2; ++pk)
                for(int bt = 0; bt < 2; ++bt)
                {
                    base_t bs = {MULTI_KNS[ki], 0, pk == 0 ? PK_VALUE : PK_INT, guard, bt == 0};
                    bases.push_back(bs);
                }
    for(int t = 0; t < 4; ++t)
        for(int guard = 0; guard < 2; ++guard)
            for(int bt = 0; bt < 2; ++bt)
            {
                base_t bs = {SUB_TILES[t][0], SUB_TILES[t][1], PK_SUB, guard, bt == 0};
                bases.push_back(bs);
            }
    std::vector<config_t> out;
    int i = 0;
    for(int a = 0; a < 2; ++a)
        for(size_t bi = 0; bi < bases.size(); ++bi)
        {
            config_t c;
            c.arch = ARCHES[a];
            c.kn = bases[bi].kn;
            c.sub = bases[bi].sub;
            c.page_kind = bases[bi].page_kind;
            c.guard = bases[bi].guard;
            c.bt_value = bases[bi].bt_value;
            c.head = i % 3;
            c.clamp = (i / 3) % 2;
            c.dt = DTYPES[(i / 2) % 3];
            c.sv = SVS[(i / 5) % 2];
            c.strides_kind = ((i / 4) % 2) == 0 ? STR_VALUE : STR_INT;
            c.vec = VECS[(i / 3) % 4];
            c.col = COLS[(i / 2) % 3];
            c.ws_width = 64;
            out.push_back(c);
            ++i;
        }
    for(size_t r = 0; r < sizeof(REJECTS) / sizeof(REJECTS[0]); ++r)
        out.push_back(REJECTS[r]);
    for(size_t r = 0; r < sizeof(STRIDES) / sizeof(STRIDES[0]); ++r)
        out.push_back(STRIDES[r]);
    return out;
}

static const std::vector<config_t>& configs()
{
    static const std::vector<config_t> c = build_configs();
    return c;
}

static const char* arch_of(int idx)
{
    if(idx < 0 || idx >= (int)configs().size())
        return NULL;
    return configs()[(size_t)idx].arch;
}

static rocke_value_t* c32(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

static const rocke_type_t* dtype_of(const char* dt)
{
    if(strcmp(dt, "f16") == 0)
        return rocke_f16();
    if(strcmp(dt, "bf16") == 0)
        return rocke_bf16();
    if(strcmp(dt, "f32") == 0)
        return rocke_f32();
    return rocke_i64();
}

static bool build_probe(rocke_ir_builder_t* b, const config_t& cfg)
{
    const int kn = cfg.kn;
    const rocke_type_t* elem = dtype_of(cfg.dt);
    const rocke_type_t* pi32 = rocke_ptr_type(b, rocke_i32(), "global");
    rocke_value_t* bt = rocke_b_param(b, "bt", pi32, NULL);
    const rocke_type_t* pe = rocke_ptr_type(b, elem, "global");
    rocke_value_t* kc = rocke_b_param(b, "kc", pe, NULL);
    rocke_value_t* dkc = rocke_b_param(b, "dkc", pe, NULL);
    rocke_value_t* ws = rocke_b_param(b, "ws", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* seq_lens = rocke_b_param(b, "seq_lens", pi32, NULL);
    rocke_value_t* bt_stride = rocke_b_param(b, "bt_stride", rocke_i32(), NULL);
    rocke_value_t* s_max = rocke_b_param(b, "s_max", rocke_i32(), NULL);
    rocke_value_t* page = rocke_b_param(b, "page", rocke_i32(), NULL);
    rocke_value_t* page_rows = rocke_b_param(b, "page_rows", rocke_i32(), NULL);
    rocke_value_t* kc_blk = rocke_b_param(b, "kc_blk", rocke_i32(), NULL);
    rocke_value_t* kc_tok = rocke_b_param(b, "kc_tok", rocke_i32(), NULL);
    rocke_value_t* kc_h = rocke_b_param(b, "kc_h", rocke_i32(), NULL);
    rocke_value_t* hk_p = rocke_b_param(b, "hk", rocke_i32(), NULL);
    rocke_value_t* col_p = rocke_b_param(b, "col", rocke_i32(), NULL);
    rocke_value_t* val = rocke_b_param(b, "val", rocke_f32(), NULL);
    rocke_value_t* s_idx = rocke_b_block_id_y(b);
    int seq_guards = 0;
    rocke_attn_bwd_decode_args_t args = rocke_attn_bwd_decode_args_default();
    args.s_max = rocke_attn_bwd_iv_val(s_max);
    args.seq_ptr = seq_lens;
    args.has_len = rocke_attn_bwd_iv_int(1);
    rocke_attn_bwd_seq_info_t seq_kv = rocke_attn_bwd_decode_batched(b, &seq_guards, s_idx, &args);
    const int sub_page = cfg.sub;
    const rocke_attn_bwd_iv_t page_arg = cfg.page_kind == PK_VALUE ? rocke_attn_bwd_iv_val(page)
                                         : cfg.page_kind == PK_INT
                                             ? rocke_attn_bwd_iv_int(2 * kn)
                                             : rocke_attn_bwd_iv_int(sub_page);
    rocke_attn_bwd_paged_kv_t paged;
    const rocke_attn_bwd_iv_t pr_iv = rocke_attn_bwd_iv_val(page_rows);
    if(!rocke_attn_bwd_paged_kv_init(&paged,
                                     b,
                                     bt,
                                     cfg.bt_value ? rocke_attn_bwd_iv_val(bt_stride)
                                                  : rocke_attn_bwd_iv_int(64),
                                     s_idx,
                                     &seq_kv,
                                     kn,
                                     page_arg,
                                     sub_page ? &sub_page : NULL,
                                     cfg.guard ? &pr_iv : NULL))
        return false;
    bool ok = false;
    do
    {
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* ckn = c32(b, kn);
        rocke_value_t* k0 = rocke_b_mul(b, bx, ckn);
        const rocke_attn_bwd_paged_kv_tile_t* tile = rocke_attn_bwd_paged_kv_tile(&paged, k0);
        if(tile == NULL)
            break;
        const rocke_attn_bwd_paged_kv_tile_t* tile_again = rocke_attn_bwd_paged_kv_tile(&paged, k0);
        if(tile_again != tile)
            break;
        rocke_value_t* tid = rocke_b_thread_id_x(b);
        rocke_value_t* ckn2 = c32(b, kn);
        rocke_value_t* lr = rocke_b_mod(b, tid, ckn2);
        rocke_value_t* pid = NULL;
        rocke_value_t* slot = NULL;
        rocke_attn_bwd_paged_kv_tile_locate(tile, lr, &pid, &slot);
        rocke_value_t* aid = NULL;
        rocke_value_t* ok_v = NULL;
        rocke_attn_bwd_paged_kv_tile_checked(tile, pid, &aid, &ok_v);
        rocke_value_t* k = rocke_b_add(b, k0, lr);
        rocke_value_t* store = NULL;
        rocke_value_t* row_live = NULL;
        rocke_attn_bwd_paged_kv_tile_store_rule(tile, k, &store, &row_live);
        rocke_value_t* lb = rocke_attn_bwd_live_blocks(b, seq_kv.length, page_arg);
        rocke_value_t* ckn3 = c32(b, kn);
        rocke_value_t* k1 = rocke_b_add(b, k0, ckn3);
        const rocke_attn_bwd_paged_kv_tile_t* tile2 = rocke_attn_bwd_paged_kv_tile(&paged, k1);
        if(tile2 == NULL)
            break;
        rocke_value_t* pid2 = NULL;
        rocke_value_t* slot2 = NULL;
        rocke_attn_bwd_paged_kv_tile_locate(tile2, lr, &pid2, &slot2);
        rocke_attn_bwd_iv_t strides[3];
        switch(cfg.strides_kind)
        {
        case STR_VALUE:
            strides[0] = rocke_attn_bwd_iv_val(kc_blk);
            strides[1] = rocke_attn_bwd_iv_val(kc_tok);
            strides[2] = rocke_attn_bwd_iv_val(kc_h);
            break;
        case STR_INT:
            strides[0] = rocke_attn_bwd_iv_int(16384);
            strides[1] = rocke_attn_bwd_iv_int(128);
            strides[2] = rocke_attn_bwd_iv_int(64);
            break;
        case STR_UNIT:
            strides[0] = rocke_attn_bwd_iv_int(1);
            strides[1] = rocke_attn_bwd_iv_int(0);
            strides[2] = rocke_attn_bwd_iv_int(1);
            break;
        default:
            strides[0] = rocke_attn_bwd_iv_int(0);
            strides[1] = rocke_attn_bwd_iv_int(1);
            strides[2] = rocke_attn_bwd_iv_int(0);
            break;
        }
        rocke_attn_bwd_paged_cache_rows_t k_cache
            = rocke_attn_bwd_paged_kv_cache(&paged, kc, elem, strides, cfg.sv);
        rocke_attn_bwd_paged_cache_rows_t out_cache
            = rocke_attn_bwd_paged_kv_cache(&paged, dkc, elem, strides, cfg.sv);
        const rocke_attn_bwd_iv_t head = cfg.head == HEAD_ZERO  ? rocke_attn_bwd_iv_int(0)
                                         : cfg.head == HEAD_INT ? rocke_attn_bwd_iv_int(3)
                                                                : rocke_attn_bwd_iv_val(hk_p);
        const rocke_attn_bwd_iv_t col
            = cfg.col < 0 ? rocke_attn_bwd_iv_val(col_p) : rocke_attn_bwd_iv_int(cfg.col);
        rocke_attn_bwd_paged_slab_t slab;
        if(!rocke_attn_bwd_paged_cache_rows_slab(&k_cache, head, k0, cfg.clamp != 0, &slab))
            break;
        rocke_value_t* row = rocke_b_add(b, k0, lr);
        rocke_value_t* v = rocke_attn_bwd_paged_slab_load_vec(&slab, row, col, cfg.vec);
        if(v == NULL)
            break;
        rocke_attn_bwd_paged_slab_t dst;
        if(!rocke_attn_bwd_paged_cache_rows_slab(&out_cache, head, k0, false, &dst))
            break;
        rocke_value_t* e0 = rocke_b_vec_extract(b, v, 0);
        rocke_attn_bwd_paged_slab_store(&dst, row, col, e0);
        rocke_attn_bwd_paged_slab_store_vec(&dst, row, col, v);
        const rocke_attn_bwd_iv_t ws_pr
            = cfg.guard ? rocke_attn_bwd_iv_val(page_rows) : rocke_attn_bwd_iv_int(8192);
        rocke_attn_bwd_paged_ws_row_t wr;
        if(!rocke_attn_bwd_paged_kv_tile_workspace_row(tile, ws, ws_pr, cfg.ws_width, head, &wr))
            break;
        rocke_attn_bwd_ws_slab_t rs;
        rocke_value_t* rr = NULL;
        if(!rocke_attn_bwd_paged_ws_row_slab(&wr, lr, &rs, &rr))
            break;
        rocke_attn_bwd_ws_slab_atomic_add(&rs, rr, rocke_attn_bwd_iv_int(0), val);
        rocke_attn_bwd_ws_slab_store(&rs, rr, col, val);
        rocke_value_t* c1 = c32(b, 1);
        rocke_value_t* lr1 = rocke_b_add(b, lr, c1);
        rocke_attn_bwd_ws_slab_t rs2;
        rocke_value_t* rr2 = NULL;
        if(!rocke_attn_bwd_paged_ws_row_slab(&wr, lr1, &rs2, &rr2))
            break;
        rocke_attn_bwd_ws_slab_atomic_add(&rs2, rr2, rocke_attn_bwd_iv_int(8), val);
        /* keep the scalar results live */
        rocke_value_t* out = rocke_b_param(b, "out", pi32, NULL);
        std::vector<rocke_value_t*> res;
        res.push_back(slot);
        res.push_back(aid);
        res.push_back(k);
        res.push_back(pid2);
        res.push_back(slot2);
        res.push_back(lb);
        res.push_back(rocke_b_zext(b, store, rocke_i32()));
        res.push_back(rocke_b_zext(b, row_live, rocke_i32()));
        if(ok_v != NULL)
            res.push_back(rocke_b_zext(b, ok_v, rocke_i32()));
        for(size_t i = 0; i < res.size(); ++i)
        {
            rocke_value_t* at = c32(b, (int64_t)i);
            rocke_b_global_store(b, out, at, res[i], 1);
        }
        ok = true;
    } while(0);
    rocke_attn_bwd_paged_kv_free(&paged);
    return ok;
}

/* Build the probe kernel of config `idx` into `b`. 0 on success. */
static int build(int idx, rocke_ir_builder_t* b)
{
    if(idx < 0 || idx >= (int)configs().size())
        return -1;
    const config_t cfg = configs()[(size_t)idx];
    if(!build_probe(b, cfg) || !rocke_ir_builder_ok(b))
        return 1;
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
