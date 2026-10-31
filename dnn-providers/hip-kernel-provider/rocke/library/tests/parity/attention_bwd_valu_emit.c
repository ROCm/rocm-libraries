/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_valu_emit.c -- C-side emitter for the
 * softmax-gradient VALU levers of the attention backward main kernel. Builds,
 * identically to the Python emitter attention_bwd_valu_emit.py, a probe kernel
 * that calls the entry points of
 * helper_rocke.instances.common._attention_bwd_valu.h, lowers it via
 * rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so the
 * two outputs can be byte-compared.
 *
 * argv[1] is the config index, decoded as in _configs of the Python emitter
 * (sections cast, softmax, layout, stats; the same loops, the layout section
 * sized by the MFMA f16 / bf16 atoms of each arch). Optional argv[2] selects
 * the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_valu.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define PROBE_NAME "attn_bwd_valu_probe"

enum
{
    SEC_CAST,
    SEC_SOFTMAX,
    SEC_LAYOUT,
    SEC_STATS
};

/* One config; the fields a section does not use are 0 / NULL. */
typedef struct
{
    int section;
    const char* arch;
    int n; /* cast: fragment length; layout: op index; stats: width */
    int elem_f16; /* cast */
    const char* str; /* cast: bf16_cast; softmax: bwd_exp2; stats: variant */
    int flag; /* softmax: stats_fma; layout: width; stats: runs */
    int q0; /* stats: 0 zero, 1 int, 2 value */
    int nh_int; /* stats: n_heads is the build-time int 8 */
} config_t;

static const char* const ARCHES[] = {"gfx942", "gfx950"};
static const int CAST_NS[] = {2, 4, 8, 16};
static const char* const CAST_ELEM_NAME[] = {"f16", "bf16", "bf16", "f16"};
static const char* const CAST_KIND[] = {"generic", "generic", "rne_noguard", "rne_noguard"};
static const char* const EXP2S[] = {"exact", "fast", "bogus"};
static const int LAYOUT_WIDTHS[] = {1, 2, 4, 3, 8};
static const int STATS_WIDTHS[] = {2, 4};
static const int STATS_RUNS[] = {1, 2};
static const char* const VARIANTS[]
    = {"int_ws_rows", "token_major", "wide_ws", "distinct_ws_rows", "distinct_row0", "bad_rows"};

/* The f16 then bf16 MFMA atoms of `arch`, as Python's _ops(arch). */
static std::vector<const rocke_mma_op_t*> ops_of(const char* arch)
{
    std::vector<const rocke_mma_op_t*> out;
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
    if(t == NULL)
        return out;
    const char* dts[2] = {"f16", "bf16"};
    for(int d = 0; d < 2; ++d)
    {
        const int n = rocke_mma_catalog_enumerate(
            &t->mma, "mma", dts[d], dts[d], "fp32", -1, -1, NULL, 0, NULL);
        std::vector<const rocke_mma_op_t*> got((size_t)(n > 0 ? n : 0));
        if(n > 0)
            rocke_mma_catalog_enumerate(
                &t->mma, "mma", dts[d], dts[d], "fp32", -1, -1, got.data(), n, NULL);
        out.insert(out.end(), got.begin(), got.end());
    }
    return out;
}

static config_t make(int section, const char* arch)
{
    config_t c;
    memset(&c, 0, sizeof(c));
    c.section = section;
    c.arch = arch;
    return c;
}

static std::vector<config_t> build_configs()
{
    std::vector<config_t> out;
    for(int a = 0; a < 2; ++a)
        for(int ni = 0; ni < 4; ++ni)
            for(int k = 0; k < 4; ++k)
            {
                config_t c = make(SEC_CAST, ARCHES[a]);
                c.n = CAST_NS[ni];
                c.elem_f16 = strcmp(CAST_ELEM_NAME[k], "f16") == 0;
                c.str = CAST_KIND[k];
                out.push_back(c);
            }
    {
        config_t c = make(SEC_CAST, "gfx950");
        c.n = 3;
        c.elem_f16 = 0;
        c.str = "rne_noguard";
        out.push_back(c);
        c = make(SEC_CAST, "gfx942");
        c.n = 4;
        c.elem_f16 = 0;
        c.str = "bogus";
        out.push_back(c);
        c = make(SEC_CAST, "gfx942");
        c.n = 12;
        c.elem_f16 = 0;
        c.str = "rne_noguard";
        out.push_back(c);
        c = make(SEC_CAST, "gfx950");
        c.n = 6;
        c.elem_f16 = 1;
        c.str = "generic";
        out.push_back(c);
    }
    for(int a = 0; a < 2; ++a)
        for(int e = 0; e < 3; ++e)
            for(int fma = 0; fma < 2; ++fma)
            {
                config_t c = make(SEC_SOFTMAX, ARCHES[a]);
                c.str = EXP2S[e];
                c.flag = fma;
                out.push_back(c);
            }
    for(int a = 0; a < 2; ++a)
    {
        const int n_ops = (int)ops_of(ARCHES[a]).size();
        for(int op = 0; op < n_ops; ++op)
            for(int w = 0; w < 5; ++w)
            {
                config_t c = make(SEC_LAYOUT, ARCHES[a]);
                c.n = op;
                c.flag = LAYOUT_WIDTHS[w];
                out.push_back(c);
            }
    }
    for(int a = 0; a < 2; ++a)
        for(int w = 0; w < 2; ++w)
            for(int r = 0; r < 2; ++r)
                for(int q0 = 0; q0 < 3; ++q0)
                    for(int nh = 0; nh < 2; ++nh)
                    {
                        config_t c = make(SEC_STATS, ARCHES[a]);
                        c.n = STATS_WIDTHS[w];
                        c.flag = STATS_RUNS[r];
                        c.q0 = q0;
                        c.nh_int = nh;
                        c.str = "ok";
                        out.push_back(c);
                    }
    for(int v = 0; v < 6; ++v)
    {
        config_t c = make(SEC_STATS, "gfx950");
        c.n = 4;
        c.flag = 2;
        c.q0 = 2;
        c.nh_int = 0;
        c.str = VARIANTS[v];
        out.push_back(c);
        c = make(SEC_STATS, "gfx942");
        c.n = 2;
        c.flag = 1;
        c.q0 = 1;
        c.nh_int = 1;
        c.str = VARIANTS[v];
        out.push_back(c);
    }
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

static void build_cast(rocke_ir_builder_t* b, const config_t& cfg)
{
    int n = cfg.n;
    const rocke_type_t* elem = cfg.elem_f16 ? rocke_f16() : rocke_bf16();
    rocke_value_t* src = rocke_b_param(b, "src", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* out = rocke_b_param(b, "out", rocke_ptr_type(b, elem, "global"), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    std::vector<rocke_value_t*> elems((size_t)n);
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* ci = c32(b, i);
        rocke_value_t* at = rocke_b_add(b, tid, ci);
        elems[(size_t)i] = rocke_b_global_load_f32(b, src, at, 4);
    }
    rocke_value_t* frag = rocke_b_vec_pack(b, elems.data(), n, rocke_f32());
    rocke_value_t* r = rocke_attn_bwd_cast_c_frag(b, frag, elem, cfg.str);
    if(r == NULL)
        return;
    if(strcmp(cfg.str, "rne_noguard") == 0 && !cfg.elem_f16)
    {
        rocke_value_t* r2 = rocke_attn_bwd_rne_noguard_bf16(b, frag);
        r = rocke_b_vec_concat(b, r, r2);
        n *= 2;
    }
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* at = c32(b, i);
        rocke_value_t* e = rocke_b_vec_extract(b, r, i);
        rocke_b_global_store(b, out, at, e, 1);
    }
}

static void build_softmax(rocke_ir_builder_t* b, const config_t& cfg)
{
    rocke_value_t* sv_p = rocke_b_param(b, "sv", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* stats_p
        = rocke_b_param(b, "stats", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* scale_log2 = rocke_b_param(b, "scale_log2", rocke_f32(), NULL);
    rocke_value_t* out = rocke_b_param(b, "out", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* lse2[3];
    for(int k = 0; k < 3; ++k)
    {
        rocke_value_t* i0 = c32(b, 2 * k);
        lse2[k] = rocke_b_global_load_f32(b, stats_p, i0, 4);
        rocke_value_t* i1 = c32(b, 2 * k + 1);
        rocke_b_global_load_f32(b, stats_p, i1, 4); /* dsum */
    }
    rocke_value_t* neg[3] = {NULL, NULL, NULL};
    if(cfg.flag)
        rocke_attn_bwd_stats_rows_neg(b, lse2, 3, neg);
    rocke_value_t* res[3];
    for(int k = 0; k < 3; ++k)
    {
        rocke_value_t* ik = c32(b, k);
        rocke_value_t* sv = rocke_b_global_load_f32(b, sv_p, ik, 4);
        res[k] = rocke_attn_bwd_softmax_p(b, sv, scale_log2, lse2[k], neg[k], cfg.str);
        if(res[k] == NULL)
            return;
    }
    for(int i = 0; i < 3; ++i)
    {
        rocke_value_t* at = c32(b, i);
        rocke_b_global_store(b, out, at, res[i], 1);
    }
}

static void build_layout(rocke_ir_builder_t* b, const config_t& cfg)
{
    rocke_value_t* out = rocke_b_param(b, "out", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    const std::vector<const rocke_mma_op_t*> ops = ops_of(cfg.arch);
    if(!rocke_attn_bwd_check_stats_vec_layout(b, ops[(size_t)cfg.n], cfg.flag))
        return;
    rocke_value_t* at = c32(b, 0);
    rocke_value_t* w = c32(b, cfg.flag);
    rocke_b_global_store(b, out, at, w, 1);
}

static void build_stats(rocke_ir_builder_t* b, const config_t& cfg)
{
    const int width = cfg.n;
    const int runs = cfg.flag;
    const char* variant = cfg.str;
    const rocke_type_t* pf = rocke_ptr_type(b, rocke_f32(), "global");
    rocke_value_t* lse_p = rocke_b_param(b, "lse", pf, NULL);
    rocke_value_t* dsum_p = rocke_b_param(b, "dsum", pf, NULL);
    rocke_value_t* out = rocke_b_param(b, "out", pf, NULL);
    rocke_value_t* ws_rows = rocke_b_param(b, "ws_rows", rocke_i32(), NULL);
    rocke_value_t* ws_rows2 = rocke_b_param(b, "ws_rows2", rocke_i32(), NULL);
    rocke_value_t* row0 = rocke_b_param(b, "row0", rocke_i64(), NULL);
    rocke_value_t* row0_d = rocke_b_param(b, "row0_d", rocke_i64(), NULL);
    rocke_value_t* n_heads = rocke_b_param(b, "n_heads", rocke_i32(), NULL);
    rocke_value_t* h = rocke_b_param(b, "h", rocke_i32(), NULL);
    rocke_value_t* q0p = rocke_b_param(b, "q0", rocke_i32(), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    const bool int_ws = strcmp(variant, "int_ws_rows") == 0;
    const rocke_attn_bwd_iv_t ws_arg
        = int_ws ? rocke_attn_bwd_iv_int(1000) : rocke_attn_bwd_iv_val(ws_rows);
    const rocke_attn_bwd_ws_layout_t layout = strcmp(variant, "token_major") == 0
                                                  ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR
                                                  : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
    const rocke_attn_bwd_iv_t nh_arg
        = cfg.nh_int ? rocke_attn_bwd_iv_int(8) : rocke_attn_bwd_iv_val(n_heads);
    rocke_attn_bwd_workspace_addr_t lse;
    rocke_attn_bwd_workspace_addr_t dsum;
    if(!rocke_attn_bwd_workspace_addr_init(&lse, b, lse_p, ws_arg, 1, row0, layout, &nh_arg))
        return;
    const rocke_attn_bwd_iv_t dsum_rows
        = strcmp(variant, "distinct_ws_rows") == 0 ? rocke_attn_bwd_iv_val(ws_rows2) : ws_arg;
    if(!rocke_attn_bwd_workspace_addr_init(&dsum,
                                           b,
                                           dsum_p,
                                           dsum_rows,
                                           strcmp(variant, "wide_ws") == 0 ? 2 : 1,
                                           strcmp(variant, "distinct_row0") == 0 ? row0_d : row0,
                                           layout,
                                           &nh_arg))
        return;
    rocke_value_t* c16 = c32(b, 16);
    rocke_value_t* q16 = rocke_b_div(b, tid, c16);
    rocke_value_t* c4 = c32(b, 4);
    rocke_value_t* base = rocke_b_mul(b, q16, c4);
    const int n_rows = runs * width + (strcmp(variant, "bad_rows") == 0 ? 1 : 0);
    std::vector<rocke_value_t*> rows((size_t)n_rows);
    for(int i = 0; i < n_rows; ++i)
    {
        rocke_value_t* ci = c32(b, i + 4 * (i / width));
        rows[(size_t)i] = rocke_b_add(b, base, ci);
    }
    rocke_attn_bwd_iv_t q0 = cfg.q0 == 0   ? rocke_attn_bwd_iv_int(0)
                             : cfg.q0 == 1 ? rocke_attn_bwd_iv_int(5)
                                           : rocke_attn_bwd_iv_val(q0p);
    std::vector<rocke_value_t*> o_lse((size_t)n_rows);
    std::vector<rocke_value_t*> o_dsum((size_t)n_rows);
    if(!rocke_attn_bwd_batched_stats_rows(
           b, &lse, &dsum, h, q0, rows.data(), n_rows, width, nh_arg, o_lse.data(), o_dsum.data()))
        return;
    std::vector<rocke_value_t*> neg((size_t)n_rows);
    rocke_attn_bwd_stats_rows_neg(b, o_lse.data(), n_rows, neg.data());
    int k = 0;
    for(int i = 0; i < n_rows; ++i)
    {
        rocke_value_t* a0 = c32(b, k);
        rocke_b_global_store(b, out, a0, o_lse[(size_t)i], 1);
        rocke_value_t* a1 = c32(b, k + 1);
        rocke_b_global_store(b, out, a1, o_dsum[(size_t)i], 1);
        rocke_value_t* a2 = c32(b, k + 2);
        rocke_b_global_store(b, out, a2, neg[(size_t)i], 1);
        k += 3;
    }
}

/* Build the probe kernel of config `idx` into `b`. 0 on success. */
static int build(int idx, rocke_ir_builder_t* b)
{
    if(idx < 0 || idx >= (int)configs().size())
        return -1;
    const config_t cfg = configs()[(size_t)idx];
    switch(cfg.section)
    {
    case SEC_CAST:
        build_cast(b, cfg);
        break;
    case SEC_SOFTMAX:
        build_softmax(b, cfg);
        break;
    case SEC_LAYOUT:
        build_layout(b, cfg);
        break;
    default:
        build_stats(b, cfg);
        break;
    }
    if(!rocke_ir_builder_ok(b))
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
