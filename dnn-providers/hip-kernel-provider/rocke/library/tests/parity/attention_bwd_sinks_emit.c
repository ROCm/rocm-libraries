/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_sinks_emit.c -- C-side emitter for the dQ sinks of
 * the attention backward tile step. Builds, identically to the Python emitter
 * attention_bwd_sinks_emit.py, a probe kernel that drives the sinks of
 * helper_rocke.instances.common._attention_bwd_sinks.h, lowers it via
 * rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so the
 * two outputs can be byte-compared.
 *
 * argv[1] is the config index, decoded as in _configs of the Python emitter
 * (sections atomic, register, null, pending; the same loops). Optional argv[2]
 * selects the output mode:
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
#include "rocke/helper_rocke.instances.common._attention_bwd_sinks.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define PROBE_NAME "attn_bwd_sinks_probe"

enum
{
    SEC_ATOMIC,
    SEC_REGISTER,
    SEC_NULL,
    SEC_PENDING
};

/* One config; the fields a section does not use are 0 / NULL. */
typedef struct
{
    int section;
    const char* arch;
    /* atomic */
    int packed;
    int token_major;
    int thd;
    int clamp;
    int scale;
    int rows; /* 0 int, 1 value, 2 mixed */
    int reuse;
    int d;
    /* register */
    int bf16;
    int sv;
    int stride_value;
    const char* variant; /* register / null / pending */
    /* pending */
    int n;
    int with_nh;
} config_t;

static const char* const ARCHES[] = {"gfx942", "gfx950"};
static const int HEADS[] = {64, 128};
static const char* const REG_VARIANTS[] = {"one", "two", "state", "empty"};
static const char* const REG_REJECTS[]
    = {"shape_changed", "q0_changed", "head_changed", "state_len"};
static const char* const NULL_VARIANTS[]
    = {"consume", "ok", "split_slice", "bytes_ok", "bytes_round", "bytes_neg", "bytes_zero"};
static const int PENDING_NS[] = {4, 8};

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
    int k = 0;
    for(int sink = 0; sink < 2; ++sink)
        for(int layout = 0; layout < 2; ++layout)
            for(int mode = 0; mode < 2; ++mode)
                for(int clamp = 1; clamp >= 0; --clamp)
                    for(int scale = 0; scale < 2; ++scale)
                        for(int rows = 0; rows < 3; ++rows)
                        {
                            config_t c = make(SEC_ATOMIC, ARCHES[k % 2]);
                            c.packed = sink;
                            c.token_major = layout;
                            c.thd = mode;
                            c.clamp = clamp;
                            c.scale = scale;
                            c.rows = rows;
                            c.reuse = k % 3;
                            c.d = HEADS[(k / 2) % 2];
                            out.push_back(c);
                            ++k;
                        }
    k = 0;
    for(int dt = 0; dt < 2; ++dt)
        for(int mode = 0; mode < 2; ++mode)
            for(int scale = 0; scale < 2; ++scale)
                for(int sv = 0; sv < 2; ++sv)
                    for(int strides = 0; strides < 2; ++strides)
                        for(int rows = 0; rows < 2; ++rows)
                        {
                            config_t c = make(SEC_REGISTER, ARCHES[k % 2]);
                            c.bf16 = dt;
                            c.thd = mode;
                            c.scale = scale;
                            c.sv = sv == 0 ? 8 : 1;
                            c.stride_value = strides;
                            c.rows = rows;
                            c.variant = REG_VARIANTS[k % 4];
                            out.push_back(c);
                            ++k;
                        }
    for(int i = 0; i < 4; ++i)
    {
        config_t c = make(SEC_REGISTER, ARCHES[i % 2]);
        c.bf16 = i % 2;
        c.thd = (i / 2) % 2;
        c.scale = i % 2;
        c.sv = 8;
        c.stride_value = 0;
        c.rows = i % 2;
        c.variant = REG_REJECTS[i];
        out.push_back(c);
    }
    for(int i = 0; i < 7; ++i)
    {
        config_t c = make(SEC_NULL, ARCHES[i % 2]);
        c.variant = NULL_VARIANTS[i];
        out.push_back(c);
    }
    for(int i = 0; i < 2; ++i)
        for(int nh = 0; nh < 2; ++nh)
        {
            config_t c = make(SEC_PENDING, ARCHES[(i + nh) % 2]);
            c.n = PENDING_NS[i];
            c.with_nh = nh;
            c.variant = "ok";
            out.push_back(c);
        }
    {
        config_t c = make(SEC_PENDING, "gfx950");
        c.n = 4;
        c.with_nh = 1;
        c.variant = "bad_len";
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

/* The kernel parameters every probe shares (Python _Params). */
typedef struct
{
    rocke_value_t *dq, *cu, *src, *outp, *s_max, *ws_rows, *n_heads, *h, *h2, *scale, *q0, *s_t;
    rocke_value_t *pk_head0, *pk_n, *pk_rph, *pk_row0;
} params_t;

static params_t make_params(rocke_ir_builder_t* b, const rocke_type_t* elem)
{
    params_t p;
    const rocke_type_t* pf = rocke_ptr_type(b, rocke_f32(), "global");
    p.dq = rocke_b_param(b, "dq", pf, NULL);
    p.cu = rocke_b_param(b, "cu", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    p.src = rocke_b_param(b, "src", pf, NULL);
    p.outp = rocke_b_param(b, "outp", rocke_ptr_type(b, elem, "global"), NULL);
    p.s_max = rocke_b_param(b, "s_max", rocke_i32(), NULL);
    p.ws_rows = rocke_b_param(b, "ws_rows", rocke_i32(), NULL);
    p.n_heads = rocke_b_param(b, "n_heads", rocke_i32(), NULL);
    p.h = rocke_b_param(b, "h", rocke_i32(), NULL);
    p.h2 = rocke_b_param(b, "h2", rocke_i32(), NULL);
    p.scale = rocke_b_param(b, "scale", rocke_f32(), NULL);
    p.q0 = rocke_b_param(b, "q0", rocke_i32(), NULL);
    p.s_t = rocke_b_param(b, "s_t", rocke_i32(), NULL);
    p.pk_head0 = rocke_b_param(b, "pk_head0", rocke_i32(), NULL);
    p.pk_n = rocke_b_param(b, "pk_n", rocke_i32(), NULL);
    p.pk_rph = rocke_b_param(b, "pk_rph", rocke_i32(), NULL);
    p.pk_row0 = rocke_b_param(b, "pk_row0", rocke_i32(), NULL);
    return p;
}

static rocke_attn_bwd_seq_info_t
    make_seq(rocke_ir_builder_t* b, int* guards, bool thd, const params_t& p)
{
    rocke_value_t* bid = rocke_b_block_id_x(b);
    rocke_attn_bwd_decode_args_t args = rocke_attn_bwd_decode_args_default();
    args.s_max = rocke_attn_bwd_iv_val(p.s_max);
    if(!thd)
        return rocke_attn_bwd_decode_batched(b, guards, bid, &args);
    args.off_ptr = p.cu;
    return rocke_attn_bwd_decode_thd(b, guards, bid, &args);
}

/* Python _frag: a dQ fragment whose slot rows are ints, SSA values or a mix. */
static std::vector<rocke_attn_bwd_dq_slot_t> make_frag(
    rocke_ir_builder_t* b, const params_t& p, int kind, int shift, const int* cols, int n_cols)
{
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* c8 = c32(b, 8);
    rocke_value_t* col_v = rocke_b_mul(b, tid, c8);
    rocke_value_t* row_vals[6] = {NULL, NULL, NULL, NULL, NULL, NULL}; /* by r in {0,1,4,5} */
    std::vector<rocke_attn_bwd_dq_slot_t> slots;
    int n = 0;
    const int RS[4] = {0, 1, 4, 5};
    for(int ri = 0; ri < 4; ++ri)
    {
        const int r = RS[ri];
        for(int ci = 0; ci < n_cols; ++ci)
        {
            rocke_attn_bwd_iv_t row;
            if(kind == 0)
            {
                row = rocke_attn_bwd_iv_int(r + shift);
            }
            else if(kind == 1 || r % 2 != 0)
            {
                if(row_vals[r] == NULL)
                {
                    rocke_value_t* cr = c32(b, r + shift);
                    row_vals[r] = rocke_b_add(b, tid, cr);
                }
                row = rocke_attn_bwd_iv_val(row_vals[r]);
            }
            else
            {
                row = rocke_attn_bwd_iv_int(r + shift);
            }
            const rocke_attn_bwd_iv_t col = (kind != 0 && n == 3) ? rocke_attn_bwd_iv_val(col_v)
                                                                  : rocke_attn_bwd_iv_int(cols[ci]);
            rocke_value_t* ci32 = c32(b, n + 16 * shift);
            rocke_value_t* v = rocke_b_global_load_f32(b, p.src, ci32, 4);
            rocke_attn_bwd_dq_slot_t s;
            s.row = row;
            s.col = col;
            s.value = v;
            slots.push_back(s);
            ++n;
        }
    }
    return slots;
}

static bool build_atomic(rocke_ir_builder_t* b, const config_t& cfg)
{
    params_t p = make_params(b, rocke_f32());
    int guards = 0;
    rocke_attn_bwd_seq_info_t seq = make_seq(b, &guards, cfg.thd, p);
    rocke_attn_bwd_workspace_addr_t ws;
    const rocke_attn_bwd_iv_t nh = rocke_attn_bwd_iv_val(p.n_heads);
    if(!rocke_attn_bwd_workspace_addr_init(&ws,
                                           b,
                                           p.dq,
                                           rocke_attn_bwd_iv_val(p.ws_rows),
                                           cfg.d,
                                           seq.ws_row0,
                                           cfg.token_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR
                                                           : ROCKE_ATTN_BWD_WS_HEAD_MAJOR,
                                           &nh))
        return false;
    rocke_attn_bwd_dq_sink_t sink;
    rocke_value_t* scale = cfg.scale ? p.scale : NULL;
    if(cfg.packed)
        rocke_attn_bwd_dq_sink_init_packed_atomic(&sink, b, &ws, &seq, scale);
    else
        rocke_attn_bwd_dq_sink_init_atomic(&sink, b, &ws, &seq, scale);
    const int cols[3] = {0, 8, 16};
    std::vector<rocke_attn_bwd_dq_slot_t> frag = make_frag(b, p, cfg.rows, 0, cols, 3);
    std::vector<rocke_attn_bwd_dq_slot_t> frag2 = make_frag(b, p, cfg.rows, 2, cols, 3);
    if(cfg.reuse == 2)
        rocke_attn_bwd_dq_sink_begin_head(&sink, b, p.h);
    bool ok = true;
    if(cfg.packed)
    {
        rocke_attn_bwd_packed_rows_t rows;
        rows.head0 = p.pk_head0;
        rows.n_heads = p.pk_n;
        rows.rows_per_head = p.pk_rph;
        rows.row0 = p.pk_row0;
        ok = rocke_attn_bwd_dq_sink_consume(
                 &sink, b, frag.data(), (int)frag.size(), NULL, &rows, NULL, cfg.clamp)
             && rocke_attn_bwd_dq_sink_consume(
                 &sink, b, frag2.data(), (int)frag2.size(), NULL, &rows, NULL, cfg.clamp);
    }
    else
    {
        rocke_value_t* h_b = cfg.reuse == 1 ? p.h2 : p.h;
        ok = rocke_attn_bwd_dq_sink_consume(
                 &sink, b, frag.data(), (int)frag.size(), p.q0, NULL, p.h, cfg.clamp)
             && rocke_attn_bwd_dq_sink_consume(
                 &sink, b, frag2.data(), (int)frag2.size(), p.q0, NULL, h_b, cfg.clamp);
    }
    ok = ok && rocke_attn_bwd_dq_sink_finish(&sink, b);
    rocke_attn_bwd_dq_sink_free(&sink);
    return ok;
}

static bool build_register(rocke_ir_builder_t* b, const config_t& cfg)
{
    const rocke_type_t* elem = cfg.bf16 ? rocke_bf16() : rocke_f16();
    params_t p = make_params(b, elem);
    int guards = 0;
    rocke_attn_bwd_seq_info_t seq = make_seq(b, &guards, cfg.thd, p);
    const rocke_attn_bwd_iv_t strides[3]
        = {rocke_attn_bwd_iv_int(65536),
           rocke_attn_bwd_iv_int(64),
           cfg.stride_value ? rocke_attn_bwd_iv_val(p.s_t) : rocke_attn_bwd_iv_int(512)};
    rocke_attn_bwd_strided_tensor_addr_t out;
    if(!rocke_attn_bwd_strided_tensor_addr_init(&out, b, p.outp, elem, strides, &seq, cfg.sv, 1))
        return false;
    rocke_attn_bwd_dq_sink_t sink;
    if(!rocke_attn_bwd_dq_sink_init_register(&sink, b, &out, cfg.scale ? p.scale : NULL))
        return false;
    const int cols[2] = {0, 8};
    std::vector<rocke_attn_bwd_dq_slot_t> frag = make_frag(b, p, cfg.rows, 0, cols, 2);
    /* the same rows and columns (the sink keys them by identity), new values */
    std::vector<rocke_attn_bwd_dq_slot_t> frag2 = frag;
    for(size_t i = 0; i < frag2.size(); ++i)
    {
        rocke_value_t* ci = c32(b, 32 + (int64_t)i);
        frag2[i].value = rocke_b_global_load_f32(b, p.src, ci, 4);
    }
    const char* v = cfg.variant;
    bool ok;
    if(strcmp(v, "empty") == 0)
        ok = rocke_attn_bwd_dq_sink_consume(&sink, b, NULL, 0, p.q0, NULL, p.h, true);
    else
        ok = rocke_attn_bwd_dq_sink_consume(
            &sink, b, frag.data(), (int)frag.size(), p.q0, NULL, p.h, true);
    if(ok && strcmp(v, "two") == 0)
        ok = rocke_attn_bwd_dq_sink_consume(
            &sink, b, frag2.data(), (int)frag2.size(), p.q0, NULL, p.h, true);
    else if(ok && strcmp(v, "state") == 0)
    {
        std::vector<rocke_value_t*> st((size_t)rocke_attn_bwd_dq_sink_register_state_len(&sink));
        rocke_attn_bwd_dq_sink_register_state(&sink, st.data());
        std::vector<rocke_value_t*> ns(st.size());
        for(size_t i = 0; i < st.size(); ++i)
            ns[i] = rocke_b_fmul(b, st[i], p.scale);
        ok = rocke_attn_bwd_dq_sink_register_set_state(&sink, b, ns.data(), (int)ns.size());
    }
    else if(ok && strcmp(v, "shape_changed") == 0)
        ok = rocke_attn_bwd_dq_sink_consume(
            &sink, b, frag2.data(), (int)frag2.size() - 1, p.q0, NULL, p.h, true);
    else if(ok && strcmp(v, "q0_changed") == 0)
    {
        rocke_value_t* c1 = c32(b, 1);
        rocke_value_t* q1 = rocke_b_add(b, p.q0, c1);
        ok = rocke_attn_bwd_dq_sink_consume(
            &sink, b, frag2.data(), (int)frag2.size(), q1, NULL, p.h, true);
    }
    else if(ok && strcmp(v, "head_changed") == 0)
        ok = rocke_attn_bwd_dq_sink_consume(
            &sink, b, frag2.data(), (int)frag2.size(), p.q0, NULL, p.h2, true);
    else if(ok && strcmp(v, "state_len") == 0)
    {
        std::vector<rocke_value_t*> st((size_t)rocke_attn_bwd_dq_sink_register_state_len(&sink));
        rocke_attn_bwd_dq_sink_register_state(&sink, st.data());
        ok = rocke_attn_bwd_dq_sink_register_set_state(&sink, b, st.data(), (int)st.size() - 1);
    }
    ok = ok && rocke_attn_bwd_dq_sink_finish(&sink, b);
    rocke_attn_bwd_dq_sink_free(&sink);
    return ok;
}

static bool build_null(rocke_ir_builder_t* b, const config_t& cfg)
{
    const char* v = cfg.variant;
    rocke_value_t* out = rocke_b_param(b, "out", rocke_ptr_type(b, rocke_i64(), "global"), NULL);
    rocke_attn_bwd_dq_sink_t sink;
    if(strcmp(v, "consume") == 0)
    {
        rocke_attn_bwd_dq_sink_init_null(&sink);
        rocke_value_t* z0 = c32(b, 0);
        rocke_value_t* z1 = c32(b, 0);
        return rocke_attn_bwd_dq_sink_consume(&sink, b, NULL, 0, z0, NULL, z1, true);
    }
    if(strcmp(v, "ok") == 0)
    {
        rocke_attn_bwd_dq_sink_init_null(&sink);
        rocke_value_t* z = c32(b, 0);
        rocke_attn_bwd_dq_sink_begin_head(&sink, b, z);
        rocke_attn_bwd_dq_sink_finish(&sink, b);
        rocke_value_t* at = c32(b, 0);
        rocke_value_t* g4 = rocke_b_const_i64(b, sink.emits_g4 ? 1 : 0);
        rocke_b_global_store(b, out, at, g4, 1);
        return true;
    }
    if(strcmp(v, "split_slice") == 0)
        return rocke_attn_bwd_dq_sink_init_split_slice(b);
    int64_t n = 0;
    bool ok;
    if(strcmp(v, "bytes_ok") == 0)
        ok = rocke_attn_bwd_split_slice_workspace_bytes(b, 3, 2, 100, 8, 64, &n);
    else if(strcmp(v, "bytes_round") == 0)
        ok = rocke_attn_bwd_split_slice_workspace_bytes(b, 1, 1, 1, 1, 3, &n);
    else if(strcmp(v, "bytes_zero") == 0)
        ok = rocke_attn_bwd_split_slice_workspace_bytes(b, (int64_t)1 << 62, 4, 0, 8, 64, &n);
    else
        ok = rocke_attn_bwd_split_slice_workspace_bytes(b, -1, 2, 100, 8, 64, &n);
    if(!ok || strcmp(v, "bytes_neg") == 0)
        return false;
    rocke_value_t* at = c32(b, 0);
    rocke_value_t* cn = rocke_b_const_i64(b, n);
    rocke_b_global_store(b, out, at, cn, 1);
    return true;
}

static bool build_pending(rocke_ir_builder_t* b, const config_t& cfg)
{
    const int n = cfg.n;
    params_t p = make_params(b, rocke_f32());
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    std::vector<rocke_value_t*> values((size_t)n);
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* ci = c32(b, i);
        values[(size_t)i] = rocke_b_global_load_f32(b, p.src, ci, 4);
    }
    rocke_attn_bwd_pending_dq_t pd;
    pd.values = values.data();
    pd.n_values = n;
    pd.q0 = p.q0;
    pd.h = p.h;
    pd.n_heads = cfg.with_nh ? p.pk_n : NULL;
    pd.rows_per_head = cfg.with_nh ? p.pk_rph : NULL;
    rocke_value_t* zero = c32(b, 0);
    rocke_value_t* c2 = c32(b, 2);
    rocke_value_t* trip = rocke_b_add(b, tid, c2);
    const int flat_len = rocke_attn_bwd_pending_dq_flat_len(&pd);
    std::vector<rocke_value_t*> flat((size_t)flat_len);
    rocke_attn_bwd_pending_dq_flat(&pd, flat.data());
    std::vector<std::vector<char>> names((size_t)flat_len, std::vector<char>(64));
    std::vector<rocke_iter_arg_t> iter_args((size_t)flat_len);
    for(int i = 0; i < flat_len; ++i)
    {
        if(!rocke_attn_bwd_pending_dq_name(&pd, "p_", i, names[(size_t)i].data(), 64))
            return false;
        iter_args[(size_t)i].name = names[(size_t)i].data();
        iter_args[(size_t)i].init = flat[(size_t)i];
    }
    rocke_value_t* c1 = c32(b, 1);
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, zero, trip, c1, iter_args.data(), flat_len, "qi", false, true);
    if(loop.op == NULL)
        return false;
    rocke_b_region_enter(b, loop.body);
    {
        std::vector<rocke_value_t*> cur(loop.iter_vars, loop.iter_vars + flat_len);
        int cur_n = flat_len;
        if(strcmp(cfg.variant, "bad_len") == 0)
            cur_n -= 1;
        rocke_attn_bwd_pending_dq_t nxt;
        if(!rocke_attn_bwd_pending_dq_from_flat(b, &pd, cur.data(), cur_n, &nxt))
            return false;
        std::vector<rocke_value_t*> y((size_t)rocke_attn_bwd_pending_dq_flat_len(&nxt));
        rocke_attn_bwd_pending_dq_flat(&nxt, y.data());
        rocke_b_scf_yield(b, y.data(), (int)y.size());
    }
    rocke_b_region_leave(b);
    rocke_attn_bwd_pending_dq_t done;
    if(!rocke_attn_bwd_pending_dq_from_flat(b, &pd, loop.op->results, loop.op->num_results, &done))
        return false;
    rocke_value_t* bid = rocke_b_block_id_x(b);
    int guards = 0;
    rocke_attn_bwd_decode_args_t args = rocke_attn_bwd_decode_args_default();
    args.s_max = rocke_attn_bwd_iv_val(p.s_max);
    rocke_attn_bwd_seq_info_t seq = rocke_attn_bwd_decode_batched(b, &guards, bid, &args);
    rocke_attn_bwd_workspace_addr_t ws;
    const rocke_attn_bwd_iv_t nh = rocke_attn_bwd_iv_val(p.n_heads);
    if(!rocke_attn_bwd_workspace_addr_init(&ws,
                                           b,
                                           p.dq,
                                           rocke_attn_bwd_iv_val(p.ws_rows),
                                           64,
                                           seq.ws_row0,
                                           ROCKE_ATTN_BWD_WS_HEAD_MAJOR,
                                           &nh))
        return false;
    rocke_attn_bwd_dq_sink_t sink;
    rocke_attn_bwd_dq_sink_init_atomic(&sink, b, &ws, &seq, p.scale);
    std::vector<rocke_attn_bwd_dq_slot_t> slots((size_t)done.n_values);
    for(int i = 0; i < done.n_values; ++i)
    {
        slots[(size_t)i].row = rocke_attn_bwd_iv_int(i / 2);
        slots[(size_t)i].col = rocke_attn_bwd_iv_int(8 * (i % 2));
        slots[(size_t)i].value = done.values[i];
    }
    bool ok = rocke_attn_bwd_dq_sink_consume(
        &sink, b, slots.data(), (int)slots.size(), done.q0, NULL, done.h, true);
    ok = ok && rocke_attn_bwd_dq_sink_finish(&sink, b);
    rocke_attn_bwd_dq_sink_free(&sink);
    return ok;
}

/* Build the probe kernel of config `idx` into `b`. 0 on success. */
static int build(int idx, rocke_ir_builder_t* b)
{
    if(idx < 0 || idx >= (int)configs().size())
        return -1;
    const config_t cfg = configs()[(size_t)idx];
    bool ok;
    switch(cfg.section)
    {
    case SEC_ATOMIC:
        ok = build_atomic(b, cfg);
        break;
    case SEC_REGISTER:
        ok = build_register(b, cfg);
        break;
    case SEC_NULL:
        ok = build_null(b, cfg);
        break;
    default:
        ok = build_pending(b, cfg);
        break;
    }
    if(!ok || !rocke_ir_builder_ok(b))
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
