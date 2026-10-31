// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/attention_bwd_aux.py: the prep, convert,
 * reduce, page_zero and page_convert kernels of the attention backward.
 *
 * Every op is created in the Python evaluation order: a nested Python call
 * `f(g(), h())` becomes `x = g(); y = h(); f(x, y)` here, never two builder
 * calls in one argument list (C++ leaves their order unspecified). Comments
 * `py:` quote the Python expression a block reproduces.
 *
 * See rocke/instance_attention_bwd_aux.h for the mapping.
 */
#include "rocke/instance_attention_bwd_aux.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "rocke/arch_target.h" /* rocke_arch_target_from_gfx */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* rocke.helpers.attention_fwd_ext.LOG2E. Defined here until the forward
 * helper has its C twin (the port of attention_fwd_ext is separate work). */
#define ROCKE_ATTN_BWD_LOG2E 1.4426950408889634

/* The lowering's flat-work-group-size attribute has a fixed minimum of 64. */
#define AUX_MIN_WG_ATTR 64
#define AUX_MAX_BLOCK 1024
/* Widest CONVERT_VECS entry: sizes the per-lane value buffers of the
 * convert / reduce. read_sum refuses a wider convert_vec instead of
 * overflowing them. */
#define AUX_MAX_CONVERT_VEC 16

/* ====================================================================== *
 * Spec
 * ====================================================================== */

rocke_attn_bwd_aux_spec_t rocke_attn_bwd_aux_spec_default(void)
{
    rocke_attn_bwd_aux_spec_t s;
    memset(&s, 0, sizeof(s));
    s.head_size = 0;
    s.dtype = "fp16";
    s.seq_mode = "batched";
    s.stage_vec = 8;
    s.ws_layout = "head_major";
    s.kv_target = "contiguous";
    s.has_rows_per_cta = false;
    s.rows_per_cta = 0;
    s.convert_vec = 8;
    s.worklist = false;
    s.narrow_vec = 1;
    s.name = "rocke_attn_bwd";
    return s;
}

static bool streq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

static bool fail(char* reason, size_t cap, const char* msg)
{
    if(reason != NULL && cap > 0)
        snprintf(reason, cap, "%s", msg);
    return false;
}

bool rocke_attn_bwd_aux_spec_validate(const rocke_attn_bwd_aux_spec_t* spec,
                                      char* reason,
                                      size_t reason_cap)
{
    if(spec == NULL)
        return fail(reason, reason_cap, "NULL spec");
    if(spec->name == NULL)
        return fail(reason, reason_cap, "name must be a str");
    if(spec->head_size != 32 && spec->head_size != 64 && spec->head_size != 128)
        return fail(reason, reason_cap, "head_size must be one of (32, 64, 128)");
    if(!streq(spec->dtype, "fp16") && !streq(spec->dtype, "bf16"))
        return fail(reason, reason_cap, "dtype must be one of ('fp16', 'bf16')");
    if(!streq(spec->seq_mode, "batched") && !streq(spec->seq_mode, "thd"))
        return fail(reason, reason_cap, "seq_mode must be one of ('batched', 'thd')");
    if(spec->stage_vec != 8 && spec->stage_vec != 1)
        return fail(reason, reason_cap, "stage_vec must be 8 or 1");
    if(!streq(spec->ws_layout, "head_major") && !streq(spec->ws_layout, "token_major"))
        return fail(reason, reason_cap, "ws_layout must be one of ('head_major', 'token_major')");
    if(!streq(spec->kv_target, "contiguous") && !streq(spec->kv_target, "paged"))
        return fail(reason, reason_cap, "kv_target must be one of ('contiguous', 'paged')");
    if(spec->has_rows_per_cta && spec->rows_per_cta < 1)
        return fail(reason, reason_cap, "rows_per_cta must be positive");
    if(spec->convert_vec != 8 && spec->convert_vec != 4 && spec->convert_vec != 16)
        return fail(reason, reason_cap, "convert_vec must be one of (8, 4, 16)");
    if(spec->worklist && (!streq(spec->seq_mode, "thd") || !streq(spec->kv_target, "contiguous")))
        return fail(
            reason, reason_cap, "worklist=True needs seq_mode='thd' and kv_target='contiguous'");
    if((spec->narrow_vec != 1 && spec->narrow_vec != 2 && spec->narrow_vec != 4)
       || (spec->narrow_vec != 1 && spec->stage_vec != 1))
        return fail(reason, reason_cap, "narrow_vec must be 1, 2 or 4 (not 1: stage_vec = 1 only)");
    return true;
}

static bool is_stage(const char* stage)
{
    return streq(stage, "prep") || streq(stage, "convert") || streq(stage, "page_zero")
           || streq(stage, "page_convert") || streq(stage, "reduce");
}

static void append(char* out, size_t cap, size_t* len, const char* part, bool* overflow)
{
    int n = snprintf(out + *len, cap - *len, "%s%s", *len ? "_" : "", part);
    if(n < 0 || (size_t)n >= cap - *len)
    {
        *overflow = true;
        return;
    }
    *len += (size_t)n;
}

rocke_status_t rocke_attn_bwd_aux_kernel_name(const rocke_attn_bwd_aux_spec_t* spec,
                                              const char* stage,
                                              char* out,
                                              size_t out_cap)
{
    if(spec == NULL || out == NULL || out_cap == 0 || !is_stage(stage))
        return ROCKE_ERR_VALUE;
    size_t len = 0;
    bool overflow = false;
    char buf[32];
    out[0] = '\0';
    append(out, out_cap, &len, spec->name, &overflow);
    append(out, out_cap, &len, stage, &overflow);
    append(out, out_cap, &len, spec->dtype, &overflow);
    snprintf(buf, sizeof(buf), "d%d", spec->head_size);
    append(out, out_cap, &len, buf, &overflow);
    if(streq(stage, "prep") || streq(stage, "convert") || streq(stage, "reduce"))
        append(out, out_cap, &len, spec->seq_mode, &overflow);
    if(spec->stage_vec != 8)
    {
        snprintf(buf, sizeof(buf), "sv%d", spec->stage_vec);
        append(out, out_cap, &len, buf, &overflow);
    }
    if(!streq(spec->ws_layout, "head_major"))
        append(out, out_cap, &len, "tm", &overflow);
    if(streq(stage, "prep") && !streq(spec->kv_target, "contiguous"))
        append(out, out_cap, &len, "paged", &overflow);
    if(spec->has_rows_per_cta)
    {
        snprintf(buf, sizeof(buf), "r%d", spec->rows_per_cta);
        append(out, out_cap, &len, buf, &overflow);
    }
    if(spec->convert_vec != 8)
    {
        snprintf(buf, sizeof(buf), "v%d", spec->convert_vec);
        append(out, out_cap, &len, buf, &overflow);
    }
    if(spec->worklist)
        append(out, out_cap, &len, "wl", &overflow);
    if(spec->narrow_vec != 1)
    {
        snprintf(buf, sizeof(buf), "nv%d", spec->narrow_vec);
        append(out, out_cap, &len, buf, &overflow);
    }
    return overflow ? ROCKE_ERR_VALUE : ROCKE_OK;
}

static int lanes_per_row(const rocke_attn_bwd_aux_spec_t* spec)
{
    return spec->head_size / spec->convert_vec;
}

int rocke_attn_bwd_aux_rows_per_cta(const rocke_attn_bwd_aux_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap)
{
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
    if(t == NULL)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "unknown arch '%s'", arch ? arch : "(null)");
        return -1;
    }
    const int wave = t->wave_size;
    const int lanes = lanes_per_row(spec);
    if(!spec->has_rows_per_cta)
        return (wave > AUX_MIN_WG_ATTR ? wave : AUX_MIN_WG_ATTR) / lanes;
    /* 64-bit: Python ints do not overflow, a huge rows_per_cta must fail the
     * size check rather than wrap. */
    const long long threads = (long long)spec->rows_per_cta * lanes;
    if(threads % wave || threads > AUX_MAX_BLOCK)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "rows_per_cta=%d gives %lld threads; need a multiple of the wave size %d "
                     "and at most %d",
                     spec->rows_per_cta,
                     threads,
                     wave,
                     AUX_MAX_BLOCK);
        return -1;
    }
    return spec->rows_per_cta;
}

int rocke_attn_bwd_aux_block_threads(const rocke_attn_bwd_aux_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap)
{
    const int rows = rocke_attn_bwd_aux_rows_per_cta(spec, arch, reason, reason_cap);
    return rows < 0 ? -1 : rows * lanes_per_row(spec);
}

/* ====================================================================== *
 * Kernel arguments
 * ====================================================================== */

/* Parameter lists in kernel-argument order: pointers, i64, f32, i32. */
static const char* const PREP_PTRS[] = {"O",
                                        "dO",
                                        "LSE",
                                        "SEQ_Q",
                                        "SEQ_KV",
                                        "OFF_Q",
                                        "OFF_KV",
                                        "WS_LSE2",
                                        "WS_DSUM",
                                        "WS_DQ",
                                        "WS_DK",
                                        "WS_DV",
                                        "WORKLIST",
                                        NULL};
static const char* const PREP_I64[] = {"o_b", "o_h", "do_b", "do_h", "l_b", "l_h", NULL};
static const char* const PREP_I32[]
    = {"o_t",     "do_t",     "l_t",          "h_q",        "h_k",        "h_v",
       "S_q_max", "S_kv_max", "has_len",      "len_stride", "off64",      "q_mult",
       "q_div",   "kv_mult",  "kv_div",       "ws_rows_q",  "ws_rows_kv", "ws_seg",
       "zero_kv", "zero_dq",  "use_worklist", "n_batch",    "wl_kn0",     NULL};
static const char* const CONVERT_PTRS[] = {"SRC", "DST", "SEQ", "OFF", NULL};
static const char* const CONVERT_I64[] = {"d_b", "d_h", NULL};
static const char* const CONVERT_I32[] = {"H",
                                          "d_t",
                                          "S_max",
                                          "has_len",
                                          "len_stride",
                                          "off64",
                                          "tok_mult",
                                          "tok_div",
                                          "ws_rows",
                                          "ws_seg",
                                          NULL};
static const char* const REDUCE_I64[] = {"d_b", "d_h", "s_stride", NULL};
static const char* const REDUCE_I32[] = {"H",
                                         "d_t",
                                         "S_max",
                                         "has_len",
                                         "len_stride",
                                         "off64",
                                         "tok_mult",
                                         "tok_div",
                                         "ws_rows",
                                         "ws_seg",
                                         "n_slices",
                                         NULL};
static const char* const PAGE_ZERO_PTRS[]
    = {"SEQ_LENS_KV", "BLOCK_TABLE", "WS_PDK", "WS_PDV", NULL};
static const char* const PAGE_I32[]
    = {"h_kv", "num_seqs", "max_blocks_per_seq", "bt_stride", "page", "page_rows", NULL};
static const char* const PAGE_CONVERT_PTRS[] = {"SRC", "SEQ_LENS_KV", "BLOCK_TABLE", "DST", NULL};
static const char* const PAGE_CONVERT_I32[] = {"h_kv",
                                               "num_seqs",
                                               "max_blocks_per_seq",
                                               "bt_stride",
                                               "page",
                                               "page_rows",
                                               "d_blk",
                                               "d_tok",
                                               "d_h",
                                               NULL};
static const char* const MULT_F32[] = {"mult", NULL};
static const char* const NONE[] = {NULL};

static const char* const READONLY_PREP[]
    = {"O", "dO", "LSE", "SEQ_Q", "SEQ_KV", "OFF_Q", "OFF_KV", NULL};
static const char* const READONLY_CONVERT[] = {"SRC", "SEQ", "OFF", NULL};
static const char* const READONLY_PAGE_ZERO[] = {"SEQ_LENS_KV", "BLOCK_TABLE", NULL};
static const char* const READONLY_PAGE_CONVERT[] = {"SRC", "SEQ_LENS_KV", "BLOCK_TABLE", NULL};

static bool in_list(const char* const* list, const char* name)
{
    for(; *list != NULL; ++list)
        if(strcmp(*list, name) == 0)
            return true;
    return false;
}

static const rocke_type_t* io_type(const rocke_attn_bwd_aux_spec_t* spec)
{
    return streq(spec->dtype, "bf16") ? rocke_bf16() : rocke_f16();
}

static const rocke_type_t* ptr_type(rocke_ir_builder_t* b, const char* name, const rocke_type_t* io)
{
    if(streq(name, "O") || streq(name, "dO") || streq(name, "DST"))
        return rocke_ptr_type(b, io, "global");
    static const char* const I32_PTRS[] = {"SEQ_Q",
                                           "SEQ_KV",
                                           "OFF_Q",
                                           "OFF_KV",
                                           "SEQ",
                                           "OFF",
                                           "WORKLIST",
                                           "SEQ_LENS_KV",
                                           "BLOCK_TABLE",
                                           NULL};
    if(in_list(I32_PTRS, name))
        return rocke_ptr_type(b, rocke_i32(), "global");
    return rocke_ptr_type(b, rocke_f32(), "global");
}

/* _declare: every kernel argument of `stage`, in attn_bwd_aux_params order. */
static void declare(rocke_ir_builder_t* b, const char* stage, const rocke_attn_bwd_aux_spec_t* spec)
{
    const char* const* ptrs = NONE;
    const char* const* i64s = NONE;
    const char* const* f32s = NONE;
    const char* const* i32s = NONE;
    const char* const* ro = NONE;
    if(streq(stage, "prep"))
    {
        ptrs = PREP_PTRS;
        i64s = PREP_I64;
        i32s = PREP_I32;
        ro = READONLY_PREP;
    }
    else if(streq(stage, "convert"))
    {
        ptrs = CONVERT_PTRS;
        i64s = CONVERT_I64;
        f32s = MULT_F32;
        i32s = CONVERT_I32;
        ro = READONLY_CONVERT;
    }
    else if(streq(stage, "reduce"))
    {
        ptrs = CONVERT_PTRS;
        i64s = REDUCE_I64;
        f32s = MULT_F32;
        i32s = REDUCE_I32;
        ro = READONLY_CONVERT;
    }
    else if(streq(stage, "page_zero"))
    {
        ptrs = PAGE_ZERO_PTRS;
        i32s = PAGE_I32;
        ro = READONLY_PAGE_ZERO;
    }
    else
    {
        ptrs = PAGE_CONVERT_PTRS;
        f32s = MULT_F32;
        i32s = PAGE_CONVERT_I32;
        ro = READONLY_PAGE_CONVERT;
    }
    const rocke_type_t* io = io_type(spec);
    rocke_param_opts_t ro_opts;
    memset(&ro_opts, 0, sizeof(ro_opts));
    ro_opts.readonly = true;
    ro_opts.readonly_set = true;
    for(const char* const* n = ptrs; *n != NULL; ++n)
        rocke_b_param(b, *n, ptr_type(b, *n, io), in_list(ro, *n) ? &ro_opts : NULL);
    for(const char* const* n = i64s; *n != NULL; ++n)
        rocke_b_param(b, *n, rocke_i64(), NULL);
    for(const char* const* n = f32s; *n != NULL; ++n)
        rocke_b_param(b, *n, rocke_f32(), NULL);
    for(const char* const* n = i32s; *n != NULL; ++n)
        rocke_b_param(b, *n, rocke_i32(), NULL);
}

/* a["name"] */
static rocke_value_t* A(rocke_ir_builder_t* b, const char* name)
{
    return rocke_b_get_param(b, name);
}

static rocke_attn_bwd_iv_t V(rocke_value_t* v)
{
    return rocke_attn_bwd_iv_val(v);
}

static rocke_attn_bwd_iv_t I(int64_t i)
{
    return rocke_attn_bwd_iv_int(i);
}

/* ====================================================================== *
 * Emission helpers
 * ====================================================================== */

/* _Lanes: D / v lanes per row (v = convert_vec). */
typedef struct lanes
{
    int lanes;
    rocke_value_t* row;
    rocke_value_t* seg;
    rocke_value_t* col0;
    rocke_value_t* tile_row0;
    rocke_value_t* r;
} lanes_t;

static lanes_t make_lanes(rocke_ir_builder_t* b, const rocke_attn_bwd_aux_spec_t* spec, int rows)
{
    lanes_t ln;
    const int vec = spec->convert_vec;
    ln.lanes = spec->head_size / vec;
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* c_lanes = rocke_b_const_i32(b, ln.lanes);
    ln.row = rocke_b_div(b, tid, c_lanes);
    rocke_value_t* c_lanes2 = rocke_b_const_i32(b, ln.lanes);
    ln.seg = rocke_b_mod(b, tid, c_lanes2);
    rocke_value_t* c_vec = rocke_b_const_i32(b, vec);
    ln.col0 = rocke_b_mul(b, ln.seg, c_vec);
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* c_rows = rocke_b_const_i32(b, rows);
    ln.tile_row0 = rocke_b_mul(b, bx, c_rows);
    ln.r = rocke_b_add(b, ln.tile_row0, ln.row);
    return ln;
}

static rocke_value_t* zero_vec8(rocke_ir_builder_t* b)
{
    rocke_value_t* z = rocke_b_const_f32(b, 0.0);
    rocke_value_t* comps[8] = {z, z, z, z, z, z, z, z};
    return rocke_b_vec_pack(b, comps, 8, rocke_f32());
}

/* _seq: decode_thd / decode_batched of batch `batch` from the kernargs. */
static rocke_attn_bwd_seq_info_t seq_decode(rocke_ir_builder_t* b,
                                            int* seq_guards,
                                            const rocke_attn_bwd_aux_spec_t* spec,
                                            rocke_value_t* batch,
                                            rocke_value_t* s_max,
                                            rocke_value_t* seq_ptr,
                                            rocke_value_t* off_ptr,
                                            const char* mult,
                                            const char* div)
{
    rocke_attn_bwd_decode_args_t d = rocke_attn_bwd_decode_args_default();
    d.s_max = V(s_max);
    d.seq_ptr = seq_ptr;
    d.has_len = V(A(b, "has_len"));
    d.len_stride = V(A(b, "len_stride"));
    if(streq(spec->seq_mode, "thd"))
    {
        d.off_ptr = off_ptr;
        d.off64 = V(A(b, "off64"));
        d.mult = V(A(b, mult));
        d.div = V(A(b, div));
        d.ws_seg = V(A(b, "ws_seg"));
        return rocke_attn_bwd_decode_thd(b, seq_guards, batch, &d);
    }
    return rocke_attn_bwd_decode_batched(b, seq_guards, batch, &d);
}

static rocke_attn_bwd_ws_layout_t ws_layout(const rocke_attn_bwd_aux_spec_t* spec)
{
    return streq(spec->ws_layout, "token_major") ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR
                                                 : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
}

static rocke_kernel_def_t*
    finish(rocke_ir_builder_t* b, const rocke_attn_bwd_aux_spec_t* spec, const char* arch)
{
    rocke_b_ret(b);
    char reason[256];
    const int threads = rocke_attn_bwd_aux_block_threads(spec, arch, reason, sizeof(reason));
    if(threads < 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
        return NULL;
    }
    rocke_attr_set_int(b,
                       &b->kernel->attrs,
                       "max_workgroup_size",
                       threads > AUX_MIN_WG_ATTR ? threads : AUX_MIN_WG_ATTR);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

static bool
    check_vec8(rocke_ir_builder_t* b, const rocke_attn_bwd_aux_spec_t* spec, const char* stage)
{
    if(spec->convert_vec != 8)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "convert_vec=%d applies to the convert only, not %s",
                        spec->convert_vec,
                        stage);
        return false;
    }
    return true;
}

static bool check_arch(rocke_ir_builder_t* b, const char* arch)
{
    if(rocke_arch_target_from_gfx(arch) == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown arch '%s'", arch ? arch : "(null)");
        return false;
    }
    return true;
}

static int
    rows_or_fail(rocke_ir_builder_t* b, const rocke_attn_bwd_aux_spec_t* spec, const char* arch)
{
    char reason[256];
    const int rows = rocke_attn_bwd_aux_rows_per_cta(spec, arch, reason, sizeof(reason));
    if(rows < 0)
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
    return rows;
}

/* ====================================================================== *
 * prep
 * ====================================================================== */

typedef struct prep_q_ctx
{
    rocke_ir_builder_t* b;
    const rocke_attn_bwd_aux_spec_t* spec;
    const lanes_t* ln;
    rocke_value_t* h;
    const rocke_type_t* io;
    rocke_attn_bwd_seq_info_t seq;
    rocke_attn_bwd_ws_slab_t lse2_ws;
    rocke_attn_bwd_ws_slab_t dsum_ws;
    rocke_value_t* rr;
    rocke_value_t* store_ok;
    rocke_value_t* inf;
    rocke_value_t* zero;
} prep_q_ctx_t;

/* q-side workspace slab: WorkspaceAddr(...).slab(h, ln.tile_row0). */
static rocke_attn_bwd_ws_slab_t
    prep_ws(const prep_q_ctx_t* c, rocke_value_t* ptr, int width, const rocke_attn_bwd_iv_t* nh)
{
    rocke_attn_bwd_workspace_addr_t w;
    rocke_attn_bwd_workspace_addr_init(
        &w, c->b, ptr, V(A(c->b, "ws_rows_q")), width, c->seq.ws_row0, ws_layout(c->spec), nh);
    return rocke_attn_bwd_workspace_addr_slab(&w, V(c->h), V(c->ln->tile_row0));
}

/* live_body.tensor(ptr, prefix, dtype): the user-tensor slab of this tile. */
static rocke_attn_bwd_row_slab_t
    prep_tensor(const prep_q_ctx_t* c, const char* ptr, const char* prefix, const rocke_type_t* dt)
{
    rocke_ir_builder_t* b = c->b;
    char nb[16];
    char nh[16];
    char nt[16];
    snprintf(nb, sizeof(nb), "%s_b", prefix);
    snprintf(nh, sizeof(nh), "%s_h", prefix);
    snprintf(nt, sizeof(nt), "%s_t", prefix);
    rocke_attn_bwd_iv_t strides[3] = {V(A(b, nb)), V(A(b, nh)), V(A(b, nt))};
    rocke_attn_bwd_strided_tensor_addr_t t;
    rocke_attn_bwd_strided_tensor_addr_init(
        &t, b, A(b, ptr), dt, strides, &c->seq, c->spec->stage_vec, c->spec->narrow_vec);
    return rocke_attn_bwd_strided_tensor_addr_slab(&t, V(c->h), V(c->ln->tile_row0), true);
}

static void prep_live_body(const prep_q_ctx_t* c)
{
    rocke_ir_builder_t* b = c->b;
    const lanes_t* ln = c->ln;
    rocke_attn_bwd_row_slab_t o_slab = prep_tensor(c, "O", "o", c->io);
    rocke_value_t* o = rocke_attn_bwd_row_slab_load_vec(&o_slab, ln->r, V(ln->col0), 8);
    rocke_attn_bwd_row_slab_t do_slab = prep_tensor(c, "dO", "do", c->io);
    rocke_value_t* dov = rocke_attn_bwd_row_slab_load_vec(&do_slab, ln->r, V(ln->col0), 8);
    rocke_value_t* acc = c->zero;
    for(int i = 0; i < 8; ++i)
    {
        rocke_value_t* oe = rocke_b_vec_extract(b, o, i);
        rocke_value_t* of = rocke_b_cast_to_f32(b, oe);
        rocke_value_t* de = rocke_b_vec_extract(b, dov, i);
        rocke_value_t* df = rocke_b_cast_to_f32(b, de);
        acc = rocke_b_fma(b, of, df, acc);
    }
    for(int m = ln->lanes / 2; m >= 1; m /= 2)
    {
        rocke_value_t* sh = rocke_b_warp_shuffle_xor(b, acc, m);
        acc = rocke_b_fadd(b, acc, sh);
    }
    rocke_attn_bwd_row_slab_t l_slab = prep_tensor(c, "LSE", "l", rocke_f32());
    rocke_value_t* lse = rocke_attn_bwd_row_slab_load(&l_slab, ln->r, I(0));
    /* py: row_live = land(cmp_lt(r, length), land(fcmp(ogt, lse, -inf), fcmp(olt, lse, inf))) */
    rocke_value_t* in_len = rocke_b_cmp_lt(b, ln->r, c->seq.length);
    rocke_value_t* ninf = rocke_b_const_f32(b, -INFINITY);
    rocke_value_t* gt = rocke_b_fcmp(b, "ogt", lse, ninf);
    rocke_value_t* lt = rocke_b_fcmp(b, "olt", lse, c->inf);
    rocke_value_t* finite = rocke_b_land(b, gt, lt);
    rocke_value_t* row_live = rocke_b_land(b, in_len, finite);
    rocke_value_t* log2e = rocke_b_const_f32(b, ROCKE_ATTN_BWD_LOG2E);
    rocke_value_t* scaled = rocke_b_fmul(b, lse, log2e);
    rocke_value_t* lse2 = rocke_b_select(b, row_live, scaled, c->inf);
    rocke_value_t* dsum = rocke_b_select(b, row_live, acc, c->zero);
    rocke_if_t st = rocke_b_scf_if(b, c->store_ok);
    rocke_b_region_enter(b, st.then_region);
    rocke_attn_bwd_ws_slab_store(&c->lse2_ws, c->rr, I(0), lse2);
    rocke_attn_bwd_ws_slab_store(&c->dsum_ws, c->rr, I(0), dsum);
    rocke_b_region_leave(b);
}

static void prep_q_role(rocke_ir_builder_t* b,
                        int* seq_guards,
                        const rocke_attn_bwd_aux_spec_t* spec,
                        const lanes_t* ln,
                        rocke_value_t* h,
                        rocke_value_t* z,
                        const rocke_type_t* io)
{
    const int D = spec->head_size;
    const bool batched = streq(spec->seq_mode, "batched");
    prep_q_ctx_t c;
    memset(&c, 0, sizeof(c));
    c.b = b;
    c.spec = spec;
    c.ln = ln;
    c.h = h;
    c.io = io;
    c.seq = seq_decode(
        b, seq_guards, spec, z, A(b, "S_q_max"), A(b, "SEQ_Q"), A(b, "OFF_Q"), "q_mult", "q_div");
    rocke_attn_bwd_iv_t nh_v = V(A(b, "h_q"));
    const rocke_attn_bwd_iv_t* nh = ws_layout(spec) == ROCKE_ATTN_BWD_WS_TOKEN_MAJOR ? &nh_v : NULL;
    c.lse2_ws = prep_ws(&c, A(b, "WS_LSE2"), 1, nh);
    c.dsum_ws = prep_ws(&c, A(b, "WS_DSUM"), 1, nh);
    c.rr = ln->row;
    rocke_value_t* in_role = rocke_b_cmp_lt(b, ln->r, batched ? c.seq.s_max : c.seq.length);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* leader = rocke_b_cmp_eq(b, ln->seg, c0);
    rocke_value_t* lead_in = rocke_b_land(b, leader, in_role);
    rocke_value_t* lse2_ok = rocke_attn_bwd_ws_slab_row_ok(&c.lse2_ws, c.rr);
    c.store_ok = rocke_b_land(b, lead_in, lse2_ok);
    rocke_value_t* live_cta = rocke_b_cmp_lt(b, ln->tile_row0, c.seq.length);
    c.inf = rocke_b_const_f32(b, INFINITY);
    c.zero = rocke_b_const_f32(b, 0.0);

    if(batched)
    {
        rocke_if_else_t ie = rocke_b_scf_if_else(b, live_cta);
        rocke_b_region_enter(b, ie.then_region);
        prep_live_body(&c);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        rocke_if_t st = rocke_b_scf_if(b, c.store_ok);
        rocke_b_region_enter(b, st.then_region);
        rocke_attn_bwd_ws_slab_store(&c.lse2_ws, c.rr, I(0), c.inf);
        rocke_attn_bwd_ws_slab_store(&c.dsum_ws, c.rr, I(0), c.zero);
        rocke_b_region_leave(b);
        rocke_b_region_leave(b);
    }
    else
    {
        rocke_if_t live = rocke_b_scf_if(b, live_cta);
        rocke_b_region_enter(b, live.then_region);
        prep_live_body(&c);
        rocke_b_region_leave(b);
    }

    rocke_attn_bwd_ws_slab_t dq_ws = prep_ws(&c, A(b, "WS_DQ"), D, nh);
    rocke_value_t* c0b = rocke_b_const_i32(b, 0);
    rocke_value_t* zero_dq = rocke_b_cmp_ne(b, A(b, "zero_dq"), c0b);
    rocke_value_t* dq_ok = rocke_attn_bwd_ws_slab_row_ok(&dq_ws, c.rr);
    rocke_value_t* role_ok = rocke_b_land(b, in_role, dq_ok);
    rocke_value_t* cond = rocke_b_land(b, zero_dq, role_ok);
    rocke_if_t zq = rocke_b_scf_if(b, cond);
    rocke_b_region_enter(b, zq.then_region);
    rocke_value_t* zv = zero_vec8(b);
    rocke_attn_bwd_ws_slab_store_vec(&dq_ws, c.rr, V(ln->col0), zv);
    rocke_b_region_leave(b);
}

static void prep_kv_role(rocke_ir_builder_t* b,
                         int* seq_guards,
                         const rocke_attn_bwd_aux_spec_t* spec,
                         const lanes_t* ln,
                         rocke_value_t* y,
                         rocke_value_t* z)
{
    const int D = spec->head_size;
    rocke_attn_bwd_seq_info_t seq = seq_decode(b,
                                               seq_guards,
                                               spec,
                                               z,
                                               A(b, "S_kv_max"),
                                               A(b, "SEQ_KV"),
                                               A(b, "OFF_KV"),
                                               "kv_mult",
                                               "kv_div");
    rocke_value_t* h_k = A(b, "h_k");
    rocke_value_t* h_v = A(b, "h_v");
    rocke_value_t* y2 = rocke_b_sub(b, y, A(b, "h_q"));
    rocke_value_t* is_dk = rocke_b_cmp_lt(b, y2, h_k);
    rocke_value_t* y2k = rocke_b_sub(b, y2, h_k);
    rocke_value_t* head = rocke_b_select(b, is_dk, y2, y2k);
    rocke_value_t* ptr = rocke_b_select(b, is_dk, A(b, "WS_DK"), A(b, "WS_DV"));
    rocke_attn_bwd_iv_t nh_v = I(0);
    const rocke_attn_bwd_iv_t* nh = NULL;
    if(ws_layout(spec) == ROCKE_ATTN_BWD_WS_TOKEN_MAJOR)
    {
        nh_v = V(rocke_b_select(b, is_dk, h_k, h_v));
        nh = &nh_v;
    }
    rocke_attn_bwd_workspace_addr_t w;
    rocke_attn_bwd_workspace_addr_init(
        &w, b, ptr, V(A(b, "ws_rows_kv")), D, seq.ws_row0, ws_layout(spec), nh);
    rocke_attn_bwd_ws_slab_t slab
        = rocke_attn_bwd_workspace_addr_slab(&w, V(head), V(ln->tile_row0));
    rocke_value_t* bound = streq(spec->seq_mode, "batched") ? seq.s_max : seq.length;
    rocke_value_t* in_b = rocke_b_cmp_lt(b, ln->r, bound);
    rocke_value_t* row_ok = rocke_attn_bwd_ws_slab_row_ok(&slab, ln->row);
    rocke_value_t* ok = rocke_b_land(b, in_b, row_ok);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* zkv = rocke_b_cmp_ne(b, A(b, "zero_kv"), c0);
    ok = rocke_b_land(b, ok, zkv);
    rocke_value_t* hkv = rocke_b_add(b, h_k, h_v);
    rocke_value_t* in_heads = rocke_b_cmp_lt(b, y2, hkv);
    ok = rocke_b_land(b, ok, in_heads);
    rocke_if_t st = rocke_b_scf_if(b, ok);
    rocke_b_region_enter(b, st.then_region);
    rocke_value_t* zv = zero_vec8(b);
    rocke_attn_bwd_ws_slab_store_vec(&slab, ln->row, V(ln->col0), zv);
    rocke_b_region_leave(b);
}

/* _wl_weight: (query length, live kv tiles) of THD sequence s. */
static void wl_weight(rocke_ir_builder_t* b,
                      int* seq_guards,
                      rocke_value_t* s,
                      rocke_value_t** key,
                      rocke_value_t** tiles)
{
    static const char* const SIDES[2][5] = {
        {"S_q_max", "OFF_Q", "SEQ_Q", "q_mult", "q_div"},
        {"S_kv_max", "OFF_KV", "SEQ_KV", "kv_mult", "kv_div"},
    };
    rocke_value_t* lens[2];
    for(int side = 0; side < 2; ++side)
    {
        rocke_attn_bwd_decode_args_t d = rocke_attn_bwd_decode_args_default();
        d.s_max = V(A(b, SIDES[side][0]));
        d.off_ptr = A(b, SIDES[side][1]);
        d.off64 = V(A(b, "off64"));
        d.mult = V(A(b, SIDES[side][3]));
        d.div = V(A(b, SIDES[side][4]));
        d.seq_ptr = A(b, SIDES[side][2]);
        d.has_len = V(A(b, "has_len"));
        d.len_stride = V(A(b, "len_stride"));
        lens[side] = rocke_attn_bwd_decode_thd(b, seq_guards, s, &d).length;
    }
    *key = lens[0];
    *tiles = rocke_attn_bwd_worklist_tiles(b, lens[1], A(b, "wl_kn0"));
}

static rocke_value_t* lds_at(rocke_ir_builder_t* b, rocke_value_t* lds, rocke_value_t* i)
{
    rocke_value_t* idx[1] = {i};
    rocke_value_t* v = rocke_b_smem_load_vN(b, lds, idx, 1, rocke_i32(), 1);
    return rocke_b_vec_extract(b, v, 0);
}

/* _worklist_role: write the work list; every thread of the CTA runs it. */
static void worklist_role(rocke_ir_builder_t* b, int* seq_guards, int threads)
{
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* tc = rocke_b_const_i32(b, threads);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* nb = A(b, "n_batch");
    rocke_value_t* wl = A(b, "WORKLIST");
    const int shape[1] = {2 * threads};
    rocke_value_t* lds = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "wl_lds");
    rocke_value_t* tm1 = rocke_b_const_i32(b, threads - 1);
    rocke_value_t* nb_up = rocke_b_add(b, nb, tm1);
    rocke_value_t* n_pass = rocke_b_div(b, nb_up, tc);

    rocke_iter_arg_t outer_args[1] = {{"wl_count", zero}};
    rocke_for_t outer
        = rocke_b_scf_for_iter(b, zero, n_pass, one, outer_args, 1, "wl_sb", false, false);
    if(outer.op == NULL)
        return;
    rocke_b_region_enter(b, outer.body);
    {
        rocke_value_t* sb_tc = rocke_b_mul(b, outer.iv, tc);
        rocke_value_t* s = rocke_b_add(b, sb_tc, tid);
        rocke_value_t* s_ok = rocke_b_cmp_lt(b, s, nb);
        rocke_value_t* s_sel = rocke_b_select(b, s_ok, s, zero);
        rocke_value_t* key_s = NULL;
        rocke_value_t* nt_s = NULL;
        wl_weight(b, seq_guards, s_sel, &key_s, &nt_s);
        rocke_iter_arg_t chunk_args[2] = {{"wl_start", zero}, {"wl_total", zero}};
        rocke_for_t chunks
            = rocke_b_scf_for_iter(b, zero, n_pass, one, chunk_args, 2, "wl_c", false, false);
        if(chunks.op == NULL)
            return;
        rocke_b_region_enter(b, chunks.body);
        {
            rocke_value_t* start = chunks.iter_vars[0];
            rocke_value_t* total = chunks.iter_vars[1];
            rocke_value_t* c0 = rocke_b_mul(b, chunks.iv, tc);
            rocke_value_t* o = rocke_b_add(b, c0, tid);
            rocke_value_t* o_ok = rocke_b_cmp_lt(b, o, nb);
            rocke_value_t* o_sel = rocke_b_select(b, o_ok, o, zero);
            rocke_value_t* key_o = NULL;
            rocke_value_t* nt_o = NULL;
            wl_weight(b, seq_guards, o_sel, &key_o, &nt_o);
            rocke_b_sync_fence(b, ROCKE_ATTN_BWD_BARRIER_FENCE);
            rocke_value_t* idx0[1] = {tid};
            rocke_value_t* m1 = rocke_b_const_i32(b, -1);
            rocke_value_t* kv = rocke_b_select(b, o_ok, key_o, m1);
            rocke_b_smem_store_vN(b, lds, idx0, 1, kv, 1);
            rocke_value_t* t_tc = rocke_b_add(b, tid, tc);
            rocke_value_t* idx1[1] = {t_tc};
            rocke_value_t* ntv = rocke_b_select(b, o_ok, nt_o, zero);
            rocke_b_smem_store_vN(b, lds, idx1, 1, ntv, 1);
            rocke_b_sync_fence(b, ROCKE_ATTN_BWD_BARRIER_FENCE);
            rocke_iter_arg_t scan_args[2] = {{"wl_st", start}, {"wl_tt", total}};
            rocke_for_t scan
                = rocke_b_scf_for_iter(b, zero, tc, one, scan_args, 2, "wl_j", false, false);
            if(scan.op == NULL)
                return;
            rocke_b_region_enter(b, scan.body);
            {
                rocke_value_t* j = scan.iv;
                rocke_value_t* st = scan.iter_vars[0];
                rocke_value_t* tt = scan.iter_vars[1];
                rocke_value_t* j_tc = rocke_b_add(b, j, tc);
                rocke_value_t* n_j = lds_at(b, lds, j_tc);
                rocke_value_t* key_j = lds_at(b, lds, j);
                rocke_value_t* o_j = rocke_b_add(b, c0, j);
                rocke_value_t* before
                    = rocke_attn_bwd_worklist_sorts_before(b, key_j, o_j, key_s, s);
                rocke_value_t* inc = rocke_b_select(b, before, n_j, zero);
                rocke_value_t* y[2];
                y[0] = rocke_b_add(b, st, inc);
                y[1] = rocke_b_add(b, tt, n_j);
                rocke_b_scf_yield(b, y, 2);
            }
            rocke_b_region_leave(b);
            rocke_b_scf_yield(b, scan.op->results, scan.op->num_results);
        }
        rocke_b_region_leave(b);
        rocke_value_t* start = chunks.op->results[0];
        rocke_value_t* total = chunks.op->results[1];
        rocke_value_t* n_items = rocke_b_select(b, s_ok, nt_s, zero);
        rocke_for_t writes = rocke_b_scf_for(b, zero, n_items, one, "wl_t");
        if(writes.op == NULL)
            return;
        rocke_b_region_enter(b, writes.body);
        rocke_value_t* pos = rocke_b_add(b, start, writes.iv);
        rocke_attn_bwd_worklist_store_item(b, wl, pos, s, writes.iv);
        rocke_b_region_leave(b);
        rocke_b_scf_yield(b, &total, 1);
    }
    rocke_b_region_leave(b);
    rocke_value_t* is0 = rocke_b_cmp_eq(b, tid, zero);
    rocke_if_t hdr = rocke_b_scf_if(b, is0);
    rocke_b_region_enter(b, hdr.then_region);
    rocke_attn_bwd_worklist_store_count(b, wl, outer.op->results[0]);
    rocke_b_region_leave(b);
}

static rocke_kernel_def_t*
    build_prep(rocke_ir_builder_t* b, const rocke_attn_bwd_aux_spec_t* spec, const char* arch)
{
    if(!check_arch(b, arch) || !check_vec8(b, spec, "prep"))
        return NULL;
    const int rows = rows_or_fail(b, spec, arch);
    if(rows < 0)
        return NULL;
    /* py: _SEQ_GUARDS is one counter per builder. Every decode_* of this
     * kernel (q role, kv role, worklist role) must share this one int, or
     * the seqlenN / seq_iN names diverge from Python. */
    int seq_guards = 0;
    declare(b, "prep", spec);
    const rocke_type_t* io = io_type(spec);
    lanes_t ln = make_lanes(b, spec, rows);
    rocke_value_t* y = rocke_b_block_id_y(b);
    rocke_value_t* z = rocke_b_block_id_z(b);
    rocke_value_t* is_q = rocke_b_cmp_lt(b, y, A(b, "h_q"));
    if(streq(spec->kv_target, "contiguous"))
    {
        rocke_if_else_t ie = rocke_b_scf_if_else(b, is_q);
        rocke_b_region_enter(b, ie.then_region);
        prep_q_role(b, &seq_guards, spec, &ln, y, z, io);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        prep_kv_role(b, &seq_guards, spec, &ln, y, z);
        rocke_b_region_leave(b);
    }
    else
    {
        rocke_if_t q = rocke_b_scf_if(b, is_q);
        rocke_b_region_enter(b, q.then_region);
        prep_q_role(b, &seq_guards, spec, &ln, y, z, io);
        rocke_b_region_leave(b);
    }
    if(spec->worklist)
    {
        /* one CTA: (0, 0, 0), a CTA-uniform branch around the role's barriers */
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* x0 = rocke_b_cmp_eq(b, bx, zero);
        rocke_value_t* y0 = rocke_b_cmp_eq(b, y, zero);
        rocke_value_t* xy0 = rocke_b_land(b, x0, y0);
        rocke_value_t* z0 = rocke_b_cmp_eq(b, z, zero);
        rocke_value_t* first = rocke_b_land(b, xy0, z0);
        rocke_value_t* use = rocke_b_cmp_ne(b, A(b, "use_worklist"), zero);
        rocke_value_t* cond = rocke_b_land(b, use, first);
        rocke_if_t role = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, role.then_region);
        char reason[256];
        const int threads = rocke_attn_bwd_aux_block_threads(spec, arch, reason, sizeof(reason));
        worklist_role(b, &seq_guards, threads);
        rocke_b_region_leave(b);
    }
    return finish(b, spec, arch);
}

/* ====================================================================== *
 * convert / reduce
 * ====================================================================== */

/* _ws_vec8: n (a multiple of 4) fp32 workspace values, 0 outside it. */
static int ws_vec8(rocke_ir_builder_t* b,
                   const rocke_attn_bwd_ws_slab_t* slab,
                   rocke_value_t* rr,
                   rocke_value_t* col0,
                   int n,
                   rocke_value_t** out)
{
    rocke_value_t* ok = rocke_attn_bwd_ws_slab_row_ok(slab, rr);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    int k = 0;
    for(int half = 0; half < n / 4; ++half)
    {
        rocke_value_t* col = col0;
        if(!(half == 0 && n == 4))
        {
            rocke_value_t* c = rocke_b_const_i32(b, 4 * half);
            col = rocke_b_add(b, col0, c);
        }
        rocke_value_t* v4 = rocke_attn_bwd_ws_slab_load_vec4(slab, rr, V(col));
        for(int i = 0; i < 4; ++i)
        {
            rocke_value_t* e = rocke_b_vec_extract(b, v4, i);
            out[k++] = rocke_b_select(b, ok, e, zero);
        }
    }
    return k;
}

typedef struct convert_ctx
{
    rocke_ir_builder_t* b;
    const rocke_attn_bwd_aux_spec_t* spec;
    bool reduce;
    const lanes_t* ln;
    rocke_attn_bwd_ws_slab_t slab;
    rocke_attn_bwd_row_slab_t dst;
    rocke_value_t* mult;
    rocke_value_t* zero;
    const rocke_type_t* io;
} convert_ctx_t;

/* read_sum: the workspace values (convert), or the in-order sum of the
 * n_slices slices (reduce). */
static int read_sum(const convert_ctx_t* c, rocke_value_t** vals)
{
    rocke_ir_builder_t* b = c->b;
    const int vec = c->spec->convert_vec;
    if(vec < 1 || vec > AUX_MAX_CONVERT_VEC)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "convert_vec=%d exceeds the C buffers", vec);
        return 0;
    }
    const int n = ws_vec8(b, &c->slab, c->ln->row, c->ln->col0, vec, vals);
    if(!c->reduce)
        return n;
    rocke_value_t* c4 = rocke_b_const_i64(b, 4);
    rocke_value_t* stride = rocke_b_mul(b, A(b, "s_stride"), c4);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_iter_arg_t args[AUX_MAX_CONVERT_VEC];
    char names[AUX_MAX_CONVERT_VEC][16];
    for(int i = 0; i < n; ++i)
    {
        snprintf(names[i], sizeof(names[i]), "rsum%d", i);
        args[i].name = names[i];
        args[i].init = vals[i];
    }
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, one, A(b, "n_slices"), one, args, n, "rslice", false, true);
    if(loop.op == NULL)
        return 0;
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* it64 = rocke_b_zext(b, loop.iv, rocke_i64());
        rocke_value_t* off = rocke_b_mul(b, it64, stride);
        rocke_value_t* ptr = rocke_b_global_ptr_add(b, c->slab.ptr, off);
        rocke_attn_bwd_ws_slab_t moved = c->slab;
        moved.ptr = ptr;
        rocke_value_t* part[AUX_MAX_CONVERT_VEC];
        ws_vec8(b, &moved, c->ln->row, c->ln->col0, vec, part);
        rocke_value_t* sums[AUX_MAX_CONVERT_VEC];
        for(int i = 0; i < n; ++i)
            sums[i] = rocke_b_fadd(b, loop.iter_vars[i], part[i]);
        rocke_b_scf_yield(b, sums, n);
    }
    rocke_b_region_leave(b);
    for(int i = 0; i < n; ++i)
        vals[i] = loop.op->results[i];
    return n;
}

static void write_row(const convert_ctx_t* c, rocke_value_t* valid)
{
    rocke_ir_builder_t* b = c->b;
    const lanes_t* ln = c->ln;
    rocke_value_t* vals[AUX_MAX_CONVERT_VEC];
    const int n = read_sum(c, vals);
    if(n == 0)
        return; /* error already set */
    for(int i = 0; i < n; ++i)
        vals[i] = rocke_b_fmul(b, c->mult, vals[i]);
    if(valid != NULL)
        for(int i = 0; i < n; ++i)
            vals[i] = rocke_b_select(b, valid, vals[i], c->zero);
    if(c->spec->convert_vec == 16)
    {
        /* two 8-element output pieces (16-byte stores) */
        for(int part = 0; part < 2; ++part)
        {
            rocke_value_t* piece = rocke_b_vec_pack(b, vals + 8 * part, 8, rocke_f32());
            rocke_value_t* col = ln->col0;
            if(part != 0)
            {
                rocke_value_t* c8 = rocke_b_const_i32(b, 8);
                col = rocke_b_add(b, ln->col0, c8);
            }
            rocke_value_t* cast = rocke_b_vec_cast_f32_to(b, piece, c->io);
            rocke_attn_bwd_row_slab_store_vec(&c->dst, ln->r, V(col), cast);
        }
        return;
    }
    rocke_value_t* packed = rocke_b_vec_pack(b, vals, n, rocke_f32());
    rocke_value_t* out = rocke_b_vec_cast_f32_to(b, packed, c->io);
    rocke_attn_bwd_row_slab_store_vec(&c->dst, ln->r, V(ln->col0), out);
}

static rocke_kernel_def_t* build_convert(rocke_ir_builder_t* b,
                                         const rocke_attn_bwd_aux_spec_t* spec,
                                         const char* arch,
                                         const char* stage)
{
    if(!check_arch(b, arch))
        return NULL;
    if(spec->worklist)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "worklist=True applies to the prep kernel only");
        return NULL;
    }
    const int rows = rows_or_fail(b, spec, arch);
    if(rows < 0)
        return NULL;
    int seq_guards = 0;
    declare(b, stage, spec);
    const rocke_type_t* io = io_type(spec);
    lanes_t ln = make_lanes(b, spec, rows);
    rocke_value_t* h = rocke_b_block_id_y(b);
    rocke_value_t* z = rocke_b_block_id_z(b);
    rocke_attn_bwd_seq_info_t seq = seq_decode(
        b, &seq_guards, spec, z, A(b, "S_max"), A(b, "SEQ"), A(b, "OFF"), "tok_mult", "tok_div");

    convert_ctx_t c;
    memset(&c, 0, sizeof(c));
    c.b = b;
    c.spec = spec;
    c.reduce = streq(stage, "reduce");
    c.ln = &ln;
    c.io = io;
    {
        rocke_attn_bwd_iv_t nh_v = V(A(b, "H"));
        const rocke_attn_bwd_iv_t* nh
            = ws_layout(spec) == ROCKE_ATTN_BWD_WS_TOKEN_MAJOR ? &nh_v : NULL;
        rocke_attn_bwd_workspace_addr_t w;
        rocke_attn_bwd_workspace_addr_init(&w,
                                           b,
                                           A(b, "SRC"),
                                           V(A(b, "ws_rows")),
                                           spec->head_size,
                                           seq.ws_row0,
                                           ws_layout(spec),
                                           nh);
        c.slab = rocke_attn_bwd_workspace_addr_slab(&w, V(h), V(ln.tile_row0));
    }
    {
        rocke_attn_bwd_iv_t strides[3] = {V(A(b, "d_b")), V(A(b, "d_h")), V(A(b, "d_t"))};
        rocke_attn_bwd_strided_tensor_addr_t t;
        rocke_attn_bwd_strided_tensor_addr_init(
            &t, b, A(b, "DST"), io, strides, &seq, spec->stage_vec, spec->narrow_vec);
        c.dst = rocke_attn_bwd_strided_tensor_addr_slab(&t, V(h), V(ln.tile_row0), false);
    }
    c.mult = A(b, "mult");
    c.zero = rocke_b_const_f32(b, 0.0);

    if(streq(spec->seq_mode, "batched"))
    {
        rocke_value_t* in_rows = rocke_b_cmp_lt(b, ln.r, seq.s_max);
        rocke_if_t st = rocke_b_scf_if(b, in_rows);
        rocke_b_region_enter(b, st.then_region);
        rocke_value_t* valid = rocke_b_cmp_lt(b, ln.r, seq.length);
        write_row(&c, valid);
        rocke_b_region_leave(b);
    }
    else
    {
        rocke_value_t* cta_live = rocke_b_cmp_lt(b, ln.tile_row0, seq.length);
        rocke_if_t live = rocke_b_scf_if(b, cta_live);
        rocke_b_region_enter(b, live.then_region);
        rocke_value_t* in_len = rocke_b_cmp_lt(b, ln.r, seq.length);
        rocke_if_t row = rocke_b_scf_if(b, in_len);
        rocke_b_region_enter(b, row.then_region);
        write_row(&c, NULL);
        rocke_b_region_leave(b);
        rocke_b_region_leave(b);
    }
    return finish(b, spec, arch);
}

/* ====================================================================== *
 * paged workspace roles
 * ====================================================================== */

typedef struct page_walk
{
    rocke_value_t* s;
    rocke_value_t* j;
    rocke_value_t* in0;
    rocke_value_t* live;
} page_walk_t;

static page_walk_t page_walk(rocke_ir_builder_t* b, const lanes_t* ln)
{
    page_walk_t w;
    w.s = rocke_b_block_id_z(b);
    rocke_value_t* page = A(b, "page");
    rocke_value_t* slot0 = ln->tile_row0;
    w.j = rocke_b_div(b, slot0, page);
    w.in0 = rocke_b_mod(b, slot0, page);
    rocke_value_t* span = rocke_b_mul(b, A(b, "max_blocks_per_seq"), page);
    rocke_value_t* raw = rocke_b_global_load_i32(b, A(b, "SEQ_LENS_KV"), w.s, 4);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* lo = rocke_b_smax(b, raw, c0);
    rocke_value_t* len_kv = rocke_b_smin(b, lo, span);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* pm1 = rocke_b_sub(b, page, c1);
    rocke_value_t* up = rocke_b_add(b, len_kv, pm1);
    rocke_value_t* n_blocks = rocke_b_div(b, up, page);
    w.live = rocke_b_cmp_lt(b, w.j, n_blocks);
    return w;
}

typedef struct page_slab
{
    rocke_value_t* pid;
    rocke_attn_bwd_ws_slab_t ws;
    rocke_value_t* ok;
} page_slab_t;

static page_slab_t page_slab(rocke_ir_builder_t* b,
                             const rocke_attn_bwd_aux_spec_t* spec,
                             rocke_value_t* ptr,
                             const page_walk_t* w,
                             rocke_value_t* h)
{
    page_slab_t out;
    rocke_value_t* page = A(b, "page");
    rocke_value_t* page_rows = A(b, "page_rows");
    rocke_value_t* raw = rocke_attn_bwd_block_table_entry(
        b, A(b, "BLOCK_TABLE"), w->s, w->j, V(A(b, "bt_stride")));
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* num_blocks = rocke_b_div(b, page_rows, page);
    rocke_value_t* ge = rocke_b_cmp_ge(b, raw, zero);
    rocke_value_t* lt = rocke_b_cmp_lt(b, raw, num_blocks);
    out.ok = rocke_b_land(b, ge, lt);
    out.pid = rocke_b_select(b, out.ok, raw, zero);
    /* py: b.add(b.mul(b.zext(pid, I64), b.zext(page, I64)), b.zext(in0, I64)) */
    rocke_value_t* pid64 = rocke_b_zext(b, out.pid, rocke_i64());
    rocke_value_t* page64 = rocke_b_zext(b, page, rocke_i64());
    rocke_value_t* pbase = rocke_b_mul(b, pid64, page64);
    rocke_value_t* in064 = rocke_b_zext(b, w->in0, rocke_i64());
    rocke_value_t* phys0 = rocke_b_add(b, pbase, in064);
    rocke_value_t* rows64 = rocke_b_zext(b, page_rows, rocke_i64());
    phys0 = rocke_b_select(b, out.ok, phys0, rows64);
    rocke_attn_bwd_workspace_addr_t wa;
    rocke_attn_bwd_workspace_addr_init(
        &wa, b, ptr, V(page_rows), spec->head_size, phys0, ROCKE_ATTN_BWD_WS_HEAD_MAJOR, NULL);
    out.ws = rocke_attn_bwd_workspace_addr_slab(&wa, V(h), I(0));
    return out;
}

static bool check_paged(rocke_ir_builder_t* b, const rocke_attn_bwd_aux_spec_t* spec)
{
    if(!streq(spec->kv_target, "paged"))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "page roles need kv_target='paged'");
        return false;
    }
    if(!streq(spec->ws_layout, "head_major"))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "the page workspace is head-major");
        return false;
    }
    return true;
}

/* ok = land(land(cmp_lt(add(in0, row), page), ws.row_ok(row)), id_ok) */
static rocke_value_t* page_store_ok(rocke_ir_builder_t* b,
                                    const lanes_t* ln,
                                    const page_walk_t* w,
                                    const page_slab_t* ps)
{
    rocke_value_t* slot = rocke_b_add(b, w->in0, ln->row);
    rocke_value_t* in_page = rocke_b_cmp_lt(b, slot, A(b, "page"));
    rocke_value_t* row_ok = rocke_attn_bwd_ws_slab_row_ok(&ps->ws, ln->row);
    rocke_value_t* ok = rocke_b_land(b, in_page, row_ok);
    return rocke_b_land(b, ok, ps->ok);
}

static rocke_kernel_def_t*
    build_page_zero(rocke_ir_builder_t* b, const rocke_attn_bwd_aux_spec_t* spec, const char* arch)
{
    if(!check_arch(b, arch) || !check_paged(b, spec) || !check_vec8(b, spec, "page_zero"))
        return NULL;
    const int rows = rows_or_fail(b, spec, arch);
    if(rows < 0)
        return NULL;
    declare(b, "page_zero", spec);
    lanes_t ln = make_lanes(b, spec, rows);
    rocke_value_t* h = rocke_b_block_id_y(b);
    page_walk_t w = page_walk(b, &ln);
    rocke_if_t live = rocke_b_scf_if(b, w.live);
    rocke_b_region_enter(b, live.then_region);
    {
        page_slab_t k = page_slab(b, spec, A(b, "WS_PDK"), &w, h);
        page_slab_t v = page_slab(b, spec, A(b, "WS_PDV"), &w, h);
        rocke_value_t* ok = page_store_ok(b, &ln, &w, &k);
        rocke_if_t st = rocke_b_scf_if(b, ok);
        rocke_b_region_enter(b, st.then_region);
        rocke_value_t* zk = zero_vec8(b);
        rocke_attn_bwd_ws_slab_store_vec(&k.ws, ln.row, V(ln.col0), zk);
        rocke_value_t* zv = zero_vec8(b);
        rocke_attn_bwd_ws_slab_store_vec(&v.ws, ln.row, V(ln.col0), zv);
        rocke_b_region_leave(b);
    }
    rocke_b_region_leave(b);
    return finish(b, spec, arch);
}

static rocke_kernel_def_t* build_page_convert(rocke_ir_builder_t* b,
                                              const rocke_attn_bwd_aux_spec_t* spec,
                                              const char* arch)
{
    if(!check_arch(b, arch) || !check_paged(b, spec) || !check_vec8(b, spec, "page_convert"))
        return NULL;
    const int rows = rows_or_fail(b, spec, arch);
    if(rows < 0)
        return NULL;
    const rocke_type_t* io = io_type(spec);
    declare(b, "page_convert", spec);
    lanes_t ln = make_lanes(b, spec, rows);
    rocke_value_t* h = rocke_b_block_id_y(b);
    page_walk_t w = page_walk(b, &ln);
    rocke_if_t live = rocke_b_scf_if(b, w.live);
    rocke_b_region_enter(b, live.then_region);
    {
        page_slab_t src = page_slab(b, spec, A(b, "SRC"), &w, h);
        rocke_attn_bwd_iv_t strides[3] = {V(A(b, "d_blk")), V(A(b, "d_tok")), V(A(b, "d_h"))};
        rocke_attn_bwd_paged_cache_addr_t pc;
        rocke_attn_bwd_paged_cache_addr_init(&pc, b, A(b, "DST"), io, strides, spec->stage_vec);
        rocke_attn_bwd_row_slab_t dst
            = rocke_attn_bwd_paged_cache_addr_slab(&pc, src.pid, V(h), V(w.in0));
        /* a block id outside the cache is never written (dK / dV of a bad entry) */
        rocke_value_t* ok = page_store_ok(b, &ln, &w, &src);
        rocke_if_t st = rocke_b_scf_if(b, ok);
        rocke_b_region_enter(b, st.then_region);
        rocke_value_t* vals[8];
        const int n = ws_vec8(b, &src.ws, ln.row, ln.col0, 8, vals);
        for(int i = 0; i < n; ++i)
            vals[i] = rocke_b_fmul(b, A(b, "mult"), vals[i]);
        rocke_value_t* packed = rocke_b_vec_pack(b, vals, n, rocke_f32());
        rocke_value_t* out = rocke_b_vec_cast_f32_to(b, packed, io);
        rocke_value_t* slot = rocke_b_add(b, w.in0, ln.row);
        rocke_attn_bwd_row_slab_store_vec(&dst, slot, V(ln.col0), out);
        rocke_b_region_leave(b);
    }
    rocke_b_region_leave(b);
    return finish(b, spec, arch);
}

/* ====================================================================== *
 * Public entry points
 * ====================================================================== */

rocke_kernel_def_t* rocke_build_attn_bwd_aux_new(rocke_ir_builder_t* b,
                                                 const char* stage,
                                                 const rocke_attn_bwd_aux_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(b == NULL)
            return NULL;
        /* py validates in AttnBwdAuxSpec.__post_init__, before any builder
         * exists; here the builder must exist to carry the error, so it is
         * initialised with a placeholder name when the spec is bad. */
        const bool stage_ok = spec != NULL && is_stage(stage);
        char reason[256] = "";
        const bool valid
            = stage_ok && rocke_attn_bwd_aux_spec_validate(spec, reason, sizeof(reason));
        /* Sized from the spec: only `name` is unbounded once the spec is
         * valid (every other part is at most a few dozen chars). */
        std::vector<char> name;
        bool named = false;
        if(valid)
        {
            name.resize(strlen(spec->name) + 128);
            named
                = rocke_attn_bwd_aux_kernel_name(spec, stage, name.data(), name.size()) == ROCKE_OK;
        }
        if(rocke_ir_builder_init(b, named ? name.data() : "rocke_attn_bwd_invalid") != ROCKE_OK)
            return NULL;
        if(!stage_ok)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "stage must be one of ('prep', 'convert', 'page_zero', "
                            "'page_convert', 'reduce')");
            return NULL;
        }
        if(!valid)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
            return NULL;
        }
        if(!named)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "kernel name does not fit");
            return NULL;
        }
        if(streq(stage, "prep"))
            return build_prep(b, spec, arch);
        if(streq(stage, "convert") || streq(stage, "reduce"))
            return build_convert(b, spec, arch, stage);
        if(streq(stage, "page_zero"))
            return build_page_zero(b, spec, arch);
        return build_page_convert(b, spec, arch);
    });
}

rocke_kernel_def_t* rocke_build_attn_bwd_prep_new(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_aux_spec_t* spec,
                                                  const char* arch)
{
    return rocke_build_attn_bwd_aux_new(b, "prep", spec, arch);
}

rocke_kernel_def_t* rocke_build_attn_bwd_convert_new(rocke_ir_builder_t* b,
                                                     const rocke_attn_bwd_aux_spec_t* spec,
                                                     const char* arch)
{
    return rocke_build_attn_bwd_aux_new(b, "convert", spec, arch);
}

rocke_kernel_def_t* rocke_build_attn_bwd_reduce_new(rocke_ir_builder_t* b,
                                                    const rocke_attn_bwd_aux_spec_t* spec,
                                                    const char* arch)
{
    return rocke_build_attn_bwd_aux_new(b, "reduce", spec, arch);
}

rocke_kernel_def_t* rocke_build_attn_bwd_page_zero_new(rocke_ir_builder_t* b,
                                                       const rocke_attn_bwd_aux_spec_t* spec,
                                                       const char* arch)
{
    return rocke_build_attn_bwd_aux_new(b, "page_zero", spec, arch);
}

rocke_kernel_def_t* rocke_build_attn_bwd_page_convert_new(rocke_ir_builder_t* b,
                                                          const rocke_attn_bwd_aux_spec_t* spec,
                                                          const char* arch)
{
    return rocke_build_attn_bwd_aux_new(b, "page_convert", spec, arch);
}
