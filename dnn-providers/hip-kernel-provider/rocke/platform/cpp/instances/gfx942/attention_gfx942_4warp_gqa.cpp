// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of the gfx942 4-warp GQA natural-QK paged-attention builder
 * (library/kernels/gfx942/attention_tiled_2d.py: _online_softmax_rescale,
 * _build_gfx942_4warp_gqa_lean, build_gfx942_4warp_gqa, supports_tiled_2d_spec,
 * supports_gfx942_4warp_gqa_spec) and of the two host predicates in
 * library/kernels/common/attention_unified.py it shares with dispatch
 * (gfx942_gqa_fold_eligible, gfx942_4warp_launch_grid).
 *
 * See rocke/instance_gfx942_attention_4warp_gqa.h for the symbol map.
 *
 * C evaluates call arguments in an unspecified order and every builder call
 * (even a constant) emits an op, so each Python expression is spelled out as a
 * sequence of statements in Python's left-to-right order.
 */

#include "rocke/instance_gfx942_attention_4warp_gqa.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_helper_rocke.helpers.attention.h" /* rocke_binary_search_seq_idx */
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h" /* rocke_decode_mfma_lanes */
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/helper_rocke.instances.common.attention_arch.h" /* rocke_require_tiled_attention_arch */
#include "rocke/instance_gfx942_attention_tiled_2d.h"

#define FWG_LOG2E 1.4426950408889634
/* Head size the lean body is written for: its V-staging stride is hardwired to
 * the 256-wide row. */
#define FWG_LEAN_HEAD_SIZE 256
/* Query heads sharing the 128-row M-tile under the head-fold (tile geometry). */
#define FWG_FOLD_HEADS 4
#define FWG_MAX_NDDIM 8 /* HD / 32 at HD = 256 */
#define FWG_MAX_NKEYT 2 /* BN / 32 at BN = 64  */
#define FWG_MAX_NK 32 /* HD / K at HD = 256  */
#define FWG_MAX_NKPV 8 /* BN / K at BN = 64   */
#define FWG_MAX_CPL 16
#define FWG_MAX_APL 4
#define FWG_MAX_NLOADS 8

static bool fwg_streq(const char* a, const char* c)
{
    if(a == c)
        return true;
    if(a == NULL || c == NULL)
        return false;
    return strcmp(a, c) == 0;
}

static void fwg_set_msg(char* out, size_t cap, const char* m)
{
    if(out == NULL || cap == 0)
        return;
    snprintf(out, cap, "%s", m ? m : "");
}

static void fwg_fail(rocke_ir_builder_t* b, rocke_status_t st, const char* m)
{
    if(b->status != ROCKE_OK)
        return;
    b->status = st;
    snprintf(b->err, ROCKE_ERR_MSG_CAP, "%s", m);
}

/* ===================================================================== *
 *  Host predicates
 * ===================================================================== */

bool rocke_gfx942_gqa_fold_eligible(
    int head_size, int num_queries_per_kv, int sliding_window, const char* dtype, int block_size)
{
    return head_size == 128 && num_queries_per_kv == 4 && sliding_window > 0
           && fwg_streq(dtype, "bf16") && block_size <= 32;
}

bool rocke_gfx942_4warp_launch_grid(const rocke_unified_attention_problem_t* p, int grid[3])
{
    if(p == NULL || grid == NULL)
        return false;
    int nqpk = rocke_unified_attention_problem_num_queries_per_kv(p);
    if(nqpk < 0)
        return false;
    if(rocke_gfx942_gqa_fold_eligible(
           p->head_size, nqpk, p->sliding_window, p->dtype, p->block_size))
    {
        grid[0] = p->num_kv_heads;
        grid[1] = p->total_q / 32 + p->num_seqs;
    }
    else
    {
        grid[0] = p->num_query_heads;
        grid[1] = p->total_q / 128 + p->num_seqs;
    }
    grid[2] = 1;
    return true;
}

static bool fwg_supports_spec(const rocke_attention_tiled_2d_spec_t* spec,
                              const char* arch,
                              bool use_d256_fast,
                              char* reason,
                              size_t reason_cap)
{
    if(spec == NULL)
    {
        fwg_set_msg(reason, reason_cap, "null spec");
        return false;
    }
    rocke_gfx942_attention_tiled_2d_supports_args_t a
        = rocke_gfx942_attention_tiled_2d_supports_args_default();
    a.head_size = spec->head_size;
    a.block_size = spec->block_size;
    a.dtype = spec->dtype;
    a.num_queries_per_kv = rocke_attention_tiled_2d_spec_num_queries_per_kv(spec);
    a.use_alibi = spec->use_alibi;
    a.use_qq_bias = spec->use_qq_bias;
    a.use_fp8 = spec->kv_storage_dtype != NULL;
    a.q_dtype = spec->dtype;
    a.num_warps = spec->num_warps;
    a.block_m_per_warp = spec->block_m_per_warp;
    a.kv_storage_dtype = spec->kv_storage_dtype;
    a.has_tile_size = spec->has_tile_size;
    a.tile_size = spec->tile_size;
    a.arch = arch ? arch : "gfx942";
    a.use_mfma_32x32x8 = spec->use_mfma_32x32x8;
    a.use_transposed_qk_32x32 = spec->use_transposed_qk_32x32;
    a.use_k_single_buffer = spec->use_k_single_buffer;
    a.use_conflict_free_v_store = spec->use_conflict_free_v_store;
    a.use_k_sliced_ring = spec->use_k_sliced_ring;
    a.ring_depth = spec->ring_depth;
    a.k_slice_hd = spec->k_slice_hd;
    a.use_d256_fast = use_d256_fast;
    return rocke_gfx942_attention_tiled_2d_supports(&a, reason, reason_cap);
}

bool rocke_gfx942_attention_tiled_2d_supports_spec(const rocke_attention_tiled_2d_spec_t* spec,
                                                   const char* arch,
                                                   char* reason,
                                                   size_t reason_cap)
{
    return fwg_supports_spec(spec, arch, false, reason, reason_cap);
}

bool rocke_gfx942_attention_4warp_gqa_supports_spec(const rocke_attention_tiled_2d_spec_t* spec,
                                                    const char* arch,
                                                    char* reason,
                                                    size_t reason_cap)
{
    return fwg_supports_spec(spec, arch, true, reason, reason_cap);
}

/* ===================================================================== *
 *  Build context: the values the Python closures capture.
 * ===================================================================== */
typedef struct fwg_ctx
{
    rocke_ir_builder_t* b;
    const rocke_type_t* dtype;
    const rocke_mfma_atom_t* at;
    rocke_lane_decode_t ld;
    rocke_value_t* tid;
    rocke_value_t* lane;
    rocke_value_t* wq;
    rocke_value_t* qhead; /* NULL under the fold */
    rocke_value_t* kvh;
    rocke_value_t* sid;
    rocke_value_t* bt_stride;
    rocke_value_t* BT;
    rocke_value_t* Q;
    rocke_value_t* Kp;
    rocke_value_t* Vp;
    rocke_value_t* sc;
    rocke_value_t* ninf;
    rocke_value_t* zf;
    rocke_tensor_descriptor_t* q_desc;
    int HD;
    int HKV;
    int BS;
    int BN;
    int BPT;
    int APL;
    int BPL;
    int CPL;
    int K;
    int NKEYT;
    int NK;
    int NDdim;
    int NKpv;
} fwg_ctx_t;

static rocke_value_t* ci(fwg_ctx_t* c, int v)
{
    return rocke_b_const_i32(c->b, v);
}

/* const(x) * const(y), constants emitted in that order. */
static rocke_value_t* fwg_mul_consts(fwg_ctx_t* c, int x, int y)
{
    rocke_value_t* cx = ci(c, x);
    rocke_value_t* cy = ci(c, y);
    return rocke_b_mul(c->b, cx, cy);
}

/* phys_key(kv, keytile): paged key row of logical key `keytile` in tile `kv`. */
static rocke_value_t* fwg_phys_key(fwg_ctx_t* c, rocke_value_t* kv, rocke_value_t* keytile)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* sb = rocke_b_mul(b, c->sid, c->bt_stride);
    rocke_value_t* kb = rocke_b_mul(b, kv, ci(c, c->BPT));
    rocke_value_t* kd = rocke_b_div(b, keytile, ci(c, c->BS));
    rocke_value_t* lblk = rocke_b_add(b, sb, rocke_b_add(b, kb, kd));
    rocke_value_t* pb = rocke_b_global_load_i32(b, c->BT, lblk, 4);
    rocke_value_t* base = rocke_b_mul(b, pb, ci(c, c->BS));
    rocke_value_t* off = rocke_b_mod(b, keytile, ci(c, c->BS));
    return rocke_b_add(b, base, off);
}

/* (row * HKV + kvh) * HD + col: element offset of a K/V cache row. */
static rocke_value_t* fwg_kv_elem(fwg_ctx_t* c, rocke_value_t* row, rocke_value_t* col)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* r = rocke_b_mul(b, row, ci(c, c->HKV));
    r = rocke_b_add(b, r, c->kvh);
    r = rocke_b_mul(b, r, ci(c, c->HD));
    return rocke_b_add(b, r, col);
}

static rocke_value_t* fwg_bperm(fwg_ctx_t* c, rocke_value_t* v)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* x = rocke_b_xor(b, c->lane, ci(c, 32));
    rocke_value_t* partner = rocke_b_mul(b, x, ci(c, 4));
    rocke_value_t* vi = rocke_b_bitcast(b, v, rocke_i32());
    rocke_value_t* p = rocke_b_ds_bpermute(b, partner, vi);
    return rocke_b_bitcast(b, p, rocke_f32());
}

static rocke_value_t* fwg_mma(fwg_ctx_t* c, rocke_value_t* a, rocke_value_t* bb, rocke_value_t* acc)
{
    return rocke_b_mma(c->b, c->at->name, a, bb, acc, NULL, 0);
}

/* MfmaAtom.lane_to_output for the 32x32 atoms. */
static void fwg_lane_to_output(fwg_ctx_t* c, int i, rocke_value_t** row, rocke_value_t** col)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* c_atom_n = ci(c, 32);
    rocke_value_t* n_in_atom = rocke_b_mod(b, c->lane, c_atom_n);
    rocke_value_t* m_blk = rocke_b_div(b, c->lane, c_atom_n);
    int rb = i / 4;
    int ri = i % 4;
    rocke_value_t* r0 = ci(c, rb * 8);
    rocke_value_t* r1 = rocke_b_mul(b, m_blk, ci(c, 4));
    rocke_value_t* r = rocke_b_add(b, r0, r1);
    *row = rocke_b_add(b, r, ci(c, ri));
    *col = n_in_atom;
}

static rocke_value_t*
    fwg_q_offset(fwg_ctx_t* c, rocke_value_t* token, rocke_value_t* head, rocke_value_t* dim)
{
    static const char* const names[3] = {"token", "head", "dim"};
    rocke_value_t* vals[3] = {token, head, dim};
    rocke_value_t* off = NULL;
    rocke_transforms_descriptor_offset(c->b, c->q_desc, names, vals, 3, &off, NULL);
    return off;
}

/* koff = h * K + k_blk * APL */
static rocke_value_t* fwg_koff(fwg_ctx_t* c, int h)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* hk = fwg_mul_consts(c, h, c->K);
    rocke_value_t* kb = rocke_b_mul(b, c->ld.k_blk, ci(c, c->APL));
    return rocke_b_add(b, hk, kb);
}

/* _online_softmax_rescale(..., exp=b.exp2_fast). Sm is [NKEYT][CPL]. */
static void fwg_online_softmax_rescale(fwg_ctx_t* c,
                                       rocke_value_t* Sm[FWG_MAX_NKEYT][FWG_MAX_CPL],
                                       rocke_value_t* m_old,
                                       rocke_value_t* l_old,
                                       rocke_value_t** m_new_out,
                                       rocke_value_t** alpha_out,
                                       rocke_value_t** l_new_out,
                                       rocke_value_t** Bp)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* local = c->ninf;
    for(int kt = 0; kt < c->NKEYT; kt++)
        for(int i = 0; i < c->CPL; i++)
            local = rocke_b_fmax(b, local, rocke_b_fmul(b, Sm[kt][i], c->sc));
    rocke_value_t* pl = fwg_bperm(c, local);
    rocke_value_t* m_new = rocke_b_fmax(b, m_old, rocke_b_fmax(b, local, pl));
    rocke_value_t* alpha = rocke_b_exp2_fast(b, rocke_b_fsub(b, m_old, m_new));
    rocke_value_t* P[FWG_MAX_NKEYT][FWG_MAX_CPL];
    rocke_value_t* lsum = c->zf;
    for(int kt = 0; kt < c->NKEYT; kt++)
    {
        for(int i = 0; i < c->CPL; i++)
        {
            rocke_value_t* s = rocke_b_fmul(b, Sm[kt][i], c->sc);
            rocke_value_t* p = rocke_b_exp2_fast(b, rocke_b_fsub(b, s, m_new));
            lsum = rocke_b_fadd(b, lsum, p);
            P[kt][i] = rocke_b_cast_f32_to(b, p, c->dtype);
        }
    }
    rocke_value_t* la = rocke_b_fmul(b, l_old, alpha);
    rocke_value_t* ps = fwg_bperm(c, lsum);
    rocke_value_t* l_new = rocke_b_fadd(b, la, rocke_b_fadd(b, lsum, ps));
    for(int kk = 0; kk < c->NKpv; kk++)
    {
        rocke_value_t* parts[FWG_MAX_APL];
        for(int j = 0; j < c->BPL; j++)
            parts[j] = P[kk / 4][(kk % 4) * 4 + j];
        Bp[kk] = rocke_b_vec_pack(b, parts, c->BPL, c->dtype);
    }
    *m_new_out = m_new;
    *alpha_out = alpha;
    *l_new_out = l_new;
}

static const char* fwg_name(rocke_ir_builder_t* b, const char* prefix, int idx)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%s%d", prefix, idx);
    return rocke_arena_strdup(&b->arena, buf);
}

/* The 18-argument production paged ABI (_attn_signature). Writes C, Q, K, V,
 * BT, KL, CUQ, scale, num_seqs, block_table_stride into out[0..9]. */
static void fwg_params(rocke_ir_builder_t* b, const rocke_type_t* dtype, rocke_value_t* out[10])
{
    rocke_param_opts_t o;
    const rocke_type_t* ptr_dt = rocke_ptr_type(b, dtype, "global");
    const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");
    const rocke_type_t* ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");

    memset(&o, 0, sizeof(o));
    o.noalias = o.noalias_set = true;
    o.writeonly = o.writeonly_set = true;
    o.align = 16;
    o.align_set = true;
    out[0] = rocke_b_param(b, "output_ptr", ptr_dt, &o);

    memset(&o, 0, sizeof(o));
    o.noalias = o.noalias_set = true;
    o.readonly = o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    out[1] = rocke_b_param(b, "query_ptr", ptr_dt, &o);
    out[2] = rocke_b_param(b, "key_cache_ptr", ptr_dt, &o);
    out[3] = rocke_b_param(b, "value_cache_ptr", ptr_dt, &o);

    memset(&o, 0, sizeof(o));
    o.readonly = o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    rocke_b_param(b, "sink_ptr", ptr_dt, &o);
    out[4] = rocke_b_param(b, "block_tables_ptr", ptr_i32, &o);

    memset(&o, 0, sizeof(o));
    o.readonly = o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    out[5] = rocke_b_param(b, "seq_lens_ptr", ptr_i32, &o);
    rocke_b_param(b, "alibi_slopes_ptr", ptr_f32, &o);
    rocke_b_param(b, "qq_bias_ptr", ptr_f32, &o);

    memset(&o, 0, sizeof(o));
    o.readonly = o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    out[6] = rocke_b_param(b, "query_start_len_ptr", ptr_i32, &o);

    out[7] = rocke_b_param(b, "scale", rocke_f32(), NULL);
    rocke_b_param(b, "k_scale", rocke_f32(), NULL);
    rocke_b_param(b, "v_scale", rocke_f32(), NULL);
    rocke_b_param(b, "out_scale", rocke_f32(), NULL);
    rocke_b_param(b, "softcap", rocke_f32(), NULL);
    out[8] = rocke_b_param(b, "num_seqs", rocke_i32(), NULL);
    out[9] = rocke_b_param(b, "block_table_stride", rocke_i32(), NULL);
}

/* tid, wid, lane, decode_mfma_lanes, wq: the shared lane prologue. */
static void fwg_lanes(fwg_ctx_t* c)
{
    rocke_ir_builder_t* b = c->b;
    c->tid = rocke_b_thread_id_x(b);
    rocke_value_t* wid = rocke_b_div(b, c->tid, ci(c, 64));
    c->lane = rocke_b_mod(b, c->tid, ci(c, 64));
    c->ld = rocke_decode_mfma_lanes(b, c->at, c->lane);
    c->wq = rocke_b_mul(b, wid, ci(c, 32));
}

/* Per-CTA sequence context: binary search on cu_q, then the padding-block
 * early return. */
typedef struct fwg_seq
{
    rocke_value_t* qlen;
    rocke_value_t* klen;
    rocke_value_t* qbase;
    rocke_value_t* qstart;
} fwg_seq_t;

static fwg_seq_t fwg_seq_prologue(fwg_ctx_t* c,
                                  rocke_value_t* CUQ,
                                  rocke_value_t* KL,
                                  rocke_value_t* num_seqs,
                                  rocke_value_t* gqb,
                                  int tokblk,
                                  int iters)
{
    rocke_ir_builder_t* b = c->b;
    fwg_seq_t s;
    c->sid = rocke_binary_search_seq_idx(b, CUQ, gqb, num_seqs, tokblk, iters, false);
    rocke_value_t* cu_q_start = rocke_b_global_load_i32(b, CUQ, c->sid, 4);
    rocke_value_t* sid1 = rocke_b_add(b, c->sid, ci(c, 1));
    rocke_value_t* cu_q_stop = rocke_b_global_load_i32(b, CUQ, sid1, 4);
    s.qlen = rocke_b_sub(b, cu_q_stop, cu_q_start);
    rocke_value_t* qbs = rocke_b_div(b, cu_q_start, ci(c, tokblk));
    rocke_value_t* q_block_start = rocke_b_add(b, qbs, c->sid);
    rocke_value_t* lqb = rocke_b_sub(b, gqb, q_block_start);
    s.klen = rocke_b_global_load_i32(b, KL, c->sid, 4);
    s.qbase = rocke_b_mul(b, lqb, ci(c, tokblk));
    s.qstart = rocke_b_add(b, cu_q_start, s.qbase);
    rocke_if_t f = rocke_b_scf_if(b, rocke_b_cmp_ge(b, s.qbase, s.qlen));
    rocke_b_region_enter(b, f.then_region);
    rocke_b_ret(b);
    rocke_b_region_leave(b);
    return s;
}

static void fwg_kernel_attrs(rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec)
{
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 256);
    if(spec->has_waves_per_eu)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
}

static void fwg_q_desc(fwg_ctx_t* c, int H)
{
    static const char* const coord_names[3] = {"token", "head", "dim"};
    int lengths[3] = {1 << 30, H, c->HD};
    c->q_desc = rocke_tensor_descriptor_naive(c->b, "query_ptr", lengths, 3, NULL, coord_names, 3);
}

static void fwg_softmax_consts(fwg_ctx_t* c, rocke_value_t* scale_p)
{
    rocke_ir_builder_t* b = c->b;
    c->sc = rocke_b_fmul(b, scale_p, rocke_b_const_f32(b, FWG_LOG2E));
    c->ninf = rocke_b_const_f32(b, -1e30);
    c->zf = rocke_b_const_f32(b, 0.0);
}

static void fwg_geometry(fwg_ctx_t* c, int BN)
{
    c->BN = BN;
    c->BPT = BN / c->BS;
    c->APL = c->at->a_per_lane;
    c->BPL = c->at->b_per_lane;
    c->CPL = c->at->c_per_lane;
    c->K = c->at->k;
    c->NKEYT = BN / 32;
    c->NK = c->HD / c->K;
    c->NDdim = c->HD / 32;
    c->NKpv = BN / c->K;
}

/* The output epilogue: acc * (1/l), cast, masked per-lane store. */
static void fwg_epilogue(fwg_ctx_t* c,
                         rocke_value_t* out_ptr,
                         const fwg_seq_t* s,
                         rocke_value_t* l_f,
                         rocke_value_t* const* accs_f,
                         int H,
                         int GQAG,
                         bool fold)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* recip = rocke_b_rcp_fast(b, l_f);
    for(int nt = 0; nt < c->NDdim; nt++)
    {
        for(int i = 0; i < c->CPL; i++)
        {
            rocke_value_t *r, *col;
            fwg_lane_to_output(c, i, &r, &col);
            rocke_value_t* d0 = fwg_mul_consts(c, nt, 32);
            rocke_value_t* dim = rocke_b_add(b, d0, r);
            rocke_value_t *q_inseq, *oi;
            if(fold)
            {
                rocke_value_t* m = rocke_b_add(b, c->wq, col);
                rocke_value_t* tl = rocke_b_div(b, m, ci(c, FWG_FOLD_HEADS));
                rocke_value_t* hl = rocke_b_mod(b, m, ci(c, FWG_FOLD_HEADS));
                q_inseq = rocke_b_add(b, s->qbase, tl);
                rocke_value_t* tok = rocke_b_add(b, s->qstart, tl);
                rocke_value_t* row = rocke_b_mul(b, tok, ci(c, H));
                rocke_value_t* hb = rocke_b_mul(b, c->kvh, ci(c, GQAG));
                oi = rocke_b_add(b, row, rocke_b_add(b, hb, hl));
            }
            else
            {
                q_inseq = rocke_b_add(b, s->qbase, rocke_b_add(b, c->wq, col));
                rocke_value_t* w = rocke_b_add(b, c->wq, col);
                rocke_value_t* tok = rocke_b_add(b, s->qstart, w);
                rocke_value_t* row = rocke_b_mul(b, tok, ci(c, H));
                oi = rocke_b_add(b, row, c->qhead);
            }
            rocke_value_t* e = rocke_b_vec_extract(b, accs_f[nt], i);
            rocke_value_t* val = rocke_b_cast_f32_to(b, rocke_b_fmul(b, e, recip), c->dtype);
            rocke_if_t f = rocke_b_scf_if(b, rocke_b_cmp_lt(b, q_inseq, s->qlen));
            rocke_b_region_enter(b, f.then_region);
            {
                rocke_value_t* o = rocke_b_mul(b, oi, ci(c, c->HD));
                rocke_b_global_store(b, out_ptr, rocke_b_add(b, o, dim), val, 2);
            }
            rocke_b_region_leave(b);
        }
    }
    rocke_b_ret(b);
}

/* ===================================================================== *
 *  _build_gfx942_4warp_gqa_lean (D256)
 * ===================================================================== */
static rocke_kernel_def_t* fwg_build_lean(rocke_ir_builder_t* b,
                                          const rocke_attention_tiled_2d_spec_t* spec)
{
    fwg_ctx_t cx;
    fwg_ctx_t* c = &cx;
    memset(c, 0, sizeof(*c));
    c->b = b;
    c->dtype = rocke_attention_tiled_2d_spec_dtype_ir(spec);
    c->HD = spec->head_size;
    c->HKV = spec->num_kv_heads;
    c->BS = spec->block_size;
    c->at = rocke_mfma_atom("bf16", 32, 32, 8);
    fwg_geometry(c, 64);
    const int H = spec->num_query_heads;
    const int GQAG = rocke_attention_tiled_2d_spec_num_queries_per_kv(spec);
    const int ITERS = rocke_attention_tiled_2d_spec_binary_search_iters(spec);

    fwg_kernel_attrs(b, spec);
    rocke_value_t* P[10];
    fwg_params(b, c->dtype, P);
    rocke_value_t* Cout = P[0];
    c->Q = P[1];
    c->Kp = P[2];
    c->Vp = P[3];
    c->BT = P[4];
    c->bt_stride = P[9];

    fwg_lanes(c);
    c->qhead = rocke_b_block_id_x(b);
    c->kvh = rocke_b_div(b, c->qhead, ci(c, GQAG));
    rocke_value_t* gqb = rocke_b_block_id_y(b);
    fwg_seq_t s = fwg_seq_prologue(c, P[6], P[5], P[8], gqb, 128, ITERS);

    int vshape[2] = {c->BN, c->HD};
    rocke_value_t* V_lds = rocke_b_smem_alloc(b, c->dtype, vshape, 2, "Vlds");
    fwg_q_desc(c, H);
    fwg_softmax_consts(c, P[7]);

    rocke_iter_arg_t iters[2 + FWG_MAX_NDDIM];
    iters[0].name = "m";
    iters[0].init = c->ninf;
    iters[1].name = "l";
    iters[1].init = c->zf;
    for(int nt = 0; nt < c->NDdim; nt++)
    {
        iters[2 + nt].name = fwg_name(b, "a", nt);
        iters[2 + nt].init = rocke_b_zero_vec_f32(b, c->CPL);
    }
    rocke_value_t* context_off = rocke_b_sub(b, s.klen, s.qlen);
    rocke_value_t* cq = rocke_b_add(b, context_off, s.qbase);
    cq = rocke_b_add(b, cq, ci(c, 128 + c->BN - 1));
    rocke_value_t* causal_t = rocke_b_div(b, cq, ci(c, c->BN));
    rocke_value_t* kl = rocke_b_add(b, s.klen, ci(c, c->BN - 1));
    rocke_value_t* klen_t = rocke_b_div(b, kl, ci(c, c->BN));
    rocke_value_t* kvend = rocke_b_select(b, rocke_b_cmp_lt(b, causal_t, klen_t), causal_t, klen_t);
    rocke_value_t* lo = ci(c, 0);
    rocke_value_t* step = ci(c, 1);
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, lo, kvend, step, iters, 2 + c->NDdim, "kv", false, true);
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* kv = loop.iv;
        rocke_value_t* m_old = loop.iter_vars[0];
        rocke_value_t* l_old = loop.iter_vars[1];
        rocke_value_t* const* accs = loop.iter_vars + 2;
        rocke_b_sync(b);
        for(int cc = 0; cc < 8; cc++)
        {
            rocke_value_t* t = rocke_b_mul(b, c->tid, ci(c, 64));
            rocke_value_t* lin = rocke_b_add(b, t, ci(c, cc * 8));
            rocke_value_t* key = rocke_b_div(b, lin, ci(c, c->HD));
            rocke_value_t* hd = rocke_b_mod(b, lin, ci(c, c->HD));
            rocke_value_t* pk = fwg_phys_key(c, kv, key);
            rocke_value_t* velem = fwg_kv_elem(c, pk, hd);
            rocke_value_t* idx[2] = {key, hd};
            rocke_value_t* v = rocke_b_global_load_vN(b, c->Vp, velem, c->dtype, 8, 16);
            rocke_b_smem_store_vN(b, V_lds, idx, 2, v, 8);
        }
        rocke_b_sync(b);
        rocke_value_t* S_T[FWG_MAX_NKEYT];
        for(int kt = 0; kt < c->NKEYT; kt++)
            S_T[kt] = rocke_b_zero_vec_f32(b, c->CPL);
        rocke_value_t* pk_kt[FWG_MAX_NKEYT];
        for(int kt = 0; kt < c->NKEYT; kt++)
        {
            rocke_value_t* kr = rocke_b_add(b, ci(c, kt * 32), c->ld.m_in_atom);
            pk_kt[kt] = fwg_phys_key(c, kv, kr);
        }
        for(int h = 0; h < c->NK; h++)
        {
            rocke_value_t* koff = fwg_koff(c, h);
            rocke_value_t* q_tok = rocke_b_add(b, s.qstart, c->wq);
            q_tok = rocke_b_add(b, q_tok, c->ld.n_in_atom);
            rocke_value_t* q_off = fwg_q_offset(c, q_tok, c->qhead, koff);
            rocke_value_t* q = rocke_b_global_load_vN(b, c->Q, q_off, c->dtype, c->BPL, c->BPL * 2);
            for(int kt = 0; kt < c->NKEYT; kt++)
            {
                rocke_value_t* kelem = fwg_kv_elem(c, pk_kt[kt], koff);
                rocke_value_t* kf
                    = rocke_b_global_load_vN(b, c->Kp, kelem, c->dtype, c->APL, c->APL * 2);
                S_T[kt] = fwg_mma(c, kf, q, S_T[kt]);
            }
        }
        rocke_value_t* Sm[FWG_MAX_NKEYT][FWG_MAX_CPL];
        for(int kt = 0; kt < c->NKEYT; kt++)
        {
            for(int i = 0; i < c->CPL; i++)
            {
                rocke_value_t *rr, *col;
                fwg_lane_to_output(c, i, &rr, &col);
                rocke_value_t* k0 = rocke_b_mul(b, kv, ci(c, c->BN));
                k0 = rocke_b_add(b, k0, ci(c, kt * 32));
                rocke_value_t* key_g = rocke_b_add(b, k0, rr);
                rocke_value_t* q0 = rocke_b_add(b, c->wq, col);
                q0 = rocke_b_add(b, s.qbase, q0);
                rocke_value_t* q_g = rocke_b_add(b, context_off, q0);
                rocke_value_t* m_causal = rocke_b_cmp_gt(b, key_g, q_g);
                rocke_value_t* m_varlen = rocke_b_cmp_ge(b, key_g, s.klen);
                rocke_value_t* mask = rocke_b_lor(b, m_causal, m_varlen);
                rocke_value_t* raw = rocke_b_vec_extract(b, S_T[kt], i);
                Sm[kt][i] = rocke_b_select(b, mask, c->ninf, raw);
            }
        }
        rocke_value_t *m_new, *alpha, *l_new;
        rocke_value_t* Bp[FWG_MAX_NKPV];
        fwg_online_softmax_rescale(c, Sm, m_old, l_old, &m_new, &alpha, &l_new, Bp);
        rocke_value_t* yields[2 + FWG_MAX_NDDIM];
        yields[0] = m_new;
        yields[1] = l_new;
        for(int nt = 0; nt < c->NDdim; nt++)
        {
            rocke_value_t* pv = rocke_b_zero_vec_f32(b, c->CPL);
            for(int kk = 0; kk < c->NKpv; kk++)
            {
                rocke_value_t* parts[FWG_MAX_APL];
                for(int j = 0; j < c->APL; j++)
                {
                    rocke_value_t* r0 = fwg_mul_consts(c, kk, c->K);
                    rocke_value_t* r1 = rocke_b_mul(b, c->ld.k_blk, ci(c, c->APL));
                    r1 = rocke_b_add(b, r1, ci(c, j));
                    rocke_value_t* row = rocke_b_add(b, r0, r1);
                    rocke_value_t* c0 = fwg_mul_consts(c, nt, 32);
                    rocke_value_t* col = rocke_b_add(b, c0, c->ld.m_in_atom);
                    rocke_value_t* idx[2] = {row, col};
                    rocke_value_t* v = rocke_b_smem_load_vN(b, V_lds, idx, 2, c->dtype, 1);
                    parts[j] = rocke_b_vec_extract(b, v, 0);
                }
                rocke_value_t* va = rocke_b_vec_pack(b, parts, c->APL, c->dtype);
                pv = fwg_mma(c, va, Bp[kk], pv);
            }
            rocke_value_t* na[FWG_MAX_CPL];
            for(int i = 0; i < c->CPL; i++)
            {
                rocke_value_t* a = rocke_b_vec_extract(b, accs[nt], i);
                rocke_value_t* p = rocke_b_vec_extract(b, pv, i);
                na[i] = rocke_b_fma(b, a, alpha, p);
            }
            yields[2 + nt] = rocke_b_vec_pack(b, na, c->CPL, rocke_f32());
        }
        rocke_b_scf_yield(b, yields, 2 + c->NDdim);
    }
    rocke_b_region_leave(b);
    if(loop.op == NULL || b->status != ROCKE_OK)
        return NULL;
    rocke_value_t** res = loop.op->results;
    fwg_epilogue(c, Cout, &s, res[1], res + 2, H, GQAG, false);
    return b->status == ROCKE_OK ? b->kernel : NULL;
}

/* ===================================================================== *
 *  build_gfx942_4warp_gqa: the general body (D128 pipe / bs64 / fold)
 * ===================================================================== */
typedef struct fwg_gen
{
    fwg_ctx_t* c;
    const fwg_seq_t* s;
    bool pipe;
    bool fold;
    bool swzh;
    int window;
    int GQAG;
    int v_fill_epw;
    int v_fill_nloads;
    rocke_value_t* V_lds;
    rocke_value_t* K_lds;
    rocke_value_t* context_off;
    rocke_value_t* kvend;
    rocke_value_t* SWZ_ST[FWG_MAX_NLOADS];
    rocke_value_t* SWZ_K[FWG_MAX_NK][FWG_MAX_NKEYT];
    rocke_value_t* SWZ_V[FWG_MAX_NKPV][FWG_MAX_NDDIM][FWG_MAX_APL];
} fwg_gen_t;

/* zero-waste 8-block XOR swizzle: ((col / 8) ^ (row % 16)) * 8 + col % 8 */
static rocke_value_t* fwg_swz(fwg_ctx_t* c, rocke_value_t* row, rocke_value_t* col)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* cd = rocke_b_div(b, col, ci(c, 8));
    rocke_value_t* rm = rocke_b_mod(b, row, ci(c, 16));
    rocke_value_t* x = rocke_b_xor(b, cd, rm);
    rocke_value_t* xm = rocke_b_mul(b, x, ci(c, 8));
    rocke_value_t* cm = rocke_b_mod(b, col, ci(c, 8));
    return rocke_b_add(b, xm, cm);
}

static rocke_value_t*
    fwg_clamp(fwg_ctx_t* c, rocke_value_t* x, rocke_value_t* lo, rocke_value_t* hi)
{
    rocke_ir_builder_t* b = c->b;
    x = rocke_b_select(b, rocke_b_cmp_lt(b, x, lo), lo, x);
    return rocke_b_select(b, rocke_b_cmp_lt(b, hi, x), hi, x);
}

/* lin = tid * v_fill_epw + idx * 8; key = lin / HD; hd = lin % HD */
static void fwg_fill_coords(fwg_gen_t* g, int idx, rocke_value_t** key, rocke_value_t** hd)
{
    fwg_ctx_t* c = g->c;
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* t = rocke_b_mul(b, c->tid, ci(c, g->v_fill_epw));
    rocke_value_t* lin = rocke_b_add(b, t, ci(c, idx * 8));
    *key = rocke_b_div(b, lin, ci(c, c->HD));
    *hd = rocke_b_mod(b, lin, ci(c, c->HD));
}

static void fwg_fill_tile(fwg_gen_t* g, rocke_value_t* tkv, rocke_value_t* buf_off)
{
    fwg_ctx_t* c = g->c;
    rocke_ir_builder_t* b = c->b;
    for(int i = 0; i < g->v_fill_nloads; i++)
    {
        rocke_value_t *key, *hd;
        fwg_fill_coords(g, i, &key, &hd);
        rocke_value_t* pk = fwg_phys_key(c, tkv, key);
        rocke_value_t* velem = fwg_kv_elem(c, pk, hd);
        rocke_value_t* row = rocke_b_add(b, buf_off, key);
        rocke_value_t* vcol = g->swzh ? g->SWZ_ST[i] : (g->pipe ? fwg_swz(c, row, hd) : hd);
        rocke_value_t* vidx[2] = {row, vcol};
        rocke_value_t* v = rocke_b_global_load_vN(b, c->Vp, velem, c->dtype, 8, 16);
        rocke_b_smem_store_vN(b, g->V_lds, vidx, 2, v, 8);
        if(g->pipe)
        {
            rocke_value_t* kcol = g->swzh ? g->SWZ_ST[i] : fwg_swz(c, row, hd);
            rocke_value_t* kidx[2] = {row, kcol};
            rocke_value_t* k = rocke_b_global_load_vN(b, c->Kp, velem, c->dtype, 8, 16);
            rocke_b_smem_store_vN(b, g->K_lds, kidx, 2, k, 8);
        }
    }
}

typedef struct fwg_prefetch
{
    rocke_value_t* vr[FWG_MAX_NLOADS];
    rocke_value_t* kr[FWG_MAX_NLOADS];
    rocke_value_t* key[FWG_MAX_NLOADS];
    rocke_value_t* hd[FWG_MAX_NLOADS];
} fwg_prefetch_t;

static void fwg_fill_load(fwg_gen_t* g, rocke_value_t* tkv, fwg_prefetch_t* pf)
{
    fwg_ctx_t* c = g->c;
    rocke_ir_builder_t* b = c->b;
    for(int i = 0; i < g->v_fill_nloads; i++)
    {
        rocke_value_t *key, *hd;
        fwg_fill_coords(g, i, &key, &hd);
        rocke_value_t* pk = fwg_phys_key(c, tkv, key);
        rocke_value_t* velem = fwg_kv_elem(c, pk, hd);
        pf->vr[i] = rocke_b_global_load_vN(b, c->Vp, velem, c->dtype, 8, 16);
        pf->kr[i] = rocke_b_global_load_vN(b, c->Kp, velem, c->dtype, 8, 16);
        pf->key[i] = key;
        pf->hd[i] = hd;
    }
}

static void fwg_fill_store(fwg_gen_t* g, const fwg_prefetch_t* pf, rocke_value_t* buf_off)
{
    fwg_ctx_t* c = g->c;
    rocke_ir_builder_t* b = c->b;
    for(int i = 0; i < g->v_fill_nloads; i++)
    {
        rocke_value_t* row = rocke_b_add(b, buf_off, pf->key[i]);
        rocke_value_t* col = g->swzh ? g->SWZ_ST[i] : fwg_swz(c, row, pf->hd[i]);
        rocke_value_t* idx[2] = {row, col};
        rocke_b_smem_store_vN(b, g->V_lds, idx, 2, pf->vr[i], 8);
        rocke_b_smem_store_vN(b, g->K_lds, idx, 2, pf->kr[i], 8);
    }
}

/* body(kv, carry, masked): one KV tile. Writes the yielded values into out. */
static void fwg_body(
    fwg_gen_t* g, rocke_value_t* kv, rocke_value_t* const* carry, bool masked, rocke_value_t** out)
{
    fwg_ctx_t* c = g->c;
    rocke_ir_builder_t* b = c->b;
    const fwg_seq_t* s = g->s;
    rocke_value_t* m_old = carry[0];
    rocke_value_t* l_old = carry[1];
    rocke_value_t* const* accs = carry + 2;
    rocke_value_t* buf_off;
    rocke_value_t* nbuf_off = NULL;
    rocke_value_t* pk_kt[FWG_MAX_NKEYT] = {NULL};
    fwg_prefetch_t pf;
    if(g->pipe)
    {
        rocke_value_t* km = rocke_b_mod(b, kv, ci(c, 2));
        buf_off = rocke_b_mul(b, km, ci(c, 32));
        rocke_value_t* kn = rocke_b_add(b, kv, ci(c, 1));
        rocke_value_t* in_range = rocke_b_cmp_lt(b, kn, g->kvend);
        rocke_value_t* kn2 = rocke_b_add(b, kv, ci(c, 1));
        rocke_value_t* last = rocke_b_sub(b, g->kvend, ci(c, 1));
        rocke_value_t* knext = rocke_b_select(b, in_range, kn2, last);
        rocke_value_t* kn3 = rocke_b_add(b, kv, ci(c, 1));
        rocke_value_t* nm = rocke_b_mod(b, kn3, ci(c, 2));
        nbuf_off = rocke_b_mul(b, nm, ci(c, 32));
        fwg_fill_load(g, knext, &pf);
    }
    else
    {
        buf_off = ci(c, 0);
        rocke_b_sync(b);
        fwg_fill_tile(g, kv, buf_off);
        rocke_b_sync(b);
        for(int kt = 0; kt < c->NKEYT; kt++)
        {
            rocke_value_t* kr = rocke_b_add(b, ci(c, kt * 32), c->ld.m_in_atom);
            pk_kt[kt] = fwg_phys_key(c, kv, kr);
        }
    }
    rocke_value_t* S_T[FWG_MAX_NKEYT];
    for(int kt = 0; kt < c->NKEYT; kt++)
        S_T[kt] = rocke_b_zero_vec_f32(b, c->CPL);
    for(int h = 0; h < c->NK; h++)
    {
        rocke_value_t* koff = fwg_koff(c, h);
        rocke_value_t* q_off;
        if(g->fold)
        {
            rocke_value_t* m = rocke_b_add(b, c->wq, c->ld.n_in_atom);
            rocke_value_t* md = rocke_b_div(b, m, ci(c, FWG_FOLD_HEADS));
            rocke_value_t* tok = rocke_b_add(b, s->qstart, md);
            rocke_value_t* hb = rocke_b_mul(b, c->kvh, ci(c, g->GQAG));
            rocke_value_t* mm = rocke_b_mod(b, m, ci(c, FWG_FOLD_HEADS));
            rocke_value_t* head = rocke_b_add(b, hb, mm);
            q_off = fwg_q_offset(c, tok, head, koff);
        }
        else
        {
            rocke_value_t* q_tok = rocke_b_add(b, s->qstart, c->wq);
            q_tok = rocke_b_add(b, q_tok, c->ld.n_in_atom);
            q_off = fwg_q_offset(c, q_tok, c->qhead, koff);
        }
        rocke_value_t* q = rocke_b_global_load_vN(b, c->Q, q_off, c->dtype, c->BPL, c->BPL * 2);
        for(int kt = 0; kt < c->NKEYT; kt++)
        {
            rocke_value_t* kf;
            if(g->pipe)
            {
                rocke_value_t* r = rocke_b_add(b, ci(c, kt * 32), c->ld.m_in_atom);
                rocke_value_t* row_k = rocke_b_add(b, buf_off, r);
                rocke_value_t* kcol = g->swzh ? g->SWZ_K[h][kt] : fwg_swz(c, row_k, koff);
                rocke_value_t* idx[2] = {row_k, kcol};
                kf = rocke_b_smem_load_vN(b, g->K_lds, idx, 2, c->dtype, c->APL);
            }
            else
            {
                rocke_value_t* kelem = fwg_kv_elem(c, pk_kt[kt], koff);
                kf = rocke_b_global_load_vN(b, c->Kp, kelem, c->dtype, c->APL, c->APL * 2);
            }
            S_T[kt] = fwg_mma(c, kf, q, S_T[kt]);
        }
    }
    rocke_value_t* Sm[FWG_MAX_NKEYT][FWG_MAX_CPL];
    for(int kt = 0; kt < c->NKEYT; kt++)
    {
        for(int i = 0; i < c->CPL; i++)
        {
            rocke_value_t* raw = rocke_b_vec_extract(b, S_T[kt], i);
            if(!masked)
            {
                Sm[kt][i] = raw;
                continue;
            }
            rocke_value_t *rr, *col;
            fwg_lane_to_output(c, i, &rr, &col);
            rocke_value_t* k0 = rocke_b_mul(b, kv, ci(c, c->BN));
            k0 = rocke_b_add(b, k0, ci(c, kt * 32));
            rocke_value_t* key_g = rocke_b_add(b, k0, rr);
            rocke_value_t* q0 = rocke_b_add(b, c->wq, col);
            if(g->fold)
                q0 = rocke_b_div(b, q0, ci(c, FWG_FOLD_HEADS));
            q0 = rocke_b_add(b, s->qbase, q0);
            rocke_value_t* q_g = rocke_b_add(b, g->context_off, q0);
            rocke_value_t* mc = rocke_b_cmp_gt(b, key_g, q_g);
            rocke_value_t* mv = rocke_b_cmp_ge(b, key_g, s->klen);
            rocke_value_t* mask_cond = rocke_b_lor(b, mc, mv);
            if(g->window > 0)
            {
                rocke_value_t* dist = rocke_b_sub(b, q_g, key_g);
                rocke_value_t* mw = rocke_b_cmp_ge(b, dist, ci(c, g->window));
                mask_cond = rocke_b_lor(b, mask_cond, mw);
            }
            Sm[kt][i] = rocke_b_select(b, mask_cond, c->ninf, raw);
        }
    }
    rocke_value_t *m_new, *alpha, *l_new;
    rocke_value_t* Bp[FWG_MAX_NKPV];
    fwg_online_softmax_rescale(c, Sm, m_old, l_old, &m_new, &alpha, &l_new, Bp);
    /* The pipe rescales acc in place and accumulates PV into it; D256 / bs64
     * keep a zero-initialized PV accumulator plus an fma. */
    rocke_value_t* acc_tgt[FWG_MAX_NDDIM];
    for(int nt = 0; nt < c->NDdim; nt++)
    {
        if(g->pipe)
        {
            rocke_value_t* sv[FWG_MAX_CPL];
            for(int i = 0; i < c->CPL; i++)
                sv[i] = rocke_b_fmul(b, rocke_b_vec_extract(b, accs[nt], i), alpha);
            acc_tgt[nt] = rocke_b_vec_pack(b, sv, c->CPL, rocke_f32());
        }
        else
        {
            acc_tgt[nt] = rocke_b_zero_vec_f32(b, c->CPL);
        }
    }
    for(int kk = 0; kk < c->NKpv; kk++)
    {
        for(int nt = 0; nt < c->NDdim; nt++)
        {
            rocke_value_t* c0 = fwg_mul_consts(c, nt, 32);
            rocke_value_t* col_v = rocke_b_add(b, c0, c->ld.m_in_atom);
            rocke_value_t* vparts[FWG_MAX_APL];
            for(int j = 0; j < c->APL; j++)
            {
                rocke_value_t* r0 = fwg_mul_consts(c, kk, c->K);
                rocke_value_t* r1 = rocke_b_mul(b, c->ld.k_blk, ci(c, c->APL));
                r1 = rocke_b_add(b, r1, ci(c, j));
                rocke_value_t* row_v = rocke_b_add(b, buf_off, rocke_b_add(b, r0, r1));
                rocke_value_t* vcol
                    = g->swzh ? g->SWZ_V[kk][nt][j] : (g->pipe ? fwg_swz(c, row_v, col_v) : col_v);
                rocke_value_t* idx[2] = {row_v, vcol};
                rocke_value_t* v = rocke_b_smem_load_vN(b, g->V_lds, idx, 2, c->dtype, 1);
                vparts[j] = rocke_b_vec_extract(b, v, 0);
            }
            rocke_value_t* va = rocke_b_vec_pack(b, vparts, c->APL, c->dtype);
            acc_tgt[nt] = fwg_mma(c, va, Bp[kk], acc_tgt[nt]);
        }
    }
    out[0] = m_new;
    out[1] = l_new;
    for(int nt = 0; nt < c->NDdim; nt++)
    {
        if(g->pipe)
        {
            out[2 + nt] = acc_tgt[nt];
            continue;
        }
        rocke_value_t* na[FWG_MAX_CPL];
        for(int i = 0; i < c->CPL; i++)
        {
            rocke_value_t* a = rocke_b_vec_extract(b, accs[nt], i);
            rocke_value_t* p = rocke_b_vec_extract(b, acc_tgt[nt], i);
            na[i] = rocke_b_fma(b, a, alpha, p);
        }
        out[2 + nt] = rocke_b_vec_pack(b, na, c->CPL, rocke_f32());
    }
    if(g->pipe)
    {
        fwg_fill_store(g, &pf, nbuf_off);
        rocke_b_sync(b);
    }
}

static rocke_for_t fwg_run_loop(fwg_gen_t* g,
                                rocke_value_t* start,
                                rocke_value_t* end,
                                const rocke_iter_arg_t* init,
                                bool masked,
                                const char* iv)
{
    fwg_ctx_t* c = g->c;
    rocke_ir_builder_t* b = c->b;
    int n = 2 + c->NDdim;
    rocke_value_t* step = ci(c, 1);
    rocke_for_t lp = rocke_b_scf_for_iter(b, start, end, step, init, n, iv, false, true);
    rocke_b_region_enter(b, lp.body);
    if(lp.op != NULL)
    {
        rocke_value_t* out[2 + FWG_MAX_NDDIM];
        fwg_body(g, lp.iv, lp.iter_vars, masked, out);
        rocke_b_scf_yield(b, out, n);
    }
    rocke_b_region_leave(b);
    return lp;
}

/* (m{tag}, l{tag}, {acc_prefix}{nt}) iter args seeded from a loop's results. */
static void fwg_chain_iters(fwg_ctx_t* c,
                            const rocke_for_t* prev,
                            const char* m_name,
                            const char* l_name,
                            const char* acc_prefix,
                            rocke_iter_arg_t* out)
{
    out[0].name = m_name;
    out[0].init = prev->op->results[0];
    out[1].name = l_name;
    out[1].init = prev->op->results[1];
    for(int nt = 0; nt < c->NDdim; nt++)
    {
        out[2 + nt].name = fwg_name(c->b, acc_prefix, nt);
        out[2 + nt].init = prev->op->results[2 + nt];
    }
}

static rocke_kernel_def_t* fwg_build_general(rocke_ir_builder_t* b,
                                             const rocke_attention_tiled_2d_spec_t* spec,
                                             const char* arch,
                                             bool pipe,
                                             bool fold)
{
    fwg_ctx_t cx;
    fwg_ctx_t* c = &cx;
    memset(c, 0, sizeof(*c));
    fwg_gen_t gx;
    fwg_gen_t* g = &gx;
    memset(g, 0, sizeof(*g));
    g->c = c;
    g->pipe = pipe;
    g->fold = fold;
    c->b = b;
    c->dtype = rocke_attention_tiled_2d_spec_dtype_ir(spec);
    c->HD = spec->head_size;
    c->HKV = spec->num_kv_heads;
    c->BS = spec->block_size;
    c->at = rocke_mfma_atom(fwg_streq(spec->dtype, "bf16") ? "bf16" : "f16", 32, 32, 8);
    fwg_geometry(c, pipe ? 32 : 64);
    const int H = spec->num_query_heads;
    g->GQAG = rocke_attention_tiled_2d_spec_num_queries_per_kv(spec);
    g->v_fill_epw = (c->BN * c->HD) / 256;
    g->v_fill_nloads = g->v_fill_epw / 8;
    const int ITERS = rocke_attention_tiled_2d_spec_binary_search_iters(spec);
    const int TOKBLK = fold ? 128 / FWG_FOLD_HEADS : 128;

    fwg_kernel_attrs(b, spec);
    rocke_value_t* P[10];
    fwg_params(b, c->dtype, P);
    rocke_value_t* Cout = P[0];
    c->Q = P[1];
    c->Kp = P[2];
    c->Vp = P[3];
    c->BT = P[4];
    c->bt_stride = P[9];

    fwg_lanes(c);
    if(fold)
    {
        c->qhead = NULL;
        c->kvh = rocke_b_block_id_x(b);
    }
    else
    {
        c->qhead = rocke_b_block_id_x(b);
        c->kvh = rocke_b_div(b, c->qhead, ci(c, g->GQAG));
    }
    rocke_value_t* gqb = rocke_b_block_id_y(b);
    fwg_seq_t s = fwg_seq_prologue(c, P[6], P[5], P[8], gqb, TOKBLK, ITERS);
    g->s = &s;

    int shape[2] = {64, c->HD};
    g->V_lds = rocke_b_smem_alloc(b, c->dtype, shape, 2, "Vlds");
    if(pipe)
        g->K_lds = rocke_b_smem_alloc(b, c->dtype, shape, 2, "Klds");

    /* Build-time LDS-capacity guard behind the admission gate. */
    {
        const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
        int cap = t != NULL ? t->lds_capacity_bytes : 65536;
        int v_bytes = 64 * c->HD * 2;
        int k_bytes = pipe ? 64 * c->HD * 2 : 0;
        if(v_bytes + k_bytes > cap)
        {
            char m[ROCKE_ERR_MSG_CAP];
            snprintf(m,
                     sizeof(m),
                     "4-warp GQA kernel LDS footprint %d B (V_lds %d + K_lds %d, HD=%d, "
                     "HD128_PIPE=%s) exceeds the %s %d B LDS capacity",
                     v_bytes + k_bytes,
                     v_bytes,
                     k_bytes,
                     c->HD,
                     pipe ? "True" : "False",
                     arch,
                     cap);
            fwg_fail(b, ROCKE_ERR_VALUE, m);
            return NULL;
        }
    }
    fwg_q_desc(c, H);
    fwg_softmax_consts(c, P[7]);

    rocke_iter_arg_t iters[2 + FWG_MAX_NDDIM];
    iters[0].name = "m";
    iters[0].init = c->ninf;
    iters[1].name = "l";
    iters[1].init = c->zf;
    for(int nt = 0; nt < c->NDdim; nt++)
    {
        iters[2 + nt].name = fwg_name(b, "a", nt);
        iters[2 + nt].init = rocke_b_zero_vec_f32(b, c->CPL);
    }
    g->context_off = rocke_b_sub(b, s.klen, s.qlen);
    g->window = spec->sliding_window;
    const int BN = c->BN;
    rocke_value_t* cq = rocke_b_add(b, g->context_off, s.qbase);
    cq = rocke_b_add(b, cq, ci(c, TOKBLK + BN - 1));
    rocke_value_t* causal_t = rocke_b_div(b, cq, ci(c, BN));
    rocke_value_t* kl = rocke_b_add(b, s.klen, ci(c, BN - 1));
    rocke_value_t* klen_t = rocke_b_div(b, kl, ci(c, BN));
    g->kvend = rocke_b_select(b, rocke_b_cmp_lt(b, causal_t, klen_t), causal_t, klen_t);
    rocke_value_t* kvstart;
    if(g->window > 0)
    {
        rocke_value_t* ks = rocke_b_add(b, g->context_off, s.qbase);
        ks = rocke_b_sub(b, ks, ci(c, g->window));
        rocke_value_t* pos = rocke_b_cmp_gt(b, ks, ci(c, 0));
        ks = rocke_b_select(b, pos, ks, ci(c, 0));
        kvstart = rocke_b_div(b, ks, ci(c, BN));
    }
    else
    {
        kvstart = ci(c, 0);
    }
    /* Loop split: fully interior tiles skip the per-element causal / window /
     * varlen mask. */
    rocke_value_t* q_blk_lo = rocke_b_add(b, g->context_off, s.qbase);
    rocke_value_t* q_blk_hi = rocke_b_add(b, g->context_off, s.qbase);
    q_blk_hi = rocke_b_add(b, q_blk_hi, ci(c, TOKBLK - 1));
    rocke_value_t* cn = rocke_b_sub(b, q_blk_lo, ci(c, BN - 1));
    rocke_value_t* cn_neg = rocke_b_cmp_lt(b, cn, ci(c, 0));
    rocke_value_t* zero = ci(c, 0);
    rocke_value_t* cnd = rocke_b_div(b, cn, ci(c, BN));
    rocke_value_t* cn1 = rocke_b_add(b, cnd, ci(c, 1));
    rocke_value_t* int_end_c = rocke_b_select(b, cn_neg, zero, cn1);
    rocke_value_t* int_end_r = rocke_b_div(b, s.klen, ci(c, BN));
    rocke_value_t* int_end
        = rocke_b_select(b, rocke_b_cmp_lt(b, int_end_c, int_end_r), int_end_c, int_end_r);
    rocke_value_t* int_start;
    if(g->window > 0)
    {
        rocke_value_t* sn = rocke_b_add(b, q_blk_hi, ci(c, 1));
        sn = rocke_b_sub(b, sn, ci(c, g->window));
        rocke_value_t* sc = rocke_b_add(b, sn, ci(c, BN - 1));
        rocke_value_t* cs = rocke_b_div(b, sc, ci(c, BN));
        int_start = rocke_b_select(b, rocke_b_cmp_lt(b, sn, ci(c, 0)), kvstart, cs);
    }
    else
    {
        int_start = kvstart;
    }
    rocke_value_t* a_bnd = fwg_clamp(c, int_start, kvstart, g->kvend);
    rocke_value_t* b_bnd = fwg_clamp(c, int_end, a_bnd, g->kvend);

    /* Swizzle-hoist: the bank-swizzle columns depend only on the lane, so the
     * pipe computes them once instead of per K/V store and read. */
    g->swzh = pipe && (fwg_streq(spec->dtype, "fp16") || fwg_streq(spec->dtype, "bf16"));
    if(g->swzh)
    {
        rocke_value_t* mia = c->ld.m_in_atom;
        rocke_value_t* kbapl = rocke_b_mul(b, c->ld.k_blk, ci(c, c->APL));
        for(int i = 0; i < g->v_fill_nloads; i++)
        {
            rocke_value_t *key, *hd;
            fwg_fill_coords(g, i, &key, &hd);
            g->SWZ_ST[i] = fwg_swz(c, key, hd);
        }
        for(int h = 0; h < c->NK; h++)
        {
            rocke_value_t* koff = rocke_b_add(b, ci(c, h * c->K), kbapl);
            for(int kt = 0; kt < c->NKEYT; kt++)
            {
                rocke_value_t* r = rocke_b_add(b, ci(c, kt * 32), mia);
                g->SWZ_K[h][kt] = fwg_swz(c, r, koff);
            }
        }
        for(int kk = 0; kk < c->NKpv; kk++)
        {
            for(int nt = 0; nt < c->NDdim; nt++)
            {
                rocke_value_t* colv = rocke_b_add(b, ci(c, nt * 32), mia);
                for(int j = 0; j < c->APL; j++)
                {
                    rocke_value_t* c0 = ci(c, kk * c->K);
                    rocke_value_t* rnb = rocke_b_add(b, c0, rocke_b_add(b, kbapl, ci(c, j)));
                    g->SWZ_V[kk][nt][j] = fwg_swz(c, rnb, colv);
                }
            }
        }
    }

    if(pipe)
    {
        rocke_if_t f = rocke_b_scf_if(b, rocke_b_cmp_lt(b, kvstart, g->kvend));
        rocke_b_region_enter(b, f.then_region);
        {
            rocke_value_t* m = rocke_b_mod(b, kvstart, ci(c, 2));
            fwg_fill_tile(g, kvstart, rocke_b_mul(b, m, ci(c, 32)));
        }
        rocke_b_region_leave(b);
        rocke_b_sync(b);
    }
    rocke_for_t l1 = fwg_run_loop(g, kvstart, a_bnd, iters, true, "kva");
    if(l1.op == NULL || b->status != ROCKE_OK)
        return NULL;
    rocke_iter_arg_t it1[2 + FWG_MAX_NDDIM];
    fwg_chain_iters(c, &l1, "m2", "l2", "b", it1);
    rocke_for_t l2 = fwg_run_loop(g, a_bnd, b_bnd, it1, false, "kvb");
    if(l2.op == NULL || b->status != ROCKE_OK)
        return NULL;
    rocke_iter_arg_t it2[2 + FWG_MAX_NDDIM];
    fwg_chain_iters(c, &l2, "m3", "l3", "c", it2);
    rocke_for_t l3 = fwg_run_loop(g, b_bnd, g->kvend, it2, true, "kvc");
    if(l3.op == NULL || b->status != ROCKE_OK)
        return NULL;
    rocke_value_t** res = l3.op->results;
    fwg_epilogue(c, Cout, &s, res[1], res + 2, H, g->GQAG, fold);
    return b->status == ROCKE_OK ? b->kernel : NULL;
}

/* ===================================================================== *
 *  build_gfx942_4warp_gqa(spec, arch)
 * ===================================================================== */
static rocke_kernel_def_t*
    fwg_build(rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch)
{
    const int HD = spec->head_size;
    const int BS = spec->block_size;
    const bool pipe = HD == 128 && BS <= 32;
    const int GQAG = rocke_attention_tiled_2d_spec_num_queries_per_kv(spec);
    const bool fold
        = pipe && rocke_gfx942_gqa_fold_eligible(HD, GQAG, spec->sliding_window, spec->dtype, BS);

    char base[1024];
    char kname[1100];
    bool named = rocke_attention_tiled_2d_spec_kernel_name(spec, base, sizeof(base)) == ROCKE_OK;
    if(named)
        snprintf(kname, sizeof(kname), "%s%s", base, fold ? "_4wgqa_fold" : "_4wgqa");
    if(rocke_ir_builder_init(b, named ? kname : "gfx942_4warp_gqa") != ROCKE_OK)
        return NULL;
    if(!named)
    {
        fwg_fail(b, ROCKE_ERR_VALUE, "spec.kernel_name() failed");
        return NULL;
    }
    /* Python validates the spec when it is constructed, before the builder. */
    if(!rocke_attention_tiled_2d_spec_validate(b, spec))
        return NULL;
    if(!rocke_require_tiled_attention_arch(b, arch))
        return NULL;
    if(spec->use_i64_kv_addr)
    {
        fwg_fail(b,
                 ROCKE_ERR_NOTIMPL,
                 "4-warp GQA kernel does not support use_i64_kv_addr: its K/V are i32 "
                 "element offsets, correct up to 2^31 elements of the cache dtype");
        return NULL;
    }
    if(!(fwg_streq(spec->dtype, "bf16") || fwg_streq(spec->dtype, "fp16")))
    {
        fwg_fail(b, ROCKE_ERR_NOTIMPL, "4-warp GQA kernel supports dtype in {bf16, fp16}");
        return NULL;
    }
    if(HD != 128 && HD != 256)
    {
        fwg_fail(b, ROCKE_ERR_NOTIMPL, "4-warp GQA kernel supports head_size in {128, 256}");
        return NULL;
    }
    if(HD == FWG_LEAN_HEAD_SIZE)
    {
        if(!fwg_streq(spec->dtype, "bf16"))
        {
            fwg_fail(b, ROCKE_ERR_NOTIMPL, "4-warp GQA kernel is bf16-only");
            return NULL;
        }
        if(spec->sliding_window)
        {
            fwg_fail(b,
                     ROCKE_ERR_NOTIMPL,
                     "4-warp GQA lean body is causal-only (no sliding-window mask)");
            return NULL;
        }
        return fwg_build_lean(b, spec);
    }
    if(fold && GQAG != FWG_FOLD_HEADS)
    {
        char m[ROCKE_ERR_MSG_CAP];
        snprintf(m,
                 sizeof(m),
                 "GQA head-fold is wired for exactly %d query heads per KV head (128-row "
                 "M-tile = %d heads x %d tokens), got num_queries_per_kv=%d. Widening the "
                 "fold cohort requires re-deriving the M-tile row split, not just the "
                 "predicate.",
                 FWG_FOLD_HEADS,
                 FWG_FOLD_HEADS,
                 128 / FWG_FOLD_HEADS,
                 GQAG);
        fwg_fail(b, ROCKE_ERR_VALUE, m);
        return NULL;
    }
    return fwg_build_general(b, spec, arch, pipe, fold);
}

rocke_kernel_def_t* rocke_build_gfx942_4warp_gqa(rocke_ir_builder_t* b,
                                                 const rocke_attention_tiled_2d_spec_t* spec,
                                                 const char* arch)
{
    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx942";
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return fwg_build(b, spec, arch); });
}

rocke_status_t
    rocke_gfx942_attention_4warp_gqa_lower_to_llvm(const rocke_attention_tiled_2d_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        fwg_set_msg(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_gfx942_4warp_gqa(&b, spec, arch);
    if(kernel == NULL)
    {
        rocke_status_t st = rocke_ir_builder_status(&b);
        fwg_set_msg(err, err_cap, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return st == ROCKE_OK ? ROCKE_ERR_VALUE : st;
    }
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, arch ? arch : "gfx942", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
