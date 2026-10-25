// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/attention_dense_bwd_dq.py: the dense split
 * dQ kernel's spec (__post_init__, kernel_name), validation, host-side band
 * classification (dense_dq_static_classes) and emitter (emit_dense_bwd_dq)
 * (rocke/instance_attention_dense_bwd_dq.h).
 *
 * Byte identity: every builder call (constants included) is bound to a local
 * in Python's left-to-right, inner-first evaluation order. A Python raise is
 * a ckc::Error throw with the same message; the entry points catch it.
 */
#include "rocke/instance_attention_dense_bwd_dq.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_dqbody.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"
#include "rocke/helper_rocke.instances.common._attention_dense_bwd_shell.h"
#include "rocke/instance_attention_bwd.h"
#include "rocke/instance_attention_dense_bwd_spec.h"
#include "rocke/ir_internal.h"

namespace rocke_attn_bwd
{
namespace
{

typedef rocke_ir_builder_t B;
typedef rocke_value_t* V;
typedef rocke_attn_bwd_iv_t IV;
typedef rocke_attn_bwd_dense_dq_spec_t DSpec;
typedef rocke_attn_bwd_dq_spec_t DqSpec;
typedef std::vector<std::pair<std::string, std::string>> Items;

const int NONE = ROCKE_ATTN_BWD_NONE;
const int ELEM_BYTES = 2;
const int MIN_WG_ATTR = 64;
const long long VOFFSET_LIMIT = 1LL << 31;
const long long TILE_SLACK_ROWS = 256;

bool eq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
        return a == b;
    return strcmp(a, b) == 0;
}

std::string s_(const char* s)
{
    return s ? std::string(s) : std::string();
}

std::string repr_str(const char* s)
{
    return s == NULL ? std::string("None") : py_repr(s);
}

bool live(const B* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

[[noreturn]] void raise_builder(B* b)
{
    if(b != NULL && b->status != ROCKE_OK)
        ckc::raise_status(b->status, b->err);
    throw ckc::ValueError("dense attention backward dQ kernel: a helper failed");
}

void need(B* b, bool ok)
{
    if(!ok || !live(b))
        raise_builder(b);
}

V needv(B* b, V v)
{
    if(v == NULL || !live(b))
        raise_builder(b);
    return v;
}

V ci(B* b, int64_t x)
{
    return rocke_b_const_i32(b, x);
}

IV ivv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

void raise_err(const rocke_attn_bwd_err_t& e)
{
    ckc::raise_status(e.code, e.msg);
}

/* problem_spec(): the dense main spec of the same problem (base class). */
rocke_attn_bwd_dense_spec_t problem_spec(const DSpec& s)
{
    rocke_attn_bwd_dense_spec_t p = rocke_attn_bwd_dense_spec_default(NULL);
    p.seqlen_q = s.seqlen_q;
    p.seqlen_kv = s.seqlen_kv;
    p.num_query_heads = s.num_query_heads;
    p.num_kv_heads = s.num_kv_heads;
    p.head_size = s.head_size;
    p.batch = s.batch;
    p.causal = s.causal;
    p.causal_bottom_right = s.causal_bottom_right;
    p.dtype = s.dtype;
    p.layout = s.layout;
    p.sliding_window = s.sliding_window;
    p.window_left = s.window_left;
    p.window_right = s.window_right;
    p.ragged = true;
    p.oob = s.oob;
    p.ws_layout = s.ws_layout;
    p.name = s.name;
    return p;
}

int group_of(const DSpec& s)
{
    return s.num_query_heads / s.num_kv_heads;
}

void strides_of(const DSpec& s, const char* tensor, long long out[3])
{
    rocke_attn_bwd_dense_spec_t p = problem_spec(s);
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_dense_spec_strides(&p, tensor, out, &e) != ROCKE_OK)
        raise_err(e);
}

/* __post_init__ */
void dense_dq_init(DSpec& s)
{
    /* the dense spec's own problem rules (the tile is checked per kernel) */
    rocke_attn_bwd_dense_spec_t p = problem_spec(s);
    dense_spec_init(p);
    bool masked = rocke_attn_bwd_dense_spec_masked(&p);
    const char* mask_class = masked ? "band" : "none";
    rocke_attn_bwd_err_t e;
    if(!s.has_dq)
    {
        DqSpec d = rocke_attn_bwd_dq_spec_default();
        d.head_size = s.head_size;
        d.dtype = s.dtype;
        d.seq_mode = "batched";
        d.stage_vec = 8;
        d.mask_class = mask_class;
        d.ws_layout = s.ws_layout;
        if(rocke_attn_bwd_dq_spec_init(&d, &e) != ROCKE_OK)
            raise_err(e);
        s.dq = d;
        s.has_dq = true;
    }
    auto mismatch = [](const char* k, const std::string& got, const std::string& want) {
        throw ckc::ValueError("dq." + std::string(k) + "=" + got
                              + " does not match the dense problem (" + want + ")");
    };
    if(s.dq.head_size != s.head_size)
        mismatch("head_size", std::to_string(s.dq.head_size), std::to_string(s.head_size));
    if(!eq(s.dq.dtype, s.dtype))
        mismatch("dtype", repr_str(s.dq.dtype), repr_str(s.dtype));
    if(!eq(s.dq.seq_mode, "batched"))
        mismatch("seq_mode", repr_str(s.dq.seq_mode), "'batched'");
    if(s.dq.stage_vec != 8)
        mismatch("stage_vec", std::to_string(s.dq.stage_vec), "8");
    if(!eq(s.dq.mask_class, mask_class))
        mismatch("mask_class", repr_str(s.dq.mask_class), repr_str(mask_class));
    if(!eq(s.dq.ws_layout, s.ws_layout))
        mismatch("ws_layout", repr_str(s.dq.ws_layout), repr_str(s.ws_layout));
}

std::string dense_dq_kernel_name(const DSpec& s)
{
    rocke_attn_bwd_dense_spec_t p = problem_spec(s);
    int left, right;
    bool br;
    rocke_attn_bwd_dense_spec_band(&p, &left, &right, &br);
    std::string base = s_(s.name) + "_dq_" + s_(s.dtype) + "_d" + std::to_string(s.head_size) + "_"
                       + s_(s.layout) + "_sq" + std::to_string(s.seqlen_q) + "_sk"
                       + std::to_string(s.seqlen_kv) + "_hq" + std::to_string(s.num_query_heads)
                       + "_hk" + std::to_string(s.num_kv_heads) + "_"
                       + dense_mask_token(left, right, br);
    if(s.batch != NONE)
        base += "_b" + std::to_string(s.batch);
    Items salted;
    if(!eq(s.oob, "buffer_oob"))
        salted.emplace_back("oob", repr_str(s.oob));
    if(!eq(s.ws_layout, "head_major"))
        salted.emplace_back("ws_layout", repr_str(s.ws_layout));
    static const char* const dq_problem[]
        = {"head_size", "dtype", "seq_mode", "stage_vec", "mask_class", "ws_layout"};
    for(const auto& kv : dq_non_default_fields(s.dq))
    {
        bool skip = false;
        for(const char* x : dq_problem)
            skip = skip || kv.first == x;
        if(!skip)
            salted.emplace_back("dq_" + kv.first, kv.second);
    }
    return salted_kernel_name(base, salted);
}

/* dq_rows_per_head(spec, group) of a resolved spec. */
int dq_rows_per_head(const DqSpec& s, int group)
{
    if(eq(s.head_loop, "single"))
        return s.block_m;
    if(group < 1 || s.block_m % group)
        throw ckc::ValueError("head_loop='group' needs the head group G=" + std::to_string(group)
                              + " to divide block_m=" + std::to_string(s.block_m));
    return s.block_m / group;
}

DqSpec dense_dq_validate(const DSpec& spec, const std::string& arch)
{
    if(std::find(DENSE_BWD_ARCHS.begin(), DENSE_BWD_ARCHS.end(), arch) == DENSE_BWD_ARCHS.end())
        throw ckc::ValueError("arch=" + py_repr(arch) + " not in " + py_tuple(DENSE_BWD_ARCHS));
    DqSpec dq;
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_validate_dq_spec(&spec.dq, arch.c_str(), &dq, &e) != ROCKE_OK)
        raise_err(e);
    dq_rows_per_head(dq, group_of(spec));
    const char* tensors[2] = {"q", "k"};
    for(const char* tensor : tensors)
    {
        long long st[3];
        strides_of(spec, tensor, st);
        long long seq = eq(tensor, "q") ? spec.seqlen_q : spec.seqlen_kv;
        long long rows
            = seq + std::max(TILE_SLACK_ROWS, (long long)std::max(dq.block_n, dq.block_m));
        if(rows * st[2] * ELEM_BYTES >= VOFFSET_LIMIT)
            throw ckc::ValueError(std::string(tensor) + " slab of " + std::to_string(seq)
                                  + " rows at token stride " + std::to_string(st[2])
                                  + " passes the 32-bit buffer offset");
    }
    if(spec.batch != NONE)
    {
        long long rows = (long long)spec.batch * std::max(spec.seqlen_q, spec.seqlen_kv);
        if(rows * spec.num_query_heads * spec.head_size >= VOFFSET_LIMIT)
            throw ckc::ValueError("batch * rows * heads * head_size passes 2^31 elements");
    }
    return dq;
}

/* ---------------------------------------------- host-side classification */

/* HostInt.div */
int64_t hdiv(int64_t x, int64_t y)
{
    int64_t ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
    int64_t q = ax / ay;
    return (x >= 0) == (y > 0) ? q : -q;
}

/* _floor_div(HostInt(), a, m) */
int64_t hfloor_div(int64_t a, int64_t m)
{
    int64_t neg = 0 - hdiv((m - 1) - a, m);
    return a < 0 ? neg : hdiv(a, m);
}

/* kv_tile_range(HostInt(), band, q_first, n_rows, k_n0=, uniform=False) */
void host_kv_tile_range(const rocke_attn_bwd_host_band_t& band,
                        int64_t q_first,
                        int64_t rows,
                        int k_n0,
                        int64_t* kt_start,
                        int64_t* n_kt)
{
    if(k_n0 < 1)
        throw ckc::ValueError("tile sizes must be positive");
    int64_t cn = k_n0;
    int64_t q_last = std::min(q_first + rows - 1, band.sq - 1);
    int64_t c_first = band.has_off ? q_first + band.off : q_first;
    int64_t c_last = band.has_off ? q_last + band.off : q_last;
    int64_t k_lo = 0;
    if(band.has_left)
        k_lo = std::max<int64_t>(c_first - band.left, 0);
    int64_t k_hi = band.sk_minus1;
    if(band.has_right)
        k_hi = std::min(c_last + band.right, band.sk_minus1);
    bool dead = (q_last < q_first) | (k_hi < k_lo);
    k_hi = dead ? -1 : k_hi;
    *kt_start = hdiv(k_lo, cn);
    int64_t kt_stop = hdiv(std::max<int64_t>(k_hi, -1) + cn, cn);
    *n_kt = std::max<int64_t>(kt_stop - *kt_start, 0);
}

/* interior_kv_range(HostInt(), band, q_first, n_rows, kt_start, n_kt,
 * k_n0=, masked=, uniform=False) */
void host_interior_kv_range(const rocke_attn_bwd_host_band_t& band,
                            int64_t q_first,
                            int64_t rows,
                            int64_t kt_start,
                            int64_t n_kt,
                            int k_n0,
                            bool masked,
                            int64_t* i0,
                            int64_t* i1)
{
    int64_t kt_end = kt_start + n_kt;
    int64_t q_last = q_first + rows - 1;
    int64_t lo = kt_start;
    int64_t hi = std::min(kt_end, hdiv(band.sk, k_n0));
    int64_t off = band.has_off ? band.off : 0;
    if(masked && band.has_right)
    {
        int64_t top = q_first + off + band.right;
        int64_t last = hfloor_div(top - (k_n0 - 1), k_n0);
        hi = std::min(hi, last + 1);
    }
    if(masked && band.has_left)
    {
        int64_t need = q_last + off - band.left;
        int64_t first = 0 - hfloor_div(0 - need, k_n0);
        lo = std::max(lo, first);
    }
    hi = q_last < band.sq ? hi : lo;
    *i0 = std::min(lo, kt_end);
    *i1 = std::max(std::min(hi, kt_end), *i0);
}

void dense_dq_static_classes(long long s_q,
                             long long s_kv,
                             int rows,
                             int k_n0,
                             int left,
                             int right,
                             bool bottom_right,
                             bool masked,
                             bool* any_edge,
                             bool* any_interior)
{
    int64_t off = bottom_right ? s_kv - s_q : 0;
    rocke_attn_bwd_host_band_t band;
    char msg[160] = "";
    rocke_status_t st = rocke_attn_bwd_host_band_init(
        &band, s_q, s_kv, off, masked ? left : -1, masked ? right : -1, msg, sizeof(msg));
    if(st != ROCKE_OK)
        ckc::raise_status(st, msg);
    bool edge = false, interior = false;
    int64_t n_x = rows > 0 ? (s_q + rows - 1) / rows : 0;
    if(rows <= 0)
        throw ckc::ValueError("integer division or modulo by zero");
    for(int64_t x = 0; x < n_x; ++x)
    {
        int64_t q0 = x * rows;
        int64_t kt0, n_kt, i0, i1;
        host_kv_tile_range(band, q0, rows, k_n0, &kt0, &n_kt);
        host_interior_kv_range(band, q0, rows, kt0, n_kt, k_n0, masked, &i0, &i1);
        edge = edge || (i0 - kt0) + (kt0 + n_kt - i1) > 0;
        interior = interior || i1 > i0;
        if(edge && interior)
            break;
    }
    *any_edge = edge;
    *any_interior = interior;
}

/* ---------------------------------------------------------------- emitter */

/* inline_asm("", constraints, [v], result_type=v.type, sideeffect=, hint) */
V empty_asm(B* b, V v, const char* constraints, bool sideeffect, const char* hint)
{
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "template", "");
    rocke_attr_set_str(b, &attrs, "constraints", constraints);
    rocke_attr_set_bool(b, &attrs, "sideeffect", sideeffect);
    rocke_attr_set_bool(b, &attrs, "convergent", false);
    const rocke_type_t* ty = v->type;
    rocke_op_t* op
        = rocke_b_op(b, ROCKE_OP_TILE_INLINE_ASM, &v, 1, &ty, 1, &attrs, NULL, 0, hint, NULL);
    need(b, op != NULL && op->num_results == 1);
    return op->results[0];
}

/* _pinned(b, slab, hint): a buffer slab (no ptr) is returned as it is. */
rocke_attn_bwd_row_slab_t pinned(B* b, const rocke_attn_bwd_row_slab_t& slab, const char* hint)
{
    if(slab.rsrc != NULL)
        return slab;
    V ptr = empty_asm(b, slab.ptr, "=s,0", false, hint);
    rocke_attn_bwd_row_slab_t out = slab;
    out.ptr = ptr;
    return out;
}

/* Everything emit_dense_bwd_dq keeps alive while it emits. */
struct Dq
{
    B* b;
    DSpec spec;
    DqSpec s;
    std::string arch;
    int seq_guards = 0;
    rocke_attn_bwd_dq_geometry_t geom;
    rocke_attn_bwd_dq_lds_plan_t plan;
    rocke_attn_bwd_seq_info_t seq_q, seq_kv;
    rocke_attn_bwd_dense_fixed_addr_t q_addr, do_addr, k_addr, v_addr;
    rocke_attn_bwd_strided_tensor_addr_t dq_addr;
    rocke_attn_bwd_workspace_addr_t ws_lse2, ws_dsum;
    rocke_attn_bwd_stats_view_t stats;
    rocke_attn_bwd_mask_policy_t mask;
    rocke_attn_bwd_dq_lds_views_t lds;
    rocke_attn_bwd_dq_step_t step;
    rocke_attn_bwd_packed_rows_t rows;
};

rocke_kernel_def_t* emit_dense_dq(Dq& m)
{
    B* b = m.b;
    const DqSpec& s = m.s;
    const DSpec& spec = m.spec;
    const rocke_attn_bwd_arch_facts_t& facts = arch_facts(m.arch);
    need(b, rocke_attn_bwd_dq_geometry_from_spec(b, &m.s, m.arch.c_str(), &m.geom));
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_dq_lds_plan(&s, m.arch.c_str(), &m.plan, &e) != ROCKE_OK)
        raise_err(e);
    const rocke_type_t* io = rocke_attn_bwd_ir_type(b, spec.dtype);
    need(b, io != NULL);
    int km = s.block_m, kn = s.block_n;
    (void)km;
    long long s_q = spec.seqlen_q, s_kv = spec.seqlen_kv;
    int h_q = spec.num_query_heads;
    int G = group_of(spec);
    bool group = eq(s.head_loop, "group");
    int rph = dq_rows_per_head(s, G); /* rows of one head in the CTA */
    long long gx_n = (s_q + rph - 1) / rph;

    /* _declare(b, io) */
    std::map<std::string, V> a;
    {
        const char* names[ROCKE_ATTN_BWD_DENSE_DQ_PARAMS];
        const char* formats[ROCKE_ATTN_BWD_DENSE_DQ_PARAMS];
        int n = rocke_attn_bwd_dense_dq_params(names, formats);
        static const char* const readonly[] = {"Q", "K", "V", "dO", "WS_LSE2", "WS_DSUM"};
        for(int i = 0; i < n; ++i)
        {
            const char* name = names[i];
            V v;
            if(eq(formats[i], "Q"))
            {
                const rocke_type_t* pointee = strncmp(name, "WS_", 3) == 0 ? rocke_f32() : io;
                bool ro = false;
                for(const char* x : readonly)
                    ro = ro || eq(name, x);
                rocke_param_opts_t opts;
                memset(&opts, 0, sizeof(opts));
                opts.readonly = true;
                opts.readonly_set = true;
                v = rocke_b_param(b, name, rocke_ptr_type(b, pointee, "global"), ro ? &opts : NULL);
            }
            else if(eq(formats[i], "f"))
                v = rocke_b_param(b, name, rocke_f32(), NULL);
            else
                v = rocke_b_param(b, name, rocke_i32(), NULL);
            a[name] = needv(b, v);
        }
    }
    auto A = [&](const char* k) -> V {
        auto it = a.find(k);
        if(it == a.end())
            throw ckc::KeyError("'" + std::string(k) + "'");
        return it->second;
    };
    /* Multipliers that are 1.0 under the compiled placement are not applied. */
    unsigned unit = 0;
    if(rocke_attn_bwd_dq_static_unit_mults(s.static_scale, &unit, &e) != ROCKE_OK)
        raise_err(e);
    if(unit & ROCKE_ATTN_BWD_UNIT_DS_MULT)
        a["ds_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
        a["dq_mult"] = NULL;
    int threads = rocke_attn_bwd_dq_geometry_threads(&m.geom);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", std::max(threads, MIN_WG_ATTR));
    if(s.agpr_alloc.kind == ROCKE_ATTN_BWD_AGPR_PAIR)
    {
        int64_t pair[2] = {s.agpr_alloc.a, s.agpr_alloc.b};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "agpr_alloc", pair, 2);
    }
    if(s.waves_per_eu != NONE)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", s.waves_per_eu);
    if(s.scheduler_strategy != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", s.scheduler_strategy);
    need(b, true);

    V tid, lane, wave;
    rocke_attn_bwd_lane_and_wave(b, facts.wave_size, NULL, &tid, &lane, &wave);
    need(b, true);
    V zero = ci(b, 0);
    V one = ci(b, 1);
    V batch = spec.batch != NONE ? ci(b, spec.batch) : A("batch");
    V x = needv(b, rocke_b_block_id_x(b));
    V y = needv(b, rocke_b_block_id_y(b));
    V z = needv(b, rocke_b_block_id_z(b));
    if(s.xcd_remap)
    {
        /* per batch plane, as in the general dQ kernel (identity at xcd_n = 0) */
        V gx = ci(b, gx_n);
        V gy = ci(b, group ? h_q / G : h_q);
        V gxy = rocke_b_mul(b, gx, y);
        V wgid = rocke_b_add(b, x, gxy);
        V num = rocke_b_mul(b, gx, gy);
        V plane = needv(b, rocke_attn_bwd_xcd_remap_id(b, wgid, num, A("xcd_n"), A("xcd_chunk")));
        x = needv(b, rocke_b_mod(b, plane, gx));
        y = needv(b, rocke_b_div(b, plane, gx));
    }
    rocke_attn_bwd_decode_args_t qa = rocke_attn_bwd_decode_args_default();
    qa.s_max = rocke_attn_bwd_iv_int(s_q);
    m.seq_q = rocke_attn_bwd_decode_batched(b, &m.seq_guards, z, &qa);
    need(b, true);
    rocke_attn_bwd_decode_args_t ka = rocke_attn_bwd_decode_args_default();
    ka.s_max = rocke_attn_bwd_iv_int(s_kv);
    m.seq_kv = rocke_attn_bwd_decode_batched(b, &m.seq_guards, z, &ka);
    need(b, true);
    V h;
    if(group)
    {
        V cg = ci(b, G);
        h = rocke_b_mul(b, y, cg);
    }
    else
        h = y; /* the (first) query head of the CTA */
    V cg = ci(b, G);
    V hk = rocke_b_div(b, h, cg);
    V crph = ci(b, rph);
    V q0 = needv(b, rocke_b_mul(b, x, crph));
    if(group)
    {
        V ng = ci(b, G);
        V rp = ci(b, rph);
        m.rows.head0 = h;
        m.rows.n_heads = ng;
        m.rows.rows_per_head = rp;
        m.rows.row0 = q0;
    }
    need(b, true);
    const rocke_attn_bwd_packed_rows_t* rows = group ? &m.rows : NULL;
    /* a q block can pass the sequence end only when rph does not divide s_q */
    bool q_tail = s_q % rph != 0;

    auto tensor = [&](rocke_attn_bwd_dense_fixed_addr_t* out,
                      const char* name,
                      const char* role,
                      const rocke_attn_bwd_seq_info_t* seq,
                      long long seqlen,
                      bool global_view,
                      bool ragged) {
        long long st[3];
        strides_of(spec, role, st);
        need(b,
             rocke_attn_bwd_dense_fixed_addr_init(out,
                                                  b,
                                                  A(name),
                                                  io,
                                                  st,
                                                  seq,
                                                  seqlen,
                                                  spec.head_size,
                                                  spec.oob,
                                                  global_view,
                                                  ragged));
    };
    /* a packed CTA reads one head per row: Q / dO through the global pointer */
    tensor(&m.q_addr, "Q", "q", &m.seq_q, s_q, group, q_tail);
    tensor(&m.do_addr, "dO", "do", &m.seq_q, s_q, group, q_tail);
    tensor(&m.k_addr, "K", "k", &m.seq_kv, s_kv, false, true);
    tensor(&m.v_addr, "V", "v", &m.seq_kv, s_kv, false, true);
    {
        long long st[3];
        strides_of(spec, "dq", st);
        IV ivs[3] = {rocke_attn_bwd_iv_int(st[0]),
                     rocke_attn_bwd_iv_int(st[1]),
                     rocke_attn_bwd_iv_int(st[2])};
        need(b,
             rocke_attn_bwd_strided_tensor_addr_init(
                 &m.dq_addr, b, A("dQ"), io, ivs, &m.seq_q, 8, 1));
    }
    V csq = ci(b, s_q);
    V rows_q = needv(b, rocke_b_mul(b, batch, csq));
    bool token_major = eq(spec.ws_layout, "token_major");
    IV nh;
    if(token_major)
        nh = ivv(ci(b, h_q));
    rocke_attn_bwd_ws_layout_t layout
        = token_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
    auto ws = [&](const char* name, rocke_attn_bwd_workspace_addr_t* out) {
        need(b,
             rocke_attn_bwd_workspace_addr_init(out,
                                                b,
                                                A(name),
                                                ivv(rows_q),
                                                1,
                                                m.seq_q.ws_row0,
                                                layout,
                                                token_major ? &nh : NULL));
    };
    ws("WS_LSE2", &m.ws_lse2);
    ws("WS_DSUM", &m.ws_dsum);
    need(b, rocke_attn_bwd_stats_view_init(&m.stats, b, &m.ws_lse2, &m.ws_dsum, NULL, "clamp", 0));
    rocke_attn_bwd_dense_spec_t p = problem_spec(spec);
    int left, right;
    bool br;
    rocke_attn_bwd_dense_spec_band(&p, &left, &right, &br);
    bool masked = rocke_attn_bwd_dense_spec_masked(&p);
    bool st_edge, st_interior;
    if(masked)
        need(b,
             rocke_attn_bwd_diagonal_band_init(
                 &m.mask, &st_edge, &st_interior, b, s_q, s_kv, left, right, br, s.block_m, kn));
    else
        need(b,
             rocke_attn_bwd_dense_no_mask_init(
                 &m.mask, &st_edge, &st_interior, b, s_q, s_kv, s.block_m, kn));
    /* _lds_views(b, plan, io) */
    int shape[1] = {m.plan.total_bytes / 2};
    V smem = needv(b, rocke_b_smem_alloc(b, io, shape, 1, "bwd_dq_lds"));
    auto view = [&](const char* name, rocke_attn_bwd_lds_view_t* out) {
        const rocke_attn_bwd_dq_lds_buffer_t* bf = NULL;
        for(int i = 0; i < ROCKE_ATTN_BWD_DQ_LDS_BUFFERS; ++i)
            if(eq(m.plan.buffers[i].name, name))
                bf = &m.plan.buffers[i];
        if(bf == NULL)
            throw ckc::KeyError(py_repr(name));
        need(b,
             rocke_attn_bwd_lds_view_init(
                 out, b, smem, bf->offset / 2, bf->pitch, bf->rows, 1, NULL));
    };
    view("q", &m.lds.q);
    view("do", &m.lds.do_);
    view("k", &m.lds.k);
    view("v", &m.lds.v);
    view("out", &m.lds.out);
    need(b,
         rocke_attn_bwd_dq_step_init(&m.step,
                                     b,
                                     &m.geom,
                                     &m.mask,
                                     &m.stats,
                                     &m.lds,
                                     tid,
                                     lane,
                                     wave,
                                     A("scale_log2"),
                                     A("ds_mult"),
                                     s.stats_prescale,
                                     true));

    /* Stage 0 (only a CTA with a row < len_q reads Q / dO). */
    V live_ = needv(b, rocke_b_cmp_lt(b, q0, m.seq_q.length)); /* CTA-uniform */
    rocke_attn_bwd_row_slab_t q_slab, do_slab;
    if(!group)
    {
        q_slab = rocke_attn_bwd_dense_fixed_addr_slab(&m.q_addr, ivv(h), ivv(q0), true);
        need(b, true);
        do_slab = rocke_attn_bwd_dense_fixed_addr_slab(&m.do_addr, ivv(h), ivv(q0), true);
        need(b, true);
    }
    /* a packed CTA's Q / dO addresses are the global views */
    need(b,
         rocke_attn_bwd_dq_step_stage_q(&m.step,
                                        b,
                                        live_,
                                        h,
                                        q0,
                                        group ? NULL : &q_slab,
                                        group ? NULL : &do_slab,
                                        m.q_addr.has_strided ? &m.q_addr.strided : NULL,
                                        m.do_addr.has_strided ? &m.do_addr.strided : NULL,
                                        rows,
                                        8));

    /* Key tiles of the rows; the edge / interior runs proved empty at build
     * time are not emitted (same key order, so the same dQ bits). */
    V kt_start, n_kt;
    need(b,
         rocke_attn_bwd_kv_tile_range(
             b, &m.mask.band, q0, rocke_attn_bwd_iv_int(rph), kn, true, &kt_start, &n_kt));
    V kt_end = needv(b, rocke_b_add(b, kt_start, n_kt));
    struct Range
    {
        V s0, e;
        bool edge;
    };
    std::vector<Range> ranges;
    if(s.edge_tiles)
    {
        bool any_edge, any_interior;
        dense_dq_static_classes(
            s_q, s_kv, rph, kn, left, right, br, masked, &any_edge, &any_interior);
        if(!any_edge)
            ranges = {{kt_start, kt_end, false}};
        else if(!any_interior)
            ranges = {{kt_start, kt_end, true}};
        else
        {
            V i0, i1;
            need(b,
                 rocke_attn_bwd_interior_kv_range(b,
                                                  &m.mask.band,
                                                  q0,
                                                  rocke_attn_bwd_iv_int(rph),
                                                  kt_start,
                                                  n_kt,
                                                  kn,
                                                  m.mask.masked,
                                                  true,
                                                  &i0,
                                                  &i1));
            ranges = {{kt_start, i0, true}, {i0, i1, false}, {i1, kt_end, true}};
        }
    }
    else
        ranges = {{kt_start, kt_end, true}};
    std::vector<V> trips;
    for(const Range& r : ranges)
    {
        V d = rocke_b_sub(b, r.e, r.s0);
        trips.push_back(needv(b, rocke_b_readfirstlane(b, d)));
    }

    rocke_attn_bwd_row_slab_t k_raw
        = rocke_attn_bwd_dense_fixed_addr_slab(&m.k_addr, ivv(hk), rocke_attn_bwd_iv_int(0), true);
    need(b, true);
    rocke_attn_bwd_row_slab_t k_slab = pinned(b, k_raw, "kptr");
    rocke_attn_bwd_row_slab_t v_raw
        = rocke_attn_bwd_dense_fixed_addr_slab(&m.v_addr, ivv(hk), rocke_attn_bwd_iv_int(0), true);
    need(b, true);
    rocke_attn_bwd_row_slab_t v_slab = pinned(b, v_raw, "vptr");
    /* dQ at head h, row 0 (rows are addressed as given; the store predicates) */
    rocke_attn_bwd_row_slab_t dq_raw = rocke_attn_bwd_strided_tensor_addr_slab(
        &m.dq_addr, ivv(h), rocke_attn_bwd_iv_int(0), false);
    need(b, true);
    rocke_attn_bwd_row_slab_t dq_slab = pinned(b, dq_raw, "dqptr");
    V head_step = NULL;
    if(group)
    {
        /* the head stride in bytes (fp16 / bf16 elements) */
        long long st[3];
        strides_of(spec, "dq", st);
        head_step = needv(b, rocke_b_const_i64(b, st[1] * ELEM_BYTES));
    }
    if(s.kv_prefetch)
    {
        V ckn = ci(b, kn);
        V k0 = rocke_b_mul(b, kt_start, ckn);
        V any = needv(b, rocke_b_cmp_lt(b, zero, n_kt));
        need(b, rocke_attn_bwd_dq_step_prologue(&m.step, b, &k_slab, &v_slab, k0, any));
    }
    const rocke_attn_bwd_block_gemm_t* g4t = &m.geom.g4t;
    int fm = rocke_attn_bwd_block_gemm_frags_m(g4t), fn = rocke_attn_bwd_block_gemm_frags_n(g4t);
    std::vector<V> dq((size_t)fm * fn);
    need(b, rocke_attn_bwd_dq_step_zero_dq(&m.step, b, dq.data()));
    static const char* const names3[3] = {"k", "ki_", "ke_"};
    for(size_t ri = 0; ri < ranges.size(); ++ri)
    {
        const Range& r = ranges[ri];
        std::string name = ranges.size() == 3 ? names3[ri] : "k";
        std::vector<std::string> arg_names;
        std::vector<rocke_iter_arg_t> ia;
        for(size_t i = 0; i < dq.size(); ++i)
            arg_names.push_back("dq_" + name + std::to_string(i));
        for(size_t i = 0; i < dq.size(); ++i)
        {
            rocke_iter_arg_t it;
            it.name = arg_names[i].c_str();
            it.init = dq[i];
            ia.push_back(it);
        }
        std::string iv = "dq_" + name + "it";
        rocke_for_t loop = rocke_b_scf_for_iter(
            b, zero, trips[ri], one, ia.data(), (int)ia.size(), iv.c_str(), false, false);
        if(loop.op == NULL)
            raise_builder(b);
        rocke_b_region_enter(b, loop.body);
        {
            V it = loop.iv;
            V sum = rocke_b_add(b, r.s0, it);
            V ckn = ci(b, kn);
            V k0 = needv(b, rocke_b_mul(b, sum, ckn));
            std::vector<V> cur(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
            std::vector<V> out(cur.size());
            need(b,
                 rocke_attn_bwd_dq_step_run_kv_step(&m.step,
                                                    b,
                                                    cur.data(),
                                                    &k_slab,
                                                    &v_slab,
                                                    k0,
                                                    r.edge,
                                                    s.kv_prefetch,
                                                    8,
                                                    out.data()));
            rocke_b_scf_yield(b, out.data(), (int)out.size());
            need(b, true);
        }
        rocke_b_region_leave(b);
        need(b, true);
        for(int i = 0; i < loop.op->num_results; ++i)
            dq[(size_t)i] = loop.op->results[i];
    }

    V sqm = ci(b, s_q);
    need(b,
         rocke_attn_bwd_dq_step_store_dq(&m.step,
                                         b,
                                         dq.data(),
                                         A("dq_mult"),
                                         &dq_slab,
                                         head_step,
                                         h,
                                         q0,
                                         rows,
                                         m.seq_q.length,
                                         sqm,
                                         ROCKE_ATTN_BWD_SEQ_BATCHED));
    rocke_b_ret(b);
    need(b, true);
    return rocke_ir_builder_kernel(b);
}

template <class Fn>
rocke_status_t boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
{
    try
    {
        fn();
        return store_err(err, ROCKE_OK, "");
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
    catch(const std::bad_alloc& e)
    {
        return store_err(err, ROCKE_ERR_OOM, e.what());
    }
}

} // namespace
} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_attn_bwd_dense_dq_spec_t rocke_attn_bwd_dense_dq_spec_default(void)
{
    rocke_attn_bwd_dense_dq_spec_t s;
    memset(&s, 0, sizeof(s));
    s.batch = NONE;
    s.causal = true;
    s.causal_bottom_right = false;
    s.dtype = "bf16";
    s.layout = "bshd";
    s.sliding_window = 0;
    s.window_left = NONE;
    s.window_right = NONE;
    s.oob = "buffer_oob";
    s.ws_layout = "head_major";
    s.has_dq = false;
    s.dq = rocke_attn_bwd_dq_spec_default();
    s.name = "rocke_attn_dense_bwd";
    return s;
}

rocke_status_t rocke_attn_bwd_dense_dq_spec_init(rocke_attn_bwd_dense_dq_spec_t* spec,
                                                 rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { dense_dq_init(*spec); });
}

rocke_status_t rocke_attn_bwd_dense_dq_spec_kernel_name(const rocke_attn_bwd_dense_dq_spec_t* spec,
                                                        char* out,
                                                        size_t out_cap,
                                                        rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        std::string n = dense_dq_kernel_name(*spec);
        if(out == NULL || n.size() >= out_cap)
            throw ckc::ValueError("kernel name does not fit the caller's buffer");
        memcpy(out, n.c_str(), n.size() + 1);
    });
}

rocke_status_t rocke_attn_bwd_dense_dq_validate(const rocke_attn_bwd_dense_dq_spec_t* spec,
                                                const char* arch,
                                                rocke_attn_bwd_dq_spec_t* out,
                                                rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dense_dq_validate(*spec, arch ? arch : ""); });
}

rocke_status_t rocke_attn_bwd_dense_dq_static_classes(long long s_q,
                                                      long long s_kv,
                                                      int rows,
                                                      int k_n0,
                                                      int left,
                                                      int right,
                                                      bool bottom_right,
                                                      bool masked,
                                                      bool* any_edge,
                                                      bool* any_interior,
                                                      rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        dense_dq_static_classes(
            s_q, s_kv, rows, k_n0, left, right, bottom_right, masked, any_edge, any_interior);
    });
}

int rocke_attn_bwd_dense_dq_params(const char* names[ROCKE_ATTN_BWD_DENSE_DQ_PARAMS],
                                   const char* formats[ROCKE_ATTN_BWD_DENSE_DQ_PARAMS])
{
    static const char* const ptrs[] = {"Q", "K", "V", "dO", "dQ", "WS_LSE2", "WS_DSUM"};
    static const char* const f32s[] = {"scale_log2", "ds_mult", "dq_mult"};
    static const char* const i32s[] = {"batch", "xcd_n", "xcd_chunk"};
    int n = 0;
    for(const char* p : ptrs)
    {
        names[n] = p;
        formats[n++] = "Q";
    }
    for(const char* f : f32s)
    {
        names[n] = f;
        formats[n++] = "f";
    }
    for(const char* i : i32s)
    {
        names[n] = i;
        formats[n++] = "i";
    }
    return n;
}

rocke_kernel_def_t* rocke_build_attn_dense_bwd_dq_new(rocke_ir_builder_t* b,
                                                      const rocke_attn_bwd_dense_dq_spec_t* spec,
                                                      const char* arch)
{
    if(b == NULL)
        return NULL;
    /* the validator raises before the Python builder exists; the C builder
     * must exist to carry the error, so a refused spec gets a placeholder
     * name. */
    std::unique_ptr<Dq> m(new Dq());
    std::string name;
    rocke_status_t code = ROCKE_OK;
    std::string msg;
    try
    {
        if(spec == NULL)
            throw ckc::ValueError("spec must not be NULL");
        m->arch = arch ? arch : "";
        m->spec = *spec;
        m->s = dense_dq_validate(*spec, m->arch);
        name = dense_dq_kernel_name(*spec);
    }
    catch(const ckc::Error& e)
    {
        code = e.code();
        msg = e.what();
    }
    catch(const std::bad_alloc& e)
    {
        code = ROCKE_ERR_OOM;
        msg = e.what();
    }
    if(rocke_ir_builder_init(b, code == ROCKE_OK ? name.c_str() : "rocke_attn_dense_bwd_dq_invalid")
       != ROCKE_OK)
        return NULL;
    if(code != ROCKE_OK)
    {
        rocke_i_set_err_msg(b, code, msg.c_str());
        return NULL;
    }
    m->b = b;
    m->seq_guards = 0;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return emit_dense_dq(*m); });
}

} /* extern "C" */
