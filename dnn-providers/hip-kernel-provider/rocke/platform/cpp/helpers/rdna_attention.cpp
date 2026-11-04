// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99-API port of rocke/helpers/rdna_attention.py -- the multi-wave WMMA
 * flash-attention forward body for RDNA (gfx11 / gfx12). See the header for
 * the symbol map and the byte-fidelity contract.
 *
 * C/C++ argument-evaluation order is unspecified (Python's is left to
 * right), so every nested rocke_b_* expression of the Python source is
 * hoisted into ordered local temporaries that reproduce the Python
 * value-creation order exactly. Python closures map onto C++ lambdas of the
 * same name.
 */
#include "rocke/helper_rocke.helpers.rdna_attention.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/helper_rocke.helpers.attention.h" /* rocke_decode_fp8e4m3fn_to_f32 */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_spec_set_reason */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

namespace
{

constexpr int k_wave = 32;
/* Lazy rescale (log2 domain): skip the O/l rescale when every lane's
 * tile_max - m <= threshold. exp2(8) = 256 bounds P before its 16-bit cast. */
constexpr double k_lazy_rescale_threshold = 8.0;
/* A byte offset >= 2**31 is out of range of every rsrc (num_records < 2**31). */
constexpr int64_t k_oob = -0x80000000LL;
constexpr double k_ln2 = 0.6931471805599453;
constexpr double k_log2e = 1.4426950408889634;

using Val = rocke_value_t*;
using Vals = std::vector<Val>;

bool is_bf16(const rocke_rdna_attn_config_t* cfg)
{
    return strcmp(cfg->dtype, "bf16") == 0;
}

int buffers(const rocke_rdna_attn_config_t* cfg)
{
    return rocke_rdna_attn_pair(cfg) ? 1 : 2;
}

void tile_elems(const rocke_rdna_attn_config_t* cfg, int* k_tile, int* v_tile)
{
    *k_tile = cfg->block_n * (cfg->head_dim + cfg->k_pad);
    *v_tile = cfg->head_dim_v * (cfg->block_n + cfg->v_pad);
}

int xs_floats(const rocke_rdna_attn_config_t* cfg)
{
    return rocke_rdna_attn_pair(cfg) ? rocke_rdna_attn_waves(cfg) * k_wave * (cfg->block_n / 16) * 8
                                     : 0;
}

/* _v_group: (rows per V unit G, units per thread), minimising
 * units * (G + 8); the first G of (4, 2, 1) wins ties. */
void v_group(const rocke_rdna_attn_config_t* cfg, int* g_out, int* per_out)
{
    const int nt = rocke_rdna_attn_threads(cfg);
    const int gs[] = {4, 2, 1};
    int best_cost = -1, best_g = 0, best_per = 0;
    for(int g : gs)
    {
        if(cfg->block_n % g)
            continue;
        const int units = (cfg->block_n / g) * (cfg->head_dim_v / 8);
        const int per = (units + nt - 1) / nt;
        const int cost = per * (g + 8);
        if(best_cost < 0 || cost < best_cost)
        {
            best_cost = cost;
            best_g = g;
            best_per = per;
        }
    }
    *g_out = best_g;
    *per_out = best_per;
}

bool partial_rounds(const rocke_rdna_attn_config_t* cfg)
{
    int g, per;
    v_group(cfg, &g, &per);
    const int nt = rocke_rdna_attn_threads(cfg);
    const int k_chunks = cfg->block_n * (cfg->head_dim / 8);
    const int v_units = (cfg->block_n / g) * (cfg->head_dim_v / 8);
    return (k_chunks % nt) != 0 || (v_units % nt) != 0;
}

int lds_elems(const rocke_rdna_attn_config_t* cfg)
{
    int k_tile, v_tile;
    tile_elems(cfg, &k_tile, &v_tile);
    const int dummy = partial_rounds(cfg) ? rocke_rdna_attn_threads(cfg) * 8 : 0;
    return buffers(cfg) * (k_tile + v_tile) + dummy;
}

} // namespace

bool rocke_rdna_attn_pair(const rocke_rdna_attn_config_t* cfg)
{
    if(cfg->wave_pair >= 0)
        return cfg->wave_pair != 0;
    return cfg->head_dim > 128 || cfg->head_dim_v > 128;
}

int rocke_rdna_attn_waves(const rocke_rdna_attn_config_t* cfg)
{
    return (cfg->block_m / 16) * (rocke_rdna_attn_pair(cfg) ? 2 : 1);
}

int rocke_rdna_attn_threads(const rocke_rdna_attn_config_t* cfg)
{
    return k_wave * rocke_rdna_attn_waves(cfg);
}

bool rocke_rdna_attn_config_validate(const rocke_rdna_attn_config_t* cfg,
                                     char* reason,
                                     size_t reason_cap)
{
    char msg[160];
    if(cfg == NULL || cfg->dtype == NULL
       || (strcmp(cfg->dtype, "fp16") != 0 && strcmp(cfg->dtype, "bf16") != 0))
    {
        snprintf(msg,
                 sizeof(msg),
                 "dtype must be fp16/bf16, got '%s'",
                 cfg != NULL && cfg->dtype != NULL ? cfg->dtype : "None");
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    const struct
    {
        const char* name;
        int d;
    } widths[] = {{"head_dim", cfg->head_dim}, {"head_dim_v", cfg->head_dim_v}};
    for(const auto& w : widths)
    {
        if(w.d <= 0 || w.d % 16 || w.d > 256)
        {
            snprintf(
                msg, sizeof(msg), "%s must be a multiple of 16 in 16..256, got %d", w.name, w.d);
            rocke_spec_set_reason(reason, reason_cap, msg);
            return false;
        }
    }
    const bool pair = rocke_rdna_attn_pair(cfg);
    if(pair && (cfg->head_dim % 32 || cfg->head_dim_v % 32))
    {
        rocke_spec_set_reason(
            reason, reason_cap, "the wave-pair split needs head widths that are multiples of 32");
        return false;
    }
    const int waves = rocke_rdna_attn_waves(cfg);
    if(cfg->block_m % 16 || !(1 <= waves && waves <= 16))
    {
        snprintf(msg, sizeof(msg), "block_m %d gives %d waves (1..16)", cfg->block_m, waves);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(cfg->block_n % 16 || cfg->block_n < 16)
    {
        snprintf(msg, sizeof(msg), "block_n must be a multiple of 16, got %d", cfg->block_n);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(cfg->k_pad % 8 || cfg->v_pad % 8)
    {
        rocke_spec_set_reason(reason, reason_cap, "LDS paddings must keep 16-byte row alignment");
        return false;
    }
    if(cfg->bias != NULL && strcmp(cfg->bias, "f32") != 0 && strcmp(cfg->bias, "q") != 0)
    {
        snprintf(msg, sizeof(msg), "bias must be None, 'f32' or 'q', got '%s'", cfg->bias);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!pair && (cfg->head_dim_v > 128 || cfg->head_dim > (cfg->gfx12 ? 256 : 128)))
    {
        rocke_spec_set_reason(
            reason, reason_cap, "unsplit waves hold Q <= 128 (gfx11) / 256 (gfx12) and V <= 128");
        return false;
    }
    const int ps = cfg->page_block_size;
    if(ps < 0 || (ps & (ps - 1)) != 0)
    {
        snprintf(msg, sizeof(msg), "page_block_size must be 0 or a power of two, got %d", ps);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    return true;
}

bool rocke_rdna_attn_log_domain(const rocke_rdna_attn_config_t* cfg)
{
    return cfg->bias != NULL || cfg->softcap || cfg->alibi || cfg->qq_bias;
}

int rocke_rdna_attention_lds_bytes(const rocke_rdna_attn_config_t* cfg)
{
    return lds_elems(cfg) * 2 + xs_floats(cfg) * 4;
}

rocke_status_t rocke_rdna_attention_fwd_body(rocke_ir_builder_t* b,
                                             const rocke_rdna_attn_config_t* cfg,
                                             const rocke_rdna_attn_params_t* p)
{
    if(b == NULL || cfg == NULL || p == NULL)
        return ROCKE_ERR_VALUE;
    char reason[256];
    if(!rocke_rdna_attn_config_validate(cfg, reason, sizeof(reason)))
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
        return ROCKE_ERR_VALUE;
    }
    const bool bf16 = is_bf16(cfg);
    const rocke_type_t* elem = bf16 ? rocke_bf16() : rocke_f16();
    const bool gfx12 = cfg->gfx12;
    const bool pair = rocke_rdna_attn_pair(cfg);
    const int BN = cfg->block_n, DQ = cfg->head_dim, DV = cfg->head_dim_v;
    const int NT = rocke_rdna_attn_threads(cfg);
    const int n_kv = BN / 16;
    const int DQW = pair ? DQ / 2 : DQ;
    const int DVW = pair ? DV / 2 : DV;
    const int n_dq = DQW / 16, n_dv = DVW / 16;
    const std::string op_id_s
        = std::string(gfx12 ? "wmma_gfx12_f32_16x16x16_" : "wmma_f32_16x16x16_")
          + (bf16 ? "bf16" : "f16");
    const char* op_id = op_id_s.c_str();
    const int KS = DQ + cfg->k_pad;
    const int VS = BN + cfg->v_pad;
    int k_tile, v_tile;
    tile_elems(cfg, &k_tile, &v_tile);
    const int buf_elems = k_tile + v_tile;
    const int n_buffers = buffers(cfg);
    const bool causal = cfg->causal;
    const bool has_bias = cfg->bias != NULL;
    const bool bias_f32 = has_bias && strcmp(cfg->bias, "f32") == 0;
    const bool bias_q = has_bias && !bias_f32;
    if(causal && (p->diag_offset == NULL || p->window_left == NULL || p->window_right == NULL))
    {
        (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "the causal class needs diag_offset and both window bounds");
        return ROCKE_ERR_VALUE;
    }
    if(has_bias && (p->bias_rsrc == NULL || p->stride_bias_q == NULL))
    {
        (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "an additive bias needs its resource and query stride");
        return ROCKE_ERR_VALUE;
    }
    if(bias_q && (p->bias_lead == NULL || p->bias_bytes == NULL))
    {
        (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "a Q-dtype bias needs its lead element and byte size");
        return ROCKE_ERR_VALUE;
    }
    const bool paged = cfg->page_block_size > 0;
    {
        const struct
        {
            const char* name;
            bool need;
            int count;
            rocke_value_t* vals[5];
        } groups[] = {
            {"softcap", cfg->softcap, 1, {p->softcap}},
            {"sinks", cfg->sinks, 1, {p->sink_log2}},
            {"alibi", cfg->alibi, 2, {p->alibi_slope_log2, p->context}},
            {"qq_bias",
             cfg->qq_bias,
             5,
             {p->qq_rsrc, p->qq_rows, p->qq_cols, p->qq_stride, p->context}},
            {"kv_fp8", cfg->kv_fp8, 2, {p->k_scale, p->v_scale}},
            {"paged",
             paged,
             5,
             {p->block_table, p->table_row, p->stride_k_block, p->stride_v_block, p->num_blocks}},
            {"dense K/V", !paged, 2, {p->k_bytes, p->v_bytes}},
        };
        std::string missing;
        for(const auto& g : groups)
        {
            bool any_null = false;
            for(int i = 0; i < g.count; ++i)
                any_null = any_null || g.vals[i] == NULL;
            if(g.need && any_null)
                missing += (missing.empty() ? "" : ", ") + std::string(g.name);
        }
        if(!missing.empty())
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "missing operands for %s", missing.c_str());
            return ROCKE_ERR_VALUE;
        }
    }

    auto ci = [&](int64_t v) { return rocke_b_const_i32(b, v); };
    auto vec_of = [&](const rocke_type_t* e, int n) { return rocke_vector_type(b, e, n); };

    Val c0 = ci(0);
    Val c1 = ci(1);
    Val oob = ci(k_oob);
    Val neg_inf = rocke_b_const_f32(b, -INFINITY);
    Val zero_f = rocke_b_const_f32(b, 0.0);
    /* With a bias (or any other score feature) the scores move to the log2
     * domain up front (s * c + b * log2e) and the softmax runs at unit scale;
     * otherwise the scale folds into one FMA per score (p = exp2(s * c - m *
     * c)). */
    Val c_in = p->scale_log2;
    Val c = rocke_rdna_attn_log_domain(cfg) ? rocke_b_const_f32(b, 1.0) : p->scale_log2;

    Val tid = rocke_b_thread_id_x(b);
    Val lane;
    {
        Val t = ci(31);
        lane = rocke_b_land(b, tid, t);
    }
    Val col;
    {
        Val t = ci(15);
        col = rocke_b_land(b, tid, t);
    }
    Val half;
    {
        Val t = ci(4);
        Val s = rocke_b_lshr(b, tid, t);
        half = rocke_b_land(b, s, c1);
    }
    Val lo_half = rocke_b_cmp_eq(b, half, c0);
    Val wave;
    {
        Val t = ci(5);
        Val s = rocke_b_lshr(b, tid, t);
        wave = rocke_b_readfirstlane(b, s);
    }
    Val row_wave, side, dq_base, dv_base;
    if(pair)
    {
        row_wave = rocke_b_lshr(b, wave, c1);
        side = rocke_b_land(b, wave, c1);
        Val tq = ci(DQW);
        dq_base = rocke_b_mul(b, side, tq);
        Val tv = ci(DVW);
        dv_base = rocke_b_mul(b, side, tv);
    }
    else
    {
        row_wave = wave;
        side = NULL;
        dq_base = c0;
        dv_base = c0;
    }
    Val q_row;
    {
        Val t = ci(16);
        Val m = rocke_b_mul(b, row_wave, t);
        Val a = rocke_b_add(b, p->q_start, m);
        q_row = rocke_b_add(b, a, col);
    }

    auto i32x4_to = [&](Val v, int n) { return rocke_b_vec_bitcast(b, v, vec_of(elem, n)); };
    auto load16 = [&](Val rsrc, Val byte_off) {
        return rocke_b_buffer_load_vN(b, rsrc, byte_off, c0, rocke_i32(), 4);
    };
    auto masked_off = [&](Val byte_off, Val ok) { return rocke_b_select(b, ok, byte_off, oob); };

    /* Two f32 -> one dword of two 16-bit values (p0 low). */
    auto pack2 = [&](Val p0, Val p1) -> Val {
        if(bf16)
        {
            /* Round-to-nearest-even without NaN handling (P is finite). */
            Val r[2];
            Val src[2] = {p0, p1};
            for(int k = 0; k < 2; ++k)
            {
                Val x = rocke_b_bitcast(b, src[k], rocke_i32());
                Val t16 = ci(16);
                Val sh = rocke_b_lshr(b, x, t16);
                Val lsb = rocke_b_land(b, sh, c1);
                Val t7 = ci(0x7FFF);
                Val a = rocke_b_add(b, x, t7);
                r[k] = rocke_b_add(b, a, lsb);
            }
            Val sel = ci(0x07060302);
            return rocke_b_perm_b32(b, r[1], r[0], sel);
        }
        Val lo;
        {
            Val h = rocke_b_cast_f32_to(b, p0, rocke_f16());
            Val i = rocke_b_bitcast(b, h, rocke_i16());
            lo = rocke_b_zext(b, i, rocke_i32());
        }
        Val hi;
        {
            Val h = rocke_b_cast_f32_to(b, p1, rocke_f16());
            Val i = rocke_b_bitcast(b, h, rocke_i16());
            hi = rocke_b_zext(b, i, rocke_i32());
        }
        Val t16 = ci(16);
        Val s = rocke_b_shl(b, hi, t16);
        return rocke_b_lor(b, lo, s);
    };

    auto permx16_f32 = [&](Val v) {
        Val i = rocke_b_bitcast(b, v, rocke_i32());
        Val pl = rocke_b_permlanex16(b, i);
        return rocke_b_bitcast(b, pl, rocke_f32());
    };

    using Bin = rocke_value_t* (*)(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*);
    using Tri
        = rocke_value_t* (*)(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*, rocke_value_t*);
    auto tree = [&](Vals vals, Bin op, Tri op3) -> Val {
        while(vals.size() > 1)
        {
            Vals nxt;
            size_t i = 0;
            if(op3 != NULL)
            {
                while(i + 3 <= vals.size())
                {
                    nxt.push_back(op3(b, vals[i], vals[i + 1], vals[i + 2]));
                    i += 3;
                }
            }
            while(i + 2 <= vals.size())
            {
                nxt.push_back(op(b, vals[i], vals[i + 1]));
                i += 2;
            }
            for(; i < vals.size(); ++i)
                nxt.push_back(vals[i]);
            vals.swap(nxt);
        }
        return vals[0];
    };

    /* ---- Q: B operand of S^T = K Q^T, resident in registers ---- */
    Val q_row_off;
    {
        Val m = rocke_b_mul(b, q_row, p->stride_q);
        q_row_off = rocke_b_add(b, m, dq_base);
    }
    Vals q_frags;
    for(int dt = 0; dt < n_dq; ++dt)
    {
        if(gfx12)
        {
            Val a = ci(dt * 16);
            Val t8 = ci(8);
            Val m = rocke_b_mul(b, half, t8);
            Val d0 = rocke_b_add(b, a, m);
            Val s = rocke_b_add(b, q_row_off, d0);
            Val t2 = ci(2);
            Val off = rocke_b_mul(b, s, t2);
            Val db = rocke_b_add(b, d0, dq_base);
            Val ok = rocke_b_cmp_lt(b, db, p->head_dim_q);
            Val mo = masked_off(off, ok);
            q_frags.push_back(i32x4_to(load16(p->q_rsrc, mo), 8));
        }
        else
        {
            Val parts[2];
            for(int h = 0; h < 2; ++h)
            {
                const int d0 = dt * 16 + 8 * h;
                Val td = ci(d0);
                Val s = rocke_b_add(b, q_row_off, td);
                Val t2 = ci(2);
                Val off = rocke_b_mul(b, s, t2);
                Val td2 = ci(d0);
                Val db = rocke_b_add(b, td2, dq_base);
                Val ok = rocke_b_cmp_lt(b, db, p->head_dim_q);
                Val mo = masked_off(off, ok);
                parts[h] = i32x4_to(load16(p->q_rsrc, mo), 8);
            }
            q_frags.push_back(rocke_b_vec_concat(b, parts[0], parts[1]));
        }
    }

    /* ---- LDS: [K tile | V^T tile] x buffers (+ wave-pair score exchange) ---- */
    const int lds_shape[] = {lds_elems(cfg)};
    Val lds = rocke_b_smem_alloc(b, elem, lds_shape, 1, "attn_kv");
    /* Private 16-byte slot per thread for stores of idle cooperative rounds. */
    Val dummy;
    {
        Val a = ci(n_buffers * buf_elems);
        Val t8 = ci(8);
        Val m = rocke_b_mul(b, tid, t8);
        dummy = rocke_b_add(b, a, m);
    }
    Val xs = NULL;
    if(pair)
    {
        const int xs_shape[] = {xs_floats(cfg)};
        xs = rocke_b_smem_alloc(b, rocke_f32(), xs_shape, 1, "attn_xs");
    }

    /* K / V element bytes, page geometry. Pages that are not a whole number
     * of key tiles take one block-table lookup per cooperative row instead of
     * one per tile. */
    const int ps = cfg->page_block_size;
    const bool row_paged = paged && ps % BN != 0;
    const bool fp8 = cfg->kv_fp8;

    /* K / V element offset -> byte offset. */
    auto kv_byte = [&](Val x) -> Val {
        if(fp8)
            return x;
        Val t2 = ci(2);
        return rocke_b_mul(b, x, t2);
    };

    /* One cooperative chunk: 8 K / V elements (16 bytes, or 8 FP8 bytes). */
    auto load_kv = [&](Val rsrc, Val byte_off) -> Val {
        if(fp8)
            return rocke_b_buffer_load_vN(b, rsrc, byte_off, c0, rocke_i32(), 2);
        return load16(rsrc, byte_off);
    };

    /* Cooperative K mapping: 16-byte chunk -> (row, col chunk). */
    const int cpr_k = DQ / 8;
    const int tot_k = BN * cpr_k;
    const int nk = (tot_k + NT - 1) / NT;
    Vals k_goff, k_loff, k_valid, k_rows;
    for(int i = 0; i < nk; ++i)
    {
        Val ti = ci(i * NT);
        Val cidx = rocke_b_add(b, tid, ti);
        Val valid = NULL;
        if((i + 1) * NT > tot_k)
        {
            Val tt = ci(tot_k);
            valid = rocke_b_cmp_lt(b, cidx, tt);
        }
        Val tc = ci(cpr_k);
        Val r = rocke_b_div(b, cidx, tc);
        Val tc2 = ci(cpr_k);
        Val md = rocke_b_mod(b, cidx, tc2);
        Val t8 = ci(8);
        Val d0 = rocke_b_mul(b, md, t8);
        Val off;
        if(row_paged)
            off = kv_byte(d0);
        else
        {
            Val rs = rocke_b_mul(b, r, p->stride_k);
            Val s = rocke_b_add(b, rs, d0);
            off = kv_byte(s);
        }
        Val ok = rocke_b_cmp_lt(b, d0, p->head_dim_q);
        if(valid != NULL)
            ok = rocke_b_land(b, ok, valid);
        k_goff.push_back(masked_off(off, ok));
        Val tks = ci(KS);
        Val rk = rocke_b_mul(b, r, tks);
        k_loff.push_back(rocke_b_add(b, rk, d0));
        k_valid.push_back(valid);
        k_rows.push_back(r);
    }

    /* Cooperative V mapping: G key rows x 8 columns per unit, stored transposed. */
    int G, nv;
    v_group(cfg, &G, &nv);
    const int groups = BN / G;
    const int tot_units = groups * (DV / 8);
    std::vector<Vals> v_goff, v_rows;
    Vals v_loff, v_valid;
    for(int i = 0; i < nv; ++i)
    {
        Val ti = ci(i * NT);
        Val uidx = rocke_b_add(b, tid, ti);
        Val valid = NULL;
        if((i + 1) * NT > tot_units)
        {
            Val tt = ci(tot_units);
            valid = rocke_b_cmp_lt(b, uidx, tt);
        }
        Val tg = ci(groups);
        Val g = rocke_b_mod(b, uidx, tg);
        Val tg2 = ci(groups);
        Val dv = rocke_b_div(b, uidx, tg2);
        Val t8 = ci(8);
        Val d0 = rocke_b_mul(b, dv, t8);
        Val ok = rocke_b_cmp_lt(b, d0, p->head_dim_v);
        if(valid != NULL)
            ok = rocke_b_land(b, ok, valid);
        Vals offs, rows;
        for(int rr = 0; rr < G; ++rr)
        {
            Val tG = ci(G);
            Val gm = rocke_b_mul(b, g, tG);
            Val tr = ci(rr);
            Val row = rocke_b_add(b, gm, tr);
            Val off;
            if(row_paged)
                off = kv_byte(d0);
            else
            {
                Val rsv = rocke_b_mul(b, row, p->stride_v);
                Val s = rocke_b_add(b, rsv, d0);
                off = kv_byte(s);
            }
            offs.push_back(masked_off(off, ok));
            rows.push_back(row);
        }
        v_goff.push_back(offs);
        Val tvs = ci(VS);
        Val m1 = rocke_b_mul(b, d0, tvs);
        Val tG = ci(G);
        Val m2 = rocke_b_mul(b, g, tG);
        v_loff.push_back(rocke_b_add(b, m1, m2));
        v_valid.push_back(valid);
        v_rows.push_back(rows);
    }
    Val tile_k_bytes = NULL, tile_v_bytes = NULL;
    if(!paged)
    {
        const int kv_eb = fp8 ? 1 : 2;
        Val tk = ci(BN * kv_eb);
        tile_k_bytes = rocke_b_mul(b, p->stride_k, tk);
        Val tv = ci(BN * kv_eb);
        tile_v_bytes = rocke_b_mul(b, p->stride_v, tv);
    }

    /* Paged K / V: rows past the sequence (or of an invalid tile) read
     * nothing; their block-table index is clamped to 0. A page id outside
     * [0, num_blocks) reads nothing either. */
    auto page_ok = [&](Val phys) {
        Val ge = rocke_b_cmp_ge(b, phys, c0);
        Val lt = rocke_b_cmp_lt(b, phys, p->num_blocks);
        return rocke_b_land(b, ge, lt);
    };
    auto issue_paged_loads = [&](Val t, Val valid, Vals* kr, std::vector<Vals>* vr) {
        int lps = 0;
        while((1 << lps) < ps)
            ++lps;
        Val tok0;
        {
            Val tb = ci(BN);
            tok0 = rocke_b_mul(b, t, tb);
        }
        Val rem; /* live rows of the tile */
        {
            Val s = rocke_b_sub(b, p->seqlen_k, tok0);
            rem = rocke_b_select(b, valid, s, c0);
        }
        kr->clear();
        vr->clear();
        if(!row_paged)
        {
            /* The whole tile sits in one page: one lookup per tile. */
            Val blk;
            {
                Val tl = ci(lps);
                Val sh = rocke_b_lshr(b, tok0, tl);
                blk = rocke_b_select(b, valid, sh, c0);
            }
            Val phys;
            {
                Val idx = rocke_b_add(b, p->table_row, blk);
                phys = rocke_b_global_load_i32(b, p->block_table, idx, 4);
            }
            {
                Val in_cache = page_ok(phys);
                rem = rocke_b_select(b, in_cache, rem, c0);
            }
            Val in_page;
            {
                Val tm = ci(ps - 1);
                in_page = rocke_b_land(b, tok0, tm);
            }
            auto tile_off = [&](Val stride_blk, Val stride_tok) {
                Val m1 = rocke_b_mul(b, phys, stride_blk);
                Val m2 = rocke_b_mul(b, in_page, stride_tok);
                Val a = rocke_b_add(b, m1, m2);
                return kv_byte(a);
            };
            Val tk = tile_off(p->stride_k_block, p->stride_k);
            Val tv = tile_off(p->stride_v_block, p->stride_v);
            auto at_tile = [&](Val o, Val r, Val off) {
                Val live = rocke_b_cmp_lt(b, r, rem);
                Val a = rocke_b_add(b, o, off);
                return rocke_b_select(b, live, a, oob);
            };
            for(size_t i = 0; i < k_goff.size(); ++i)
                kr->push_back(load_kv(p->k_rsrc, at_tile(k_goff[i], k_rows[i], tk)));
            for(size_t u = 0; u < v_goff.size(); ++u)
            {
                Vals out;
                for(size_t rr = 0; rr < v_goff[u].size(); ++rr)
                    out.push_back(load_kv(p->v_rsrc, at_tile(v_goff[u][rr], v_rows[u][rr], tv)));
                vr->push_back(out);
            }
            return;
        }
        auto at_row = [&](Val o, Val r, Val stride_blk, Val stride_tok) {
            Val tok = rocke_b_add(b, tok0, r);
            Val ok = rocke_b_cmp_lt(b, r, rem);
            Val blk;
            {
                Val tl = ci(lps);
                Val sh = rocke_b_lshr(b, tok, tl);
                blk = rocke_b_select(b, ok, sh, c0);
            }
            Val phys;
            {
                Val idx = rocke_b_add(b, p->table_row, blk);
                phys = rocke_b_global_load_i32(b, p->block_table, idx, 4);
            }
            {
                Val in_cache = page_ok(phys);
                ok = rocke_b_land(b, ok, in_cache);
            }
            Val in_page;
            {
                Val tm = ci(ps - 1);
                in_page = rocke_b_land(b, tok, tm);
            }
            Val m1 = rocke_b_mul(b, phys, stride_blk);
            Val m2 = rocke_b_mul(b, in_page, stride_tok);
            Val row = rocke_b_add(b, m1, m2);
            Val rb = kv_byte(row);
            Val a = rocke_b_add(b, o, rb);
            return rocke_b_select(b, ok, a, oob);
        };
        for(size_t i = 0; i < k_goff.size(); ++i)
        {
            Val off = at_row(k_goff[i], k_rows[i], p->stride_k_block, p->stride_k);
            kr->push_back(load_kv(p->k_rsrc, off));
        }
        for(size_t u = 0; u < v_goff.size(); ++u)
        {
            Vals out;
            for(size_t rr = 0; rr < v_goff[u].size(); ++rr)
            {
                Val off = at_row(v_goff[u][rr], v_rows[u][rr], p->stride_v_block, p->stride_v);
                out.push_back(load_kv(p->v_rsrc, off));
            }
            vr->push_back(out);
        }
    };

    /* Global loads of key tile `t` (zeros, no traffic, if not valid). Dense
     * K/V addressing is the per-thread byte offsets above plus one per-tile
     * offset here. */
    auto issue_loads = [&](Val t, Val valid, Vals* kr, std::vector<Vals>* vr) {
        if(paged)
        {
            issue_paged_loads(t, valid, kr, vr);
            return;
        }
        Val mk = rocke_b_mul(b, t, tile_k_bytes);
        Val tk = rocke_b_select(b, valid, mk, p->k_bytes);
        Val mv = rocke_b_mul(b, t, tile_v_bytes);
        Val tv = rocke_b_select(b, valid, mv, p->v_bytes);
        kr->clear();
        for(Val o : k_goff)
        {
            Val a = rocke_b_add(b, o, tk);
            kr->push_back(load_kv(p->k_rsrc, a));
        }
        vr->clear();
        for(const Vals& offs : v_goff)
        {
            Vals out;
            for(Val o : offs)
            {
                Val a = rocke_b_add(b, o, tv);
                out.push_back(load_kv(p->v_rsrc, a));
            }
            vr->push_back(out);
        }
    };

    /* Eight OCP E4M3FN bytes (<2 x i32>) -> <8 x elem>: decoded, multiplied
     * by `scale` and rounded to the Q dtype. */
    auto widen_fp8 = [&](Val raw, Val scale) -> Val {
        Val words[2];
        words[0] = rocke_b_vec_extract(b, raw, 0);
        words[1] = rocke_b_vec_extract(b, raw, 1);
        Val vals[8];
        for(int k = 0; k < 8; ++k)
        {
            Val w = words[k / 4];
            if(k % 4)
            {
                Val ts = ci(8 * (k % 4));
                w = rocke_b_lshr(b, w, ts);
            }
            Val byte = rocke_b_trunc(b, w, rocke_i8());
            Val f = rocke_decode_fp8e4m3fn_to_f32(b, byte);
            Val fs = rocke_b_fmul(b, f, scale);
            vals[k] = rocke_b_cast_f32_to(b, fs, elem);
        }
        return rocke_b_vec_pack(b, vals, 8, elem);
    };

    auto dst_of = [&](Val addr, Val valid) {
        return valid == NULL ? addr : rocke_b_select(b, valid, addr, dummy);
    };

    auto store_k = [&](Val base, const Vals& kr) {
        for(size_t i = 0; i < k_loff.size(); ++i)
        {
            Val a = rocke_b_add(b, base, k_loff[i]);
            Val idx[1] = {dst_of(a, k_valid[i])};
            Val v = fp8 ? widen_fp8(kr[i], p->k_scale) : i32x4_to(kr[i], 8);
            rocke_b_smem_store_vN(b, lds, idx, 1, v, 8);
        }
    };

    auto store_v_unit = [&](Val vbase, Val o, Vals rows, Val valid) {
        if(fp8)
        {
            for(Val& rv : rows)
            {
                Val w = widen_fp8(rv, p->v_scale);
                rv = rocke_b_vec_bitcast(b, w, vec_of(rocke_i32(), 4));
            }
        }
        std::vector<Vals> words;
        for(Val rv : rows)
        {
            Vals w;
            for(int j = 0; j < 4; ++j)
                w.push_back(rocke_b_vec_extract(b, rv, j));
            words.push_back(w);
        }
        for(int j = 0; j < 4; ++j)
        {
            for(int odd = 0; odd < 2; ++odd)
            {
                Val tj = ci((2 * j + odd) * VS);
                Val oj = rocke_b_add(b, o, tj);
                Val a = rocke_b_add(b, vbase, oj);
                Val idx[1] = {dst_of(a, valid)};
                if(G == 1)
                {
                    Val v8 = i32x4_to(rows[0], 8);
                    Val e = rocke_b_vec_extract(b, v8, 2 * j + odd);
                    rocke_b_smem_store_vN(b, lds, idx, 1, e, 1);
                    continue;
                }
                Val sel = ci(odd ? 0x07060302 : 0x05040100);
                Vals pairs;
                for(int rr = 0; rr < G; rr += 2)
                    pairs.push_back(rocke_b_perm_b32(b, words[rr + 1][j], words[rr][j], sel));
                Val vec = rocke_b_vec_pack(b, pairs.data(), (int)pairs.size(), rocke_i32());
                Val vb = rocke_b_vec_bitcast(b, vec, vec_of(elem, G));
                rocke_b_smem_store_vN(b, lds, idx, 1, vb, G);
            }
        }
    };

    auto store_v = [&](Val base, const std::vector<Vals>& vr) {
        Val tk = ci(k_tile);
        Val vbase = rocke_b_add(b, base, tk);
        for(size_t i = 0; i < v_loff.size(); ++i)
            store_v_unit(vbase, v_loff[i], vr[i], v_valid[i]);
    };

    /* Per-lane LDS operand bases (elements, before the buffer offset). */
    Val hoff;
    if(gfx12)
    {
        Val t8 = ci(8);
        hoff = rocke_b_mul(b, half, t8);
    }
    else
        hoff = c0;
    Val k_lane;
    {
        Val t = ci(KS);
        Val m = rocke_b_mul(b, col, t);
        Val a = rocke_b_add(b, m, hoff);
        k_lane = rocke_b_add(b, a, dq_base);
    }
    Val v_lane;
    {
        Val t = ci(VS);
        Val m = rocke_b_mul(b, col, t);
        Val a = rocke_b_add(b, m, hoff);
        Val t2 = ci(VS);
        Val m2 = rocke_b_mul(b, dv_base, t2);
        v_lane = rocke_b_add(b, a, m2);
    }

    auto lds_frag = [&](Val off) -> Val {
        Val idx[1] = {off};
        if(gfx12)
            return rocke_b_smem_load_vN(b, lds, idx, 1, elem, 8);
        Val lo = rocke_b_smem_load_vN(b, lds, idx, 1, elem, 8);
        Val t8 = ci(8);
        Val idx2[1] = {rocke_b_add(b, off, t8)};
        Val hi = rocke_b_smem_load_vN(b, lds, idx2, 1, elem, 8);
        return rocke_b_vec_concat(b, lo, hi);
    };

    /* gfx11 C->B transpose selectors (swapped for the upper half). */
    Val sel_lo;
    {
        Val a = ci(0x05040100);
        Val d = ci(0x01000504);
        sel_lo = rocke_b_select(b, lo_half, a, d);
    }
    Val sel_hi;
    {
        Val a = ci(0x07060302);
        Val d = ci(0x03020706);
        sel_hi = rocke_b_select(b, lo_half, a, d);
    }

    auto p_operand = [&](const Vals& ps) -> Val {
        if(gfx12)
        {
            Vals words;
            for(int j = 0; j < 4; ++j)
                words.push_back(pack2(ps[2 * j], ps[2 * j + 1]));
            Val vp = rocke_b_vec_pack(b, words.data(), 4, rocke_i32());
            return rocke_b_vec_bitcast(b, vp, vec_of(elem, 8));
        }
        Vals out;
        for(int m = 0; m < 4; ++m)
        {
            Val v = pack2(ps[2 * m], ps[2 * m + 1]); /* keys {4m+h, 4m+2+h} */
            Val w = rocke_b_permlanex16(b, v);
            out.push_back(rocke_b_perm_b32(b, w, v, sel_lo)); /* keys {4m, 4m+1} */
            out.push_back(rocke_b_perm_b32(b, w, v, sel_hi)); /* keys {4m+2, 4m+3} */
        }
        Val vp = rocke_b_vec_pack(b, out.data(), 8, rocke_i32());
        return rocke_b_vec_bitcast(b, vp, vec_of(elem, 16));
    };

    /* Key offset of accumulator slot i inside a 16-key sub-tile, minus the
     * lane-half part (folded into the per-lane bounds). */
    auto slot_key = [&](int ns, int i) { return ns * 16 + (gfx12 ? i : 2 * i); };

    Val half_key;
    if(gfx12)
    {
        Val t8 = ci(8);
        half_key = rocke_b_mul(b, half, t8);
    }
    else
        half_key = half;

    /* ---- additive bias: this lane's query row, its slots' keys ---- */
    Val log2e = rocke_b_const_f32(b, k_log2e);
    Val bias_row = has_bias ? rocke_b_mul(b, q_row, p->stride_bias_q) : NULL;
    Val bias_sh = NULL, bias_tail_off = NULL, bias_tail = NULL;
    if(bias_q)
    {
        Val row = rocke_b_add(b, bias_row, p->bias_lead);
        if(gfx12)
        {
            /* Rows load from their exact start, which may sit mid-dword. Of a
             * dword dropped for ending past num_records, only the low half at
             * num_records - 2 can be in range: the resource's last element,
             * key seqlen_k - 1 of the last row. A mid-dword row ending on a
             * dword has an odd seqlen_k, so that key is in a masked tile; the
             * element's dword is fetched once here. */
            bias_row = row;
            Val t2 = ci(2);
            bias_tail_off = rocke_b_sub(b, p->bias_bytes, t2);
            Val t4 = ci(4);
            Val toff = rocke_b_sub(b, p->bias_bytes, t4);
            Val tail = rocke_b_buffer_load_vN(b, p->bias_rsrc, toff, c0, rocke_i32(), 1);
            Val tw = rocke_b_vec_extract(b, tail, 0);
            Val t16 = ci(16);
            bias_tail = rocke_b_lshr(b, tw, t16);
        }
        else
        {
            /* Loads start at the dword holding this lane's first key (relative
             * to the dword-aligned resource); bias_par is the 16-bit half it
             * sits in. Tile and sub-tile key offsets are even, so the parity is
             * per lane. */
            row = rocke_b_add(b, row, half);
            Val bias_par = rocke_b_land(b, row, c1);
            bias_row = rocke_b_sub(b, row, bias_par);
            Val t16 = ci(16);
            bias_sh = rocke_b_mul(b, bias_par, t16);
        }
    }

    /* ---- score features: per-CTA constants, per-lane QQ-bias row ---- */
    Val cap2 = NULL, cap_in = NULL;
    if(cfg->softcap)
    {
        cap2 = rocke_b_fmul(b, p->softcap, log2e);
        cap_in = rocke_b_fdiv(b, c_in, cap2); /* raw score -> tanh argument */
    }
    Val qq_row = NULL, qq_q_ok = NULL;
    if(cfg->qq_bias)
    {
        qq_row = rocke_b_mul(b, q_row, p->qq_stride);
        qq_q_ok = rocke_b_cmp_lt(b, q_row, p->qq_rows);
    }

    /* Key of slot 0 of this lane in tile `t`, minus `context`. */
    auto rel_keys = [&](Val t) {
        Val tB = ci(BN);
        Val tm = rocke_b_mul(b, t, tB);
        Val a = rocke_b_add(b, tm, half_key);
        return rocke_b_sub(b, a, p->context);
    };

    /* QQ-bias (raw) of every score slot; 0 outside its bounds. */
    auto load_qq = [&](Val rel0) {
        std::vector<Vals> out;
        for(int ns = 0; ns < n_kv; ++ns)
        {
            Vals vals;
            for(int i = 0; i < 8; ++i)
            {
                Val tk = ci(slot_key(ns, i));
                Val rel = rocke_b_add(b, rel0, tk);
                Val ge = rocke_b_cmp_ge(b, rel, c0);
                Val lt = rocke_b_cmp_lt(b, rel, p->qq_cols);
                Val in_cols = rocke_b_land(b, ge, lt);
                Val ok = rocke_b_land(b, qq_q_ok, in_cols);
                Val a = rocke_b_add(b, qq_row, rel);
                Val t4 = ci(4);
                Val off = rocke_b_mul(b, a, t4);
                Val mo = masked_off(off, ok);
                Val v1 = rocke_b_buffer_load_vN(b, p->qq_rsrc, mo, c0, rocke_f32(), 1);
                vals.push_back(rocke_b_vec_extract(b, v1, 0));
            }
            out.push_back(vals);
        }
        return out;
    };

    /* Bias (log2 domain) of every score slot of key tile `t`. */
    auto load_bias = [&](Val t, bool masked) {
        const int eb = bias_f32 ? 4 : 2;
        std::vector<Vals> out;
        for(int ns = 0; ns < n_kv; ++ns)
        {
            /* gfx12: the 8 contiguous keys at 8 * half; gfx11: all 16 keys of
             * the sub-tile, of which the lane keeps parity `half` (Q dtype: the
             * 16 keys from the lane's aligned start, keeping bias_par). */
            Val tB = ci(BN);
            Val tm = rocke_b_mul(b, t, tB);
            Val tn = ci(ns * 16);
            Val k0 = rocke_b_add(b, tm, tn);
            if(gfx12)
            {
                Val t8 = ci(8);
                Val hm = rocke_b_mul(b, half, t8);
                k0 = rocke_b_add(b, k0, hm);
            }
            const int n_keys = gfx12 ? 8 : 16;
            Val br = rocke_b_add(b, bias_row, k0);
            Val te = ci(eb);
            Val base = rocke_b_mul(b, br, te);
            Vals words;
            for(int q = 0; q < n_keys * eb / 16; ++q)
            {
                Val tq = ci(16 * q);
                Val a = rocke_b_add(b, base, tq);
                Val v4 = load16(p->bias_rsrc, a);
                for(int j = 0; j < 4; ++j)
                    words.push_back(rocke_b_vec_extract(b, v4, j));
            }
            if(bias_q && gfx12 && masked)
            {
                /* Restore the resource's last element if this lane's load lost it. */
                for(int j = 0; j < 4; ++j)
                {
                    Val tj = ci(4 * j);
                    Val a = rocke_b_add(b, base, tj);
                    Val hit = rocke_b_cmp_eq(b, a, bias_tail_off);
                    words[j] = rocke_b_select(b, hit, bias_tail, words[j]);
                }
            }
            Vals vals;
            for(int i = 0; i < 8; ++i)
            {
                Val x;
                if(bias_f32)
                {
                    Val w = gfx12 ? words[i]
                                  : rocke_b_select(b, lo_half, words[2 * i], words[2 * i + 1]);
                    x = rocke_b_bitcast(b, w, rocke_f32());
                }
                else
                {
                    Val w = gfx12 ? words[i / 2] : words[i];
                    Val sh = gfx12 ? ci(16 * (i % 2)) : bias_sh;
                    Val ls = rocke_b_lshr(b, w, sh);
                    Val tf = ci(0xFFFF);
                    Val h = rocke_b_land(b, ls, tf);
                    if(bf16)
                    {
                        Val t16 = ci(16);
                        Val s = rocke_b_shl(b, h, t16);
                        x = rocke_b_bitcast(b, s, rocke_f32());
                    }
                    else
                    {
                        Val tr = rocke_b_trunc(b, h, rocke_i16());
                        Val hf = rocke_b_bitcast(b, tr, rocke_f16());
                        x = rocke_b_cast_to_f32(b, hf);
                    }
                }
                vals.push_back(rocke_b_fmul(b, x, log2e));
            }
            out.push_back(vals);
        }
        return out;
    };

    /* ---- key-tile ranges (all scalar) ---- */
    auto cdiv = [&](Val x, int d) {
        Val t = ci(d - 1);
        Val a = rocke_b_add(b, x, t);
        Val td = ci(d);
        return rocke_b_div(b, a, td);
    };

    Val zero_rows = rocke_b_cmp_le(b, p->seqlen_k, c0);
    Val sk = rocke_b_smax(b, p->seqlen_k, c0);
    Val n_tiles = cdiv(sk, BN);
    Val lo_t, full_lo, full_hi, hi_t, lane_hi, lane_lo;
    if(!causal)
    {
        lo_t = c0;
        full_lo = c0;
        Val tB = ci(BN);
        full_hi = rocke_b_div(b, sk, tB);
        hi_t = n_tiles;
        lane_hi = sk;
        lane_lo = NULL;
    }
    else
    {
        Val diag_off = p->diag_offset;
        Val q_last;
        {
            Val t = ci(cfg->block_m);
            Val a = rocke_b_add(b, p->q_start, t);
            Val mn = rocke_b_smin(b, a, p->seqlen_q);
            q_last = rocke_b_sub(b, mn, c1);
        }
        q_last = rocke_b_smax(b, q_last, p->q_start);
        Val wl = p->window_left, wr = p->window_right;
        Val has_l = rocke_b_cmp_ge(b, wl, c0);
        Val has_r = rocke_b_cmp_ge(b, wr, c0);

        auto clamp_k = [&](Val x) {
            Val mx = rocke_b_smax(b, x, c0);
            return rocke_b_smin(b, mx, sk);
        };

        Val k_lo;
        {
            Val a = rocke_b_add(b, p->q_start, diag_off);
            Val s = rocke_b_sub(b, a, wl);
            Val cl = clamp_k(s);
            k_lo = rocke_b_select(b, has_l, cl, c0);
        }
        Val k_hi;
        {
            Val a1 = rocke_b_add(b, q_last, diag_off);
            Val a2 = rocke_b_add(b, a1, wr);
            Val a3 = rocke_b_add(b, a2, c1);
            Val cl = clamp_k(a3);
            k_hi = rocke_b_select(b, has_r, cl, sk);
        }
        {
            Val tB = ci(BN);
            lo_t = rocke_b_div(b, k_lo, tB);
        }
        {
            Val cd = cdiv(k_hi, BN);
            hi_t = rocke_b_smax(b, cd, lo_t);
        }
        Val fl;
        {
            Val a = rocke_b_add(b, q_last, diag_off);
            Val s = rocke_b_sub(b, a, wl);
            Val cl = clamp_k(s);
            Val cd = cdiv(cl, BN);
            fl = rocke_b_select(b, has_l, cd, lo_t);
        }
        {
            Val mx = rocke_b_smax(b, fl, lo_t);
            full_lo = rocke_b_smin(b, mx, hi_t);
        }
        Val fh;
        {
            Val a1 = rocke_b_add(b, p->q_start, diag_off);
            Val a2 = rocke_b_add(b, a1, wr);
            Val a3 = rocke_b_add(b, a2, c1);
            Val cl = clamp_k(a3);
            Val tB = ci(BN);
            Val d1 = rocke_b_div(b, cl, tB);
            Val tB2 = ci(BN);
            Val d2 = rocke_b_div(b, sk, tB2);
            fh = rocke_b_select(b, has_r, d1, d2);
        }
        {
            Val mx = rocke_b_smax(b, fh, full_lo);
            full_hi = rocke_b_smin(b, mx, hi_t);
        }
        Val diag = rocke_b_add(b, q_row, diag_off);
        {
            Val a = rocke_b_add(b, diag, wr);
            Val a2 = rocke_b_add(b, a, c1);
            Val mn = rocke_b_smin(b, a2, sk);
            lane_hi = rocke_b_select(b, has_r, mn, sk);
        }
        {
            Val s = rocke_b_sub(b, diag, wl);
            Val t = ci(-(1LL << 30));
            lane_lo = rocke_b_select(b, has_l, s, t);
        }
    }
    Val n_full = rocke_b_sub(b, full_hi, full_lo);
    Val n_left = rocke_b_sub(b, full_lo, lo_t);
    Val total;
    {
        Val s = rocke_b_sub(b, hi_t, full_hi);
        Val a = rocke_b_add(b, n_left, s);
        total = rocke_b_add(b, n_full, a);
    }
    total = rocke_b_select(b, zero_rows, c0, total);
    n_full = rocke_b_select(b, zero_rows, c0, n_full);

    auto tile_of = [&](Val j) -> Val {
        if(!causal)
            return j;
        Val jl = rocke_b_sub(b, j, n_full);
        Val c_full = rocke_b_cmp_lt(b, j, n_full);
        Val a1 = rocke_b_add(b, full_lo, j);
        Val c_left = rocke_b_cmp_lt(b, jl, n_left);
        Val a2 = rocke_b_add(b, lo_t, jl);
        Val s = rocke_b_sub(b, jl, n_left);
        Val a3 = rocke_b_add(b, full_hi, s);
        Val inner = rocke_b_select(b, c_left, a2, a3);
        return rocke_b_select(b, c_full, a1, inner);
    };

    /* ---- prologue: tile 0 -> buffer 0 ---- */
    {
        Vals kr;
        std::vector<Vals> vr;
        Val t0 = tile_of(c0);
        Val v0 = rocke_b_cmp_gt(b, total, c0);
        issue_loads(t0, v0, &kr, &vr);
        store_k(c0, kr);
        store_v(c0, vr);
        rocke_b_sync_lds_only(b);
    }
    const int n_quads = n_kv * 2;

    /* Wave pair: publish this wave's partial S^T; returns the partner's base. */
    auto exchange_scores = [&](const Vals& s) -> Val {
        Val mine;
        {
            Val t = ci(n_quads * k_wave);
            Val m = rocke_b_mul(b, wave, t);
            Val a = rocke_b_add(b, m, lane);
            Val t4 = ci(4);
            mine = rocke_b_mul(b, a, t4);
        }
        Val theirs;
        {
            Val x = rocke_b_xor(b, wave, c1);
            Val t = ci(n_quads * k_wave);
            Val m = rocke_b_mul(b, x, t);
            Val a = rocke_b_add(b, m, lane);
            Val t4 = ci(4);
            theirs = rocke_b_mul(b, a, t4);
        }
        for(int ns = 0; ns < n_kv; ++ns)
        {
            for(int h = 0; h < 2; ++h)
            {
                Val comps[4];
                for(int j = 0; j < 4; ++j)
                    comps[j] = rocke_b_vec_extract(b, s[ns], 4 * h + j);
                Val v4 = rocke_b_vec_pack(b, comps, 4, rocke_f32());
                Val t = ci((ns * 2 + h) * k_wave * 4);
                Val idx[1] = {rocke_b_add(b, mine, t)};
                rocke_b_smem_store_vN(b, xs, idx, 1, v4, 4);
            }
        }
        return theirs;
    };

    auto step = [&](Val j, rocke_value_t* const* state, bool masked) {
        const char* tag = masked ? "msk" : "full";
        Val m_old = state[0], l_old = state[1];
        Vals accs(state + 2, state + 2 + n_dv);
        Val t = tile_of(j);
        Val buf, nbuf;
        if(n_buffers == 2)
        {
            Val l = rocke_b_land(b, j, c1);
            Val tb = ci(buf_elems);
            buf = rocke_b_mul(b, l, tb);
            Val tb2 = ci(buf_elems);
            nbuf = rocke_b_sub(b, tb2, buf);
        }
        else
        {
            buf = c0;
            nbuf = c0;
        }
        Val j1 = rocke_b_add(b, j, c1);
        Val has_next = rocke_b_cmp_lt(b, j1, total);
        Vals nkr;
        std::vector<Vals> nvr;
        {
            Val s = rocke_b_sub(b, total, c1);
            Val mn = rocke_b_smin(b, j1, s);
            Val tn = tile_of(mn);
            issue_loads(tn, has_next, &nkr, &nvr);
        }
        std::vector<Vals> bias;
        if(has_bias)
            bias = load_bias(t, masked);
        Val rel0 = (cfg->alibi || cfg->qq_bias) ? rel_keys(t) : NULL;
        std::vector<Vals> qq;
        if(cfg->qq_bias)
            qq = load_qq(rel0);

        /* S^T = K Q^T, d-outer so the n_kv chains stay independent. */
        Val kb = rocke_b_add(b, buf, k_lane);
        Vals s;
        for(int ns = 0; ns < n_kv; ++ns)
            s.push_back(rocke_b_zero_vec_f32(b, 8));
        for(int dt = 0; dt < n_dq; ++dt)
        {
            for(int ns = 0; ns < n_kv; ++ns)
            {
                Val to = ci(ns * 16 * KS + dt * 16);
                Val kf = lds_frag(rocke_b_add(b, kb, to));
                s[ns] = rocke_b_mma(b, op_id, kf, q_frags[dt], s[ns], NULL, 0);
            }
        }

        if(pair)
        {
            Val theirs = exchange_scores(s);
            rocke_b_sync_lds_only(b); /* partials visible; every wave is done reading K */
            store_k(nbuf, nkr);
            for(int ns = 0; ns < n_kv; ++ns)
            {
                Val parts[2];
                for(int h = 0; h < 2; ++h)
                {
                    Val tq = ci((ns * 2 + h) * k_wave * 4);
                    Val idx[1] = {rocke_b_add(b, theirs, tq)};
                    parts[h] = rocke_b_smem_load_vN(b, xs, idx, 1, rocke_f32(), 4);
                }
                Val cat = rocke_b_vec_concat(b, parts[0], parts[1]);
                s[ns] = rocke_b_vector_add(b, s[ns], cat);
            }
        }

        /* Score transforms run on the per-slot scores here, in order: the
         * move to the log2 domain (through softcap), ALiBi, QQ-bias, the
         * additive bias, then the masks. */
        std::vector<Vals> vals(n_kv);
        for(int ns = 0; ns < n_kv; ++ns)
            for(int i = 0; i < 8; ++i)
                vals[ns].push_back(rocke_b_vec_extract(b, s[ns], i));
        if(cfg->softcap || cfg->alibi || cfg->qq_bias)
        {
            Val fk0 = cfg->alibi ? rocke_b_sitofp_f32(b, rel0) : NULL;
            for(int ns = 0; ns < n_kv; ++ns)
            {
                for(int i = 0; i < 8; ++i)
                {
                    Val x = vals[ns][i];
                    if(cfg->softcap)
                    {
                        Val a = rocke_b_fmul(b, x, cap_in);
                        Val th = rocke_b_tanh(b, a);
                        x = rocke_b_fmul(b, cap2, th);
                    }
                    else
                        x = rocke_b_fmul(b, x, c_in);
                    if(cfg->alibi)
                    {
                        Val kc = rocke_b_const_f32(b, (double)slot_key(ns, i));
                        Val kf = rocke_b_fadd(b, fk0, kc);
                        x = rocke_b_fma(b, p->alibi_slope_log2, kf, x);
                    }
                    if(cfg->qq_bias)
                        x = rocke_b_fma(b, qq[ns][i], log2e, x);
                    if(has_bias)
                        x = rocke_b_fadd(b, x, bias[ns][i]);
                    vals[ns][i] = x;
                }
            }
        }
        else if(has_bias)
        {
            for(int ns = 0; ns < n_kv; ++ns)
                for(int i = 0; i < 8; ++i)
                    vals[ns][i] = rocke_b_fma(b, vals[ns][i], c_in, bias[ns][i]);
        }
        if(masked)
        {
            Val tB = ci(BN);
            Val tm = rocke_b_mul(b, t, tB);
            Val kbase = rocke_b_add(b, tm, half_key);
            Val hi_rel = rocke_b_sub(b, lane_hi, kbase);
            Val lo_rel = lane_lo != NULL ? rocke_b_sub(b, lane_lo, kbase) : NULL;
            for(int ns = 0; ns < n_kv; ++ns)
            {
                for(int i = 0; i < 8; ++i)
                {
                    Val kv = ci(slot_key(ns, i));
                    Val keep = rocke_b_cmp_lt(b, kv, hi_rel);
                    if(lo_rel != NULL)
                    {
                        Val ge = rocke_b_cmp_ge(b, kv, lo_rel);
                        keep = rocke_b_land(b, keep, ge);
                    }
                    vals[ns][i] = rocke_b_select(b, keep, vals[ns][i], neg_inf);
                }
            }
        }
        Vals flat;
        for(const Vals& row : vals)
            flat.insert(flat.end(), row.begin(), row.end());
        Val local_max = tree(flat, rocke_b_fmax, rocke_b_fmax3);
        Val tile_max = rocke_b_fmax(b, local_max, permx16_f32(local_max));
        Val skip = NULL, m_new;
        if(cfg->lazy_rescale)
        {
            Val d = rocke_b_fsub(b, tile_max, m_old);
            Val dm = rocke_b_fmul(b, d, c);
            Val th = rocke_b_const_f32(b, k_lazy_rescale_threshold);
            Val within = rocke_b_fcmp(b, "ole", dm, th);
            Val sel = rocke_b_select(b, within, c0, c1);
            Val any_out = rocke_b_wave_any(b, sel);
            skip = rocke_b_cmp_eq(b, any_out, c0);
            Val mx = rocke_b_fmax(b, m_old, tile_max);
            m_new = rocke_b_select(b, skip, m_old, mx);
        }
        else
            m_new = rocke_b_fmax(b, m_old, tile_max);
        Val neg_mc;
        {
            Val mm = rocke_b_fmul(b, m_new, c);
            neg_mc = rocke_b_fneg(b, mm);
        }
        /* A row whose running max is still -inf (every visible key so far is
         * masked, or carries a -inf bias / QQ-bias) must not form -inf - -inf.
         * Only the masked tiles and the additive score terms can produce -inf. */
        if(masked || has_bias || cfg->qq_bias)
        {
            Val is_inf = rocke_b_fcmp(b, "oeq", m_new, neg_inf);
            neg_mc = rocke_b_select(b, is_inf, zero_f, neg_mc);
        }
        std::vector<Vals> ps(n_kv);
        for(int ns = 0; ns < n_kv; ++ns)
        {
            for(Val v : vals[ns])
            {
                Val f = rocke_b_fma(b, v, c, neg_mc);
                ps[ns].push_back(rocke_b_exp2_fast(b, f));
            }
        }
        Val alpha;
        {
            Val f = rocke_b_fma(b, m_old, c, neg_mc);
            alpha = rocke_b_exp2_fast(b, f);
        }
        Vals flat_p;
        for(const Vals& row : ps)
            flat_p.insert(flat_p.end(), row.begin(), row.end());
        Val lsum = tree(flat_p, rocke_b_fadd, NULL);
        Val l_scaled = rocke_b_fmul(b, l_old, alpha);
        if(skip != NULL)
            l_scaled = rocke_b_select(b, skip, l_old, l_scaled);
        Val l_new = rocke_b_fadd(b, l_scaled, lsum);
        Vals p_ops;
        for(int ns = 0; ns < n_kv; ++ns)
            p_ops.push_back(p_operand(ps[ns]));

        Val alpha_v = rocke_b_vector_splat(b, alpha, 8);
        if(skip != NULL)
        {
            Val hi = rocke_b_select(b, skip, c0, c1);
            std::vector<std::string> names;
            for(int d = 0; d < n_dv; ++d)
                names.push_back("ra" + std::to_string(d) + "_" + tag);
            std::vector<rocke_iter_arg_t> iter_args(n_dv);
            for(int d = 0; d < n_dv; ++d)
            {
                iter_args[d].name = names[d].c_str();
                iter_args[d].init = accs[d];
            }
            const std::string iv = std::string("rsc_") + tag;
            rocke_for_t rl = rocke_b_scf_for_iter(
                b, c0, hi, c1, iter_args.data(), n_dv, iv.c_str(), false, true);
            rocke_b_region_enter(b, rl.body);
            {
                Vals out;
                for(int d = 0; d < n_dv; ++d)
                    out.push_back(rocke_b_vector_mul(b, rl.iter_vars[d], alpha_v));
                rocke_b_scf_yield(b, out.data(), n_dv);
            }
            rocke_b_region_leave(b);
            for(int d = 0; d < n_dv; ++d)
                accs[d] = rl.op != NULL ? rl.op->results[d] : NULL;
        }
        else
        {
            for(int d = 0; d < n_dv; ++d)
                accs[d] = rocke_b_vector_mul(b, accs[d], alpha_v);
        }

        /* O^T += V^T P^T, V^T from LDS. */
        Val vb;
        {
            Val tk = ci(k_tile);
            Val a = rocke_b_add(b, buf, tk);
            vb = rocke_b_add(b, a, v_lane);
        }
        for(int ns = 0; ns < n_kv; ++ns)
        {
            for(int dt = 0; dt < n_dv; ++dt)
            {
                Val to = ci(dt * 16 * VS + ns * 16);
                Val vf = lds_frag(rocke_b_add(b, vb, to));
                accs[dt] = rocke_b_mma(b, op_id, vf, p_ops[ns], accs[dt], NULL, 0);
            }
        }

        if(pair)
        {
            rocke_b_sync_lds_only(b); /* every wave is done reading V and the partials */
            store_v(nbuf, nvr);
        }
        else
        {
            store_k(nbuf, nkr);
            store_v(nbuf, nvr);
            rocke_b_sync_lds_only(b);
        }
        Vals yields;
        yields.push_back(m_new);
        yields.push_back(l_new);
        yields.insert(yields.end(), accs.begin(), accs.end());
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    };

    Vals state;
    if(cfg->sinks)
    {
        /* The sink logit is the initial running max with one unit of
         * denominator mass (counted once, in the lower lane half's partial l). */
        Val m0 = rocke_rdna_attn_log_domain(cfg) ? p->sink_log2
                                                 : rocke_b_fdiv(b, p->sink_log2, p->scale_log2);
        Val one = rocke_b_const_f32(b, 1.0);
        Val l0 = rocke_b_select(b, lo_half, one, zero_f);
        state.push_back(m0);
        state.push_back(l0);
    }
    else
    {
        state.push_back(neg_inf);
        state.push_back(zero_f);
    }
    for(int d = 0; d < n_dv; ++d)
        state.push_back(rocke_b_zero_vec_f32(b, 8));
    for(int pass = 0; pass < 2; ++pass)
    {
        const bool masked = pass == 1;
        Val lo_j = masked ? n_full : c0;
        Val hi_j = masked ? total : n_full;
        const std::string tag = masked ? "msk" : "full";
        std::vector<std::string> names;
        names.push_back("m_" + tag);
        names.push_back("l_" + tag);
        for(int d = 0; d < n_dv; ++d)
            names.push_back("o" + std::to_string(d) + "_" + tag);
        std::vector<rocke_iter_arg_t> iter_args(names.size());
        for(size_t i = 0; i < names.size(); ++i)
        {
            iter_args[i].name = names[i].c_str();
            iter_args[i].init = state[i];
        }
        const std::string iv = "kt_" + tag;
        rocke_for_t loop = rocke_b_scf_for_iter(
            b, lo_j, hi_j, c1, iter_args.data(), (int)iter_args.size(), iv.c_str(), false, false);
        if(loop.op == NULL)
            return rocke_ir_builder_status(b);
        rocke_b_region_enter(b, loop.body);
        step(loop.iv, loop.iter_vars, masked);
        rocke_b_region_leave(b);
        for(size_t i = 0; i < state.size(); ++i)
            state[i] = loop.op->results[i];
    }

    /* ---- epilogue ---- */
    Val m_f = state[0], l_f = state[1];
    Val l_tot = rocke_b_fadd(b, l_f, permx16_f32(l_f));
    Val empty = rocke_b_fcmp(b, "oeq", l_tot, zero_f);
    Val inv_l;
    {
        Val r = rocke_b_rcp(b, l_tot);
        inv_l = rocke_b_select(b, empty, zero_f, r);
    }
    Val o_row_off;
    {
        Val m = rocke_b_mul(b, q_row, p->stride_o);
        o_row_off = rocke_b_add(b, m, dv_base);
    }
    Val d_lane;
    {
        Val t8 = ci(8);
        d_lane = rocke_b_mul(b, half, t8);
    }
    for(int dt = 0; dt < n_dv; ++dt)
    {
        Val acc = state[2 + dt];
        Vals v;
        for(int i = 0; i < 8; ++i)
        {
            Val e = rocke_b_vec_extract(b, acc, i);
            v.push_back(rocke_b_fmul(b, e, inv_l));
        }
        Vals h16;
        for(Val x : v)
        {
            Val h = rocke_b_cast_f32_to(b, x, elem);
            Val i = rocke_b_bitcast(b, h, rocke_i16());
            h16.push_back(rocke_b_zext(b, i, rocke_i32()));
        }
        Vals words;
        for(int m = 0; m < 4; ++m)
        {
            Val t16 = ci(16);
            Val s = rocke_b_shl(b, h16[2 * m + 1], t16);
            words.push_back(rocke_b_lor(b, h16[2 * m], s));
        }
        if(!gfx12)
        {
            /* Slot i holds d = 2i + half; trade with lane^16 so each half owns
             * 8 contiguous columns (d = 8 * half + 0..7). */
            Val send[2], keep[2], recv[2];
            for(int k = 0; k < 2; ++k)
                send[k] = rocke_b_select(b, lo_half, words[2 + k], words[k]);
            for(int k = 0; k < 2; ++k)
                keep[k] = rocke_b_select(b, lo_half, words[k], words[2 + k]);
            for(int k = 0; k < 2; ++k)
                recv[k] = rocke_b_permlanex16(b, send[k]);
            words.clear();
            for(int k = 0; k < 2; ++k)
            {
                words.push_back(rocke_b_perm_b32(b, recv[k], keep[k], sel_lo));
                words.push_back(rocke_b_perm_b32(b, recv[k], keep[k], sel_hi));
            }
        }
        Val td = ci(dt * 16);
        Val d0 = rocke_b_add(b, td, d_lane);
        Val s = rocke_b_add(b, o_row_off, d0);
        Val t2 = ci(2);
        Val off = rocke_b_mul(b, s, t2);
        Val db = rocke_b_add(b, d0, dv_base);
        Val ok = rocke_b_cmp_lt(b, db, p->head_dim_v);
        Val vp = rocke_b_vec_pack(b, words.data(), 4, rocke_i32());
        Val out = rocke_b_vec_bitcast(b, vp, vec_of(rocke_f16(), 8));
        Val mo = masked_off(off, ok);
        rocke_b_buffer_store_vN_f16(b, p->o_rsrc, mo, c0, out, 4);
    }

    if(p->lse_rsrc != NULL)
    {
        Val mc = rocke_b_fmul(b, m_f, c);
        Val lg = rocke_b_log2(b, l_tot);
        Val a = rocke_b_fadd(b, mc, lg);
        Val ln2 = rocke_b_const_f32(b, k_ln2);
        Val lse = rocke_b_fmul(b, a, ln2);
        lse = rocke_b_select(b, empty, neg_inf, lse);
        Val m = rocke_b_mul(b, q_row, p->stride_lse);
        Val t4 = ci(4);
        Val off = rocke_b_mul(b, m, t4);
        Val writer;
        if(side == NULL)
            writer = lo_half;
        else
        {
            Val is0 = rocke_b_cmp_eq(b, side, c0);
            writer = rocke_b_land(b, lo_half, is0);
        }
        Val mo = masked_off(off, writer);
        rocke_b_buffer_store_f32(b, p->lse_rsrc, mo, c0, lse);
    }
    return rocke_ir_builder_status(b);
}
