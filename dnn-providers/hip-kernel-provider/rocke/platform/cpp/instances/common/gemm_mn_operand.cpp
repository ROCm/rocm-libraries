// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gemm_mn_operand.cpp -- C++ mirror of rocke/instances/common/gemm_mn_operand.py
 * (MnLayout, the spec-level rules and the IR helpers of the DTL ring's
 * MN-contiguous operands). See rocke/instance_gemm_mn_operand.h.
 *
 * Every IR helper makes the builder calls of its Python twin in Python's
 * order (an argument that emits ops is computed in its own statement first).
 */
#include "rocke/instance_gemm_mn_operand.h"

#include "rocke/instance_tensor_gett.h" /* rocke_gett_hooks_a_sub_perm */
#include "rocke/ir_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

namespace
{

/* the K bits whose 8 combinations one half-wave of ds_read_b64_tr_b16 reads */
const int k_tr_low_bits[3] = {0, 1, 3};
/* MnLayout.sub_perm: QUAD_PERM */
const int k_quad_perm[4] = {0, 2, 1, 3};

bool reason_fmt(char* reason, size_t cap, const char* fmt, ...)
{
    if(reason != nullptr && cap > 0)
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(reason, cap, fmt, ap);
        va_end(ap);
    }
    return false;
}

/* _ilog2; -1 when x is not a power of two */
int ilog2(int x)
{
    if(x <= 0 || (x & (x - 1)))
        return -1;
    int n = 0;
    while((1 << n) < x)
        ++n;
    return n;
}

bool streq(const char* a, const char* b)
{
    return a != nullptr && b != nullptr && strcmp(a, b) == 0;
}

using V = rocke_value_t;

V* ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

} // namespace

extern "C" {

int rocke_gemm_default_mn_chunk(int tile_mn, int tile_k, int warp_ext, char* reason, size_t cap)
{
    if(tile_mn > 128)
    {
        int w = warp_ext;
        int c = ROCKE_GEMM_MN_BLOCK_BYTES * 4 / tile_k;
        if(c > w)
            w = c;
        if(16 > w)
            w = 16;
        if(w <= 256 && (w & (w - 1)) == 0 && tile_mn % w == 0
           && (warp_ext % w == 0 || w % warp_ext == 0))
            return w;
    }
    int best = 0;
    int lim = tile_mn < 256 ? tile_mn : 256;
    for(int w = 16; w <= lim; w *= 2)
    {
        bool ok = tile_mn % w == 0 && (warp_ext % w == 0 || w % warp_ext == 0);
        if(ok && tile_k * w * 2 >= ROCKE_GEMM_MN_BLOCK_BYTES)
            best = w;
    }
    if(!best)
    {
        reason_fmt(reason, cap, "no MN chunk width for tile_mn %d, tile_k %d", tile_mn, tile_k);
        return 0;
    }
    return best;
}

bool rocke_gemm_mn_layout_init(rocke_gemm_mn_layout_t* lay,
                               int tile_mn,
                               int tile_k,
                               int block_size,
                               int wave_size,
                               int warp_ext,
                               int pad,
                               int chunk,
                               bool sub_perm,
                               char* reason,
                               size_t cap)
{
    memset(lay, 0, sizeof *lay);
    lay->tile_mn = tile_mn;
    lay->tile_k = tile_k;
    lay->block_size = block_size;
    lay->wave_size = wave_size;
    lay->warp_ext = warp_ext;
    lay->pad = pad;
    lay->chunk = chunk;
    lay->sub_perm = sub_perm;
    int W = chunk;
    if(!W)
    {
        W = rocke_gemm_default_mn_chunk(tile_mn, tile_k, warp_ext, reason, cap);
        if(!W)
            return false;
    }
    lay->W = W;
    if(wave_size != 64)
        return reason_fmt(reason, cap, "MN operand DTL is built for wave64 (1 KiB per wave)");
    if(tile_k < 32)
        return reason_fmt(reason, cap, "MN operand reader needs tile_k >= 32 (16x16x32 atoms)");
    if(tile_mn % W || 2 * W > ROCKE_GEMM_MN_BLOCK_BYTES || W < 16)
        return reason_fmt(reason, cap, "MN chunk %d does not tile tile_mn %d", W, tile_mn);
    if(!(warp_ext % W == 0 || W % warp_ext == 0))
        return reason_fmt(reason, cap, "MN chunk %d vs wave extent %d", W, warp_ext);
    int R = ROCKE_GEMM_MN_BLOCK_BYTES / (2 * W);
    lay->R = R;
    int tk = ilog2(tile_k);
    if(tk < 0)
        return reason_fmt(reason, cap, "%d is not a power of two", tile_k);
    if(ilog2(W) < 0)
        return reason_fmt(reason, cap, "%d is not a power of two", W);
    std::vector<int> low, rest;
    for(int bt : k_tr_low_bits)
        if(bt < tk)
            low.push_back(bt);
    for(int bt = 0; bt < tk; ++bt)
    {
        bool in_low = false;
        for(int x : low)
            in_low = in_low || x == bt;
        if(!in_low)
            rest.push_back(bt);
    }
    int rb = ilog2(R);
    std::vector<int> row(rest.begin(),
                         rest.begin() + (rb < (int)rest.size() ? rb : (int)rest.size()));
    rest.erase(rest.begin(), rest.begin() + (int)row.size());
    while((int)row.size() < rb)
    {
        /* tile too shallow for a conflict-free block: steal low bits */
        row.push_back(low.back());
        low.pop_back();
    }
    lay->n_row_bits = (int)row.size();
    for(size_t i = 0; i < row.size(); ++i)
        lay->row_bits[i] = row[i];
    std::vector<int> blk = low;
    blk.insert(blk.end(), rest.begin(), rest.end());
    lay->n_blk_bits = (int)blk.size();
    for(size_t i = 0; i < blk.size(); ++i)
        lay->blk_bits[i] = blk[i];
    if(rocke_gemm_mn_n_blocks(lay) % rocke_gemm_mn_n_waves(lay))
        return reason_fmt(reason,
                          cap,
                          "%d LDS blocks are not whole DTL passes of %d waves",
                          rocke_gemm_mn_n_blocks(lay),
                          rocke_gemm_mn_n_waves(lay));
    return true;
}

int rocke_gemm_mn_n_waves(const rocke_gemm_mn_layout_t* l)
{
    return l->block_size / l->wave_size;
}
int rocke_gemm_mn_block_bytes(const rocke_gemm_mn_layout_t* l)
{
    return ROCKE_GEMM_MN_BLOCK_BYTES + l->pad;
}
int rocke_gemm_mn_blocks_per_chunk(const rocke_gemm_mn_layout_t* l)
{
    return l->tile_k * l->W * 2 / ROCKE_GEMM_MN_BLOCK_BYTES;
}
int rocke_gemm_mn_n_blocks(const rocke_gemm_mn_layout_t* l)
{
    return l->tile_mn * l->tile_k * 2 / ROCKE_GEMM_MN_BLOCK_BYTES;
}
int rocke_gemm_mn_n_pass(const rocke_gemm_mn_layout_t* l)
{
    return rocke_gemm_mn_n_blocks(l) / rocke_gemm_mn_n_waves(l);
}
int rocke_gemm_mn_stage_bytes(const rocke_gemm_mn_layout_t* l)
{
    return rocke_gemm_mn_n_blocks(l) * rocke_gemm_mn_block_bytes(l);
}
int rocke_gemm_mn_wave_bytes(const rocke_gemm_mn_layout_t* l)
{
    return rocke_gemm_mn_block_bytes(l);
}
int rocke_gemm_mn_pass_bytes(const rocke_gemm_mn_layout_t* l)
{
    return rocke_gemm_mn_n_waves(l) * rocke_gemm_mn_block_bytes(l);
}

int rocke_gemm_mn_k_contrib(const rocke_gemm_mn_layout_t* l, int k)
{
    int off = 0;
    for(int j = 0; j < l->n_blk_bits; ++j)
        off += ((k >> l->blk_bits[j]) & 1) * (1 << j) * rocke_gemm_mn_block_bytes(l);
    for(int j = 0; j < l->n_row_bits; ++j)
        off += ((k >> l->row_bits[j]) & 1) * (1 << j) * 2 * l->W;
    return off;
}

int rocke_gemm_mn_mn_contrib(const rocke_gemm_mn_layout_t* l, int mn)
{
    return (mn / l->W) * rocke_gemm_mn_blocks_per_chunk(l) * rocke_gemm_mn_block_bytes(l)
           + (mn % l->W) * 2;
}

int rocke_gemm_mn_read_imm_bytes(const rocke_gemm_mn_layout_t* l, int idx, int r, int kk)
{
    int atom = l->warp_ext % l->W == 0 ? rocke_gemm_mn_mn_contrib(l, 16 * idx) : 32 * idx;
    return rocke_gemm_mn_k_contrib(l, 32 * kk + 4 * r) + atom;
}

bool rocke_gemm_mn_layout_letters(const char* layout, bool* a_mn, bool* b_mn)
{
    if(layout == nullptr || strlen(layout) != 3)
        return false;
    char c0 = (char)toupper((unsigned char)layout[0]);
    char c1 = (char)toupper((unsigned char)layout[1]);
    char c2 = (char)toupper((unsigned char)layout[2]);
    if((c0 != 'R' && c0 != 'C') || (c1 != 'R' && c1 != 'C') || c2 != 'R')
        return false;
    if(a_mn != nullptr)
        *a_mn = c0 == 'C';
    if(b_mn != nullptr)
        *b_mn = c1 == 'R';
    return true;
}

bool rocke_gemm_uses_mn_dtl(const rocke_gemm_universal_spec_t* spec)
{
    const rocke_gemm_trait_spec_t* tr = &spec->trait;
    return tr->direct_to_lds && tr->dtl_prefetch && tr->dtl_stages >= 2;
}

bool rocke_gemm_spec_mn_layouts(const rocke_gemm_universal_spec_t* spec,
                                rocke_gemm_mn_layout_t* la,
                                bool* has_a,
                                rocke_gemm_mn_layout_t* lb,
                                bool* has_b,
                                char* reason,
                                size_t cap)
{
    bool a_mn = false, b_mn = false;
    *has_a = *has_b = false;
    if(!rocke_gemm_mn_layout_letters(spec->data.layout, &a_mn, &b_mn))
        return reason_fmt(reason, cap, "unsupported GEMM layout '%s'", spec->data.layout);
    if(!rocke_gemm_uses_mn_dtl(spec))
        return true;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_gemm_trait_spec_t* tr = &spec->trait;
    int pad = tr->mn_pad_set ? tr->mn_pad : 32;
    if(a_mn)
    {
        bool sub = spec->_gett != nullptr && rocke_gett_hooks_a_sub_perm(spec->_gett);
        if(!rocke_gemm_mn_layout_init(la,
                                      t->tile_m,
                                      t->tile_k,
                                      spec->block_size,
                                      spec->wave_size,
                                      t->tile_m / t->warp_m,
                                      pad,
                                      tr->mn_chunk,
                                      sub,
                                      reason,
                                      cap))
            return false;
        *has_a = true;
    }
    if(b_mn)
    {
        /* sab_vec16 (an N-contiguous B with permuted reads) is Python-engine only */
        if(!rocke_gemm_mn_layout_init(lb,
                                      t->tile_n,
                                      t->tile_k,
                                      spec->block_size,
                                      spec->wave_size,
                                      t->tile_n / t->warp_n,
                                      pad,
                                      tr->mn_chunk,
                                      false,
                                      reason,
                                      cap))
            return false;
        *has_b = true;
    }
    return true;
}

bool rocke_gemm_mn_layout_error(const rocke_gemm_universal_spec_t* spec,
                                const char* arch,
                                char* reason,
                                size_t cap)
{
    bool a_mn = false, b_mn = false;
    if(!rocke_gemm_mn_layout_letters(spec->data.layout, &a_mn, &b_mn))
    {
        reason_fmt(reason, cap, "unsupported GEMM layout '%s'", spec->data.layout);
        return true;
    }
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_gemm_trait_spec_t* tr = &spec->trait;
    if(!(a_mn || b_mn))
    {
        if(tr->mn_ld_any)
        {
            reason_fmt(
                reason, cap, "mn_ld_any needs an MN-contiguous operand (layout RRR / CRR / CCR)");
            return true;
        }
        return false;
    }
    std::string why = std::string("layout ") + spec->data.layout;
    if(tr->mn_ld_any)
    {
        /* the shifted last tile (mn_ld_any) recomputes rows of the tile before
         * it: every output path must store, never accumulate */
        if(!rocke_gemm_uses_mn_dtl(spec))
        {
            reason_fmt(
                reason, cap, "%s: mn_ld_any is a DTL-ring lever (dtl_stages >= 2)", why.c_str());
            return true;
        }
        if(tr->split_k > 1 || tr->streamk)
        {
            reason_fmt(reason,
                       cap,
                       "%s: mn_ld_any shifts the last tile onto rows of the one before it "
                       "(split-K / stream-K would add those partials twice)",
                       why.c_str());
            return true;
        }
        if(spec->batched)
        {
            reason_fmt(reason, cap, "%s: mn_ld_any is built for the non-batched body", why.c_str());
            return true;
        }
    }
    if(!rocke_gemm_uses_mn_dtl(spec))
    {
        /* Python _mn_generic_error: the VGPR-staged MN operand loader */
        reason_fmt(reason,
                   cap,
                   "%s: the VGPR-staged MN operand path (non-ring bodies) is implemented by "
                   "the Python engine only (no C++ mirror yet)",
                   why.c_str());
        return true;
    }
    if(!streq(arch, "gfx950"))
    {
        reason_fmt(reason,
                   cap,
                   "%s: MN-contiguous operands need ds_read_b64_tr_b16 (('gfx950',))",
                   why.c_str());
        return true;
    }
    if(!streq(spec->data.dtype_a, "fp16") && !streq(spec->data.dtype_a, "bf16"))
    {
        reason_fmt(reason, cap, "%s: 16-bit A/B only", why.c_str());
        return true;
    }
    if(!(tr->direct_to_lds && tr->dtl_prefetch && tr->dtl_stages >= 2))
    {
        reason_fmt(reason,
                   cap,
                   "%s: needs the alias-scoped DTL ring (direct_to_lds, dtl_prefetch, "
                   "dtl_stages >= 2)",
                   why.c_str());
        return true;
    }
    if(t->warp_tile_m != 16 || t->warp_tile_n != 16 || t->warp_tile_k != 32)
    {
        reason_fmt(
            reason, cap, "%s: transpose fragments are built for the 16x16x32 atom", why.c_str());
        return true;
    }
    if(t->warp_k != 1)
    {
        reason_fmt(reason, cap, "%s: warp_k must be 1", why.c_str());
        return true;
    }
    /* (pingpong / asm_loop / asm_loop_rawrd: Python-engine levers, refused by
     * the C++ engine on every layout) */
    const char* blocked = nullptr;
    if(b_mn && tr->rilv_vec_epilogue)
        blocked = "rilv_vec_epilogue with an N-contiguous B";
    else if(tr->preshuffle_b)
        blocked = "preshuffle_b";
    else if(tr->streamk)
        blocked = "streamk";
    if(blocked != nullptr)
    {
        reason_fmt(reason, cap, "%s: %s is K-contiguous-only", why.c_str(), blocked);
        return true;
    }
    if(tr->lds_row_interleave)
    {
        if(a_mn && b_mn)
        {
            reason_fmt(reason,
                       cap,
                       "%s: lds_row_interleave with both operands MN-contiguous (register "
                       "spills)",
                       why.c_str());
            return true;
        }
        /* sched_strategy is a Python-engine lever: the C++ engine never has
         * 'iterative-minreg' */
        reason_fmt(reason,
                   cap,
                   "%s: lds_row_interleave with an MN-contiguous operand requires "
                   "sched_strategy='iterative-minreg'",
                   why.c_str());
        return true;
    }
    int pad = tr->mn_pad_set ? tr->mn_pad : 32;
    if(!(pad == 0 || pad == 16 || pad == 32 || pad == 64))
    {
        reason_fmt(reason, cap, "%s: mn_pad must be 0 / 16 / 32 / 64 (got %d)", why.c_str(), pad);
        return true;
    }
    {
        rocke_gemm_mn_layout_t la, lb;
        bool ha = false, hb = false;
        char r2[256] = {0};
        if(!rocke_gemm_spec_mn_layouts(spec, &la, &ha, &lb, &hb, r2, sizeof r2))
        {
            reason_fmt(reason, cap, "%s: %s", why.c_str(), r2);
            return true;
        }
    }
    /* rows past the MN extent / K read past the operand's end on the last
     * tile: only the tight DTL resource bounds them (dtl_tight_ragged_only is
     * Python-engine only) */
    if(tr->pad_k && !tr->dtl_tight_rsrc)
    {
        reason_fmt(reason,
                   cap,
                   "%s: pad_k with an MN-contiguous operand needs dtl_tight_rsrc on every "
                   "operand",
                   why.c_str());
        return true;
    }
    if(a_mn && tr->pad_m && !tr->dtl_tight_rsrc)
    {
        reason_fmt(
            reason, cap, "%s: pad_m with an M-contiguous A needs dtl_tight_rsrc", why.c_str());
        return true;
    }
    if(b_mn && tr->pad_n && !tr->dtl_tight_rsrc)
    {
        reason_fmt(
            reason, cap, "%s: pad_n with an N-contiguous B needs dtl_tight_rsrc", why.c_str());
        return true;
    }
    return false;
}

int rocke_gemm_mn_lds_extra_bytes(const rocke_gemm_universal_spec_t* spec)
{
    rocke_gemm_mn_layout_t la, lb;
    bool ha = false, hb = false;
    if(!rocke_gemm_spec_mn_layouts(spec, &la, &ha, &lb, &hb, nullptr, 0))
        return 0;
    int S = spec->trait.dtl_stages;
    int out = 0;
    if(ha)
        out += rocke_gemm_mn_n_blocks(&la) * la.pad * S;
    if(hb)
        out += rocke_gemm_mn_n_blocks(&lb) * lb.pad * S;
    return out;
}

/* ----------------------------------------------------------------- IR helpers */

void rocke_gemm_mn_emit_dtl_voffs(rocke_ir_builder_t* b,
                                  const rocke_gemm_mn_layout_t* lay,
                                  rocke_value_t* tid,
                                  rocke_value_t* ld,
                                  rocke_value_t** out)
{
    V* ws = ci(b, lay->wave_size);
    V* w = rocke_b_div(b, tid, ws);
    V* lane = rocke_b_mod(b, tid, ws);
    V* c16 = ci(b, 16);
    V* lb = rocke_b_mul(b, lane, c16);
    V* c2w = ci(b, 2 * lay->W);
    V* ri = rocke_b_div(b, lb, c2w);
    V* c2w2 = ci(b, 2 * lay->W);
    V* md = rocke_b_mod(b, lb, c2w2);
    V* c2 = ci(b, 2);
    V* mnin = rocke_b_div(b, md, c2);
    /* K contribution of the in-block row (lane only) */
    V* k_ri = nullptr;
    for(int j = 0; j < lay->n_row_bits; ++j)
    {
        int bt = lay->row_bits[j];
        V* bit;
        if(j)
        {
            V* cj = ci(b, 1 << j);
            V* dv = rocke_b_div(b, ri, cj);
            V* c2b = ci(b, 2);
            bit = rocke_b_mod(b, dv, c2b);
        }
        else
        {
            V* c2b = ci(b, 2);
            bit = rocke_b_mod(b, ri, c2b);
        }
        V* term = bit;
        if(bt)
        {
            V* cb = ci(b, 1 << bt);
            term = rocke_b_mul(b, bit, cb);
        }
        k_ri = k_ri == nullptr ? term : rocke_b_add(b, k_ri, term);
    }
    int np = rocke_gemm_mn_n_pass(lay);
    for(int p = 0; p < np; ++p)
    {
        V* beta = w;
        if(p)
        {
            V* cp = ci(b, p * rocke_gemm_mn_n_waves(lay));
            beta = rocke_b_add(b, w, cp);
        }
        V* cb1 = ci(b, rocke_gemm_mn_blocks_per_chunk(lay));
        V* ch = rocke_b_div(b, beta, cb1);
        V* cb2 = ci(b, rocke_gemm_mn_blocks_per_chunk(lay));
        V* bb = rocke_b_mod(b, beta, cb2);
        V* k = k_ri;
        for(int j = 0; j < lay->n_blk_bits; ++j)
        {
            int bt = lay->blk_bits[j];
            V* bit;
            if(j)
            {
                V* cj = ci(b, 1 << j);
                V* dv = rocke_b_div(b, bb, cj);
                V* c2b = ci(b, 2);
                bit = rocke_b_mod(b, dv, c2b);
            }
            else
            {
                V* c2b = ci(b, 2);
                bit = rocke_b_mod(b, bb, c2b);
            }
            V* term = bit;
            if(bt)
            {
                V* cb = ci(b, 1 << bt);
                term = rocke_b_mul(b, bit, cb);
            }
            k = k == nullptr ? term : rocke_b_add(b, k, term);
        }
        V* cw = ci(b, lay->W);
        V* chw = rocke_b_mul(b, ch, cw);
        V* mn = rocke_b_add(b, chw, mnin);
        if(k != nullptr)
        {
            V* kl = rocke_b_mul(b, k, ld);
            out[p] = rocke_b_add(b, kl, mn);
        }
        else
            out[p] = mn;
    }
}

void rocke_gemm_mn_emit_dtl_lane_k_mn(rocke_ir_builder_t* b,
                                      const rocke_gemm_mn_layout_t* lay,
                                      rocke_value_t* tid,
                                      rocke_value_t** k_out,
                                      rocke_value_t** mn_out)
{
    V* ws = ci(b, lay->wave_size);
    V* w = rocke_b_div(b, tid, ws);
    V* lane = rocke_b_mod(b, tid, ws);
    V* c16 = ci(b, 16);
    V* lb = rocke_b_mul(b, lane, c16);
    V* c2w = ci(b, 2 * lay->W);
    V* ri = rocke_b_div(b, lb, c2w);
    V* c2w2 = ci(b, 2 * lay->W);
    V* md = rocke_b_mod(b, lb, c2w2);
    V* c2 = ci(b, 2);
    V* mnin = rocke_b_div(b, md, c2);
    V* k_ri = nullptr;
    for(int j = 0; j < lay->n_row_bits; ++j)
    {
        int bt = lay->row_bits[j];
        V* cj = ci(b, 1 << j);
        V* dv = rocke_b_div(b, ri, cj);
        V* c2b = ci(b, 2);
        V* bit = rocke_b_mod(b, dv, c2b);
        V* cb = ci(b, 1 << bt);
        V* term = rocke_b_mul(b, bit, cb);
        k_ri = k_ri == nullptr ? term : rocke_b_add(b, k_ri, term);
    }
    int np = rocke_gemm_mn_n_pass(lay);
    for(int p = 0; p < np; ++p)
    {
        V* cp = ci(b, p * rocke_gemm_mn_n_waves(lay));
        V* beta = rocke_b_add(b, w, cp);
        V* cb1 = ci(b, rocke_gemm_mn_blocks_per_chunk(lay));
        V* ch = rocke_b_div(b, beta, cb1);
        V* cb2 = ci(b, rocke_gemm_mn_blocks_per_chunk(lay));
        V* bb = rocke_b_mod(b, beta, cb2);
        V* k = k_ri;
        for(int j = 0; j < lay->n_blk_bits; ++j)
        {
            int bt = lay->blk_bits[j];
            V* cj = ci(b, 1 << j);
            V* dv = rocke_b_div(b, bb, cj);
            V* c2b = ci(b, 2);
            V* bit = rocke_b_mod(b, dv, c2b);
            V* cb = ci(b, 1 << bt);
            V* term = rocke_b_mul(b, bit, cb);
            k = k == nullptr ? term : rocke_b_add(b, k, term);
        }
        if(k == nullptr)
            k = ci(b, 0);
        V* cw = ci(b, lay->W);
        V* chw = rocke_b_mul(b, ch, cw);
        V* mn = rocke_b_add(b, chw, mnin);
        k_out[p] = k;
        mn_out[p] = mn;
    }
}

rocke_value_t* rocke_gemm_mn_emit_read_lane_elems(rocke_ir_builder_t* b,
                                                  const rocke_gemm_mn_layout_t* lay,
                                                  rocke_value_t* lane,
                                                  rocke_value_t* warp_idx)
{
    V* c4 = ci(b, 4);
    V* c16 = ci(b, 16);
    V* m16 = rocke_b_mod(b, lane, c16);
    V* g = rocke_b_div(b, m16, c4);
    V* d16 = rocke_b_div(b, lane, c16);
    V* c2 = ci(b, 2);
    V* a = rocke_b_mod(b, d16, c2);
    V* c32 = ci(b, 32);
    V* c = rocke_b_div(b, lane, c32);
    V* sub = rocke_b_mod(b, lane, c4);
    if(lay->sub_perm)
    {
        /* QUAD_PERM[s] = 2 * (s % 2) + s // 2 */
        V* c2a = ci(b, 2);
        V* sm = rocke_b_mod(b, sub, c2a);
        V* c2b = ci(b, 2);
        V* smm = rocke_b_mul(b, sm, c2b);
        V* c2c = ci(b, 2);
        V* sd = rocke_b_div(b, sub, c2c);
        sub = rocke_b_add(b, smm, sd);
    }
    std::vector<V*> terms;
    {
        /* g = K bits 0 and 1 */
        V* c2d = ci(b, 2);
        V* g0 = rocke_b_mod(b, g, c2d);
        V* c2e = ci(b, 2);
        V* g1 = rocke_b_div(b, g, c2e);
        V* k1 = ci(b, rocke_gemm_mn_k_contrib(lay, 1) / 2);
        terms.push_back(rocke_b_mul(b, g0, k1));
        V* k2 = ci(b, rocke_gemm_mn_k_contrib(lay, 2) / 2);
        terms.push_back(rocke_b_mul(b, g1, k2));
    }
    {
        V* k8 = ci(b, rocke_gemm_mn_k_contrib(lay, 1 << 3) / 2);
        terms.push_back(rocke_b_mul(b, a, k8));
    }
    {
        V* k16 = ci(b, rocke_gemm_mn_k_contrib(lay, 1 << 4) / 2);
        terms.push_back(rocke_b_mul(b, c, k16));
    }
    terms.push_back(rocke_b_mul(b, sub, c4));
    /* wave MN origin */
    if(lay->warp_ext % lay->W == 0)
    {
        V* cw = ci(b, rocke_gemm_mn_mn_contrib(lay, lay->warp_ext) / 2);
        terms.push_back(rocke_b_mul(b, warp_idx, cw));
    }
    else
    {
        V* ce = ci(b, lay->warp_ext);
        V* mn0 = rocke_b_mul(b, warp_idx, ce);
        V* cw = ci(b, lay->W);
        V* ch = rocke_b_div(b, mn0, cw);
        V* cb = ci(b, rocke_gemm_mn_blocks_per_chunk(lay) * rocke_gemm_mn_block_bytes(lay) / 2);
        terms.push_back(rocke_b_mul(b, ch, cb));
        V* cw2 = ci(b, lay->W);
        terms.push_back(rocke_b_mod(b, mn0, cw2));
    }
    V* out = terms[0];
    for(size_t i = 1; i < terms.size(); ++i)
        out = rocke_b_add(b, out, terms[i]);
    return out;
}

rocke_value_t* rocke_gemm_mn_emit_tr_fragment(rocke_ir_builder_t* b,
                                              rocke_value_t* smem,
                                              const rocke_gemm_mn_layout_t* lay,
                                              rocke_value_t* base_elems,
                                              int idx,
                                              int kk,
                                              const rocke_type_t* dtype,
                                              int sub_elems)
{
    V* parts[2];
    for(int r = 0; r < 2; ++r)
    {
        int imm = rocke_gemm_mn_read_imm_bytes(lay, idx, r, kk) / 2 - sub_elems;
        V* addr = base_elems;
        if(imm)
        {
            V* cimm = ci(b, imm);
            addr = rocke_b_add(b, base_elems, cimm);
        }
        V* z = ci(b, 0);
        V* idx2[2] = {z, addr};
        parts[r] = rocke_b_ds_read_tr16_b64(b, smem, idx2, 2, dtype);
    }
    return rocke_b_vec_concat(b, parts[0], parts[1]);
}

} /* extern "C" */
