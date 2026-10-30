/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C++ GEMM dispatcher (rocke/gemm_dispatch.h): the validity and launch rules of the universal
 * GEMM body on the full spec mirror -- instances/common/gemm_universal.py (is_valid_spec,
 * launch_shape_error, _g1_launch_error, sia_table, the LDS plan), gemm_asm_loop.py (the
 * hand-ordered K-tile tables), gemm_mn_operand.py (MN-contiguous operands) and
 * dispatch/gemm/support.py (the dispatcher's support predicates).
 */
#include "gemm_dispatch_internal.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace rocke
{
namespace gemm_dispatch
{
namespace detail
{

namespace
{

bool is_gemm_layout(const std::string& lay)
{
    return lay == "RCR" || lay == "RRR" || lay == "CRR" || lay == "CCR";
}

std::string upper(std::string s)
{
    for(auto& c : s)
        c = (char)toupper((unsigned char)c);
    return s;
}

bool ilog2_exact(int64_t x, int64_t* out)
{
    if(x <= 0 || (x & (x - 1)))
        return false;
    int64_t b = 0;
    while((int64_t(1) << b) < x)
        ++b;
    *out = b;
    return true;
}

} // namespace

/* ------------------------------------------------------------------ load vectors */

bool choose_load_vec(
    int64_t tile_m, int64_t tile_n, int64_t tile_k, int64_t block_size, int64_t* out)
{
    int64_t threads = block_size;
    int64_t v = 8;
    while(v >= 1)
    {
        if(tile_k % v)
        {
            v /= 2;
            continue;
        }
        int64_t a_vecs = (tile_m * tile_k) / v;
        int64_t b_vecs = (tile_n * tile_k) / v;
        if(threads == 0 || a_vecs < threads || b_vecs < threads || a_vecs % threads
           || b_vecs % threads)
        {
            v /= 2;
            continue;
        }
        *out = v;
        return true;
    }
    return false;
}

/* --------------------------------------------------------------------- LDS plan */

bool ab_double_buffer_decision(const UGemm& s, int64_t lds_cap)
{
    const Trait& tr = s.trait;
    if(tr.pipeline != "compv4" || tr.direct_to_lds)
        return false;
    if(tr.ab_double_buffer.has_value())
        return *tr.ab_double_buffer;
    int64_t ab_single = ((s.tile.tile_m * s.tile.tile_k) + (s.tile.tile_n * s.tile.tile_k)) * 2;
    bool db_fits_2wg = (2 * ab_single) * 2 <= lds_cap;
    return tr.epilogue != "cshuffle" && db_fits_2wg;
}

void universal_lds_bytes(const UGemm& s,
                         int64_t lds_cap,
                         int64_t eb_a,
                         int64_t eb_b,
                         int64_t eb_c,
                         int64_t* total,
                         int64_t* ab_out,
                         int64_t* c_out)
{
    const Trait& tr = s.trait;
    const Tile& t = s.tile;
    int64_t pad = tr.direct_to_lds ? 0 : tr.lds_k_pad;
    int64_t ab_single = (t.tile_m * eb_a + t.tile_n * eb_b) * (t.tile_k + pad);
    bool db = ab_double_buffer_decision(s, lds_cap);
    int64_t ab = ab_single * ((tr.dtl_prefetch || db) ? 2 : 1);
    if(tr.direct_to_lds && tr.dtl_prefetch && tr.dtl_stages >= 2)
        ab = ab_single * tr.dtl_stages;
    int64_t c = tr.epilogue == "cshuffle" ? t.tile_m * t.tile_n * eb_c : 0;
    *total = tr.cshuffle_no_alias ? ab + c : std::max(ab, c);
    *ab_out = ab;
    *c_out = c;
}

namespace
{

/* gemm_universal.lds_swizzle_params */
void lds_swizzle_params(
    int64_t a_per_lane, int64_t b_per_lane, int64_t block_k, int64_t* R, int64_t* W, int64_t* L)
{
    int64_t elem = a_per_lane == b_per_lane ? a_per_lane : 0;
    int64_t slots = (elem && block_k % elem == 0) ? block_k / elem : 0;
    int64_t al, aw;
    bool okl = ilog2_exact(elem, &al);
    bool okw = ilog2_exact(slots, &aw);
    if(okl && okw && aw >= 1)
    {
        *R = 0;
        *W = aw;
        *L = al;
        return;
    }
    *R = 3;
    *W = 1;
    *L = 4;
}

/* gemm_universal._spec_swizzle_params: lds_swizzle_params with the trait
 * overrides lds_swizzle_r and lds_swizzle_l (W = log2(tile_k >> L)) */
void spec_swizzle_params(const Trait& tr,
                         int64_t a_per_lane,
                         int64_t b_per_lane,
                         int64_t block_k,
                         int64_t* R,
                         int64_t* W,
                         int64_t* L)
{
    lds_swizzle_params(a_per_lane, b_per_lane, block_k, R, W, L);
    if(tr.lds_swizzle_r >= 0)
        *R = tr.lds_swizzle_r;
    if(tr.lds_swizzle_l >= 0)
    {
        *L = tr.lds_swizzle_l;
        int64_t w = 0;
        if(!ilog2_exact(block_k >> *L, &w))
            w = 0;
        *W = w;
    }
}

} // namespace

/* ------------------------------------------------------------- gemm_mn_operand */

bool mn_layout_letters_x(const std::string& layout, bool* a_mn, bool* b_mn)
{
    std::string lay = upper(layout);
    if(lay.size() != 3 || (lay[0] != 'R' && lay[0] != 'C') || (lay[1] != 'R' && lay[1] != 'C')
       || lay[2] != 'R')
        return false;
    *a_mn = lay[0] == 'C';
    *b_mn = lay[1] == 'R';
    return true;
}

bool MnLayout::conflict_free() const
{
    return blk_bits.size() >= 3 && blk_bits[0] == 0 && blk_bits[1] == 1 && blk_bits[2] == 3;
}

namespace
{

/* gemm_mn_operand.default_mn_chunk (false: ValueError) */
bool default_mn_chunk(int64_t tile_mn, int64_t tile_k, int64_t warp_ext, int64_t* out)
{
    if(tile_mn > 128)
    {
        int64_t w = std::max({warp_ext, (int64_t)1024 * 4 / tile_k, (int64_t)16});
        if(w <= 256 && (w & (w - 1)) == 0 && tile_mn % w == 0
           && (warp_ext % w == 0 || w % warp_ext == 0))
        {
            *out = w;
            return true;
        }
    }
    int64_t best = 0;
    int64_t w = 16;
    while(w <= std::min(tile_mn, (int64_t)256))
    {
        bool ok = tile_mn % w == 0 && (warp_ext % w == 0 || w % warp_ext == 0);
        if(ok && tile_k * w * 2 >= 1024)
            best = w;
        w *= 2;
    }
    if(!best)
        return false;
    *out = best;
    return true;
}

} // namespace

bool make_mn_layout(int64_t tile_mn,
                    int64_t tile_k,
                    int64_t block_size,
                    int64_t wave_size,
                    int64_t warp_ext,
                    int64_t pad,
                    int64_t chunk,
                    bool sub_perm,
                    MnLayout* out,
                    std::string* why)
{
    MnLayout l;
    l.tile_mn = tile_mn;
    l.tile_k = tile_k;
    l.block_size = block_size;
    l.wave_size = wave_size;
    l.warp_ext = warp_ext;
    l.pad = pad;
    l.chunk = chunk;
    l.sub_perm = sub_perm;
    int64_t W = chunk;
    if(!W && !default_mn_chunk(tile_mn, tile_k, warp_ext, &W))
        return set_why(why, "no MN chunk width");
    l.W = W;
    if(wave_size != 64)
        return set_why(why, "MN operand DTL is built for wave64 (1 KiB per wave)");
    if(tile_k < 32)
        return set_why(why, "MN operand reader needs tile_k >= 32 (16x16x32 atoms)");
    if(W == 0 || tile_mn % W || 2 * W > 1024 || W < 16)
        return set_why(why, "MN chunk does not tile tile_mn");
    if(warp_ext == 0 || !(warp_ext % W == 0 || W % warp_ext == 0))
        return set_why(why, "MN chunk vs wave extent");
    int64_t R = 1024 / (2 * W);
    l.R = R;
    int64_t tk, wb, rb;
    if(!ilog2_exact(tile_k, &tk))
        return set_why(why, "tile_k is not a power of two");
    if(!ilog2_exact(W, &wb))
        return set_why(why, "MN chunk is not a power of two");
    std::vector<int64_t> low;
    for(int64_t bt : {0, 1, 3})
        if(bt < tk)
            low.push_back(bt);
    std::vector<int64_t> rest;
    for(int64_t bt = 0; bt < tk; ++bt)
        if(std::find(low.begin(), low.end(), bt) == low.end())
            rest.push_back(bt);
    if(!ilog2_exact(R, &rb))
        return set_why(why, "rows per block is not a power of two");
    std::vector<int64_t> row(rest.begin(), rest.begin() + std::min((size_t)rb, rest.size()));
    rest.erase(rest.begin(), rest.begin() + std::min((size_t)rb, rest.size()));
    while((int64_t)row.size() < rb)
    {
        if(low.empty())
            return set_why(why, "pop from empty list");
        row.push_back(low.back());
        low.pop_back();
    }
    l.row_bits = row;
    l.blk_bits = low;
    l.blk_bits.insert(l.blk_bits.end(), rest.begin(), rest.end());
    if(l.n_waves() == 0 || l.n_blocks() % l.n_waves())
        return set_why(why, "LDS blocks are not whole DTL passes");
    *out = l;
    return true;
}

bool uses_mn_dtl(const UGemm& s)
{
    return s.trait.direct_to_lds && s.trait.dtl_prefetch && s.trait.dtl_stages >= 2;
}

bool spec_mn_layouts(const UGemm& s,
                     bool a_sub_perm,
                     bool* has_a,
                     MnLayout* la,
                     bool* has_b,
                     MnLayout* lb,
                     std::string* why)
{
    bool a_mn, b_mn;
    if(!mn_layout_letters_x(s.data.layout, &a_mn, &b_mn))
        return set_why(why, "unsupported GEMM layout '" + s.data.layout + "'");
    *has_a = false;
    *has_b = false;
    if(!uses_mn_dtl(s))
        return true;
    const Tile& t = s.tile;
    if(a_mn)
    {
        if(t.warp_m == 0)
            return set_why(why, "warp_m 0");
        if(!make_mn_layout(t.tile_m,
                           t.tile_k,
                           s.block_size,
                           s.wave_size,
                           t.tile_m / t.warp_m,
                           s.trait.mn_pad,
                           s.trait.mn_chunk,
                           a_sub_perm,
                           la,
                           why))
            return false;
        *has_a = true;
    }
    if(b_mn)
    {
        if(t.warp_n == 0)
            return set_why(why, "warp_n 0");
        if(!make_mn_layout(t.tile_n,
                           t.tile_k,
                           s.block_size,
                           s.wave_size,
                           t.tile_n / t.warp_n,
                           s.trait.mn_pad,
                           s.trait.mn_chunk,
                           s.trait.sab_vec16,
                           lb,
                           why))
            return false;
        *has_b = true;
    }
    return true;
}

namespace
{

/* gemm_mn_operand._vp_mn_error */
std::string vp_mn_error(
    const UGemm& s, bool a_mn, bool b_mn, const std::string& why0, int64_t v, int64_t kv)
{
    const Tile& t = s.tile;
    const Trait& tr = s.trait;
    if(kv != 2 && kv != 4 && kv != 8)
        return why0 + ": vp_mn_kv must be 2 / 4 / 8";
    if(v != 2 && v != 4 && v != 8)
        return why0 + ": vp_mn_kv needs mn_vec 2 / 4 / 8";
    if(tr.vp_mn_order < 0 || tr.vp_mn_order > 3)
        return why0 + ": vp_mn_order must be 0..3";
    if(tr.vp_mn_worder < 0 || tr.vp_mn_worder > 3)
        return why0 + ": vp_mn_worder must be 0..3";
    if(tr.vp_mn_pipe != 0 && tr.vp_mn_pipe != 1)
        return why0 + ": vp_mn_pipe must be 0 / 1";
    if(tr.vp_mn_wrap != 0 && tr.vp_mn_wrap != 1)
        return why0 + ": vp_mn_wrap must be 0 / 1";
    bool mns[2] = {a_mn, b_mn};
    int64_t tiles[2] = {t.tile_m, t.tile_n};
    for(int i = 0; i < 2; ++i)
    {
        if(!mns[i])
            continue;
        int64_t tile_mn = tiles[i];
        if(tile_mn % v || t.tile_k % kv)
            return why0 + ": tile is not a whole number of micro-blocks";
        if(((tile_mn / v) * (t.tile_k / kv)) % s.block_size && !tr.vp_mn_wrap)
            return why0 + ": tile is not a whole number of micro-blocks per thread";
        int64_t q = tr.vp_mn_order == 2 ? 4 : (tr.vp_mn_order == 3 ? 2 : 1);
        if((tile_mn / v) % q)
            return why0 + ": vp_mn_order groups";
    }
    return "";
}

/* gemm_mn_operand._mn_generic_error */
std::string mn_generic_error(
    const UGemm& s, const std::string& arch, bool a_mn, bool b_mn, const std::string& why0)
{
    const Tile& t = s.tile;
    const Trait& tr = s.trait;
    if(arch != "gfx942" && arch != "gfx950")
        return why0 + ": the VGPR-staged MN operand path is built for ('gfx942', 'gfx950')";
    if(s.data.dtype_a != "fp16" && s.data.dtype_a != "bf16")
        return why0 + ": 16-bit A/B only";
    if(tr.direct_to_lds)
        return why0 + ": direct_to_lds without the alias-scoped ring is K-contiguous-only";
    if(tr.preshuffle_b)
        return why0 + ": preshuffle_b is K-contiguous-only";
    if(tr.streamk)
        return why0 + ": streamk is K-contiguous-only";
    if(tr.pipeline == "wsp3")
        return why0 + ": pipeline wsp3 is K-contiguous-only";
    if(tr.vgpr_prefetch > 0 && !tr.vp_mn_kv)
        return why0 + ": vgpr_prefetch without vp_mn_kv is K-contiguous-only";
    int64_t v = tr.mn_vec;
    if(v != 1 && v != 2 && v != 4 && v != 8)
        return why0 + ": mn_vec must be 1 / 2 / 4 / 8";
    if(tr.vp_mn_kv)
        return vp_mn_error(s, a_mn, b_mn, why0, v, tr.vp_mn_kv);
    bool mns[2] = {a_mn, b_mn};
    int64_t tiles[2] = {t.tile_m, t.tile_n};
    for(int i = 0; i < 2; ++i)
    {
        if(!mns[i])
            continue;
        int64_t tile_mn = tiles[i];
        if(tile_mn % v || s.block_size == 0 || (tile_mn * t.tile_k / v) % s.block_size)
            return why0 + ": tile is not a whole number of loads per thread";
    }
    return "";
}

/* gemm_mn_operand.mn_layout_error ("" = None) */
std::string mn_layout_error(const UGemm& s, const std::string& arch, bool a_sub_perm)
{
    bool a_mn, b_mn;
    if(!mn_layout_letters_x(s.data.layout, &a_mn, &b_mn))
        return "unsupported GEMM layout '" + s.data.layout + "'";
    const Tile& t = s.tile;
    const Trait& tr = s.trait;
    if(!(a_mn || b_mn))
    {
        if(tr.mn_ld_any)
            return "mn_ld_any needs an MN-contiguous operand (layout RRR / CRR / CCR)";
        return "";
    }
    std::string why = "layout " + s.data.layout;
    if(tr.mn_ld_any)
    {
        if(!uses_mn_dtl(s))
            return why + ": mn_ld_any is a DTL-ring lever (dtl_stages >= 2)";
        if(tr.split_k > 1 || tr.split_k_serial || tr.streamk)
            return why + ": mn_ld_any shifts the last tile onto rows of the one before it";
        if(s.batched)
            return why + ": mn_ld_any is built for the non-batched body";
    }
    if(!uses_mn_dtl(s))
        return mn_generic_error(s, arch, a_mn, b_mn, why);
    if(arch != "gfx950")
        return why + ": MN-contiguous operands need ds_read_b64_tr_b16";
    if(s.data.dtype_a != "fp16" && s.data.dtype_a != "bf16")
        return why + ": 16-bit A/B only";
    if(!(tr.direct_to_lds && tr.dtl_prefetch && tr.dtl_stages >= 2))
        return why + ": needs the alias-scoped DTL ring";
    if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16 && t.warp_tile_k == 32))
        return why + ": transpose fragments are built for the 16x16x32 atom";
    if(t.warp_k != 1)
        return why + ": warp_k must be 1";
    if(tr.pingpong > 0)
        return why + ": pingpong is K-contiguous-only";
    if(!tr.asm_loop.empty())
        return why + ": asm_loop is K-contiguous-only";
    if(tr.asm_loop_rawrd)
        return why + ": asm_loop_rawrd is K-contiguous-only";
    if(b_mn && tr.rilv_vec_epilogue)
        return why + ": rilv_vec_epilogue with an N-contiguous B is K-contiguous-only";
    if(tr.preshuffle_b)
        return why + ": preshuffle_b is K-contiguous-only";
    if(tr.streamk)
        return why + ": streamk is K-contiguous-only";
    if(tr.lds_row_interleave)
    {
        if(a_mn && b_mn)
            return why + ": lds_row_interleave with both operands MN-contiguous (register spills)";
        if(tr.sched_strategy != "iterative-minreg")
            return why
                   + ": lds_row_interleave with an MN-contiguous operand requires "
                     "sched_strategy='iterative-minreg'";
    }
    if(!(tr.mn_pad == 0 || tr.mn_pad == 16 || tr.mn_pad == 32 || tr.mn_pad == 64))
        return why + ": mn_pad must be 0 / 16 / 32 / 64";
    bool ha, hb;
    MnLayout la, lb;
    std::string w;
    if(!spec_mn_layouts(s, a_sub_perm, &ha, &la, &hb, &lb, &w))
        return why + ": " + w;
    if(tr.pad_k && !(tr.dtl_tight_rsrc && !tr.dtl_tight_ragged_only))
        return why + ": pad_k with an MN-contiguous operand needs dtl_tight_rsrc on every operand";
    if(a_mn && tr.pad_m && !tr.dtl_tight_rsrc)
        return why + ": pad_m with an M-contiguous A needs dtl_tight_rsrc";
    if(b_mn && tr.pad_n && !tr.dtl_tight_rsrc)
        return why + ": pad_n with an N-contiguous B needs dtl_tight_rsrc";
    return "";
}

/* gemm_mn_operand.mn_launch_error ("" = None) */
std::string mn_launch_error(const UGemm& s, int64_t M, int64_t N, int64_t K)
{
    (void)K;
    bool a_mn, b_mn;
    if(!mn_layout_letters_x(s.data.layout, &a_mn, &b_mn))
        return "unsupported GEMM layout '" + s.data.layout + "'";
    bool dtl = uses_mn_dtl(s);
    int64_t v = dtl ? 8 : s.trait.mn_vec;
    bool any_ld = dtl && s.trait.mn_ld_any;
    bool mns[2] = {a_mn, b_mn};
    int64_t lds[2] = {M, N};
    int64_t tiles[2] = {s.tile.tile_m, s.tile.tile_n};
    for(int i = 0; i < 2; ++i)
    {
        if(mns[i] && any_ld)
        {
            if(lds[i] < tiles[i])
                return "layout " + s.data.layout + ": mn_ld_any leading dimension below the tile";
            continue;
        }
        if(mns[i] && (v == 0 || pmod(lds[i], v)))
            return "layout " + s.data.layout + ": MN-contiguous leading dimension not a multiple";
    }
    return "";
}

/* gemm_mn_operand.mn_lds_extra_bytes */
int64_t mn_lds_extra_bytes(const UGemm& s, bool a_sub_perm)
{
    bool ha, hb;
    MnLayout la, lb;
    std::string w;
    if(!spec_mn_layouts(s, a_sub_perm, &ha, &la, &hb, &lb, &w))
        return 0;
    int64_t S = s.trait.dtl_stages;
    int64_t out = 0;
    if(ha)
        out += la.n_blocks() * la.pad * S;
    if(hb)
        out += lb.n_blocks() * lb.pad * S;
    return out;
}

/* ----------------------------------------------------------------- hb / sia */

int64_t hb_k_atoms(const UGemm& s)
{
    int64_t ka = 0;
    s.tile.k_atoms(&ka);
    return std::max((int64_t)1, ka);
}

int64_t hb_last_read_slot(const UGemm& s)
{
    int64_t mm = 0, mn = 0;
    s.tile.mfmas_m(&mm);
    s.tile.mfmas_n(&mn);
    int64_t n = mm * mn;
    int64_t ka = hb_k_atoms(s);
    if(ka < 2)
        return -1;
    int64_t nrd = mm + mn;
    return (ka - 2) * n + std::min(n - 1, (nrd - 1) * std::max((int64_t)1, s.trait.hb_read_every));
}

void hb_slots(const UGemm& s, int64_t* rel, int64_t* acq)
{
    int64_t mm = 0, mn = 0;
    s.tile.mfmas_m(&mm);
    s.tile.mfmas_n(&mn);
    int64_t n = mm * mn;
    int64_t ka = hb_k_atoms(s);
    int64_t ntot = ka * n;
    const Trait& tr = s.trait;
    *rel = tr.hb_rel_slot >= 0
               ? tr.hb_rel_slot
               : std::min(ntot - 1, hb_last_read_slot(s) + std::max((int64_t)1, tr.hb_rel_gap));
    *acq = tr.hb_acq_slot >= 0 ? tr.hb_acq_slot : std::min(ntot - 1, (ka - 1) * n + 3);
}

/* gemm_universal.sia_table: "" = ok, else the ValueError text */
std::string sia_table_error(const UGemm& s)
{
    const Tile& t = s.tile;
    const Trait& tr = s.trait;
    int64_t mm = 0, mn = 0;
    if(!t.mfmas_m(&mm) || !t.mfmas_n(&mn))
        return "tile not divisible";
    int64_t n = mm * mn;
    int64_t ntot = 2 * n;
    int64_t e = std::max((int64_t)1, tr.sia_read_every);
    int64_t bs = std::max((int64_t)1, s.block_size);
    std::map<char, int64_t> cnt = {{'A', mm}, {'B', mn}};
    std::map<char, int64_t> npass
        = {{'A', cdiv(t.tile_m * t.tile_k / 8, bs)}, {'B', cdiv(t.tile_n * t.tile_k / 8, bs)}};
    char X = tr.sia_b_first ? 'B' : 'A';
    char Y = tr.sia_b_first ? 'A' : 'B';
    if(tr.sia_tensile)
    {
        if(!(mm == 8 && mn == 8 && npass['A'] == 8 && npass['B'] == 8))
            return "sia_tensile needs 8x8 MFMAs per atom and 8 + 8 DTL issues";
        return "";
    }
    int64_t gap = std::max((int64_t)1, tr.sia_rel_gap);
    struct Rd
    {
        int64_t slot;
        char op;
        int64_t idx;
    };
    std::vector<std::pair<char, int64_t>> useq;
    if(tr.sia_use_order)
    {
        useq.push_back({'A', 0});
        for(int64_t j = 0; j < mn; ++j)
            useq.push_back({'B', j});
        for(int64_t j = 1; j < mm; ++j)
            useq.push_back({'A', j});
    }
    std::vector<Rd> rd1;
    for(int64_t j = 0; j < cnt[X]; ++j)
        rd1.push_back({j * e, X, j});
    if(rd1.empty())
        return "list index out of range";
    int64_t last_x = rd1.back().slot;
    int64_t rel_x = 0, y0;
    if(tr.sia_split_rel)
    {
        rel_x = last_x + gap;
        y0 = rel_x + 1;
    }
    else
        y0 = last_x + e;
    for(int64_t j = 0; j < cnt[Y]; ++j)
        rd1.push_back({y0 + j * e, Y, j});
    if(tr.sia_use_order)
    {
        rd1.clear();
        for(size_t j = 0; j < useq.size(); ++j)
            rd1.push_back({(int64_t)j * e, useq[j].first, useq[j].second});
    }
    int64_t last_y = rd1.back().slot;
    int64_t rel_y = last_y + gap;
    std::vector<std::pair<int64_t, std::string>> rel;
    if(tr.sia_split_rel)
        rel = {{rel_x, std::string(1, X)}, {rel_y, std::string(1, Y)}};
    else
        rel = {{rel_y, std::string(1, X) + Y}};
    int64_t acq_x = tr.sia_plr ? n + tr.sia_acq_gap : n + fdiv(n, 2);
    if(tr.sia_merge_sync)
    {
        acq_x = std::max(acq_x, last_y + 1);
        rel.clear();
    }
    else if(acq_x <= rel_y)
        acq_x = rel_y + 1;
    std::vector<Rd> rd0;
    std::vector<std::pair<int64_t, std::string>> acq;
    int64_t nread0 = cnt[X] + cnt[Y];
    auto fit = [&](int64_t start, int64_t k) {
        int64_t room = ntot - start;
        return std::max((int64_t)1, std::min(e, fdiv(room, std::max((int64_t)1, k))));
    };
    if(tr.sia_split_acq)
    {
        acq.push_back({acq_x, std::string(1, X)});
        int64_t ex = fit(acq_x + 1, nread0);
        for(int64_t j = 0; j < cnt[X]; ++j)
            rd0.push_back({acq_x + 1 + j * ex, X, j});
        int64_t acq_y = rd0.back().slot + 1;
        acq.push_back({acq_y, std::string(1, Y)});
        int64_t ey = fit(acq_y + 1, cnt[Y]);
        for(int64_t j = 0; j < cnt[Y]; ++j)
            rd0.push_back({acq_y + 1 + j * ey, Y, j});
    }
    else
    {
        acq.push_back({acq_x, std::string(1, X) + Y});
        int64_t ex = fit(acq_x + 1, nread0);
        for(int64_t j = 0; j < cnt[X]; ++j)
            rd0.push_back({acq_x + 1 + j * ex, X, j});
        int64_t st = rd0.back().slot + ex;
        for(int64_t j = 0; j < cnt[Y]; ++j)
            rd0.push_back({st + j * ex, Y, j});
        if(tr.sia_merge_sync)
            for(auto& r : rd0)
                r.slot -= 1;
        if(tr.sia_use_order)
        {
            int64_t st0 = tr.sia_merge_sync ? acq_x : acq_x + 1;
            rd0.clear();
            for(size_t j = 0; j < useq.size(); ++j)
                rd0.push_back({st0 + (int64_t)j * ex, useq[j].first, useq[j].second});
        }
    }
    if(rd0.back().slot >= ntot)
        return "next-tile reads overflow the tile";
    if(last_y >= n)
        return "next-atom reads must finish inside the first atom";
    std::map<char, int64_t> rel_of;
    for(const auto& r : rel)
        for(char op : r.second)
            rel_of[op] = r.first;
    if(tr.sia_merge_sync)
        rel_of = {{X, acq_x}, {Y, acq_x}};
    std::set<int64_t> busy;
    for(const auto& r : rd1)
        busy.insert(r.slot);
    for(const auto& r : rd0)
        busy.insert(r.slot);
    for(const auto& r : rel)
        busy.insert(r.first);
    for(const auto& r : acq)
        busy.insert(r.first);
    int64_t stride = std::max((int64_t)1, tr.sia_dtl_stride);
    std::vector<std::pair<char, int64_t>> seq;
    for(int64_t p = 0; p < npass[X]; ++p)
        seq.push_back({X, p});
    for(int64_t p = 0; p < npass[Y]; ++p)
        seq.push_back({Y, p});
    if(!rel_of.count(X) || !rel_of.count(Y))
        return "KeyError";
    int64_t lo0 = rel_of[X] + 1;
    std::vector<int64_t> dtl;
    int64_t L = (int64_t)seq.size();
    if(tr.sia_dtl_spread)
    {
        std::vector<int64_t> fr;
        for(int64_t sl = lo0; sl < ntot; ++sl)
            if(!busy.count(sl))
                fr.push_back(sl);
        int64_t prev = -1;
        for(int64_t j = 0; j < L; ++j)
        {
            char op = seq[j].first;
            int64_t left = L - j;
            int64_t tgt
                = !fr.empty()
                      ? fr[std::min((int64_t)fr.size() - 1, fdiv(j * (int64_t)fr.size(), L))]
                      : lo0;
            int64_t want = std::max({tgt, rel_of[op] + 1, prev + 1});
            if(prev >= 0 && (ntot - (prev + stride)) >= left * 1)
                want = std::max(want, prev + stride);
            int64_t nxt = -1;
            for(int64_t sl : fr)
                if(sl >= want)
                {
                    nxt = sl;
                    break;
                }
            if(nxt >= 0 && ntot - nxt >= left)
                want = nxt;
            want = std::min(want, ntot - left);
            want = std::max({want, rel_of[op] + 1, prev + 1});
            dtl.push_back(want);
            prev = want;
        }
    }
    else
    {
        int64_t span = ntot - lo0;
        int64_t prev = -1;
        for(int64_t j = 0; j < L; ++j)
        {
            char op = seq[j].first;
            int64_t want = std::max({lo0 + fdiv(j * span, L), rel_of[op] + 1, prev + 1});
            dtl.push_back(want);
            prev = want;
        }
    }
    if(!dtl.empty() && *std::max_element(dtl.begin(), dtl.end()) >= ntot)
        return "DTL issues overflow the tile";
    return "";
}

/* -------------------------------------------------------------- asm loop tables */

struct Event
{
    int64_t k;
    std::string kind;
    std::string op;
    int64_t idx;
};

std::string seqs(int64_t k0, int64_t step, const std::string& ev, int n = 8)
{
    std::string o;
    for(int j = 0; j < n; ++j)
        o += (j ? " " : "") + std::to_string(k0 + j * step) + "/" + ev + "." + std::to_string(j);
    return o;
}

std::string join(const std::vector<std::string>& v)
{
    std::string o;
    for(size_t i = 0; i < v.size(); ++i)
        o += (i ? " " : "") + v[i];
    return o;
}

std::string table_tensile()
{
    return join({seqs(1, 2, "rB1"),
                 "20/wl 21/bar",
                 "23/dB.0 25/rA1.0 26/dB.1 28/rA1.1 29/dB.2 31/rA1.2",
                 "32/dB.3 34/rA1.3 35/dB.4 37/rA1.4 39/rA1.5 41/rA1.6",
                 "43/rA1.7",
                 "51/wl 52/bar",
                 "53/dB.5 56/dB.6 59/dB.7 62/dA.0 65/dA.1",
                 "68/vB 69/bar",
                 seqs(70, 2, "rB0"),
                 "86/dA.2 88/dA.3 90/dA.4 97/dA.5 101/dA.6",
                 "105/vA 106/bar",
                 seqs(107, 2, "rA0"),
                 "125/dA.7"});
}

std::string table_custom()
{
    return join({seqs(1, 2, "rB1"),
                 "21/wl 22/bar",
                 "23/dB.0 25/rA1.0 26/dB.1 28/rA1.1 29/dB.2 31/rA1.2",
                 "32/dB.3 34/rA1.3 35/dB.4 37/rA1.4 39/rA1.5 41/rA1.6",
                 "43/rA1.7",
                 "51/wl 52/bar",
                 "53/dB.5 56/dB.6 59/dB.7 62/dA.0 65/dA.1",
                 "86/dA.2 88/dA.3 90/dA.4",
                 "92/vAB 93/bar",
                 "94/rB0.0 95/rB0.1 96/rB0.2 97/dA.5 98/rB0.3 99/rB0.4",
                 "101/dA.6 103/rB0.5 104/rB0.6 105/rB0.7",
                 "106/rA0.0 107/rA0.1 110/rA0.2 113/rA0.3 115/rA0.4",
                 "118/rA0.5 121/rA0.6 124/rA0.7 125/dA.7 127/wl"});
}

std::string table_tensile_odd()
{
    return join({seqs(2, 2, "rB1"),
                 "20/wl 21/bar",
                 "23/rA1.0 24/dB.0 26/rA1.1 27/dB.1 29/rA1.2 30/dB.2",
                 "32/rA1.3 33/dB.3 35/rA1.4 36/dB.4 38/rA1.5 40/rA1.6",
                 "42/rA1.7",
                 "51/wl 52/bar",
                 "54/dB.5 57/dB.6 60/dB.7 63/dA.0 66/dA.1",
                 "68/vB 69/bar",
                 seqs(71, 2, "rB0", 7) + " 91/rB0.7",
                 "85/dA.2 87/dA.3 89/dA.4 96/dA.5 100/dA.6",
                 "105/vA 106/bar",
                 seqs(108, 2, "rA0"),
                 "124/dA.7"});
}

std::string table_custom_odd()
{
    return join({seqs(1, 2, "rB1"),
                 "21/wl 22/bar",
                 "23/rA1.0 24/dB.0 26/rA1.1 27/dB.1 29/rA1.2 30/dB.2",
                 "32/rA1.3 33/dB.3 35/rA1.4 36/dB.4 39/rA1.5 41/rA1.6",
                 "43/rA1.7",
                 "51/wl 52/bar",
                 "54/dB.5 57/dB.6 60/dB.7 63/dA.0 66/dA.1",
                 "85/dA.2 87/dA.3 89/dA.4",
                 "92/vAB 93/bar",
                 "94/rB0.0 95/rB0.1 96/dA.5 97/rB0.2 98/rB0.3 99/rB0.4",
                 "100/dA.6 103/rB0.5 104/rB0.6 105/rB0.7",
                 "106/rA0.0 107/rA0.1 110/rA0.2 113/rA0.3 115/rA0.4",
                 "118/rA0.5 121/rA0.6 123/rA0.7 124/dA.7 127/wl"});
}

std::string table_hbr(int64_t shift)
{
    std::vector<std::pair<char, int64_t>> order;
    order.push_back({'A', 0});
    for(int64_t j = 0; j < 8; ++j)
        order.push_back({'B', j});
    for(int64_t j = 1; j < 8; ++j)
        order.push_back({'A', j});
    std::vector<std::string> t;
    for(size_t j = 0; j < order.size(); ++j)
        t.push_back(std::to_string(1 + 2 * (int64_t)j + shift) + "/r" + order[j].first + "1."
                    + std::to_string(order[j].second));
    int64_t rel = 1 + 2 * ((int64_t)order.size() - 1) + 4;
    t.push_back(std::to_string(rel) + "/wl");
    t.push_back(std::to_string(rel) + "/bar");
    int64_t span = 128 - (rel - 1);
    std::vector<std::pair<char, int64_t>> dl;
    for(int64_t j = 0; j < 8; ++j)
        dl.push_back({'A', j});
    for(int64_t j = 0; j < 8; ++j)
        dl.push_back({'B', j});
    for(size_t j = 0; j < dl.size(); ++j)
        t.push_back(std::to_string(rel + fdiv((int64_t)j * span, (int64_t)dl.size()) + shift) + "/d"
                    + dl[j].first + "." + std::to_string(dl[j].second));
    int64_t acq = 64 + 4;
    t.push_back(std::to_string(acq) + "/wl");
    t.push_back(std::to_string(acq) + "/vAB");
    t.push_back(std::to_string(acq) + "/bar");
    for(size_t j = 0; j < order.size(); ++j)
        t.push_back(std::to_string(std::min((int64_t)128, acq + 1 + 2 * (int64_t)j + shift)) + "/r"
                    + order[j].first + "0." + std::to_string(order[j].second));
    return join(t);
}

bool is_digits(const std::string& s)
{
    if(s.empty())
        return false;
    for(char c : s)
        if(c < '0' || c > '9')
            return false;
    return true;
}

bool py_int(const std::string& s, int64_t* out)
{
    FieldValue v = parse_value(s);
    if(v.kind != FieldValue::INT)
        return false;
    *out = v.i;
    return true;
}

/* gemm_asm_loop.parse_table ("" = ok) */
std::string parse_table(const std::string& text, std::vector<Event>* out)
{
    struct E
    {
        int64_t k;
        size_t pos;
        Event e;
    };
    std::vector<E> evs;
    std::istringstream is(text);
    std::string tok;
    size_t pos = 0;
    while(is >> tok)
    {
        size_t sl = tok.find('/');
        if(sl == std::string::npos)
            return "asm_loop token '" + tok + "': expected K/EVENT";
        std::string ks = tok.substr(0, sl), ev = tok.substr(sl + 1);
        int64_t k;
        if(!py_int(ks, &k))
            return "invalid literal for int()";
        Event e{k, "", "", 0};
        if(ev == "wl" || ev == "bar")
            e.kind = ev;
        else if(ev == "vA" || ev == "vB" || ev == "vAB")
        {
            e.kind = "v";
            e.op = ev.substr(1);
        }
        else if(!ev.empty() && ev[0] == 'p' && is_digits(ev.substr(1)))
        {
            e.kind = "p";
            e.idx = std::strtoll(ev.c_str() + 1, nullptr, 10);
        }
        else if(!ev.empty() && ev[0] == 'l' && is_digits(ev.substr(1)))
        {
            e.kind = "l";
            e.idx = std::strtoll(ev.c_str() + 1, nullptr, 10);
        }
        else if(!ev.empty() && (ev[0] == 'r' || ev[0] == 'd') && ev.find('.') != std::string::npos)
        {
            size_t dot = ev.find('.');
            std::string head = ev.substr(0, dot), idx = ev.substr(dot + 1);
            int64_t ii;
            if(head == "rA1" || head == "rB1" || head == "rA0" || head == "rB0")
            {
                if(!py_int(idx, &ii))
                    return "invalid literal for int()";
                e.kind = std::string("r") + head[2];
                e.op = std::string(1, head[1]);
                e.idx = ii;
            }
            else if(head == "dA" || head == "dB")
            {
                if(!py_int(idx, &ii))
                    return "invalid literal for int()";
                e.kind = "d";
                e.op = std::string(1, head[1]);
                e.idx = ii;
            }
            else
                return "asm_loop token '" + tok + "': unknown event";
        }
        else
            return "asm_loop token '" + tok + "': unknown event";
        evs.push_back({k, pos, e});
        ++pos;
    }
    std::stable_sort(evs.begin(), evs.end(), [](const E& a, const E& b) {
        return a.k != b.k ? a.k < b.k : a.pos < b.pos;
    });
    out->clear();
    for(const auto& e : evs)
        out->push_back(e.e);
    return "";
}

/* gemm_asm_loop.check_table ("" = None) */
std::string check_table(const std::vector<Event>& evs)
{
    std::set<std::tuple<std::string, std::string, int64_t>> seen;
    for(const auto& e : evs)
    {
        if(!(0 <= e.k && e.k <= 128))
            return "event slot outside 0..128";
        if(e.kind == "r1" || e.kind == "r0" || e.kind == "d")
        {
            int64_t lim = 8;
            if(!(0 <= e.idx && e.idx < lim))
                return "index outside";
            auto key = std::make_tuple(e.kind, e.op, e.idx);
            if(seen.count(key))
                return "appears twice";
            seen.insert(key);
        }
        if(e.kind == "p" && !(0 <= e.idx && e.idx <= 3))
            return "s_setprio outside 0..3";
        if(e.kind == "l" && !(0 <= e.idx && e.idx <= 15))
            return "lgkmcnt outside 0..15";
    }
    for(const char* kind : {"r1", "r0", "d"})
        for(const char* op : {"A", "B"})
            for(int64_t j = 0; j < 8; ++j)
                if(!seen.count(std::make_tuple(std::string(kind), std::string(op), j)))
                    return "missing";
    for(const auto& e : evs)
        if(e.kind == "r1" && e.k > 64)
            return "after the second atom started";
    std::map<char, int> n_r1 = {{'A', 0}, {'B', 0}};
    std::map<char, bool> released = {{'A', false}, {'B', false}};
    std::map<char, bool> acquired = {{'A', false}, {'B', false}};
    std::map<char, bool> wl_ok = {{'A', false}, {'B', false}};
    std::map<char, bool> pend_v = {{'A', false}, {'B', false}};
    for(const auto& e : evs)
    {
        char op = e.op.empty() ? 0 : e.op[0];
        if(e.kind == "r1")
        {
            if(released[op])
                return "read after its stage was released";
            n_r1[op] += 1;
        }
        else if(e.kind == "wl")
        {
            for(char o : {'A', 'B'})
                if(n_r1[o] == 8)
                    wl_ok[o] = true;
        }
        else if(e.kind == "bar")
        {
            for(char o : {'A', 'B'})
            {
                if(wl_ok[o])
                    released[o] = true;
                if(pend_v[o])
                {
                    acquired[o] = true;
                    pend_v[o] = false;
                }
            }
        }
        else if(e.kind == "v")
        {
            for(char o : e.op)
                pend_v[o] = true;
        }
        else if(e.kind == "d")
        {
            if(!released[op])
                return "DTL before the stage release";
        }
        else if(e.kind == "r0")
        {
            if(!acquired[op])
                return "read before the acquire";
        }
    }
    for(char o : {'A', 'B'})
        if(!(released[o] && acquired[o]))
            return "no release / acquire in the table";
    return "";
}

/* gemm_asm_loop.table_text ("" error text in *err) */
bool table_text(const Trait& tr, bool odd, std::string* text, std::string* err)
{
    const std::string& mode = tr.asm_loop;
    if(mode == "table")
    {
        std::vector<std::string> parts;
        size_t p = 0;
        while(true)
        {
            size_t q = tr.asm_loop_table.find("||", p);
            parts.push_back(
                tr.asm_loop_table.substr(p, q == std::string::npos ? std::string::npos : q - p));
            if(q == std::string::npos)
                break;
            p = q + 2;
        }
        if(parts.size() > 2)
        {
            *err = "asm_loop_table: at most one '||' (even || odd)";
            return false;
        }
        *text = parts[odd && parts.size() == 2 ? 1 : 0];
        return true;
    }
    std::string key = mode + (odd ? "_odd" : "");
    if(key == "hbr")
        *text = table_hbr(0);
    else if(key == "hbr_odd")
        *text = table_hbr(1);
    else if(key == "tensile")
        *text = table_tensile();
    else if(key == "custom")
        *text = table_custom();
    else if(key == "tensile_odd")
        *text = table_tensile_odd();
    else if(key == "custom_odd")
        *text = table_custom_odd();
    else
    {
        *err = "'" + key + "'"; /* KeyError */
        return false;
    }
    return true;
}

/* table_events + check_table ("" = None; the caught ValueError / KeyError text otherwise) */
std::string table_check(const Trait& tr, bool odd, std::vector<Event>* evs)
{
    std::string text, err;
    if(!table_text(tr, odd, &text, &err))
        return err;
    err = parse_table(text, evs);
    if(!err.empty())
        return err;
    return "";
}

std::vector<std::vector<std::string>> table_proto(const std::vector<Event>& evs)
{
    std::vector<std::vector<std::string>> out;
    std::vector<std::string> st;
    for(const auto& e : evs)
    {
        if(e.kind == "wl" || e.kind == "v")
            st.push_back(e.kind + e.op);
        else if(e.kind == "bar")
        {
            out.push_back(st);
            st.clear();
        }
    }
    return out;
}

/* gemm_asm_loop.check_skew ("" = None) */
std::string check_skew(const Trait& tr)
{
    std::vector<Event> e, o;
    std::string w = table_check(tr, false, &e);
    if(!w.empty())
        return w;
    w = table_check(tr, true, &o);
    if(!w.empty())
        return w;
    w = check_table(e);
    if(!w.empty())
        return "even table: " + w;
    w = check_table(o);
    if(!w.empty())
        return "odd table: " + w;
    auto nbar = [](const std::vector<Event>& v) {
        return std::count_if(v.begin(), v.end(), [](const Event& x) { return x.kind == "bar"; });
    };
    if(nbar(e) != nbar(o))
        return "even / odd tables have different barrier counts";
    if(table_proto(e) != table_proto(o))
        return "even / odd tables: the barriers release / acquire different operands";
    return "";
}

} // namespace

/* --------------------------------------------------------------- is_valid_spec */

bool is_valid_spec_x(
    const UGemm& spec, const std::string& arch, bool gett, bool a_sub_perm, std::string* why)
{
    ArchFacts target;
    if(!arch_facts(arch, &target))
        return set_why(why, "unknown arch '" + arch + "'");
    const Tile& t = spec.tile;
    const Data& d = spec.data;
    if(d.dtype_a != d.dtype_b || d.dtype_a != d.dtype_c)
        return set_why(why, "UniversalGemmSpec currently requires homogeneous A/B/C dtypes");
    if(d.dtype_acc != "fp32" && d.dtype_acc != "f32")
        return set_why(why, "UniversalGemmSpec only supports fp32 accumulation");
    if(!is_gemm_layout(d.layout))
        return set_why(why, "UniversalGemmSpec supports layouts RCR / RRR / CRR / CCR");
    /* io_ir_type(dtype_a): the 16-bit types the dispatcher builds */
    if(d.dtype_a != "fp16" && d.dtype_a != "bf16")
        return set_why(why, "unsupported storage dtype '" + d.dtype_a + "'");
    const std::string family = "mma";
    const std::string& a_name = d.dtype_a;
    if(!supports_dtype(arch, a_name))
        return set_why(why, "unsupported GEMM dtype");
    int a_frag = 0, b_frag = 0;
    if(!mma_shape(arch, a_name, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k, &a_frag, &b_frag))
        return set_why(why, "unsupported warp_tile");
    if(spec.wave_size != target.wave_size)
        return set_why(why, "spec wave_size != arch wave_size");
    const Trait& tr = spec.trait;
    if(tr.direct_to_lds)
    {
        if(target.async_lds_max_dwords < 1)
            return set_why(why, "direct_to_lds: no DRAM->LDS DMA instruction");
        int64_t dtl_halves = std::min(4, target.async_lds_max_dwords) * 2;
        if(t.tile_k % dtl_halves != 0)
            return set_why(why, "direct_to_lds requires tile_k % 8 == 0");
    }
    if(t.warp_m * t.warp_tile_m == 0 || t.tile_m % (t.warp_m * t.warp_tile_m))
        return set_why(why, "tile_m not divisible by warp_m * warp_tile_m");
    if(t.warp_n * t.warp_tile_n == 0 || t.tile_n % (t.warp_n * t.warp_tile_n))
        return set_why(why, "tile_n not divisible by warp_n * warp_tile_n");
    if(t.tile_k % t.warp_tile_k)
        return set_why(why, "tile_k not divisible by warp_tile_k");
    int64_t expected_bs = t.warp_m * t.warp_n * spec.wave_size;
    if(expected_bs != spec.block_size)
        return set_why(why, "block_size != warp_m*warp_n*wave_size");
    if(tr.scheduler != "intrawave" && tr.scheduler != "interwave")
        return set_why(why, "unknown scheduler");
    if(tr.ab_double_buffer.has_value() && (tr.pipeline != "compv4" || tr.direct_to_lds))
        return set_why(why, "ab_double_buffer applies only to the compv4 VGPR-staged pipeline");
    if(!(tr.dtl_stages == 0 || tr.dtl_stages == 2 || tr.dtl_stages == 3 || tr.dtl_stages == 4))
        return set_why(why, "dtl_stages must be 0, 2, 3 or 4");
    if(tr.dtl_stages && !(tr.direct_to_lds && tr.dtl_prefetch))
        return set_why(why, "dtl_stages requires the MFMA direct_to_lds + dtl_prefetch body");
    if((tr.dtl_issue_spread || tr.dtl_frag_prefetch || tr.dtl_stream_barrier
        || tr.iglp_opt.has_value())
       && tr.dtl_stages < 2)
        return set_why(why, "requires dtl_stages >= 2");
    if(tr.persistent_ctas)
    {
        if(tr.persistent_ctas < 0)
            return set_why(why, "persistent_ctas must be >= 0");
        if(tr.dtl_stages < 2)
            return set_why(why, "persistent_ctas requires dtl_stages >= 2");
        if(tr.epilogue != "default" || spec.batched || tr.split_k > 1 || tr.active_tile_skip)
            return set_why(why,
                           "persistent_ctas requires the default epilogue, a non-batched body and "
                           "split_k == 1");
        if(tr.chiplet_swizzle
           && pmod(tr.persistent_ctas, std::max((int64_t)1, tr.chiplet_num_xcds)))
            return set_why(
                why, "persistent_ctas with chiplet_swizzle must be a multiple of chiplet_num_xcds");
    }
    if(tr.streamk && tr.persistent_ctas <= 0)
        return set_why(why, "streamk requires persistent_ctas > 0");
    if(tr.streamk && t.warp_k != 1)
        return set_why(why, "streamk requires warp_k == 1");
    if(tr.sk_xcd_local && !(tr.streamk && tr.chiplet_swizzle))
        return set_why(why, "sk_xcd_local requires streamk and chiplet_swizzle");
    if(tr.dtl_tight_rsrc && (tr.dtl_stages < 2 || spec.batched))
        return set_why(why, "dtl_tight_rsrc requires dtl_stages >= 2 and a non-batched body");
    if(tr.pad_fast_tile
       && (!(tr.pad_m || tr.pad_n) || tr.epilogue != "default" || tr.split_k > 1
           || tr.mfma_swap_ab))
        return set_why(why,
                       "pad_fast_tile requires pad_m or pad_n, the default epilogue, split_k == 1 "
                       "and no mfma_swap_ab");
    if(tr.iglp_opt.has_value() && !(*tr.iglp_opt >= 0 && *tr.iglp_opt <= 3))
        return set_why(why, "iglp_opt must be 0..3");
    /* acc_agpr / mfma_interleave / lds_swizzle_r / dtl_hoist: gfx942 / gfx950 */
    int64_t mm = 0, mn = 0, ka = 0;
    if(tr.ilv_dtl_every && (tr.ilv_dtl_every < 0 || !tr.mfma_interleave))
        return set_why(why, "ilv_dtl_every must be >= 0 and requires mfma_interleave");
    t.mfmas_m(&mm);
    t.mfmas_n(&mn);
    if(tr.ilv_barrier_after
       && (!tr.mfma_interleave || !(0 < tr.ilv_barrier_after && tr.ilv_barrier_after < mm * mn)))
        return set_why(
            why, "ilv_barrier_after requires mfma_interleave and 0 < k < the MFMAs of one K atom");
    if(tr.ilv_merge && (!tr.mfma_interleave || tr.ilv_dtl_every))
        return set_why(why, "ilv_merge requires mfma_interleave and excludes ilv_dtl_every");
    int64_t n_atom = mm * mn;
    /* ilv_early_release / ilv_read_span / ilv_dtl_span / acc_out_vgpr: gfx950 (arch is gfx950) */
    if(tr.ilv_early_release)
    {
        bool okk = t.k_atoms(&ka);
        if(!okk)
            return set_why(why, "tile_k not divisible by warp_tile_k");
        if(!tr.mfma_interleave || ka < 2)
            return set_why(why,
                           "ilv_early_release requires mfma_interleave and >= 2 K atoms per tile");
    }
    if(!(tr.ilv_early_release >= 0 && tr.ilv_early_release <= 3)
       || (tr.ilv_early_release >= 2 && tr.ilv_barrier_after))
        return set_why(why,
                       "ilv_early_release must be 0..3; modes 2 / 3 exclude ilv_barrier_after");
    if(tr.ilv_release_at
       && (tr.ilv_early_release < 2 || !(0 < tr.ilv_release_at && tr.ilv_release_at < n_atom)))
        return set_why(
            why,
            "ilv_release_at requires ilv_early_release 2 / 3 and 0 < k < the MFMAs of one K atom");
    if(tr.ilv_acquire_at
       && (!tr.ilv_early_release
           || !(tr.ilv_barrier_after < tr.ilv_acquire_at && tr.ilv_acquire_at < n_atom)))
        return set_why(why,
                       "ilv_acquire_at requires ilv_early_release and ilv_barrier_after < k < the "
                       "MFMAs of one K atom");
    if(tr.lds_swizzle_r != -1
       && !(tr.lds_swizzle && 0 <= tr.lds_swizzle_r && tr.lds_swizzle_r <= 4))
        return set_why(why, "lds_swizzle_r must be -1 (auto) or 0..4 with lds_swizzle");
    if(tr.lds_swizzle_l != -1)
    {
        int64_t w = 0;
        bool okw = 1 <= tr.lds_swizzle_l && tr.lds_swizzle_l <= 4
                   && ilog2_exact(t.tile_k >> tr.lds_swizzle_l, &w);
        if(!(tr.lds_swizzle && tr.vgpr_prefetch && 1 <= tr.lds_swizzle_l && tr.lds_swizzle_l <= 4
             && t.tile_k % (int64_t(1) << tr.lds_swizzle_l) == 0 && okw && w >= 1))
            return set_why(why,
                           "lds_swizzle_l must be -1 (auto) or 1..4 with lds_swizzle on the "
                           "vgpr_prefetch body");
    }
    if(tr.dtl_hoist && tr.dtl_stages < 2)
        return set_why(why, "dtl_hoist requires dtl_stages >= 2");
    if(tr.vgpr_prefetch_depth != 1 && !tr.vgpr_prefetch)
        return set_why(why, "vgpr_prefetch_depth requires vgpr_prefetch");
    if(tr.vp_unroll && (tr.vp_unroll != 1 || tr.vgpr_prefetch < 3))
        return set_why(why, "vp_unroll must be 0 / 1 and requires vgpr_prefetch 3 / 4");
    if(tr.vp_late_reads
       && (tr.vp_late_reads != 1 || tr.vgpr_prefetch != 4 || tr.vgpr_prefetch_depth != 1))
        return set_why(why,
                       "vp_late_reads must be 0 / 1 and requires vgpr_prefetch 4 with depth 1");
    if(tr.vp_release_atoms != 1)
    {
        bool okk = t.k_atoms(&ka);
        if(tr.vgpr_prefetch != 4 || tr.vp_late_reads || !okk
           || !(1 <= tr.vp_release_atoms && tr.vp_release_atoms < ka))
            return set_why(why,
                           "vp_release_atoms must be 1 .. K atoms per tile - 1 and requires "
                           "vgpr_prefetch 4 without vp_late_reads");
    }
    if(tr.vp_tail_pack
       && (!(tr.vp_tail_pack >= 1 && tr.vp_tail_pack <= 3) || tr.vgpr_prefetch != 4))
        return set_why(why, "vp_tail_pack must be 0 .. 3 and requires vgpr_prefetch 4");
    if(tr.vp_mfma_first
       && (!(tr.vp_mfma_first == 1 || tr.vp_mfma_first == 2)
           || !(tr.vgpr_prefetch == 3 || tr.vgpr_prefetch == 4)))
        return set_why(why, "vp_mfma_first must be 0 / 1 / 2 and requires vgpr_prefetch 3 / 4");
    if(tr.vp_pair_reads
       && (!(tr.vp_pair_reads == 1 || tr.vp_pair_reads == 2) || tr.vgpr_prefetch != 4
           || tr.vgpr_prefetch_depth != 1 || tr.vp_late_reads || tr.vp_tail_pack == 2 || [&] {
                  std::string l = d.layout;
                  for(auto& c : l)
                      c = (char)toupper((unsigned char)c);
                  return l != "RCR";
              }()))
        return set_why(why, "vp_pair_reads constraints");
    {
        std::string l = d.layout;
        for(auto& c : l)
            c = (char)toupper((unsigned char)c);
        if((tr.vp_mn_kv || tr.vp_mn_order || tr.vp_mn_worder || tr.vp_mn_pipe || tr.vp_mn_wrap)
           && (!tr.vgpr_prefetch || l == "RCR"))
            return set_why(why, "vp_mn_* require vgpr_prefetch and an MN-contiguous operand");
    }
    if(tr.vgpr_prefetch)
    {
        if(!(tr.vgpr_prefetch >= 1 && tr.vgpr_prefetch <= 4))
            return set_why(why, "vgpr_prefetch must be 0..4");
        if(!(tr.vgpr_prefetch_depth >= 1 && tr.vgpr_prefetch_depth <= 3))
            return set_why(why, "vgpr_prefetch_depth must be 1..3");
        if(tr.vgpr_prefetch_depth > 1 && tr.acc_agpr)
            return set_why(why, "vgpr_prefetch_depth > 1 with AGPR accumulators");
        if(tr.lds_swizzle)
        {
            int64_t R, W, L;
            spec_swizzle_params(tr, a_frag, b_frag, t.tile_k, &R, &W, &L);
            if(t.tile_m % (int64_t(1) << (R + W)) || t.tile_n % (int64_t(1) << (R + W)))
                return set_why(why,
                               "vgpr_prefetch with lds_swizzle: tile_m / tile_n must be multiples "
                               "of the swizzle row period");
            if(tr.lds_swizzle_l >= 0 && (int64_t(1) << L) < std::max(a_frag, b_frag))
                return set_why(why, "lds_swizzle_l: the XOR group must hold a whole MFMA fragment");
        }
        if(tr.vgpr_prefetch >= 3)
        {
            bool okk = t.k_atoms(&ka);
            if(!okk)
                return set_why(why, "tile_k not divisible by warp_tile_k");
            if(ka < 2 || tr.acc_agpr_pin)
                return set_why(
                    why, "vgpr_prefetch 3 / 4 need >= 2 K atoms per tile and exclude acc_agpr_pin");
        }
        if(!(tr.pipeline == "compv4" && !tr.direct_to_lds))
            return set_why(why, "vgpr_prefetch requires the MFMA compv4 VGPR-staged body");
        if(tr.vgpr_prefetch == 4)
        {
            if(!(tr.ab_double_buffer.has_value() && !*tr.ab_double_buffer)
               || tr.epilogue != "default")
                return set_why(why, "vgpr_prefetch 4 is the single-LDS-buffer body");
        }
        else if(!ab_double_buffer_decision(spec, target.lds_capacity_bytes))
            return set_why(why, "vgpr_prefetch 1 / 2 / 3 require the AB LDS double buffer");
        if(tr.preshuffle_b || tr.active_tile_skip || tr.lds_k_pad)
            return set_why(why,
                           "vgpr_prefetch excludes preshuffle_b, active_tile_skip and lds_k_pad");
        {
            std::string l = d.layout;
            for(auto& c : l)
                c = (char)toupper((unsigned char)c);
            if(spec.batched && (tr.split_k > 1 || tr.split_k_serial || l != "RCR"))
                return set_why(why, "vgpr_prefetch batched: the RCR layout without split-K");
        }
        if(tr.pad_k && tr.split_k > 1 && !tr.split_k_aligned)
            return set_why(why, "vgpr_prefetch with pad_k and split-K needs split_k_aligned");
        int64_t lv = 0;
        if(!choose_load_vec(t.tile_m, t.tile_n, t.tile_k, spec.block_size, &lv))
            return set_why(why, "no usable load_vec");
        if(lv != 2 && lv != 4 && lv != 8)
            return set_why(why, "vgpr_prefetch needs 4 / 8 / 16-byte global loads per lane");
    }
    for(int64_t pc : {tr.ilv_read_span, tr.ilv_dtl_span})
        if(pc && (!tr.mfma_interleave || !(0 < pc && pc <= 100) || tr.ilv_dtl_every))
            return set_why(why, "ilv_read_span / ilv_dtl_span requires mfma_interleave");
    if(tr.acc_out_vgpr && !tr.acc_agpr)
        return set_why(why, "acc_out_vgpr requires acc_agpr");
    if(tr.lds_row_interleave)
    {
        if(arch != "gfx950" || tr.dtl_stages < 2 || tr.lds_swizzle)
            return set_why(
                why, "lds_row_interleave requires gfx950, dtl_stages >= 2 and lds_swizzle off");
        if(!t.mfmas_m(&mm) || !t.mfmas_n(&mn))
            return set_why(why, "tile not divisible");
        if((d.dtype_a != "fp16" && d.dtype_a != "bf16") || d.dtype_b != d.dtype_a || t.tile_k != 64
           || !(t.warp_tile_m == 16 && t.warp_tile_n == 16 && t.warp_tile_k == 32) || mm % 8
           || mn % 8 || spec.wave_size != 64)
            return set_why(why,
                           "lds_row_interleave requires 16-bit A/B, tile_k 64, the 16x16x32 atom, "
                           "wave64 and mfmas_m / mfmas_n multiples of 8");
    }
    if(tr.rilv_vec_epilogue
       && !(tr.lds_row_interleave && tr.epilogue == "default" && !tr.mfma_swap_ab
            && (tr.rve_pad || !(tr.pad_m || tr.pad_n)) && tr.split_k == 1))
        return set_why(why,
                       "rilv_vec_epilogue requires lds_row_interleave, the default epilogue, no "
                       "mfma_swap_ab, split_k == 1 and no pad_m / pad_n unless rve_pad");
    if(!(tr.rve_pad == 0 || tr.rve_pad == 1 || tr.rve_pad == 2))
        return set_why(why, "rve_pad must be 0, 1 or 2");
    if(tr.rve_pad && !(tr.rilv_vec_epilogue && (tr.pad_m || tr.pad_n) && tr.dtl_tight_rsrc))
        return set_why(why,
                       "rve_pad requires rilv_vec_epilogue, pad_m or pad_n and dtl_tight_rsrc");
    if(tr.rve_pad == 2 && spec.batched)
        return set_why(why, "rve_pad 2 requires a non-batched body");
    if(tr.rve_store_nt && !tr.rilv_vec_epilogue)
        return set_why(why, "rve_store_nt requires rilv_vec_epilogue");
    if(tr.dtl_rsrc_word3
       && !(tr.direct_to_lds && tr.dtl_stages >= 2 && 0 < tr.dtl_rsrc_word3
            && tr.dtl_rsrc_word3 < (int64_t(1) << 32)))
        return set_why(
            why, "dtl_rsrc_word3 requires direct_to_lds with dtl_stages >= 2 and a u32 value");
    if(tr.dtl_addr_spread && !(tr.hb_sched && tr.dtl_hoist))
        return set_why(why, "dtl_addr_spread requires hb_sched and dtl_hoist");
    {
        int64_t v = tr.k_stagger_rc;
        bool p2 = v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
        if(v
           && !(p2 && tr.direct_to_lds && tr.dtl_stages >= 2 && !tr.pingpong && !tr.streamk
                && tr.asm_loop.empty() && tr.split_k == 1))
            return set_why(why, "k_stagger_rc constraints");
    }
    if(tr.dtl_tight_ragged_only && !(tr.dtl_tight_rsrc && tr.dtl_hoist && (tr.pad_m || tr.pad_n)))
        return set_why(
            why, "dtl_tight_ragged_only requires dtl_tight_rsrc, dtl_hoist and pad_m / pad_n");
    if(tr.hb_skew
       && !((tr.hb_skew == 1 || tr.hb_skew == 2) && tr.hb_sched && tr.asm_loop.empty()
            && !tr.pingpong))
        return set_why(why, "hb_skew requires 1 or 2 with hb_sched, no asm_loop / pingpong");
    if(tr.hb_tile_lgkm0
       && !(tr.hb_tile_lgkm0 == 1 && tr.hb_sched && tr.dtl_frag_prefetch && tr.mfma_interleave
            && tr.asm_loop.empty() && !tr.pingpong))
        return set_why(why,
                       "hb_tile_lgkm0 requires 1 with hb_sched + dtl_frag_prefetch + "
                       "mfma_interleave, no asm_loop / pingpong");
    {
        int64_t v = tr.k_stagger;
        bool p2 = v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
        if(v
           && !(p2 && tr.direct_to_lds && tr.dtl_stages >= 2 && !tr.dtl_tight_rsrc && !tr.streamk
                && tr.split_k == 1 && !tr.pad_k && !tr.pingpong))
            return set_why(why, "k_stagger constraints");
    }
    if(tr.dtl_clamp_k
       && !(tr.dtl_soff0 && tr.dtl_hoist && !tr.dtl_tight_rsrc && !tr.persistent_ctas && !tr.streamk
            && tr.split_k == 1 && !tr.pad_k && !tr.pingpong && !(tr.epilogue == "cshuffle")))
        return set_why(why, "dtl_clamp_k constraints");
    if(tr.k_stagger_map && !(tr.k_stagger && tr.k_stagger_map == 1))
        return set_why(why, "k_stagger_map requires k_stagger and 1");
    if(tr.k_stagger_tiles && !(tr.k_stagger && 0 < tr.k_stagger_tiles && tr.k_stagger_tiles <= 16))
        return set_why(why, "k_stagger_tiles requires k_stagger and 1..16");
    if(tr.das_late && !(tr.dtl_addr_spread && tr.hb_sched))
        return set_why(why, "das_late requires dtl_addr_spread (hb_sched)");
    if(tr.das_anchor && !tr.dtl_addr_spread)
        return set_why(why, "das_anchor requires dtl_addr_spread");
    if(tr.hb_skew_dtl && !(tr.hb_skew && 0 < tr.hb_skew_dtl && tr.hb_skew_dtl <= 8))
        return set_why(why, "hb_skew_dtl requires hb_skew and 1..8 slots");
    if(tr.dtl_base_fold
       && !(tr.dtl_soff0 && tr.dtl_hoist && !tr.dtl_tight_rsrc && !tr.persistent_ctas
            && !tr.streamk))
        return set_why(why, "dtl_base_fold constraints");
    {
        std::string l = d.layout;
        for(auto& c : l)
            c = (char)toupper((unsigned char)c);
        if(tr.wide_addr && tr.vgpr_prefetch)
        {
            if(tr.direct_to_lds || tr.epilogue != "default" || tr.rilv_vec_epilogue
               || tr.split_k > 1 || tr.split_k_serial || tr.persistent_ctas || tr.streamk
               || l != "RCR" || gett)
                return set_why(why, "wide_addr on the vgpr_prefetch body constraints");
        }
        else if(tr.wide_addr
                && !(tr.direct_to_lds && tr.dtl_hoist && (tr.dtl_soff0 || tr.dtl_tight_rsrc)
                     && tr.rilv_vec_epilogue && !tr.dtl_base_fold && !tr.persistent_ctas
                     && !tr.streamk && tr.split_k <= 1 && !tr.split_k_serial && !tr.pingpong
                     && tr.asm_loop.empty() && !tr.dtl_k_tail && l == "RCR" && !spec.batched
                     && !gett))
            return set_why(why, "wide_addr constraints");
    }
    {
        int64_t v = tr.stagger_u;
        bool p2 = v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
        if((tr.stagger_u || tr.stagger_u_stride != 1 || tr.stagger_u_map)
           && !(p2 && 1 <= tr.stagger_u_stride && tr.stagger_u_stride <= 64
                && (tr.stagger_u_map == 0 || tr.stagger_u_map == 1 || tr.stagger_u_map == 2)
                && (tr.stagger_u_map != 2 || tr.chiplet_swizzle) && tr.direct_to_lds
                && tr.dtl_stages >= 2 && !tr.persistent_ctas && !tr.streamk && !tr.pingpong
                && tr.split_k == 1))
            return set_why(why, "stagger_u constraints");
    }
    if((tr.k_stagger ? 1 : 0) + (tr.stagger_u ? 1 : 0) + (tr.k_stagger_rc ? 1 : 0) > 1)
        return set_why(
            why,
            "k_stagger, stagger_u and k_stagger_rc are alternative StaggerU forms (at most one)");
    if(tr.stagger_u && tr.dtl_clamp_k)
        return set_why(why, "stagger_u does not combine with dtl_clamp_k");
    if(tr.mfma_serpentine && !(tr.hb_sched || tr.sia_sched || tr.mfma_interleave))
        return set_why(
            why, "mfma_serpentine requires hb_sched / asm_loop / sia_sched / mfma_interleave");
    if(tr.mfma_order
       && !((tr.mfma_order == 2 || tr.mfma_order == 3)
            && (tr.hb_sched || tr.sia_sched || tr.mfma_interleave)
            && (tr.mfma_order != 2 || tr.hb_sched || tr.sia_sched) && tr.asm_loop.empty()
            && !tr.mfma_serpentine))
        return set_why(why, "mfma_order constraints");
    if(tr.hb_dtl_delay && !(tr.hb_sched && 0 < tr.hb_dtl_delay && tr.hb_dtl_delay <= 16))
        return set_why(why, "hb_dtl_delay requires hb_sched and 1..16 slots");
    if(tr.kloop_unroll != 1
       && !((tr.kloop_unroll == 2 || tr.kloop_unroll == 4) && tr.direct_to_lds && tr.dtl_stages >= 2
            && !tr.pingpong))
        return set_why(why,
                       "kloop_unroll requires 2 or 4 with direct_to_lds, dtl_stages >= 2, no "
                       "pingpong / streamk");
    if(tr.dtl_soff0
       && !(tr.direct_to_lds && tr.dtl_stages >= 2 && tr.dtl_hoist && !tr.dtl_tight_rsrc
            && !tr.pingpong))
        return set_why(why,
                       "dtl_soff0 requires direct_to_lds, dtl_stages >= 2 and dtl_hoist, without "
                       "dtl_tight_rsrc / pingpong");
    if(tr.rilv_pad != 16
       && !(tr.lds_row_interleave && (tr.rilv_pad == 32 || tr.rilv_pad == 48 || tr.rilv_pad == 64)))
        return set_why(why, "rilv_pad requires lds_row_interleave and one of 16, 32, 48, 64 bytes");
    if(tr.hb_sched)
    {
        if(!(tr.dtl_frag_prefetch && tr.mfma_interleave))
            return set_why(why, "hb_sched requires dtl_frag_prefetch and mfma_interleave");
        if(tr.ilv_merge || tr.ilv_barrier_after || tr.ilv_dtl_every || tr.iglp_opt.has_value())
            return set_why(why, "hb_sched is its own MFMA-slot schedule");
        if(!t.mfmas_m(&mm) || !t.mfmas_n(&mn))
            return set_why(why, "tile not divisible");
        int64_t ntot = hb_k_atoms(spec) * mm * mn;
        if(tr.hb_split_rel && (tr.hb_rel_slot >= 0 || hb_k_atoms(spec) < 2))
            return set_why(why, "hb_split_rel places both releases itself");
        if(tr.hb_rel_gap < 1)
            return set_why(why, "hb_rel_gap must be >= 1");
        int64_t rel, acq;
        hb_slots(spec, &rel, &acq);
        if(tr.hb_acq_b_slot >= 0 && !(acq < tr.hb_acq_b_slot && tr.hb_acq_b_slot < ntot))
            return set_why(why, "hb_acq_b_slot must lie after the A acquire");
        if(!(-1 <= tr.hb_rel_slot && tr.hb_rel_slot < ntot && -1 <= tr.hb_acq_slot
             && tr.hb_acq_slot < ntot))
            return set_why(why, "hb_rel_slot / hb_acq_slot must be -1 (auto) or an MFMA index");
        if(tr.hb_read_every < 1 || tr.hb_dtl_every < 0)
            return set_why(why, "hb_read_every must be >= 1 and hb_dtl_every >= 0");
        int64_t last_rd = hb_last_read_slot(spec);
        if(0 <= tr.hb_rel_slot && tr.hb_rel_slot <= last_rd)
            return set_why(why, "hb_rel_slot must come after the last intra-tile fragment read");
    }
    else if(!(tr.hb_rel_slot == -1 && tr.hb_acq_slot == -1 && tr.hb_read_every == 1
              && tr.hb_dtl_every == 0 && tr.hb_acq_b_slot == -1 && !tr.hb_split_rel
              && tr.hb_rel_gap == 4))
        return set_why(why,
                       "hb_rel_slot / hb_acq_slot / hb_read_every / hb_dtl_every require hb_sched");
    if(tr.sia_sched)
    {
        if(!(tr.dtl_frag_prefetch && tr.mfma_interleave && tr.lds_row_interleave))
            return set_why(
                why,
                "sia_sched requires dtl_frag_prefetch, mfma_interleave and lds_row_interleave");
        if(tr.ilv_merge || tr.ilv_barrier_after || tr.ilv_dtl_every || tr.iglp_opt.has_value()
           || tr.ilv_read_span || tr.ilv_dtl_span || tr.mfma_swap_ab || tr.streamk)
            return set_why(why, "sia_sched is its own MFMA-slot schedule");
        if(!t.k_atoms(&ka))
            return set_why(why, "tile_k not divisible by warp_tile_k");
        if(ka != 2)
            return set_why(why, "sia_sched needs exactly 2 K atoms per tile");
        if(!t.mfmas_n(&mn))
            return set_why(why, "tile not divisible");
        if(tr.sia_read_every < 1 || tr.sia_rel_gap < 1
           || tr.sia_acq_gap < (tr.sia_merge_sync ? -mn : 0) || tr.sia_dtl_stride < 1)
            return set_why(
                why,
                "sia_read_every / sia_rel_gap / sia_dtl_stride must be >= 1 and sia_acq_gap >= 0");
        if((tr.sia_use_order || tr.sia_merge_sync)
           && (tr.sia_split_rel || tr.sia_split_acq || tr.sia_tensile))
            return set_why(why, "sia_use_order / sia_merge_sync need one release and one acquire");
        std::string w = sia_table_error(spec);
        if(!w.empty())
            return set_why(why, "sia_sched: " + w);
    }
    else if(tr.sia_b_first || tr.sia_plr || tr.sia_single_lgkm || tr.sia_split_rel
            || tr.sia_split_acq || tr.sia_dtl_spread || tr.sia_read_every != 2
            || tr.sia_rel_gap != 4 || tr.sia_acq_gap != 3 || tr.sia_dtl_stride != 3
            || tr.sia_tensile || tr.sia_use_order || tr.sia_merge_sync)
        return set_why(why, "sia_* sub-levers require sia_sched");
    if(tr.fr0_loop_order && (!tr.asm_loop.empty() || !(tr.dtl_frag_prefetch && tr.mfma_interleave)))
        return set_why(
            why, "fr0_loop_order requires dtl_frag_prefetch + mfma_interleave and no asm_loop");
    {
        const std::string& m = tr.asm_loop;
        if(!(m.empty() || m == "tensile" || m == "custom" || m == "hbr" || m == "table"))
            return set_why(why, "asm_loop must be one of the modes");
    }
    if(tr.asm_loop.empty()
       && (!tr.asm_loop_table.empty() || tr.asm_loop_prio || tr.asm_loop_skew || tr.asm_loop_rawrd))
        return set_why(why, "asm_loop_table / asm_loop_prio / asm_loop_skew require asm_loop");
    if(!tr.asm_loop.empty())
    {
        if(!tr.asm_loop_table.empty() && tr.asm_loop != "table")
            return set_why(why, "asm_loop_table requires asm_loop='table'");
        if(!(tr.hb_sched && tr.lds_row_interleave && tr.dtl_stages == 2))
            return set_why(why, "asm_loop requires hb_sched, lds_row_interleave and dtl_stages 2");
        if(!(tr.hb_rel_slot == -1 && tr.hb_acq_slot == -1 && tr.hb_read_every == 1
             && tr.hb_dtl_every == 0 && tr.hb_acq_b_slot == -1 && !tr.hb_split_rel
             && tr.hb_rel_gap == 4))
            return set_why(why, "asm_loop places every event itself (no hb_* slot knob)");
        if(!(t.tile_m == 256 && t.tile_n == 256 && t.tile_k == 64)
           || !(t.warp_m == 2 && t.warp_n == 2 && t.warp_k == 1)
           || !(t.warp_tile_m == 16 && t.warp_tile_n == 16 && t.warp_tile_k == 32))
            return set_why(why, "asm_loop tables are written for the 256x256x64 tile");
        if(tr.streamk || tr.interwave_setprio)
            return set_why(why, "asm_loop excludes streamk and interwave_setprio");
        if(!(0 <= tr.asm_loop_prio && tr.asm_loop_prio <= 3))
            return set_why(why, "asm_loop_prio must be 0..3");
        std::vector<Event> evs;
        std::string w = table_check(tr, false, &evs);
        if(w.empty())
            w = check_table(evs);
        if(!w.empty())
            return set_why(why, "asm_loop table: " + w);
        if(!(tr.asm_loop_skew == 0 || tr.asm_loop_skew == 1 || tr.asm_loop_skew == 2))
            return set_why(why, "asm_loop_skew must be 0, 1 or 2");
        if(tr.asm_loop_skew)
        {
            w = check_skew(tr);
            if(!w.empty())
                return set_why(why, "asm_loop_skew: " + w);
        }
    }
    if(tr.pp_wait_late && !tr.pingpong)
        return set_why(why, "pp_wait_late requires pingpong");
    if(tr.pp_spread_lo)
    {
        if(!t.mfmas_m(&mm) || !t.mfmas_n(&mn) || !t.k_atoms(&ka))
            return set_why(why, "tile not divisible");
        if(!(tr.pingpong == 6 && 0 < tr.pp_spread_lo && tr.pp_spread_lo < mm * mn * ka))
            return set_why(why, "pp_spread_lo requires pingpong 6");
    }
    if(tr.pp_mix_reads && (tr.pp_mix_reads < 0 || tr.pingpong < 4))
        return set_why(why, "pp_mix_reads must be >= 0 and requires pingpong 4 / 5 / 6");
    if(tr.pingpong)
    {
        int64_t nw = t.warp_m * t.warp_n * t.warp_k;
        if(!(tr.pingpong >= 1 && tr.pingpong <= 7))
            return set_why(why, "pingpong must be 0..7");
        if(tr.pingpong == 7 && (tr.dtl_hoist || tr.dtl_tight_rsrc))
            return set_why(why, "pingpong 7 builds its own DMA resources");
        if(tr.pingpong == 6 || tr.pingpong == 7)
        {
            int64_t eb = (d.dtype_a == "fp16" || d.dtype_a == "bf16") ? 2 : 4;
            int64_t l6 = (2 * t.tile_m + 3 * t.tile_n) * t.tile_k * eb;
            if(l6 > 163840)
                return set_why(why, "pingpong 6 / 7 need 2 A + 3 B stages of LDS");
        }
        if(tr.pingpong >= 4 && tr.dtl_stages != 2)
            return set_why(why, "pingpong 4 / 5 / 6 requires dtl_stages == 2");
        if(tr.pingpong < 4 && tr.dtl_stages < 3)
            return set_why(why, "pingpong 1..3 requires dtl_stages >= 3");
        if(tr.pingpong >= 4 && tr.pp_wait_late)
            return set_why(why, "pingpong 4 / 5 place their own waits");
        if(tr.pingpong >= 4 && (tr.epilogue != "default" || tr.split_k > 1 || spec.batched))
            return set_why(
                why, "pingpong 4 needs the default epilogue, split_k == 1 and a non-batched body");
        if(t.warp_k != 1 || nw < 2 || nw % 2)
            return set_why(why, "pingpong requires warp_k == 1 and an even wave count");
        if(tr.dtl_frag_prefetch || tr.dtl_stream_barrier || tr.dtl_issue_spread
           || tr.mfma_interleave || tr.iglp_opt.has_value()
           || (tr.persistent_ctas && tr.pingpong < 4) || tr.streamk)
            return set_why(why, "pingpong is its own staged order");
    }
    {
        int fams = (tr.ilv_early_release ? 1 : 0) + (tr.ilv_early_free ? 1 : 0)
                   + (tr.ilv_split_wait ? 1 : 0) + (tr.hb_sched ? 1 : 0) + (tr.sia_sched ? 1 : 0)
                   + (tr.pingpong ? 1 : 0);
        if(fams > 1)
            return set_why(why, "at most one K-tile schedule family per kernel");
    }
    if(tr.hb_sched && (tr.ilv_read_span || tr.ilv_dtl_span))
        return set_why(why, "hb_sched places its own reads / DTL issues");
    if(tr.pingpong && tr.lds_row_interleave)
        return set_why(why, "pingpong builds its own DTL lane layout");
    if(tr.acc_agpr_pin && !tr.acc_agpr)
        return set_why(why, "acc_agpr_pin requires acc_agpr");
    if(tr.mfma_interleave && !tr.dtl_frag_prefetch)
        return set_why(why, "mfma_interleave requires dtl_frag_prefetch (dtl_stages >= 2)");
    if(tr.ilv_split_wait)
    {
        int64_t sw_n = (t.warp_tile_m ? fdiv(fdiv(t.tile_m, t.warp_m), t.warp_tile_m) : 0)
                       * (t.warp_tile_n ? fdiv(fdiv(t.tile_n, t.warp_n), t.warp_tile_n) : 0);
        if(!tr.mfma_interleave || tr.ilv_early_free)
            return set_why(why, "ilv_split_wait requires mfma_interleave without ilv_early_free");
        if(!(tr.ilv_barrier_after < tr.ilv_split_wait && tr.ilv_split_wait < sw_n))
            return set_why(why, "ilv_split_wait range");
    }
    if(tr.ilv_ef_spread && !tr.ilv_early_free)
        return set_why(why, "ilv_ef_spread requires ilv_early_free");
    if(tr.ilv_early_free)
    {
        int64_t ef_n = (t.warp_tile_m ? fdiv(fdiv(t.tile_m, t.warp_m), t.warp_tile_m) : 0)
                       * (t.warp_tile_n ? fdiv(fdiv(t.tile_n, t.warp_n), t.warp_tile_n) : 0);
        int64_t ef_ka = t.warp_tile_k ? fdiv(t.tile_k, t.warp_tile_k) : 0;
        if(!tr.mfma_interleave || ef_ka < 2)
            return set_why(why,
                           "ilv_early_free requires mfma_interleave and >= 2 K atoms per tile");
        if(!(4 <= tr.ilv_early_free && tr.ilv_early_free < ef_n))
            return set_why(why, "ilv_early_free range");
    }
    if(tr.mfma_interleave && tr.dtl_issue_spread)
        return set_why(why, "mfma_interleave excludes dtl_issue_spread");
    if(tr.acc_agpr)
    {
        if(!t.mfmas_m(&mm) || !t.mfmas_n(&mn))
            return set_why(why, "tile not divisible");
        int64_t regs = mm * mn * ((t.warp_tile_m * t.warp_tile_n) / 64);
        if(spec.wave_size != 64 || regs > 256)
            return set_why(why, "acc_agpr needs wave64 and <= 256 accumulator registers per lane");
    }
    if(tr.mfma_swap_ab)
    {
        if(tr.epilogue != "default" || tr.pad_n || tr.split_k > 1)
            return set_why(
                why,
                "mfma_swap_ab requires the default (direct) epilogue without pad_n or split-K");
        if(t.warp_tile_m != t.warp_tile_n)
            return set_why(why, "mfma_swap_ab requires warp_tile_m == warp_tile_n");
    }
    if(tr.sab_vec16)
    {
        /* _sab_vec16_error */
        if(!tr.mfma_swap_ab)
            return set_why(why, "sab_vec16 requires mfma_swap_ab");
        if(arch != "gfx950")
            return set_why(why, "sab_vec16 needs v_permlane32_swap (gfx950)");
        if(spec.wave_size != 64)
            return set_why(why, "sab_vec16 needs wave64");
        if(!(t.warp_tile_m == 16 && t.warp_tile_n == 16))
            return set_why(why, "sab_vec16 is built for 16x16 atoms");
        if(!t.mfmas_m(&mm))
            return set_why(why, "tile not divisible");
        if(mm % 2)
            return set_why(why, "sab_vec16 pairs M atoms: mfmas_per_warp_m must be even");
        if(d.dtype_c != "fp16" && d.dtype_c != "bf16")
            return set_why(why, "sab_vec16 stores 16-bit C");
        if(spec.batched)
            return set_why(why, "sab_vec16 is built for the non-batched body");
        bool a_mn, b_mn;
        if(!mn_layout_letters_x(d.layout, &a_mn, &b_mn))
            return set_why(why, "unsupported GEMM layout");
        if(!(b_mn && uses_mn_dtl(spec)))
            return set_why(
                why, "sab_vec16 permutes the transpose reads of an N-contiguous B on the DTL ring");
    }
    if(tr.sab_store_nt && !tr.sab_vec16)
        return set_why(why, "sab_store_nt requires sab_vec16");
    /* LDS write granularity (family mma) */
    int64_t dtl_h = std::min(4, target.async_lds_max_dwords) * 2;
    if(tr.direct_to_lds)
    {
        int64_t rows[2] = {t.tile_m, t.tile_n};
        for(int64_t r : rows)
        {
            int64_t chunks = (r * t.tile_k) / dtl_h;
            if(spec.block_size == 0 || chunks % spec.block_size)
                return set_why(why, "direct_to_lds: a partial DMA pass writes past the LDS tile");
        }
    }
    if(tr.lds_swizzle)
    {
        int64_t R, W, L;
        spec_swizzle_params(tr, a_frag, b_frag, t.tile_k, &R, &W, &L);
        int64_t chunk;
        if(tr.direct_to_lds)
            chunk = dtl_h;
        else if(!choose_load_vec(t.tile_m, t.tile_n, t.tile_k, spec.block_size, &chunk))
            return set_why(why, "no usable load_vec");
        if(tr.vgpr_prefetch && !tr.direct_to_lds && L >= 1 && chunk % (int64_t(1) << L) == 0)
            chunk = int64_t(1) << L;
        if(L < 0 || (int64_t(1) << L) % chunk)
            return set_why(why, "lds_swizzle: XOR group narrower than the LDS write chunk");
    }
    int64_t bytes_lds, ab_bytes, c_bytes;
    universal_lds_bytes(spec, target.lds_capacity_bytes, 2, 2, 2, &bytes_lds, &ab_bytes, &c_bytes);
    if(!tr.sched_strategy.empty())
    {
        /* codegen_policy.normalize_scheduler_strategy: exact values only */
        static const std::set<std::string> ok = {"max-ilp",
                                                 "max-memory-clause",
                                                 "iterative-ilp",
                                                 "iterative-minreg",
                                                 "iterative-maxocc"};
        if(!ok.count(tr.sched_strategy))
            return set_why(why, "unsupported scheduler_strategy");
    }
    if(tr.kc_row_guard && tr.direct_to_lds)
        return set_why(why, "kc_row_guard is a VGPR-staged loader lever (no direct_to_lds)");
    std::string mnerr = mn_layout_error(spec, arch, a_sub_perm);
    if(!mnerr.empty())
        return set_why(why, mnerr);
    if(d.layout != "RCR")
    {
        ab_bytes += mn_lds_extra_bytes(spec, a_sub_perm);
        bytes_lds = tr.cshuffle_no_alias ? ab_bytes + c_bytes : std::max(ab_bytes, c_bytes);
    }
    if(tr.lds_row_interleave)
    {
        bool a_mn, b_mn;
        mn_layout_letters_x(d.layout, &a_mn, &b_mn);
        int64_t rpad
            = (((a_mn ? 0 : t.tile_m) + (b_mn ? 0 : t.tile_n)) / 8) * tr.rilv_pad * tr.dtl_stages;
        ab_bytes += rpad;
        bytes_lds = tr.cshuffle_no_alias ? ab_bytes + c_bytes : std::max(ab_bytes, c_bytes);
    }
    if(bytes_lds > target.lds_capacity_bytes)
        return set_why(why, "LDS budget exceeded");
    if(tr.streamk)
    {
        int64_t waves = t.warp_m * t.warp_n * t.warp_k;
        int64_t per_lds = target.lds_capacity_bytes / std::max((int64_t)1, bytes_lds);
        int64_t wpe = 4 / 4;
        if(tr.waves_per_eu.has_value())
            wpe = std::max(wpe, *tr.waves_per_eu);
        int64_t per_reg = std::max((int64_t)1, fdiv(4 * wpe, std::max((int64_t)1, waves)));
        int64_t per_cu = std::min(per_lds, per_reg);
        if(tr.persistent_ctas > current_cus() * per_cu)
            return set_why(why, "streamk needs all persistent_ctas workgroups co-resident");
    }
    if(spec.block_size > target.max_threads_per_block)
        return set_why(why, "block_size above the hardware cap");
    int64_t threads = spec.block_size;
    if(t.tile_m * t.tile_k < threads || t.tile_n * t.tile_k < threads)
        return set_why(why, "block too small for one element/thread/phase");
    int64_t sk = tr.split_k;
    if(sk < 1)
        return set_why(why, "split_k must be >= 1");
    if(tr.dtl_prefetch && !tr.direct_to_lds)
        return set_why(why, "dtl_prefetch requires direct_to_lds");
    if(tr.pad_row_clamp && (!(tr.pad_m || tr.pad_n) || tr.direct_to_lds || tr.preshuffle_b))
        return set_why(why, "pad_row_clamp requires pad_m or pad_n on the global-load body");
    if(tr.pad_k_fast && (!tr.pad_k || tr.direct_to_lds))
        return set_why(why, "pad_k_fast requires pad_k on the global-load body");
    bool ska_ring = tr.direct_to_lds && !tr.pad_k && !tr.dtl_k_tail && !tr.pingpong
                    && tr.asm_loop.empty() && !tr.persistent_ctas;
    if(tr.split_k_aligned
       && (sk < 2 || !((tr.pad_k && !tr.direct_to_lds) || ska_ring) || tr.streamk
           || tr.preshuffle_b))
        return set_why(why, "split_k_aligned constraints");
    if(!(tr.split_k_serial >= 0 && tr.split_k_serial <= 3))
        return set_why(why, "split_k_serial must be 0, 1, 2 or 3");
    bool sks_ring = tr.direct_to_lds && !tr.pingpong && tr.asm_loop.empty();
    if(tr.split_k_serial == 1)
        return set_why(why, "split_k_serial 1 (fenced partials) returned wrong results: use 2");
    if(tr.split_k_serial
       && (sk < 2 || !(tr.vgpr_prefetch || sks_ring) || tr.epilogue != "default" || tr.streamk
           || tr.persistent_ctas || spec.batched || t.warp_k != 1))
        return set_why(why, "split_k_serial constraints");
    if(!(tr.dtl_k_tail >= 0 && tr.dtl_k_tail <= 2))
        return set_why(why, "dtl_k_tail must be 0, 1 or 2");
    if(tr.dtl_k_tail == 2 && tr.pingpong)
        return set_why(why, "dtl_k_tail 2 (peeled tail tile) has no ping-pong form");
    if(tr.dtl_k_tail
       && (!tr.direct_to_lds || tr.dtl_stages < 2 || tr.pingpong == 7 || !tr.asm_loop.empty()
           || tr.streamk || sk > 1 || tr.dtl_clamp_k || spec.batched || tr.k_stagger
           || tr.k_stagger_rc || tr.stagger_u))
        return set_why(why, "dtl_k_tail constraints");
    (void)family;
    return true;
}

/* -------------------------------------------------------------- launch rules */

namespace
{

std::string g1_launch_error(const UGemm& s, int64_t M, int64_t N, int64_t K)
{
    const Tile& t = s.tile;
    const Trait& tr = s.trait;
    bool m_rag = pmod(M, t.tile_m) != 0;
    bool n_rag = pmod(N, t.tile_n) != 0;
    if(m_rag || n_rag)
    {
        if(tr.direct_to_lds)
        {
            if(!tr.dtl_tight_rsrc)
                return "direct_to_lds with ragged M / N needs dtl_tight_rsrc";
            if(tr.dtl_tight_ragged_only && ((m_rag && !tr.pad_m) || (n_rag && !tr.pad_n)))
                return "dtl_tight_ragged_only bounds only the padded operand";
        }
        else if(!(tr.pad_row_clamp || tr.kc_row_guard || s.data.layout != "RCR"
                  || tr.vgpr_prefetch))
            return "ragged M / N on the global-load body needs pad_row_clamp";
    }
    if(n_rag && tr.preshuffle_b)
        return "preshuffle_b needs N % tile_n == 0";
    if(tr.direct_to_lds && pmod(K, t.tile_k))
    {
        if(s.data.layout != "RCR")
        {
            bool a_mn, b_mn;
            mn_layout_letters_x(s.data.layout, &a_mn, &b_mn);
            bool mns[2] = {a_mn, b_mn};
            bool pads[2] = {tr.pad_m, tr.pad_n};
            for(int i = 0; i < 2; ++i)
                if(mns[i] && !(tr.dtl_tight_rsrc && (!tr.dtl_tight_ragged_only || pads[i])))
                    return "direct_to_lds K tail with an MN-contiguous operand needs "
                           "dtl_tight_rsrc on it";
        }
        if(!tr.dtl_k_tail)
            return "direct_to_lds: K % tile_k != 0 needs dtl_k_tail";
        if(pmod(K, DTL_K_TAIL_ALIGN))
            return "dtl_k_tail: K is not a multiple of 8";
    }
    if(K < 1 || M < 1 || N < 1)
        return "empty GEMM";
    if(!tr.persistent_ctas && cdiv(M, t.tile_m) > GRID_YZ_MAX)
        return "row tiles exceed the grid.y limit";
    if(tr.split_k > GRID_YZ_MAX)
        return "split_k exceeds the grid.z limit";
    int64_t ra = M, rb = N;
    if(tr.wide_addr)
    {
        int64_t span_b = t.tile_n;
        if(t.tile_m * K * 2 > DTL_BUFFER_NUM_RECORDS || span_b * K * 2 > DTL_BUFFER_NUM_RECORDS)
            return "wide_addr: an operand tile spans past the DTL buffer range";
        if(t.tile_m * N > (int64_t(1) << 31))
            return "wide_addr: a C tile spans more than 2**31 elements";
        return "";
    }
    if(ra * K > (int64_t(1) << 31) || rb * K > (int64_t(1) << 31) || M * N > (int64_t(1) << 31))
        return "operand elements beyond the i32 element offsets";
    if(tr.direct_to_lds && std::max(ra, rb) * K * 2 > DTL_BUFFER_NUM_RECORDS)
        return "direct_to_lds: an operand exceeds the DTL buffer range";
    return "";
}

} // namespace

} // namespace detail

using namespace detail;

bool mn_layout_letters(const std::string& layout, bool* a_mn, bool* b_mn)
{
    return mn_layout_letters_x(layout, a_mn, b_mn);
}

bool is_valid_spec(const UGemm& spec, const std::string& arch, std::string* why)
{
    return is_valid_spec_x(spec, arch, false, false, why);
}

std::string launch_shape_error(const UGemm& spec, int64_t M, int64_t N, int64_t K)
{
    const Tile& t = spec.tile;
    const Trait& tr = spec.trait;
    if(!tr.pad_m && pmod(M, t.tile_m))
        return "M is not a multiple of tile_m and pad_m is off";
    if(!tr.pad_n && pmod(N, t.tile_n))
        return "N is not a multiple of tile_n and pad_n is off";
    if(!tr.pad_k && pmod(K, t.tile_k))
        return "K is not a multiple of tile_k and pad_k is off";
    if(tr.direct_to_lds && pmod(K, t.tile_k) && !tr.dtl_k_tail)
        return "direct_to_lds: K is not a multiple of tile_k (needs dtl_k_tail)";
    int64_t ra = M, rb = N;
    if(tr.dtl_tight_rsrc && !tr.wide_addr && std::max(ra, rb) * K * 2 >= (int64_t(1) << 31))
        return "dtl_tight_rsrc: operand bytes do not fit the i32 buffer range";
    if(tr.rve_pad && tr.pad_n && pmod(N, 8))
        return "rve_pad: N is not a multiple of 8";
    if(tr.rve_pad == 2 && M * N * 2 >= (int64_t(1) << 32))
        return "rve_pad 2: C bytes do not fit the u32 buffer range";
    std::string lay = spec.data.layout;
    for(auto& c : lay)
        c = (char)toupper((unsigned char)c);
    if(tr.vgpr_prefetch && tr.pad_k && lay != "CRR")
    {
        int64_t lv = 0;
        if(!choose_load_vec(t.tile_m, t.tile_n, t.tile_k, spec.block_size, &lv))
            return "no usable load_vec";
        if(pmod(K, lv))
            return "vgpr_prefetch with pad_k: K is not a multiple of the global load vector";
    }
    if(tr.split_k_serial)
    {
        int64_t tiles = cdiv(M, t.tile_m) * cdiv(N, t.tile_n);
        int64_t wf = tr.split_k_serial == 3 ? M * N : tr.split_k * tiles * t.tile_m * t.tile_n;
        if(wf * 4 >= (int64_t(1) << 31))
            return "split_k_serial: the partial workspace does not fit the i32 slot byte offsets";
    }
    int64_t vp_lim = (tr.pad_m || tr.pad_n) ? (int64_t(1) << 31) : (int64_t(1) << 32);
    if(lay != "RCR" && tr.pad_k)
        vp_lim = int64_t(1) << 31;
    /* (wide_addr: 64-bit tile origins, only a tile's span is checked) */
    if(tr.vgpr_prefetch && !tr.wide_addr && std::max(M, N) * K * 2 >= vp_lim)
        return "vgpr_prefetch: operand bytes do not fit the buffer range";
    if(tr.split_k > 1 && !tr.split_k_aligned)
    {
        if(pmod(K, tr.split_k))
            return "K is not a multiple of split_k";
        if(pmod(fdiv(K, tr.split_k), t.tile_k))
            return "K is not a multiple of split_k * tile_k";
    }
    if(tr.split_k > 1 && tr.split_k_aligned && tr.direct_to_lds)
    {
        if(pmod(K, t.tile_k))
            return "K is not a multiple of tile_k (split_k_aligned ring)";
        int64_t ks_t = cdiv(fdiv(K, t.tile_k), tr.split_k);
        if((tr.split_k - 1) * ks_t >= fdiv(K, t.tile_k))
            return "split_k gives empty aligned slices";
    }
    if(spec.data.layout != "RCR")
    {
        std::string err = mn_launch_error(spec, M, N, K);
        if(!err.empty())
            return err;
    }
    return g1_launch_error(spec, M, N, K);
}

namespace detail
{

/* ---------------------------------------------------------------- support.py */

namespace
{

int64_t dtype_bytes(const std::string& dt, bool* ok)
{
    std::string d = dt;
    for(auto& c : d)
        c = (char)tolower((unsigned char)c);
    if(d == "f16" || d == "half")
        d = "fp16";
    static const std::map<std::string, int64_t> m = {{"fp16", 2},
                                                     {"f16", 2},
                                                     {"bf16", 2},
                                                     {"fp8", 1},
                                                     {"f8", 1},
                                                     {"bf8", 1},
                                                     {"fp8e4m3", 1},
                                                     {"bf8e5m2", 1},
                                                     {"i8", 1},
                                                     {"fp32", 4},
                                                     {"f32", 4},
                                                     {"i32", 4}};
    auto it = m.find(d);
    *ok = it != m.end();
    return *ok ? it->second : 0;
}

/* support.gemm_config_supported */
bool gemm_config_supported(const UGemm& s, const std::string& arch, std::string* why)
{
    ArchFacts target;
    if(!arch_facts(arch, &target))
        return set_why(why, "unknown arch");
    std::string lay = s.data.layout;
    for(auto& c : lay)
        c = (char)toupper((unsigned char)c);
    if(!is_gemm_layout(lay))
        return set_why(why, "unsupported GEMM layout");
    if(s.data.dtype_a != s.data.dtype_b || s.data.dtype_a != s.data.dtype_c)
        return set_why(why, "phase-1 UniversalGemm requires homogeneous A/B/C dtypes");
    std::string acc = s.data.dtype_acc;
    for(auto& c : acc)
        c = (char)tolower((unsigned char)c);
    if(acc == "f16" || acc == "half")
        acc = "fp16";
    if(acc != "fp32")
        return set_why(why, "phase-1 UniversalGemm requires fp32 accumulation");
    const Tile& t = s.tile;
    std::string dtype = s.data.dtype_a;
    for(auto& c : dtype)
        c = (char)tolower((unsigned char)c);
    if(dtype == "f16" || dtype == "half")
        dtype = "fp16";
    if(!supports_dtype(arch, dtype))
        return set_why(why, "unsupported GEMM dtype");
    if(!mma_shape(arch, dtype, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k, nullptr, nullptr))
        return set_why(why, "unsupported warp_tile");
    if(s.wave_size != target.wave_size)
        return set_why(why, "spec wave_size != arch wave_size");
    if(t.warp_m * t.warp_tile_m == 0 || t.tile_m % (t.warp_m * t.warp_tile_m))
        return set_why(why, "tile_m not divisible by warp_m * warp_tile_m");
    if(t.warp_n * t.warp_tile_n == 0 || t.tile_n % (t.warp_n * t.warp_tile_n))
        return set_why(why, "tile_n not divisible by warp_n * warp_tile_n");
    if(t.warp_k * t.warp_tile_k == 0 || t.tile_k % (t.warp_k * t.warp_tile_k))
        return set_why(why, "tile_k not divisible by warp_k * warp_tile_k");
    int64_t expected_bs = t.warp_m * t.warp_n * t.warp_k * s.wave_size;
    if(expected_bs != s.block_size)
        return set_why(why, "block_size != warp_m*warp_n*warp_k*wave_size");
    if(s.block_size > target.max_threads_per_block)
        return set_why(why, "block_size above the hardware cap");
    if(s.trait.scheduler != "intrawave" && s.trait.scheduler != "interwave")
        return set_why(why, "unknown scheduler");
    int64_t lds_cap = target.lds_capacity_bytes;
    bool oka, okb, okc;
    int64_t ea = dtype_bytes(s.data.dtype_a, &oka), eb = dtype_bytes(s.data.dtype_b, &okb),
            ec = dtype_bytes(s.data.dtype_c, &okc);
    if(!s.trait.direct_to_lds && !s.trait.vgpr_prefetch)
    {
        int64_t cap
            = (s.trait.pipeline == "compv4" || s.trait.pipeline == "preshufflev2") ? 32768 : 65536;
        cap = std::min(cap, lds_cap);
        if(!oka || !okb)
            return set_why(why, "unknown GEMM dtype for LDS sizing");
        int64_t used = t.tile_m * t.tile_k * ea + t.tile_n * t.tile_k * eb;
        if(used > cap)
            return set_why(why, "LDS budget (CK estimate) exceeded");
    }
    if(!oka || !okb || !okc)
        return set_why(why, "unknown GEMM dtype for LDS sizing");
    int64_t total, ab, c;
    universal_lds_bytes(s, lds_cap, ea, eb, ec, &total, &ab, &c);
    if(total > lds_cap)
        return set_why(why, "LDS budget exceeded");
    if(t.tile_m * t.tile_k < s.block_size || t.tile_n * t.tile_k < s.block_size)
        return set_why(why, "block too small for one element/thread/phase");
    return true;
}

} // namespace

bool universal_spec_supported(const UGemm& s, const std::string& arch, std::string* why)
{
    if(!gemm_config_supported(s, arch, why))
        return false;
    std::string w;
    if(!is_valid_spec_x(s, arch, false, false, &w))
        return set_why(why, "is_valid_spec: " + w);
    return true;
}

bool request_shape_supported(const Request& r, const UGemm& s, std::string* why)
{
    const Tile& t = s.tile;
    const Trait& tr = s.trait;
    if(!tr.pad_m && pmod(r.M, t.tile_m))
        return set_why(why, "M not divisible by tile_m and pad_m is disabled");
    if(!tr.pad_n && pmod(r.N, t.tile_n))
        return set_why(why, "N not divisible by tile_n and pad_n is disabled");
    if(!tr.pad_k && pmod(r.K, t.tile_k))
        return set_why(why, "K not divisible by tile_k and pad_k is disabled");
    std::string err = launch_shape_error(s, r.M, r.N, r.K);
    if(!err.empty())
        return set_why(why, err);
    std::string lay = s.data.layout;
    for(auto& c : lay)
        c = (char)toupper((unsigned char)c);
    if(!tr.direct_to_lds && lay == "RCR" && !tr.kc_row_guard && !tr.pad_row_clamp
       && !tr.vgpr_prefetch)
    {
        if(pmod(r.M, t.tile_m) || pmod(r.N, t.tile_n))
            return set_why(why, "the unguarded VGPR-staged loader would read past the operand");
    }
    /* operand_range_supported */
    bool ok;
    int64_t eb = dtype_bytes(s.data.dtype_a, &ok);
    if(!ok)
        return set_why(why, "unknown GEMM dtype for LDS sizing");
    int64_t M = r.M, N = r.N, K = r.K;
    if(M * K > (int64_t(1) << 31) || N * K > (int64_t(1) << 31) || M * N > (int64_t(1) << 31))
        return set_why(why, "operand elements beyond the i32 element offsets");
    if(tr.direct_to_lds)
        if(M * K * eb > DTL_BUFFER_NUM_RECORDS || N * K * eb > DTL_BUFFER_NUM_RECORDS)
            return set_why(why, "direct_to_lds: operand beyond the DTL buffer range");
    return true;
}

} // namespace detail
} // namespace gemm_dispatch
} // namespace rocke
