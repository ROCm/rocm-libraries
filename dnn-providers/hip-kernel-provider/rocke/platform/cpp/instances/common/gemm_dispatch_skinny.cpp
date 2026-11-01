/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C++ GEMM dispatcher (rocke/gemm_dispatch.h): the gemm_skinny body -- the spec rules of
 * instances/common/gemm_skinny.py (is_valid_spec, skinny_launch_error, the LDS plan), the
 * configuration tables (dispatch/gemm/skinny_table.py, skinny_native_table.py), the region,
 * plans and candidates of dispatch/gemm/skinny.py and the learned configuration choice of
 * dispatch/gemm/skinny_select.py.
 */
#include "gemm_dispatch_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

namespace rocke
{
namespace gemm_dispatch
{
namespace detail
{

namespace
{

std::string upper(std::string s)
{
    for(auto& c : s)
        c = (char)toupper((unsigned char)c);
    return s;
}

/* ------------------------------------------------------------------- tables */

struct TableRow
{
    const char* dt;
    int mle, bs, bg, bk;
    int64_t v[9];
};

const TableRow k_table[] = {
#define ROCKE_GD_SKINNY_TABLE(dt, mle, bs, bg, bk, a, b, c, d, e, f, g, h, i) \
    {dt, mle, bs, bg, bk, {a, b, c, d, e, f, g, h, i}},
#include "gemm_dispatch_skinny_tables.inc"
#undef ROCKE_GD_SKINNY_TABLE
};

struct NativeRow
{
    const char* arch;
    const char* lay;
    int mle, odd, bs, bg, bk;
    int64_t v[10];
};

const NativeRow k_native[] = {
#define ROCKE_GD_SKINNY_NATIVE_TABLE(                              \
    arch, lay, mle, odd, bs, bg, bk, a, b, c, d, e, f, g, h, i, j) \
    {arch, lay, mle, odd, bs, bg, bk, {a, b, c, d, e, f, g, h, i, j}},
#include "gemm_dispatch_skinny_tables.inc"
#undef ROCKE_GD_SKINNY_NATIVE_TABLE
};

const std::map<std::tuple<std::string, int, int, int, int>, const TableRow*>& table_index()
{
    static const auto m = [] {
        std::map<std::tuple<std::string, int, int, int, int>, const TableRow*> r;
        for(const auto& row : k_table)
            r[{row.dt, row.mle, row.bs, row.bg, row.bk}] = &row;
        return r;
    }();
    return m;
}

const std::map<std::tuple<std::string, std::string, int, int, int, int, int>, const NativeRow*>&
    native_index()
{
    static const auto m = [] {
        std::map<std::tuple<std::string, std::string, int, int, int, int, int>, const NativeRow*> r;
        for(const auto& row : k_native)
            r[{std::string(row.arch) + "|" + row.lay,
               row.lay,
               row.mle,
               row.odd,
               row.bs,
               row.bg,
               row.bk}]
                = &row;
        return r;
    }();
    return m;
}

} // namespace

bool skinny_table_entry(
    int64_t M, int64_t N, int64_t K, const std::string& dtype, std::array<int64_t, 9>* out)
{
    int64_t s = std::min(M, N), g = std::max(M, N);
    int64_t bs = floor_log2(std::max((int64_t)1, s));
    if(bs > 9)
        return false;
    int64_t bg = std::min((int64_t)15, std::max((int64_t)7, floor_log2(std::max((int64_t)1, g))));
    int64_t bk = std::min((int64_t)15, std::max((int64_t)6, floor_log2(std::max((int64_t)1, K))));
    std::string dt = (dtype == "fp16" || dtype == "f16") ? "f16" : "bf16";
    auto it = table_index().find({dt, M <= N ? 1 : 0, (int)bs, (int)bg, (int)bk});
    if(it == table_index().end())
        return false;
    for(int i = 0; i < 9; ++i)
        (*out)[i] = it->second->v[i];
    return true;
}

bool skinny_native_entry(const std::string& arch,
                         int64_t M,
                         int64_t N,
                         int64_t K,
                         const std::string& layout,
                         bool ld_odd,
                         std::array<int64_t, 10>* out)
{
    int64_t s = std::min(M, N), g = std::max(M, N);
    int64_t bs = floor_log2(std::max((int64_t)1, s));
    if(bs > 9)
        return false;
    int64_t bg = std::min((int64_t)15, std::max((int64_t)7, floor_log2(std::max((int64_t)1, g))));
    int64_t bk = std::min((int64_t)15, std::max((int64_t)6, floor_log2(std::max((int64_t)1, K))));
    std::string lay = upper(layout);
    auto it = native_index().find(
        {arch + "|" + lay, lay, M <= N ? 1 : 0, ld_odd ? 1 : 0, (int)bs, (int)bg, (int)bk});
    if(it == native_index().end())
        return false;
    for(int i = 0; i < 10; ++i)
        (*out)[i] = it->second->v[i];
    return true;
}

/* ------------------------------------------------------------- gemm_skinny */

int64_t skinny_lds_bytes(const Skinny& s)
{
    int64_t red = 0;
    if(s.warp_k > 1)
        red = s.warp_k * s.tile_m() * (s.tile_n() + 4) * 4;
    bool a_mn, b_mn;
    s.mn_operands(&a_mn, &b_mn);
    int64_t pieces = s.kc / 8;
    int64_t ea = a_mn ? pieces * 32 * (16 * s.wave_m_atoms + s.mn_pad / 2) : 0;
    int64_t eb = b_mn ? pieces * 32 * (16 * s.wave_n_atoms + s.mn_pad / 2) : 0;
    int64_t mn = s.waves() * (ea + eb) * 2;
    return red + mn + 16;
}

bool skinny_is_valid(const Skinny& s, const std::string& arch, std::string* why)
{
    if(arch != "gfx942" && arch != "gfx950")
        return set_why(why, "gemm_skinny is built for gfx942 / gfx950");
    if(s.dtype != "fp16" && s.dtype != "bf16")
        return set_why(why, "dtype must be fp16 or bf16");
    for(int64_t v : {s.wave_m_atoms, s.wave_n_atoms, s.warp_m, s.warp_n, s.warp_k})
        if(!(1 <= v && v <= 16))
            return set_why(why, "atoms / warps must be in [1, 16]");
    if(s.kc != 8 && s.kc != 16 && s.kc != 32)
        return set_why(why, "kc must be 8, 16 or 32");
    if(s.b_load != "buffer" && s.b_load != "global" && s.b_load != "global_nt")
        return set_why(why, "b_load must be buffer, global or global_nt");
    if(s.red_sync != "agent" && s.red_sync != "sc1")
        return set_why(why, "red_sync must be 'agent' or 'sc1'");
    if(s.kmap != "mfma" && s.kmap != "lane")
        return set_why(why, "kmap must be 'mfma' or 'lane'");
    if(s.kalign != 1 && s.kalign != 2 && s.kalign != 8)
        return set_why(why, "kalign must be 1, 2 or 8");
    if(s.red_unroll != 1 && s.red_unroll != 2 && s.red_unroll != 4 && s.red_unroll != 8)
        return set_why(why, "red_unroll must be 1, 2, 4 or 8");
    if(!(1 <= s.prefetch && s.prefetch <= 4))
        return set_why(why, "prefetch must be in [1, 4]");
    if(!(1 <= s.split && s.split <= 65535))
        return set_why(why, "split must be in [1, 65535]");
    std::string lay = upper(s.layout);
    if(lay != "RCR" && lay != "RRR" && lay != "CRR" && lay != "CCR")
        return set_why(why, "layout must be RCR, RRR, CRR or CCR");
    bool a_mn, b_mn;
    s.mn_operands(&a_mn, &b_mn);
    if(s.mnalign != 1 && s.mnalign != 2 && s.mnalign != 8)
        return set_why(why, "mnalign must be 1, 2 or 8");
    if(s.mn_pad != 0 && s.mn_pad != 16 && s.mn_pad != 32)
        return set_why(why, "mn_pad must be 0, 16 or 32");
    if(s.mn_lds != "tr" && s.mn_lds != "u16")
        return set_why(why, "mn_lds must be 'tr' or 'u16'");
    if(!(a_mn || b_mn))
    {
        if(s.mnalign != 8 || s.mn_pad || s.mn_lds != "tr")
            return set_why(why, "mnalign / mn_pad / mn_lds need an MN-contiguous operand");
    }
    else
    {
        if(s.kmap != "mfma")
            return set_why(why, "an MN-contiguous operand needs kmap='mfma'");
        if(b_mn && s.b_load != "buffer")
            return set_why(why, "an N-contiguous B needs b_load='buffer'");
        if(a_mn && b_mn && s.kalign != 8)
            return set_why(why, "kalign is unused with both operands MN-contiguous (keep 8)");
        if(s.mn_lds == "tr" && arch != "gfx950")
            return set_why(why, "mn_lds='tr' needs ds_read_b64_tr_b16 (gfx950)");
    }
    if(s.mn_uload && (!(a_mn || b_mn) || s.mnalign == 8))
        return set_why(why, "mn_uload needs an MN-contiguous operand with mnalign 1 or 2");
    if(s.mn_uload && arch != "gfx950")
        return set_why(why, "mn_uload is admitted on gfx950 only");
    if(s.block_size() > 1024)
        return set_why(why, "block_size > 1024");
    int64_t acc = s.wave_m_atoms * s.wave_n_atoms * 4;
    int64_t frag = (s.wave_m_atoms + s.wave_n_atoms) * (s.kc / 2);
    int64_t regs = acc + (s.prefetch + 1) * frag;
    if(regs > 384)
        return set_why(why, "register estimate > 384");
    ArchFacts f;
    int64_t cap = arch_facts(arch, &f) ? f.lds_capacity_bytes : 65536;
    if(skinny_lds_bytes(s) > cap)
        return set_why(why, "LDS above the capacity");
    return true;
}

int64_t skinny_steps(const Skinny& s, int64_t K)
{
    return cdiv(K, s.step_k());
}

int64_t skinny_kps(const Skinny& s, int64_t K, int64_t split)
{
    return cdiv(skinny_steps(s, K), std::max((int64_t)1, split));
}

std::string skinny_launch_error(const Skinny& s, int64_t M, int64_t N, int64_t K, int64_t split)
{
    if(std::min({M, N, K}) < 1)
        return "M, N, K must be >= 1";
    if(split < 1 || split > 65535)
        return "split must be in [1, 65535]";
    bool a_mn, b_mn;
    s.mn_operands(&a_mn, &b_mn);
    if(pmod(K, s.kalign) && !(a_mn && b_mn))
        return "K is not a multiple of kalign";
    bool mns[2] = {a_mn, b_mn};
    int64_t lds[2] = {M, N};
    int64_t tiles[2] = {s.tile_m(), s.tile_n()};
    for(int i = 0; i < 2; ++i)
    {
        if(mns[i] && s.mn_uload && lds[i] < tiles[i])
            return "mn_uload: the leading dimension is below the tile";
        if(mns[i] && pmod(lds[i], s.mnalign))
            return "MN-contiguous leading dimension is not a multiple of mnalign";
    }
    int64_t gx = cdiv(N, s.tile_n()), gy = cdiv(M, s.tile_m());
    if(gy > 65535)
        return "grid.y > 65535";
    int64_t kc_rows = 0;
    bool have_kc = false;
    if(!a_mn)
    {
        kc_rows = std::max(kc_rows, s.tile_m());
        have_kc = true;
    }
    if(!b_mn)
    {
        kc_rows = std::max(kc_rows, s.tile_n());
        have_kc = true;
    }
    if(have_kc && kc_rows * K * 2 >= (int64_t(1) << 31))
        return "tile rows x K x 2 bytes must fit in i32";
    if(a_mn || b_mn)
    {
        int64_t rows = (skinny_kps(s, K, split) + 1) * s.step_k();
        for(int i = 0; i < 2; ++i)
            if(mns[i] && (rows * lds[i] + 64) * 2 >= (int64_t(1) << 31))
                return "K slice x MN leading dimension x 2 bytes must fit in i32";
    }
    if(s.tile_m() * N * 2 >= (int64_t(1) << 31))
        return "tile_m x N x 2 bytes must fit in i32";
    int64_t ws = split <= 1 ? 0 : split * gx * gy * s.tile_m() * s.tile_n();
    if(ws >= (int64_t(1) << 31))
        return "workspace too large for i32 indexing";
    if(cdiv(K, s.step_k()) >= (int64_t(1) << 31) / s.step_k())
        return "K too large";
    return "";
}

/* ----------------------------------------------------------------- skinny.py */

int64_t skinny_kalign(int64_t K)
{
    return pmod(K, 8) == 0 ? 8 : (pmod(K, 2) == 0 ? 2 : 1);
}

int64_t skinny_split(const Skinny& spec, int64_t M, int64_t N, int64_t K, int64_t waves_target)
{
    if(waves_target <= 0)
        return 1;
    int64_t gx = cdiv(N, spec.tile_n()), gy = cdiv(M, spec.tile_m());
    int64_t waves = gx * gy * spec.waves();
    int64_t steps = cdiv(K, spec.step_k());
    int64_t s = pround((double)waves_target / (double)std::max((int64_t)1, waves));
    return std::max((int64_t)1,
                    std::min({(int64_t)256, s, std::max((int64_t)1, fdiv(steps, spec.warp_k))}));
}

namespace
{

void small_tile(int64_t s, int64_t* atoms, int64_t* warps)
{
    if(s <= 16)
        *atoms = 1, *warps = 1;
    else if(s <= 32)
        *atoms = 2, *warps = 1;
    else if(s <= 64)
        *atoms = 4, *warps = 1;
    else if(s <= 128)
        *atoms = 4, *warps = 2;
    else
        *atoms = 4, *warps = 4;
}

Skinny fallback_plan(int64_t M, int64_t N, int64_t K, const std::string& dtype)
{
    int64_t s = std::min(M, N), g = std::max(M, N);
    if(s > 128)
    {
        Skinny spec;
        spec.dtype = dtype;
        spec.wave_m_atoms = 4;
        spec.wave_n_atoms = 4;
        spec.warp_m = 2;
        spec.warp_n = 2;
        spec.kc = 16;
        spec.prefetch = 2;
        spec.red_sync = "sc1";
        spec.kalign = skinny_kalign(K);
        int64_t tiles = cdiv(M, 128) * cdiv(N, 128);
        int64_t split = 1;
        if(tiles < 256)
        {
            int64_t steps = cdiv(K, spec.step_k());
            int64_t q = fdiv(steps, 8);
            split = std::max((int64_t)1,
                             std::min({(int64_t)64, fdiv(256, tiles), q ? q : (int64_t)1}));
        }
        spec.split = split;
        spec.red_unroll = split >= 16 ? 8 : 4;
        return spec;
    }
    int64_t sa, sw;
    small_tile(s, &sa, &sw);
    int64_t kc = K <= 384 ? 32 : 16;
    int64_t steps = cdiv(K, 4 * kc);
    int64_t pf = steps <= 16 ? 4 : 2;
    while(pf > 1 && sa * 4 * 4 + fdiv((pf + 1) * (sa + 4) * kc, 2) > 384)
        pf -= 1;
    int64_t stiles = cdiv(s, 16 * sa * sw);
    int64_t ab = 1;
    for(int64_t c : {2, 4})
        if(stiles * cdiv(g, 16 * c) * sw >= 512)
            ab = c;
    int64_t tiles = stiles * cdiv(g, 16 * ab);
    int64_t wk = 1;
    if(tiles * sw < 256 && steps >= 16)
        wk = tiles * sw >= 64 ? 2 : 4;
    int64_t ma, wm, na, wn;
    if(M <= N)
        ma = sa, wm = sw, na = ab, wn = 1;
    else
        ma = ab, wm = 1, na = sa, wn = sw;
    Skinny spec;
    spec.dtype = dtype;
    spec.wave_m_atoms = ma;
    spec.wave_n_atoms = na;
    spec.warp_m = wm;
    spec.warp_n = wn;
    spec.warp_k = wk;
    spec.kc = kc;
    spec.prefetch = pf;
    spec.red_sync = "sc1";
    spec.kalign = skinny_kalign(K);
    int64_t target = tiles * sw * wk < 2048 / 2 ? 2048 : 0;
    int64_t split = skinny_split(spec, M, N, K, target);
    int64_t q = fdiv(steps, std::max((int64_t)1, wk * 8));
    split = std::max((int64_t)1, std::min(split, q ? q : (int64_t)1));
    spec.split = split;
    spec.red_unroll = split >= 16 ? 8 : 4;
    return spec;
}

bool req_ld_odd(const Request& req)
{
    std::string lay = upper(req.layout);
    bool a_mn = !lay.empty() && lay[0] == 'C', b_mn = lay.size() > 1 && lay[1] == 'R';
    return (a_mn && pmod(req.M, 8)) || (b_mn && pmod(req.N, 8));
}

bool native_table_on(const Request& req)
{
    return upper(req.layout) != "RCR" && (req.arch == "gfx950" || req.arch == "gfx942");
}

Skinny native_split_fit(const Skinny& spec, int64_t M, int64_t N, int64_t K)
{
    bool a_mn, b_mn;
    spec.mn_operands(&a_mn, &b_mn);
    int64_t ld = 1;
    bool any = false;
    if(a_mn)
    {
        ld = M;
        any = true;
    }
    if(b_mn)
        ld = any ? std::max(ld, N) : N;
    int64_t kps_max = fdiv((int64_t(1) << 31) - 1, 2 * spec.step_k() * ld) - 2;
    if(kps_max < 1)
        return spec;
    int64_t steps = cdiv(K, spec.step_k());
    int64_t split = std::max(spec.split, cdiv(steps, kps_max));
    if(split == spec.split)
        return spec;
    Skinny o = spec;
    o.split = split;
    o.red_unroll = split >= 16 ? 8 : spec.red_unroll;
    return o;
}

} // namespace

bool skinny_plan_ok(const Skinny& s, const std::string& arch, int64_t M, int64_t N, int64_t K)
{
    return skinny_is_valid(s, arch, nullptr) && skinny_launch_error(s, M, N, K, s.split).empty();
}

Skinny
    skinny_native_fit(const Skinny& spec, const std::string& arch, int64_t M, int64_t N, int64_t K)
{
    if(upper(spec.layout) == "RCR" || skinny_plan_ok(spec, arch, M, N, K))
        return spec;
    for(int v = 0; v < 3; ++v)
    {
        Skinny alt = spec;
        if(v >= 1)
            alt.mn_pad = 0;
        if(v == 2)
            alt.kc = 8;
        if(!skinny_is_valid(alt, arch, nullptr))
            continue;
        alt = native_split_fit(alt, M, N, K);
        if(skinny_plan_ok(alt, arch, M, N, K))
            return alt;
    }
    return spec;
}

Skinny skinny_native(const Skinny& spec, const Request& req)
{
    std::string lay = upper(req.layout);
    if(lay == "RCR")
        return spec;
    bool a_mn = lay[0] == 'C', b_mn = lay[1] == 'R';
    std::vector<int64_t> lds;
    if(a_mn)
        lds.push_back(req.M);
    if(b_mn)
        lds.push_back(req.N);
    bool all8 = true, all2 = true;
    for(int64_t x : lds)
    {
        all8 = all8 && pmod(x, 8) == 0;
        all2 = all2 && pmod(x, 2) == 0;
    }
    int64_t mna = all8 ? 8 : (all2 ? 2 : 1);
    Skinny out = spec;
    out.layout = lay;
    out.mnalign = mna;
    out.mn_pad = 16;
    out.mn_lds = req.arch == "gfx950" ? "tr" : "u16";
    out.kalign = (a_mn && b_mn) ? 8 : spec.kalign;
    out.kmap = "mfma";
    out.b_load = b_mn ? "buffer" : spec.b_load;
    if(mna < 8 && req.arch == "gfx950")
    {
        bool ok = true;
        if(a_mn && !(req.M >= out.tile_m()))
            ok = false;
        if(b_mn && !(req.N >= out.tile_n()))
            ok = false;
        if(ok)
            out.mn_uload = true;
    }
    return out;
}

namespace
{

Skinny spec_from_entry(const std::string& dt,
                       int64_t ma,
                       int64_t na,
                       int64_t wm,
                       int64_t wn,
                       int64_t wk,
                       int64_t kc,
                       int64_t pf,
                       int64_t K)
{
    Skinny s;
    s.dtype = dt;
    s.wave_m_atoms = ma;
    s.wave_n_atoms = na;
    s.warp_m = wm;
    s.warp_n = wn;
    s.warp_k = wk;
    s.kc = kc;
    s.prefetch = pf;
    s.red_sync = "sc1";
    s.kalign = skinny_kalign(K);
    return s;
}

int64_t table_split(const Skinny& spec, int64_t tw, int64_t wk, int64_t M, int64_t N, int64_t K)
{
    int64_t split = 1;
    if(tw)
    {
        int64_t tiles = cdiv(M, spec.tile_m()) * cdiv(N, spec.tile_n());
        int64_t steps = cdiv(K, spec.step_k());
        split = std::max(
            (int64_t)1,
            std::min({(int64_t)256,
                      pround((double)tw / (double)std::max((int64_t)1, tiles * spec.waves())),
                      std::max((int64_t)1, fdiv(steps, wk))}));
    }
    return split;
}

/* skinny_native_table_plan (false: None) */
bool skinny_native_table_plan(const Request& req, Skinny* out)
{
    int64_t M = req.M, N = req.N, K = req.K;
    std::array<int64_t, 10> e;
    if(!skinny_native_entry(req.arch, M, N, K, req.layout, req_ld_odd(req), &e))
        return false;
    std::string dt = (req.dtype == "fp16" || req.dtype == "f16") ? "fp16" : "bf16";
    Skinny spec = spec_from_entry(dt, e[0], e[1], e[2], e[3], e[4], e[5], e[6], K);
    int64_t split = table_split(spec, e[7], e[4], M, N, K);
    spec.split = split;
    spec.red_unroll = split >= 16 ? 8 : 4;
    spec = skinny_native_fit(skinny_native(spec, req), req.arch, M, N, K);
    if(!skinny_plan_ok(spec, req.arch, M, N, K))
        return false;
    *out = spec;
    return true;
}

/* skinny_table.skinny_table_plan (false: None) */
bool skinny_table_plan(
    const std::string& arch, int64_t M, int64_t N, int64_t K, const std::string& dtype, Skinny* out)
{
    if(arch != "gfx950" && arch != "gfx942")
        return false;
    std::array<int64_t, 9> e;
    if(!skinny_table_entry(M, N, K, dtype, &e))
        return false;
    std::string dt = (dtype == "fp16" || dtype == "f16") ? "fp16" : "bf16";
    Skinny spec = spec_from_entry(dt, e[0], e[1], e[2], e[3], e[4], e[5], e[6], K);
    int64_t split = table_split(spec, e[7], e[4], M, N, K);
    spec.split = split;
    spec.red_unroll = split >= 16 ? 8 : 4;
    *out = spec;
    return true;
}

} // namespace

Skinny skinny_plan(const Request& req, const Context& ctx)
{
    int64_t M = req.M, N = req.N, K = req.K;
    std::string dtype = (req.dtype == "fp16" || req.dtype == "f16") ? "fp16" : "bf16";
    Skinny spec;
    bool have = false;
    if(native_table_on(req))
        have = skinny_native_table_plan(req, &spec);
    if(!have)
        have = skinny_table_plan(req.arch, M, N, K, dtype, &spec);
    if(have)
        spec = skinny_native_fit(skinny_native(spec, req), req.arch, M, N, K);
    if(!have || !skinny_plan_ok(spec, req.arch, M, N, K))
        spec = skinny_native_fit(
            skinny_native(fallback_plan(M, N, K, dtype), req), req.arch, M, N, K);
    for(int64_t prefetch : {2, 1})
    {
        if(skinny_plan_ok(spec, req.arch, M, N, K))
            break;
        spec.wave_m_atoms = 1;
        spec.wave_n_atoms = 1;
        spec.warp_m = 1;
        spec.warp_n = 1;
        spec.warp_k = 1;
        spec.kc = 16;
        spec.prefetch = prefetch;
    }
    if(skinny_plan_ok(spec, req.arch, M, N, K))
        spec = skinny_select(req, ctx, spec);
    return spec;
}

/* ------------------------------------------------------------- admission */

namespace
{

bool native_tiles2_admit(const Request& req)
{
    int64_t tiles = cdiv(req.M, 128) * cdiv(req.N, 128);
    if(req_ld_odd(req))
    {
        /* _odd_dtl_native: on gfx950 */
        if(req.arch != "gfx950")
            return true;
        if(tiles < 32)
            return true;
        if(req.K < 96)
            return true;
        if(tiles < 64 && req.K >= 4096)
            return true;
        return false;
    }
    if(tiles < 8)
        return true;
    if(tiles < 48 && req.K >= 2048)
        return true;
    return false;
}

bool skinny_admit_region(const Request& req)
{
    if(req.arch != "gfx950" && req.arch != "gfx942")
        return false;
    int64_t s = std::min(req.M, req.N);
    if(native_table_on(req))
    {
        if(s <= 32)
            return true;
        if(req.arch == "gfx950")
            return native_tiles2_admit(req);
        return false; /* gfx942 rules are not mirrored (the tensor GETT is gfx950 only) */
    }
    if(req.arch != "gfx950")
        return false;
    std::array<int64_t, 9> e;
    if(s <= 32)
    {
        bool have = skinny_table_entry(req.M, req.N, req.K, req.dtype, &e);
        if(have && e[8] < 100 && ring_small_takes(req))
            return false;
        return true;
    }
    bool mid = s > 256;
    if(s > 1023)
        return false;
    if(!skinny_table_entry(req.M, req.N, req.K, req.dtype, &e))
        return false;
    int64_t bound = mid ? 105 : 95;
    if(e[8] < bound)
        return false;
    if(mid && e[8] < 110 && ring_small_takes(req))
        return false;
    return true;
}

} // namespace

Candidate make_skinny_candidate(const std::string& name,
                                const std::string& dtype,
                                bool fallback,
                                const Context& ctx)
{
    Candidate c;
    c.name = name;
    c.algorithm = "gemm_skinny";
    c.spec_id = fallback ? "cdna_skinny_fallback" : "cdna_skinny_splitk";
    c.priority = fallback ? 99 : 3;
    c.arches = {"gfx942", "gfx950"};
    c.layouts = {"RCR", "RRR", "CRR", "CCR"};
    Context cx = ctx;
    std::string dt = dtype;
    c.admits = [cx, dt, fallback](const Request& req, std::string* why) {
        if(req.arch != "gfx942" && req.arch != "gfx950")
            return set_why(why, "capability");
        if(req.M <= 0 || req.N <= 0 || req.K <= 0 || req.dtype != dt)
            return set_why(why, "request errors");
        std::string lay = upper(req.layout);
        if(lay != "RCR" && lay != "RRR" && lay != "CRR" && lay != "CCR")
            return set_why(why, "request errors");
        if(!fallback && !skinny_admit_region(req))
            return set_why(why, "outside the gemm_skinny region");
        Skinny spec = skinny_plan(req, cx);
        if(!skinny_is_valid(spec, req.arch, why))
            return false;
        std::string err = skinny_launch_error(spec, req.M, req.N, req.K, spec.split);
        if(!err.empty())
            return set_why(why, err);
        return true;
    };
    auto admits = c.admits;
    c.select = [cx, admits](const Request& req, Spec* out, std::string* why) {
        if(!admits(req, why))
            return false;
        out->skinny = true;
        out->s = skinny_plan(req, cx);
        return true;
    };
    auto select = c.select;
    c.sweep = [select, admits](const Request& req) {
        std::vector<Spec> out;
        if(!admits(req, nullptr))
            return out;
        Spec base;
        if(!select(req, &base, nullptr))
            return out;
        out.push_back(base);
        for(int64_t s : {std::max((int64_t)1, fdiv(base.s.split, 2)), base.s.split * 2})
        {
            Skinny v = base.s;
            v.split = s;
            if(s != base.s.split && skinny_is_valid(v, req.arch, nullptr))
            {
                v.red_unroll = s >= 16 ? 8 : 4;
                Spec sp;
                sp.skinny = true;
                sp.s = v;
                out.push_back(sp);
            }
        }
        return out;
    };
    return c;
}

/* ---------------------------------------------------------- skinny_select */

namespace
{

double l2(double x)
{
    return std::log2(std::max(1.0, x));
}

std::vector<double>
    skinny_features(const std::string& arch, int64_t M, int64_t N, int64_t K, const Skinny& spec)
{
    int64_t cus = arch == "gfx942" ? 304 : 256;
    bool a_mn, b_mn;
    spec.mn_operands(&a_mn, &b_mn);
    int64_t tm = spec.tile_m(), tn = spec.tile_n();
    int64_t gx = cdiv(N, tn), gy = cdiv(M, tm);
    int64_t tiles = gx * gy;
    int64_t split = std::max((int64_t)1, spec.split);
    int64_t waves = tiles * spec.waves() * split;
    int64_t steps = cdiv(K, spec.step_k());
    double sps
        = std::max(1.0, (double)cdiv(steps, split) / (double)std::max((int64_t)1, spec.warp_k));
    int64_t load
        = tiles * spec.warp_m * spec.warp_n * 16 * (spec.wave_m_atoms + spec.wave_n_atoms) * K * 2;
    int64_t ws = split > 1 ? split * tiles * tm * tn * 4 : 0;
    int64_t acc = spec.wave_m_atoms * spec.wave_n_atoms * 4;
    int64_t frag = (spec.wave_m_atoms + spec.wave_n_atoms) * (spec.kc / 2);
    int64_t regs = acc + (spec.prefetch + 1) * frag;
    return {arch == "gfx942" ? 1.0 : 0.0,
            spec.dtype == "bf16" ? 1.0 : 0.0,
            a_mn ? 1.0 : 0.0,
            b_mn ? 1.0 : 0.0,
            spec.mnalign < 8 ? 1.0 : 0.0,
            spec.kalign < 8 ? 1.0 : 0.0,
            l2((double)M),
            l2((double)N),
            l2((double)K),
            l2((double)std::min(M, N)),
            l2((double)std::max(M, N)),
            M <= N ? 1.0 : 0.0,
            (double)spec.wave_m_atoms,
            (double)spec.wave_n_atoms,
            (double)spec.warp_m,
            (double)spec.warp_n,
            (double)spec.warp_k,
            (double)spec.kc,
            (double)spec.prefetch,
            l2((double)split),
            (double)tm,
            (double)tn,
            l2((double)tiles),
            l2((double)waves),
            (double)waves / (double)cus,
            std::log2(sps),
            (double)(gy * tm) / (double)M,
            (double)(gx * tn) / (double)N,
            l2((double)load),
            l2((double)(1 + ws)),
            (double)regs,
            (double)skinny_lds_bytes(spec) / 1024.0,
            (spec.mn_lds == "u16" && (a_mn || b_mn)) ? 1.0 : 0.0,
            (double)spec.mn_pad,
            spec.red_unroll == 8 ? 1.0 : 0.0,
            l2(2.0 * (double)M * (double)N * (double)K),
            l2((double)(2 * (M * K + N * K + M * N)))};
}

std::array<int64_t, 7> base_key(const Skinny& s)
{
    return {s.wave_m_atoms, s.wave_n_atoms, s.warp_m, s.warp_n, s.warp_k, s.kc, s.prefetch};
}

std::vector<Skinny> skinny_menu(const Request& req, const Skinny& plan, const SkinnyModel& model)
{
    int64_t M = req.M, N = req.N, K = req.K;
    std::string dt = (req.dtype == "fp16" || req.dtype == "f16") ? "fp16" : "bf16";
    std::vector<Skinny> out = {plan};
    std::vector<std::pair<std::array<int64_t, 7>, int64_t>> seen = {{base_key(plan), plan.split}};
    std::vector<std::array<int64_t, 7>> bases;
    for(const auto& b : model.menu)
    {
        std::array<int64_t, 7> a{};
        for(size_t i = 0; i < 7 && i < b.size(); ++i)
            a[i] = b[i];
        bases.push_back(a);
    }
    if(std::find(bases.begin(), bases.end(), base_key(plan)) == bases.end())
        bases.push_back(base_key(plan));
    for(const auto& b : bases)
    {
        Skinny rcr = spec_from_entry(dt, b[0], b[1], b[2], b[3], b[4], b[5], b[6], K);
        for(int64_t tw : {0, 256, 512, 1024, 2048, 4096})
        {
            int64_t s = tw ? skinny_split(rcr, M, N, K, tw) : 1;
            Skinny sp = rcr;
            sp.split = s;
            sp.red_unroll = s >= 16 ? 8 : 4;
            sp = skinny_native_fit(skinny_native(sp, req), req.arch, M, N, K);
            std::pair<std::array<int64_t, 7>, int64_t> key = {base_key(sp), sp.split};
            if(std::find(seen.begin(), seen.end(), key) != seen.end()
               || !skinny_plan_ok(sp, req.arch, M, N, K))
                continue;
            seen.push_back(key);
            out.push_back(sp);
        }
    }
    return out;
}

} // namespace

Skinny skinny_select(const Request& req, const Context& ctx, const Skinny& plan)
{
    if(req.arch != "gfx950" && req.arch != "gfx942")
        return plan;
    if(std::min(req.M, req.N) < 17)
        return plan;
    int64_t tiles = cdiv(req.M, 128) * cdiv(req.N, 128);
    int64_t max_tiles = req.arch == "gfx950" ? 1024 : 256;
    if(tiles > max_tiles)
        return plan;
    if(ctx.models == nullptr)
        return plan;
    auto it = ctx.models->skinny.find(req.arch);
    if(it == ctx.models->skinny.end())
        return plan;
    const SkinnyModel& m = it->second;
    std::vector<Skinny> cands = skinny_menu(req, plan, m);
    if(cands.size() <= 1)
        return plan;
    std::vector<std::vector<double>> X;
    for(const auto& c : cands)
        X.push_back(skinny_features(req.arch, req.M, req.N, req.K, c));
    std::vector<double> p0;
    for(const auto& x : X)
        p0.push_back(m.model.predict_packed(x));
    std::vector<size_t> idx(cands.size() - 1);
    std::iota(idx.begin(), idx.end(), 1);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return p0[a] < p0[b]; });
    std::vector<size_t> order = {0};
    for(size_t i = 0; i < idx.size() && i < 8; ++i)
        order.push_back(idx[i]);
    size_t T = order.size();
    std::vector<std::vector<double>> P;
    {
        std::vector<double> r;
        for(size_t j : order)
            r.push_back(p0[j]);
        P.push_back(r);
    }
    for(const auto& b : m.bags)
    {
        std::vector<double> r;
        for(size_t j : order)
            r.push_back(b.predict_packed(X[j]));
        P.push_back(r);
    }
    const double thr = std::log1p(0.03);
    double best = -std::numeric_limits<double>::infinity();
    int bestj = -1;
    for(size_t j = 0; j < T; ++j)
    {
        bool allpos = true;
        double sum = 0.0;
        for(size_t mi = 0; mi < P.size(); ++mi)
        {
            double g = P[mi][0] - P[mi][j];
            allpos = allpos && (g > 0);
            sum += g;
        }
        double mean = sum / (double)P.size();
        bool ok = allpos && mean >= thr && j != 0;
        double v = ok ? mean : -std::numeric_limits<double>::infinity();
        if(ok && (bestj < 0 || v > best))
        {
            best = v;
            bestj = (int)j;
        }
    }
    if(bestj < 0)
        return plan;
    return cands[order[bestj]];
}

} // namespace detail
} // namespace gemm_dispatch
} // namespace rocke
