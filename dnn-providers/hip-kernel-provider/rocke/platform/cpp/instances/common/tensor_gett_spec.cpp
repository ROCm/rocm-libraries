// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tensor_gett_spec.cpp -- C++ mirror of the family spec of
 * rocke/instances/common/tensor/gett.py (TensorGettSpec, is_valid_spec,
 * build_tensor_gett, tensor_gett_grid, launch_operands, body_spec) and of the
 * tuned-route planning of rocke/dispatch/families/tensor_gett.py
 * (tensor_gett_route_key, the tuned table, _tuned_plan / _plan_mn / _spec_mn)
 * with dispatch/gemm/selector.parse_config. See rocke/instance_tensor_gett.h.
 */
#include "rocke/instance_tensor_gett.h"

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"
#include "rocke/json_dom.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <sstream>

namespace rocke
{
namespace gett
{
namespace
{

/* ----------------------------------------------------------------- hashes */

uint32_t rol(uint32_t x, int n)
{
    return (x << n) | (x >> (32 - n));
}
uint32_t ror(uint32_t x, int n)
{
    return (x >> n) | (x << (32 - n));
}

std::vector<uint8_t> md_pad(const std::string& s)
{
    std::vector<uint8_t> m(s.begin(), s.end());
    uint64_t bits = (uint64_t)s.size() * 8;
    m.push_back(0x80);
    while(m.size() % 64 != 56)
        m.push_back(0);
    for(int i = 7; i >= 0; --i)
        m.push_back((uint8_t)(bits >> (8 * i)));
    return m;
}

std::string hex_words(const uint32_t* h, int n)
{
    static const char* d = "0123456789abcdef";
    std::string out;
    for(int i = 0; i < n; ++i)
        for(int s = 28; s >= 0; s -= 4)
            out.push_back(d[(h[i] >> s) & 0xF]);
    return out;
}

/* ------------------------------------------------------------------- JSON */

std::string jstr(const std::string& s)
{
    /* json.dumps(str) with ensure_ascii */
    std::string o = "\"";
    for(unsigned char c : s)
    {
        if(c == '"')
            o += "\\\"";
        else if(c == '\\')
            o += "\\\\";
        else if(c == '\n')
            o += "\\n";
        else if(c == '\r')
            o += "\\r";
        else if(c == '\t')
            o += "\\t";
        else if(c < 0x20 || c >= 0x7F)
        {
            char buf[8];
            snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
        }
        else
            o.push_back((char)c);
    }
    return o + "\"";
}

std::string jbool(bool b)
{
    return b ? "true" : "false";
}

/* an object of already-rendered members, keys sorted (sort_keys=True) */
std::string
    jobj(std::vector<std::pair<std::string, std::string>> m, const char* sep, const char* kv)
{
    std::sort(m.begin(), m.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string o = "{";
    for(size_t i = 0; i < m.size(); ++i)
        o += (i ? sep : "") + jstr(m[i].first) + kv + m[i].second;
    return o + "}";
}

std::string jlist(const std::vector<std::string>& items, const char* sep)
{
    std::string o = "[";
    for(size_t i = 0; i < items.size(); ++i)
        o += (i ? sep : "") + items[i];
    return o + "]";
}

std::string jmodes(const std::vector<std::string>& v, const char* sep)
{
    std::vector<std::string> items;
    for(const auto& s : v)
        items.push_back(jstr(s));
    return jlist(items, sep);
}

std::string jmodemap(const ModeMap& m, const char* sep)
{
    std::vector<std::string> items;
    for(const auto& kv : m)
        items.push_back(jlist({jstr(kv.first), std::to_string(kv.second)}, sep));
    return jlist(items, sep);
}

/* the Python TraitSpec fields and their defaults (json.dumps text) */
const std::pair<const char*, const char*> k_trait_fields[] = {
#include "gemm_traitspec_fields.inc"
};

std::string jopt_int(bool set, int v)
{
    return set ? std::to_string(v) : "null";
}
std::string jopt_bool(bool set, bool v)
{
    return set ? jbool(v) : "null";
}

/* dataclasses.asdict(TraitSpec) as json.dumps(sort_keys=True) */
std::string trait_json(const rocke_gemm_trait_spec_t& t)
{
    std::map<std::string, std::string> v;
    for(const auto& f : k_trait_fields)
        v[f.first] = f.second;
    auto I = [](int x) { return std::to_string(x); };
    auto S = [](const char* s) { return s == nullptr ? std::string("null") : jstr(s); };
    v["pipeline"] = S(t.pipeline);
    v["scheduler"] = S(t.scheduler);
    v["epilogue"] = S(t.epilogue);
    v["pad_m"] = jbool(t.pad_m);
    v["pad_n"] = jbool(t.pad_n);
    v["pad_k"] = jbool(t.pad_k);
    v["persistent"] = jbool(t.persistent);
    v["chiplet_swizzle"] = jbool(t.chiplet_swizzle);
    v["chiplet_wgm"] = I(t.chiplet_wgm);
    v["chiplet_num_xcds"] = I(t.chiplet_num_xcds);
    v["chiplet_chunk_size"] = I(t.chiplet_chunk_size);
    v["waves_per_eu"] = jopt_int(t.waves_per_eu_set, t.waves_per_eu);
    v["preshuffle_b"] = jbool(t.preshuffle_b);
    v["direct_to_lds"] = jbool(t.direct_to_lds);
    v["dtl_cache_a"] = I(t.dtl_cache_a);
    v["dtl_cache_b"] = I(t.dtl_cache_b);
    v["dtl_prefetch"] = jbool(t.dtl_prefetch);
    v["active_tile_skip"] = jbool(t.active_tile_skip);
    v["lds_k_pad"] = I(t.lds_k_pad);
    v["lds_swizzle"] = jbool(t.lds_swizzle);
    v["emit_sched_hints"] = jopt_bool(t.emit_sched_hints_set, t.emit_sched_hints);
    v["split_k"] = I(t.split_k);
    v["cshuffle_no_alias"] = jbool(t.cshuffle_no_alias);
    v["ab_double_buffer"] = jopt_bool(t.ab_double_buffer_set, t.ab_double_buffer);
    v["interwave_setprio"] = jbool(t.interwave_setprio);
    v["dtl_stages"] = I(t.dtl_stages);
    v["dtl_issue_spread"] = jbool(t.dtl_issue_spread);
    v["dtl_frag_prefetch"] = jbool(t.dtl_frag_prefetch);
    v["dtl_stream_barrier"] = jbool(t.dtl_stream_barrier);
    v["mfma_swap_ab"] = jbool(t.mfma_swap_ab);
    v["iglp_opt"] = jopt_int(t.iglp_opt_set, t.iglp_opt);
    v["persistent_ctas"] = I(t.persistent_ctas);
    v["acc_agpr"] = jbool(t.acc_agpr);
    v["mfma_interleave"] = jbool(t.mfma_interleave);
    v["acc_agpr_pin"] = jbool(t.acc_agpr_pin);
    v["ilv_dtl_every"] = I(t.ilv_dtl_every);
    v["ilv_barrier_after"] = I(t.ilv_barrier_after);
    v["ilv_merge"] = jbool(t.ilv_merge);
    v["lds_swizzle_r"] = I(t.lds_swizzle_r);
    v["dtl_hoist"] = jbool(t.dtl_hoist);
    v["streamk"] = jbool(t.streamk);
    v["dtl_tight_rsrc"] = jbool(t.dtl_tight_rsrc);
    v["pad_fast_tile"] = jbool(t.pad_fast_tile);
    v["ilv_early_release"] = I(t.ilv_early_release);
    v["ilv_release_at"] = I(t.ilv_release_at);
    v["ilv_acquire_at"] = I(t.ilv_acquire_at);
    v["acc_out_vgpr"] = jbool(t.acc_out_vgpr);
    v["ilv_read_span"] = I(t.ilv_read_span);
    v["ilv_dtl_span"] = I(t.ilv_dtl_span);
    v["lds_row_interleave"] = jbool(t.lds_row_interleave);
    v["rilv_pad"] = I(t.rilv_pad ? t.rilv_pad : 16);
    v["rilv_vec_epilogue"] = jbool(t.rilv_vec_epilogue);
    v["rve_store_nt"] = jbool(t.rve_store_nt);
    v["dtl_soff0"] = jbool(t.dtl_soff0);
    v["wide_addr"] = jbool(t.wide_addr);
    v["mn_pad"] = I(t.mn_pad_set ? t.mn_pad : 32);
    v["mn_chunk"] = I(t.mn_chunk);
    v["mn_ld_any"] = jbool(t.mn_ld_any);
    std::vector<std::pair<std::string, std::string>> m(v.begin(), v.end());
    return jobj(m, ", ", ": ");
}

std::string tile_json(const rocke_gemm_tile_spec_t& t)
{
    auto I = [](int x) { return std::to_string(x); };
    return jobj({{"tile_m", I(t.tile_m)},
                 {"tile_n", I(t.tile_n)},
                 {"tile_k", I(t.tile_k)},
                 {"warp_m", I(t.warp_m)},
                 {"warp_n", I(t.warp_n)},
                 {"warp_k", I(t.warp_k)},
                 {"warp_tile_m", I(t.warp_tile_m)},
                 {"warp_tile_n", I(t.warp_tile_n)},
                 {"warp_tile_k", I(t.warp_tile_k)}},
                ", ",
                ": ");
}

std::string knobs_json(const Knobs& k)
{
    std::vector<std::pair<std::string, std::string>> m = {
        {"pad", std::to_string(k.pad)},
        {"bilinear", jbool(k.bilinear)},
        {"epi_order", jstr(k.epi_order)},
        {"epi_pair", jbool(k.epi_pair)},
        {"lane_split", jbool(k.lane_split)},
        {"acc_pin", std::to_string(k.acc_pin)},
        {"frag_pin", jbool(k.frag_pin)},
        {"llvm_opts", jstr(k.llvm_opts)},
    };
    /* fields added after the first release are hashed only when set */
    if(k.epi_v16)
        m.push_back({"epi_v16", jbool(k.epi_v16)});
    if(k.c_prefetch != 0)
        m.push_back({"c_prefetch", std::to_string(k.c_prefetch)});
    if(k.store_nt)
        m.push_back({"store_nt", jbool(k.store_nt)});
    return jobj(m, ", ", ": ");
}

std::string short_dt(const std::string& dtype)
{
    return dtype == "f16" || dtype == "fp16" ? "f16" : dtype;
}

std::string gemm_dt(const std::string& dtype)
{
    return dtype == "f16" || dtype == "fp16" ? "fp16" : dtype;
}

std::map<std::string, int64_t> as_map(const ModeMap& m)
{
    return std::map<std::string, int64_t>(m.begin(), m.end());
}

bool set_why(std::string* why, const std::string& m)
{
    if(why != nullptr)
        *why = m;
    return false;
}

} // namespace

/* ------------------------------------------------------------------ hashes */

std::string sha1_hex(const std::string& s)
{
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    std::vector<uint8_t> m = md_pad(s);
    for(size_t off = 0; off < m.size(); off += 64)
    {
        uint32_t w[80];
        for(int i = 0; i < 16; ++i)
            w[i] = (uint32_t)m[off + 4 * i] << 24 | (uint32_t)m[off + 4 * i + 1] << 16
                   | (uint32_t)m[off + 4 * i + 2] << 8 | (uint32_t)m[off + 4 * i + 3];
        for(int i = 16; i < 80; ++i)
            w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for(int i = 0; i < 80; ++i)
        {
            uint32_t f, k;
            if(i < 20)
                f = (b & c) | (~b & d), k = 0x5A827999u;
            else if(i < 40)
                f = b ^ c ^ d, k = 0x6ED9EBA1u;
            else if(i < 60)
                f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDCu;
            else
                f = b ^ c ^ d, k = 0xCA62C1D6u;
            uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rol(b, 30);
            b = a;
            a = t;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    return hex_words(h, 5);
}

std::string sha256_hex(const std::string& s)
{
    static const uint32_t K[64]
        = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
           0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
           0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
           0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
           0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
           0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
           0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
           0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
           0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
           0xc67178f2};
    uint32_t h[8] = {0x6a09e667,
                     0xbb67ae85,
                     0x3c6ef372,
                     0xa54ff53a,
                     0x510e527f,
                     0x9b05688c,
                     0x1f83d9ab,
                     0x5be0cd19};
    std::vector<uint8_t> m = md_pad(s);
    for(size_t off = 0; off < m.size(); off += 64)
    {
        uint32_t w[64];
        for(int i = 0; i < 16; ++i)
            w[i] = (uint32_t)m[off + 4 * i] << 24 | (uint32_t)m[off + 4 * i + 1] << 16
                   | (uint32_t)m[off + 4 * i + 2] << 8 | (uint32_t)m[off + 4 * i + 3];
        for(int i = 16; i < 64; ++i)
        {
            uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for(int i = 0; i < 64; ++i)
        {
            uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    return hex_words(h, 8);
}

/* ------------------------------------------------------------ body specs */

rocke_gemm_trait_spec_t mn_body_trait(const rocke_gemm_trait_spec_t& trait)
{
    rocke_gemm_trait_spec_t t = trait;
    t.epilogue = "default";
    t.cshuffle_no_alias = false;
    t.pad_m = false;
    t.pad_n = false;
    t.dtl_tight_rsrc = false;
    t.pad_fast_tile = false;
    t.dtl_soff0 = false;
    return t;
}

rocke_gemm_universal_spec_t body_spec(const char* dtype, int tile_k)
{
    rocke_gemm_universal_spec_t s = rocke_gemm_universal_spec_default();
    s.name = "tensor_gett";
    s.tile.tile_m = 256;
    s.tile.tile_n = 256;
    s.tile.tile_k = tile_k;
    s.tile.warp_m = 2;
    s.tile.warp_n = 2;
    s.tile.warp_k = 1;
    s.tile.warp_tile_m = 16;
    s.tile.warp_tile_n = 16;
    s.tile.warp_tile_k = 32;
    s.trait = body_trait();
    std::string dt = gemm_dt(dtype != nullptr ? dtype : "fp16");
    const char* d = dt == "fp16" ? "fp16" : dt == "bf16" ? "bf16" : "fp16";
    s.data.dtype_a = d;
    s.data.dtype_b = d;
    s.data.dtype_c = d;
    rocke_gemm_universal_spec_finalize(&s);
    return s;
}

/* ------------------------------------------------------------ TensorGettSpec */

TensorGettSpec TensorGettSpec::make(const std::vector<std::string>& modes_a,
                                    const std::vector<std::string>& modes_b,
                                    const std::vector<std::string>& modes_d,
                                    const std::map<std::string, int64_t>& extents)
{
    TensorGettSpec s;
    s.modes_a = modes_a;
    s.modes_b = modes_b;
    s.modes_d = modes_d;
    s.extents = ModeMap(extents.begin(), extents.end()); /* sorted by name */
    return s;
}

bool TensorGettSpec::plan(Plan* out, std::string* why) const
{
    std::map<std::string, int64_t> sa, sb, sd;
    if(has_strides_a)
        sa = as_map(strides_a);
    if(has_strides_b)
        sb = as_map(strides_b);
    if(has_strides_d)
        sd = as_map(strides_d);
    return plan_gett(modes_a,
                     modes_b,
                     modes_d,
                     as_map(extents),
                     has_strides_a ? &sa : nullptr,
                     has_strides_b ? &sb : nullptr,
                     has_strides_d ? &sd : nullptr,
                     tile_k,
                     false,
                     'A',
                     out,
                     why);
}

std::string TensorGettSpec::json_key() const
{
    const char* sep = ", ";
    std::vector<std::pair<std::string, std::string>> m = {
        {"modes_a", jmodes(modes_a, sep)},
        {"modes_b", jmodes(modes_b, sep)},
        {"modes_d", jmodes(modes_d, sep)},
        {"extents", jmodemap(extents, sep)},
        {"strides_a", has_strides_a ? jmodemap(strides_a, sep) : "null"},
        {"strides_b", has_strides_b ? jmodemap(strides_b, sep) : "null"},
        {"strides_d", has_strides_d ? jmodemap(strides_d, sep) : "null"},
        {"dtype", jstr(dtype)},
        {"knobs", knobs_json(knobs)},
        {"tile_k", std::to_string(tile_k)},
        {"dtl_stages", std::to_string(dtl_stages)},
        {"persistent_ctas", std::to_string(persistent_ctas)},
        {"chiplet_wgm", std::to_string(chiplet_wgm)},
        {"chiplet_chunk_size", std::to_string(chiplet_chunk_size)},
    };
    if(has_mn_tile)
        m.push_back({"mn_tile", tile_json(mn_tile)});
    if(has_mn_trait)
        m.push_back({"mn_trait", trait_json(mn_trait)});
    return jobj(m, ", ", ": ");
}

std::string TensorGettSpec::kernel_name() const
{
    return "tensor_gett_" + short_dt(dtype) + "_" + sha1_hex(json_key()).substr(0, 12);
}

rocke_gemm_universal_spec_t TensorGettSpec::body(std::string* name_store) const
{
    *name_store = kernel_name();
    rocke_gemm_universal_spec_t s;
    if(has_mn_tile)
    {
        s = rocke_gemm_universal_spec_default();
        s.tile = mn_tile;
        s.trait
            = mn_body_trait(has_mn_trait ? mn_trait : rocke_gemm_universal_spec_default().trait);
        std::string dt = gemm_dt(dtype);
        const char* d = dt == "bf16" ? "bf16" : "fp16";
        s.data.dtype_a = d;
        s.data.dtype_b = d;
        s.data.dtype_c = d;
        s.data.layout = "CRR";
        s.block_size = 0;
        rocke_gemm_universal_spec_finalize(&s);
    }
    else
    {
        s = body_spec(dtype.c_str(), tile_k);
        s.trait.dtl_stages = dtl_stages;
        s.trait.persistent_ctas = persistent_ctas;
        s.trait.chiplet_wgm = chiplet_wgm;
        s.trait.chiplet_chunk_size = chiplet_chunk_size;
        s.block_size = 0;
        rocke_gemm_universal_spec_finalize(&s);
    }
    s.name = name_store->c_str();
    return s;
}

bool is_valid_spec(const TensorGettSpec& spec, const char* arch, std::string* why)
{
    if(arch == nullptr || strcmp(arch, "gfx950") != 0)
        return set_why(why,
                       std::string("arch '") + (arch ? arch : "")
                           + "' not in ('gfx950',) (ds_read_b64_tr_b16)");
    if(!(spec.dtype == "fp16" || spec.dtype == "bf16" || spec.dtype == "f16"))
        return set_why(why, "dtype '" + spec.dtype + "' not in ('fp16', 'bf16')");
    if(!(spec.tile_k == 32 || spec.tile_k == 64))
        return set_why(why, "tile_k " + std::to_string(spec.tile_k) + " not in (32, 64)");
    const Knobs& k = spec.knobs;
    if(!(k.pad == 0 || k.pad == 32))
        return set_why(why, "knobs.pad " + std::to_string(k.pad) + " not in (0, 32)");
    if(!(k.epi_order == "mi" || k.epi_order == "ni"))
        return set_why(why, "knobs.epi_order '" + k.epi_order + "' not in ('mi', 'ni')");
    if(!(k.acc_pin == 0 || k.acc_pin == 1 || k.acc_pin == 2))
        return set_why(why, "knobs.acc_pin " + std::to_string(k.acc_pin) + " not in (0, 1, 2)");
    if(!(0 <= k.c_prefetch && k.c_prefetch <= 8))
        return set_why(why, "knobs.c_prefetch " + std::to_string(k.c_prefetch) + " not in 0..8");
    if(spec.has_mn_tile)
    {
        if(spec.mn_tile.tile_k != spec.tile_k)
            return set_why(why,
                           "tile_k " + std::to_string(spec.tile_k) + " != the MN body's tile_k "
                               + std::to_string(spec.mn_tile.tile_k));
        if(spec.dtl_stages != 3)
            return set_why(why, "dtl_stages is a ring-body field; set it on mn_trait");
        if(spec.persistent_ctas != 0)
            return set_why(why, "persistent_ctas is a ring-body field; set it on mn_trait");
        if(spec.chiplet_wgm != 8)
            return set_why(why, "chiplet_wgm is a ring-body field; set it on mn_trait");
        if(spec.chiplet_chunk_size != 64)
            return set_why(why, "chiplet_chunk_size is a ring-body field; set it on mn_trait");
    }
    else if(spec.has_mn_trait)
        return set_why(why, "mn_trait without mn_tile");
    Plan plan;
    std::string pw;
    if(!spec.plan(&plan, &pw))
        return set_why(why, "no GETT plan: " + pw);
    for(int o = 0; o < 2; ++o)
        if(plan.ops[o].nbytes > MAX_TENSOR_BYTES)
            return set_why(why, "operand " + plan.ops[o].name + " spans more than 2 GiB");
    int64_t span = 0;
    for(const auto& m : plan.modes_d)
        span += (plan.ext.at(m) - 1) * plan.strides_d.at(m);
    if(2 * (1 + span) > MAX_D_BYTES)
        return set_why(
            why, "D spans more than " + std::to_string(MAX_D_BYTES) + " bytes (32-bit offsets)");
    std::string name;
    rocke_gemm_universal_spec_t body = spec.body(&name);
    rocke_gett_hooks hooks(plan, k, body, spec.dtype.c_str());
    if(!hooks.ok())
        return set_why(why, hooks.error());
    char reason[512] = {0};
    if(!rocke_gemm_universal_is_valid_spec(&body, arch, reason, sizeof reason))
        return set_why(why, reason);
    return true;
}

rocke_kernel_def_t*
    build_tensor_gett(rocke_ir_builder_t* b, const TensorGettSpec& spec, const char* arch)
{
    std::string why;
    std::string err;
    if(!is_valid_spec(spec, arch, &why))
        err = "invalid tensor GETT spec: " + why;
    else if(!spec.knobs.llvm_opts.empty())
        err = "knobs.llvm_opts (the llvm_options kernel attr) is implemented by the Python engine "
              "only (no C++ mirror yet)";
    if(!err.empty())
    {
        std::string name = spec.kernel_name();
        if(rocke_ir_builder_init(b, name.c_str()) != ROCKE_OK)
            return nullptr;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, err.c_str());
        return nullptr;
    }
    Plan plan;
    spec.plan(&plan, &why);
    std::string bname;
    rocke_gemm_universal_spec_t body = spec.body(&bname);
    rocke_gett_hooks hooks(plan, spec.knobs, body, spec.dtype.c_str());
    body._gett = &hooks;
    /* the IR builder is named by the body's kernel_name(), as in Python */
    return rocke_build_universal_gemm_new(b, &body, arch);
}

void tensor_gett_grid(const TensorGettSpec& spec, int grid[3])
{
    Plan plan;
    std::string why;
    spec.plan(&plan, &why);
    grid[1] = grid[2] = 1;
    if(spec.has_mn_tile)
    {
        std::string name;
        rocke_gemm_universal_spec_t body = spec.body(&name);
        int64_t tiles_n = (plan.N + body.tile.tile_n - 1) / body.tile.tile_n;
        int64_t tiles_m = (plan.M + body.tile.tile_m - 1) / body.tile.tile_m;
        int pc = body.trait.persistent_ctas;
        if(pc)
        {
            int64_t n = std::min<int64_t>(pc, tiles_n * tiles_m);
            grid[0] = (int)std::max<int64_t>(1, n);
            return;
        }
        grid[0] = (int)tiles_n;
        grid[1] = (int)tiles_m;
        return;
    }
    if(spec.persistent_ctas)
    {
        grid[0] = spec.persistent_ctas;
        return;
    }
    grid[0] = (int)((plan.N + 255) / 256);
    grid[1] = (int)((plan.M + 255) / 256);
}

int tensor_gett_block(const TensorGettSpec& spec)
{
    std::string name;
    return spec.body(&name).block_size;
}

std::pair<std::string, std::string> launch_operands(const TensorGettSpec& spec)
{
    Plan plan;
    std::string why;
    spec.plan(&plan, &why);
    return {plan.ops[0].name, plan.ops[1].name};
}

/* ------------------------------------------------------------ parse_config */

namespace
{

/* Python selector._parse_value */
struct PVal
{
    enum Kind
    {
        BOOL,
        NONE,
        INT,
        FLOAT,
        STR
    } k;
    bool b = false;
    long long i = 0;
    std::string s;
};

PVal parse_value(const std::string& v)
{
    PVal p;
    if(v == "True" || v == "False")
    {
        p.k = PVal::BOOL;
        p.b = v == "True";
        return p;
    }
    if(v == "None")
    {
        p.k = PVal::NONE;
        return p;
    }
    {
        /* int(v): optional sign, digits (Python also strips whitespace) */
        size_t a = 0;
        std::string t = v;
        if(!t.empty() && (t[0] == '-' || t[0] == '+'))
            a = 1;
        bool digits = a < t.size();
        for(size_t i = a; i < t.size(); ++i)
            digits = digits && isdigit((unsigned char)t[i]);
        if(digits)
        {
            p.k = PVal::INT;
            p.i = std::stoll(t);
            return p;
        }
    }
    {
        char* end = nullptr;
        (void)strtod(v.c_str(), &end);
        if(!v.empty() && end != nullptr && *end == '\0')
        {
            p.k = PVal::FLOAT;
            p.s = v;
            return p;
        }
    }
    p.k = PVal::STR;
    p.s = v;
    return p;
}

bool parse_uint(const std::string& s, size_t* pos, int* out)
{
    size_t a = *pos;
    while(*pos < s.size() && isdigit((unsigned char)s[*pos]))
        ++*pos;
    if(*pos == a)
        return false;
    *out = std::stoi(s.substr(a, *pos - a));
    return true;
}

bool expect(const std::string& s, size_t* pos, char c)
{
    if(*pos < s.size() && s[*pos] == c)
    {
        ++*pos;
        return true;
    }
    return false;
}

std::string strip(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, e - a + 1);
}

/* the strings a trait refers to must outlive it */
const char* intern(const std::string& s)
{
    static std::vector<std::unique_ptr<std::string>> pool;
    for(const auto& p : pool)
        if(*p == s)
            return p->c_str();
    pool.emplace_back(new std::string(s));
    return pool.back()->c_str();
}

bool set_field(rocke_gemm_trait_spec_t* t, const std::string& k, const PVal& v, std::string* why)
{
    auto B = [&](bool* f) {
        if(v.k != PVal::BOOL && v.k != PVal::INT)
            return set_why(why, "trait " + k + ": not a bool");
        *f = v.k == PVal::BOOL ? v.b : v.i != 0;
        return true;
    };
    auto I = [&](int* f) {
        if(v.k == PVal::BOOL)
            *f = v.b ? 1 : 0;
        else if(v.k == PVal::INT)
            *f = (int)v.i;
        else
            return set_why(why, "trait " + k + ": not an int");
        return true;
    };
    auto OI = [&](bool* set, int* f) {
        if(v.k == PVal::NONE)
        {
            *set = false;
            return true;
        }
        *set = true;
        return I(f);
    };
    auto OB = [&](bool* set, bool* f) {
        if(v.k == PVal::NONE)
        {
            *set = false;
            return true;
        }
        *set = true;
        return B(f);
    };
    auto S = [&](const char** f) {
        if(v.k != PVal::STR)
            return set_why(why, "trait " + k + ": not a string");
        *f = intern(v.s);
        return true;
    };
    if(k == "pipeline")
        return S(&t->pipeline);
    if(k == "scheduler")
        return S(&t->scheduler);
    if(k == "epilogue")
        return S(&t->epilogue);
    if(k == "pad_m")
        return B(&t->pad_m);
    if(k == "pad_n")
        return B(&t->pad_n);
    if(k == "pad_k")
        return B(&t->pad_k);
    if(k == "persistent")
        return B(&t->persistent);
    if(k == "chiplet_swizzle")
        return B(&t->chiplet_swizzle);
    if(k == "chiplet_wgm")
        return I(&t->chiplet_wgm);
    if(k == "chiplet_num_xcds")
        return I(&t->chiplet_num_xcds);
    if(k == "chiplet_chunk_size")
        return I(&t->chiplet_chunk_size);
    if(k == "waves_per_eu")
        return OI(&t->waves_per_eu_set, &t->waves_per_eu);
    if(k == "preshuffle_b")
        return B(&t->preshuffle_b);
    if(k == "direct_to_lds")
        return B(&t->direct_to_lds);
    if(k == "dtl_cache_a")
        return I(&t->dtl_cache_a);
    if(k == "dtl_cache_b")
        return I(&t->dtl_cache_b);
    if(k == "dtl_prefetch")
        return B(&t->dtl_prefetch);
    if(k == "active_tile_skip")
        return B(&t->active_tile_skip);
    if(k == "lds_k_pad")
        return I(&t->lds_k_pad);
    if(k == "lds_swizzle")
        return B(&t->lds_swizzle);
    if(k == "emit_sched_hints")
        return OB(&t->emit_sched_hints_set, &t->emit_sched_hints);
    if(k == "split_k")
        return I(&t->split_k);
    if(k == "cshuffle_no_alias")
        return B(&t->cshuffle_no_alias);
    if(k == "ab_double_buffer")
        return OB(&t->ab_double_buffer_set, &t->ab_double_buffer);
    if(k == "interwave_setprio")
        return B(&t->interwave_setprio);
    if(k == "dtl_stages")
        return I(&t->dtl_stages);
    if(k == "dtl_issue_spread")
        return B(&t->dtl_issue_spread);
    if(k == "dtl_frag_prefetch")
        return B(&t->dtl_frag_prefetch);
    if(k == "dtl_stream_barrier")
        return B(&t->dtl_stream_barrier);
    if(k == "mfma_swap_ab")
        return B(&t->mfma_swap_ab);
    if(k == "iglp_opt")
        return OI(&t->iglp_opt_set, &t->iglp_opt);
    if(k == "persistent_ctas")
        return I(&t->persistent_ctas);
    if(k == "acc_agpr")
        return B(&t->acc_agpr);
    if(k == "mfma_interleave")
        return B(&t->mfma_interleave);
    if(k == "acc_agpr_pin")
        return B(&t->acc_agpr_pin);
    if(k == "ilv_dtl_every")
        return I(&t->ilv_dtl_every);
    if(k == "ilv_barrier_after")
        return I(&t->ilv_barrier_after);
    if(k == "ilv_merge")
        return B(&t->ilv_merge);
    if(k == "lds_swizzle_r")
        return I(&t->lds_swizzle_r);
    if(k == "dtl_hoist")
        return B(&t->dtl_hoist);
    if(k == "streamk")
        return B(&t->streamk);
    if(k == "dtl_tight_rsrc")
        return B(&t->dtl_tight_rsrc);
    if(k == "pad_fast_tile")
        return B(&t->pad_fast_tile);
    if(k == "ilv_early_release")
        return I(&t->ilv_early_release);
    if(k == "ilv_release_at")
        return I(&t->ilv_release_at);
    if(k == "ilv_acquire_at")
        return I(&t->ilv_acquire_at);
    if(k == "acc_out_vgpr")
        return B(&t->acc_out_vgpr);
    if(k == "ilv_read_span")
        return I(&t->ilv_read_span);
    if(k == "ilv_dtl_span")
        return I(&t->ilv_dtl_span);
    if(k == "lds_row_interleave")
        return B(&t->lds_row_interleave);
    if(k == "rilv_pad")
        return I(&t->rilv_pad);
    if(k == "rilv_vec_epilogue")
        return B(&t->rilv_vec_epilogue);
    if(k == "rve_store_nt")
        return B(&t->rve_store_nt);
    if(k == "dtl_soff0")
        return B(&t->dtl_soff0);
    if(k == "wide_addr")
        return B(&t->wide_addr);
    if(k == "mn_pad")
    {
        t->mn_pad_set = true;
        return I(&t->mn_pad);
    }
    if(k == "mn_chunk")
        return I(&t->mn_chunk);
    if(k == "mn_ld_any")
        return B(&t->mn_ld_any);
    for(const auto& f : k_trait_fields)
        if(k == f.first)
            return set_why(why,
                           "TraitSpec " + k
                               + ": implemented by the Python engine only (no C++ mirror yet)");
    return set_why(why, "TraitSpec has no field '" + k + "'");
}

} // namespace

bool parse_config(const std::string& cfg_in,
                  rocke_gemm_tile_spec_t* tile,
                  rocke_gemm_trait_spec_t* trait,
                  std::string* why)
{
    std::string cfg = strip(cfg_in);
    size_t pos = 0;
    int v[9];
    const char* seps = "xx_xx_xx";
    bool ok = expect(cfg, &pos, 't');
    for(int i = 0; ok && i < 9; ++i)
    {
        ok = parse_uint(cfg, &pos, &v[i]);
        if(ok && i < 8)
        {
            ok = expect(cfg, &pos, seps[i]);
            if(ok && i == 2)
                ok = expect(cfg, &pos, 'w');
            if(ok && i == 5)
                ok = expect(cfg, &pos, 'a');
        }
    }
    if(!ok)
        return set_why(why, "not a gemm config string: '" + cfg_in + "'");
    expect(cfg, &pos, '|');
    std::string rest = cfg.substr(pos);
    rocke_gemm_universal_spec_t d = rocke_gemm_universal_spec_default();
    *tile = d.tile;
    tile->tile_m = v[0];
    tile->tile_n = v[1];
    tile->tile_k = v[2];
    tile->warp_m = v[3];
    tile->warp_n = v[4];
    tile->warp_k = v[5];
    tile->warp_tile_m = v[6];
    tile->warp_tile_n = v[7];
    tile->warp_tile_k = v[8];
    *trait = d.trait;
    size_t a = 0;
    while(!rest.empty() && a <= rest.size())
    {
        size_t e = rest.find(',', a);
        std::string kv = rest.substr(a, e == std::string::npos ? std::string::npos : e - a);
        if(!kv.empty())
        {
            size_t eq = kv.find('=');
            std::string key = kv.substr(0, eq);
            std::string val = eq == std::string::npos ? std::string() : kv.substr(eq + 1);
            if(!set_field(trait, key, parse_value(val), why))
                return false;
        }
        if(e == std::string::npos)
            break;
        a = e + 1;
    }
    return true;
}

/* ------------------------------------------------------------ tuned table */

namespace
{

struct Arena
{
    rocke_arena_t a{};
    Arena()
    {
        rocke_arena_init(&a, (size_t)1 << 20);
    }
    ~Arena()
    {
        rocke_arena_destroy(&a);
    }
};

bool str_list(const jd_val_t* v, std::vector<std::string>* out)
{
    if(v == nullptr || v->kind != JD_ARR)
        return false;
    for(int i = 0; i < v->arr_len; ++i)
    {
        const char* s = rocke_jstr(v->arr[i]);
        if(s == nullptr)
            return false;
        out->push_back(s);
    }
    return true;
}

} // namespace

bool tuned_entries(const std::string& json_text,
                   const std::string& arch,
                   std::vector<TunedEntry>* out,
                   std::string* why)
{
    Arena arena;
    char err[512] = {0};
    jd_val_t* root = rocke_json_parse(json_text.c_str(), &arena.a, err, sizeof err);
    if(root == nullptr)
        return set_why(why, std::string("tuned GETT table: ") + err);
    out->clear();
    const jd_val_t* entries = rocke_jget(root, "entries");
    const jd_val_t* table = entries != nullptr ? rocke_jget(entries, arch.c_str()) : nullptr;
    if(table == nullptr || table->kind != JD_OBJ)
        return true;
    for(int i = 0; i < table->obj_len; ++i)
    {
        TunedEntry e;
        e.key = table->obj[i].key;
        const jd_val_t* ev = table->obj[i].val;
        const char* route = rocke_jstr(rocke_jget(ev, "route"));
        const char* body = rocke_jstr(rocke_jget(ev, "body"));
        const jd_val_t* pr = rocke_jget(ev, "problem");
        if(route == nullptr || body == nullptr || pr == nullptr)
            return set_why(why, "tuned GETT entry " + e.key + ": route / body / problem missing");
        e.route = route;
        e.body = body;
        const char* dt = rocke_jstr(rocke_jget(pr, "dtype"));
        const jd_val_t* bil = rocke_jget(pr, "bilinear");
        const jd_val_t* ext = rocke_jget(pr, "extents");
        if(dt == nullptr || bil == nullptr || bil->kind != JD_BOOL || ext == nullptr
           || ext->kind != JD_OBJ || !str_list(rocke_jget(pr, "modes_a"), &e.modes_a)
           || !str_list(rocke_jget(pr, "modes_b"), &e.modes_b)
           || !str_list(rocke_jget(pr, "modes_d"), &e.modes_d))
            return set_why(why, "tuned GETT entry " + e.key + ": malformed problem");
        e.dtype = dt;
        e.bilinear = bil->b;
        for(int j = 0; j < ext->obj_len; ++j)
        {
            double n = 0;
            if(!rocke_jnum(ext->obj[j].val, &n))
                return set_why(why, "tuned GETT entry " + e.key + ": extent is not a number");
            e.extents[ext->obj[j].key] = (int64_t)n;
        }
        out->push_back(e);
    }
    std::sort(out->begin(), out->end(), [](const TunedEntry& a, const TunedEntry& b) {
        return a.key < b.key;
    });
    return true;
}

TensorGettSpec tuned_problem(const TunedEntry& entry)
{
    TensorGettSpec s
        = TensorGettSpec::make(entry.modes_a, entry.modes_b, entry.modes_d, entry.extents);
    s.dtype = entry.dtype == "f16" || entry.dtype == "fp16" ? "fp16" : entry.dtype;
    s.knobs.bilinear = entry.bilinear;
    return s;
}

std::string route_key(const TensorGettSpec& p)
{
    std::map<std::string, std::string> nm;
    std::vector<std::string> all = p.modes_d;
    all.insert(all.end(), p.modes_a.begin(), p.modes_a.end());
    all.insert(all.end(), p.modes_b.begin(), p.modes_b.end());
    for(const auto& m : all)
        if(nm.find(m) == nm.end())
        {
            std::string x = "x" + std::to_string(nm.size());
            nm[m] = x;
        }
    auto names = [&](const std::vector<std::string>& v) {
        std::vector<std::string> out;
        for(const auto& m : v)
            out.push_back(jstr(nm.at(m)));
        return jlist(out, ",");
    };
    std::vector<std::pair<std::string, int64_t>> ext;
    for(const auto& kv : p.extents)
        if(nm.find(kv.first) != nm.end())
            ext.push_back({nm.at(kv.first), kv.second});
    std::sort(ext.begin(), ext.end());
    std::vector<std::string> exts;
    for(const auto& e : ext)
        exts.push_back(jlist({jstr(e.first), std::to_string(e.second)}, ","));
    auto strides = [&](bool has, const ModeMap& st) -> std::string {
        if(!has)
            return "null";
        std::vector<std::pair<std::string, int64_t>> v;
        for(const auto& kv : st)
            if(nm.find(kv.first) != nm.end())
                v.push_back({nm.at(kv.first), kv.second});
        std::sort(v.begin(), v.end());
        std::vector<std::string> items;
        for(const auto& e : v)
            items.push_back(jlist({jstr(e.first), std::to_string(e.second)}, ","));
        return jlist(items, ",");
    };
    std::string dt = p.dtype == "f16" || p.dtype == "fp16" ? "fp16" : p.dtype;
    std::string body = jobj({{"dtype", jstr(dt)},
                             {"bilinear", jbool(p.knobs.bilinear)},
                             {"a", names(p.modes_a)},
                             {"b", names(p.modes_b)},
                             {"d", names(p.modes_d)},
                             {"ext", jlist(exts, ",")},
                             {"sa", strides(p.has_strides_a, p.strides_a)},
                             {"sb", strides(p.has_strides_b, p.strides_b)},
                             {"sd", strides(p.has_strides_d, p.strides_d)}},
                            ",",
                            ":");
    return sha256_hex(body).substr(0, 20);
}

bool tuned_plan(const TunedEntry& entry,
                const std::string& arch,
                TensorGettSpec* out,
                std::string* why)
{
    TensorGettSpec p = tuned_problem(entry);
    if(entry.route != "mn")
        return set_why(
            why,
            "tuned GETT route '" + entry.route
                + "': only the 'mn' route is mirrored by the C++ engine (the K-contiguous "
                  "'kc' bodies are Python-engine only)");
    if(!(p.dtype == "fp16" || p.dtype == "bf16"))
        return set_why(why, "tuned GETT entry dtype '" + p.dtype + "'");
    rocke_gemm_tile_spec_t tile;
    rocke_gemm_trait_spec_t trait;
    if(!parse_config(entry.body, &tile, &trait, why))
        return false;
    /* _spec_mn(req, [(tile, trait)]) */
    TensorGettSpec sp = p;
    sp.knobs.bilinear = p.knobs.bilinear;
    sp.knobs.c_prefetch = p.knobs.bilinear ? 2 : 0; /* TENSOR_GETT_MN_C_PREFETCH */
    sp.tile_k = tile.tile_k;
    sp.dtl_stages = 3;
    sp.persistent_ctas = 0;
    sp.chiplet_wgm = 8;
    sp.chiplet_chunk_size = 64;
    sp.has_mn_tile = true;
    sp.mn_tile = tile;
    sp.has_mn_trait = true;
    sp.mn_trait = trait;
    TensorGettSpec v16 = sp;
    v16.knobs.epi_v16 = true; /* TENSOR_GETT_MN_EPI_V16 */
    std::string w1, w2;
    if(is_valid_spec(v16, arch.c_str(), &w1))
        *out = v16;
    else if(is_valid_spec(sp, arch.c_str(), &w2))
        *out = sp;
    else
        return set_why(why,
                       "tuned GETT entry " + entry.key
                           + " gives no single-launch MN plan (the pack-hybrid fallback is "
                             "Python-engine only): "
                           + w2);
    return true;
}

} // namespace gett
} // namespace rocke
