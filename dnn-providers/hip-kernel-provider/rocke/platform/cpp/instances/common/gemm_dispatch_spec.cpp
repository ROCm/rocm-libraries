/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C++ GEMM dispatcher (rocke/gemm_dispatch.h): the spec mirrors (TileSpec / TraitSpec /
 * UniversalGemmSpec / SkinnyGemmSpec field access, JSON and config strings of
 * dispatch/gemm/selector.py) and the arch facts the dispatcher reads (core.arch.ArchTarget).
 */
#include "gemm_dispatch_internal.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>

namespace rocke
{
namespace gemm_dispatch
{

/* ---------------------------------------------------------------------- TileSpec */

bool Tile::mfmas_m(int64_t* out) const
{
    int64_t d = warp_m * warp_tile_m;
    if(d == 0 || tile_m % d)
        return false;
    *out = tile_m / d;
    return true;
}

bool Tile::mfmas_n(int64_t* out) const
{
    int64_t d = warp_n * warp_tile_n;
    if(d == 0 || tile_n % d)
        return false;
    *out = tile_n / d;
    return true;
}

bool Tile::k_atoms(int64_t* out) const
{
    if(warp_tile_k == 0 || tile_k % warp_tile_k)
        return false;
    *out = tile_k / warp_tile_k;
    return true;
}

void UGemm::post_init()
{
    if(block_size == 0)
        block_size = tile.warp_m * tile.warp_n * tile.warp_k * wave_size;
}

void Skinny::mn_operands(bool* a_mn, bool* b_mn) const
{
    std::string lay = layout;
    for(auto& c : lay)
        c = (char)toupper((unsigned char)c);
    *a_mn = !lay.empty() && lay[0] == 'C';
    *b_mn = lay.size() > 1 && lay[1] == 'R';
}

namespace detail
{

/* ------------------------------------------------------------------------- numbers */

int64_t pround(double x)
{
    /* Python round(float) -> int: round half to even */
    double r = std::nearbyint(x); /* FE_TONEAREST: ties to even */
    return (int64_t)r;
}

int64_t floor_log2(int64_t x)
{
    /* int(math.floor(math.log2(max(1, x)))) for an int: math.log2 of a Python int is exact
     * at powers of two and below them never rounds up to the next integer for x < 2^53 */
    if(x < 1)
        x = 1;
    return (int64_t)std::floor(std::log2((double)x));
}

/* --------------------------------------------------------------------- arch facts */

bool arch_facts(const std::string& arch, ArchFacts* out)
{
    if(arch == "gfx950")
    {
        *out = ArchFacts{64, 4, 163840, 1024};
        return true;
    }
    return false;
}

bool mma_shape(const std::string& arch,
               const std::string& dtype,
               int64_t m,
               int64_t n,
               int64_t k,
               int* a_frag,
               int* b_frag)
{
    if(arch != "gfx950" || !(dtype == "fp16" || dtype == "bf16"))
        return false;
    struct S
    {
        int64_t m, n, k;
        int a, b;
    };
    static const S shapes[]
        = {{16, 16, 16, 4, 4}, {16, 16, 32, 8, 8}, {32, 32, 8, 4, 4}, {32, 32, 16, 8, 8}};
    for(const S& s : shapes)
        if(s.m == m && s.n == n && s.k == k)
        {
            if(a_frag != nullptr)
                *a_frag = s.a;
            if(b_frag != nullptr)
                *b_frag = s.b;
            return true;
        }
    return false;
}

bool supports_dtype(const std::string& arch, const std::string& dtype)
{
    return arch == "gfx950" && (dtype == "fp16" || dtype == "bf16");
}

/* -------------------------------------------------------------------- field access */

namespace
{

FieldValue fv_B(bool v)
{
    FieldValue f;
    f.kind = FieldValue::BOOL;
    f.b = v;
    return f;
}
FieldValue fv_I(int64_t v)
{
    FieldValue f;
    f.kind = FieldValue::INT;
    f.i = v;
    return f;
}
FieldValue fv_S(const std::string& v)
{
    FieldValue f;
    f.kind = FieldValue::STR;
    f.s = v;
    return f;
}
FieldValue fv_OI(const std::optional<int64_t>& v)
{
    return v.has_value() ? fv_I(*v) : FieldValue{};
}
FieldValue fv_OB(const std::optional<bool>& v)
{
    return v.has_value() ? fv_B(*v) : FieldValue{};
}

bool set_B(bool& dst, const FieldValue& v)
{
    if(v.kind == FieldValue::BOOL)
        dst = v.b;
    else if(v.kind == FieldValue::INT)
        dst = v.i != 0;
    else
        return false;
    return true;
}
bool set_I(int64_t& dst, const FieldValue& v)
{
    if(v.kind == FieldValue::INT)
        dst = v.i;
    else if(v.kind == FieldValue::BOOL)
        dst = v.b ? 1 : 0;
    else
        return false;
    return true;
}
bool set_S(std::string& dst, const FieldValue& v)
{
    if(v.kind != FieldValue::STR)
        return false;
    dst = v.s;
    return true;
}
bool set_OI(std::optional<int64_t>& dst, const FieldValue& v)
{
    if(v.kind == FieldValue::NONE)
        dst.reset();
    else if(v.kind == FieldValue::INT)
        dst = v.i;
    else if(v.kind == FieldValue::BOOL)
        dst = (int64_t)(v.b ? 1 : 0);
    else
        return false;
    return true;
}
bool set_OB(std::optional<bool>& dst, const FieldValue& v)
{
    if(v.kind == FieldValue::NONE)
        dst.reset();
    else if(v.kind == FieldValue::BOOL)
        dst = v.b;
    else if(v.kind == FieldValue::INT)
        dst = v.i != 0;
    else
        return false;
    return true;
}

using TraitGetter = FieldValue (*)(const Trait&);
using TraitSetter = bool (*)(Trait&, const FieldValue&);

const TraitGetter k_getters[] = {
#define ROCKE_GD_TRAIT(kind, name, dflt) [](const Trait& t) { return fv_##kind(t.name); },
#include "rocke/gemm_dispatch_fields.inc"
#undef ROCKE_GD_TRAIT
};

const TraitSetter k_setters[] = {
#define ROCKE_GD_TRAIT(kind, name, dflt) \
    [](Trait& t, const FieldValue& v) { return set_##kind(t.name, v); },
#include "rocke/gemm_dispatch_fields.inc"
#undef ROCKE_GD_TRAIT
};

const char* const k_names[] = {
#define ROCKE_GD_TRAIT(kind, name, dflt) #name,
#include "rocke/gemm_dispatch_fields.inc"
#undef ROCKE_GD_TRAIT
};

constexpr int k_nfields = (int)(sizeof(k_names) / sizeof(k_names[0]));

const std::map<std::string, int>& name_index()
{
    static const std::map<std::string, int> m = [] {
        std::map<std::string, int> r;
        for(int i = 0; i < k_nfields; ++i)
            r[k_names[i]] = i;
        return r;
    }();
    return m;
}

} // namespace

std::vector<FieldValue> trait_values(const Trait& t)
{
    std::vector<FieldValue> out;
    out.reserve(k_nfields);
    for(int i = 0; i < k_nfields; ++i)
        out.push_back(k_getters[i](t));
    return out;
}

const std::vector<std::string>& trait_field_names()
{
    static const std::vector<std::string> v(k_names, k_names + k_nfields);
    return v;
}

int trait_field_index(const std::string& name)
{
    const auto& m = name_index();
    auto it = m.find(name);
    return it == m.end() ? -1 : it->second;
}

bool set_trait_field(Trait* t, const std::string& name, const FieldValue& v)
{
    int i = trait_field_index(name);
    if(i < 0)
        return false;
    return k_setters[i](*t, v);
}

FieldValue get_trait_field(const Trait& t, const std::string& name)
{
    int i = trait_field_index(name);
    return i < 0 ? FieldValue{} : k_getters[i](t);
}

std::vector<std::pair<std::string, FieldValue>> skinny_fields(const Skinny& s)
{
    std::vector<std::pair<std::string, FieldValue>> out;
#define ROCKE_GD_SKINNY(kind, name, dflt) out.push_back({#name, fv_##kind(s.name)});
#include "rocke/gemm_dispatch_fields.inc"
#undef ROCKE_GD_SKINNY
    return out;
}

std::string py_str(const FieldValue& v)
{
    switch(v.kind)
    {
    case FieldValue::NONE:
        return "None";
    case FieldValue::BOOL:
        return v.b ? "True" : "False";
    case FieldValue::INT:
        return std::to_string(v.i);
    case FieldValue::FLOAT:
    {
        char buf[64];
        snprintf(buf, sizeof buf, "%.17g", v.f);
        return buf;
    }
    case FieldValue::STR:
        return v.s;
    }
    return "";
}

bool field_eq(const FieldValue& a, const FieldValue& b)
{
    /* Python ==: bool and int compare by value; None only equals None */
    auto num = [](const FieldValue& x, double* d) {
        if(x.kind == FieldValue::BOOL)
            *d = x.b ? 1.0 : 0.0;
        else if(x.kind == FieldValue::INT)
            *d = (double)x.i;
        else if(x.kind == FieldValue::FLOAT)
            *d = x.f;
        else
            return false;
        return true;
    };
    double da, db;
    if(num(a, &da) && num(b, &db))
    {
        if(a.kind != FieldValue::FLOAT && b.kind != FieldValue::FLOAT)
        {
            int64_t ia = a.kind == FieldValue::BOOL ? (a.b ? 1 : 0) : a.i;
            int64_t ib = b.kind == FieldValue::BOOL ? (b.b ? 1 : 0) : b.i;
            return ia == ib;
        }
        return da == db;
    }
    if(a.kind != b.kind)
        return false;
    if(a.kind == FieldValue::NONE)
        return true;
    return a.s == b.s;
}

FieldValue parse_value(const std::string& v)
{
    if(v == "True")
        return fv_B(true);
    if(v == "False")
        return fv_B(false);
    if(v == "None")
        return FieldValue{};
    /* int(v): optional sign, digits, surrounding whitespace, underscores between digits */
    {
        std::string s = v;
        size_t a = s.find_first_not_of(" \t\n\r\f\v");
        size_t b = s.find_last_not_of(" \t\n\r\f\v");
        if(a != std::string::npos)
        {
            s = s.substr(a, b - a + 1);
            size_t i = 0;
            bool neg = false;
            if(s[0] == '+' || s[0] == '-')
            {
                neg = s[0] == '-';
                i = 1;
            }
            std::string digits;
            bool ok = i < s.size();
            bool last_us = true;
            for(; i < s.size(); ++i)
            {
                char c = s[i];
                if(c >= '0' && c <= '9')
                {
                    digits.push_back(c);
                    last_us = false;
                }
                else if(c == '_' && !last_us)
                    last_us = true;
                else
                {
                    ok = false;
                    break;
                }
            }
            if(ok && !digits.empty() && !last_us)
            {
                FieldValue f;
                f.kind = FieldValue::INT;
                f.i = std::strtoll(digits.c_str(), nullptr, 10) * (neg ? -1 : 1);
                return f;
            }
        }
    }
    {
        const char* c = v.c_str();
        char* end = nullptr;
        errno = 0;
        double d = std::strtod(c, &end);
        if(end != c)
        {
            while(*end != '\0' && isspace((unsigned char)*end))
                ++end;
            if(*end == '\0')
            {
                FieldValue f;
                f.kind = FieldValue::FLOAT;
                f.f = d;
                return f;
            }
        }
    }
    return fv_S(v);
}

} // namespace detail

using namespace detail;

/* ------------------------------------------------------------------------ JSON */

namespace
{

std::string jstr(const std::string& s)
{
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

std::string jval(const FieldValue& v)
{
    switch(v.kind)
    {
    case FieldValue::NONE:
        return "null";
    case FieldValue::BOOL:
        return v.b ? "true" : "false";
    case FieldValue::INT:
        return std::to_string(v.i);
    case FieldValue::FLOAT:
    {
        char buf[64];
        snprintf(buf, sizeof buf, "%.17g", v.f);
        return buf;
    }
    case FieldValue::STR:
        return jstr(v.s);
    }
    return "null";
}

} // namespace

std::vector<std::pair<std::string, FieldValue>> trait_fields(const Trait& t)
{
    std::vector<std::pair<std::string, FieldValue>> out;
    std::vector<FieldValue> v = trait_values(t);
    const auto& n = trait_field_names();
    for(size_t i = 0; i < v.size(); ++i)
        out.push_back({n[i], v[i]});
    return out;
}

std::string trait_json(const Trait& t)
{
    std::vector<std::pair<std::string, FieldValue>> f = trait_fields(t);
    std::sort(f.begin(), f.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string o = "{";
    for(size_t i = 0; i < f.size(); ++i)
        o += (i ? ", " : "") + jstr(f[i].first) + ": " + jval(f[i].second);
    return o + "}";
}

std::string tile_json(const Tile& t)
{
    std::vector<std::pair<std::string, int64_t>> f = {{"tile_m", t.tile_m},
                                                      {"tile_n", t.tile_n},
                                                      {"tile_k", t.tile_k},
                                                      {"warp_m", t.warp_m},
                                                      {"warp_n", t.warp_n},
                                                      {"warp_k", t.warp_k},
                                                      {"warp_tile_m", t.warp_tile_m},
                                                      {"warp_tile_n", t.warp_tile_n},
                                                      {"warp_tile_k", t.warp_tile_k}};
    std::sort(f.begin(), f.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string o = "{";
    for(size_t i = 0; i < f.size(); ++i)
        o += (i ? ", " : "") + jstr(f[i].first) + ": " + std::to_string(f[i].second);
    return o + "}";
}

/* -------------------------------------------------------------- config strings */

std::string config_string(const UGemm& spec)
{
    const Tile& t = spec.tile;
    std::vector<std::pair<std::string, FieldValue>> f = trait_fields(spec.trait);
    std::vector<std::pair<std::string, FieldValue>> d = trait_fields(Trait{});
    std::vector<std::pair<std::string, std::string>> nd;
    for(size_t i = 0; i < f.size(); ++i)
        if(!field_eq(f[i].second, d[i].second))
            nd.push_back({f[i].first, py_str(f[i].second)});
    std::sort(nd.begin(), nd.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string items;
    for(size_t i = 0; i < nd.size(); ++i)
        items += (i ? "," : "") + nd[i].first + "=" + nd[i].second;
    char head[256];
    snprintf(head,
             sizeof head,
             "t%lldx%lldx%lld_w%lldx%lldx%lld_a%lldx%lldx%lld|",
             (long long)t.tile_m,
             (long long)t.tile_n,
             (long long)t.tile_k,
             (long long)t.warp_m,
             (long long)t.warp_n,
             (long long)t.warp_k,
             (long long)t.warp_tile_m,
             (long long)t.warp_tile_n,
             (long long)t.warp_tile_k);
    return head + items;
}

std::string config_string(const Spec& spec)
{
    if(!spec.skinny)
        return config_string(spec.u);
    std::vector<std::pair<std::string, FieldValue>> f = skinny_fields(spec.s);
    std::vector<std::pair<std::string, FieldValue>> d = skinny_fields(Skinny{});
    std::vector<std::pair<std::string, std::string>> kv;
    for(size_t i = 0; i < f.size(); ++i)
    {
        const std::string& k = f[i].first;
        if(k == "dtype" || k == "name" || k == "split")
            continue;
        if(!field_eq(f[i].second, d[i].second))
            kv.push_back({k, py_str(f[i].second)});
    }
    kv.push_back({"split", std::to_string(spec.s.split)});
    std::sort(kv.begin(), kv.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string o = "sk|";
    for(size_t i = 0; i < kv.size(); ++i)
        o += (i ? "," : "") + kv[i].first + "=" + kv[i].second;
    return o;
}

bool parse_config(const std::string& cfg, Tile* tile, Trait* trait, std::string* why)
{
    /* cfg.strip() then the selector's _CFG_RE */
    size_t a = cfg.find_first_not_of(" \t\n\r\f\v");
    size_t b = cfg.find_last_not_of(" \t\n\r\f\v");
    std::string s = a == std::string::npos ? "" : cfg.substr(a, b - a + 1);
    static const std::regex re(
        "^t(\\d+)x(\\d+)x(\\d+)_w(\\d+)x(\\d+)x(\\d+)_a(\\d+)x(\\d+)x(\\d+)\\|?(.*)$");
    std::smatch m;
    if(!std::regex_match(s, m, re))
        return set_why(why, "not a gemm config string: '" + cfg + "'");
    Tile t;
    int64_t* dst[9] = {&t.tile_m,
                       &t.tile_n,
                       &t.tile_k,
                       &t.warp_m,
                       &t.warp_n,
                       &t.warp_k,
                       &t.warp_tile_m,
                       &t.warp_tile_n,
                       &t.warp_tile_k};
    for(int i = 0; i < 9; ++i)
        *dst[i] = std::strtoll(m[i + 1].str().c_str(), nullptr, 10);
    Trait tr;
    std::string rest = m[10].str();
    size_t pos = 0;
    while(!rest.empty() && pos <= rest.size())
    {
        size_t comma = rest.find(',', pos);
        std::string kv
            = rest.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if(!kv.empty())
        {
            size_t eq = kv.find('=');
            std::string k = eq == std::string::npos ? kv : kv.substr(0, eq);
            std::string v = eq == std::string::npos ? "" : kv.substr(eq + 1);
            if(trait_field_index(k) < 0)
                return set_why(why, "unknown TraitSpec field '" + k + "' in '" + cfg + "'");
            if(!set_trait_field(&tr, k, parse_value(v)))
                return set_why(why,
                               "TraitSpec field '" + k + "': value '" + v + "' of another type");
        }
        if(comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    *tile = t;
    *trait = tr;
    return true;
}

std::string python_only_levers(const Trait& t)
{
    static const char* const names[] = {
#define ROCKE_GD_PYONLY(name) #name,
#include "rocke/gemm_dispatch_fields.inc"
#undef ROCKE_GD_PYONLY
    };
    Trait d;
    std::string out;
    for(const char* n : names)
        if(!field_eq(get_trait_field(t, n), get_trait_field(d, n)))
            out += (out.empty() ? "" : ", ") + std::string(n);
    return out;
}

} // namespace gemm_dispatch
} // namespace rocke
