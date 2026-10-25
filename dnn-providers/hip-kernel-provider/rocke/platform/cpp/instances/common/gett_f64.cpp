// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gett_f64.cpp -- spec-driven native mirror of instances/common/gett_f64.py.
 *
 * Every builder call is its own statement, in the order the Python builder
 * makes it (Python evaluates arguments left to right, innermost first): each
 * call consumes an SSA id, so any reordering emits the same IR under different
 * names. See the Python module for the kernel design.
 */
#include "rocke/instance_gett_f64.h"

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <functional>
#include <string>
#include <vector>

namespace
{

constexpr int32_t k_mask_offset = INT32_MIN; /* 0x80000000: always out of range */
constexpr int64_t k_max_span = (int64_t(1) << 31) - 16;

using digits_t = rocke_gett_digits_t;

digits_t one_digit(int64_t e, int64_t s)
{
    digits_t d = {};
    d.n = 1;
    d.extent[0] = e;
    d.stride[0] = s;
    return d;
}

bool digits_equal(const digits_t& x, const digits_t& y)
{
    if(x.n != y.n)
        return false;
    for(int i = 0; i < x.n; ++i)
        if(x.extent[i] != y.extent[i] || x.stride[i] != y.stride[i])
            return false;
    return true;
}

int64_t digits_span(const digits_t& d)
{
    int64_t s = 0;
    for(int i = 0; i < d.n; ++i)
        s += (d.extent[i] - 1) * d.stride[i];
    return s + 1;
}

int64_t digits_extent(const digits_t& d)
{
    int64_t p = 1;
    for(int i = 0; i < d.n; ++i)
        p *= d.extent[i];
    return p;
}

const digits_t& c_m_of(const rocke_gett_f64_problem_t& p)
{
    return p.has_c ? p.c_m : p.d_m;
}
const digits_t& c_n_of(const rocke_gett_f64_problem_t& p)
{
    return p.has_c ? p.c_n : p.d_n;
}
const digits_t& c_l_of(const rocke_gett_f64_problem_t& p)
{
    return p.has_c ? p.c_l : p.d_l;
}

int elem_bytes(const rocke_gett_f64_problem_t& p)
{
    return p.complex_ ? 16 : 8;
}

/* Python f64_atom. */
const rocke_mma_op_t* f64_atom(const char* arch)
{
    if(arch == nullptr)
        return nullptr;
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == nullptr)
        return nullptr;
    return rocke_mma_catalog_op_for_shape(
        &target->mma, "mma", "fp64", "fp64", "fp64", 16, 16, 4, NULL);
}

/* Python k_split: (local, tiled) or false. */
bool k_split(const digits_t& d, int tile, digits_t* local, digits_t* tiled)
{
    *local = {};
    *tiled = {};
    if(d.n == 1)
    {
        int64_t e = d.extent[0], s = d.stride[0];
        int64_t ntiles = (e + tile - 1) / tile;
        *local = one_digit(tile, s);
        *tiled = one_digit(ntiles < 1 ? 1 : ntiles, tile * s);
        return true;
    }
    int64_t rem = tile;
    for(int i = 0; i < d.n; ++i)
    {
        int64_t e = d.extent[i], s = d.stride[i];
        if(rem == 1)
        {
            tiled->extent[tiled->n] = e;
            tiled->stride[tiled->n++] = s;
        }
        else if(e <= rem)
        {
            if(rem % e)
                return false;
            local->extent[local->n] = e;
            local->stride[local->n++] = s;
            rem /= e;
        }
        else
        {
            if(e % rem)
                return false;
            local->extent[local->n] = rem;
            local->stride[local->n++] = s;
            tiled->extent[tiled->n] = e / rem;
            tiled->stride[tiled->n++] = rem * s;
            rem = 1;
        }
    }
    if(rem != 1)
        return false;
    if(tiled->n == 0)
        *tiled = one_digit(1, 0);
    return true;
}

int64_t addr_span_bytes(const rocke_gett_f64_problem_t& p, char which, int tile_k)
{
    if(which == 'A' || which == 'B')
    {
        const digits_t& side = which == 'A' ? p.a_m : p.b_n;
        const digits_t& kd = which == 'A' ? p.a_k : p.b_k;
        digits_t local, tiled;
        int64_t k_span = k_split(kd, tile_k, &local, &tiled) ? digits_span(local) : digits_span(kd);
        return (digits_span(side) - 1 + k_span - 1 + 1) * elem_bytes(p);
    }
    const digits_t& m = which == 'C' ? c_m_of(p) : p.d_m;
    const digits_t& n = which == 'C' ? c_n_of(p) : p.d_n;
    return (digits_span(m) - 1 + digits_span(n) - 1 + 1) * elem_bytes(p);
}

/* Python runtime_template. */
bool runtime_template(const rocke_gett_f64_problem_t& p, rocke_gett_f64_problem_t* out)
{
    const digits_t* spaces[] = {&p.a_m,
                                &p.a_k,
                                &p.a_l,
                                &p.b_n,
                                &p.b_k,
                                &p.b_l,
                                &p.d_m,
                                &p.d_n,
                                &p.d_l,
                                &c_m_of(p),
                                &c_n_of(p),
                                &c_l_of(p)};
    for(const digits_t* d : spaces)
        if(d->n != 1)
            return false;
    auto unit = [](const digits_t& d) { return d.stride[0] == 1 && d.extent[0] > 1; };
    /* 3 for an odd run an operand loads along (no two-element vectors), else 2. */
    int64_t em = unit(p.a_m) && p.M % 2 ? 3 : 2;
    int64_t en = unit(p.b_n) && p.N % 2 ? 3 : 2;
    int64_t ek = (unit(p.a_k) || unit(p.b_k)) && p.K % 2 ? 3 : 2;
    auto dig = [&](const digits_t& d, int64_t ext) { return one_digit(ext, unit(d) ? 1 : 2); };
    rocke_gett_f64_problem_t t = {};
    t.M = em;
    t.N = en;
    t.K = ek;
    t.L = 2;
    t.a_m = dig(p.a_m, em);
    t.a_k = dig(p.a_k, ek);
    t.b_n = dig(p.b_n, en);
    t.b_k = dig(p.b_k, ek);
    t.d_m = dig(p.d_m, em);
    t.d_n = dig(p.d_n, en);
    t.a_l = one_digit(2, 2);
    t.b_l = one_digit(2, 2);
    t.d_l = one_digit(2, 2);
    t.has_c = true;
    t.c_m = dig(c_m_of(p), em);
    t.c_n = dig(c_n_of(p), en);
    t.c_l = one_digit(2, 2);
    t.complex_ = p.complex_;
    t.conj_a = p.conj_a;
    t.conj_b = p.conj_b;
    *out = t;
    return true;
}

/* Python GettF64Problem equality (c digits resolved). */
bool problems_equal(const rocke_gett_f64_problem_t& x, const rocke_gett_f64_problem_t& y)
{
    return x.M == y.M && x.N == y.N && x.K == y.K && x.L == y.L && digits_equal(x.a_m, y.a_m)
           && digits_equal(x.a_k, y.a_k) && digits_equal(x.a_l, y.a_l) && digits_equal(x.b_n, y.b_n)
           && digits_equal(x.b_k, y.b_k) && digits_equal(x.b_l, y.b_l) && digits_equal(x.d_m, y.d_m)
           && digits_equal(x.d_n, y.d_n) && digits_equal(x.d_l, y.d_l)
           && digits_equal(c_m_of(x), c_m_of(y)) && digits_equal(c_n_of(x), c_n_of(y))
           && digits_equal(c_l_of(x), c_l_of(y)) && x.complex_ == y.complex_ && x.conj_a == y.conj_a
           && x.conj_b == y.conj_b;
}

int threads(const rocke_gett_f64_spec_t& s)
{
    return 64 * s.warps_m * s.warps_n;
}

bool uses_swizzle(const rocke_gett_f64_spec_t& s)
{
    return s.lds_swizzle && !s.problem.complex_ && s.tile_m % 32 == 0 && s.tile_n % 32 == 0;
}

int swizzle_mask(int side)
{
    return side % 64 == 0 ? 63 : 31;
}

int resolved_pad(const rocke_gett_f64_spec_t& s)
{
    if(s.lds_pad >= 0)
        return s.lds_pad;
    if(s.problem.complex_ || uses_swizzle(s))
        return 0;
    const int pads[] = {0, 16, 8, 24};
    for(int pad : pads)
    {
        if(((s.tile_m + pad) * 8) % 256 == 128 && ((s.tile_n + pad) * 8) % 256 == 128)
            return pad;
    }
    return 16;
}

int contiguous(const digits_t& d)
{
    return d.stride[0] == 1 ? (int)d.extent[0] : 0;
}

struct load_plan_t
{
    bool ok;
    bool along_k;
    int vec;
    int cpl;
    int lpp;
    int passes;
};

/* Python _load_plan. */
load_plan_t load_plan(const rocke_gett_f64_spec_t& spec,
                      const digits_t& side_digits,
                      const digits_t& k_digits,
                      int side_tile)
{
    const rocke_gett_f64_problem_t& p = spec.problem;
    int nt = threads(spec);
    struct cand
    {
        bool along_k;
        int vec;
    };
    std::vector<cand> cands;
    if(p.complex_)
    {
        if(contiguous(side_digits))
            cands.push_back({false, 1});
        if(contiguous(k_digits))
            cands.push_back({true, 1});
        cands.push_back({false, 1});
    }
    else
    {
        int run_s = contiguous(side_digits), run_k = contiguous(k_digits);
        if(run_s && run_s % 2 == 0)
            cands.push_back({false, 2});
        if(run_k && run_k % 2 == 0 && spec.tile_k % 2 == 0)
            cands.push_back({true, 2});
        if(run_s)
            cands.push_back({false, 1});
        if(run_k)
            cands.push_back({true, 1});
        cands.push_back({false, 1});
    }
    for(const cand& c : cands)
    {
        int line = c.along_k ? spec.tile_k : side_tile;
        int other = c.along_k ? side_tile : spec.tile_k;
        if(line % c.vec)
            continue;
        int cpl = line / c.vec;
        if(cpl > nt || nt % cpl)
            continue;
        int lpp = nt / cpl;
        if(other % lpp)
            continue;
        return {true, c.along_k, c.vec, cpl, lpp, other / lpp};
    }
    return {false, false, 0, 0, 0, 0};
}

bool problem_check(const rocke_gett_f64_problem_t& p, std::string* why)
{
    struct pair
    {
        const char* label;
        int64_t ext;
        const digits_t* d[3];
        int n;
    } pairs[] = {
        {"M", p.M, {&p.a_m, &p.d_m, nullptr}, 2},
        {"N", p.N, {&p.b_n, &p.d_n, nullptr}, 2},
        {"K", p.K, {&p.a_k, &p.b_k, nullptr}, 2},
        {"L", p.L, {&p.a_l, &p.b_l, &p.d_l}, 3},
    };
    for(const pair& q : pairs)
    {
        if(q.ext <= 0)
        {
            *why = std::string(q.label) + " must be positive, got " + std::to_string(q.ext);
            return false;
        }
        for(int i = 0; i < q.n; ++i)
        {
            if(digits_extent(*q.d[i]) != q.ext)
            {
                *why
                    = std::string(q.label) + " digits do not cover extent " + std::to_string(q.ext);
                return false;
            }
        }
    }
    if(digits_extent(c_m_of(p)) != p.M || digits_extent(c_n_of(p)) != p.N
       || digits_extent(c_l_of(p)) != p.L)
    {
        *why = "C digits do not cover (M, N, L)";
        return false;
    }
    if((p.conj_a || p.conj_b) && !p.complex_)
    {
        *why = "conjugation needs complex data";
        return false;
    }
    return true;
}

bool valid(const rocke_gett_f64_spec_t& spec_in, const char* arch, std::string* why)
{
    rocke_gett_f64_spec_t spec = spec_in;
    if(!problem_check(spec.problem, why))
        return false;
    if(spec.runtime_sizes)
    {
        rocke_gett_f64_problem_t tp;
        if(!runtime_template(spec.problem, &tp))
        {
            *why = "runtime_sizes needs one digit per index space (a strided GEMM)";
            return false;
        }
        /* The kernel depends on the problem only through its layout class. */
        spec.problem = tp;
    }
    const rocke_gett_f64_problem_t& p = spec.problem;
    const rocke_mma_op_t* atom = f64_atom(arch);
    if(atom == nullptr)
    {
        *why = std::string("'") + (arch ? arch : "")
               + "' has no 16x16x4 f64 MMA atom in its catalog";
        return false;
    }
    if(!spec.runtime_sizes && p.L > ROCKE_GETT_F64_MAX_GRID_Y)
    {
        *why = "L = " + std::to_string(p.L) + " exceeds the grid limit "
               + std::to_string(ROCKE_GETT_F64_MAX_GRID_Y) + "; split the batch";
        return false;
    }
    int w = spec.warps_m * spec.warps_n;
    if(!(w == 1 || w == 2 || w == 4 || w == 8))
    {
        *why = "warps_m * warps_n must be 1, 2, 4 or 8";
        return false;
    }
    if(spec.tile_m % (atom->m * spec.warps_m) || spec.tile_n % (atom->n * spec.warps_n))
    {
        *why = "each wave tile must be a whole number of 16x16 atoms";
        return false;
    }
    if(spec.tile_k % atom->k || spec.tile_k <= 0)
    {
        *why = "tile_k must be a positive multiple of 4";
        return false;
    }
    if(spec.k_block < 0)
    {
        *why = "k_block must be >= 0";
        return false;
    }
    if(spec.split_k < 1 || spec.split_k > ROCKE_GETT_F64_MAX_GRID_Y)
    {
        *why = "split_k must be in [1, " + std::to_string(ROCKE_GETT_F64_MAX_GRID_Y) + "]";
        return false;
    }
    if(spec.split_k > 1 && (spec.unroll2 || spec.k_stagger))
    {
        *why = "split_k does not combine with unroll2 or k_stagger";
        return false;
    }
    if(spec.split_k > 1 && !spec.runtime_sizes
       && (p.K + spec.tile_k - 1) / spec.tile_k < spec.split_k)
    {
        *why = "split_k needs at least one K tile per split";
        return false;
    }
    if(spec.k_block && spec.unroll2 && spec.k_block % 2)
    {
        *why = "k_block must be even with unroll2";
        return false;
    }
    if(spec.group_m < 1)
    {
        *why = "group_m must be >= 1";
        return false;
    }
    if(spec.xcds < 0)
    {
        *why = "xcds must be >= 0";
        return false;
    }
    if(spec.lds_swizzle_mode < 0 || spec.lds_swizzle_mode > 2)
    {
        *why = "lds_swizzle_mode must be 0, 1 or 2";
        return false;
    }
    if(spec.lds_stages != 1 && spec.lds_stages != 2)
    {
        *why = "lds_stages must be 1 or 2";
        return false;
    }
    if(spec.prefetch_depth != 1 && spec.prefetch_depth != 2)
    {
        *why = "prefetch_depth must be 1 or 2";
        return false;
    }
    if(spec.prefetch_depth == 2 && !spec.prefetch)
    {
        *why = "prefetch_depth 2 needs prefetch";
        return false;
    }
    if(spec.complex_accs != 2 && spec.complex_accs != 4)
    {
        *why = "complex_accs must be 2 or 4";
        return false;
    }
    if(spec.k_stagger < 0)
    {
        *why = "k_stagger must be >= 0";
        return false;
    }
    if(spec.unroll2 && (spec.lds_stages != 2 || spec.prefetch_depth != 1 || !spec.prefetch))
    {
        *why = "unroll2 needs two LDS stages and prefetch depth 1";
        return false;
    }
    int acc_words = (spec.tile_m / spec.warps_m / atom->m) * (spec.tile_n / spec.warps_n / atom->n)
                    * atom->c_frag_len * (p.complex_ ? spec.complex_accs : 1)
                    * (spec.k_block ? 2 : 1);
    if(acc_words * 2 > 384)
    {
        *why = "accumulator needs " + std::to_string(acc_words * 2)
               + " registers per lane (max 384)";
        return false;
    }
    const char names[] = {'A', 'B', 'C', 'D'};
    for(char which : names)
    {
        int64_t span = addr_span_bytes(p, which, spec.tile_k);
        if(span > k_max_span)
        {
            *why = std::string(1, which) + " addresses " + std::to_string(span)
                   + " bytes per buffer; split the launch";
            return false;
        }
    }
    if(!load_plan(spec, p.a_m, p.a_k, spec.tile_m).ok)
    {
        *why = "no thread mapping for the A tile";
        return false;
    }
    if(!load_plan(spec, p.b_n, p.b_k, spec.tile_n).ok)
    {
        *why = "no thread mapping for the B tile";
        return false;
    }
    int words = p.complex_ ? 2 : 1;
    int lds = spec.lds_stages * spec.tile_k
              * ((spec.tile_m + spec.tile_n) * words + 2 * resolved_pad(spec)) * 8;
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    int cap = target ? target->lds_capacity_bytes : 0;
    if(lds > cap)
    {
        *why = "LDS " + std::to_string(lds) + " bytes exceeds " + std::to_string(cap) + " on "
               + arch;
        return false;
    }
    return true;
}

/* Python runtime_compatible. */
bool runtime_compatible(const rocke_gett_f64_spec_t& spec,
                        const rocke_gett_f64_problem_t& problem,
                        std::string* why)
{
    if(!spec.runtime_sizes)
    {
        *why = "the spec bakes its sizes in (runtime_sizes is off)";
        return false;
    }
    if(!problem_check(problem, why))
        return false;
    rocke_gett_f64_problem_t tp, ts;
    if(!runtime_template(problem, &tp))
    {
        *why = "runtime_sizes needs one digit per index space (a strided GEMM)";
        return false;
    }
    if(!runtime_template(spec.problem, &ts) || !problems_equal(tp, ts))
    {
        *why = "the problem's layout class differs from the kernel's template";
        return false;
    }
    const digits_t* spaces[] = {&problem.a_m,
                                &problem.a_k,
                                &problem.a_l,
                                &problem.b_n,
                                &problem.b_k,
                                &problem.b_l,
                                &problem.d_m,
                                &problem.d_n,
                                &problem.d_l,
                                &c_m_of(problem),
                                &c_n_of(problem),
                                &c_l_of(problem)};
    for(const digits_t* d : spaces)
    {
        if(d->stride[0] < 0)
        {
            *why = "runtime_sizes needs non-negative strides";
            return false;
        }
    }
    const char names[] = {'A', 'B', 'C', 'D'};
    for(char which : names)
    {
        int64_t span = addr_span_bytes(problem, which, spec.tile_k);
        if(span > k_max_span)
        {
            *why = std::string(1, which) + " addresses " + std::to_string(span)
                   + " bytes per buffer; split the launch";
            return false;
        }
    }
    if(problem.L > ROCKE_GETT_F64_MAX_GRID_Y)
    {
        *why = "L = " + std::to_string(problem.L) + " exceeds the grid limit "
               + std::to_string(ROCKE_GETT_F64_MAX_GRID_Y) + "; split the batch";
        return false;
    }
    if(spec.split_k > 1 && (problem.K + spec.tile_k - 1) / spec.tile_k < spec.split_k)
    {
        *why = "split_k needs at least one K tile per split";
        return false;
    }
    return true;
}

/* Python RUNTIME_SIZE_ARGS: (name, is_i64). */
struct rt_arg_t
{
    const char* name;
    bool i64;
};
const rt_arg_t k_rt_args[ROCKE_GETT_F64_RUNTIME_ARGS] = {
    {"M", false},
    {"N", false},
    {"K", false},
    {"L", false},
    {"a_sm", false},
    {"a_sk", false},
    {"b_sn", false},
    {"b_sk", false},
    {"d_sm", false},
    {"d_sn", false},
    {"c_sm", false},
    {"c_sn", false},
    {"a_sl", true},
    {"b_sl", true},
    {"d_sl", true},
    {"c_sl", true},
};

bool is_c_arg(const char* name)
{
    return strcmp(name, "c_sm") == 0 || strcmp(name, "c_sn") == 0 || strcmp(name, "c_sl") == 0;
}

std::string digits_key(const digits_t& d)
{
    std::string s;
    for(int i = 0; i < d.n; ++i)
    {
        if(i)
            s += ",";
        s += std::to_string(d.extent[i]) + ":" + std::to_string(d.stride[i]);
    }
    return s;
}

/* Python _problem_hash: 10 hex digits of FNV-1a 64 over the index maps. */
std::string problem_hash(const rocke_gett_f64_problem_t& p)
{
    const digits_t* ds[] = {&p.a_m,
                            &p.a_k,
                            &p.a_l,
                            &p.b_n,
                            &p.b_k,
                            &p.b_l,
                            &p.d_m,
                            &p.d_n,
                            &p.d_l,
                            &c_m_of(p),
                            &c_n_of(p),
                            &c_l_of(p)};
    std::string key;
    for(size_t i = 0; i < sizeof(ds) / sizeof(ds[0]); ++i)
    {
        if(i)
            key += "|";
        key += digits_key(*ds[i]);
    }
    uint64_t h = 0xCBF29CE484222325ull;
    for(unsigned char c : key)
    {
        h ^= c;
        h *= 0x100000001B3ull;
    }
    char buf[17];
    snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
    return std::string(buf, 10);
}

/* ------------------------------------------------------------- device side */

typedef rocke_ir_builder_t B;
typedef rocke_value_t V;

V* ci(B* b, int64_t v)
{
    return rocke_b_const_i32(b, (int32_t)v);
}

/* An i32 operand: a compile-time int (v == nullptr) or a runtime Value (Python _ci). */
struct ival
{
    V* v;
    int64_t c;
};

ival iv_const(int64_t c)
{
    return {nullptr, c};
}

ival iv_val(V* v)
{
    return {v, 0};
}

V* get(B* b, const ival& x)
{
    return x.v ? x.v : ci(b, x.c);
}

/* Python _mixed (compile-time digits). */
V* mixed(B* b, V* idx, const digits_t& d)
{
    V* off = nullptr;
    V* rest = idx;
    int last = d.n - 1;
    for(int i = 0; i < d.n; ++i)
    {
        int64_t e = d.extent[i], s = d.stride[i];
        if(s == 0 || e == 1)
        {
            if(i < last)
            {
                V* ce = ci(b, e);
                rest = rocke_b_div(b, rest, ce);
            }
            continue;
        }
        V* dv;
        if(i == last)
            dv = rest;
        else
        {
            V* ce = ci(b, e);
            dv = rocke_b_mod(b, rest, ce);
        }
        V* term;
        if(s == 1)
            term = dv;
        else
        {
            V* cs = ci(b, s);
            term = rocke_b_mul(b, dv, cs);
        }
        off = off == nullptr ? term : rocke_b_add(b, off, term);
        if(i < last)
        {
            V* ce = ci(b, e);
            rest = rocke_b_div(b, rest, ce);
        }
    }
    return off != nullptr ? off : ci(b, 0);
}

/* An index map: compile-time digits, or (runtime_sizes) one digit whose stride
 * is a kernel argument (stride == nullptr: unit stride). */
struct gdig
{
    bool rt;
    digits_t d;
    V* stride;
};

/* Python _mixed over a static or runtime index map. */
V* gmixed(B* b, V* idx, const gdig& g)
{
    if(!g.rt)
        return mixed(b, idx, g.d);
    return g.stride ? rocke_b_mul(b, idx, g.stride) : idx;
}

/* Python _mixed64. */
V* mixed64(B* b, V* idx, const digits_t& d, int64_t scale)
{
    V* off = nullptr;
    V* rest = idx;
    int last = d.n - 1;
    for(int i = 0; i < d.n; ++i)
    {
        int64_t e = d.extent[i], s = d.stride[i];
        if(s == 0 || e == 1)
        {
            if(i < last)
            {
                V* ce = ci(b, e);
                rest = rocke_b_div(b, rest, ce);
            }
            continue;
        }
        V* dv;
        if(i == last)
            dv = rest;
        else
        {
            V* ce = ci(b, e);
            dv = rocke_b_mod(b, rest, ce);
        }
        V* wide = rocke_b_zext(b, dv, rocke_i64());
        V* cs = rocke_b_const_i64(b, s * scale);
        V* term = rocke_b_mul(b, wide, cs);
        off = off == nullptr ? term : rocke_b_add(b, off, term);
        if(i < last)
        {
            V* ce = ci(b, e);
            rest = rocke_b_div(b, rest, ce);
        }
    }
    return off != nullptr ? off : rocke_b_const_i64(b, 0);
}

/* Python _rt_span: (sum (extent - 1) * stride + 1) * es as an i32. A null
 * stride is a unit stride (no multiply). */
struct span_term
{
    ival em1;
    V* stride;
};

V* rt_span(B* b, const span_term* terms, int n, int es)
{
    V* acc = nullptr;
    for(int i = 0; i < n; ++i)
    {
        V* t;
        if(terms[i].stride == nullptr)
            t = get(b, terms[i].em1);
        else
        {
            V* e = get(b, terms[i].em1);
            t = rocke_b_mul(b, e, terms[i].stride);
        }
        acc = acc == nullptr ? t : rocke_b_add(b, acc, t);
    }
    V* c1 = ci(b, 1);
    acc = rocke_b_add(b, acc, c1);
    V* ces = ci(b, es);
    return rocke_b_mul(b, acc, ces);
}

/* Python _swz. */
V* swz(B* b, V* k, int mask, int mode)
{
    if(mode == 2)
    {
        V* c1 = ci(b, 1);
        V* sh = rocke_b_lshr(b, k, c1);
        V* c7 = ci(b, 7);
        V* pair = rocke_b_land(b, sh, c7);
        V* c4 = ci(b, 4);
        V* x = rocke_b_mul(b, pair, c4);
        V* cm = ci(b, mask);
        return rocke_b_land(b, x, cm);
    }
    V* c1 = ci(b, 1);
    V* low = rocke_b_land(b, k, c1);
    V* c16 = ci(b, 16);
    V* odd = rocke_b_mul(b, low, c16);
    if(mode == 1)
    {
        V* c1b = ci(b, 1);
        V* sh = rocke_b_lshr(b, k, c1b);
        V* c3 = ci(b, 3);
        V* p0 = rocke_b_land(b, sh, c3);
        V* c4 = ci(b, 4);
        V* pair = rocke_b_mul(b, p0, c4);
        V* x = rocke_b_xor(b, odd, pair);
        V* cm = ci(b, mask);
        return rocke_b_land(b, x, cm);
    }
    V* c2 = ci(b, 2);
    V* sh = rocke_b_lshr(b, k, c2);
    V* c7 = ci(b, 7);
    V* blk0 = rocke_b_land(b, sh, c7);
    V* c8 = ci(b, 8);
    V* blk = rocke_b_mul(b, blk0, c8);
    V* x = rocke_b_xor(b, odd, blk);
    V* cm = ci(b, mask);
    return rocke_b_land(b, x, cm);
}

/* Python _Geo: sizes and index maps the kernel addresses with. */
struct geo_t
{
    bool rt;
    ival M, N, K, L;
    gdig a_m, a_k, b_n, b_k, d_m, d_n, c_m, c_n;
    V* batch_a;
    V* batch_b;
    V* batch_c;
    V* batch_d;
};

struct coord_t
{
    V* s_loc;
    V* k_loc;
    V* base;
    V* s_ok;
};

/* Python _Operand. */
struct operand_t
{
    B* b;
    const rocke_gett_f64_spec_t* spec;
    bool cx;
    int es;
    V* base_ptr;
    bool rt;
    int64_t span;
    V* span_v;
    V* kstep;
    ival K;
    ival nk;
    digits_t k_digits;
    int side_tile;
    V* lds;
    load_plan_t plan;
    bool split;
    gdig local;
    digits_t tiled;
    V* rsrc;
    std::vector<coord_t> coords;
    std::vector<std::vector<V*>> store_cols;
    std::function<V*(V*)> kmap; /* K-tile index remap (k_stagger), set by the builder */

    void init(B* bb,
              const rocke_gett_f64_spec_t* sp,
              char name,
              V* bptr,
              const gdig& side_digits,
              const gdig& kd,
              int st,
              V* side_base,
              const ival& side_extent,
              V* lds_,
              const geo_t& geo)
    {
        b = bb;
        spec = sp;
        const rocke_gett_f64_problem_t& p = sp->problem;
        cx = p.complex_;
        es = elem_bytes(p);
        base_ptr = bptr;
        k_digits = kd.d;
        side_tile = st;
        lds = lds_;
        nk = iv_const(0);
        K = geo.K;
        rt = geo.rt;
        const digits_t& tmpl_side = name == 'A' ? p.a_m : p.b_n;
        const digits_t& tmpl_k = name == 'A' ? p.a_k : p.b_k;
        plan = load_plan(*sp, tmpl_side, tmpl_k, st);
        span = 0;
        span_v = nullptr;
        kstep = nullptr;
        rsrc = nullptr;
        local = {};
        tiled = {};
        if(rt)
        {
            /* One digit per space: the K tile always splits. */
            split = true;
            local.rt = true;
            local.stride = kd.stride;
            V* c1 = ci(b, 1);
            V* em1 = rocke_b_sub(b, side_extent.v, c1);
            span_term terms[2]
                = {{iv_val(em1), side_digits.stride}, {iv_const(sp->tile_k - 1), kd.stride}};
            span_v = rt_span(b, terms, 2, es);
            if(kd.stride == nullptr)
                kstep = rocke_b_const_i64(b, (int64_t)sp->tile_k * es);
            else
            {
                V* wide = rocke_b_zext(b, kd.stride, rocke_i64());
                V* cst = rocke_b_const_i64(b, (int64_t)sp->tile_k * es);
                kstep = rocke_b_mul(b, wide, cst);
            }
        }
        else
        {
            span = addr_span_bytes(p, name, sp->tile_k);
            digits_t loc;
            split = k_split(kd.d, sp->tile_k, &loc, &tiled);
            local.rt = false;
            local.d = loc;
            local.stride = nullptr;
            if(!split)
            {
                V* cspan = ci(b, span);
                rsrc = rocke_b_buffer_rsrc(b, base_ptr, cspan);
            }
        }
        V* tid = rocke_b_thread_id_x(b);
        V* ccpl = ci(b, plan.cpl);
        V* c = rocke_b_mod(b, tid, ccpl);
        V* ccpl2 = ci(b, plan.cpl);
        V* line = rocke_b_div(b, tid, ccpl2);
        for(int ps = 0; ps < plan.passes; ++ps)
        {
            V* ln = line;
            if(ps != 0)
            {
                V* cl = ci(b, ps * plan.lpp);
                ln = rocke_b_add(b, line, cl);
            }
            V* v0 = c;
            if(plan.vec > 1)
            {
                V* cv = ci(b, plan.vec);
                v0 = rocke_b_mul(b, c, cv);
            }
            V* s_loc = plan.along_k ? ln : v0;
            V* k_loc = plan.along_k ? v0 : ln;
            V* s_g = rocke_b_add(b, side_base, s_loc);
            V* cext = get(b, side_extent);
            V* s_ok = rocke_b_cmp_lt(b, s_g, cext);
            V* c0 = ci(b, 0);
            V* s_g_safe = rocke_b_select(b, s_ok, s_g, c0);
            V* off = gmixed(b, s_g_safe, side_digits);
            if(split)
            {
                V* kl = gmixed(b, k_loc, local);
                off = rocke_b_add(b, off, kl);
            }
            V* ces = ci(b, es);
            off = rocke_b_mul(b, off, ces);
            coords.push_back({s_loc, k_loc, off, s_ok});
        }
        int wpe = cx ? 2 : 1;
        bool sw = uses_swizzle(*sp);
        for(const coord_t& co : coords)
        {
            int nrows = plan.along_k ? plan.vec : 1;
            std::vector<V*> cols;
            for(int j = 0; j < nrows; ++j)
            {
                V* col = co.s_loc;
                if(wpe > 1)
                {
                    V* cw = ci(b, wpe);
                    col = rocke_b_mul(b, co.s_loc, cw);
                }
                if(sw)
                {
                    V* kr = co.k_loc;
                    if(j)
                    {
                        V* cj = ci(b, j);
                        kr = rocke_b_add(b, co.k_loc, cj);
                    }
                    V* sz = swz(b, kr, swizzle_mask(side_tile), spec->lds_swizzle_mode);
                    col = rocke_b_xor(b, col, sz);
                }
                cols.push_back(col);
            }
            store_cols.push_back(cols);
        }
    }

    std::vector<V*> load(V* kt)
    {
        V* live = nullptr;
        if(kmap)
        {
            V* cnk = get(b, nk);
            live = rocke_b_cmp_lt(b, kt, cnk);
            kt = kmap(kt);
        }
        V* ctk = ci(b, spec->tile_k);
        V* k0 = rocke_b_mul(b, kt, ctk);
        V* rs = rsrc;
        if(split)
        {
            V* koff;
            if(rt)
            {
                V* wide = rocke_b_zext(b, kt, rocke_i64());
                koff = rocke_b_mul(b, wide, kstep);
            }
            else
                koff = mixed64(b, kt, tiled, es);
            V* tile_ptr = rocke_b_global_ptr_add(b, base_ptr, koff);
            V* cspan = rt ? span_v : ci(b, span);
            rs = rocke_b_buffer_rsrc(b, tile_ptr, cspan);
        }
        std::vector<V*> vals;
        int words = plan.vec * (cx ? 2 : 1);
        const rocke_type_t* vty = rocke_vector_type(b, rocke_f64(), words);
        for(const coord_t& co : coords)
        {
            V* kg = rocke_b_add(b, k0, co.k_loc);
            V* cK = get(b, K);
            V* kin = rocke_b_cmp_lt(b, kg, cK);
            V* ok = rocke_b_land(b, co.s_ok, kin);
            if(live != nullptr)
                ok = rocke_b_land(b, ok, live);
            V* voff;
            if(split)
                voff = co.base;
            else
            {
                V* c0 = ci(b, 0);
                V* kg_safe = rocke_b_select(b, ok, kg, c0);
                V* ko = mixed(b, kg_safe, k_digits);
                V* ces = ci(b, es);
                V* kb = rocke_b_mul(b, ko, ces);
                voff = rocke_b_add(b, co.base, kb);
            }
            V* cmask = ci(b, k_mask_offset);
            voff = rocke_b_select(b, ok, voff, cmask);
            V* c0s = ci(b, 0);
            V* raw = rocke_b_buffer_load_vN(b, rs, voff, c0s, rocke_i32(), 2 * words);
            vals.push_back(rocke_b_vec_bitcast(b, raw, vty));
        }
        return vals;
    }

    void store_lds(const std::vector<V*>& vals, V* stage)
    {
        V* ctk = ci(b, spec->tile_k);
        V* row0 = rocke_b_mul(b, stage, ctk);
        int wpe = cx ? 2 : 1;
        for(size_t ps = 0; ps < coords.size(); ++ps)
        {
            const coord_t& co = coords[ps];
            V* v = vals[ps];
            if(plan.along_k)
            {
                for(int j = 0; j < plan.vec; ++j)
                {
                    V* kr = co.k_loc;
                    if(j)
                    {
                        V* cj = ci(b, j);
                        kr = rocke_b_add(b, co.k_loc, cj);
                    }
                    V* row = rocke_b_add(b, row0, kr);
                    V* col = store_cols[ps][j];
                    V* idx[2] = {row, col};
                    if(cx)
                        rocke_b_smem_store_vN(b, lds, idx, 2, v, 2);
                    else
                    {
                        V* e = rocke_b_vec_extract(b, v, j);
                        rocke_b_smem_store_vN(b, lds, idx, 2, e, 1);
                    }
                }
            }
            else
            {
                V* row = rocke_b_add(b, row0, co.k_loc);
                V* col = store_cols[ps][0];
                V* idx[2] = {row, col};
                int n = plan.vec * wpe;
                if(n == 1)
                {
                    V* e = rocke_b_vec_extract(b, v, 0);
                    rocke_b_smem_store_vN(b, lds, idx, 2, e, 1);
                }
                else
                    rocke_b_smem_store_vN(b, lds, idx, 2, v, n);
            }
        }
    }
};

/* Python _raster (compile-time sizes). */
void raster(B* b, const rocke_gett_f64_spec_t& spec, V* bid, V** tm, V** tn)
{
    const rocke_gett_f64_problem_t& p = spec.problem;
    int64_t ntm = (p.M + spec.tile_m - 1) / spec.tile_m;
    int64_t ntn = (p.N + spec.tile_n - 1) / spec.tile_n;
    int64_t total = ntm * ntn;
    if(spec.xcds && total % spec.xcds == 0)
    {
        int64_t per = total / spec.xcds;
        V* cx1 = ci(b, spec.xcds);
        V* md = rocke_b_mod(b, bid, cx1);
        V* cper = ci(b, per);
        V* hi = rocke_b_mul(b, md, cper);
        V* cx2 = ci(b, spec.xcds);
        V* lo = rocke_b_div(b, bid, cx2);
        bid = rocke_b_add(b, hi, lo);
    }
    int64_t g = spec.group_m < ntm ? spec.group_m : ntm;
    if(g <= 1)
    {
        V* c1 = ci(b, ntn);
        *tm = rocke_b_div(b, bid, c1);
        V* c2 = ci(b, ntn);
        *tn = rocke_b_mod(b, bid, c2);
        return;
    }
    int64_t in_group = g * ntn;
    V* cig = ci(b, in_group);
    V* group = rocke_b_div(b, bid, cig);
    V* cg = ci(b, g);
    V* first = rocke_b_mul(b, group, cg);
    V* cntm = ci(b, ntm);
    V* rest = rocke_b_sub(b, cntm, first);
    V* cg2 = ci(b, g);
    V* size = rocke_b_smin(b, rest, cg2);
    V* cig2 = ci(b, in_group);
    V* r = rocke_b_mod(b, bid, cig2);
    V* rm = rocke_b_mod(b, r, size);
    *tm = rocke_b_add(b, first, rm);
    *tn = rocke_b_div(b, r, size);
}

/* Python _raster_rt: tile counts from the size arguments. */
void raster_rt(B* b, const rocke_gett_f64_spec_t& spec, V* bid, const geo_t& geo, V** tm, V** tn)
{
    V* cm1 = ci(b, spec.tile_m - 1);
    V* am = rocke_b_add(b, geo.M.v, cm1);
    V* ctm = ci(b, spec.tile_m);
    V* ntm = rocke_b_div(b, am, ctm);
    V* cn1 = ci(b, spec.tile_n - 1);
    V* an = rocke_b_add(b, geo.N.v, cn1);
    V* ctn = ci(b, spec.tile_n);
    V* ntn = rocke_b_div(b, an, ctn);
    if(spec.xcds)
    {
        V* total = rocke_b_mul(b, ntm, ntn);
        V* cx0 = ci(b, spec.xcds);
        V* per = rocke_b_div(b, total, cx0);
        V* cx1 = ci(b, spec.xcds);
        V* md = rocke_b_mod(b, bid, cx1);
        V* hi = rocke_b_mul(b, md, per);
        V* cx2 = ci(b, spec.xcds);
        V* lo = rocke_b_div(b, bid, cx2);
        V* remap = rocke_b_add(b, hi, lo);
        V* cx3 = ci(b, spec.xcds);
        V* tmod = rocke_b_mod(b, total, cx3);
        V* c0 = ci(b, 0);
        V* even = rocke_b_cmp_eq(b, tmod, c0);
        bid = rocke_b_select(b, even, remap, bid);
    }
    if(spec.group_m <= 1)
    {
        *tm = rocke_b_div(b, bid, ntn);
        *tn = rocke_b_mod(b, bid, ntn);
        return;
    }
    V* cgm = ci(b, spec.group_m);
    V* g = rocke_b_smin(b, ntm, cgm);
    V* in_group = rocke_b_mul(b, g, ntn);
    V* group = rocke_b_div(b, bid, in_group);
    V* first = rocke_b_mul(b, group, g);
    V* rest = rocke_b_sub(b, ntm, first);
    V* size = rocke_b_smin(b, rest, g);
    V* r = rocke_b_mod(b, bid, in_group);
    V* rm = rocke_b_mod(b, r, size);
    *tm = rocke_b_add(b, first, rm);
    *tn = rocke_b_div(b, r, size);
}

/* Python _frag: one f64, or the (re, im) pair of a complex element. */
void frag(B* b, V* lds, V* row, V* col, bool cx, V** re, V** im)
{
    if(cx)
    {
        V* c2 = ci(b, 2);
        V* col2 = rocke_b_mul(b, col, c2);
        V* idx[2] = {row, col2};
        V* v = rocke_b_smem_load_vN(b, lds, idx, 2, rocke_f64(), 2);
        *re = rocke_b_vec_extract(b, v, 0);
        *im = rocke_b_vec_extract(b, v, 1);
        return;
    }
    V* idx[2] = {row, col};
    V* v = rocke_b_smem_load_vN(b, lds, idx, 2, rocke_f64(), 1);
    *re = rocke_b_vec_extract(b, v, 0);
    *im = nullptr;
}

/* Python _mma_step. */
void mma_step(B* b,
              const char* atom,
              const std::vector<V*>& far,
              const std::vector<V*>& fai,
              const std::vector<V*>& fbr,
              const std::vector<V*>& fbi,
              std::vector<V*>& accs,
              const rocke_gett_f64_problem_t& p,
              int complex_accs)
{
    int am_n = (int)far.size(), an_n = (int)fbr.size();
    if(!p.complex_)
    {
        for(int am = 0; am < am_n; ++am)
            for(int an = 0; an < an_n; ++an)
            {
                int i = am * an_n + an;
                accs[i] = rocke_b_mma(b, atom, far[am], fbr[an], accs[i], NULL, 0);
            }
        return;
    }
    if(complex_accs == 4)
    {
        for(int am = 0; am < am_n; ++am)
            for(int an = 0; an < an_n; ++an)
            {
                int i = 4 * (am * an_n + an);
                accs[i] = rocke_b_mma(b, atom, far[am], fbr[an], accs[i], NULL, 0);
                accs[i + 1] = rocke_b_mma(b, atom, fai[am], fbi[an], accs[i + 1], NULL, 0);
                accs[i + 2] = rocke_b_mma(b, atom, far[am], fbi[an], accs[i + 2], NULL, 0);
                accs[i + 3] = rocke_b_mma(b, atom, fai[am], fbr[an], accs[i + 3], NULL, 0);
            }
        return;
    }
    int sa = p.conj_a ? -1 : 1;
    int sb = p.conj_b ? -1 : 1;
    std::vector<V*> neg_ai, neg_bi;
    if(sa * sb == 1 || sa == -1)
        for(int am = 0; am < am_n; ++am)
            neg_ai.push_back(rocke_b_fneg(b, fai[am]));
    if(sb == -1)
        for(int an = 0; an < an_n; ++an)
            neg_bi.push_back(rocke_b_fneg(b, fbi[an]));
    for(int am = 0; am < am_n; ++am)
        for(int an = 0; an < an_n; ++an)
        {
            int i = 2 * (am * an_n + an);
            V* acc_r = accs[i];
            V* acc_i = accs[i + 1];
            acc_r = rocke_b_mma(b, atom, far[am], fbr[an], acc_r, NULL, 0);
            acc_r = rocke_b_mma(
                b, atom, sa * sb == 1 ? neg_ai[am] : fai[am], fbi[an], acc_r, NULL, 0);
            acc_i = rocke_b_mma(b, atom, far[am], sb == -1 ? neg_bi[an] : fbi[an], acc_i, NULL, 0);
            acc_i = rocke_b_mma(b, atom, sa == -1 ? neg_ai[am] : fai[am], fbr[an], acc_i, NULL, 0);
            accs[i] = acc_r;
            accs[i + 1] = acc_i;
        }
}

/* Python _Frag: the K row of the lane and the LDS column of each atom per k-step. */
struct frag_t
{
    V* k_row;
    std::vector<std::vector<V*>> cols;
};

/* Python _LoopCtx. */
struct loop_ctx
{
    B* b;
    const rocke_gett_f64_spec_t* spec;
    const rocke_mma_op_t* atom;
    operand_t* opa;
    operand_t* opb;
    V* a_lds;
    V* b_lds;
    frag_t fa;
    frag_t fb;
    ival nk;
    V* zero;
    int n_acc;
    V* zero_i;
    int na;
    bool cx;
    V* k_lo; /* split_k: the first K tile of this CTA (nullptr: 0) */

    std::vector<V*> mfma_block(std::vector<V*> accs, V* row0)
    {
        const rocke_gett_f64_spec_t& s = *spec;
        for(int ks = 0; ks < s.tile_k / atom->k; ++ks)
        {
            V* ka = fa.k_row;
            if(ks)
            {
                V* c = ci(b, ks * atom->k);
                ka = rocke_b_add(b, fa.k_row, c);
            }
            V* row_a = rocke_b_add(b, row0, ka);
            std::vector<V*> far, fai, fbr, fbi;
            for(V* col : fa.cols[ks])
            {
                V *re, *im;
                frag(b, a_lds, row_a, col, cx, &re, &im);
                far.push_back(re);
                fai.push_back(im);
            }
            V* kb = fb.k_row;
            if(ks)
            {
                V* c = ci(b, ks * atom->k);
                kb = rocke_b_add(b, fb.k_row, c);
            }
            V* row_b = rocke_b_add(b, row0, kb);
            for(V* col : fb.cols[ks])
            {
                V *re, *im;
                frag(b, b_lds, row_b, col, cx, &re, &im);
                fbr.push_back(re);
                fbi.push_back(im);
            }
            mma_step(b, atom->op_id, far, fai, fbr, fbi, accs, s.problem, s.complex_accs);
        }
        return accs;
    }
};

typedef std::function<void(V*, V*, std::vector<V*>&, std::vector<V*>&)> run_fn;

/* Python _blocked: k_block blocks of loop units, each from zero accumulators. */
void blocked(loop_ctx& ctx,
             const ival& n_units,
             int per_block,
             const run_fn& run,
             std::vector<V*>& carry,
             std::vector<V*>* sums_out)
{
    B* b = ctx.b;
    V* nblk;
    if(n_units.v == nullptr)
        nblk = ci(b, (n_units.c + per_block - 1) / per_block);
    else
    {
        V* c = ci(b, per_block - 1);
        V* a = rocke_b_add(b, n_units.v, c);
        V* cp = ci(b, per_block);
        nblk = rocke_b_div(b, a, cp);
    }
    std::vector<std::string> names;
    for(int i = 0; i < ctx.n_acc; ++i)
        names.push_back("sum" + std::to_string(i));
    for(size_t i = 0; i < carry.size(); ++i)
        names.push_back("cr" + std::to_string(i));
    std::vector<rocke_iter_arg_t> args;
    for(int i = 0; i < ctx.n_acc; ++i)
        args.push_back({names[i].c_str(), ctx.zero});
    for(size_t i = 0; i < carry.size(); ++i)
        args.push_back({names[ctx.n_acc + i].c_str(), carry[i]});
    V* lo0 = ci(b, 0);
    V* st = ci(b, 1);
    rocke_for_t outer
        = rocke_b_scf_for_iter(b, lo0, nblk, st, args.data(), (int)args.size(), "kb", false, true);
    rocke_b_region_enter(b, outer.body);
    {
        std::vector<V*> sums(outer.iter_vars, outer.iter_vars + ctx.n_acc);
        std::vector<V*> rest(outer.iter_vars + ctx.n_acc, outer.iter_vars + outer.num_iter_vars);
        V* cpb = ci(b, per_block);
        V* lo = rocke_b_mul(b, outer.iv, cpb);
        V* cpb2 = ci(b, per_block);
        V* lpb = rocke_b_add(b, lo, cpb2);
        V* cnu = get(b, n_units);
        V* hi = rocke_b_smin(b, lpb, cnu);
        std::vector<V*> accs(ctx.n_acc, ctx.zero);
        run(lo, hi, accs, rest);
        for(int i = 0; i < ctx.n_acc; ++i)
            sums[i] = rocke_b_fadd(b, sums[i], accs[i]);
        std::vector<V*> yields = sums;
        yields.insert(yields.end(), rest.begin(), rest.end());
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    }
    rocke_b_region_leave(b);
    sums_out->assign(outer.op->results, outer.op->results + ctx.n_acc);
    carry.assign(outer.op->results + ctx.n_acc, outer.op->results + outer.num_iter_vars);
}

/* Python _k_loop: every K tile but the last in the loop (no branch in the
 * body); the last tile's MFMAs after it, with no loads, stores or barrier. */
std::vector<V*> k_loop(loop_ctx& ctx, std::vector<V*> carry)
{
    B* b = ctx.b;
    const rocke_gett_f64_spec_t& s = *ctx.spec;
    int ns = s.lds_stages, pd = s.prefetch_depth;
    operand_t& opa = *ctx.opa;
    operand_t& opb = *ctx.opb;
    run_fn run = [&](V* lo, V* hi, std::vector<V*>& accs_io, std::vector<V*>& carry_io) {
        int n_acc = ctx.n_acc;
        std::vector<std::string> names;
        for(int i = 0; i < n_acc; ++i)
            names.push_back("acc" + std::to_string(i));
        if(pd == 2)
        {
            for(int i = 0; i < ctx.na; ++i)
                names.push_back("ra" + std::to_string(i));
            for(int i = ctx.na; i < (int)carry_io.size(); ++i)
                names.push_back("rb" + std::to_string(i - ctx.na));
        }
        std::vector<rocke_iter_arg_t> args;
        for(int i = 0; i < n_acc; ++i)
            args.push_back({names[i].c_str(), accs_io[i]});
        if(pd == 2)
            for(size_t i = 0; i < carry_io.size(); ++i)
                args.push_back({names[n_acc + i].c_str(), carry_io[i]});
        V* st = ci(b, 1);
        rocke_for_t loop
            = rocke_b_scf_for_iter(b, lo, hi, st, args.data(), (int)args.size(), "kt", false, true);
        rocke_b_region_enter(b, loop.body);
        {
            V* kt = loop.iv;
            std::vector<V*> accs(loop.iter_vars, loop.iter_vars + n_acc);
            std::vector<V*> cur_a, cur_b;
            if(pd == 2)
            {
                cur_a.assign(loop.iter_vars + n_acc, loop.iter_vars + n_acc + ctx.na);
                cur_b.assign(loop.iter_vars + n_acc + ctx.na, loop.iter_vars + loop.num_iter_vars);
            }
            V* stage = ctx.zero_i;
            if(ns == 2)
            {
                V* c1 = ci(b, 1);
                stage = rocke_b_land(b, kt, c1);
            }
            /* split_k: kt counts this CTA's tiles; the loads take the global index. */
            V* ktg = ctx.k_lo != nullptr ? rocke_b_add(b, kt, ctx.k_lo) : kt;
            V* cpd = ci(b, pd);
            V* nxt = rocke_b_add(b, ktg, cpd);
            std::vector<V*> pre_a, pre_b;
            if(s.prefetch)
            {
                pre_a = opa.load(nxt);
                pre_b = opb.load(nxt);
            }
            if(s.sched_fence)
                rocke_b_sched_barrier(b, 0);
            V* ctk = ci(b, s.tile_k);
            V* row0 = rocke_b_mul(b, stage, ctk);
            accs = ctx.mfma_block(accs, row0);
            if(s.sched_fence)
                rocke_b_sched_barrier(b, 0);
            if(!s.prefetch)
            {
                pre_a = opa.load(nxt);
                pre_b = opb.load(nxt);
            }
            V* nstage = ctx.zero_i;
            if(ns == 2)
            {
                V* c1c = ci(b, 1);
                nstage = rocke_b_xor(b, stage, c1c);
            }
            else
                rocke_b_sync(b);
            opa.store_lds(pd == 2 ? cur_a : pre_a, nstage);
            opb.store_lds(pd == 2 ? cur_b : pre_b, nstage);
            rocke_b_sync(b);
            std::vector<V*> yields = accs;
            if(pd == 2)
            {
                yields.insert(yields.end(), pre_a.begin(), pre_a.end());
                yields.insert(yields.end(), pre_b.begin(), pre_b.end());
            }
            rocke_b_scf_yield(b, yields.data(), (int)yields.size());
        }
        rocke_b_region_leave(b);
        accs_io.assign(loop.op->results, loop.op->results + n_acc);
        carry_io.assign(loop.op->results + n_acc, loop.op->results + loop.num_iter_vars);
    };
    ival last;
    if(ctx.nk.v == nullptr)
        last = iv_const(ctx.nk.c - 1);
    else
    {
        V* c1 = ci(b, 1);
        last = iv_val(rocke_b_sub(b, ctx.nk.v, c1));
    }
    std::vector<V*> accs;
    if(s.k_block)
        blocked(ctx, last, s.k_block, run, carry, &accs);
    else
    {
        V* lo = ci(b, 0);
        V* hi = get(b, last);
        accs.assign(ctx.n_acc, ctx.zero);
        run(lo, hi, accs, carry);
    }
    /* The last K tile: its MFMAs only, from the stage the loop left it in. */
    V* kt = get(b, last);
    V* stage = ctx.zero_i;
    if(ns == 2)
    {
        V* c1 = ci(b, 1);
        stage = rocke_b_land(b, kt, c1);
    }
    if(s.sched_fence)
        rocke_b_sched_barrier(b, 0);
    V* ctk = ci(b, s.tile_k);
    V* row0 = rocke_b_mul(b, stage, ctk);
    accs = ctx.mfma_block(accs, row0);
    if(s.sched_fence)
        rocke_b_sched_barrier(b, 0);
    return accs;
}

/* Python _k_loop_unroll2. */
std::vector<V*> k_loop_unroll2(loop_ctx& ctx)
{
    B* b = ctx.b;
    const rocke_gett_f64_spec_t& s = *ctx.spec;
    operand_t& opa = *ctx.opa;
    operand_t& opb = *ctx.opb;
    auto half = [&](std::vector<V*> accs, V* kt, int stg, bool do_next) {
        std::vector<V*> pre_a, pre_b;
        if(do_next)
        {
            V* c1 = ci(b, 1);
            V* nxt = rocke_b_add(b, kt, c1);
            pre_a = opa.load(nxt);
            pre_b = opb.load(nxt);
        }
        if(s.sched_fence)
            rocke_b_sched_barrier(b, 0);
        V* row0 = ci(b, stg * s.tile_k);
        accs = ctx.mfma_block(accs, row0);
        if(s.sched_fence)
            rocke_b_sched_barrier(b, 0);
        if(do_next)
        {
            V* other = ci(b, 1 - stg);
            opa.store_lds(pre_a, other);
            opb.store_lds(pre_b, other);
            rocke_b_sync(b);
        }
        return accs;
    };
    run_fn run = [&](V* lo, V* hi, std::vector<V*>& accs_io, std::vector<V*>& carry_io) {
        (void)carry_io;
        std::vector<std::string> names;
        for(int i = 0; i < ctx.n_acc; ++i)
            names.push_back("acc" + std::to_string(i));
        std::vector<rocke_iter_arg_t> args;
        for(int i = 0; i < ctx.n_acc; ++i)
            args.push_back({names[i].c_str(), accs_io[i]});
        V* st = ci(b, 1);
        rocke_for_t loop2
            = rocke_b_scf_for_iter(b, lo, hi, st, args.data(), ctx.n_acc, "kt2", false, true);
        rocke_b_region_enter(b, loop2.body);
        {
            std::vector<V*> accs(loop2.iter_vars, loop2.iter_vars + ctx.n_acc);
            V* c2 = ci(b, 2);
            V* kt = rocke_b_mul(b, loop2.iv, c2);
            accs = half(accs, kt, 0, true);
            V* c1 = ci(b, 1);
            V* kt1 = rocke_b_add(b, kt, c1);
            accs = half(accs, kt1, 1, true);
            rocke_b_scf_yield(b, accs.data(), (int)accs.size());
        }
        rocke_b_region_leave(b);
        accs_io.assign(loop2.op->results, loop2.op->results + ctx.n_acc);
    };
    ival looped;
    bool tail_pair;
    if(ctx.nk.v == nullptr)
    {
        /* odd: every pair's stores are live, the last tile follows alone;
         * even: the last pair follows the loop. */
        looped = iv_const(ctx.nk.c % 2 ? ctx.nk.c / 2 : ctx.nk.c / 2 - 1);
        tail_pair = ctx.nk.c % 2 == 0;
    }
    else
    {
        V* c1 = ci(b, 1);
        V* a = rocke_b_add(b, ctx.nk.v, c1);
        V* c2 = ci(b, 2);
        V* pairs = rocke_b_div(b, a, c2);
        V* c1b = ci(b, 1);
        looped = iv_val(rocke_b_sub(b, pairs, c1b));
        tail_pair = true;
    }
    std::vector<V*> accs;
    std::vector<V*> none;
    if(s.k_block)
        blocked(ctx, looped, s.k_block / 2, run, none, &accs);
    else
    {
        V* lo = ci(b, 0);
        V* hi = get(b, looped);
        accs.assign(ctx.n_acc, ctx.zero);
        run(lo, hi, accs, none);
    }
    if(tail_pair)
    {
        V* lp = get(b, looped);
        V* c2 = ci(b, 2);
        V* kt = rocke_b_mul(b, lp, c2);
        accs = half(accs, kt, 0, true);
        V* c1 = ci(b, 1);
        V* kt1 = rocke_b_add(b, kt, c1);
        accs = half(accs, kt1, 1, false);
    }
    else
    {
        V* last = ci(b, ctx.nk.c - 1);
        accs = half(accs, last, 0, false);
    }
    return accs;
}

/* Python _Scalars. */
struct scalars_t
{
    bool cx;
    std::vector<V*> alpha, beta;
    V* zero;
    V* a_zero;
    V* ai_zero;
    V* bi_zero;
};

scalars_t make_scalars(B* b, const std::vector<V*>& alpha, const std::vector<V*>& beta, bool cx)
{
    scalars_t sc = {};
    sc.cx = cx;
    sc.alpha = alpha;
    sc.beta = beta;
    sc.zero = rocke_b_const_f64(b, 0.0);
    if(cx)
    {
        V* ar_z = rocke_b_fcmp(b, "oeq", alpha[0], sc.zero);
        sc.ai_zero = rocke_b_fcmp(b, "oeq", alpha[1], sc.zero);
        sc.a_zero = rocke_b_land(b, ar_z, sc.ai_zero);
        sc.bi_zero = beta.empty() ? nullptr : rocke_b_fcmp(b, "oeq", beta[1], sc.zero);
    }
    else
        sc.a_zero = rocke_b_fcmp(b, "oeq", alpha[0], sc.zero);
    return sc;
}

/* Python _combine. */
void combine(B* b, const scalars_t& sc, V* const* acc, V* const* c_val, V** out)
{
    if(!sc.cx)
    {
        V* a = sc.alpha[0];
        if(c_val == nullptr)
        {
            V* t = rocke_b_fmul(b, a, acc[0]);
            out[0] = rocke_b_select(b, sc.a_zero, sc.zero, t);
            return;
        }
        V* bc = rocke_b_fmul(b, sc.beta[0], c_val[0]);
        V* t = rocke_b_fma(b, a, acc[0], bc);
        out[0] = rocke_b_select(b, sc.a_zero, bc, t);
        return;
    }
    V *xr = acc[0], *xi = acc[1];
    V *ar = sc.alpha[0], *ai = sc.alpha[1];
    if(c_val == nullptr)
    {
        V* t = rocke_b_fmul(b, ai, xi);
        V* nt = rocke_b_fneg(b, t);
        V* re_f = rocke_b_fma(b, ar, xr, nt);
        V* u = rocke_b_fmul(b, ai, xr);
        V* im_f = rocke_b_fma(b, ar, xi, u);
        V* re_r = rocke_b_fmul(b, ar, xr);
        V* im_r = rocke_b_fmul(b, ar, xi);
        V* re_s = rocke_b_select(b, sc.ai_zero, re_r, re_f);
        out[0] = rocke_b_select(b, sc.a_zero, sc.zero, re_s);
        V* im_s = rocke_b_select(b, sc.ai_zero, im_r, im_f);
        out[1] = rocke_b_select(b, sc.a_zero, sc.zero, im_s);
        return;
    }
    V *br = sc.beta[0], *bi = sc.beta[1];
    V *cr = c_val[0], *cim = c_val[1];
    V* t1 = rocke_b_fmul(b, bi, cim);
    V* nt1 = rocke_b_fneg(b, t1);
    V* tr_f = rocke_b_fma(b, br, cr, nt1);
    V* t2 = rocke_b_fmul(b, bi, cr);
    V* ti_f = rocke_b_fma(b, br, cim, t2);
    V* brc = rocke_b_fmul(b, br, cr);
    V* tr = rocke_b_select(b, sc.bi_zero, brc, tr_f);
    V* bri = rocke_b_fmul(b, br, cim);
    V* ti = rocke_b_select(b, sc.bi_zero, bri, ti_f);
    V* nai = rocke_b_fneg(b, ai);
    V* r0 = rocke_b_fma(b, nai, xi, tr);
    V* re_f = rocke_b_fma(b, ar, xr, r0);
    V* i0 = rocke_b_fma(b, ai, xr, ti);
    V* im_f = rocke_b_fma(b, ar, xi, i0);
    V* re_r = rocke_b_fma(b, ar, xr, tr);
    V* im_r = rocke_b_fma(b, ar, xi, ti);
    V* re_s = rocke_b_select(b, sc.ai_zero, re_r, re_f);
    out[0] = rocke_b_select(b, sc.a_zero, tr, re_s);
    V* im_s = rocke_b_select(b, sc.ai_zero, im_r, im_f);
    out[1] = rocke_b_select(b, sc.a_zero, ti, im_s);
}

/* Python _epilogue. */
void epilogue(B* b,
              const rocke_gett_f64_spec_t& spec,
              const rocke_mma_op_t* atom,
              const std::vector<V*>& accs,
              V* lane,
              V* wm,
              V* wn,
              V* m0,
              V* n0,
              V* rd,
              V* rc,
              const std::vector<V*>& alpha,
              const std::vector<V*>& beta,
              const geo_t& geo)
{
    const rocke_gett_f64_problem_t& p = spec.problem;
    bool cx = p.complex_;
    int es = elem_bytes(p);
    int wave_m = spec.tile_m / spec.warps_m, wave_n = spec.tile_n / spec.warps_n;
    int am_n = wave_m / atom->m, an_n = wave_n / atom->n;
    /* Output coordinates of each accumulator slot from the catalog lane map. */
    std::vector<V*> c_rows;
    V* col_l = nullptr;
    for(int slot = 0; slot < atom->c_frag_len; ++slot)
    {
        V *r, *c;
        rocke_layout_map_coord(atom->c_layout, b, lane, slot, &r, &c);
        c_rows.push_back(r);
        if(col_l == nullptr)
            col_l = c;
    }
    V* cwm = ci(b, wave_m);
    V* wmo = rocke_b_mul(b, wm, cwm);
    V* mw = rocke_b_add(b, m0, wmo);
    V* cwn = ci(b, wave_n);
    V* wno = rocke_b_mul(b, wn, cwn);
    V* nw = rocke_b_add(b, n0, wno);

    struct col_t
    {
        V* ok;
        V* dn;
        V* c_n;
    };
    std::vector<col_t> cols;
    for(int an = 0; an < an_n; ++an)
    {
        V* inner = col_l;
        if(an)
        {
            V* c = ci(b, an * atom->n);
            inner = rocke_b_add(b, col_l, c);
        }
        V* n = rocke_b_add(b, nw, inner);
        V* cN = get(b, geo.N);
        V* ok = rocke_b_cmp_lt(b, n, cN);
        V* c0 = ci(b, 0);
        V* n_safe = rocke_b_select(b, ok, n, c0);
        V* dn = gmixed(b, n_safe, geo.d_n);
        V* c_n = rc != nullptr ? gmixed(b, n_safe, geo.c_n) : nullptr;
        cols.push_back({ok, dn, c_n});
    }
    V* use_c = nullptr;
    if(rc != nullptr)
    {
        V* z = rocke_b_const_f64(b, 0.0);
        V* eq = rocke_b_fcmp(b, "oeq", beta[0], z);
        use_c = rocke_b_lnot(b, eq);
        if(cx)
        {
            V* z2 = rocke_b_const_f64(b, 0.0);
            V* eq2 = rocke_b_fcmp(b, "oeq", beta[1], z2);
            V* ne2 = rocke_b_lnot(b, eq2);
            use_c = rocke_b_lor(b, use_c, ne2);
        }
    }
    scalars_t sc = make_scalars(b, alpha, beta, cx);
    int words = cx ? 2 : 1;
    const rocke_type_t* cvty = rocke_vector_type(b, rocke_f64(), words);
    const rocke_type_t* svty = rocke_vector_type(b, rocke_f32(), 2 * words);
    /* Per row group of atoms: every C load first, then the combines and the D
     * stores. C may be D itself (in place), so the compiler may not move a C
     * load above an earlier D store; issuing the loads of a group first keeps
     * them in flight together, and reading every C element of the group before
     * writing it keeps the in-place update correct. */
    struct elem_t
    {
        int slot;
        int an;
        V* ok;
        V* dm;
        V* dn;
        V* cv;
    };
    for(int am = 0; am < am_n; ++am)
    {
        std::vector<elem_t> elems;
        for(int slot = 0; slot < atom->c_frag_len; ++slot)
        {
            V* r = c_rows[slot];
            V* rr = r;
            if(am)
            {
                V* cr = ci(b, am * atom->m);
                rr = rocke_b_add(b, r, cr);
            }
            V* m = rocke_b_add(b, mw, rr);
            V* cM = get(b, geo.M);
            V* mok = rocke_b_cmp_lt(b, m, cM);
            V* c0 = ci(b, 0);
            V* m_safe = rocke_b_select(b, mok, m, c0);
            V* dm = gmixed(b, m_safe, geo.d_m);
            V* c_m = rc != nullptr ? gmixed(b, m_safe, geo.c_m) : nullptr;
            for(int an = 0; an < an_n; ++an)
            {
                const col_t& col = cols[an];
                V* ok = rocke_b_land(b, mok, col.ok);
                V* cv = nullptr;
                if(rc != nullptr)
                {
                    V* cs = rocke_b_add(b, c_m, col.c_n);
                    V* ces = ci(b, es);
                    V* coff = rocke_b_mul(b, cs, ces);
                    V* okc = rocke_b_land(b, ok, use_c);
                    V* cmask = ci(b, k_mask_offset);
                    coff = rocke_b_select(b, okc, coff, cmask);
                    V* c0s = ci(b, 0);
                    V* raw = rocke_b_buffer_load_vN(b, rc, coff, c0s, rocke_i32(), 2 * words);
                    cv = rocke_b_vec_bitcast(b, raw, cvty);
                }
                elems.push_back({slot, an, ok, dm, col.dn, cv});
            }
        }
        for(const elem_t& e : elems)
        {
            int slot = e.slot;
            int i = am * an_n + e.an;
            V* acc[2] = {nullptr, nullptr};
            if(cx && spec.complex_accs == 4)
            {
                /* Cr = rr - sa*sb*ii ; Ci = sb*ri + sa*ir (sa / sb = -1: CONJ) */
                V* q[4];
                for(int j = 0; j < 4; ++j)
                    q[j] = rocke_b_vec_extract(b, accs[4 * i + j], slot);
                int sa = p.conj_a ? -1 : 1;
                int sb = p.conj_b ? -1 : 1;
                acc[0] = sa * sb == 1 ? rocke_b_fsub(b, q[0], q[1]) : rocke_b_fadd(b, q[0], q[1]);
                if(sa == sb)
                {
                    V* t = rocke_b_fadd(b, q[2], q[3]);
                    acc[1] = sa == 1 ? t : rocke_b_fneg(b, t);
                }
                else
                    acc[1] = sa == 1 ? rocke_b_fsub(b, q[3], q[2]) : rocke_b_fsub(b, q[2], q[3]);
            }
            else if(cx)
            {
                acc[0] = rocke_b_vec_extract(b, accs[2 * i], slot);
                acc[1] = rocke_b_vec_extract(b, accs[2 * i + 1], slot);
            }
            else
                acc[0] = rocke_b_vec_extract(b, accs[i], slot);
            V* c_val[2] = {nullptr, nullptr};
            if(e.cv != nullptr)
                for(int j = 0; j < words; ++j)
                    c_val[j] = rocke_b_vec_extract(b, e.cv, j);
            V* out[2] = {nullptr, nullptr};
            combine(b, sc, acc, e.cv != nullptr ? c_val : nullptr, out);
            V* ds = rocke_b_add(b, e.dm, e.dn);
            V* ces2 = ci(b, es);
            V* doff = rocke_b_mul(b, ds, ces2);
            V* cmask2 = ci(b, k_mask_offset);
            doff = rocke_b_select(b, e.ok, doff, cmask2);
            V* packed = rocke_b_vec_pack(b, out, words, rocke_f64());
            V* vec = rocke_b_vec_bitcast(b, packed, svty);
            V* c0s2 = ci(b, 0);
            rocke_b_buffer_store_vN_f32(b, rd, doff, c0s2, vec, 2 * words);
        }
    }
}

/* Python _workspace_rsrc: the partial-sum slice W[z][l] (M*N elements, N
 * fastest) this CTA writes (split_k). */
V* workspace_rsrc(
    B* b, const rocke_gett_f64_spec_t& s, const geo_t& geo, V* W, V* kz, V* lb, int es)
{
    const rocke_gett_f64_problem_t& p = s.problem;
    V *off, *span;
    if(geo.rt)
    {
        V* zL = rocke_b_mul(b, kz, geo.L.v);
        V* zl = rocke_b_add(b, zL, lb);
        V* mn = rocke_b_mul(b, geo.M.v, geo.N.v);
        V* zl64 = rocke_b_zext(b, zl, rocke_i64());
        V* mn64 = rocke_b_zext(b, mn, rocke_i64());
        V* prod = rocke_b_mul(b, zl64, mn64);
        V* ces = rocke_b_const_i64(b, es);
        off = rocke_b_mul(b, prod, ces);
        V* ces32 = ci(b, es);
        span = rocke_b_mul(b, mn, ces32);
    }
    else
    {
        V* cL = ci(b, p.L);
        V* zL = rocke_b_mul(b, kz, cL);
        V* zl = rocke_b_add(b, zL, lb);
        V* zl64 = rocke_b_zext(b, zl, rocke_i64());
        V* cmn = rocke_b_const_i64(b, p.M * p.N * es);
        off = rocke_b_mul(b, zl64, cmn);
        span = ci(b, p.M * p.N * es);
    }
    V* base = rocke_b_global_ptr_add(b, W, off);
    return rocke_b_buffer_rsrc(b, base, span);
}

/* Python _acc_value: the (re[, im]) accumulator value of atom i at slot. */
void acc_value(
    B* b, const rocke_gett_f64_spec_t& spec, const std::vector<V*>& accs, int i, int slot, V** acc)
{
    const rocke_gett_f64_problem_t& p = spec.problem;
    bool cx = p.complex_;
    if(cx && spec.complex_accs == 4)
    {
        /* Cr = rr - sa*sb*ii ; Ci = sb*ri + sa*ir (sa / sb = -1: CONJ) */
        V* q[4];
        for(int j = 0; j < 4; ++j)
            q[j] = rocke_b_vec_extract(b, accs[4 * i + j], slot);
        int sa = p.conj_a ? -1 : 1;
        int sb = p.conj_b ? -1 : 1;
        acc[0] = sa * sb == 1 ? rocke_b_fsub(b, q[0], q[1]) : rocke_b_fadd(b, q[0], q[1]);
        if(sa == sb)
        {
            V* t = rocke_b_fadd(b, q[2], q[3]);
            acc[1] = sa == 1 ? t : rocke_b_fneg(b, t);
        }
        else
            acc[1] = sa == 1 ? rocke_b_fsub(b, q[3], q[2]) : rocke_b_fsub(b, q[2], q[3]);
    }
    else if(cx)
    {
        acc[0] = rocke_b_vec_extract(b, accs[2 * i], slot);
        acc[1] = rocke_b_vec_extract(b, accs[2 * i + 1], slot);
    }
    else
    {
        acc[0] = rocke_b_vec_extract(b, accs[i], slot);
        acc[1] = nullptr;
    }
}

/* Python _epilogue_partial: the accumulators to the workspace slice, row m at
 * m*N (split_k; no C, no alpha / beta). */
void epilogue_partial(B* b,
                      const rocke_gett_f64_spec_t& spec,
                      const rocke_mma_op_t* atom,
                      const std::vector<V*>& accs,
                      V* lane,
                      V* wm,
                      V* wn,
                      V* m0,
                      V* n0,
                      V* rw,
                      const geo_t& geo)
{
    const rocke_gett_f64_problem_t& p = spec.problem;
    int words = p.complex_ ? 2 : 1;
    int es = elem_bytes(p);
    int wave_m = spec.tile_m / spec.warps_m, wave_n = spec.tile_n / spec.warps_n;
    int am_n = wave_m / atom->m, an_n = wave_n / atom->n;
    std::vector<V*> c_rows;
    V* col_l = nullptr;
    for(int slot = 0; slot < atom->c_frag_len; ++slot)
    {
        V *r, *c;
        rocke_layout_map_coord(atom->c_layout, b, lane, slot, &r, &c);
        c_rows.push_back(r);
        if(col_l == nullptr)
            col_l = c;
    }
    V* cwm = ci(b, wave_m);
    V* wmo = rocke_b_mul(b, wm, cwm);
    V* mw = rocke_b_add(b, m0, wmo);
    V* cwn = ci(b, wave_n);
    V* wno = rocke_b_mul(b, wn, cwn);
    V* nw = rocke_b_add(b, n0, wno);
    std::vector<V*> nok, ncol;
    for(int an = 0; an < an_n; ++an)
    {
        V* inner = col_l;
        if(an)
        {
            V* c = ci(b, an * atom->n);
            inner = rocke_b_add(b, col_l, c);
        }
        V* n = rocke_b_add(b, nw, inner);
        V* cN = get(b, geo.N);
        nok.push_back(rocke_b_cmp_lt(b, n, cN));
        ncol.push_back(n);
    }
    const rocke_type_t* svty = rocke_vector_type(b, rocke_f32(), 2 * words);
    for(int am = 0; am < am_n; ++am)
        for(int slot = 0; slot < atom->c_frag_len; ++slot)
        {
            V* r = c_rows[slot];
            V* rr = r;
            if(am)
            {
                V* cr = ci(b, am * atom->m);
                rr = rocke_b_add(b, r, cr);
            }
            V* m = rocke_b_add(b, mw, rr);
            V* cM = get(b, geo.M);
            V* mok = rocke_b_cmp_lt(b, m, cM);
            V* cN = get(b, geo.N);
            V* row = rocke_b_mul(b, m, cN);
            for(int an = 0; an < an_n; ++an)
            {
                V* ok = rocke_b_land(b, mok, nok[an]);
                V* acc[2] = {nullptr, nullptr};
                acc_value(b, spec, accs, am * an_n + an, slot, acc);
                V* rn = rocke_b_add(b, row, ncol[an]);
                V* ces = ci(b, es);
                V* off = rocke_b_mul(b, rn, ces);
                V* cmask = ci(b, k_mask_offset);
                off = rocke_b_select(b, ok, off, cmask);
                V* packed = rocke_b_vec_pack(b, acc, words, rocke_f64());
                V* vec = rocke_b_vec_bitcast(b, packed, svty);
                V* c0 = ci(b, 0);
                rocke_b_buffer_store_vN_f32(b, rw, off, c0, vec, 2 * words);
            }
        }
}

/* Python gett_f64_reduce_name. */
std::string reduce_name(const rocke_gett_f64_spec_t& spec)
{
    const rocke_gett_f64_problem_t& p = spec.problem;
    std::string parts[4];
    parts[0] = p.complex_ ? "c128" : "f64";
    std::string h;
    if(spec.runtime_sizes)
    {
        rocke_gett_f64_problem_t tp;
        parts[1] = "rt";
        h = runtime_template(p, &tp) ? problem_hash(tp) : std::string("invalid");
    }
    else
    {
        parts[1] = "M" + std::to_string(p.M) + "N" + std::to_string(p.N) + "K" + std::to_string(p.K)
                   + "L" + std::to_string(p.L);
        h = problem_hash(p);
    }
    parts[2] = "sk" + std::to_string(spec.split_k);
    parts[3] = "h" + h;
    const char* cparts[4]
        = {parts[0].c_str(), parts[1].c_str(), parts[2].c_str(), parts[3].c_str()};
    const char* flag_names[] = {"bil"};
    const int flag_on[] = {spec.bilinear ? 1 : 0};
    std::string prefix = std::string(spec.name ? spec.name : "rocke_gett_f64") + "_reduce";
    char out[512];
    if(rocke_kernel_name_join(
           prefix.c_str(), cparts, 4, flag_names, flag_on, 1, out, sizeof out, NULL)
       != ROCKE_OK)
        return "";
    return out;
}

} // namespace

extern "C" {

void rocke_gett_digits_norm(rocke_gett_digits_t* d)
{
    if(d == nullptr)
        return;
    int w = 0;
    for(int i = 0; i < d->n; ++i)
    {
        if(d->extent[i] != 1)
        {
            d->extent[w] = d->extent[i];
            d->stride[w] = d->stride[i];
            ++w;
        }
    }
    d->n = w;
    if(w == 0)
    {
        d->n = 1;
        d->extent[0] = 1;
        d->stride[0] = 0;
    }
}

rocke_gett_f64_spec_t rocke_gett_f64_spec_default(const rocke_gett_f64_problem_t* problem)
{
    rocke_gett_f64_spec_t s = {};
    if(problem != nullptr)
        s.problem = *problem;
    s.tile_m = 128;
    s.tile_n = 128;
    s.tile_k = 16;
    s.warps_m = 2;
    s.warps_n = 2;
    s.lds_pad = -1;
    s.lds_swizzle = true;
    s.group_m = 8;
    s.xcds = 0;
    s.waves_per_eu = 0;
    s.bilinear = true;
    s.prefetch = true;
    s.sched_fence = true;
    s.lds_swizzle_mode = 0;
    s.lds_stages = 2;
    s.prefetch_depth = 1;
    s.complex_accs = 2;
    s.unroll2 = false;
    s.k_stagger = 0;
    s.name = "rocke_gett_f64";
    s.k_block = 0;
    s.runtime_sizes = false;
    s.split_k = 1;
    return s;
}

bool rocke_gett_f64_arch_supported(const char* arch)
{
    return f64_atom(arch) != nullptr;
}

bool rocke_gett_f64_runtime_template(const rocke_gett_f64_problem_t* problem,
                                     rocke_gett_f64_problem_t* out)
{
    if(problem == nullptr || out == nullptr)
        return false;
    return runtime_template(*problem, out);
}

bool rocke_gett_f64_runtime_compatible(const rocke_gett_f64_spec_t* spec,
                                       const rocke_gett_f64_problem_t* problem,
                                       char* reason,
                                       size_t reason_cap)
{
    std::string why = "ok";
    bool ok = spec != nullptr && problem != nullptr && runtime_compatible(*spec, *problem, &why);
    if(spec == nullptr || problem == nullptr)
        why = "null spec or problem";
    if(reason != nullptr && reason_cap > 0)
        snprintf(reason, reason_cap, "%s", why.c_str());
    return ok;
}

bool rocke_gett_f64_runtime_args(const rocke_gett_f64_spec_t* spec,
                                 const rocke_gett_f64_problem_t* problem,
                                 int64_t out[ROCKE_GETT_F64_RUNTIME_ARGS],
                                 int* count)
{
    if(count != nullptr)
        *count = 0;
    std::string why;
    if(spec == nullptr || problem == nullptr || out == nullptr || count == nullptr
       || !runtime_compatible(*spec, *problem, &why))
        return false;
    const rocke_gett_f64_problem_t& p = *problem;
    const int64_t vals[ROCKE_GETT_F64_RUNTIME_ARGS] = {p.M,
                                                       p.N,
                                                       p.K,
                                                       p.L,
                                                       p.a_m.stride[0],
                                                       p.a_k.stride[0],
                                                       p.b_n.stride[0],
                                                       p.b_k.stride[0],
                                                       p.d_m.stride[0],
                                                       p.d_n.stride[0],
                                                       c_m_of(p).stride[0],
                                                       c_n_of(p).stride[0],
                                                       p.a_l.stride[0],
                                                       p.b_l.stride[0],
                                                       p.d_l.stride[0],
                                                       c_l_of(p).stride[0]};
    int n = 0;
    for(int i = 0; i < ROCKE_GETT_F64_RUNTIME_ARGS; ++i)
    {
        if(!spec->bilinear && is_c_arg(k_rt_args[i].name))
            continue;
        out[n++] = vals[i];
    }
    *count = n;
    return true;
}

bool rocke_gett_f64_is_valid_spec(const rocke_gett_f64_spec_t* spec,
                                  const char* arch,
                                  char* reason,
                                  size_t reason_cap)
{
    std::string why = "ok";
    bool ok = spec != nullptr && valid(*spec, arch, &why);
    if(spec == nullptr)
        why = "null spec";
    if(reason != nullptr && reason_cap > 0)
        snprintf(reason, reason_cap, "%s", why.c_str());
    return ok;
}

rocke_status_t
    rocke_gett_f64_kernel_name(const rocke_gett_f64_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    const rocke_gett_f64_problem_t& p = spec->problem;
    std::string parts[18];
    parts[0] = p.complex_ ? "c128" : "f64";
    std::string h;
    if(spec->runtime_sizes)
    {
        /* The template's layout class names the kernel; sizes are arguments. */
        rocke_gett_f64_problem_t tp;
        parts[1] = "rt";
        h = runtime_template(p, &tp) ? problem_hash(tp) : std::string("invalid");
    }
    else
    {
        parts[1] = "M" + std::to_string(p.M) + "N" + std::to_string(p.N) + "K" + std::to_string(p.K)
                   + "L" + std::to_string(p.L);
        h = problem_hash(p);
    }
    parts[2] = "t" + std::to_string(spec->tile_m) + "x" + std::to_string(spec->tile_n) + "x"
               + std::to_string(spec->tile_k);
    parts[3] = "w" + std::to_string(spec->warps_m) + "x" + std::to_string(spec->warps_n);
    parts[4] = "g" + std::to_string(spec->group_m);
    parts[5] = spec->xcds ? "x" + std::to_string(spec->xcds) : "";
    parts[6] = spec->lds_pad >= 0 ? "p" + std::to_string(spec->lds_pad) : "";
    parts[7] = (uses_swizzle(*spec) || p.complex_) ? "" : "nsw";
    parts[8] = (uses_swizzle(*spec) && spec->lds_swizzle_mode)
                   ? "z" + std::to_string(spec->lds_swizzle_mode)
                   : "";
    parts[9] = spec->waves_per_eu ? "e" + std::to_string(spec->waves_per_eu) : "";
    parts[10] = spec->lds_stages != 2 ? "s" + std::to_string(spec->lds_stages) : "";
    parts[11] = spec->prefetch_depth != 1 ? "d" + std::to_string(spec->prefetch_depth) : "";
    parts[12]
        = (p.complex_ && spec->complex_accs != 2) ? "a" + std::to_string(spec->complex_accs) : "";
    parts[13] = spec->k_stagger ? "k" + std::to_string(spec->k_stagger) : "";
    parts[14] = spec->k_block ? "b" + std::to_string(spec->k_block) : "";
    parts[15] = spec->split_k > 1 ? "sk" + std::to_string(spec->split_k) : "";
    parts[16] = "h" + h;
    const char* cparts[17];
    for(int i = 0; i < 17; ++i)
        cparts[i] = parts[i].c_str();
    const char* flag_names[] = {"bil", "nopf", "nofence", "u2", "cja", "cjb"};
    const int flag_on[] = {spec->bilinear ? 1 : 0,
                           spec->prefetch ? 0 : 1,
                           spec->sched_fence ? 0 : 1,
                           spec->unroll2 ? 1 : 0,
                           p.conj_a ? 1 : 0,
                           p.conj_b ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "rocke_gett_f64",
                                  cparts,
                                  17,
                                  flag_names,
                                  flag_on,
                                  6,
                                  out,
                                  out_cap,
                                  NULL);
}

void rocke_gett_f64_grid(const rocke_gett_f64_spec_t* spec, int grid[3])
{
    const rocke_gett_f64_problem_t& p = spec->problem;
    grid[0] = (int)(((p.M + spec->tile_m - 1) / spec->tile_m)
                    * ((p.N + spec->tile_n - 1) / spec->tile_n));
    grid[1] = (int)p.L;
    grid[2] = spec->split_k;
}

bool rocke_gett_f64_grid_for(const rocke_gett_f64_spec_t* spec,
                             const rocke_gett_f64_problem_t* problem,
                             int grid[3])
{
    if(spec == nullptr || problem == nullptr || problem->L > ROCKE_GETT_F64_MAX_GRID_Y)
        return false;
    const rocke_gett_f64_problem_t& p = *problem;
    grid[0] = (int)(((p.M + spec->tile_m - 1) / spec->tile_m)
                    * ((p.N + spec->tile_n - 1) / spec->tile_n));
    grid[1] = (int)p.L;
    grid[2] = spec->split_k;
    return true;
}

rocke_kernel_def_t*
    rocke_build_gett_f64(rocke_ir_builder_t* b, const rocke_gett_f64_spec_t* spec, const char* arch)
{
    char name[512];
    if(spec == nullptr || rocke_gett_f64_kernel_name(spec, name, sizeof name) != ROCKE_OK)
        snprintf(name, sizeof name, "rocke_gett_f64_invalid");
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        return NULL;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        std::string why;
        if(spec == nullptr || !valid(*spec, arch, &why))
            return (rocke_kernel_def_t*)rocke_i_set_err(b,
                                                        ROCKE_ERR_VALUE,
                                                        "invalid gett_f64 spec for %s: %s",
                                                        arch ? arch : "(null)",
                                                        why.c_str());
        rocke_gett_f64_spec_t s = *spec;
        if(s.runtime_sizes)
        {
            rocke_gett_f64_problem_t tp;
            runtime_template(spec->problem, &tp);
            s.problem = tp;
        }
        const rocke_gett_f64_problem_t& p = s.problem;
        bool cx = p.complex_;
        int es = elem_bytes(p);
        int wpe = cx ? 2 : 1;
        const rocke_mma_op_t* atom = f64_atom(arch);
        int nt = threads(s);
        int wave_m = s.tile_m / s.warps_m, wave_n = s.tile_n / s.warps_n;
        int am_n = wave_m / atom->m, an_n = wave_n / atom->n;
        int pad = resolved_pad(s);

        rocke_kernel_def_t* k = rocke_ir_builder_kernel(b);
        rocke_attr_set_int(b, &k->attrs, "max_workgroup_size", (int64_t)nt);
        if(s.waves_per_eu)
            rocke_attr_set_int(b, &k->attrs, "waves_per_eu", (int64_t)s.waves_per_eu);
        const rocke_type_t* gptr = rocke_ptr_type(b, rocke_f64(), "global");
        /* A and B never overlap D. C may be D itself (in-place update), so
         * neither carries noalias. 8-byte alignment: an f64 subtensor view need
         * not be 16-byte aligned (the buffer loads only need 4). */
        rocke_param_opts_t ab = {};
        ab.noalias = true;
        ab.noalias_set = true;
        ab.readonly = true;
        ab.readonly_set = true;
        ab.align = 8;
        ab.align_set = true;
        rocke_param_opts_t co = {};
        co.readonly = true;
        co.readonly_set = true;
        co.align = 8;
        co.align_set = true;
        rocke_param_opts_t dopt = {};
        dopt.align = 8;
        dopt.align_set = true;
        bool split = s.split_k > 1;
        V* A = rocke_b_param(b, "A", gptr, &ab);
        V* Bp = rocke_b_param(b, "B", gptr, &ab);
        V *C = nullptr, *D = nullptr, *W = nullptr;
        std::vector<V*> alpha, beta;
        if(split)
        {
            /* Partial sums only: the reduce kernel reads C and applies alpha / beta. */
            rocke_param_opts_t wo = {};
            wo.noalias = true;
            wo.noalias_set = true;
            wo.align = 8;
            wo.align_set = true;
            W = rocke_b_param(b, "W", gptr, &wo);
        }
        else
        {
            C = s.bilinear ? rocke_b_param(b, "C", gptr, &co) : nullptr;
            D = rocke_b_param(b, "D", gptr, &dopt);
            alpha.push_back(rocke_b_param(b, cx ? "alpha_re" : "alpha", rocke_f64(), NULL));
            if(cx)
                alpha.push_back(rocke_b_param(b, "alpha_im", rocke_f64(), NULL));
            if(s.bilinear)
            {
                beta.push_back(rocke_b_param(b, cx ? "beta_re" : "beta", rocke_f64(), NULL));
                if(cx)
                    beta.push_back(rocke_b_param(b, "beta_im", rocke_f64(), NULL));
            }
        }
        /* Size / stride arguments (runtime_sizes), by name. */
        V* rt[ROCKE_GETT_F64_RUNTIME_ARGS] = {};
        if(s.runtime_sizes)
        {
            for(int i = 0; i < ROCKE_GETT_F64_RUNTIME_ARGS; ++i)
            {
                if(!s.bilinear && is_c_arg(k_rt_args[i].name))
                    continue;
                rt[i] = rocke_b_param(
                    b, k_rt_args[i].name, k_rt_args[i].i64 ? rocke_i64() : rocke_i32(), NULL);
            }
        }
        geo_t geo = {};
        geo.rt = s.runtime_sizes;
        if(!geo.rt)
        {
            geo.M = iv_const(p.M);
            geo.N = iv_const(p.N);
            geo.K = iv_const(p.K);
            geo.a_m = {false, p.a_m, nullptr};
            geo.a_k = {false, p.a_k, nullptr};
            geo.b_n = {false, p.b_n, nullptr};
            geo.b_k = {false, p.b_k, nullptr};
            geo.d_m = {false, p.d_m, nullptr};
            geo.d_n = {false, p.d_n, nullptr};
            geo.c_m = {false, c_m_of(p), nullptr};
            geo.c_n = {false, c_n_of(p), nullptr};
        }
        else
        {
            geo.M = iv_val(rt[0]);
            geo.N = iv_val(rt[1]);
            geo.K = iv_val(rt[2]);
            geo.L = iv_val(rt[3]);
            auto one = [](const digits_t& d, V* stride) {
                gdig g = {};
                g.rt = true;
                g.d = d;
                g.stride = d.stride[0] == 1 ? nullptr : stride;
                return g;
            };
            geo.a_m = one(p.a_m, rt[4]);
            geo.a_k = one(p.a_k, rt[5]);
            geo.b_n = one(p.b_n, rt[6]);
            geo.b_k = one(p.b_k, rt[7]);
            geo.d_m = one(p.d_m, rt[8]);
            geo.d_n = one(p.d_n, rt[9]);
            geo.c_m = one(c_m_of(p), rt[10]);
            geo.c_n = one(c_n_of(p), rt[11]);
            geo.batch_a = rt[12];
            geo.batch_b = rt[13];
            geo.batch_d = rt[14];
            geo.batch_c = rt[15];
        }

        V* tid = rocke_b_thread_id_x(b);
        V* c64a = ci(b, 64);
        V* lane = rocke_b_mod(b, tid, c64a);
        V* c64b = ci(b, 64);
        V* wv = rocke_b_div(b, tid, c64b);
        V* wave = rocke_b_readfirstlane(b, wv);
        V* cwm = ci(b, s.warps_m);
        V* wm = rocke_b_mod(b, wave, cwm);
        V* cwm2 = ci(b, s.warps_m);
        V* wn = rocke_b_div(b, wave, cwm2);
        V* bidx = rocke_b_block_id_x(b);
        V *tm, *tn;
        if(geo.rt)
            raster_rt(b, s, bidx, geo, &tm, &tn);
        else
            raster(b, s, bidx, &tm, &tn);
        V* ctm = ci(b, s.tile_m);
        V* m0 = rocke_b_mul(b, tm, ctm);
        V* ctn = ci(b, s.tile_n);
        V* n0 = rocke_b_mul(b, tn, ctn);
        V* lb = rocke_b_block_id_y(b);

        auto at_batch = [&](V* ptr, const digits_t& d, V* rt_stride) -> V* {
            if(geo.rt)
            {
                V* wide = rocke_b_zext(b, lb, rocke_i64());
                V* off = rocke_b_mul(b, wide, rt_stride);
                V* ces = rocke_b_const_i64(b, es);
                V* bytes = rocke_b_mul(b, off, ces);
                return rocke_b_global_ptr_add(b, ptr, bytes);
            }
            if(digits_extent(d) == 1)
                return ptr;
            V* off = mixed64(b, lb, d, es);
            return rocke_b_global_ptr_add(b, ptr, off);
        };
        auto out_span = [&](const gdig& md, const gdig& nd, char which) -> V* {
            if(geo.rt)
            {
                V* c1 = ci(b, 1);
                V* em = rocke_b_sub(b, geo.M.v, c1);
                V* c1b = ci(b, 1);
                V* en = rocke_b_sub(b, geo.N.v, c1b);
                span_term terms[2] = {{iv_val(em), md.stride}, {iv_val(en), nd.stride}};
                return rt_span(b, terms, 2, es);
            }
            return ci(b, addr_span_bytes(p, which, s.tile_k));
        };

        V *rd = nullptr, *rc = nullptr, *rw = nullptr, *kz = nullptr;
        if(split)
        {
            kz = rocke_b_block_id_z(b);
            rw = workspace_rsrc(b, s, geo, W, kz, lb, es);
        }
        else
        {
            V* dptr = at_batch(D, p.d_l, geo.batch_d);
            V* dspan = out_span(geo.d_m, geo.d_n, 'D');
            rd = rocke_b_buffer_rsrc(b, dptr, dspan);
            if(C != nullptr)
            {
                V* cptr = at_batch(C, c_l_of(p), geo.batch_c);
                V* cspan = out_span(geo.c_m, geo.c_n, 'C');
                rc = rocke_b_buffer_rsrc(b, cptr, cspan);
            }
        }

        int ns = s.lds_stages;
        int a_shape[2] = {ns * s.tile_k, s.tile_m * wpe + pad};
        int b_shape[2] = {ns * s.tile_k, s.tile_n * wpe + pad};
        V* a_lds = rocke_b_smem_alloc(b, rocke_f64(), a_shape, 2, "a_lds");
        V* b_lds = rocke_b_smem_alloc(b, rocke_f64(), b_shape, 2, "b_lds");
        operand_t opa, opb;
        V* aptr = at_batch(A, p.a_l, geo.batch_a);
        opa.init(b, &s, 'A', aptr, geo.a_m, geo.a_k, s.tile_m, m0, geo.M, a_lds, geo);
        V* bptr = at_batch(Bp, p.b_l, geo.batch_b);
        opb.init(b, &s, 'B', bptr, geo.b_n, geo.b_k, s.tile_n, n0, geo.N, b_lds, geo);

        /* Fragment read coordinates from the atom's catalog lane maps. */
        V *a_mn, *a_k, *b_k, *b_mn;
        rocke_layout_map_coord(atom->a_layout, b, lane, 0, &a_mn, &a_k);
        rocke_layout_map_coord(atom->b_layout, b, lane, 0, &b_k, &b_mn);
        V* cwave_m = ci(b, wave_m);
        V* wmo = rocke_b_mul(b, wm, cwave_m);
        V* a_col0 = rocke_b_add(b, wmo, a_mn);
        V* cwave_n = ci(b, wave_n);
        V* wno = rocke_b_mul(b, wn, cwave_n);
        V* b_col0 = rocke_b_add(b, wno, b_mn);
        bool swz_on = uses_swizzle(s);
        auto frag_cols = [&](V* col0, int n_atoms, int side, V* k_row) {
            int mask = swizzle_mask(side);
            V* odd = nullptr;
            if(swz_on && !s.lds_swizzle_mode)
            {
                V* c1 = ci(b, 1);
                V* low = rocke_b_land(b, k_row, c1);
                V* c16 = ci(b, 16);
                odd = rocke_b_mul(b, low, c16);
            }
            frag_t out;
            out.k_row = k_row;
            for(int ks = 0; ks < s.tile_k / atom->k; ++ks)
            {
                std::vector<V*> cols;
                for(int i = 0; i < n_atoms; ++i)
                {
                    V* col = col0;
                    if(i)
                    {
                        V* c = ci(b, i * atom->m);
                        col = rocke_b_add(b, col0, c);
                    }
                    if(swz_on && s.lds_swizzle_mode)
                    {
                        V* kr = k_row;
                        if(ks)
                        {
                            V* c4 = ci(b, atom->k * ks);
                            kr = rocke_b_add(b, k_row, c4);
                        }
                        V* sz = swz(b, kr, mask, s.lds_swizzle_mode);
                        col = rocke_b_xor(b, col, sz);
                    }
                    else if(swz_on)
                    {
                        V* cf = ci(b, (8 * (ks & 7)) & mask);
                        V* flip = rocke_b_xor(b, odd, cf);
                        V* cm = ci(b, mask);
                        V* fm = rocke_b_land(b, flip, cm);
                        col = rocke_b_xor(b, col, fm);
                    }
                    cols.push_back(col);
                }
                out.cols.push_back(cols);
            }
            return out;
        };
        frag_t fa = frag_cols(a_col0, am_n, s.tile_m, a_k);
        frag_t fb = frag_cols(b_col0, an_n, s.tile_n, b_k);

        ival nk;
        if(geo.rt)
        {
            V* ctk1 = ci(b, s.tile_k - 1);
            V* kk = rocke_b_add(b, geo.K.v, ctk1);
            V* ctk = ci(b, s.tile_k);
            nk = iv_val(rocke_b_div(b, kk, ctk));
        }
        else
            nk = iv_const((p.K + s.tile_k - 1) / s.tile_k);
        if(s.k_stagger && (geo.rt || nk.c > 1))
        {
            /* start tile ((bid % 8) * n) % nk, then wrap: kt -> kt + start (- nk) */
            V* bid_s = rocke_b_block_id_x(b);
            V* c8 = ci(b, 8);
            V* sid = rocke_b_mod(b, bid_s, c8);
            V* cst = ci(b, s.k_stagger);
            V* sm = rocke_b_mul(b, sid, cst);
            V* cnk = get(b, nk);
            V* start = rocke_b_mod(b, sm, cnk);
            auto kmap = [b, start, nk](V* kt) {
                V* t = rocke_b_add(b, kt, start);
                V* cn1 = get(b, nk);
                V* wrap = rocke_b_cmp_lt(b, t, cn1);
                V* cn2 = get(b, nk);
                V* tw = rocke_b_sub(b, t, cn2);
                return rocke_b_select(b, wrap, t, tw);
            };
            opa.kmap = kmap;
            opa.nk = nk;
            opb.kmap = kmap;
            opb.nk = nk;
        }
        int n_acc = am_n * an_n * (cx ? s.complex_accs : 1);
        V* z = rocke_b_const_f64(b, 0.0);
        std::vector<V*> zs(atom->c_frag_len, z);
        V* zero = rocke_b_vec_pack(b, zs.data(), atom->c_frag_len, rocke_f64());

        /* split_k: this CTA's K tiles are [lo, hi), lo = z*nk/S, hi = (z+1)*nk/S. */
        V* k_lo = nullptr;
        ival count = nk;
        if(split)
        {
            V* cnk = get(b, nk);
            V* zn = rocke_b_mul(b, kz, cnk);
            V* cs = ci(b, s.split_k);
            k_lo = rocke_b_div(b, zn, cs);
            V* c1 = ci(b, 1);
            V* z1 = rocke_b_add(b, kz, c1);
            V* cnk2 = get(b, nk);
            V* z1n = rocke_b_mul(b, z1, cnk2);
            V* cs2 = ci(b, s.split_k);
            V* k_hi = rocke_b_div(b, z1n, cs2);
            count = iv_val(rocke_b_sub(b, k_hi, k_lo));
        }
        V* zero_i = ci(b, 0);
        V* first = k_lo != nullptr ? k_lo : zero_i;
        std::vector<V*> regs_a = opa.load(first);
        std::vector<V*> regs_b = opb.load(first);
        std::vector<V*> nx_a, nx_b;
        if(s.prefetch_depth == 2)
        {
            V* one_i;
            if(k_lo == nullptr)
                one_i = ci(b, 1);
            else
            {
                V* c1 = ci(b, 1);
                one_i = rocke_b_add(b, k_lo, c1);
            }
            nx_a = opa.load(one_i);
            nx_b = opb.load(one_i);
        }
        opa.store_lds(regs_a, zero_i);
        opb.store_lds(regs_b, zero_i);
        rocke_b_sync(b);

        loop_ctx ctx;
        ctx.b = b;
        ctx.spec = &s;
        ctx.atom = atom;
        ctx.opa = &opa;
        ctx.opb = &opb;
        ctx.a_lds = a_lds;
        ctx.b_lds = b_lds;
        ctx.fa = fa;
        ctx.fb = fb;
        ctx.nk = count;
        ctx.zero = zero;
        ctx.n_acc = n_acc;
        ctx.zero_i = zero_i;
        ctx.na = (int)nx_a.size();
        ctx.cx = cx;
        ctx.k_lo = k_lo;
        std::vector<V*> accs;
        if(s.unroll2)
            accs = k_loop_unroll2(ctx);
        else
        {
            std::vector<V*> carry = nx_a;
            carry.insert(carry.end(), nx_b.begin(), nx_b.end());
            accs = k_loop(ctx, carry);
        }
        if(split)
            epilogue_partial(b, s, atom, accs, lane, wm, wn, m0, n0, rw, geo);
        else
            epilogue(b, s, atom, accs, lane, wm, wn, m0, n0, rd, rc, alpha, beta, geo);
        rocke_b_ret(b);
        return rocke_ir_builder_kernel(b);
    });
}

int64_t rocke_gett_f64_workspace_bytes(const rocke_gett_f64_spec_t* spec,
                                       const rocke_gett_f64_problem_t* problem)
{
    if(spec == nullptr || spec->split_k <= 1)
        return 0;
    const rocke_gett_f64_problem_t* p = problem != nullptr ? problem : &spec->problem;
    return (int64_t)spec->split_k * p->L * p->M * p->N * elem_bytes(*p);
}

rocke_status_t
    rocke_gett_f64_reduce_name(const rocke_gett_f64_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = reduce_name(*spec);
    if(n.empty() || n.size() + 1 > out_cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

bool rocke_gett_f64_reduce_grid(const rocke_gett_f64_spec_t* spec,
                                const rocke_gett_f64_problem_t* problem,
                                int grid[3])
{
    if(spec == nullptr)
        return false;
    const rocke_gett_f64_problem_t* p = problem != nullptr ? problem : &spec->problem;
    if(p->L > ROCKE_GETT_F64_MAX_GRID_Y)
        return false;
    grid[0]
        = (int)((p->M * p->N + ROCKE_GETT_F64_REDUCE_THREADS - 1) / ROCKE_GETT_F64_REDUCE_THREADS);
    grid[1] = (int)p->L;
    grid[2] = 1;
    return true;
}

rocke_kernel_def_t* rocke_build_gett_f64_reduce(rocke_ir_builder_t* b,
                                                const rocke_gett_f64_spec_t* spec,
                                                const char* arch)
{
    std::string name = spec != nullptr ? reduce_name(*spec) : std::string();
    if(name.empty())
        name = "rocke_gett_f64_reduce_invalid";
    if(rocke_ir_builder_init(b, name.c_str()) != ROCKE_OK)
        return NULL;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        std::string why;
        if(spec == nullptr || !valid(*spec, arch, &why))
            return (rocke_kernel_def_t*)rocke_i_set_err(b,
                                                        ROCKE_ERR_VALUE,
                                                        "invalid gett_f64 spec for %s: %s",
                                                        arch ? arch : "(null)",
                                                        why.c_str());
        if(spec->split_k <= 1)
            return (rocke_kernel_def_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "build_gett_f64_reduce needs split_k > 1");
        rocke_gett_f64_spec_t s = *spec;
        if(s.runtime_sizes)
        {
            rocke_gett_f64_problem_t tp;
            runtime_template(spec->problem, &tp);
            s.problem = tp;
        }
        const rocke_gett_f64_problem_t& p = s.problem;
        bool cx = p.complex_;
        int es = elem_bytes(p);
        int words = cx ? 2 : 1;
        rocke_kernel_def_t* k = rocke_ir_builder_kernel(b);
        rocke_attr_set_int(b, &k->attrs, "max_workgroup_size", ROCKE_GETT_F64_REDUCE_THREADS);
        const rocke_type_t* gptr = rocke_ptr_type(b, rocke_f64(), "global");
        rocke_param_opts_t wo = {};
        wo.noalias = true;
        wo.noalias_set = true;
        wo.readonly = true;
        wo.readonly_set = true;
        wo.align = 8;
        wo.align_set = true;
        rocke_param_opts_t co = {};
        co.readonly = true;
        co.readonly_set = true;
        co.align = 8;
        co.align_set = true;
        rocke_param_opts_t dopt = {};
        dopt.align = 8;
        dopt.align_set = true;
        V* W = rocke_b_param(b, "W", gptr, &wo);
        V* C = s.bilinear ? rocke_b_param(b, "C", gptr, &co) : nullptr;
        V* D = rocke_b_param(b, "D", gptr, &dopt);
        std::vector<V*> alpha, beta;
        alpha.push_back(rocke_b_param(b, cx ? "alpha_re" : "alpha", rocke_f64(), NULL));
        if(cx)
            alpha.push_back(rocke_b_param(b, "alpha_im", rocke_f64(), NULL));
        if(s.bilinear)
        {
            beta.push_back(rocke_b_param(b, cx ? "beta_re" : "beta", rocke_f64(), NULL));
            if(cx)
                beta.push_back(rocke_b_param(b, "beta_im", rocke_f64(), NULL));
        }
        V* rt[ROCKE_GETT_F64_RUNTIME_ARGS] = {};
        if(s.runtime_sizes)
        {
            for(int i = 0; i < ROCKE_GETT_F64_RUNTIME_ARGS; ++i)
            {
                if(!s.bilinear && is_c_arg(k_rt_args[i].name))
                    continue;
                rt[i] = rocke_b_param(
                    b, k_rt_args[i].name, k_rt_args[i].i64 ? rocke_i64() : rocke_i32(), NULL);
            }
        }
        geo_t geo = {};
        geo.rt = s.runtime_sizes;
        if(!geo.rt)
        {
            geo.M = iv_const(p.M);
            geo.N = iv_const(p.N);
            geo.K = iv_const(p.K);
            geo.d_m = {false, p.d_m, nullptr};
            geo.d_n = {false, p.d_n, nullptr};
            geo.c_m = {false, c_m_of(p), nullptr};
            geo.c_n = {false, c_n_of(p), nullptr};
        }
        else
        {
            geo.M = iv_val(rt[0]);
            geo.N = iv_val(rt[1]);
            geo.K = iv_val(rt[2]);
            geo.L = iv_val(rt[3]);
            auto one = [](const digits_t& d, V* stride) {
                gdig g = {};
                g.rt = true;
                g.d = d;
                g.stride = d.stride[0] == 1 ? nullptr : stride;
                return g;
            };
            geo.d_m = one(p.d_m, rt[8]);
            geo.d_n = one(p.d_n, rt[9]);
            geo.c_m = one(c_m_of(p), rt[10]);
            geo.c_n = one(c_n_of(p), rt[11]);
            geo.batch_d = rt[14];
            geo.batch_c = rt[15];
        }

        V* tid = rocke_b_thread_id_x(b);
        V* bx = rocke_b_block_id_x(b);
        V* cthr = ci(b, ROCKE_GETT_F64_REDUCE_THREADS);
        V* bxt = rocke_b_mul(b, bx, cthr);
        V* idx = rocke_b_add(b, bxt, tid);
        V* lb = rocke_b_block_id_y(b);
        ival total;
        V *slice64 = nullptr, *span;
        if(geo.rt)
        {
            V* t = rocke_b_mul(b, geo.M.v, geo.N.v);
            total = iv_val(t);
            V* t64 = rocke_b_zext(b, t, rocke_i64());
            V* ces = rocke_b_const_i64(b, es);
            slice64 = rocke_b_mul(b, t64, ces);
            V* ces32 = ci(b, es);
            span = rocke_b_mul(b, t, ces32);
        }
        else
        {
            total = iv_const(p.M * p.N);
            span = ci(b, p.M * p.N * es);
        }
        V* ctot = get(b, total);
        V* ok = rocke_b_cmp_lt(b, idx, ctot);
        V* c0 = ci(b, 0);
        V* idx_safe = rocke_b_select(b, ok, idx, c0);
        V* cN1 = get(b, geo.N);
        V* n = rocke_b_mod(b, idx_safe, cN1);
        V* cN2 = get(b, geo.N);
        V* m = rocke_b_div(b, idx_safe, cN2);
        V* ces = ci(b, es);
        V* woff0 = rocke_b_mul(b, idx, ces);
        V* cmask = ci(b, k_mask_offset);
        V* woff = rocke_b_select(b, ok, woff0, cmask);
        const rocke_type_t* vty = rocke_vector_type(b, rocke_f64(), words);
        std::vector<V*> parts;
        for(int zi = 0; zi < s.split_k; ++zi)
        {
            V* base;
            if(geo.rt)
            {
                V* cz = ci(b, zi);
                V* zL = rocke_b_mul(b, cz, geo.L.v);
                V* zl = rocke_b_add(b, zL, lb);
                V* zl64 = rocke_b_zext(b, zl, rocke_i64());
                V* off = rocke_b_mul(b, zl64, slice64);
                base = rocke_b_global_ptr_add(b, W, off);
            }
            else
            {
                V* czl = ci(b, (int64_t)zi * p.L);
                V* zl = rocke_b_add(b, czl, lb);
                V* zl64 = rocke_b_zext(b, zl, rocke_i64());
                V* cmn = rocke_b_const_i64(b, p.M * p.N * es);
                V* off = rocke_b_mul(b, zl64, cmn);
                base = rocke_b_global_ptr_add(b, W, off);
            }
            V* rsrc = rocke_b_buffer_rsrc(b, base, span);
            V* cz0 = ci(b, 0);
            V* raw = rocke_b_buffer_load_vN(b, rsrc, woff, cz0, rocke_i32(), 2 * words);
            parts.push_back(rocke_b_vec_bitcast(b, raw, vty));
        }
        V* acc = parts[0];
        for(size_t zi = 1; zi < parts.size(); ++zi)
            acc = rocke_b_fadd(b, acc, parts[zi]);

        auto at_batch = [&](V* ptr, const digits_t& d, V* rt_stride) -> V* {
            if(geo.rt)
            {
                V* wide = rocke_b_zext(b, lb, rocke_i64());
                V* off = rocke_b_mul(b, wide, rt_stride);
                V* ces64 = rocke_b_const_i64(b, es);
                V* bytes = rocke_b_mul(b, off, ces64);
                return rocke_b_global_ptr_add(b, ptr, bytes);
            }
            if(digits_extent(d) == 1)
                return ptr;
            V* off = mixed64(b, lb, d, es);
            return rocke_b_global_ptr_add(b, ptr, off);
        };
        auto out_span = [&](const gdig& md, const gdig& nd, char which) -> V* {
            if(geo.rt)
            {
                V* c1 = ci(b, 1);
                V* em = rocke_b_sub(b, geo.M.v, c1);
                V* c1b = ci(b, 1);
                V* en = rocke_b_sub(b, geo.N.v, c1b);
                span_term terms[2] = {{iv_val(em), md.stride}, {iv_val(en), nd.stride}};
                return rt_span(b, terms, 2, es);
            }
            return ci(b, addr_span_bytes(p, which, s.tile_k));
        };
        V* dptr = at_batch(D, p.d_l, geo.batch_d);
        V* dspan = out_span(geo.d_m, geo.d_n, 'D');
        V* rd = rocke_b_buffer_rsrc(b, dptr, dspan);
        V* c_val[2] = {nullptr, nullptr};
        bool have_c = false;
        if(C != nullptr)
        {
            V* cptr = at_batch(C, c_l_of(p), geo.batch_c);
            V* cspan = out_span(geo.c_m, geo.c_n, 'C');
            V* rc = rocke_b_buffer_rsrc(b, cptr, cspan);
            V* z0 = rocke_b_const_f64(b, 0.0);
            V* eq = rocke_b_fcmp(b, "oeq", beta[0], z0);
            V* use_c = rocke_b_lnot(b, eq);
            if(cx)
            {
                V* z1 = rocke_b_const_f64(b, 0.0);
                V* eq1 = rocke_b_fcmp(b, "oeq", beta[1], z1);
                V* ne1 = rocke_b_lnot(b, eq1);
                use_c = rocke_b_lor(b, use_c, ne1);
            }
            V* cmo = gmixed(b, m, geo.c_m);
            V* cno = gmixed(b, n, geo.c_n);
            V* cs = rocke_b_add(b, cmo, cno);
            V* ces2 = ci(b, es);
            V* coff = rocke_b_mul(b, cs, ces2);
            V* okc = rocke_b_land(b, ok, use_c);
            V* cmask2 = ci(b, k_mask_offset);
            coff = rocke_b_select(b, okc, coff, cmask2);
            V* cz = ci(b, 0);
            V* craw = rocke_b_buffer_load_vN(b, rc, coff, cz, rocke_i32(), 2 * words);
            V* cv = rocke_b_vec_bitcast(b, craw, vty);
            for(int j = 0; j < words; ++j)
                c_val[j] = rocke_b_vec_extract(b, cv, j);
            have_c = true;
        }
        scalars_t sc = make_scalars(b, alpha, beta, cx);
        V* accv[2] = {nullptr, nullptr};
        for(int j = 0; j < words; ++j)
            accv[j] = rocke_b_vec_extract(b, acc, j);
        V* out[2] = {nullptr, nullptr};
        combine(b, sc, accv, have_c ? c_val : nullptr, out);
        V* dmo = gmixed(b, m, geo.d_m);
        V* dno = gmixed(b, n, geo.d_n);
        V* ds = rocke_b_add(b, dmo, dno);
        V* ces3 = ci(b, es);
        V* doff = rocke_b_mul(b, ds, ces3);
        V* cmask3 = ci(b, k_mask_offset);
        doff = rocke_b_select(b, ok, doff, cmask3);
        V* packed = rocke_b_vec_pack(b, out, words, rocke_f64());
        const rocke_type_t* svty = rocke_vector_type(b, rocke_f32(), 2 * words);
        V* vec = rocke_b_vec_bitcast(b, packed, svty);
        V* cz2 = ci(b, 0);
        rocke_b_buffer_store_vN_f32(b, rd, doff, cz2, vec, 2 * words);
        rocke_b_ret(b);
        return rocke_ir_builder_kernel(b);
    });
}

} /* extern "C" */
