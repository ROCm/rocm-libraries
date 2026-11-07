// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_sched.py: the stage tables
 * of the attention backward q step (StageTable) and of the split dQ kernel's
 * key step (DqStageTable), and the scheduling hints they emit. See
 * rocke/helper_rocke.instances.common._attention_bwd_sched.h for the mapping.
 *
 * Every Python raise is one ckc::Error throw with the same message (str(exc));
 * the C entry points catch it. Python's // and % floor (fdiv / fmod below),
 * and ceil_div is -(-a // b). Each method recomputes what Python recomputes,
 * in Python's order, so a raise surfaces at the same point.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_sched.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"
#include "rocke/ir_internal.h"

/* ====================================================================== *
 * constants
 * ====================================================================== */

extern "C" {
const char* const ROCKE_ATTN_BWD_SCHEDS[ROCKE_ATTN_BWD_N_SCHEDS]
    = {"none", "iglp0", "iglp1", "fences_only", "stage_table", "stage_table_atomics"};
const char* const ROCKE_ATTN_BWD_STAGE_TABLE_SCHEDS[ROCKE_ATTN_BWD_N_STAGE_TABLE_SCHEDS]
    = {"stage_table", "stage_table_atomics"};
const char* const ROCKE_ATTN_BWD_FENCED_SCHEDS[ROCKE_ATTN_BWD_N_FENCED_SCHEDS]
    = {"fences_only", "stage_table", "stage_table_atomics"};
const char* const ROCKE_ATTN_BWD_IGLP_SCHEDS[ROCKE_ATTN_BWD_N_IGLP_SCHEDS] = {"iglp0", "iglp1"};
const char* const ROCKE_ATTN_BWD_PIPELINES[ROCKE_ATTN_BWD_N_PIPELINES]
    = {"sync", "ring", "merged", "ck"};
const char* const ROCKE_ATTN_BWD_BUILT_PIPELINES[ROCKE_ATTN_BWD_N_BUILT_PIPELINES]
    = {"sync", "ring", "merged"};
const char* const ROCKE_ATTN_BWD_COUNTED_LATE_DEFERRED
    = "sched='stage_table_atomics' with loop_sync='counted' needs "
      "counted_issue='early': the late issue would load the next step after the "
      "deferred dQ atomics of stage A (the counted waits keep every load of the "
      "step ahead of its atomics)";
const char* const ROCKE_ATTN_BWD_STAGES[ROCKE_ATTN_BWD_N_STAGES] = {"A", "B", "C", "D"};
const char* const ROCKE_ATTN_BWD_SPLIT_STAGES[ROCKE_ATTN_BWD_N_SPLIT_STAGES] = {"A", "B", "C"};
const char* const ROCKE_ATTN_BWD_SPLIT_DEFERRED
    = "sched='stage_table_atomics' needs dq_mode='atomic': the split-dQ main "
      "kernel issues no dQ atomics to defer (its stage table is "
      "sched='stage_table')";
const char* const ROCKE_ATTN_BWD_DQ_SCHEDS[ROCKE_ATTN_BWD_N_DQ_SCHEDS]
    = {"none", "iglp0", "iglp1", "fences_only", "stage_table", "stage_table_setprio"};
const char* const ROCKE_ATTN_BWD_DQ_FENCED_SCHEDS[ROCKE_ATTN_BWD_N_DQ_FENCED_SCHEDS]
    = {"fences_only", "stage_table", "stage_table_setprio"};
const char* const ROCKE_ATTN_BWD_DQ_STAGES[ROCKE_ATTN_BWD_N_DQ_STAGES] = {"A", "B", "C"};
const char* const ROCKE_ATTN_BWD_GEMMS[ROCKE_ATTN_BWD_N_GEMMS] = {"g0", "g1", "g2", "g3", "g4"};
const char* const ROCKE_ATTN_BWD_SCHED_G4_SLICINGS[ROCKE_ATTN_BWD_SCHED_N_G4_SLICINGS]
    = {"step", "slice", "slice_db", "slice_db_g3"};
} // extern "C"

namespace rocke_attn_bwd
{
namespace
{

typedef rocke_attn_bwd_sched_pair_t Pair;
typedef std::vector<Pair> Pairs;
typedef std::vector<std::string> Strs;
typedef rocke_attn_bwd_stage_table_t Table;
typedef rocke_attn_bwd_dq_stage_table_t DqTable;
/* StageCounts: the C ABI struct borrows its lead / spans / fixed arrays, so
 * the module works on this owning twin (no cap on the tuples, as Python). */
struct Counts
{
    const char* stage;
    int mfma;
    int vmem_read;
    int vmem_write;
    int ds_read;
    int ds_write;
    Pairs lead;
    Pairs spans;
    Pairs fixed;
};

const int MFMA = ROCKE_SCHED_MFMA;
const int VMEM_READ = ROCKE_SCHED_VMEM_READ;
const int VMEM_WRITE = ROCKE_SCHED_VMEM_WRITE;
const int DS_READ = ROCKE_SCHED_DS_READ;
const int DS_WRITE = ROCKE_SCHED_DS_WRITE;
const int ELEM_BYTES = 2;

const char* const TRANSPOSE_SOURCES[] = {"xt_lds", "tr_read", "lds_plain", "wmma_lds"};
const char* const DQ_TRANSPOSE_SOURCES[] = {"tr_read", "lds_plain"};
const int DQ_KV_PADS[] = {0, 8, 16, 32};

/* ------------------------------------------------------------ helpers */

bool eq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

template <size_t N>
bool in(const char* const (&set)[N], const char* s)
{
    for(const char* x : set)
        if(eq(x, s))
            return true;
    return false;
}

bool in_n(const char* const* set, int n, const char* s)
{
    for(int i = 0; i < n; ++i)
        if(eq(set[i], s))
            return true;
    return false;
}

bool in_scheds(const char* s)
{
    return in_n(ROCKE_ATTN_BWD_SCHEDS, ROCKE_ATTN_BWD_N_SCHEDS, s);
}
bool in_table(const char* s)
{
    return in_n(ROCKE_ATTN_BWD_STAGE_TABLE_SCHEDS, ROCKE_ATTN_BWD_N_STAGE_TABLE_SCHEDS, s);
}
bool in_fenced(const char* s)
{
    return in_n(ROCKE_ATTN_BWD_FENCED_SCHEDS, ROCKE_ATTN_BWD_N_FENCED_SCHEDS, s);
}
bool in_iglp(const char* s)
{
    return in_n(ROCKE_ATTN_BWD_IGLP_SCHEDS, ROCKE_ATTN_BWD_N_IGLP_SCHEDS, s);
}

std::string repr(const char* s)
{
    return s == NULL ? std::string("None") : py_repr(s);
}

template <size_t N>
std::string tuple_repr(const char* const (&set)[N])
{
    return py_tuple(Strs(set, set + N));
}

std::string tuple_repr_n(const char* const* set, int n)
{
    return py_tuple(Strs(set, set + n));
}

std::string grid_repr(const int g[2])
{
    return "(" + std::to_string(g[0]) + ", " + std::to_string(g[1]) + ")";
}

std::string hex_repr(long long m)
{
    char buf[40];
    if(m < 0)
        snprintf(buf, sizeof(buf), "-0x%llx", (unsigned long long)(-m));
    else
        snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)m);
    return buf;
}

long long fdiv(long long a, long long b)
{
    if(b == 0)
        throw ckc::ValueError("integer division or modulo by zero");
    long long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

long long fmod_(long long a, long long b)
{
    return a - fdiv(a, b) * b;
}

long long cdiv(long long a, long long b)
{
    return -fdiv(-a, b);
}

int I(long long v)
{
    return (int)v;
}

bool starts_with(const std::string& s, const char* p)
{
    return s.rfind(p, 0) == 0;
}

bool is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::string strip(const std::string& s)
{
    size_t a = 0, e = s.size();
    while(a < e && is_space(s[a]))
        ++a;
    while(e > a && is_space(s[e - 1]))
        --e;
    return s.substr(a, e - a);
}

/* int(text) of a str (base 10, sign, surrounding whitespace, '_' between
 * digits). */
long long py_int(const std::string& text)
{
    std::string s = strip(text);
    size_t i = 0;
    bool neg = false;
    if(i < s.size() && (s[i] == '+' || s[i] == '-'))
        neg = s[i++] == '-';
    long long v = 0;
    bool digit = false, prev_us = false, ok = i < s.size();
    for(; i < s.size() && ok; ++i)
    {
        char c = s[i];
        if(c >= '0' && c <= '9')
        {
            v = v * 10 + (c - '0');
            digit = true;
            prev_us = false;
        }
        else if(c == '_' && digit && !prev_us)
            prev_us = true;
        else
            ok = false;
    }
    if(!ok || !digit || prev_us)
        throw ckc::ValueError("invalid literal for int() with base 10: " + py_repr(text));
    return neg ? -v : v;
}

/* An insertion-ordered {mask: count} dict. */
struct ODict
{
    Pairs items;
    int* find(int m)
    {
        for(Pair& p : items)
            if(p.mask == m)
                return &p.count;
        return NULL;
    }
    bool has(int m) const
    {
        for(const Pair& p : items)
            if(p.mask == m)
                return true;
        return false;
    }
    int get(int m, int dflt) const
    {
        for(const Pair& p : items)
            if(p.mask == m)
                return p.count;
        return dflt;
    }
    void set(int m, int v)
    {
        int* c = find(m);
        if(c != NULL)
            *c = v;
        else
            items.push_back(Pair{m, v});
    }
};

ODict make_dict(const Pair* p, int n)
{
    ODict d;
    for(int i = 0; i < n; ++i)
        d.set(p[i].mask, p[i].count);
    return d;
}

/* ------------------------------------------------------------- atoms */

const rocke_arch_target_t* target_of(const char* arch)
{
    const char* a = arch != NULL ? arch : "";
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(a);
    if(t == NULL)
    {
        char reason[1024];
        /* already str(KeyError(...)): the repr of Python's message */
        rocke_set_unknown_arch_reason(reason, sizeof(reason), a);
        throw ckc::KeyError(reason);
    }
    return t;
}

const char* norm_dt(const char* dtype)
{
    if(eq(dtype, "fp16") || eq(dtype, "f16"))
        return "fp16";
    if(eq(dtype, "bf16"))
        return "bf16";
    return NULL;
}

std::vector<const rocke_mma_op_t*> mma_ops(const char* arch, const char* dtype)
{
    const rocke_arch_target_t* tgt = target_of(arch);
    const char* dt = norm_dt(dtype);
    if(dt == NULL)
        throw ckc::ValueError("unsupported dtype " + repr(dtype));
    bool has_mfma = false;
    for(int i = 0; i < tgt->mma.num_ops; ++i)
        if(strcmp(tgt->mma.ops[i].family, "mma") == 0)
            has_mfma = true;
    const char* family = has_mfma ? "mma" : "wmma";
    int n = rocke_mma_catalog_enumerate(&tgt->mma, family, dt, dt, "fp32", 16, 16, NULL, 0, NULL);
    std::vector<const rocke_mma_op_t*> all((size_t)(n > 0 ? n : 0));
    if(n > 0)
        rocke_mma_catalog_enumerate(&tgt->mma, family, dt, dt, "fp32", 16, 16, all.data(), n, NULL);
    std::vector<const rocke_mma_op_t*> ops;
    for(const rocke_mma_op_t* op : all)
        if(op->wave_size == tgt->wave_size && op->a_frag_len > 0)
            ops.push_back(op);
    if(ops.empty())
        throw ckc::ValueError(std::string("no 16x16 ") + dt + " MMA atom on " + arch);
    std::stable_sort(ops.begin(), ops.end(), [](const rocke_mma_op_t* a, const rocke_mma_op_t* b) {
        return a->k < b->k;
    });
    return ops;
}

std::vector<int> native_atom_ks(const char* arch, const char* dtype)
{
    std::vector<int> ks;
    for(const rocke_mma_op_t* op : mma_ops(arch, dtype))
        ks.push_back(op->k);
    return ks;
}

int instruction_k(const char* arch, int atom_k, const char* dtype)
{
    std::vector<int> ks = native_atom_ks(arch, dtype);
    for(int k : ks)
        if(k == atom_k)
            return atom_k;
    int best = 0;
    bool any = false;
    for(int k : ks)
        if(fmod_(atom_k, k) == 0)
        {
            if(!any || k > best)
                best = k;
            any = true;
        }
    if(!any)
        throw ckc::ValueError("no instruction can implement a K=" + std::to_string(atom_k)
                              + " step on " + arch);
    return best;
}

/* ----------------------------------------------------- interleave_groups */

bool is_pre(int m)
{
    return m == VMEM_READ;
}

Pairs interleave_groups(int mfma, const Pairs& mem, const ODict& lead, const ODict& spans)
{
    Pairs raw;
    if(mfma <= 0)
    {
        for(const Pair& p : mem)
            if(p.count > 0)
                raw.push_back(p);
    }
    else
    {
        ODict total;
        for(const Pair& p : mem)
            total.set(p.mask, p.count);
        for(const Pair& p : lead.items)
            if(!(0 <= p.count && p.count <= total.get(p.mask, 0)))
                throw ckc::ValueError("lead " + std::to_string(p.count) + " of mask "
                                      + hex_repr(p.mask) + " exceeds its count");
        for(const Pair& p : spans.items)
            if(!(0 < p.count && p.count <= mfma))
                throw ckc::ValueError("span " + std::to_string(p.count) + " of mask "
                                      + hex_repr(p.mask) + " outside 1.." + std::to_string(mfma));
        std::vector<int> order;
        for(const Pair& p : mem)
            order.push_back(p.mask);
        std::stable_sort(order.begin(), order.end(), [](int a, int b) {
            return (int)!is_pre(a) < (int)!is_pre(b);
        });
        for(int m : order)
            if(lead.get(m, 0))
                raw.push_back(Pair{m, lead.get(m, 0)});
        Pairs pre, post;
        for(const Pair& p : mem)
        {
            Pair r{p.mask, p.count - lead.get(p.mask, 0)};
            if(is_pre(r.mask) && r.count > 0)
                pre.push_back(r);
            else if(!is_pre(r.mask) && r.count > 0)
                post.push_back(r);
        }
        auto share = [&](int m, int c, int i) -> int {
            int span = spans.get(m, mfma);
            long long j = (long long)i - (mfma - span);
            if(j < 0)
                return 0;
            return I(fdiv((j + 1) * c, span) - fdiv(j * c, span));
        };
        for(int i = 0; i < mfma; ++i)
        {
            for(const Pair& p : pre)
            {
                int s = share(p.mask, p.count, i);
                if(s)
                    raw.push_back(Pair{p.mask, s});
            }
            raw.push_back(Pair{MFMA, 1});
            for(const Pair& p : post)
            {
                int s = share(p.mask, p.count, i);
                if(s)
                    raw.push_back(Pair{p.mask, s});
            }
        }
    }
    Pairs out;
    for(const Pair& p : raw)
    {
        if(!out.empty() && out.back().mask == p.mask)
            out.back().count += p.count;
        else
            out.push_back(p);
    }
    return out;
}

/* ---------------------------------------------------------- StageCounts */

Counts counts(const char* stage, int mfma)
{
    Counts c = Counts();
    c.stage = stage;
    c.mfma = mfma;
    return c;
}

Pairs pairs_of(const Pair* p, int n, const char* what)
{
    if(n < 0 || (n > 0 && p == NULL))
        throw ckc::ValueError(std::string("StageCounts.") + what + ": bad (pointer, count)");
    return Pairs(p, p + n);
}

/* The scalar fields of the C ABI struct (memory / by_mask read only those). */
Counts counts_scalars(const rocke_attn_bwd_stage_counts_t& s)
{
    Counts c = counts(s.stage, s.mfma);
    c.vmem_read = s.vmem_read;
    c.vmem_write = s.vmem_write;
    c.ds_read = s.ds_read;
    c.ds_write = s.ds_write;
    return c;
}

/* The C ABI struct (borrowed arrays) to the owning twin. */
Counts counts_in(const rocke_attn_bwd_stage_counts_t& s)
{
    Counts c = counts_scalars(s);
    c.lead = pairs_of(s.lead, s.n_lead, "lead");
    c.spans = pairs_of(s.spans, s.n_spans, "spans");
    c.fixed = pairs_of(s.fixed, s.n_fixed, "fixed");
    return c;
}

const Pair* store_pairs(rocke_attn_bwd_stage_pairs_store_t* st, const Pairs& src, int* n)
{
    *n = (int)src.size();
    if(src.empty())
        return NULL;
    if(st == NULL || st->n < 0 || src.size() > (size_t)(ROCKE_ATTN_BWD_STAGE_STORE_PAIRS - st->n))
        throw ckc::ValueError("stage pair store too small");
    Pair* out = st->pairs + st->n;
    for(size_t i = 0; i < src.size(); ++i)
        out[i] = src[i];
    st->n += (int)src.size();
    return out;
}

/* The owning twin to the C ABI struct, its arrays in the caller's store. */
rocke_attn_bwd_stage_counts_t counts_out(const Counts& c, rocke_attn_bwd_stage_pairs_store_t* st)
{
    rocke_attn_bwd_stage_counts_t s;
    memset(&s, 0, sizeof(s));
    s.stage = c.stage;
    s.mfma = c.mfma;
    s.vmem_read = c.vmem_read;
    s.vmem_write = c.vmem_write;
    s.ds_read = c.ds_read;
    s.ds_write = c.ds_write;
    s.lead = store_pairs(st, c.lead, &s.n_lead);
    s.spans = store_pairs(st, c.spans, &s.n_spans);
    s.fixed = store_pairs(st, c.fixed, &s.n_fixed);
    return s;
}

Pairs memory(const Counts& s)
{
    return Pairs{{VMEM_READ, s.vmem_read},
                 {DS_READ, s.ds_read},
                 {DS_WRITE, s.ds_write},
                 {VMEM_WRITE, s.vmem_write}};
}

Pairs by_mask(const Counts& s)
{
    return Pairs{{MFMA, s.mfma},
                 {VMEM_READ, s.vmem_read},
                 {VMEM_WRITE, s.vmem_write},
                 {DS_READ, s.ds_read},
                 {DS_WRITE, s.ds_write}};
}

Pairs census(const Counts& s)
{
    ODict out;
    out.items = by_mask(s);
    for(const Pair& f : s.fixed)
    {
        int* c = out.find(f.mask);
        if(c == NULL)
            throw ckc::KeyError(std::to_string(f.mask));
        *c += f.count;
    }
    return out.items;
}

Pairs stage_groups(const Counts& s)
{
    return interleave_groups(s.mfma,
                             memory(s),
                             make_dict(s.lead.data(), (int)s.lead.size()),
                             make_dict(s.spans.data(), (int)s.spans.size()));
}

/* ----------------------------------------------------------- StageTable */

std::vector<const char*> stage_names(const Table& t)
{
    if(eq(t.dq_mode, "split") && !eq(t.pipeline, "ck"))
        return std::vector<const char*>(ROCKE_ATTN_BWD_SPLIT_STAGES,
                                        ROCKE_ATTN_BWD_SPLIT_STAGES
                                            + ROCKE_ATTN_BWD_N_SPLIT_STAGES);
    return std::vector<const char*>(ROCKE_ATTN_BWD_STAGES,
                                    ROCKE_ATTN_BWD_STAGES + ROCKE_ATTN_BWD_N_STAGES);
}

bool counted(const Table& t)
{
    return eq(t.loop_sync, "counted");
}

void check_staging(const Table& t)
{
    if(!(t.ring_depth == 1 || t.ring_depth == 2 || t.ring_depth == 3))
        throw ckc::ValueError("ring_depth must be 1, 2 or 3");
    if(!(eq(t.barrier_merge, "off") || eq(t.barrier_merge, "on")))
        throw ckc::ValueError("barrier_merge must be 'off' or 'on'");
    if(!(eq(t.ring_issue, "early") || eq(t.ring_issue, "late")))
        throw ckc::ValueError("ring_issue must be 'early' or 'late'");
    if(!(eq(t.loop_sync, "full") || eq(t.loop_sync, "lds_only") || eq(t.loop_sync, "counted")))
        throw ckc::ValueError("loop_sync must be 'full', 'lds_only' or 'counted'");
    if(!(eq(t.counted_issue, "early") || eq(t.counted_issue, "late")))
        throw ckc::ValueError("counted_issue must be 'early' or 'late'");
    if(eq(t.pipeline, "ck"))
        return;
    if(eq(t.dq_mode, "split"))
    {
        if(eq(t.sched, "stage_table_atomics"))
            throw ckc::ValueError(ROCKE_ATTN_BWD_SPLIT_DEFERRED);
        if(eq(t.barrier_merge, "on") || counted(t))
            throw ckc::ValueError("the split-dQ main kernel runs on the synchronous body or a "
                                  "prefetch ring (barrier_merge='off', loop_sync != 'counted')");
    }
    if(counted(t))
    {
        if(!eq(t.pipeline, "sync"))
            throw ckc::ValueError("loop_sync='counted' runs on the synchronous body");
        if(eq(t.sched, "stage_table_atomics") && eq(t.counted_issue, "late"))
            throw ckc::ValueError(ROCKE_ATTN_BWD_COUNTED_LATE_DEFERRED);
    }
    bool merged = eq(t.barrier_merge, "on");
    bool staged = !(eq(t.global_path, "vgpr") && t.ring_depth == 1);
    const char* want = merged ? "merged" : (staged ? "ring" : "sync");
    if(!eq(t.pipeline, want))
        throw ckc::ValueError("pipeline=" + repr(t.pipeline) + " does not model global_path="
                              + repr(t.global_path) + ", ring_depth=" + std::to_string(t.ring_depth)
                              + ", barrier_merge=" + repr(t.barrier_merge) + " (that body is "
                              + repr(want) + ")");
    if(merged && !(t.ring_depth == 2 && eq(t.global_path, "vgpr")))
        throw ckc::ValueError("barrier_merge='on' needs ring_depth=2, global_path='vgpr'");
    if(eq(t.pipeline, "ring") && eq(t.global_path, "dma"))
        if(t.stage_vec != 8 || t.head_pack)
            throw ckc::ValueError("the DMA ring needs stage_vec = 8 and no head packing");
    if(!eq(t.pipeline, "sync") && !eq(t.stats_load, "at_use"))
        throw ckc::ValueError("a ring stages the stats: stats_load must be 'at_use'");
}

void post_init(Table& t)
{
    const int w = t.waves;
    struct
    {
        int* g;
        int natural[2];
    } grids[3] = {{t.grid_g4, {1, w}}, {t.grid_g02, {1, w}}, {t.grid_g13, {w, 1}}};
    for(auto& gr : grids)
        if((gr.g[0] == 1 && gr.g[1] == 0) || (gr.g[0] == 0 && gr.g[1] == 1))
        {
            gr.g[0] = gr.natural[0];
            gr.g[1] = gr.natural[1];
        }
    long long w02 = (long long)t.grid_g02[0] * t.grid_g02[1];
    if(w02 < 1 || fmod_(w, w02))
        throw ckc::ValueError("warp grid G0 " + grid_repr(t.grid_g02)
                              + " does not divide the waves");
    if((long long)t.grid_g13[0] * t.grid_g13[1] != w)
        throw ckc::ValueError("warp grid G1 " + grid_repr(t.grid_g13) + " does not hold the waves");
    if(in_table(t.sched) && w02 != w)
        throw ckc::ValueError("a stage table needs a G0 grid of every wave");
    const int g4r = t.grid_g4[0], g4c = t.grid_g4[1];
    const long long d = t.head_size, m0 = t.k_m0, n0 = t.k_n0, k4 = t.k_k4;
    if(!in_scheds(t.sched))
        throw ckc::ValueError("unknown sched " + repr(t.sched) + "; expected one of "
                              + tuple_repr_n(ROCKE_ATTN_BWD_SCHEDS, ROCKE_ATTN_BWD_N_SCHEDS));
    if(t.setprio && !in_table(t.sched))
        throw ckc::ValueError("setprio is legal only inside a stage table");
    if(!in_n(ROCKE_ATTN_BWD_PIPELINES, ROCKE_ATTN_BWD_N_PIPELINES, t.pipeline))
        throw ckc::ValueError("pipeline must be one of "
                              + tuple_repr_n(ROCKE_ATTN_BWD_PIPELINES, ROCKE_ATTN_BWD_N_PIPELINES));
    check_staging(t);
    if(!(eq(t.stats_load, "at_use") || eq(t.stats_load, "step_top")))
        throw ckc::ValueError("stats_load must be 'at_use' or 'step_top'");
    if(!(eq(t.dq_issue, "block") || eq(t.dq_issue, "fragment") || eq(t.dq_issue, "fragment_regs")))
        throw ckc::ValueError("dq_issue must be 'block', 'fragment' or 'fragment_regs'");
    const int g0r = t.grid_g02[0], g0c = t.grid_g02[1];
    const int g1r = t.grid_g13[0], g1c = t.grid_g13[1];
    if(!in(TRANSPOSE_SOURCES, t.transpose_source))
        throw ckc::ValueError("unknown transpose_source " + repr(t.transpose_source));
    if(!(eq(t.kt_source, "reg") || eq(t.kt_source, "lds")))
        throw ckc::ValueError("kt_source must be 'reg' or 'lds'");
    if(!(eq(t.kv_residency, "reg") || eq(t.kv_residency, "lds")))
        throw ckc::ValueError("kv_residency must be 'reg' or 'lds'");
    if(eq(t.kt_source, "reg") && !eq(t.kv_residency, "reg"))
        throw ckc::ValueError("kt_source='reg' requires kv_residency='reg'");
    if(!(eq(t.global_path, "vgpr") || eq(t.global_path, "dma")))
        throw ckc::ValueError("global_path must be 'vgpr' or 'dma'");
    if(!(eq(t.pt_route, "relabel") || eq(t.pt_route, "lds")))
        throw ckc::ValueError("pt_route must be 'relabel' or 'lds'");
    if(!(eq(t.dq_mode, "atomic") || eq(t.dq_mode, "split")))
        throw ckc::ValueError("dq_mode must be 'atomic' or 'split'");
    if(!(t.stage_vec == 1 || t.stage_vec == 8))
        throw ckc::ValueError("stage_vec must be 8 or 1");
    if(!in_n(ROCKE_ATTN_BWD_SCHED_G4_SLICINGS, ROCKE_ATTN_BWD_SCHED_N_G4_SLICINGS, t.g4_slicing))
        throw ckc::ValueError(
            "g4_slicing must be one of "
            + tuple_repr_n(ROCKE_ATTN_BWD_SCHED_G4_SLICINGS, ROCKE_ATTN_BWD_SCHED_N_G4_SLICINGS));
    if((long long)g4r * g4c != w)
        throw ckc::ValueError("warp grid G4 " + grid_repr(t.grid_g4) + " does not hold "
                              + std::to_string(w) + " waves");
    const bool cnt = in_table(t.sched);
    const std::string W = std::to_string(w);
    const std::string D = std::to_string(d), M0 = std::to_string(m0), N0 = std::to_string(n0);
    /* the tuple is built first (every % runs), then checked in order */
    const std::pair<bool, std::string> checks[] = {
        {!cnt || fmod_(n0, 16LL * w) == 0, "kN0 " + N0 + " % (16 * waves " + W + ") != 0"},
        {fmod_(m0, 16) == 0, "kM0 " + M0 + " % 16 != 0"},
        {fmod_(d, 16LL * g4c) == 0, "D " + D + " % (16 * g4c " + std::to_string(g4c) + ") != 0"},
        {fmod_(m0, 16LL * g4r) == 0,
         "kM0 " + M0 + " % (16 * g4r " + std::to_string(g4r) + ") != 0"},
        {fmod_(n0, k4) == 0, "kN0 " + N0 + " % kK4 " + std::to_string(k4) + " != 0"},
        {fmod_(k4, t.atom_g4) == 0,
         "kK4 " + std::to_string(k4) + " % atom_g4 K " + std::to_string(t.atom_g4) + " != 0"},
        {fmod_(d, t.atom_g02) == 0,
         "D " + D + " % atom_g02 K " + std::to_string(t.atom_g02) + " != 0"},
        {fmod_(m0, t.atom_g13) == 0,
         "kM0 " + M0 + " % atom_g13 K " + std::to_string(t.atom_g13) + " != 0"},
        {t.atom_g13 != 32 || m0 >= 32, "atom_g13 K = 32 needs kM0 >= 32"},
        {fmod_(m0, 16LL * g0r) == 0,
         "kM0 " + M0 + " % (16 * g0r " + std::to_string(g0r) + ") != 0"},
        {fmod_(n0, 16LL * g0c) == 0,
         "kN0 " + N0 + " % (16 * g0c " + std::to_string(g0c) + ") != 0"},
        {fmod_(n0, 16LL * g1r) == 0,
         "kN0 " + N0 + " % (16 * g1r " + std::to_string(g1r) + ") != 0"},
        {fmod_(d, 16LL * g1c) == 0, "D " + D + " % (16 * g1c " + std::to_string(g1c) + ") != 0"},
    };
    for(const auto& c : checks)
        if(!c.first)
            throw ckc::ValueError("illegal backward stage-table geometry: " + c.second);
    for(int k : {t.atom_g02, t.atom_g13, t.atom_g4})
        instruction_k(t.arch, k, t.dtype);
}

int wave_size(const Table& t)
{
    return target_of(t.arch)->wave_size;
}

const rocke_mma_op_t* op_of(const Table& t, int k)
{
    for(const rocke_mma_op_t* op : mma_ops(t.arch, t.dtype))
        if(op->k == k)
            return op;
    throw ckc::ValueError("no native 16x16x" + std::to_string(k) + " atom on " + t.arch);
}

struct InstrK
{
    int g[5];
    int at(const char* gemm) const
    {
        for(int i = 0; i < ROCKE_ATTN_BWD_N_GEMMS; ++i)
            if(eq(gemm, ROCKE_ATTN_BWD_GEMMS[i]))
                return g[i];
        throw ckc::KeyError(repr(gemm));
    }
};

InstrK instr_k(const Table& t)
{
    int k02 = instruction_k(t.arch, t.atom_g02, t.dtype);
    int k13 = instruction_k(t.arch, t.atom_g13, t.dtype);
    int k4 = instruction_k(t.arch, t.atom_g4, t.dtype);
    return InstrK{{k02, k13, k02, k13, k4}};
}

int frag_len(const Table& t, const char* gemm)
{
    return op_of(t, instr_k(t).at(gemm))->a_frag_len;
}

int c_frag_len(const Table& t)
{
    return op_of(t, 16)->c_frag_len;
}

rocke_attn_bwd_gemm_counts_t mfma(const Table& t)
{
    const long long w = t.waves, d = t.head_size, m0 = t.k_m0, n0 = t.k_n0;
    const int g4r = t.grid_g4[0], g4c = t.grid_g4[1];
    InstrK k = instr_k(t);
    long long w02 = (long long)t.grid_g02[0] * t.grid_g02[1];
    long long g0 = fdiv(fdiv(m0, 16) * fdiv(n0, 16), w02) * fdiv(d, k.g[0]);
    long long g1 = fdiv(fdiv(n0, 16) * fdiv(d, 16), w) * fdiv(m0, k.g[1]);
    long long g4 = 0;
    if(eq(t.dq_mode, "atomic"))
        g4 = fdiv(m0, 16LL * g4r) * fdiv(d, 16LL * g4c) * fdiv(n0, k.g[4]);
    return rocke_attn_bwd_gemm_counts_t{I(g0), I(g1), I(g0), I(g1), I(g4)};
}

int dq_atomics_per_lane(const Table& t)
{
    if(!eq(t.dq_mode, "atomic"))
        return 0;
    return I(fdiv((long long)t.k_m0 * t.head_size, (long long)t.waves * wave_size(t)));
}

int reads_per_frag(const Table& t, const char* gemm, const char* source)
{
    int n = frag_len(t, gemm);
    if(eq(source, "tr_read"))
        return I(cdiv(n, 4));
    if(eq(source, "lds_plain"))
        return n;
    return I(cdiv((long long)n * ELEM_BYTES, 16));
}

const char* tsrc(const Table& t)
{
    if(eq(t.transpose_source, "xt_lds") || eq(t.transpose_source, "wmma_lds"))
        return "contig";
    return t.transpose_source;
}

int prefetch_loads(const Table& t)
{
    long long vec = eq(t.global_path, "dma") ? 8 : t.stage_vec;
    long long threads = (long long)t.waves * wave_size(t);
    long long tile = cdiv((long long)t.k_m0 * t.head_size, threads * vec);
    long long stats = cdiv(t.k_m0, wave_size(t));
    return I(2 * tile + 2 * stats);
}

int next_tile_lds_writes(const Table& t)
{
    if(eq(t.global_path, "dma"))
        return 0;
    long long threads = (long long)t.waves * wave_size(t);
    long long tile = cdiv((long long)t.k_m0 * t.head_size, threads * t.stage_vec);
    long long n = 2 * tile;
    if(eq(t.transpose_source, "xt_lds"))
    {
        rocke_attn_bwd_xt_writer_t w;
        rocke_attn_bwd_err_t e;
        if(rocke_attn_bwd_default_xt_writer(&w, t.k_m0, t.head_size, (int)threads, 2, 16, &e)
           != ROCKE_OK)
            ckc::raise_status(e.code, e.msg);
        n += 2LL * rocke_attn_bwd_xt_writer_stores_per_thread(&w);
    }
    n += 2 * cdiv(t.k_m0, threads);
    return I(n);
}

int ds_tile_writes(const Table& t)
{
    long long tiles = fdiv(t.k_m0, 16) * fdiv(t.k_n0, 16LL * t.waves);
    int per_tile = wave_size(t) == 64 ? 1 : c_frag_len(t);
    return I(tiles * per_tile);
}

int kv_frag_reads(const Table& t, const char* gemm)
{
    if(!eq(t.kv_residency, "lds"))
        return 0;
    int k = instr_k(t).at(gemm);
    long long frags = fdiv(t.k_n0, 16LL * t.waves) * fdiv(t.head_size, k);
    return I(frags * reads_per_frag(t, gemm, "contig"));
}

int acc_rmw(const Table& t)
{
    if(!t.acc_in_lds)
        return 0;
    long long tiles = fdiv(t.k_n0, 16LL * t.waves) * fdiv(t.head_size, 16);
    return I(tiles * c_frag_len(t));
}

/* _swizzle_pairs(value).get(buf, "none") */
std::string swizzle_of(const char* value, const char* buf)
{
    std::string text = value != NULL ? value : "";
    std::string found = "none";
    size_t start = 0;
    while(true)
    {
        size_t comma = text.find(',', start);
        std::string item
            = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        std::string s = strip(item);
        size_t eqpos = s.find('=');
        if(eqpos != std::string::npos && strip(s.substr(0, eqpos)) == buf)
            found = strip(s.substr(eqpos + 1));
        if(comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return found;
}

int align(const Table& t, const char* buf)
{
    std::string sw = swizzle_of(t.lds_swizzle, buf);
    if(starts_with(sw, "pad"))
    {
        long long pitch = 16 + fdiv(py_int(sw.substr(3)), ELEM_BYTES);
        int a = 8;
        while(a > 1 && fmod_(pitch, a))
            a /= 2;
        return a;
    }
    return 8;
}

int contig_reads(const Table& t, const char* gemm, const char* buf)
{
    int n = frag_len(t, gemm);
    return I(cdiv(n, std::min(8, align(t, buf))));
}

int transposed_reads(const Table& t, const char* gemm)
{
    int n = frag_len(t, gemm);
    if(eq(t.transpose_source, "tr_read"))
        return I(fdiv(n, 4));
    if(eq(t.transpose_source, "xt_lds"))
        return I(cdiv((long long)n * ELEM_BYTES, 16));
    return n;
}

struct Frags
{
    long long fm, fn, ks;
};

Frags frags(const Table& t, const char* gemm)
{
    const long long d = t.head_size, m0 = t.k_m0, n0 = t.k_n0;
    InstrK k = instr_k(t);
    if(eq(gemm, "g0") || eq(gemm, "g2"))
    {
        int r = t.grid_g02[0], c = t.grid_g02[1];
        return Frags{fdiv(m0, 16LL * r), fdiv(n0, 16LL * c), fdiv(d, k.g[0])};
    }
    if(eq(gemm, "g1") || eq(gemm, "g3"))
    {
        int r = t.grid_g13[0], c = t.grid_g13[1];
        return Frags{fdiv(n0, 16LL * r), fdiv(d, 16LL * c), fdiv(m0, k.g[1])};
    }
    int r = t.grid_g4[0], c = t.grid_g4[1];
    return Frags{fdiv(m0, 16LL * r), fdiv(d, 16LL * c), fdiv(n0, k.g[4])};
}

struct Three
{
    long long r, w, lead;
};

Three g13_reads(const Table& t, bool b_hoisted)
{
    Frags f = frags(t, "g1");
    long long tr = transposed_reads(t, "g1");
    long long a = eq(t.pt_route, "lds") ? tr : 0;
    long long bb = b_hoisted ? 0 : tr;
    if(t.acc_in_lds)
    {
        long long cf = c_frag_len(t);
        return Three{f.fm * f.fn * (cf + f.ks * (a + bb)), f.fm * f.fn * cf, cf + a + bb};
    }
    return Three{f.ks * (f.fm * a + f.fn * bb), 0, f.fm * a + f.fn * bb};
}

std::pair<long long, long long> g4_reads(const Table& t, bool kt_hoisted)
{
    Frags f = frags(t, "g4");
    long long a = contig_reads(t, "g4", "ds");
    long long bb = 0;
    if(eq(t.kt_source, "lds") && !kt_hoisted)
        bb = transposed_reads(t, "g4");
    const char* issue = t.acc_in_lds ? "fragment" : t.dq_issue;
    if(eq(issue, "block"))
        return {f.ks * (f.fm * a + f.fn * bb), f.fm * a + f.fn * bb};
    if(eq(issue, "fragment"))
        return {f.fm * f.fn * f.ks * (a + bb), a + bb};
    return {f.fm * f.ks * a + f.fm * f.fn * f.ks * bb, f.fm * f.ks * a + bb};
}

int stats_loads(const Table& t)
{
    Frags f = frags(t, "g0");
    return I(2 * f.fm * c_frag_len(t));
}

/* ----- the rings ----- */

int threads_of(const Table& t)
{
    return I((long long)t.waves * wave_size(t));
}

int chunk_loads(const Table& t)
{
    return t.stage_vec == 8 ? 1 : 8;
}

int chunk_stores(const Table& t, const char* buf)
{
    return I(cdiv(8, std::min(8, align(t, buf))));
}

std::pair<int, bool> ring_reg_chunks(const Table& t)
{
    long long total = fdiv((long long)t.k_m0 * t.head_size, 8);
    long long threads = threads_of(t);
    bool joint = !t.head_pack && total < threads && fmod_(total, wave_size(t)) == 0;
    return {joint ? 1 : I(2 * cdiv(total, threads)), joint};
}

rocke_attn_bwd_ring_prefetch_t ring_prefetch(const Table& t)
{
    rocke_attn_bwd_ring_prefetch_t pf;
    memset(&pf, 0, sizeof(pf));
    if(!(eq(t.pipeline, "ring") || eq(t.pipeline, "merged")))
        return pf;
    pf.present = true;
    bool table = in_table(t.sched);
    if(eq(t.pipeline, "ring") && eq(t.global_path, "dma"))
    {
        int ws = wave_size(t);
        rocke_attn_bwd_dma_batch_t batch;
        rocke_attn_bwd_err_t e;
        if(rocke_attn_bwd_dma_batch_init(&batch, t.k_m0, t.head_size, t.waves, ws, 2, &e)
           != ROCKE_OK)
            ckc::raise_status(e.code, e.msg);
        int lo = 0, hi = 0;
        for(int w = 0; w < t.waves; ++w)
        {
            int n = rocke_attn_bwd_dma_batch_of_wave(&batch, w, NULL, 0);
            lo = w == 0 ? n : std::min(lo, n);
            hi = w == 0 ? n : std::max(hi, n);
        }
        pf.issue = "dma";
        pf.where = "top";
        pf.vmem_read = 0;
        pf.ds_write = 0;
        pf.has_dma = true;
        pf.dma_per_wave[0] = lo;
        pf.dma_per_wave[1] = hi;
        pf.dma_x4 = 2 * rocke_attn_bwd_dma_batch_pieces(&batch);
        pf.dma_x1 = 2;
        return pf;
    }
    const int stats_stores = 2;
    if(eq(t.pipeline, "ring"))
    {
        std::pair<int, bool> cj = ring_reg_chunks(t);
        int chunks = cj.first;
        long long stores;
        if(cj.second)
            stores = 1;
        else
        {
            long long half = fdiv(chunks, 2);
            int q = chunk_stores(t, "q");
            int dd = chunk_stores(t, "do");
            stores = half * (q + dd);
        }
        pf.issue = "loads";
        pf.where = table ? "A" : "top";
        pf.vmem_read = I((long long)chunks * chunk_loads(t) + 2);
        pf.ds_write = I(stores + stats_stores);
        return pf;
    }
    long long per_tensor = cdiv(fdiv((long long)t.k_m0 * t.head_size, 8), threads_of(t));
    bool late = eq(t.ring_issue, "late");
    pf.issue = "merged";
    pf.where = late ? "barrier" : (table ? "A" : "top");
    pf.vmem_read = I(2 * per_tensor * chunk_loads(t) + 2);
    int q = chunk_stores(t, "q");
    int dd = chunk_stores(t, "do");
    pf.ds_write = I(per_tensor * (q + dd) + stats_stores);
    return pf;
}

rocke_attn_bwd_counted_terms_t counted_terms(const Table& t)
{
    rocke_attn_bwd_counted_terms_t c;
    memset(&c, 0, sizeof(c));
    if(!counted(t))
        return c;
    long long chunks = cdiv(fdiv((long long)t.k_m0 * t.head_size, 8), threads_of(t));
    c.present = true;
    c.issue = t.counted_issue;
    c.loads = I(stats_loads(t) + 2 * chunks * chunk_loads(t));
    int q = chunk_stores(t, "q");
    int dd = chunk_stores(t, "do");
    c.stores = I(chunks * (q + dd));
    return c;
}

struct RingTerms
{
    int a_vmem_read;
    Pairs a_fixed;
    Pairs d_fixed;
};

RingTerms ring_terms(const Table& t)
{
    RingTerms out{0, Pairs(), Pairs()};
    rocke_attn_bwd_counted_terms_t cnt = counted_terms(t);
    if(cnt.present && eq(cnt.issue, "late"))
        out.a_fixed = Pairs{{DS_WRITE, cnt.stores}};
    rocke_attn_bwd_ring_prefetch_t pf = ring_prefetch(t);
    if(!pf.present)
        return out;
    if(eq(pf.where, "A"))
        out.a_vmem_read = pf.vmem_read;
    else if(eq(pf.issue, "merged") && eq(pf.where, "top"))
        out.a_fixed = Pairs{{VMEM_READ, pf.vmem_read}};
    Pairs d_fixed;
    if(eq(pf.issue, "merged") && eq(pf.where, "barrier"))
        d_fixed.push_back(Pair{VMEM_READ, pf.vmem_read});
    if(pf.ds_write)
        d_fixed.push_back(Pair{DS_WRITE, pf.ds_write});
    out.d_fixed = d_fixed;
    return out;
}

std::vector<Counts> sync_stages(const Table& t, bool prev_atomics)
{
    rocke_attn_bwd_gemm_counts_t mf = mfma(t);
    const bool hoist = in_fenced(t.sched);
    const bool deferred = eq(t.sched, "stage_table_atomics");
    const bool slot_stats = eq(t.pipeline, "ring") || eq(t.pipeline, "merged");
    const char* k_buf = (eq(t.kv_residency, "lds") || eq(t.kt_source, "lds")) ? "k_res" : "k_stage";
    Frags f0 = frags(t, "g0");
    long long q_r = f0.fm * contig_reads(t, "g0", "q");
    long long do_r = f0.fm * contig_reads(t, "g2", "do");
    long long k_r = 0, v_r = 0;
    if(eq(t.kv_residency, "lds"))
    {
        k_r = f0.fn * contig_reads(t, "g0", k_buf);
        v_r = f0.fn * contig_reads(t, "g2", "v_res");
    }
    long long stats = eq(t.stats_load, "at_use") ? stats_loads(t) : 0;
    if(counted(t))
        stats = 0;
    long long vm_stats = slot_stats ? 0 : stats;
    long long ds_stats = slot_stats ? stats : 0;
    RingTerms ring = ring_terms(t);
    Frags f1 = frags(t, "g1");
    long long bt = f1.fn * f1.ks * transposed_reads(t, "g1");
    long long cf = c_frag_len(t);
    long long tile_w = f0.fm * f0.fn * cf;
    long long ds_w = tile_w * (eq(t.pt_route, "lds") ? 2 : (mf.g4 ? 1 : 0));
    Three g13 = g13_reads(t, hoist);
    Frags f4 = frags(t, "g4");
    long long kt_ahead = 0;
    if(hoist && eq(t.kt_source, "lds") && mf.g4)
        kt_ahead = f4.fn * f4.ks * transposed_reads(t, "g4");
    std::pair<long long, long long> g4r(0, 0);
    if(mf.g4)
        g4r = g4_reads(t, hoist);
    int atoms = dq_atomics_per_lane(t);
    int a_w = deferred && prev_atomics ? atoms : 0;

    Counts a = counts("A", mf.g0);
    a.vmem_read = I((hoist ? vm_stats : 0) + ring.a_vmem_read);
    a.vmem_write = a_w;
    a.ds_read = I(f0.ks * (q_r + k_r) + (hoist ? bt : 0) + (hoist ? ds_stats : 0));
    a.lead = Pairs{{VMEM_READ, I(hoist ? vm_stats : 0)}, {DS_READ, I(q_r + k_r)}};
    a.fixed = ring.a_fixed;

    Counts b = counts("B", mf.g2 + mf.g1);
    b.vmem_read = I(hoist ? 0 : vm_stats);
    b.ds_read = I(f0.ks * (do_r + v_r) + g13.r + (hoist ? bt : 0) + (hoist ? 0 : ds_stats));
    b.ds_write = I(ds_w + g13.w);
    b.lead = Pairs{{DS_READ, I(do_r + v_r)}};
    if(mf.g1)
        b.spans = Pairs{{DS_WRITE, mf.g1}};

    Counts c = counts("C", mf.g3);
    c.ds_read = I(g13.r + kt_ahead);
    c.ds_write = I(g13.w);
    c.lead = Pairs{{DS_READ, I(g13.lead)}};
    std::vector<const char*> names = stage_names(t);
    if(names.size() == ROCKE_ATTN_BWD_N_SPLIT_STAGES)
    {
        c.fixed = ring.d_fixed;
        return std::vector<Counts>{a, b, c};
    }
    Counts d = counts("D", mf.g4);
    d.vmem_write = deferred ? 0 : atoms;
    d.ds_read = I(g4r.first);
    d.lead = Pairs{{DS_READ, I(g4r.second)}};
    if(mf.g4 && !deferred)
        d.spans = Pairs{{VMEM_WRITE, I(f4.fm * f4.fn)}};
    d.fixed = ring.d_fixed;
    return std::vector<Counts>{a, b, c, d};
}

std::vector<Counts> ck_stages(const Table& t, bool prefetch, bool prev_atomics)
{
    const long long w = t.waves, d = t.head_size, m0 = t.k_m0, n0 = t.k_n0, kk4 = t.k_k4;
    const int g4r = t.grid_g4[0], g4c = t.grid_g4[1];
    InstrK k = instr_k(t);
    rocke_attn_bwd_gemm_counts_t mf = mfma(t);
    const char* ts = tsrc(t);
    const bool g4_on = eq(t.dq_mode, "atomic");

    long long bt_frags = fdiv(d, 16) * fdiv(m0, k.g[1]);
    long long bt_reads = bt_frags * reads_per_frag(t, "g1", ts);
    long long next_a_reads = fdiv(m0, 16) * fdiv(d, k.g[0]) * reads_per_frag(t, "g0", "contig");
    long long stats_reads = fdiv(m0, 16);
    long long slice_reads
        = fdiv(m0, 16LL * g4r) * fdiv(kk4, k.g[4]) * reads_per_frag(t, "g4", "contig");
    long long n_slices = g4_on ? fdiv(n0, kk4) : 0;

    long long a_vmem = prefetch ? prefetch_loads(t) : 0;
    long long a_vmem_w
        = (eq(t.sched, "stage_table_atomics") && prev_atomics) ? dq_atomics_per_lane(t) : 0;
    long long a_ds_r = bt_reads;
    long long a_ds_w = 0;
    long long b_ds_r = bt_reads;
    long long b_ds_w = prefetch ? next_tile_lds_writes(t) : 0;
    if(eq(t.pt_route, "lds"))
    {
        a_ds_w += ds_tile_writes(t);
        long long pt_frags = fdiv(n0, 16 * w) * fdiv(m0, k.g[1]);
        b_ds_r += pt_frags * reads_per_frag(t, "g1", "contig");
    }
    long long c_ds_w = (g4_on || eq(t.pt_route, "lds")) ? ds_tile_writes(t) : 0;
    long long c_slices = (n_slices && eq(t.g4_slicing, "slice_db_g3")) ? 1 : 0;
    long long c_ds_r = c_slices * slice_reads + (prefetch ? next_a_reads + stats_reads : 0);
    if(eq(t.pt_route, "lds"))
        c_ds_r += fdiv(n0, 16 * w) * fdiv(m0, k.g[3]) * reads_per_frag(t, "g3", "contig");
    a_ds_r += kv_frag_reads(t, "g0");
    b_ds_r += kv_frag_reads(t, "g2");
    b_ds_r += acc_rmw(t);
    b_ds_w += acc_rmw(t);
    c_ds_r += acc_rmw(t);
    c_ds_w += acc_rmw(t);
    long long d_ds_r = (n_slices - c_slices) * slice_reads;
    if(g4_on && eq(t.kt_source, "lds"))
    {
        long long kt_frags = fdiv(d, 16LL * g4c) * fdiv(n0, k.g[4]);
        d_ds_r += kt_frags * reads_per_frag(t, "g4", ts);
    }
    if(prefetch)
        d_ds_r += next_a_reads + stats_reads;
    Counts a = counts("A", mf.g0);
    a.vmem_read = I(a_vmem);
    a.vmem_write = I(a_vmem_w);
    a.ds_read = I(a_ds_r);
    a.ds_write = I(a_ds_w);
    Counts b = counts("B", mf.g1 + mf.g2);
    b.ds_read = I(b_ds_r);
    b.ds_write = I(b_ds_w);
    Counts c = counts("C", mf.g3);
    c.ds_read = I(c_ds_r);
    c.ds_write = I(c_ds_w);
    Counts dd = counts("D", mf.g4);
    dd.ds_read = I(d_ds_r);
    return std::vector<Counts>{a, b, c, dd};
}

std::vector<Counts> stages(const Table& t, bool prefetch = true, bool prev_atomics = true)
{
    if(eq(t.pipeline, "ck"))
        return ck_stages(t, prefetch, prev_atomics);
    return sync_stages(t, prev_atomics);
}

size_t name_index(const std::vector<const char*>& names, const char* name)
{
    for(size_t i = 0; i < names.size(); ++i)
        if(eq(names[i], name))
            return i;
    throw ckc::ValueError("tuple.index(x): x not in tuple");
}

Counts stage(const Table& t, const char* name, bool prefetch, bool prev_atomics)
{
    std::vector<Counts> st = stages(t, prefetch, prev_atomics);
    return st[name_index(stage_names(t), name)];
}

Pairs groups(const Table& t, const char* name, bool prefetch = true, bool prev_atomics = true)
{
    if(!in_table(t.sched))
        return Pairs();
    return stage_groups(stage(t, name, prefetch, prev_atomics));
}

bool prefetch_region(const Table& t, int* vmem_read)
{
    rocke_attn_bwd_ring_prefetch_t pf = ring_prefetch(t);
    if(pf.present && eq(pf.issue, "loads") && eq(pf.where, "top"))
    {
        *vmem_read = pf.vmem_read;
        return true;
    }
    return false;
}

bool prefetch_fence(const Table& t)
{
    int v;
    return prefetch_region(t, &v);
}

std::vector<int> ring_waits(const Table& t)
{
    rocke_attn_bwd_ring_prefetch_t pf = ring_prefetch(t);
    if(!(pf.present && eq(pf.issue, "dma")))
        return std::vector<int>();
    rocke_attn_bwd_ring_waits_t w;
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_dma_wait_counts(
           t.ring_depth, pf.dma_per_wave[0], dq_atomics_per_lane(t), &w, &e)
       != ROCKE_OK)
        ckc::raise_status(e.code, e.msg);
    return std::vector<int>{t.ring_depth == 1 ? w.step_top : w.step_end};
}

int vm_after_prefetch(const Table& t, bool prev_atomics)
{
    long long total = 0;
    for(const Counts& st : stages(t, true, prev_atomics))
    {
        ODict c;
        c.items = census(st);
        total += c.get(VMEM_READ, 0) + c.get(VMEM_WRITE, 0);
    }
    return I(total);
}

std::vector<int> counted_waits(const Table& t)
{
    if(!counted(t))
        return std::vector<int>();
    int n = dq_atomics_per_lane(t);
    if(eq(t.counted_issue, "late"))
        return std::vector<int>{n, n, n};
    return std::vector<int>{n, eq(t.sched, "stage_table_atomics") ? n : 0, n};
}

std::vector<int> step_waits(const Table& t)
{
    std::vector<int> out = ring_waits(t);
    std::vector<int> c = counted_waits(t);
    out.insert(out.end(), c.begin(), c.end());
    return out;
}

typedef rocke_attn_bwd_hint_t Hint;

Hint hint(rocke_attn_bwd_hint_kind_t k, int a, int b = 0)
{
    return Hint{k, a, b};
}

std::vector<Hint> hint_sequence(const Table& t)
{
    std::vector<Hint> out;
    rocke_attn_bwd_counted_terms_t cnt = counted_terms(t);
    if(prefetch_fence(t))
        out.push_back(hint(ROCKE_ATTN_BWD_HINT_FENCE, 0));
    if(cnt.present && eq(cnt.issue, "early"))
    {
        out.push_back(hint(ROCKE_ATTN_BWD_HINT_FENCE, 0));
        out.push_back(hint(ROCKE_ATTN_BWD_HINT_FENCE, 0));
    }
    if(in_iglp(t.sched))
        out.push_back(hint(ROCKE_ATTN_BWD_HINT_IGLP, t.sched[strlen(t.sched) - 1] - '0'));
    std::vector<const char*> names = stage_names(t);
    const char* last = names.back();
    for(const Counts& st : stages(t))
    {
        if(eq(st.stage, "D") && cnt.present && eq(cnt.issue, "late") && mfma(t).g4)
        {
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_FENCE, 0));
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_FENCE, 0));
        }
        if(t.setprio)
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_PRIO, 1));
        if(in_table(t.sched))
            for(const Pair& g : stage_groups(st))
                out.push_back(hint(ROCKE_ATTN_BWD_HINT_GROUP, g.mask, g.count));
        if(t.setprio)
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_PRIO, 0));
        if(in_fenced(t.sched) && !eq(st.stage, last))
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_FENCE, 0));
    }
    return out;
}

typedef rocke_attn_bwd_sched_region_t Region;

Region region(const char* name, const Pairs& ops, const Pairs* lead)
{
    Region r;
    memset(&r, 0, sizeof(r));
    r.name = name;
    if(ops.size() > 5)
        throw ckc::ValueError("more than 5 classes in a region");
    for(size_t i = 0; i < ops.size(); ++i)
        r.ops[i] = ops[i];
    r.n_ops = (int)ops.size();
    if(lead != NULL)
    {
        r.has_lead = true;
        if(lead->size() > ROCKE_ATTN_BWD_REGION_LEAD_MAX)
            throw ckc::ValueError("more than " + std::to_string(ROCKE_ATTN_BWD_REGION_LEAD_MAX)
                                  + " leading classes in a region");
        for(size_t i = 0; i < lead->size(); ++i)
            r.lead[i] = (*lead)[i];
        r.n_lead = (int)lead->size();
    }
    return r;
}

Region stage_region(const Counts& st)
{
    Pairs c = census(st);
    Pairs lead = make_dict(st.lead.data(), (int)st.lead.size()).items;
    return region(st.stage, c, &lead);
}

std::vector<Region> regions(const Table& t)
{
    if(!in_fenced(t.sched))
        throw ckc::ValueError("regions exist under a fenced sched only");
    std::vector<Region> out;
    int pre;
    if(prefetch_region(t, &pre))
        out.push_back(region("prefetch", Pairs{{VMEM_READ, pre}}, NULL));
    rocke_attn_bwd_counted_terms_t cnt = counted_terms(t);
    if(cnt.present && eq(cnt.issue, "early"))
    {
        out.push_back(region("staging", Pairs{{DS_WRITE, cnt.stores}}, NULL));
        out.push_back(region("issue", Pairs{{VMEM_READ, cnt.loads}}, NULL));
    }
    for(const Counts& st : stages(t))
    {
        if(eq(st.stage, "D") && cnt.present && eq(cnt.issue, "late") && mfma(t).g4)
        {
            out.push_back(region("barrier", Pairs(), NULL));
            out.push_back(region("issue", Pairs{{VMEM_READ, cnt.loads}}, NULL));
        }
        out.push_back(stage_region(st));
    }
    return out;
}

void add_sum(ODict& sums, const Pair& g)
{
    sums.set(g.mask, sums.get(g.mask, 0) + g.count);
}

rocke_attn_bwd_ir_count_model_t ir_count_model(const Table& t, bool prefetch, bool prev_atomics)
{
    rocke_attn_bwd_ir_count_model_t m;
    memset(&m, 0, sizeof(m));
    int n_groups = 0;
    ODict sums;
    std::vector<const char*> names = stage_names(t);
    for(const char* name : names)
        for(const Pair& g : groups(t, name, prefetch, prev_atomics))
        {
            ++n_groups;
            add_sum(sums, g);
        }
    int fences = in_fenced(t.sched) ? (int)names.size() - 1 : 0;
    if(!eq(t.pipeline, "ck"))
        fences += (prefetch_fence(t) ? 1 : 0) + (counted(t) ? 2 : 0);
    std::vector<Counts> st = stages(t, prefetch, prev_atomics);
    std::vector<int> waits;
    if(!eq(t.pipeline, "ck"))
        waits = step_waits(t);
    m.sched_group_barrier = n_groups;
    m.sched_barrier = fences;
    m.s_setprio = t.setprio ? 2 * (int)names.size() : 0;
    m.iglp_opt = in_iglp(t.sched) ? 1 : 0;
    m.s_waitcnt = (int)waits.size();
    m.n_group_counts = (int)sums.items.size();
    for(size_t i = 0; i < sums.items.size(); ++i)
        m.group_counts[i] = sums.items[i];
    m.mfma = mfma(t);
    m.mfma_total = rocke_attn_bwd_gemm_counts_total(&m.mfma);
    ODict sm; /* stage_mfma, keyed by the stage's index in STAGES */
    std::vector<const char*> keys;
    for(const Counts& s : st)
    {
        size_t i = 0;
        for(; i < keys.size(); ++i)
            if(eq(keys[i], s.stage))
                break;
        if(i == keys.size())
            keys.push_back(s.stage);
        sm.set((int)i, s.mfma);
    }
    m.n_stages = (int)keys.size();
    for(size_t i = 0; i < keys.size(); ++i)
    {
        m.stage_names[i] = keys[i];
        m.stage_mfma[i] = sm.get((int)i, 0);
    }
    m.pipeline = t.pipeline;
    m.n_ring_waits = (int)waits.size();
    for(size_t i = 0; i < waits.size(); ++i)
        m.ring_waits[i] = waits[i];
    if(!eq(t.pipeline, "ck"))
        m.prefetch = ring_prefetch(t);
    return m;
}

/* ----------------------------------------------------------- from_geometry */

int atom_k_of(const rocke_attn_bwd_sched_geom_t& g, int value)
{
    if(!g.has_atoms || value == ROCKE_ATTN_BWD_NONE)
        return 16; /* _atom_k(None) */
    return value; /* an int atom is its own K, 0 included */
}

Table from_geometry(const rocke_attn_bwd_sched_geom_t& geom,
                    const char* arch,
                    const char* dtype,
                    int stage_vec,
                    const char* dq_mode,
                    bool acc_in_lds)
{
    const int none = ROCKE_ATTN_BWD_NONE;
    int depth = geom.ring_depth == none ? 1 : geom.ring_depth;
    if(depth == 0)
        depth = 1;
    const char* path = geom.global_path != NULL ? geom.global_path : "vgpr";
    const char* merge = geom.barrier_merge != NULL ? geom.barrier_merge : "off";
    const char* pipeline;
    if(eq(merge, "on"))
        pipeline = "merged";
    else if(!(eq(path, "vgpr") && depth == 1))
        pipeline = "ring";
    else
        pipeline = "sync";
    auto str = [](const char* v, const char* dflt) { return v != NULL ? v : dflt; };
    Table t = rocke_attn_bwd_stage_table_default();
    t.arch = arch;
    t.head_size = geom.head_size;
    t.k_m0 = geom.k_m0;
    t.k_n0 = geom.k_n0;
    t.k_k4 = geom.k_k4;
    t.waves = geom.waves;
    if(geom.has_grid_g4)
        t.grid_g4[0] = geom.grid_g4[0], t.grid_g4[1] = geom.grid_g4[1];
    t.atom_g02 = atom_k_of(geom, geom.atom_g02);
    t.atom_g13 = atom_k_of(geom, geom.atom_g13);
    t.atom_g4 = atom_k_of(geom, geom.atom_g4);
    t.dtype = dtype;
    t.transpose_source = str(geom.transpose_source, "xt_lds");
    t.kt_source = str(geom.kt_source, "reg");
    t.global_path = path;
    t.pt_route = str(geom.pt_route, "relabel");
    t.stage_vec = geom.stage_vec != none ? geom.stage_vec : stage_vec;
    t.dq_mode = str(geom.dq_mode, dq_mode);
    t.sched = str(geom.sched, "stage_table");
    t.setprio = geom.setprio;
    t.kv_residency = str(geom.kv_residency, "reg");
    t.acc_in_lds = geom.acc_in_lds != none ? geom.acc_in_lds != 0 : acc_in_lds;
    if(geom.has_grid_g0)
        t.grid_g02[0] = geom.grid_g0[0], t.grid_g02[1] = geom.grid_g0[1];
    if(geom.has_grid_g1)
        t.grid_g13[0] = geom.grid_g1[0], t.grid_g13[1] = geom.grid_g1[1];
    t.pipeline = pipeline;
    t.stats_load = str(geom.stats_load, "at_use");
    t.dq_issue = str(geom.dq_issue, "block");
    t.lds_swizzle = geom.has_lds_swizzle ? geom.lds_swizzle : NULL;
    t.g4_slicing = str(geom.g4_slicing, "step");
    t.ring_depth = depth;
    t.barrier_merge = merge;
    t.ring_issue = str(geom.ring_issue, "early");
    t.head_pack = geom.head_pack;
    t.loop_sync = str(geom.loop_sync, "full");
    t.counted_issue = str(geom.counted_issue, "early");
    post_init(t);
    return t;
}

/* ----------------------------------------------------------- DqStageTable */

void dq_post_init(const DqTable& t)
{
    if(!in_n(ROCKE_ATTN_BWD_DQ_SCHEDS, ROCKE_ATTN_BWD_N_DQ_SCHEDS, t.sched))
        throw ckc::ValueError("dq sched " + repr(t.sched) + " not in "
                              + tuple_repr_n(ROCKE_ATTN_BWD_DQ_SCHEDS, ROCKE_ATTN_BWD_N_DQ_SCHEDS));
    if(!in(DQ_TRANSPOSE_SOURCES, t.transpose_source))
        throw ckc::ValueError("dQ transpose_source must be one of "
                              + tuple_repr(DQ_TRANSPOSE_SOURCES));
    bool pad_ok = false;
    for(int p : DQ_KV_PADS)
        pad_ok = pad_ok || p == t.kv_pad;
    if(!pad_ok)
        throw ckc::ValueError("kv_pad must be one of "
                              + py_tuple(std::vector<int>(DQ_KV_PADS, DQ_KV_PADS + 4)));
    if(!(t.stage_vec == 1 || t.stage_vec == 8))
        throw ckc::ValueError("stage_vec must be 8 or 1");
    const long long w = t.waves, d = t.head_size, m0 = t.k_m0, n0 = t.k_n0;
    const std::string W = std::to_string(w), D = std::to_string(d), M0 = std::to_string(m0),
                      N0 = std::to_string(n0);
    const std::pair<bool, std::string> checks[] = {
        {fmod_(m0, 16 * w) == 0, "kM0 " + M0 + " % (16 * waves " + W + ") != 0"},
        {fmod_(n0, 16) == 0, "kN0 " + N0 + " % 16 != 0"},
        {fmod_(d, 16) == 0, "D " + D + " % 16 != 0"},
        {fmod_(d, t.atom_g02) == 0,
         "D " + D + " % atom_g02 K " + std::to_string(t.atom_g02) + " != 0"},
        {fmod_(n0, t.atom_g4) == 0,
         "kN0 " + N0 + " % atom_g4 K " + std::to_string(t.atom_g4) + " != 0"},
    };
    for(const auto& c : checks)
        if(!c.first)
            throw ckc::ValueError("illegal dQ stage-table geometry: " + c.second);
    for(int k : {t.atom_g02, t.atom_g4})
        instruction_k(t.arch, k, t.dtype);
}

int dq_wave_size(const DqTable& t)
{
    return target_of(t.arch)->wave_size;
}

bool dq_setprio(const DqTable& t)
{
    return eq(t.sched, "stage_table_setprio");
}

bool dq_table(const DqTable& t)
{
    return eq(t.sched, "stage_table") || eq(t.sched, "stage_table_setprio");
}

bool dq_fenced(const DqTable& t)
{
    return in_n(ROCKE_ATTN_BWD_DQ_FENCED_SCHEDS, ROCKE_ATTN_BWD_N_DQ_FENCED_SCHEDS, t.sched);
}

int dq_frag_len(const DqTable& t, int atom_k)
{
    int k = instruction_k(t.arch, atom_k, t.dtype);
    for(const rocke_mma_op_t* op : mma_ops(t.arch, t.dtype))
        if(op->k == k)
            return op->a_frag_len;
    throw ckc::ValueError("no native 16x16x" + std::to_string(k) + " atom on " + t.arch);
}

int dq_kv_align(const DqTable& t)
{
    long long pitch = (long long)t.head_size + fdiv(t.kv_pad, ELEM_BYTES);
    int a = 8;
    while(a > 1 && fmod_(pitch, a))
        a /= 2;
    return a;
}

rocke_attn_bwd_dq_mfma_t dq_mfma(const DqTable& t)
{
    const long long w = t.waves, d = t.head_size, m0 = t.k_m0, n0 = t.k_n0;
    int k02 = instruction_k(t.arch, t.atom_g02, t.dtype);
    int k4 = instruction_k(t.arch, t.atom_g4, t.dtype);
    long long g0 = fdiv(n0, 16) * fdiv(m0, 16 * w) * fdiv(d, k02);
    long long g4 = fdiv(m0, 16 * w) * fdiv(d, 16) * fdiv(n0, k4);
    return rocke_attn_bwd_dq_mfma_t{I(g0), I(g0), I(g4)};
}

std::pair<long long, long long> dq_kv_reads(const DqTable& t)
{
    int k02 = instruction_k(t.arch, t.atom_g02, t.dtype);
    int n = dq_frag_len(t, t.atom_g02);
    long long per = cdiv(n, std::min(8, dq_kv_align(t)));
    long long first = fdiv(t.k_n0, 16) * per;
    return {fdiv(t.head_size, k02) * first, first};
}

std::pair<long long, long long> dq_kt_reads(const DqTable& t)
{
    int k4 = instruction_k(t.arch, t.atom_g4, t.dtype);
    int n = dq_frag_len(t, t.atom_g4);
    long long per = eq(t.transpose_source, "tr_read") ? fdiv(n, 4) : n;
    long long first = fdiv(t.head_size, 16) * per;
    return {fdiv(t.k_n0, k4) * first, first};
}

rocke_attn_bwd_dq_prefetch_t dq_prefetch(const DqTable& t)
{
    rocke_attn_bwd_dq_prefetch_t p = {false, 0, 0};
    if(!t.kv_prefetch)
        return p;
    long long threads = (long long)t.waves * dq_wave_size(t);
    long long chunks = cdiv(fdiv((long long)t.k_n0 * t.head_size, 8), threads);
    long long loads = t.stage_vec == 8 ? 1 : 8;
    long long stores = cdiv(8, std::min(8, dq_kv_align(t)));
    p.present = true;
    p.vmem_read = I(2 * chunks * loads);
    p.ds_write = I(2 * chunks * stores);
    return p;
}

std::vector<Counts> dq_stages(const DqTable& t)
{
    rocke_attn_bwd_dq_mfma_t mf = dq_mfma(t);
    std::pair<long long, long long> kv = dq_kv_reads(t);
    std::pair<long long, long long> kt = dq_kt_reads(t);
    rocke_attn_bwd_dq_prefetch_t pf = dq_prefetch(t);
    Counts a = counts("A", mf.g0);
    a.vmem_read = pf.present ? pf.vmem_read : 0;
    a.ds_read = I(kv.first);
    a.lead = Pairs{{DS_READ, I(kv.second)}};
    Counts b = counts("B", mf.g2);
    b.ds_read = I(kv.first);
    b.lead = Pairs{{DS_READ, I(kv.second)}};
    Counts c = counts("C", mf.g4);
    c.ds_read = I(kt.first);
    c.lead = Pairs{{DS_READ, I(kt.second)}};
    if(pf.present)
        c.fixed = Pairs{{DS_WRITE, pf.ds_write}};
    return std::vector<Counts>{a, b, c};
}

std::vector<const char*> dq_names()
{
    return std::vector<const char*>(ROCKE_ATTN_BWD_DQ_STAGES,
                                    ROCKE_ATTN_BWD_DQ_STAGES + ROCKE_ATTN_BWD_N_DQ_STAGES);
}

Counts dq_stage(const DqTable& t, const char* name)
{
    std::vector<Counts> st = dq_stages(t);
    return st[name_index(dq_names(), name)];
}

Pairs dq_groups(const DqTable& t, const char* name)
{
    if(!dq_table(t))
        return Pairs();
    return stage_groups(dq_stage(t, name));
}

std::vector<Hint> dq_hint_sequence(const DqTable& t)
{
    std::vector<Hint> out;
    if(in_iglp(t.sched))
        out.push_back(hint(ROCKE_ATTN_BWD_HINT_IGLP, t.sched[strlen(t.sched) - 1] - '0'));
    for(const char* name : dq_names())
    {
        if(dq_setprio(t))
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_PRIO, 1));
        for(const Pair& g : dq_groups(t, name))
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_GROUP, g.mask, g.count));
        if(dq_setprio(t))
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_PRIO, 0));
        if(dq_fenced(t) && !eq(name, "C"))
            out.push_back(hint(ROCKE_ATTN_BWD_HINT_FENCE, 0));
    }
    return out;
}

/* --------------------------------------------------------- C boundaries */

template <class Fn>
rocke_status_t boundary(rocke_attn_bwd_err_t* err, Fn&& fn)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        fn();
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
    catch(const std::bad_alloc&)
    {
        return store_err(err, ROCKE_ERR_OOM, "out of memory");
    }
}

/* An emit_* entry: fn computes everything that can raise and returns the
 * emission to run; a raise records the builder error and emits nothing. */
template <class Fn>
bool emit_boundary(rocke_ir_builder_t* b, Fn&& fn)
{
    try
    {
        fn();
        return rocke_ir_builder_ok(b);
    }
    catch(const ckc::Error& e)
    {
        rocke_i_set_err_msg(b, e.code(), e.what());
        return false;
    }
    catch(const std::bad_alloc&)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_OOM, "out of memory");
        return false;
    }
}

/* The (out, cap, n_out) contract: *n_out is always the full count; out NULL
 * is a sizing query; a non-NULL out with cap below the count is refused (a
 * ValueError, nothing written) rather than silently truncated. */
template <typename T>
void copy_out(const std::vector<T>& src, T* out, int cap, int* n_out)
{
    if(n_out != NULL)
        *n_out = (int)src.size();
    if(out == NULL)
        return;
    if(cap < 0 || src.size() > (size_t)cap)
        throw ckc::ValueError("output array too small: " + std::to_string(src.size())
                              + " entries, cap " + std::to_string(cap));
    for(size_t i = 0; i < src.size(); ++i)
        out[i] = src[i];
}

void copy_pairs(const Pairs& src, Pair* out, int cap, int* n_out)
{
    copy_out(src, out, cap, n_out);
}

void copy_ints(const std::vector<int>& src, int* out, int cap, int* n_out)
{
    copy_out(src, out, cap, n_out);
}

void copy_hints(const std::vector<Hint>& src, Hint* out, int cap, int* n_out)
{
    copy_out(src, out, cap, n_out);
}

void emit_groups(rocke_ir_builder_t* b, const Pairs& gs)
{
    for(const Pair& g : gs)
        rocke_b_sched_group_barrier(b, g.mask, g.count, 0);
}

} // namespace
} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

/* ====================================================================== *
 * C ABI
 * ====================================================================== */

extern "C" {

int rocke_attn_bwd_stage_gemms(const char* stage, const char* out[2])
{
    if(eq(stage, "A"))
        return out[0] = "g0", 1;
    if(eq(stage, "B"))
        return out[0] = "g1", out[1] = "g2", 2;
    if(eq(stage, "C"))
        return out[0] = "g3", 1;
    if(eq(stage, "D"))
        return out[0] = "g4", 1;
    return -1;
}

rocke_status_t rocke_attn_bwd_native_atom_ks(
    const char* arch, const char* dtype, int* out, int cap, int* n_out, rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_ints(native_atom_ks(arch, dtype), out, cap, n_out); });
}

rocke_status_t rocke_attn_bwd_instruction_k(
    const char* arch, int atom_k, const char* dtype, int* out, rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = instruction_k(arch, atom_k, dtype); });
}

rocke_status_t rocke_attn_bwd_interleave_groups(int mfma,
                                                const rocke_attn_bwd_sched_pair_t* mem,
                                                int n_mem,
                                                const rocke_attn_bwd_sched_pair_t* lead,
                                                int n_lead,
                                                const rocke_attn_bwd_sched_pair_t* spans,
                                                int n_spans,
                                                rocke_attn_bwd_sched_pair_t* out,
                                                int cap,
                                                int* n_out,
                                                rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        Pairs m(mem, mem + (n_mem > 0 ? n_mem : 0));
        Pairs gs = interleave_groups(mfma, m, make_dict(lead, n_lead), make_dict(spans, n_spans));
        copy_pairs(gs, out, cap, n_out);
    });
}

int rocke_attn_bwd_gemm_counts_total(const rocke_attn_bwd_gemm_counts_t* c)
{
    return c->g0 + c->g1 + c->g2 + c->g3 + c->g4;
}

void rocke_attn_bwd_stage_counts_memory(const rocke_attn_bwd_stage_counts_t* s,
                                        rocke_attn_bwd_sched_pair_t out[4])
{
    Pairs m = memory(counts_scalars(*s));
    for(size_t i = 0; i < 4; ++i)
        out[i] = m[i];
}

void rocke_attn_bwd_stage_counts_by_mask(const rocke_attn_bwd_stage_counts_t* s,
                                         rocke_attn_bwd_sched_pair_t out[5])
{
    Pairs m = by_mask(counts_scalars(*s));
    for(size_t i = 0; i < 5; ++i)
        out[i] = m[i];
}

rocke_status_t rocke_attn_bwd_stage_counts_census(const rocke_attn_bwd_stage_counts_t* s,
                                                  rocke_attn_bwd_sched_pair_t out[5],
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_pairs(census(counts_in(*s)), out, 5, NULL); });
}

rocke_status_t rocke_attn_bwd_stage_counts_groups(const rocke_attn_bwd_stage_counts_t* s,
                                                  rocke_attn_bwd_sched_pair_t* out,
                                                  int cap,
                                                  int* n_out,
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_pairs(stage_groups(counts_in(*s)), out, cap, n_out); });
}

rocke_attn_bwd_stage_table_t rocke_attn_bwd_stage_table_default(void)
{
    rocke_attn_bwd_stage_table_t t;
    memset(&t, 0, sizeof(t));
    t.grid_g4[0] = 1, t.grid_g4[1] = 0;
    t.atom_g02 = 16;
    t.atom_g13 = 16;
    t.atom_g4 = 16;
    t.dtype = "fp16";
    t.transpose_source = "xt_lds";
    t.kt_source = "reg";
    t.global_path = "vgpr";
    t.pt_route = "relabel";
    t.stage_vec = 8;
    t.dq_mode = "atomic";
    t.sched = "stage_table";
    t.setprio = false;
    t.kv_residency = "reg";
    t.acc_in_lds = false;
    t.grid_g02[0] = 1, t.grid_g02[1] = 0;
    t.grid_g13[0] = 0, t.grid_g13[1] = 1;
    t.pipeline = "sync";
    t.stats_load = "at_use";
    t.dq_issue = "block";
    t.lds_swizzle = NULL;
    t.g4_slicing = "step";
    t.ring_depth = 1;
    t.barrier_merge = "off";
    t.ring_issue = "early";
    t.head_pack = false;
    t.loop_sync = "full";
    t.counted_issue = "early";
    return t;
}

rocke_status_t rocke_attn_bwd_stage_table_init(rocke_attn_bwd_stage_table_t* t,
                                               rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        rocke_attn_bwd_stage_table_t c = *t;
        post_init(c);
        *t = c;
    });
}

rocke_attn_bwd_sched_geom_t rocke_attn_bwd_sched_geom_empty(void)
{
    rocke_attn_bwd_sched_geom_t g;
    memset(&g, 0, sizeof(g));
    g.atom_g02 = ROCKE_ATTN_BWD_NONE;
    g.atom_g13 = ROCKE_ATTN_BWD_NONE;
    g.atom_g4 = ROCKE_ATTN_BWD_NONE;
    g.ring_depth = ROCKE_ATTN_BWD_NONE;
    g.stage_vec = ROCKE_ATTN_BWD_NONE;
    g.acc_in_lds = ROCKE_ATTN_BWD_NONE;
    return g;
}

rocke_status_t rocke_attn_bwd_stage_table_from_geometry(const rocke_attn_bwd_sched_geom_t* geom,
                                                        const char* arch,
                                                        const char* dtype,
                                                        int stage_vec,
                                                        const char* dq_mode,
                                                        bool acc_in_lds,
                                                        rocke_attn_bwd_stage_table_t* out,
                                                        rocke_attn_bwd_err_t* err)
{
    return boundary(
        err, [&] { *out = from_geometry(*geom, arch, dtype, stage_vec, dq_mode, acc_in_lds); });
}

rocke_status_t rocke_attn_bwd_stage_table_wave_size(const rocke_attn_bwd_stage_table_t* t,
                                                    int* out,
                                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = wave_size(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_instr_k(const rocke_attn_bwd_stage_table_t* t,
                                                  int out[5],
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        InstrK k = instr_k(*t);
        for(int i = 0; i < 5; ++i)
            out[i] = k.g[i];
    });
}

rocke_status_t rocke_attn_bwd_stage_table_mma_op_id(const rocke_attn_bwd_stage_table_t* t,
                                                    const char* gemm,
                                                    const char** out,
                                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = op_of(*t, instr_k(*t).at(gemm))->op_id; });
}

rocke_status_t rocke_attn_bwd_stage_table_frag_len(const rocke_attn_bwd_stage_table_t* t,
                                                   const char* gemm,
                                                   int* out,
                                                   rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = frag_len(*t, gemm); });
}

rocke_status_t rocke_attn_bwd_stage_table_c_frag_len(const rocke_attn_bwd_stage_table_t* t,
                                                     int* out,
                                                     rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = c_frag_len(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_mfma(const rocke_attn_bwd_stage_table_t* t,
                                               rocke_attn_bwd_gemm_counts_t* out,
                                               rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = mfma(*t); });
}

int rocke_attn_bwd_stage_table_stage_names(const rocke_attn_bwd_stage_table_t* t,
                                           const char* const** out)
{
    if(eq(t->dq_mode, "split") && !eq(t->pipeline, "ck"))
    {
        *out = ROCKE_ATTN_BWD_SPLIT_STAGES;
        return ROCKE_ATTN_BWD_N_SPLIT_STAGES;
    }
    *out = ROCKE_ATTN_BWD_STAGES;
    return ROCKE_ATTN_BWD_N_STAGES;
}

rocke_status_t rocke_attn_bwd_stage_table_dq_atomics_per_lane(const rocke_attn_bwd_stage_table_t* t,
                                                              int* out,
                                                              rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_atomics_per_lane(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_stats_loads(const rocke_attn_bwd_stage_table_t* t,
                                                      int* out,
                                                      rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = stats_loads(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_stages(const rocke_attn_bwd_stage_table_t* t,
                                                 bool prefetch,
                                                 bool prev_atomics,
                                                 rocke_attn_bwd_stage_counts_t out[4],
                                                 int* n_out,
                                                 rocke_attn_bwd_stage_pairs_store_t* store,
                                                 rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        std::vector<Counts> st = stages(*t, prefetch, prev_atomics);
        if(store != NULL)
            store->n = 0;
        for(size_t i = 0; i < st.size(); ++i)
            out[i] = counts_out(st[i], store);
        *n_out = (int)st.size();
    });
}

rocke_status_t rocke_attn_bwd_stage_table_stage(const rocke_attn_bwd_stage_table_t* t,
                                                const char* name,
                                                bool prefetch,
                                                bool prev_atomics,
                                                rocke_attn_bwd_stage_counts_t* out,
                                                rocke_attn_bwd_stage_pairs_store_t* store,
                                                rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        Counts c = stage(*t, name, prefetch, prev_atomics);
        if(store != NULL)
            store->n = 0;
        *out = counts_out(c, store);
    });
}

rocke_status_t rocke_attn_bwd_stage_table_groups(const rocke_attn_bwd_stage_table_t* t,
                                                 const char* name,
                                                 bool prefetch,
                                                 bool prev_atomics,
                                                 rocke_attn_bwd_sched_pair_t* out,
                                                 int cap,
                                                 int* n_out,
                                                 rocke_attn_bwd_err_t* err)
{
    return boundary(err,
                    [&] { copy_pairs(groups(*t, name, prefetch, prev_atomics), out, cap, n_out); });
}

rocke_status_t rocke_attn_bwd_stage_table_ring_prefetch(const rocke_attn_bwd_stage_table_t* t,
                                                        rocke_attn_bwd_ring_prefetch_t* out,
                                                        rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = ring_prefetch(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_prefetch_region(const rocke_attn_bwd_stage_table_t* t,
                                                          bool* has,
                                                          int* vmem_read,
                                                          rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        int v = 0;
        *has = prefetch_region(*t, &v);
        *vmem_read = v;
    });
}

rocke_status_t rocke_attn_bwd_stage_table_prefetch_fence(const rocke_attn_bwd_stage_table_t* t,
                                                         bool* out,
                                                         rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = prefetch_fence(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_ring_waits(const rocke_attn_bwd_stage_table_t* t,
                                                     int out[1],
                                                     int* n_out,
                                                     rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_ints(ring_waits(*t), out, 1, n_out); });
}

rocke_status_t rocke_attn_bwd_stage_table_ring_atomics_per_lane(
    const rocke_attn_bwd_stage_table_t* t, int* out, rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_atomics_per_lane(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_vm_after_prefetch(const rocke_attn_bwd_stage_table_t* t,
                                                            bool prev_atomics,
                                                            int* out,
                                                            rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = vm_after_prefetch(*t, prev_atomics); });
}

bool rocke_attn_bwd_stage_table_counted(const rocke_attn_bwd_stage_table_t* t)
{
    return counted(*t);
}

rocke_status_t rocke_attn_bwd_stage_table_counted_terms(const rocke_attn_bwd_stage_table_t* t,
                                                        rocke_attn_bwd_counted_terms_t* out,
                                                        rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = counted_terms(*t); });
}

rocke_status_t rocke_attn_bwd_stage_table_counted_waits(const rocke_attn_bwd_stage_table_t* t,
                                                        int out[3],
                                                        int* n_out,
                                                        rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_ints(counted_waits(*t), out, 3, n_out); });
}

rocke_status_t rocke_attn_bwd_stage_table_step_waits(const rocke_attn_bwd_stage_table_t* t,
                                                     int out[4],
                                                     int* n_out,
                                                     rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_ints(step_waits(*t), out, 4, n_out); });
}

rocke_status_t rocke_attn_bwd_stage_table_hint_sequence(const rocke_attn_bwd_stage_table_t* t,
                                                        rocke_attn_bwd_hint_t* out,
                                                        int cap,
                                                        int* n_out,
                                                        rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_hints(hint_sequence(*t), out, cap, n_out); });
}

rocke_status_t rocke_attn_bwd_stage_table_regions(
    const rocke_attn_bwd_stage_table_t* t,
    rocke_attn_bwd_sched_region_t out[ROCKE_ATTN_BWD_SCHED_REGIONS_MAX],
    int* n_out,
    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        std::vector<Region> rs = regions(*t);
        if(rs.size() > ROCKE_ATTN_BWD_SCHED_REGIONS_MAX)
            throw ckc::ValueError("more than " + std::to_string(ROCKE_ATTN_BWD_SCHED_REGIONS_MAX)
                                  + " regions");
        for(size_t i = 0; i < rs.size(); ++i)
            out[i] = rs[i];
        *n_out = (int)rs.size();
    });
}

rocke_status_t rocke_attn_bwd_stage_table_ir_count_model(const rocke_attn_bwd_stage_table_t* t,
                                                         bool prefetch,
                                                         bool prev_atomics,
                                                         rocke_attn_bwd_ir_count_model_t* out,
                                                         rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = ir_count_model(*t, prefetch, prev_atomics); });
}

bool rocke_attn_bwd_stage_table_emit_loop_head(rocke_ir_builder_t* b,
                                               const rocke_attn_bwd_stage_table_t* t)
{
    return emit_boundary(b, [&] {
        if(eq(t->sched, "iglp0"))
            rocke_b_iglp_opt(b, 0);
        else if(eq(t->sched, "iglp1"))
        {
            rocke_b_iglp_opt(b, 1);
            rocke_attr_set_bool(b, &b->kernel->attrs, ROCKE_COMGR_FRESH_PROCESS_ATTR, true);
        }
    });
}

bool rocke_attn_bwd_stage_table_emit_stage_begin(rocke_ir_builder_t* b,
                                                 const rocke_attn_bwd_stage_table_t* t,
                                                 const char* name)
{
    (void)name;
    return emit_boundary(b, [&] {
        if(t->setprio)
            rocke_b_s_setprio(b, 1);
    });
}

bool rocke_attn_bwd_stage_table_emit_stage_end(rocke_ir_builder_t* b,
                                               const rocke_attn_bwd_stage_table_t* t,
                                               const char* name,
                                               bool prefetch,
                                               bool prev_atomics)
{
    return emit_boundary(b, [&] {
        std::vector<const char*> names = stage_names(*t);
        bool known = false;
        for(const char* n : names)
            known = known || eq(n, name);
        if(!known)
        {
            std::vector<std::string> ns(names.begin(), names.end());
            throw ckc::ValueError("unknown stage " + repr(name) + " (stages " + py_tuple(ns) + ")");
        }
        Pairs gs = groups(*t, name, prefetch, prev_atomics);
        emit_groups(b, gs);
        if(t->setprio)
            rocke_b_s_setprio(b, 0);
        if(!eq(name, names.back()) && in_fenced(t->sched))
            rocke_b_sched_barrier(b, 0);
    });
}

/* ----- DqStageTable ----- */

rocke_attn_bwd_dq_stage_table_t rocke_attn_bwd_dq_stage_table_default(void)
{
    rocke_attn_bwd_dq_stage_table_t t;
    memset(&t, 0, sizeof(t));
    t.atom_g02 = 16;
    t.atom_g4 = 16;
    t.dtype = "fp16";
    t.transpose_source = "tr_read";
    t.kv_pad = 0;
    t.kv_prefetch = false;
    t.stage_vec = 8;
    t.sched = "stage_table";
    return t;
}

rocke_status_t rocke_attn_bwd_dq_stage_table_init(rocke_attn_bwd_dq_stage_table_t* t,
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { dq_post_init(*t); });
}

rocke_status_t
    rocke_attn_bwd_dq_stage_table_from_geometry(const rocke_attn_bwd_dq_sched_geom_t* geom,
                                                rocke_attn_bwd_dq_stage_table_t* out,
                                                rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        const int none = ROCKE_ATTN_BWD_NONE;
        rocke_attn_bwd_dq_stage_table_t t = rocke_attn_bwd_dq_stage_table_default();
        t.arch = geom->arch;
        t.head_size = geom->head_size;
        t.k_m0 = geom->k_m0;
        t.k_n0 = geom->k_n0;
        t.waves = geom->waves;
        t.atom_g02 = geom->atom_g02;
        t.atom_g4 = geom->atom_g4;
        t.dtype = geom->dtype;
        t.transpose_source = geom->transpose_source;
        t.kv_pad = geom->kv_pad != none ? geom->kv_pad : 0;
        t.kv_prefetch = geom->has_kv_prefetch ? geom->kv_prefetch : false;
        t.stage_vec = geom->stage_vec != none ? geom->stage_vec : 8;
        t.sched = geom->sched != NULL ? geom->sched : "none";
        dq_post_init(t);
        *out = t;
    });
}

rocke_status_t rocke_attn_bwd_dq_stage_table_wave_size(const rocke_attn_bwd_dq_stage_table_t* t,
                                                       int* out,
                                                       rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_wave_size(*t); });
}

bool rocke_attn_bwd_dq_stage_table_setprio(const rocke_attn_bwd_dq_stage_table_t* t)
{
    return dq_setprio(*t);
}

bool rocke_attn_bwd_dq_stage_table_table(const rocke_attn_bwd_dq_stage_table_t* t)
{
    return dq_table(*t);
}

bool rocke_attn_bwd_dq_stage_table_fenced(const rocke_attn_bwd_dq_stage_table_t* t)
{
    return dq_fenced(*t);
}

rocke_status_t rocke_attn_bwd_dq_stage_table_mfma(const rocke_attn_bwd_dq_stage_table_t* t,
                                                  rocke_attn_bwd_dq_mfma_t* out,
                                                  rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_mfma(*t); });
}

rocke_status_t rocke_attn_bwd_dq_stage_table_prefetch(const rocke_attn_bwd_dq_stage_table_t* t,
                                                      rocke_attn_bwd_dq_prefetch_t* out,
                                                      rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { *out = dq_prefetch(*t); });
}

rocke_status_t rocke_attn_bwd_dq_stage_table_stages(const rocke_attn_bwd_dq_stage_table_t* t,
                                                    rocke_attn_bwd_stage_counts_t out[3],
                                                    rocke_attn_bwd_stage_pairs_store_t* store,
                                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        std::vector<Counts> st = dq_stages(*t);
        if(store != NULL)
            store->n = 0;
        for(size_t i = 0; i < 3; ++i)
            out[i] = counts_out(st[i], store);
    });
}

rocke_status_t rocke_attn_bwd_dq_stage_table_stage(const rocke_attn_bwd_dq_stage_table_t* t,
                                                   const char* name,
                                                   rocke_attn_bwd_stage_counts_t* out,
                                                   rocke_attn_bwd_stage_pairs_store_t* store,
                                                   rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        Counts c = dq_stage(*t, name);
        if(store != NULL)
            store->n = 0;
        *out = counts_out(c, store);
    });
}

rocke_status_t rocke_attn_bwd_dq_stage_table_groups(const rocke_attn_bwd_dq_stage_table_t* t,
                                                    const char* name,
                                                    rocke_attn_bwd_sched_pair_t* out,
                                                    int cap,
                                                    int* n_out,
                                                    rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_pairs(dq_groups(*t, name), out, cap, n_out); });
}

rocke_status_t rocke_attn_bwd_dq_stage_table_hint_sequence(const rocke_attn_bwd_dq_stage_table_t* t,
                                                           rocke_attn_bwd_hint_t* out,
                                                           int cap,
                                                           int* n_out,
                                                           rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] { copy_hints(dq_hint_sequence(*t), out, cap, n_out); });
}

rocke_status_t rocke_attn_bwd_dq_stage_table_regions(const rocke_attn_bwd_dq_stage_table_t* t,
                                                     rocke_attn_bwd_sched_region_t out[3],
                                                     rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        if(!dq_fenced(*t))
            throw ckc::ValueError("regions exist under a fenced dq sched only");
        std::vector<Counts> st = dq_stages(*t);
        for(size_t i = 0; i < 3; ++i)
            out[i] = stage_region(st[i]);
    });
}

rocke_status_t
    rocke_attn_bwd_dq_stage_table_ir_count_model(const rocke_attn_bwd_dq_stage_table_t* t,
                                                 rocke_attn_bwd_dq_ir_count_model_t* out,
                                                 rocke_attn_bwd_err_t* err)
{
    return boundary(err, [&] {
        rocke_attn_bwd_dq_ir_count_model_t m;
        memset(&m, 0, sizeof(m));
        ODict sums;
        for(const char* name : dq_names())
            for(const Pair& g : dq_groups(*t, name))
            {
                ++m.sched_group_barrier;
                add_sum(sums, g);
            }
        m.sched_barrier = dq_fenced(*t) ? ROCKE_ATTN_BWD_N_DQ_STAGES - 1 : 0;
        m.s_setprio = dq_setprio(*t) ? 2 * ROCKE_ATTN_BWD_N_DQ_STAGES : 0;
        m.iglp_opt = in_iglp(t->sched) ? 1 : 0;
        m.s_waitcnt = 0;
        m.n_group_counts = (int)sums.items.size();
        for(size_t i = 0; i < sums.items.size(); ++i)
            m.group_counts[i] = sums.items[i];
        m.mfma = dq_mfma(*t);
        std::vector<Counts> st = dq_stages(*t);
        for(size_t i = 0; i < 3; ++i)
            m.stage_mfma[i] = st[i].mfma;
        m.prefetch = dq_prefetch(*t);
        *out = m;
    });
}

bool rocke_attn_bwd_dq_stage_table_emit_loop_head(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_dq_stage_table_t* t)
{
    return emit_boundary(b, [&] {
        if(in_iglp(t->sched))
        {
            int level = t->sched[strlen(t->sched) - 1] - '0';
            rocke_b_iglp_opt(b, level);
            if(level == 1)
                rocke_attr_set_bool(b, &b->kernel->attrs, ROCKE_COMGR_FRESH_PROCESS_ATTR, true);
        }
    });
}

bool rocke_attn_bwd_dq_stage_table_emit_stage_begin(rocke_ir_builder_t* b,
                                                    const rocke_attn_bwd_dq_stage_table_t* t,
                                                    const char* name)
{
    return emit_boundary(b, [&] {
        if(!in_n(ROCKE_ATTN_BWD_DQ_STAGES, ROCKE_ATTN_BWD_N_DQ_STAGES, name))
            throw ckc::ValueError("unknown dQ stage " + repr(name));
        if(dq_setprio(*t))
            rocke_b_s_setprio(b, 1);
    });
}

bool rocke_attn_bwd_dq_stage_table_emit_stage_end(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_dq_stage_table_t* t,
                                                  const char* name)
{
    return emit_boundary(b, [&] {
        if(!in_n(ROCKE_ATTN_BWD_DQ_STAGES, ROCKE_ATTN_BWD_N_DQ_STAGES, name))
            throw ckc::ValueError("unknown dQ stage " + repr(name));
        Pairs gs = dq_groups(*t, name);
        emit_groups(b, gs);
        if(dq_setprio(*t))
            rocke_b_s_setprio(b, 0);
        if(dq_fenced(*t) && !eq(name, "C"))
            rocke_b_sched_barrier(b, 0);
    });
}

} // extern "C"
