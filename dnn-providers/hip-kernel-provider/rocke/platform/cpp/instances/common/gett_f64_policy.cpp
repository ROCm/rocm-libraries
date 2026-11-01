// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gett_f64_policy.cpp -- the spec policy of instances/common/gett_f64.py:
 * gemm_problem, pick_tile_k, arch_spec, TUNED_CONFIGS and _with_k_block. A
 * C++ caller (hipTensor's pack builder) resolves the same GettF64Spec the
 * Python policy resolves and builds it with rocke_build_gett_f64; the
 * gett_f64_pack parity family proves the resulting IR byte-identical.
 */
#include "rocke/instance_gett_f64.h"

#include <string.h>

namespace
{

rocke_gett_digits_t one(int64_t e, int64_t s)
{
    rocke_gett_digits_t d = {};
    d.n = 1;
    d.extent[0] = e;
    d.stride[0] = s;
    rocke_gett_digits_norm(&d);
    return d;
}

/* One TUNED_CONFIGS row. */
struct tuned_row
{
    const char* arch;
    const char* kind;
    const char* cls;
    rocke_gett_f64_knobs_t knobs;
};

constexpr unsigned k_base = ROCKE_GETT_F64_KNOB_TILE_M | ROCKE_GETT_F64_KNOB_TILE_N
                            | ROCKE_GETT_F64_KNOB_TILE_K | ROCKE_GETT_F64_KNOB_WARPS_M
                            | ROCKE_GETT_F64_KNOB_WARPS_N | ROCKE_GETT_F64_KNOB_GROUP_M
                            | ROCKE_GETT_F64_KNOB_WAVES_PER_EU | ROCKE_GETT_F64_KNOB_LDS_STAGES
                            | ROCKE_GETT_F64_KNOB_SCHED_FENCE;
/* Python _SWZ0 / _SWZ1 / _SWZ2: lds_pad=None, lds_swizzle=True, lds_swizzle_mode. */
constexpr unsigned k_swz = ROCKE_GETT_F64_KNOB_LDS_PAD | ROCKE_GETT_F64_KNOB_LDS_SWIZZLE
                           | ROCKE_GETT_F64_KNOB_LDS_SWIZZLE_MODE;
/* lds_swizzle=False, lds_pad=n */
constexpr unsigned k_pad = ROCKE_GETT_F64_KNOB_LDS_PAD | ROCKE_GETT_F64_KNOB_LDS_SWIZZLE;
constexpr unsigned k_xcds = ROCKE_GETT_F64_KNOB_XCDS;
constexpr unsigned k_u2 = ROCKE_GETT_F64_KNOB_UNROLL2;
constexpr unsigned k_pd = ROCKE_GETT_F64_KNOB_PREFETCH_DEPTH;

/* Knob row: set, tile m / n / k, warps m / n, lds_pad, lds_swizzle, group_m,
 * xcds, waves_per_eu, bilinear (unused), sched_fence, lds_swizzle_mode,
 * lds_stages, prefetch_depth, unroll2, runtime_sizes (unused). */
#define ROW(                                                                                     \
    arch, kind, cls, set, tm, tn, tk, wm, wn, pad, swz, gm, xcds, wpe, sf, mode, st, pd, u2)     \
    {                                                                                            \
        arch, kind, cls,                                                                         \
        {                                                                                        \
            set, tm, tn, tk, wm, wn, pad, swz, gm, xcds, wpe, false, sf, mode, st, pd, u2, false \
        }                                                                                        \
    }

/* Every knob of an f32 row is explicit (Python TUNED_CONFIGS_F32). */
constexpr unsigned k_all = k_base | ROCKE_GETT_F64_KNOB_LDS_PAD | ROCKE_GETT_F64_KNOB_LDS_SWIZZLE
                           | ROCKE_GETT_F64_KNOB_LDS_SWIZZLE_MODE | ROCKE_GETT_F64_KNOB_XCDS
                           | ROCKE_GETT_F64_KNOB_PREFETCH_DEPTH | ROCKE_GETT_F64_KNOB_UNROLL2
                           | ROCKE_GETT_F64_KNOB_K_STAGGER;

/* f32 knob row: tile m / n / k, warps m / n, lds_pad (-1: None), lds_swizzle,
 * group_m, xcds, waves_per_eu (0: None), sched_fence, lds_swizzle_mode,
 * lds_stages, prefetch_depth, unroll2, k_stagger. */
#define ROW32(                                                                               \
    arch, kind, cls, tm, tn, tk, wm, wn, pad, swz, gm, xcds, wpe, sf, mode, st, pd, u2, ks)  \
    {                                                                                        \
        arch, kind, cls,                                                                     \
        {                                                                                    \
            k_all, tm, tn, tk, wm, wn, pad, swz, gm, xcds, wpe, false, sf, mode, st, pd, u2, \
                false, ks                                                                    \
        }                                                                                    \
    }

/* Python TUNED_CONFIGS_F32, sorted by (arch, kind, cls). Keep in step with
 * gett_f64.py: the gett_f64_pack parity family compares every row. */
const tuned_row k_tuned_f32[] = {
    ROW32("gfx942",
          "complex",
          "compact",
          64,
          32,
          24,
          4,
          1,
          4,
          false,
          1,
          0,
          3,
          true,
          0,
          1,
          1,
          false,
          0),
    ROW32("gfx942",
          "complex",
          "large",
          128,
          128,
          4,
          2,
          2,
          -1,
          true,
          32,
          0,
          2,
          true,
          0,
          1,
          1,
          false,
          0),
    ROW32(
        "gfx942", "complex", "medium", 64, 64, 8, 4, 2, -1, true, 2, 0, 1, true, 0, 2, 1, false, 0),
    ROW32("gfx942",
          "complex",
          "small",
          32,
          32,
          24,
          2,
          2,
          -1,
          true,
          32,
          0,
          3,
          true,
          0,
          1,
          2,
          false,
          0),
    ROW32(
        "gfx942", "complex", "wide", 128, 64, 4, 2, 2, -1, true, 2, 0, 3, true, 0, 2, 1, false, 0),
    ROW32(
        "gfx942", "real", "compact", 32, 32, 32, 2, 2, -1, true, 16, 0, 3, true, 0, 2, 1, false, 0),
    ROW32(
        "gfx942", "real", "huge", 128, 128, 32, 2, 2, -1, true, 32, 0, 2, true, 0, 1, 1, false, 0),
    ROW32("gfx942", "real", "large", 64, 128, 16, 2, 2, -1, true, 4, 8, 3, true, 0, 1, 1, false, 0),
    ROW32(
        "gfx942", "real", "medium", 64, 64, 32, 2, 2, -1, true, 32, 0, 3, true, 0, 1, 1, false, 0),
    ROW32("gfx942", "real", "small", 32, 32, 32, 1, 2, -1, true, 8, 0, 3, true, 0, 2, 1, false, 0),
    ROW32("gfx942", "real", "tiny", 32, 32, 64, 2, 2, -1, true, 4, 0, 2, true, 0, 2, 1, false, 0),
    ROW32(
        "gfx942", "real", "xlarge", 128, 128, 16, 2, 2, -1, true, 4, 8, 3, true, 0, 1, 1, false, 0),
    ROW32("gfx950",
          "complex",
          "large",
          64,
          128,
          8,
          1,
          4,
          4,
          false,
          32,
          0,
          4,
          true,
          0,
          1,
          1,
          false,
          0),
    ROW32(
        "gfx950", "complex", "longk", 32, 32, 32, 2, 2, 4, false, 32, 0, 4, true, 0, 2, 1, true, 0),
    ROW32("gfx950",
          "complex",
          "medium",
          64,
          64,
          8,
          2,
          2,
          4,
          false,
          32,
          0,
          4,
          true,
          0,
          1,
          1,
          false,
          0),
    ROW32(
        "gfx950", "complex", "small", 32, 32, 16, 2, 2, 4, false, 4, 8, 4, true, 0, 2, 1, false, 0),
    ROW32(
        "gfx950", "real", "huge", 128, 128, 16, 2, 2, -1, true, 32, 0, 3, false, 1, 2, 1, true, 1),
    ROW32("gfx950", "real", "large", 64, 64, 32, 2, 2, -1, true, 32, 8, 4, true, 0, 1, 1, false, 0),
    ROW32(
        "gfx950", "real", "medium", 64, 32, 32, 2, 2, 1, false, 32, 8, 4, true, 0, 1, 1, false, 0),
    ROW32("gfx950", "real", "small", 32, 32, 64, 2, 2, 1, false, 4, 8, 4, true, 0, 2, 1, false, 0),
    ROW32("gfx950", "real", "tiny", 32, 32, 64, 2, 2, 1, false, 8, 8, 4, true, 0, 2, 2, false, 0),
    ROW32(
        "gfx950", "real", "xlarge", 128, 64, 16, 2, 2, -1, true, 16, 0, 4, false, 1, 2, 1, true, 0),
};

/* Python TUNED_CONFIGS, sorted by (arch, kind, cls). Keep in step with
 * gett_f64.py: the gett_f64_pack parity family compares every row. */
const tuned_row k_tuned[] = {
    ROW("gfx90a",
        "complex",
        "large",
        k_base | k_swz | k_pd,
        64,
        128,
        8,
        4,
        2,
        -1,
        true,
        1,
        0,
        4,
        false,
        1,
        1,
        2,
        false),
    ROW("gfx90a",
        "complex",
        "medium",
        k_base | k_swz,
        64,
        64,
        8,
        2,
        2,
        -1,
        true,
        8,
        0,
        4,
        true,
        1,
        1,
        1,
        false),
    ROW("gfx90a",
        "complex",
        "wide",
        k_base | k_swz | k_u2,
        64,
        128,
        4,
        2,
        2,
        -1,
        true,
        8,
        0,
        2,
        false,
        1,
        2,
        1,
        true),
    ROW("gfx90a",
        "real",
        "kk",
        k_base | k_pad,
        128,
        128,
        16,
        4,
        2,
        1,
        false,
        32,
        0,
        1,
        true,
        0,
        1,
        1,
        false),
    ROW("gfx90a",
        "real",
        "kmixed",
        k_base | k_swz,
        128,
        128,
        16,
        4,
        2,
        -1,
        true,
        8,
        0,
        4,
        true,
        0,
        1,
        1,
        false),
    ROW("gfx90a",
        "real",
        "medium",
        k_base | k_swz,
        128,
        128,
        16,
        4,
        2,
        -1,
        true,
        4,
        0,
        1,
        true,
        0,
        1,
        1,
        false),
    ROW("gfx90a",
        "real",
        "pow2",
        k_base | k_swz,
        128,
        128,
        8,
        2,
        2,
        -1,
        true,
        1,
        0,
        2,
        true,
        2,
        1,
        1,
        false),
    ROW("gfx90a",
        "real",
        "small",
        k_base | k_pad,
        64,
        128,
        12,
        2,
        2,
        4,
        false,
        4,
        0,
        3,
        true,
        0,
        1,
        1,
        false),
    ROW("gfx90a",
        "real",
        "wide",
        k_base | k_swz,
        128,
        128,
        16,
        2,
        2,
        -1,
        true,
        4,
        0,
        2,
        false,
        2,
        1,
        1,
        false),
    ROW("gfx942",
        "complex",
        "medium",
        k_base | k_pad | k_xcds | k_pd,
        128,
        64,
        8,
        4,
        2,
        4,
        false,
        1,
        8,
        4,
        false,
        0,
        1,
        2,
        false),
    ROW("gfx942",
        "complex",
        "small",
        k_base,
        64,
        64,
        8,
        2,
        2,
        -1,
        true,
        8,
        0,
        2,
        true,
        0,
        2,
        1,
        false),
    ROW("gfx942",
        "real",
        "medium",
        k_base | k_swz | k_xcds,
        128,
        128,
        16,
        2,
        2,
        -1,
        true,
        4,
        8,
        2,
        true,
        1,
        1,
        1,
        false),
    ROW("gfx942",
        "real",
        "small",
        k_base | k_swz | k_pd,
        64,
        128,
        32,
        2,
        4,
        -1,
        true,
        4,
        0,
        3,
        true,
        2,
        1,
        2,
        false),
    ROW("gfx950",
        "complex",
        "large",
        k_base | k_pad,
        128,
        64,
        8,
        1,
        4,
        4,
        false,
        16,
        0,
        2,
        false,
        0,
        1,
        1,
        false),
    ROW("gfx950",
        "complex",
        "medium",
        k_base | k_swz,
        64,
        64,
        8,
        1,
        4,
        -1,
        true,
        32,
        0,
        4,
        true,
        0,
        1,
        1,
        false),
    ROW("gfx950",
        "complex",
        "small",
        k_base | k_swz | k_xcds | k_u2,
        32,
        32,
        24,
        2,
        2,
        -1,
        true,
        32,
        8,
        3,
        false,
        0,
        2,
        1,
        true),
    ROW("gfx950",
        "real",
        "large",
        k_base | k_swz | k_u2,
        128,
        128,
        16,
        2,
        2,
        -1,
        true,
        32,
        0,
        2,
        false,
        0,
        2,
        1,
        true),
    ROW("gfx950",
        "real",
        "medium",
        k_base | k_swz,
        64,
        128,
        8,
        2,
        2,
        -1,
        true,
        32,
        0,
        4,
        true,
        2,
        1,
        1,
        false),
    ROW("gfx950",
        "real",
        "small",
        k_base | k_swz | k_xcds,
        32,
        32,
        32,
        2,
        2,
        -1,
        true,
        8,
        8,
        4,
        true,
        2,
        1,
        1,
        false),
};

#undef ROW

bool str_eq(const char* a, const char* b)
{
    return a != nullptr && b != nullptr && strcmp(a, b) == 0;
}

void apply(rocke_gett_f64_spec_t* s, const rocke_gett_f64_knobs_t& k)
{
    if(k.set & ROCKE_GETT_F64_KNOB_TILE_M)
        s->tile_m = k.tile_m;
    if(k.set & ROCKE_GETT_F64_KNOB_TILE_N)
        s->tile_n = k.tile_n;
    if(k.set & ROCKE_GETT_F64_KNOB_TILE_K)
        s->tile_k = k.tile_k;
    if(k.set & ROCKE_GETT_F64_KNOB_WARPS_M)
        s->warps_m = k.warps_m;
    if(k.set & ROCKE_GETT_F64_KNOB_WARPS_N)
        s->warps_n = k.warps_n;
    if(k.set & ROCKE_GETT_F64_KNOB_LDS_PAD)
        s->lds_pad = k.lds_pad;
    if(k.set & ROCKE_GETT_F64_KNOB_LDS_SWIZZLE)
        s->lds_swizzle = k.lds_swizzle;
    if(k.set & ROCKE_GETT_F64_KNOB_GROUP_M)
        s->group_m = k.group_m;
    if(k.set & ROCKE_GETT_F64_KNOB_XCDS)
        s->xcds = k.xcds;
    if(k.set & ROCKE_GETT_F64_KNOB_WAVES_PER_EU)
        s->waves_per_eu = k.waves_per_eu;
    if(k.set & ROCKE_GETT_F64_KNOB_BILINEAR)
        s->bilinear = k.bilinear;
    if(k.set & ROCKE_GETT_F64_KNOB_SCHED_FENCE)
        s->sched_fence = k.sched_fence;
    if(k.set & ROCKE_GETT_F64_KNOB_LDS_SWIZZLE_MODE)
        s->lds_swizzle_mode = k.lds_swizzle_mode;
    if(k.set & ROCKE_GETT_F64_KNOB_LDS_STAGES)
        s->lds_stages = k.lds_stages;
    if(k.set & ROCKE_GETT_F64_KNOB_PREFETCH_DEPTH)
        s->prefetch_depth = k.prefetch_depth;
    if(k.set & ROCKE_GETT_F64_KNOB_UNROLL2)
        s->unroll2 = k.unroll2;
    if(k.set & ROCKE_GETT_F64_KNOB_RUNTIME_SIZES)
        s->runtime_sizes = k.runtime_sizes;
    if(k.set & ROCKE_GETT_F64_KNOB_K_STAGGER)
        s->k_stagger = k.k_stagger;
}

/* Python TILE_K_CANDIDATES */
const int k_tile_k_candidates[] = {8, 16, 12, 24, 20, 4};

} // namespace

extern "C" {

rocke_gett_f64_problem_t rocke_gett_f64_gemm_problem(int64_t M,
                                                     int64_t N,
                                                     int64_t K,
                                                     const int64_t a_strides[2],
                                                     const int64_t b_strides[2],
                                                     const int64_t d_strides[2],
                                                     const int64_t* c_strides,
                                                     int64_t batch,
                                                     const int64_t batch_strides[3],
                                                     const int64_t* c_batch_stride,
                                                     bool complex_,
                                                     bool conj_a,
                                                     bool conj_b)
{
    const int64_t* cs = c_strides != nullptr ? c_strides : d_strides;
    int64_t cbs = c_batch_stride != nullptr ? *c_batch_stride : batch_strides[2];
    rocke_gett_f64_problem_t p = {};
    p.M = M;
    p.N = N;
    p.K = K;
    p.L = batch;
    p.a_m = one(M, a_strides[0]);
    p.a_k = one(K, a_strides[1]);
    p.b_n = one(N, b_strides[0]);
    p.b_k = one(K, b_strides[1]);
    p.d_m = one(M, d_strides[0]);
    p.d_n = one(N, d_strides[1]);
    p.a_l = one(batch, batch_strides[0]);
    p.b_l = one(batch, batch_strides[1]);
    p.d_l = one(batch, batch_strides[2]);
    /* gemm_problem always passes C's digits (D's strides by default). */
    p.has_c = true;
    p.c_m = one(M, cs[0]);
    p.c_n = one(N, cs[1]);
    p.c_l = one(batch, cbs);
    p.complex_ = complex_;
    p.conj_a = conj_a;
    p.conj_b = conj_b;
    return p;
}

int rocke_gett_f64_pick_tile_k(const rocke_gett_f64_problem_t* problem,
                               int preferred,
                               int max_tile_k)
{
    if(problem == nullptr)
        return preferred;
    auto fits = [&](int t) {
        if(t > max_tile_k)
            return false;
        const char sides[2] = {'A', 'B'};
        for(char which : sides)
        {
            const rocke_gett_digits_t* kd = which == 'A' ? &problem->a_k : &problem->b_k;
            if(!rocke_gett_f64_k_splits(kd, t)
               || rocke_gett_f64_addr_span_bytes(problem, which, t) > ROCKE_GETT_F64_MAX_SPAN)
                return false;
        }
        return true;
    };
    if(fits(preferred))
        return preferred;
    for(int t : k_tile_k_candidates)
        if(fits(t))
            return t;
    return preferred;
}

rocke_gett_f64_spec_t rocke_gett_f64_arch_spec(const rocke_gett_f64_problem_t* problem,
                                               const char* arch,
                                               const rocke_gett_f64_knobs_t* knobs)
{
    rocke_gett_f64_spec_t s = rocke_gett_f64_spec_default(problem);
    s.waves_per_eu = 2;
    if(problem != nullptr && problem->complex_)
    {
        s.tile_m = 64;
        s.tile_n = 64;
    }
    if(knobs == nullptr || !(knobs->set & ROCKE_GETT_F64_KNOB_TILE_K))
    {
        int preferred = str_eq(arch, "gfx950") ? 16 : 8;
        s.tile_k = rocke_gett_f64_pick_tile_k(problem, preferred, 2 * preferred);
    }
    if(knobs != nullptr)
        apply(&s, *knobs);
    return s;
}

bool rocke_gett_f64_tuned_config(const char* arch,
                                 const char* kind,
                                 const char* cls,
                                 rocke_gett_f64_knobs_t* out)
{
    for(const tuned_row& r : k_tuned)
    {
        if(str_eq(r.arch, arch) && str_eq(r.kind, kind) && str_eq(r.cls, cls))
        {
            if(out != nullptr)
                *out = r.knobs;
            return true;
        }
    }
    return false;
}

const char* rocke_gett_f64_tuned_class_at(const char* arch, const char* kind, int index)
{
    int i = 0;
    for(const tuned_row& r : k_tuned)
    {
        if(str_eq(r.arch, arch) && str_eq(r.kind, kind))
        {
            if(i == index)
                return r.cls;
            ++i;
        }
    }
    return nullptr;
}

namespace
{
/* Whether TUNED_CONFIGS_F32 has a class for (arch, kind). */
bool has_f32_classes(const char* arch, const char* kind)
{
    for(const tuned_row& r : k_tuned_f32)
    {
        if(str_eq(r.arch, arch) && str_eq(r.kind, kind))
            return true;
    }
    return false;
}
} // namespace

bool rocke_gett_f32_tuned_config(const char* arch,
                                 const char* kind,
                                 const char* cls,
                                 rocke_gett_f64_knobs_t* out)
{
    if(!has_f32_classes(arch, kind))
        return rocke_gett_f64_tuned_config(arch, kind, cls, out);
    for(const tuned_row& r : k_tuned_f32)
    {
        if(str_eq(r.arch, arch) && str_eq(r.kind, kind) && str_eq(r.cls, cls))
        {
            if(out != nullptr)
                *out = r.knobs;
            return true;
        }
    }
    return false;
}

const char* rocke_gett_f32_tuned_class_at(const char* arch, const char* kind, int index)
{
    if(!has_f32_classes(arch, kind))
        return rocke_gett_f64_tuned_class_at(arch, kind, index);
    int i = 0;
    for(const tuned_row& r : k_tuned_f32)
    {
        if(str_eq(r.arch, arch) && str_eq(r.kind, kind))
        {
            if(i == index)
                return r.cls;
            ++i;
        }
    }
    return nullptr;
}

rocke_gett_f64_spec_t rocke_gett_f64_with_k_block(const rocke_gett_f64_spec_t* spec,
                                                  const char* arch)
{
    rocke_gett_f64_spec_t cand = *spec;
    int kb = ROCKE_GETT_F64_K_BLOCK_ELEMS / spec->tile_k;
    if(kb < 1)
        kb = 1;
    if(spec->unroll2 && kb % 2)
        kb += 1;
    cand.k_block = kb;
    return rocke_gett_f64_is_valid_spec(&cand, arch, nullptr, 0) ? cand : *spec;
}

} /* extern "C" */
