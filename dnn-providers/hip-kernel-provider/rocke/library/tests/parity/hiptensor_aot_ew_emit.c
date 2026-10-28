/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/hiptensor_aot_ew_emit.c -- C-side emitter for the shape-generic elementwise kernel of hipTensor's kernel pack.
 * Enumerates the configs of hiptensor_aot_ew_emit.py in the same order (the rows of
 * _hiptensor_aot_rows.py) and builds each kernel with
 * rocke/instance_hiptensor_aot.h.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/instance_hiptensor_aot.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

namespace
{

/* ---- the pack rows (_hiptensor_aot_rows.py) ---- */

const char* const REAL[4] = {"f16", "bf16", "f32", "f64"};

struct Row
{
    int arity;
    const char* a;
    const char* b;
    const char* c;
    const char* d;
    bool cplx;
    bool idx64;
    bool math;
};

const char* const R1[11][2] = {{"f16", "f16"},
                               {"f16", "f32"},
                               {"bf16", "bf16"},
                               {"bf16", "f32"},
                               {"f32", "f32"},
                               {"f32", "f16"},
                               {"f32", "bf16"},
                               {"f64", "f64"},
                               {"f16", "f64"},
                               {"bf16", "f64"},
                               {"f32", "f64"}};
const char* const R2[11][3] = {{"f16", "f16", "f16"},
                               {"f16", "f16", "f32"},
                               {"bf16", "bf16", "bf16"},
                               {"bf16", "bf16", "f32"},
                               {"f32", "f32", "f32"},
                               {"f32", "f16", "f16"},
                               {"f32", "bf16", "bf16"},
                               {"f64", "f64", "f64"},
                               {"f64", "f16", "f16"},
                               {"f64", "bf16", "bf16"},
                               {"f64", "f32", "f32"}};
const char* const R3[6][4] = {{"f16", "f16", "f16", "f16"},
                              {"f16", "f16", "f16", "f32"},
                              {"bf16", "bf16", "bf16", "bf16"},
                              {"bf16", "bf16", "bf16", "f32"},
                              {"f32", "f32", "f32", "f32"},
                              {"f64", "f64", "f64", "f64"}};
const char* const RC1[3][2] = {{"f32", "f32"}, {"f64", "f64"}, {"f32", "f64"}};
const char* const RC2[3][3] = {{"f32", "f32", "f32"}, {"f64", "f64", "f64"}, {"f64", "f32", "f32"}};
const char* const RED[4][2] = {{"f16", "f16"}, {"bf16", "bf16"}, {"f32", "f32"}, {"f64", "f64"}};

bool eq(const char* a, const char* b)
{
    return strcmp(a, b) == 0;
}

bool in1(const char* a, const char* d)
{
    for(const auto& r : R1)
        if(eq(r[0], a) && eq(r[1], d))
            return true;
    return false;
}

bool in2(const char* t, const char* c, const char* d)
{
    for(const auto& r : R2)
        if(eq(r[0], t) && eq(r[1], c) && eq(r[2], d))
            return true;
    return false;
}

bool inc2(const char* t, const char* c, const char* d)
{
    for(const auto& r : RC2)
        if(eq(r[0], t) && eq(r[1], c) && eq(r[2], d))
            return true;
    return false;
}

std::vector<Row> ew_params(bool idx64, bool mixed)
{
    std::vector<Row> out;
    for(const auto& r : R1)
        out.push_back({1, r[0], nullptr, nullptr, r[1], false, idx64, false});
    if(mixed)
        for(const char* d : {"f16", "bf16"})
            for(const char* a : REAL)
                if(!in1(a, d))
                    out.push_back({1, a, nullptr, nullptr, d, false, idx64, false});
    for(const auto& r : R2)
        out.push_back({2, r[0], nullptr, r[1], r[2], false, idx64, false});
    if(mixed)
        for(const char* t : {"f32", "f64"})
            for(const char* c : REAL)
                for(const char* d : REAL)
                    if(!in2(t, c, d))
                        out.push_back({2, t, nullptr, c, d, false, idx64, false});
    for(const auto& r : R3)
        out.push_back({3, r[0], r[1], r[2], r[3], false, idx64, false});
    for(const auto& r : RC1)
        out.push_back({1, r[0], nullptr, nullptr, r[1], true, idx64, false});
    for(const auto& r : RC2)
        out.push_back({2, r[0], nullptr, r[1], r[2], true, idx64, false});
    if(mixed)
        for(const char* t : {"f32", "f64"})
            for(const char* c : {"f32", "f64"})
                for(const char* d : {"f32", "f64"})
                    if(!inc2(t, c, d))
                        out.push_back({2, t, nullptr, c, d, true, idx64, false});
    size_t n = out.size();
    for(size_t i = 0; i < n; ++i)
        if(!out[i].cplx)
        {
            Row m = out[i];
            m.math = true;
            out.push_back(m);
        }
    return out;
}

std::vector<Row> plain_rows()
{
    std::vector<Row> out;
    for(const Row& r : ew_params(false, false))
        if(!r.cplx && !r.math)
            out.push_back(r);
    return out;
}

rocke_htaot_ew_spec_t to_spec(const Row& r)
{
    rocke_htaot_ew_spec_t s = rocke_htaot_ew_spec_default();
    s.arity = r.arity;
    s.a = r.a;
    s.b = r.b;
    s.c = r.c;
    s.d = r.d;
    s.cplx = r.cplx;
    s.idx64 = r.idx64;
    s.math = r.math;
    return s;
}

/* jit_ew.jit_classes([base]) */
std::vector<rocke_htaot_ew_jit_spec_t> jit_classes(const Row& base)
{
    std::vector<rocke_htaot_ew_jit_spec_t> out;
    rocke_htaot_ew_jit_spec_t j;
    memset(&j, 0, sizeof j);
    j.base = to_spec(base);
    int n_in = base.arity;
    int v = rocke_htaot_ew_jit_vec(base.d);
    if(base.math)
    {
        j.vec = v;
        j.n_unit = n_in;
        for(int i = 0; i < n_in; ++i)
            j.unit[i] = true;
        out.push_back(j);
        return out;
    }
    j.vec = 1;
    out.push_back(j);
    if(base.cplx)
        return out;
    std::vector<int> masks;
    if(base.idx64)
        masks.push_back((1 << n_in) - 1);
    else
        for(int m = 0; m < (1 << n_in); ++m)
            masks.push_back(m);
    for(int mask : masks)
    {
        rocke_htaot_ew_jit_spec_t k = j;
        k.vec = v;
        k.n_unit = n_in;
        for(int i = 0; i < n_in; ++i)
            k.unit[i] = ((mask >> i) & 1) != 0;
        out.push_back(k);
    }
    return out;
}

/* _hiptensor_aot_rows.geometry: a 64-bit LCG seeded by the config index */
std::vector<int64_t> geometry(int n, int seed)
{
    const uint64_t A = 6364136223846793005ULL, C = 1442695040888963407ULL;
    uint64_t state = (uint64_t)seed * A + C;
    std::vector<int64_t> out;
    for(int i = 0; i < n; ++i)
    {
        state = state * A + C;
        out.push_back(1000 + (int64_t)((state >> 33) % (((uint64_t)1 << 30) - 1000)));
    }
    return out;
}

rocke_kernel_def_t* build(rocke_ir_builder_t* b, int idx, int* known)
{
    std::vector<Row> rows = ew_params(false, true);
    std::vector<Row> r64 = ew_params(true, true);
    rows.insert(rows.end(), r64.begin(), r64.end());
    if(idx < 0 || idx >= (int)rows.size())
        return nullptr;
    *known = 1;
    rocke_htaot_ew_spec_t s = to_spec(rows[idx]);
    return rocke_build_htaot_ew(b, &s);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: hiptensor_aot_ew_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    const char* arch = "gfx950";
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    int known = 0;
    rocke_kernel_def_t* kernel = build(&b, idx, &known);
    if(!known)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    if(kernel == NULL)
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    int rc = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rc = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rc = 1;
        }
        else
        {
            fputs(text, stdout);
            free(text);
        }
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return rc;
}
