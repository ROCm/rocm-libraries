/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/hiptensor_aot_ewp_emit.c -- C-side emitter for the tuned tiled elementwise
 * kernels of hipTensor's kernel pack. Enumerates the configs of hiptensor_aot_ewp_emit.py in the
 * same order (_hiptensor_aot_rows.ewp_rows: the knob coverage set EWP_COVER, then the classes of
 * the gfx950 and gfx942 packs not listed before) and builds each kernel with
 * rocke/instance_hiptensor_aot.h.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <set>
#include <string>
#include <vector>

#include "rocke/instance_hiptensor_aot.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

namespace
{

rocke_htaot_ewp_spec_t spec(int arity,
                            const char* a,
                            const char* b,
                            const char* c,
                            const char* d,
                            const char* kinds,
                            const char* mode,
                            int to,
                            int tt,
                            int tw,
                            int block,
                            int nt,
                            int swz,
                            int swsh,
                            const char* walk,
                            int tpc,
                            int ilp)
{
    rocke_htaot_ewp_spec_t s = rocke_htaot_ewp_spec_default();
    s.arity = arity;
    s.a = a;
    s.b = b;
    s.c = c;
    s.d = d;
    s.kinds = kinds;
    s.mode = mode;
    s.to = to;
    s.tt = tt;
    s.tw = tw;
    s.block = block;
    s.nt = nt;
    s.swz = swz;
    s.swsh = swsh;
    s.walk = walk;
    s.tpc = tpc;
    s.ilp = ilp;
    return s;
}

rocke_htaot_ewp_spec_t with_vb(rocke_htaot_ewp_spec_t s, int vb, int rag, int tb = 0)
{
    s.vb = vb;
    s.rag = rag;
    s.tb = tb;
    return s;
}

/* _hiptensor_aot_rows.EWP_COVER */
std::vector<rocke_htaot_ewp_spec_t> cover()
{
    return {
        spec(1, "f16", nullptr, nullptr, "f16", "T", "t", 64, 64, 1, 256, 3, 16, 6, "o", 1, 0),
        spec(1, "f32", nullptr, nullptr, "f32", "T", "t", 32, 32, 1, 128, 0, 0, 0, "o", 1, 0),
        spec(1, "f64", nullptr, nullptr, "f64", "T", "t", 16, 16, 1, 64, 1, 0, 0, "t", 1, 0),
        spec(1, "f16", nullptr, nullptr, "f16", "T", "a", 8, 8, 8, 256, 3, 2, 6, "o", 1, 0),
        spec(2, "f16", nullptr, "f16", "f16", "TV", "t", 64, 64, 1, 128, 3, 16, 6, "t", 2, 1),
        spec(2, "bf16", nullptr, "bf16", "bf16", "VT", "t", 32, 64, 1, 64, 2, 0, 0, "o", 1, 2),
        spec(2, "f32", nullptr, "f32", "f32", "TT", "t", 32, 32, 1, 256, 0, 0, 0, "o", 4, 0),
        spec(2, "f64", nullptr, "f64", "f64", "TV", "a", 4, 8, 1, 64, 0, 0, 0, "o", 1, 0),
        spec(3, "f16", "f16", "f16", "f16", "TTV", "t", 64, 64, 1, 128, 3, 16, 6, "o", 1, 4),
        spec(3, "f32", "f32", "f32", "f32", "VVT", "t", 16, 32, 1, 64, 0, 0, 0, "o", 1, 0),
        spec(3, "f32", "f32", "f32", "f32", "TVT", "a", 2, 4, 4, 128, 3, 0, 0, "o", 8, 8),
        spec(2, "bf16", nullptr, "bf16", "bf16", "TV", "t", 256, 64, 1, 256, 3, 16, 8, "o", 1, 4),
        /* vector widths below 16 bytes and ragged classes (vb / rag) */
        with_vb(
            spec(2, "f16", nullptr, "f16", "f16", "TV", "t", 16, 16, 1, 64, 0, 16, 4, "t", 2, 1),
            4,
            1),
        with_vb(
            spec(3, "f32", "f32", "f32", "f32", "TVT", "a", 2, 4, 8, 64, 1, 0, 0, "t", 2, 0), 8, 1),
        with_vb(
            spec(1, "f64", nullptr, nullptr, "f64", "T", "t", 16, 16, 1, 64, 2, 16, 4, "o", 1, 0),
            8,
            1),
        with_vb(
            spec(2, "bf16", nullptr, "bf16", "bf16", "VT", "t", 32, 32, 1, 64, 3, 32, 5, "o", 1, 2),
            8,
            0),
        with_vb(spec(2, "f16", nullptr, "f16", "f16", "TT", "a", 4, 4, 16, 64, 2, 4, 6, "o", 1, 0),
                2,
                1),
        with_vb(spec(3, "f16", "f16", "f16", "f16", "TVT", "t", 16, 32, 1, 64, 2, 0, 0, "t", 2, 1),
                2,
                1,
                8),
        with_vb(spec(2, "f32", nullptr, "f32", "f32", "TT", "t", 16, 16, 1, 64, 3, 0, 0, "o", 1, 2),
                8,
                0,
                16),
    };
}

/* _hiptensor_aot_rows.ewp_rows */
std::vector<rocke_htaot_ewp_spec_t> rows()
{
    std::vector<rocke_htaot_ewp_spec_t> out;
    std::set<std::string> seen;
    auto add = [&](const rocke_htaot_ewp_spec_t& s) {
        char name[256] = {0};
        if(rocke_htaot_ewp_kernel_name(&s, name, sizeof name) != ROCKE_OK)
            return;
        if(seen.insert(name).second)
            out.push_back(s);
    };
    for(const auto& s : cover())
        add(s);
    for(const char* arch : {"gfx950", "gfx942"})
    {
        int n = rocke_htaot_ewp_pack_classes(arch, nullptr, 0);
        std::vector<rocke_htaot_ewp_spec_t> cls((size_t)(n > 0 ? n : 1));
        rocke_htaot_ewp_pack_classes(arch, cls.data(), n);
        for(int i = 0; i < n; ++i)
            add(cls[(size_t)i]);
    }
    return out;
}

rocke_kernel_def_t* build(rocke_ir_builder_t* b, int idx, int* known)
{
    static const std::vector<rocke_htaot_ewp_spec_t> all = rows();
    if(idx < 0 || idx >= (int)all.size())
        return nullptr;
    *known = 1;
    return rocke_build_htaot_ewp(b, &all[(size_t)idx]);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: hiptensor_aot_ewp_emit <config_index> [ll|ir|verify]\n");
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
