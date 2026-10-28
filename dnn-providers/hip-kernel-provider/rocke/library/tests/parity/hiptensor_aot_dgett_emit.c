/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/hiptensor_aot_dgett_emit.c -- C-side emitter for the direct
 * GETT kernels of hipTensor's kernel pack. Enumerates the configs of
 * hiptensor_aot_dgett_emit.py in the same order and builds each kernel with
 * rocke_build_htaot_dgett (a descriptor the target cannot build is rejected).
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

const char* const ARCHES[3] = {"gfx950", "gfx942", "gfx90a"};

struct Cfg
{
    rocke_htaot_dgett_spec_t spec;
    const char* arch;
};

std::vector<Cfg> rows()
{
    std::vector<Cfg> out;
    for(const char* arch : ARCHES)
        for(bool cplx : {false, true})
        {
            std::vector<const char*> comps
                = cplx ? std::vector<const char*>{"f32", "f64"}
                       : std::vector<const char*>{"f16", "bf16", "f32", "f64"};
            for(const char* comp : comps)
                for(const char* a_fast : {"k", "r"})
                    for(const char* b_fast : {"k", "r"})
                        for(bool math : {false, true})
                        {
                            if(cplx && math)
                                continue;
                            for(bool idx64 : {false, true})
                            {
                                rocke_htaot_dgett_spec_t s = rocke_htaot_dgett_spec_default();
                                s.comp = comp;
                                s.cplx = cplx;
                                s.a_fast = a_fast;
                                s.b_fast = b_fast;
                                s.math = math;
                                s.idx64 = idx64;
                                out.push_back({s, arch});
                            }
                        }
        }
    return out;
}

const char* g_arch = "gfx950";

rocke_kernel_def_t* build(rocke_ir_builder_t* b, int idx, int* known)
{
    std::vector<Cfg> r = rows();
    if(idx < 0 || idx >= (int)r.size())
        return nullptr;
    *known = 1;
    g_arch = r[idx].arch;
    return rocke_build_htaot_dgett(b, &r[idx].spec, r[idx].arch);
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: hiptensor_aot_dgett_emit <config_index> [ll|ir|verify]\n");
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
    (void)arch;
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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, g_arch, &llvm_text, err, sizeof err);
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
