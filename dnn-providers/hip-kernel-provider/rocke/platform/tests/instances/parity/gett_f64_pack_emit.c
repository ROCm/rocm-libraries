/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/gett_f64_pack_emit.c -- C-side emitter for the f64 and f32 GETT
 * kernels of hipTensor's kernel pack. Resolves every spec of gett_f64_pack_emit.py
 * (same order) through the C spec policy (rocke_gett_f64_gemm_problem,
 * rocke_gett_f64_arch_spec, rocke_gett_f64_tuned_config (f32 words:
 * rocke_gett_f32_tuned_config),
 * rocke_gett_f64_runtime_template, rocke_gett_f64_with_k_block) and builds it
 * with rocke_build_gett_f64, so run_diff.py byte-compares policy and builder.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gett_f64.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static const char* const k_arches[] = {"gfx950", "gfx942", "gfx90a"};
static const char* const k_dclass = "mnx";

/* The idx-th pack descriptor: arch, complex, class, D layout, bilinear,
 * k_block, f32 words. The f64 kernels on gfx950 / gfx942 / gfx90a, then the
 * f32 kernels on gfx950 / gfx942. Returns 0 past the end. */
static int entry(int idx,
                 const char** arch,
                 int* cplx,
                 const char** cls,
                 char* dclass,
                 int* bilinear,
                 int* kblock,
                 int* f32)
{
    int i = 0;
    for(int w = 0; w < 2; ++w)
    {
        for(int a = 0; a < (w ? 2 : 3); ++a)
        {
            for(int c = 0; c < 2; ++c)
            {
                const char* kind = c ? "complex" : "real";
                for(int ci = 0;; ++ci)
                {
                    const char* name = w ? rocke_gett_f32_tuned_class_at(k_arches[a], kind, ci)
                                         : rocke_gett_f64_tuned_class_at(k_arches[a], kind, ci);
                    if(name == NULL)
                        name = "default";
                    for(int d = 0; d < 3; ++d)
                    {
                        for(int bl = 0; bl < 2; ++bl)
                        {
                            for(int kb = 0; kb < 2; ++kb)
                            {
                                if(i++ == idx)
                                {
                                    *arch = k_arches[a];
                                    *cplx = c;
                                    *cls = name;
                                    *dclass = k_dclass[d];
                                    *bilinear = bl;
                                    *kblock = kb;
                                    *f32 = w;
                                    return 1;
                                }
                            }
                        }
                    }
                    if(strcmp(name, "default") == 0)
                        break;
                }
            }
        }
    }
    return 0;
}

/* The spec of one pack descriptor; 0 when it does not apply. */
static int pack_spec(const char* arch,
                     int cplx,
                     const char* cls,
                     char dclass,
                     int bilinear,
                     int kblock,
                     int f32,
                     rocke_gett_f64_spec_t* out)
{
    const int64_t M = 64, N = 64, K = 64, L = 2;
    int64_t a_strides[2] = {K, 1};
    int64_t b_strides[2] = {K, 1};
    int64_t d_strides[2];
    if(dclass == 'm')
    {
        d_strides[0] = 1;
        d_strides[1] = M;
    }
    else if(dclass == 'n')
    {
        d_strides[0] = N;
        d_strides[1] = 1;
    }
    else
    {
        d_strides[0] = 2;
        d_strides[1] = 2 * M;
    }
    int64_t batch_strides[3] = {M * K, N * K, 4 * M * N};
    rocke_gett_f64_problem_t prob = rocke_gett_f64_gemm_problem(M,
                                                                N,
                                                                K,
                                                                a_strides,
                                                                b_strides,
                                                                d_strides,
                                                                NULL,
                                                                L,
                                                                batch_strides,
                                                                NULL,
                                                                cplx != 0,
                                                                false,
                                                                false);
    prob.f32 = f32 != 0;
    rocke_gett_f64_knobs_t knobs;
    memset(&knobs, 0, sizeof knobs);
    const char* kind = cplx ? "complex" : "real";
    if(strcmp(cls, "default") != 0
       && !(f32 ? rocke_gett_f32_tuned_config(arch, kind, cls, &knobs)
                : rocke_gett_f64_tuned_config(arch, kind, cls, &knobs)))
        return 0;
    knobs.set |= ROCKE_GETT_F64_KNOB_BILINEAR | ROCKE_GETT_F64_KNOB_RUNTIME_SIZES;
    knobs.bilinear = bilinear != 0;
    knobs.runtime_sizes = true;
    rocke_gett_f64_spec_t spec = rocke_gett_f64_arch_spec(&prob, arch, &knobs);
    spec.split_k = 1;
    rocke_gett_f64_problem_t tmpl;
    if(!rocke_gett_f64_runtime_template(&prob, &tmpl))
        return 0;
    spec.problem = tmpl;
    if(kblock)
    {
        rocke_gett_f64_spec_t kb = rocke_gett_f64_with_k_block(&spec, arch);
        if(kb.k_block == spec.k_block)
            return 0;
        spec = kb;
    }
    if(!rocke_gett_f64_is_valid_spec(&spec, arch, NULL, 0))
        return 0;
    *out = spec;
    return 1;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: gett_f64_pack_emit <config_index>\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    const char* arch;
    int cplx, bilinear, kblock, f32;
    const char* cls;
    char dclass;
    if(!entry(idx, &arch, &cplx, &cls, &dclass, &bilinear, &kblock, &f32))
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    rocke_gett_f64_spec_t spec;
    if(!pack_spec(arch, cplx, cls, dclass, bilinear, kblock, f32, &spec))
    {
        fprintf(stderr, "config does not apply\n");
        return 1;
    }
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_gett_f64(&b, &spec, arch);
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
