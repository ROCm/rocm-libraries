/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_dgrad_box_det_reduce_emit.c -- C-side emitter for the
 * dgrad pp_box_det Stage 2 (conv_dgrad_box_det_reduce) parity harness.
 * Selects one of N sampled spec configs by argv[1], builds the spec as the
 * Python emitter conv_dgrad_box_det_reduce_emit.py does, builds the kernel
 * via rocke_build_dgrad_box_det_reduce_new and lowers it (flavor AUTO).
 *
 * Config index table (must stay in sync with the Python emitter):
 *   0  t128x256 w2x4, f32 slots, msplit 1
 *   1  t256x256 w4x2, msplit 4, pix_pre
 *   2  t256x256 w4x2, bf16 slots, msplit 2, pix_pre
 *   3  t256x256 w4x2, m_chunks 2
 *   4  t128x256 w2x4, wave_wg, unroll 2, load_aux 2
 *   5  t128x256 w2x4, msplit 4, pix_pre, bf16 slots
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_dgrad_box_det.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static int make_cfg(int idx, rocke_dgrad_box_det_reduce_spec_t* spec, const char** arch)
{
    *spec = rocke_dgrad_box_det_reduce_spec_default();
    *arch = "gfx942";
    if(idx < 0 || idx > 5)
        return -1;
    const bool t256 = idx >= 1 && idx <= 3;
    spec->tile_m = t256 ? 256 : 128;
    spec->tile_n = 256;
    spec->warp_m = t256 ? 4 : 2;
    spec->warp_n = t256 ? 2 : 4;
    if(idx == 1 || idx == 5)
        spec->msplit = 4;
    if(idx == 2)
        spec->msplit = 2;
    if(idx == 1 || idx == 2 || idx == 5)
        spec->pix_pre = true;
    if(idx == 2 || idx == 5)
        spec->slot_bytes = 2;
    if(idx == 3)
        spec->m_chunks = 2;
    if(idx == 4)
    {
        spec->wave_wg = true;
        spec->unroll = 2;
        spec->load_aux = 2;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_dgrad_box_det_reduce_spec_t spec;
    const char* arch = "gfx942";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_dgrad_box_det_reduce_new(&b, &spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* ir_text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &ir_text);
        if(st != ROCKE_OK || !ir_text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(ir_text, stdout);
        free(ir_text);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t ndiag = 0;
        rocke_verify(kernel, &diags, &ndiag);
        for(size_t i = 0; i < ndiag; i++)
        {
            char* s = rocke_diag_to_string(&diags[i]);
            if(s)
            {
                fputs(s, stdout);
                fputc('\n', stdout);
                free(s);
            }
        }
        rocke_diags_free(diags, ndiag);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s' (expected ll|ir|verify)\n", mode);
        ret = 2;
    }

    rocke_ir_builder_free(&b);
    return ret;
}
