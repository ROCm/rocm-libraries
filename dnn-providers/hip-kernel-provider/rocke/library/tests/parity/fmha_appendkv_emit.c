/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_appendkv_emit.c -- C-side emitter for the fmha_appendkv
 * parity harness. Selects one of 6 FmhaAppendKvSpec configs by argv[1] (0..5),
 * builds rocke_fmha_appendkv_spec_t identically to the Python emitter
 * fmha_appendkv_emit.py, builds the kernel via rocke_build_fmha_fwd_appendkv and
 * lowers via rocke_lower_kernel_to_llvm_ex (arch gfx950, flavor AUTO), printing
 * the .ll to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.rotary.h"
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/instance_fmha_appendkv.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_fmha_appendkv_spec_t* spec)
{
    rocke_fmha_shape_t shape;
    rocke_fmha_common_spec_t common;
    rocke_rotary_spec_t rot;

    switch(idx)
    {
    case 0:
    { /* H128 q4 kv2 f16, batch1, no rope, b256 */
        shape = rocke_fmha_shape_default(128, 4, 2);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        *spec = rocke_fmha_appendkv_spec_default(common, 1);
        spec->has_rotary = false;
        spec->block_size = 256;
        break;
    }
    case 1:
    { /* H128 q4 kv2 f16, batch2, rope half H128, b256 */
        shape = rocke_fmha_shape_default(128, 4, 2);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        *spec = rocke_fmha_appendkv_spec_default(common, 2);
        if(rocke_rotary_spec_init(&rot, 128, ROCKE_ROTARY_HALF, 0) != ROCKE_OK)
            return -1;
        spec->has_rotary = true;
        spec->rotary = rot;
        spec->block_size = 256;
        break;
    }
    case 2:
    { /* H128 q8 kv4 bsq16 bsk64 bf16, batch2, rope interleaved H128, b128 */
        shape = rocke_fmha_shape_make(128, 8, 4, 16, 64);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        *spec = rocke_fmha_appendkv_spec_default(common, 2);
        if(rocke_rotary_spec_init(&rot, 128, ROCKE_ROTARY_INTERLEAVED, 0) != ROCKE_OK)
            return -1;
        spec->has_rotary = true;
        spec->rotary = rot;
        spec->block_size = 128;
        break;
    }
    case 3:
    { /* H64 q4 kv2 f16, batch1, rope half H64, b256 */
        shape = rocke_fmha_shape_default(64, 4, 2);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        *spec = rocke_fmha_appendkv_spec_default(common, 1);
        if(rocke_rotary_spec_init(&rot, 64, ROCKE_ROTARY_HALF, 0) != ROCKE_OK)
            return -1;
        spec->has_rotary = true;
        spec->rotary = rot;
        spec->block_size = 256;
        break;
    }
    case 4:
    { /* H32 q8 kv8 bf16, batch2, no rope, b256 */
        shape = rocke_fmha_shape_default(32, 8, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        *spec = rocke_fmha_appendkv_spec_default(common, 2);
        spec->has_rotary = false;
        spec->block_size = 256;
        break;
    }
    case 5:
    { /* H192 q4 kv2 f16, batch1, rope half H192, b256 */
        shape = rocke_fmha_shape_default(192, 4, 2);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        *spec = rocke_fmha_appendkv_spec_default(common, 1);
        if(rocke_rotary_spec_init(&rot, 192, ROCKE_ROTARY_HALF, 0) != ROCKE_OK)
            return -1;
        spec->has_rotary = true;
        spec->rotary = rot;
        spec->block_size = 256;
        break;
    }
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_fmha_appendkv_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    /* Build via the requested entry: rocke_build_fmha_fwd_appendkv(b, spec, arch),
     * with the builder initialised to spec.kernel_name() (the convenience
     * wrapper does exactly this). */
    rocke_ir_builder_t b;
    char namebuf[256];
    if(rocke_fmha_appendkv_kernel_name(&spec, namebuf, sizeof namebuf) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }
    if(rocke_ir_builder_init(&b, namebuf) != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_build_fmha_fwd_appendkv(&b, &spec, "gfx950");
    if(kernel == NULL)
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
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
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
