/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
#include "rocke/instance_f64_mma_probe.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    int idx = argc > 1 ? atoi(argv[1]) : -1;
    /* (m, k, arch): the gfx942 rows first, then the same atoms on gfx90a / gfx950. */
    const struct
    {
        int m;
        int k;
        const char* arch;
    } configs[] = {{16, 4, "gfx942"},
                   {16, 8, "gfx942"},
                   {16, 64, "gfx942"},
                   {4, 4, "gfx942"},
                   {4, 8, "gfx942"},
                   {4, 64, "gfx942"},
                   {16, 8, "gfx90a"},
                   {4, 8, "gfx90a"},
                   {16, 8, "gfx950"},
                   {4, 8, "gfx950"}};
    if(idx < 0 || idx >= (int)(sizeof(configs) / sizeof(configs[0])))
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const char* mode = argc > 2 ? argv[2] : "ll";
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel
        = rocke_build_f64_mma_probe_arch(&b, configs[idx].m, configs[idx].k, configs[idx].arch);
    if(!kernel)
    {
        fprintf(stderr, "%s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }
    char* text = NULL;
    rocke_status_t status = ROCKE_OK;
    if(strcmp(mode, "ll") == 0)
        status
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, configs[idx].arch, &text);
    else if(strcmp(mode, "ir") == 0)
        status = rocke_ir_serialize(kernel, &text);
    else if(strcmp(mode, "hip") == 0)
    {
        rocke_strbuf_t buf;
        rocke_strbuf_init(&buf, 0);
        rocke_lower_hip_opts_t opts = {0};
        opts.arch = configs[idx].arch;
        status = rocke_lower_kernel_to_hip(&b, kernel, &opts, &buf);
        if(status == ROCKE_OK)
            fputs(buf.data, stdout);
        rocke_strbuf_free(&buf);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diagnostics = NULL;
        size_t count = 0;
        rocke_verify(kernel, &diagnostics, &count);
        for(size_t i = 0; i < count; ++i)
        {
            char* line = rocke_diag_to_string(&diagnostics[i]);
            if(line)
            {
                puts(line);
                free(line);
            }
        }
        rocke_diags_free(diagnostics, count);
    }
    else
        status = ROCKE_ERR_VALUE;
    if(text)
    {
        fputs(text, stdout);
        free(text);
    }
    if(status != ROCKE_OK)
        fprintf(stderr, "emit failed: %d %s\n", status, b.err);
    rocke_ir_builder_free(&b);
    return status != ROCKE_OK;
}
