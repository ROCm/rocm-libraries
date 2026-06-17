/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * roundtrip.c -- portable-IR round-trip driver.
 *
 * Reads a portable CK-DSL IR JSON file (schema "ck.dsl.ir/v1", from
 * ck_dsl.core.ir_export), imports it via ckc_import_kernel_from_json, lowers it
 * to AMDGPU LLVM IR via ckc_lower_kernel_to_llvm_ex, and writes the .ll to
 * stdout.
 *
 *   usage: roundtrip <kernel.ir.json> [arch]
 *
 * The emitted .ll is byte-compared against the Python-lowered .ll (and the
 * C-native build) by run_portable_ir_parity.sh to prove the Python frontend ->
 * portable IR -> C backend boundary is lossless.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir_import.h"
#include "ckc/lower_llvm.h"

static char* read_file(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)n, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <kernel.ir.json> [arch]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const char* arch = argc >= 3 ? argv[2] : "gfx950";

    char* text = read_file(path);
    if (!text)
        return 1;

    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = '\0';
    ckc_status_t st = ckc_import_kernel_from_json(text, NULL, &b, &kernel, err, sizeof err);
    free(text);
    if (st != CKC_OK || !kernel) {
        fprintf(stderr, "import failed: status=%d err=%s\n", (int)st, err);
        return 1;
    }

    char* llvm_text = NULL;
    char lerr[CKC_ERR_MSG_CAP];
    lerr[0] = '\0';
    st = ckc_lower_kernel_to_llvm_ex(kernel, CKC_LLVM_FLAVOR_AUTO, arch, &llvm_text, lerr,
                                     sizeof lerr);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, lerr);
        ckc_ir_builder_free(&b);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_ir_builder_free(&b);
    return 0;
}
