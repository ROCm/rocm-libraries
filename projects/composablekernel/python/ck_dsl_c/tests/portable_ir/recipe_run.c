/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * recipe_run.c -- run a builder recipe (ck.dsl.recipe/v1) through the C VM with
 * a runtime spec, lower the emitted kernel to AMDGPU LLVM IR, and print the .ll.
 *
 *   usage: recipe_run <recipe.json> [--arch gfx950] [--int K=V]... [--str K=V]...
 *
 * One recipe + a runtime spec produces the specialized kernel with no CPython.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/lower_llvm.h"
#include "ckc/recipe_vm.h"

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
    char* buf = (char*)malloc((size_t)n + 1);
    size_t rd = fread(buf, 1, (size_t)n, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

/* Split "K=V" in place: returns key, *val points past '='. */
static char* split_kv(char* s, char** val)
{
    char* eq = strchr(s, '=');
    if (!eq) {
        *val = NULL;
        return s;
    }
    *eq = '\0';
    *val = eq + 1;
    return s;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <recipe.json> [--arch A] [--int K=V]... [--str K=V]...\n",
                argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const char* arch = "gfx950";

    ckc_recipe_spec_int_t ints[32];
    ckc_recipe_spec_str_t strs[32];
    int n_ints = 0, n_strs = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--arch") == 0 && i + 1 < argc) {
            arch = argv[++i];
        } else if (strcmp(argv[i], "--int") == 0 && i + 1 < argc && n_ints < 32) {
            char* val;
            char* key = split_kv(argv[++i], &val);
            ints[n_ints].name = key;
            ints[n_ints].value = val ? atol(val) : 0;
            n_ints++;
        } else if (strcmp(argv[i], "--str") == 0 && i + 1 < argc && n_strs < 32) {
            char* val;
            char* key = split_kv(argv[++i], &val);
            strs[n_strs].name = key;
            strs[n_strs].value = val ? val : "";
            n_strs++;
        }
    }

    char* text = read_file(path);
    if (!text)
        return 1;

    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = '\0';
    ckc_status_t st =
        ckc_recipe_run_from_json(text, ints, n_ints, strs, n_strs, &b, &kernel, err, sizeof err);
    free(text);
    if (st != CKC_OK || !kernel) {
        fprintf(stderr, "recipe run failed: %s\n", err);
        return 1;
    }

    char* ll = NULL;
    char lerr[CKC_ERR_MSG_CAP];
    lerr[0] = '\0';
    st = ckc_lower_kernel_to_llvm_ex(kernel, CKC_LLVM_FLAVOR_AUTO, arch, &ll, lerr, sizeof lerr);
    if (st != CKC_OK || !ll) {
        fprintf(stderr, "lower failed: %s\n", lerr);
        ckc_ir_builder_free(&b);
        return 1;
    }
    fputs(ll, stdout);
    free(ll);
    ckc_ir_builder_free(&b);
    return 0;
}
