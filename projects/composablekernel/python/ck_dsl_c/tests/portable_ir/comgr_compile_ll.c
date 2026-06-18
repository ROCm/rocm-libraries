/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * comgr_compile_ll.c -- compile an AMDGPU LLVM IR (.ll) file to a HSACO with
 * libamd_comgr, writing the HSACO bytes to an output file. Used by
 * run_recipe_demo.sh to compare the recipe-VM kernel against the Python
 * reference kernel by HSACO bytes (SSA value-name differences in the .ll do not
 * affect the compiled object).
 *
 *   usage: comgr_compile_ll <in.ll> <out.hsaco> [arch=gfx950]
 */
#include <amd_comgr/amd_comgr.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* read_file(const char* path, size_t* out_n)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n);
    *out_n = fread(buf, 1, (size_t)n, f);
    fclose(f);
    return buf;
}

#define CK(call, where)                                                  \
    do {                                                                 \
        amd_comgr_status_t _s = (call);                                  \
        if (_s != AMD_COMGR_STATUS_SUCCESS) {                            \
            fprintf(stderr, "comgr %s failed (%d)\n", where, (int)_s);   \
            return 1;                                                    \
        }                                                                \
    } while (0)

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.ll> <out.hsaco> [arch]\n", argv[0]);
        return 2;
    }
    const char* in = argv[1];
    const char* out = argv[2];
    const char* arch = argc >= 4 ? argv[3] : "gfx950";
    char isa[128];
    snprintf(isa, sizeof isa, "amdgcn-amd-amdhsa--%s", arch);

    size_t n = 0;
    char* ll = read_file(in, &n);
    if (!ll)
        return 1;

    amd_comgr_data_set_t in_set;
    CK(amd_comgr_create_data_set(&in_set), "create in");
    amd_comgr_data_t src;
    CK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src), "create src");
    CK(amd_comgr_set_data(src, n, ll), "set src");
    CK(amd_comgr_set_data_name(src, "kernel.ll"), "name src");
    CK(amd_comgr_data_set_add(in_set, src), "add src");

    amd_comgr_action_info_t info;
    CK(amd_comgr_create_action_info(&info), "info");
    CK(amd_comgr_action_info_set_isa_name(info, isa), "isa");
    CK(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR), "lang");
    const char* opts[] = {"-O3"};
    CK(amd_comgr_action_info_set_option_list(info, opts, 1), "opts");

    amd_comgr_data_set_t bc, rel, exe;
    CK(amd_comgr_create_data_set(&bc), "bc");
    CK(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc), "tobc");
    CK(amd_comgr_create_data_set(&rel), "rel");
    CK(amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc, rel), "torel");
    CK(amd_comgr_create_data_set(&exe), "exe");
    CK(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, rel, exe), "toexe");

    size_t count = 0;
    CK(amd_comgr_action_data_count(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, &count), "count");
    if (!count) {
        fprintf(stderr, "no executable\n");
        return 1;
    }
    amd_comgr_data_t e;
    CK(amd_comgr_action_data_get_data(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &e), "gete");
    size_t sz = 0;
    CK(amd_comgr_get_data(e, &sz, NULL), "size");
    char* hsaco = (char*)malloc(sz);
    CK(amd_comgr_get_data(e, &sz, hsaco), "read");

    FILE* of = fopen(out, "wb");
    if (!of) {
        fprintf(stderr, "cannot write %s\n", out);
        return 1;
    }
    fwrite(hsaco, 1, sz, of);
    fclose(of);
    free(hsaco);
    free(ll);
    printf("%zu\n", sz);
    return 0;
}
