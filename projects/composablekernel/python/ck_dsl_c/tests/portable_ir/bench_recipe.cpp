// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// bench_recipe.cpp -- time the record+roll JIT path: parse+expand a parametric
// recipe (ckc_recipe_run_from_json) for a given spec D, lower to .ll, and comgr
// to a gfx950 HSACO. Median of N (warm-up discarded). Pairs with bench_compare
// (native C + portable-IR) for the 3-way compile-time comparison.
//
//   usage: bench_recipe <recipe.json> <D> [dtype=fp16] [arch=gfx950] [iters=10]
#include <amd_comgr/amd_comgr.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "ckc/lower_llvm.h"
#include "ckc/recipe_vm.h"
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}
static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}
static std::string read_file(const char* p)
{
    FILE* f = fopen(p, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", p);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s((size_t)n, 0);
    size_t r = fread(&s[0], 1, (size_t)n, f);
    s.resize(r);
    fclose(f);
    return s;
}
static size_t comgr_compile(const std::string& ll, const std::string& isa)
{
    auto ck = [](amd_comgr_status_t s, const char* w) {
        if (s != AMD_COMGR_STATUS_SUCCESS) {
            fprintf(stderr, "comgr %s (%d)\n", w, (int)s);
            exit(1);
        }
    };
    amd_comgr_data_set_t in{};
    ck(amd_comgr_create_data_set(&in), "in");
    amd_comgr_data_t src{};
    ck(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src), "src");
    ck(amd_comgr_set_data(src, ll.size(), ll.data()), "set");
    ck(amd_comgr_set_data_name(src, "kernel.ll"), "nm");
    ck(amd_comgr_data_set_add(in, src), "add");
    amd_comgr_action_info_t info{};
    ck(amd_comgr_create_action_info(&info), "info");
    ck(amd_comgr_action_info_set_isa_name(info, isa.c_str()), "isa");
    ck(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR), "lang");
    const char* o[] = {"-O3"};
    ck(amd_comgr_action_info_set_option_list(info, o, 1), "opt");
    amd_comgr_data_set_t bc{}, rel{}, exe{};
    ck(amd_comgr_create_data_set(&bc), "bc");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in, bc), "bc");
    ck(amd_comgr_create_data_set(&rel), "rel");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc, rel), "rel");
    ck(amd_comgr_create_data_set(&exe), "exe");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, rel, exe), "exe");
    size_t c = 0;
    ck(amd_comgr_action_data_count(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, &c), "cnt");
    amd_comgr_data_t e{};
    ck(amd_comgr_action_data_get_data(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &e), "get");
    size_t sz = 0;
    ck(amd_comgr_get_data(e, &sz, nullptr), "sz");
    amd_comgr_release_data(e);
    amd_comgr_release_data(src);
    amd_comgr_destroy_data_set(in);
    amd_comgr_destroy_data_set(bc);
    amd_comgr_destroy_data_set(rel);
    amd_comgr_destroy_data_set(exe);
    amd_comgr_destroy_action_info(info);
    return sz;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <recipe.json> <D> [dtype] [arch] [iters]\n", argv[0]);
        return 2;
    }
    std::string recipe = read_file(argv[1]);
    long D = atol(argv[2]);
    const char* dtype = argc >= 4 ? argv[3] : "fp16";
    std::string arch = argc >= 5 ? argv[4] : "gfx950";
    int iters = argc >= 6 ? atoi(argv[5]) : 10;
    std::string isa = "amdgcn-amd-amdhsa--" + arch;

    std::vector<double> tr, tl, tc, tt;
    size_t hsaco = 0;
    for (int it = 0; it < iters + 1; it++) {
        ckc_recipe_spec_int_t ints[] = {{"D", D}};
        ckc_recipe_spec_str_t strs[] = {{"dtype", dtype}};
        auto t0 = clk::now();
        ckc_ir_builder_t b;
        ckc_kernel_def_t* k = nullptr;
        char err[256] = {0};
        auto tr0 = clk::now();
        ckc_status_t st = ckc_recipe_run_from_json(recipe.c_str(), ints, 1, strs, 1, &b, &k, err,
                                                   sizeof err);
        double rms = ms_since(tr0);
        if (st != CKC_OK || !k) {
            fprintf(stderr, "recipe: %s\n", err);
            return 1;
        }
        char* ll = nullptr;
        char lerr[256] = {0};
        auto tl0 = clk::now();
        st = ckc_lower_kernel_to_llvm_ex(k, CKC_LLVM_FLAVOR_AUTO, arch.c_str(), &ll, lerr,
                                         sizeof lerr);
        double lms = ms_since(tl0);
        std::string lltext(ll);
        free(ll);
        ckc_ir_builder_free(&b);
        auto tc0 = clk::now();
        size_t h = comgr_compile(lltext, isa);
        double cms = ms_since(tc0);
        double tot = ms_since(t0);
        if (it > 0) {
            tr.push_back(rms);
            tl.push_back(lms);
            tc.push_back(cms);
            tt.push_back(tot);
        }
        hsaco = h;
    }
    printf("  RECORD+ROLL  recipe(parse+expand)=%6.2f  lower=%5.2f  comgr=%7.2f  total=%7.2f ms  "
           "(hsaco=%zuB)\n",
           median(tr), median(tl), median(tc), median(tt), hsaco);
    return 0;
}
