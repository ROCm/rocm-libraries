// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// bench_compile.cpp -- end-to-end compile-time benchmark for the portable-IR
// path: read a portable CK-DSL IR JSON (from ck_dsl.core.ir_export), then for
// N iterations measure the three steps the online provider would run:
//
//   1. import : ckc_import_kernel_from_json   (JSON -> ckc_kernel_def_t)
//   2. lower  : ckc_lower_kernel_to_llvm_ex   (IR  -> AMDGPU LLVM IR text)
//   3. comgr  : amd_comgr                     (.ll -> gfx950 HSACO)
//
// This mirrors the "cold compile (per new shape)" metric in
// dsl_docs/architecture/SDPA_CKDSL_Provider_Comparison: a real comgr JIT with
// no in-memory cache. The only difference from the shipped C-interface is that
// the kernel source is a Python-authored portable-IR artifact instead of a
// C-native build -- so this validates that the portable-IR path keeps the
// C-interface's flat, comgr-bound compile profile.
//
//   usage: bench_compile <kernel.ir.json> [arch=gfx950] [iters=10]
//
// Build (see run_compile_bench.sh): cc compiles the ckc C sources to a static
// lib; g++ links this TU against it + libamd_comgr.

#include <amd_comgr/amd_comgr.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "ckc/ir_import.h"
#include "ckc/lower_llvm.h"
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static std::string read_file(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s((size_t)n, '\0');
    size_t rd = fread(&s[0], 1, (size_t)n, f);
    s.resize(rd);
    fclose(f);
    return s;
}

// Minimal comgr .ll -> HSACO, mirroring ck_dsl_runtime/comgr.hpp.
static size_t comgr_compile(const std::string& ll, const std::string& isa)
{
    auto ck = [](amd_comgr_status_t s, const char* where) {
        if (s != AMD_COMGR_STATUS_SUCCESS) {
            fprintf(stderr, "comgr %s failed (%d)\n", where, (int)s);
            exit(1);
        }
    };
    amd_comgr_data_set_t in_set{};
    ck(amd_comgr_create_data_set(&in_set), "create in");
    amd_comgr_data_t src{};
    ck(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src), "create src");
    ck(amd_comgr_set_data(src, ll.size(), ll.data()), "set src");
    ck(amd_comgr_set_data_name(src, "kernel.ll"), "name src");
    ck(amd_comgr_data_set_add(in_set, src), "add src");

    amd_comgr_action_info_t info{};
    ck(amd_comgr_create_action_info(&info), "create info");
    ck(amd_comgr_action_info_set_isa_name(info, isa.c_str()), "isa");
    ck(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR), "lang");
    const char* opts[] = {"-O3"};
    ck(amd_comgr_action_info_set_option_list(info, opts, 1), "opts");

    amd_comgr_data_set_t bc{}, rel{}, exe{};
    ck(amd_comgr_create_data_set(&bc), "create bc");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc), "to_bc");
    ck(amd_comgr_create_data_set(&rel), "create rel");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc, rel), "to_rel");
    ck(amd_comgr_create_data_set(&exe), "create exe");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, rel, exe),
       "to_exe");

    size_t count = 0;
    ck(amd_comgr_action_data_count(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, &count), "count");
    if (count == 0) {
        fprintf(stderr, "comgr produced no executable\n");
        exit(1);
    }
    amd_comgr_data_t e{};
    ck(amd_comgr_action_data_get_data(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &e), "get exe");
    size_t sz = 0;
    ck(amd_comgr_get_data(e, &sz, nullptr), "size");

    amd_comgr_release_data(e);
    amd_comgr_release_data(src);
    amd_comgr_destroy_data_set(in_set);
    amd_comgr_destroy_data_set(bc);
    amd_comgr_destroy_data_set(rel);
    amd_comgr_destroy_data_set(exe);
    amd_comgr_destroy_action_info(info);
    return sz;
}

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <kernel.ir.json> [arch=gfx950] [iters=10]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    std::string arch = argc >= 3 ? argv[2] : "gfx950";
    int iters = argc >= 4 ? atoi(argv[3]) : 10;
    std::string isa = "amdgcn-amd-amdhsa--" + arch;

    std::string text = read_file(path);

    std::vector<double> t_import, t_lower, t_comgr, t_total;
    size_t hsaco_bytes = 0, ll_bytes = 0;

    for (int it = 0; it < iters + 1; it++) {  // +1 warm-up (discarded)
        auto t0 = clk::now();

        ckc_ir_builder_t b;
        ckc_kernel_def_t* kernel = nullptr;
        char err[256];
        err[0] = '\0';
        auto ti0 = clk::now();
        ckc_status_t st = ckc_import_kernel_from_json(text.c_str(), nullptr, &b, &kernel, err,
                                                      sizeof err);
        double imp = ms_since(ti0);
        if (st != CKC_OK || !kernel) {
            fprintf(stderr, "import failed: %s\n", err);
            return 1;
        }

        char* ll = nullptr;
        char lerr[256];
        lerr[0] = '\0';
        auto tl0 = clk::now();
        st = ckc_lower_kernel_to_llvm_ex(kernel, CKC_LLVM_FLAVOR_AUTO, arch.c_str(), &ll, lerr,
                                         sizeof lerr);
        double low = ms_since(tl0);
        if (st != CKC_OK || !ll) {
            fprintf(stderr, "lower failed: %s\n", lerr);
            ckc_ir_builder_free(&b);
            return 1;
        }
        std::string ll_text(ll);
        free(ll);
        ckc_ir_builder_free(&b);

        auto tc0 = clk::now();
        size_t hb = comgr_compile(ll_text, isa);
        double cg = ms_since(tc0);

        double tot = ms_since(t0);
        if (it > 0) {  // discard warm-up
            t_import.push_back(imp);
            t_lower.push_back(low);
            t_comgr.push_back(cg);
            t_total.push_back(tot);
        }
        hsaco_bytes = hb;
        ll_bytes = ll_text.size();
    }

    printf("%-40s  import=%6.2f  lower=%6.2f  comgr=%7.2f  total=%7.2f ms   (ll=%zuKB hsaco=%zuB)\n",
           path, median(t_import), median(t_lower), median(t_comgr), median(t_total),
           ll_bytes / 1024, hsaco_bytes);
    return 0;
}
