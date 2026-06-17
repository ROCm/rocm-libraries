// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// bench_compare.cpp -- head-to-head compile-time comparison, same host / same
// codepath, of the two ways to obtain a CK-DSL kernel in the pure-C backend:
//
//   (A) NATIVE C    : ckc_build_unified_attention_2d_scalar_new  (build IR in C)
//   (B) PORTABLE IR : ckc_import_kernel_from_json                (import Python-
//                     authored IR)
//
// Both then run the IDENTICAL ckc_lower_kernel_to_llvm_ex + libamd_comgr path,
// so the only difference measured is "native build" vs "JSON import". The two
// lowered .ll are asserted byte-identical (proving the portable-IR path
// reproduces the native build exactly), then each step is timed (median of N,
// warm-up discarded).
//
//   usage: bench_compare --dtype fp16 --head-size 128 --nqh 32 --nkv 32
//                        --seqlen 2048 --json <kernel.ir.json>
//                        [--arch gfx950] [--iters 10]

#include <amd_comgr/amd_comgr.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "ckc/instance_attention_unified.h"
#include "ckc/ir.h"
#include "ckc/ir_import.h"
#include "ckc/lower_llvm.h"
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

static size_t comgr_compile(const std::string& ll, const std::string& isa)
{
    auto ck = [](amd_comgr_status_t s, const char* w) {
        if (s != AMD_COMGR_STATUS_SUCCESS) {
            fprintf(stderr, "comgr %s failed (%d)\n", w, (int)s);
            exit(1);
        }
    };
    amd_comgr_data_set_t in{};
    ck(amd_comgr_create_data_set(&in), "in");
    amd_comgr_data_t src{};
    ck(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src), "src");
    ck(amd_comgr_set_data(src, ll.size(), ll.data()), "setsrc");
    ck(amd_comgr_set_data_name(src, "kernel.ll"), "name");
    ck(amd_comgr_data_set_add(in, src), "add");
    amd_comgr_action_info_t info{};
    ck(amd_comgr_create_action_info(&info), "info");
    ck(amd_comgr_action_info_set_isa_name(info, isa.c_str()), "isa");
    ck(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR), "lang");
    const char* opts[] = {"-O3"};
    ck(amd_comgr_action_info_set_option_list(info, opts, 1), "opts");
    amd_comgr_data_set_t bc{}, rel{}, exe{};
    ck(amd_comgr_create_data_set(&bc), "bc");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in, bc), "tobc");
    ck(amd_comgr_create_data_set(&rel), "rel");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc, rel), "torel");
    ck(amd_comgr_create_data_set(&exe), "exe");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, rel, exe), "toexe");
    size_t count = 0;
    ck(amd_comgr_action_data_count(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, &count), "count");
    if (!count) {
        fprintf(stderr, "no exe\n");
        exit(1);
    }
    amd_comgr_data_t e{};
    ck(amd_comgr_action_data_get_data(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &e), "gete");
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

struct Args
{
    const char* dtype = "fp16";
    int head_size = 128, nqh = 32, nkv = 32, seqlen = 2048, batch = 1, block_size = 16,
        iters = 10;
    const char* json = nullptr;
    std::string arch = "gfx950";
};

static ckc_unified_attention_problem_t make_problem(const Args& a)
{
    // Must match tests/portable_ir/export_mha.py exactly so both paths build the
    // identical kernel (same name + body).
    ckc_unified_attention_problem_t p = ckc_unified_attention_problem_default();
    p.total_q = a.batch * a.seqlen;
    p.num_seqs = a.batch;
    p.num_query_heads = a.nqh;
    p.num_kv_heads = a.nkv;
    p.head_size = a.head_size;
    p.block_size = a.block_size;
    p.max_seqlen_q = a.seqlen;
    p.max_seqlen_k = a.seqlen;
    p.dtype = a.dtype;
    return p;
}

// Native build + lower -> .ll (timed via out-params).
static std::string native_lower(const Args& a, double* t_build, double* t_lower)
{
    ckc_unified_attention_problem_t p = make_problem(a);
    auto tb0 = clk::now();
    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, "u") != CKC_OK) {
        fprintf(stderr, "native init failed\n");
        exit(1);
    }
    ckc_kernel_def_t* k = ckc_build_unified_attention_2d_scalar_new(&b, &p, nullptr);
    if (!k || !ckc_ir_builder_ok(&b)) {
        fprintf(stderr, "native build failed: %s\n", ckc_ir_builder_error(&b));
        exit(1);
    }
    *t_build = ms_since(tb0);

    char* ll = nullptr;
    char err[256];
    err[0] = '\0';
    auto tl0 = clk::now();
    ckc_status_t st =
        ckc_lower_kernel_to_llvm_ex(k, CKC_LLVM_FLAVOR_AUTO, a.arch.c_str(), &ll, err, sizeof err);
    *t_lower = ms_since(tl0);
    if (st != CKC_OK || !ll) {
        fprintf(stderr, "native lower failed: %s\n", err);
        exit(1);
    }
    std::string out(ll);
    free(ll);
    ckc_ir_builder_free(&b);
    return out;
}

// Portable-IR import + lower -> .ll (timed via out-params).
static std::string portable_lower(const Args& a, const std::string& json, double* t_import,
                                  double* t_lower)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* k = nullptr;
    char err[256];
    err[0] = '\0';
    auto ti0 = clk::now();
    ckc_status_t st = ckc_import_kernel_from_json(json.c_str(), nullptr, &b, &k, err, sizeof err);
    *t_import = ms_since(ti0);
    if (st != CKC_OK || !k) {
        fprintf(stderr, "import failed: %s\n", err);
        exit(1);
    }
    char* ll = nullptr;
    char lerr[256];
    lerr[0] = '\0';
    auto tl0 = clk::now();
    st = ckc_lower_kernel_to_llvm_ex(k, CKC_LLVM_FLAVOR_AUTO, a.arch.c_str(), &ll, lerr, sizeof lerr);
    *t_lower = ms_since(tl0);
    if (st != CKC_OK || !ll) {
        fprintf(stderr, "portable lower failed: %s\n", lerr);
        exit(1);
    }
    std::string out(ll);
    free(ll);
    ckc_ir_builder_free(&b);
    return out;
}

int main(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; i++) {
        auto eq = [&](const char* f) { return strcmp(argv[i], f) == 0; };
        if (eq("--dtype"))
            a.dtype = argv[++i];
        else if (eq("--head-size"))
            a.head_size = atoi(argv[++i]);
        else if (eq("--nqh"))
            a.nqh = atoi(argv[++i]);
        else if (eq("--nkv"))
            a.nkv = atoi(argv[++i]);
        else if (eq("--seqlen"))
            a.seqlen = atoi(argv[++i]);
        else if (eq("--batch"))
            a.batch = atoi(argv[++i]);
        else if (eq("--block-size"))
            a.block_size = atoi(argv[++i]);
        else if (eq("--iters"))
            a.iters = atoi(argv[++i]);
        else if (eq("--json"))
            a.json = argv[++i];
        else if (eq("--arch"))
            a.arch = argv[++i];
    }
    if (!a.json) {
        fprintf(stderr, "need --json <kernel.ir.json>\n");
        return 2;
    }
    std::string isa = "amdgcn-amd-amdhsa--" + a.arch;
    std::string json = read_file(a.json);

    // Correctness cross-check: native and portable-IR must lower to the SAME .ll.
    double d0, d1;
    std::string ll_native = native_lower(a, &d0, &d1);
    std::string ll_port = portable_lower(a, json, &d0, &d1);
    bool ll_match = (ll_native == ll_port);

    std::vector<double> nb, nl, nc, nt, pi, pl, pc, pt;
    size_t hsaco = 0;
    for (int it = 0; it < a.iters + 1; it++) {
        double tb, tl, ti, tl2;
        auto n0 = clk::now();
        std::string lln = native_lower(a, &tb, &tl);
        double ncg_t0b = 0;
        auto nc0 = clk::now();
        size_t hb = comgr_compile(lln, isa);
        double ncg = ms_since(nc0);
        double ntot = ms_since(n0);
        (void)ncg_t0b;

        auto p0 = clk::now();
        std::string llp = portable_lower(a, json, &ti, &tl2);
        auto pc0 = clk::now();
        size_t hp = comgr_compile(llp, isa);
        double pcg = ms_since(pc0);
        double ptot = ms_since(p0);

        if (it > 0) {
            nb.push_back(tb);
            nl.push_back(tl);
            nc.push_back(ncg);
            nt.push_back(ntot);
            pi.push_back(ti);
            pl.push_back(tl2);
            pc.push_back(pcg);
            pt.push_back(ptot);
        }
        hsaco = hb ? hb : hp;
    }

    printf("  NATIVE-C    build=%6.2f  lower=%5.2f  comgr=%7.2f  total=%7.2f ms\n", median(nb),
           median(nl), median(nc), median(nt));
    printf("  PORTABLE-IR import=%6.2f lower=%5.2f  comgr=%7.2f  total=%7.2f ms   (.ll match: %s)\n",
           median(pi), median(pl), median(pc), median(pt), ll_match ? "YES" : "NO");
    double overhead = median(pt) - median(nt);
    printf("  delta(import path - native) total = %+.2f ms   hsaco=%zuB\n", overhead, hsaco);
    return ll_match ? 0 : 3;
}
