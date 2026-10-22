// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host helpers of the RDNA split-KV WMMA decode (kernels/gfx1151/
 * wmma_fmha_decode.py): key range, pack groups, row tiles, split count,
 * grids, workspace layout, fused-merge choice, spec resolution, kernel names,
 * LDS budget and validity. A native launcher sizes its launches and workspace
 * with these, and none of it is visible to the byte-identity gate, so the C
 * port is pinned to Python reference vectors here (mirrors
 * library/tests/test_wmma_fmha_decode_spec.py).
 */
#include <stdio.h>
#include <string.h>

#include "rocke/instance_gfx1151_wmma_fmha_decode.h"

static int g_failures = 0;

static void check_int(const char* what, long long got, long long want)
{
    if(got != want)
    {
        fprintf(stderr, "FAIL %s: got %lld, want %lld\n", what, got, want);
        ++g_failures;
    }
}

static void check_str(const char* what, const char* got, const char* want)
{
    if(strcmp(got, want) != 0)
    {
        fprintf(stderr, "FAIL %s: got \"%s\", want \"%s\"\n", what, got, want);
        ++g_failures;
    }
}

static rocke_wmma_fmha_decode_spec_t resolved(int head_size, const char* arch)
{
    rocke_wmma_fmha_decode_spec_t s = rocke_wmma_fmha_decode_spec_default();
    char reason[256];
    s.head_size = head_size;
    if(rocke_wmma_fmha_decode_resolve(&s, arch, &s, reason, sizeof reason) != ROCKE_OK)
    {
        fprintf(stderr, "FAIL resolve d%d %s: %s\n", head_size, arch, reason);
        ++g_failures;
    }
    return s;
}

static void test_bucket(void)
{
    static const int cases[][3] = {
        /* head_dim_q, head_dim_v, bucket (0: not served) */
        {64, 64, 64},
        {8, 8, 0},
        {16, 16, 64},
        {16, 64, 64},
        {40, 56, 64},
        {64, 128, 0},
        {72, 128, 128},
        {128, 64, 128},
        {128, 128, 128},
        {112, 128, 128},
        {136, 136, 256},
        {128, 256, 0},
        {200, 136, 256},
        {256, 256, 256},
        {264, 8, 0},
        {12, 16, 0},
        {0, 64, 0},
        {256, 8, 256},
    };
    for(const auto& c : cases)
        check_int("bucket", rocke_wmma_fmha_decode_bucket(c[0], c[1]), c[2]);
}

static void test_kv_range(void)
{
    static const int cases[][7] = {
        /* seqlen_q, seqlen_k, bottom_right, wl, wr, lo, hi */
        {1, 4096, 1, -1, 0, 0, 4096},
        {4, 100, 1, 10, 0, 86, 100},
        {1, 8192, 1, 127, 0, 8064, 8192},
        {3, 50, 0, -1, -1, 0, 50},
        {5, 20, 1, -1, 2, 0, 20},
        {100, 40, 1, 5, 0, 0, 40},
        {8, 8, 0, 0, 0, 0, 8},
    };
    for(const auto& c : cases)
    {
        int lo = -1, hi = -1;
        rocke_wmma_fmha_decode_kv_range(c[0], c[1], c[2] != 0, c[3], c[4], &lo, &hi);
        check_int("kv_range lo", lo, c[5]);
        check_int("kv_range hi", hi, c[6]);
    }
}

static void test_heads(void)
{
    static const int groups[][4] = {
        /* Hq, Hkv, Hv, P (-1: rejected) */
        {64, 8, 8, 8},
        {32, 8, 4, 8},
        {32, 8, 16, 16},
        {16, 16, 16, 16},
        {12, 4, 6, -1},
        {8, 3, 8, -1},
        {0, 1, 1, -1},
    };
    for(const auto& g : groups)
    {
        int out = 0;
        const rocke_status_t st = rocke_wmma_fmha_decode_pack_groups(g[0], g[1], g[2], &out);
        if(g[3] < 0)
            check_int("pack_groups rejects", st == ROCKE_ERR_VALUE, 1);
        else
        {
            check_int("pack_groups status", st, ROCKE_OK);
            check_int("pack_groups", out, g[3]);
        }
    }
    static const int tiles[][5] = {
        /* Hq, Hkv, Hv, Sq, row tiles */
        {64, 8, 8, 1, 1},
        {64, 8, 8, 4, 2},
        {32, 8, 4, 3, 1},
        {32, 32, 32, 1, 1},
        {16, 2, 2, 5, 3},
        {32, 8, 16, 8, 1},
    };
    for(const auto& t : tiles)
    {
        int out = 0;
        check_int(
            "row_tiles status", rocke_wmma_fmha_decode_row_tiles(t[0], t[1], t[2], t[3], &out), 0);
        check_int("row_tiles", out, t[4]);
    }
}

static void test_num_splits(void)
{
    static const char* archs[] = {"gfx11-generic", "gfx12-generic"};
    static const int problems[][5] = {
        /* batch, pack groups, row tiles, kv_len, num_cus */
        {1, 8, 1, 8192, 60},
        {4, 8, 1, 4096, 32},
        {1, 32, 1, 2500, 48},
        {8, 8, 1, 128, 60},
        {1, 1, 1, 100000, 64},
        {2, 16, 2, 0, 40},
    };
    /* [arch][bucket 64/128/256][problem] */
    static const int want[2][3][6] = {
        {{15, 2, 3, 1, 64, 1}, {15, 2, 3, 1, 64, 1}, {7, 1, 1, 1, 64, 1}},
        {{15, 2, 3, 1, 64, 1}, {15, 2, 3, 1, 64, 1}, {3, 1, 1, 1, 32, 1}},
    };
    static const int buckets[] = {64, 128, 256};
    for(int a = 0; a < 2; ++a)
        for(int k = 0; k < 3; ++k)
        {
            const rocke_wmma_fmha_decode_spec_t s = resolved(buckets[k], archs[a]);
            for(int p = 0; p < 6; ++p)
            {
                int out = 0;
                const int* q = problems[p];
                check_int(
                    "num_splits status",
                    rocke_wmma_fmha_decode_num_splits(&s, q[0], q[1], q[2], q[3], q[4], 0, 0, &out),
                    ROCKE_OK);
                check_int("num_splits", out, want[a][k][p]);
            }
        }
    rocke_wmma_fmha_decode_spec_t s = rocke_wmma_fmha_decode_spec_default();
    char reason[256];
    s.num_waves = 2;
    rocke_wmma_fmha_decode_resolve(&s, "gfx12-generic", &s, reason, sizeof reason);
    int out = 0;
    rocke_wmma_fmha_decode_num_splits(&s, 1, 8, 1, 8192, 60, 3, 2, &out);
    check_int("num_splits nw2 wg3 min2", out, 22);
    /* An unresolved spec has no dv_split: rejected, as Python raises. */
    rocke_wmma_fmha_decode_spec_t u = rocke_wmma_fmha_decode_spec_default();
    check_int("num_splits unresolved",
              rocke_wmma_fmha_decode_num_splits(&u, 1, 8, 1, 8192, 60, 0, 0, &out),
              ROCKE_ERR_VALUE);
}

static void test_grids_workspace(void)
{
    static const char* archs[] = {"gfx11-generic", "gfx12-generic"};
    static const int want_grid[] = {160, 320};
    static const long long want_bytes[] = {926336, 926464};
    for(int a = 0; a < 2; ++a)
    {
        const rocke_wmma_fmha_decode_spec_t s = resolved(256, archs[a]);
        int grid[3], block[3];
        check_int("grid status", rocke_wmma_fmha_decode_grid(&s, 2, 8, 1, 5, grid, block), 0);
        check_int("grid x", grid[0], want_grid[a]);
        check_int("grid y", grid[1], 1);
        check_int("block x", block[0], 128);
        check_int("grid overflow",
                  rocke_wmma_fmha_decode_grid(&s, 65536, 64, 64, 64, grid, block),
                  ROCKE_ERR_VALUE);
        check_int("grid empty",
                  rocke_wmma_fmha_decode_grid(&s, 0, 8, 1, 5, grid, block),
                  ROCKE_ERR_VALUE);
        rocke_wmma_fmha_decode_workspace_t ws;
        check_int("workspace status",
                  rocke_wmma_fmha_decode_workspace(&s, 2, 32, 8, 4, 3, 200, 6, &ws),
                  ROCKE_OK);
        check_int("ws_o", ws.ws_o, 0);
        check_int("ws_lse", ws.ws_lse, 921600);
        check_int("ws_cnt", ws.ws_cnt, 926208);
        check_int("ws bytes", ws.bytes, want_bytes[a]);
        rocke_wmma_fmha_decode_workspace(&s, 2, 32, 8, 4, 3, 200, 1, &ws);
        check_int("ws one split", ws.bytes + ws.ws_lse + ws.ws_cnt, 0);
        check_int("ws overflow",
                  rocke_wmma_fmha_decode_workspace(&s, 4096, 64, 8, 8, 64, 256, 64, &ws),
                  ROCKE_ERR_VALUE);
    }
    int grid[3], block[3];
    check_int("merge grid status", rocke_wmma_fmha_decode_merge_grid(3, 32, 4, grid, block), 0);
    check_int("merge grid x", grid[0], 384);
    check_int("merge block x", block[0], ROCKE_WMMA_FMHA_DECODE_MERGE_THREADS);
    static const int fused[] = {0, 0, 1, 1, 1, 0, 0};
    for(int n = 0; n < 7; ++n)
        check_int("fused_merge", rocke_wmma_fmha_decode_fused_merge(n), fused[n]);
}

static void test_names_lds_validity(void)
{
    struct Row
    {
        const char* arch;
        int variant;
        const char* unresolved;
        const char* resolved_name;
        int lds;
    };
    static const Row rows[] = {
        {"gfx11-generic",
         0,
         "rocke_wmma_fmha_decode_d64_fp16_w4",
         "rocke_wmma_fmha_decode_d64_fp16_w4_dvs1_t16_qreg_khalf_vlate_nopf_si_maxilp",
         17408},
        {"gfx11-generic",
         1,
         "rocke_wmma_fmha_decode_d256_bf16_w4_biasq",
         "rocke_wmma_fmha_decode_d256_bf16_w4_biasq_dvs2_t16_qlds_khalf_vlate_nopf_si_maxilp",
         42240},
        {"gfx11-generic",
         2,
         "rocke_wmma_fmha_decode_d128_fp16_w4_cap_sink_alibi_qq_kvfp8_paged16",
         "rocke_wmma_fmha_decode_d128_fp16_w4_cap_sink_alibi_qq_kvfp8_paged16_dvs1_t16_qreg_khalf_"
         "vlate_pf_si_maxilp_int8_tanhf",
         33792},
        {"gfx11-generic",
         3,
         "rocke_wmma_fmha_decode_d128_fp16_w2_default_wpe2",
         "rocke_wmma_fmha_decode_d128_fp16_w2_dvs1_t16_qreg_khalf_vlate_nopf_si_default_wpe2",
         16896},
        {"gfx12-generic",
         0,
         "rocke_wmma_fmha_decode_d64_fp16_w4",
         "rocke_wmma_fmha_decode_d64_fp16_w4_dvs1_t16_qreg_vearly_pf_si_ls_maxilp",
         17408},
        {"gfx12-generic",
         1,
         "rocke_wmma_fmha_decode_d256_bf16_w4_biasq",
         "rocke_wmma_fmha_decode_d256_bf16_w4_biasq_dvs4_t16_qreg_vearly_pf_si_ls_default",
         17408},
        {"gfx12-generic",
         2,
         "rocke_wmma_fmha_decode_d128_fp16_w4_cap_sink_alibi_qq_kvfp8_paged16",
         "rocke_wmma_fmha_decode_d128_fp16_w4_cap_sink_alibi_qq_kvfp8_paged16_dvs1_t16_qreg_vearly_"
         "nopf_si_ls_maxilp_int8_tanhf",
         33792},
        {"gfx12-generic",
         3,
         "rocke_wmma_fmha_decode_d128_fp16_w2_default_wpe2",
         "rocke_wmma_fmha_decode_d128_fp16_w2_dvs1_t16_qreg_vearly_nopf_si_default_wpe2",
         16896},
    };
    for(const Row& r : rows)
    {
        rocke_wmma_fmha_decode_spec_t s = rocke_wmma_fmha_decode_spec_default();
        switch(r.variant)
        {
        case 0:
            s.head_size = 64;
            break;
        case 1:
            s.head_size = 256;
            s.dtype = "bf16";
            s.bias = "q";
            break;
        case 2:
            s.head_size = 128;
            s.softcap = true;
            s.kv_dtype = "fp8e4m3";
            s.layout = "paged";
            s.page_block_size = 16;
            s.sinks = true;
            s.alibi = true;
            s.qq_bias = true;
            break;
        default:
            s.head_size = 128;
            s.scheduler_strategy = "default";
            s.waves_per_eu = 2;
            s.num_waves = 2;
            s.lockstep = 0;
            break;
        }
        char name[512], reason[256];
        check_int("kernel_name", rocke_wmma_fmha_decode_kernel_name(&s, name, sizeof name), 0);
        check_str("unresolved name", name, r.unresolved);
        check_int("lds_bytes", rocke_wmma_fmha_decode_lds_bytes(&s, r.arch), r.lds);
        rocke_wmma_fmha_decode_spec_t res;
        check_int(
            "resolve", rocke_wmma_fmha_decode_resolve(&s, r.arch, &res, reason, sizeof reason), 0);
        rocke_wmma_fmha_decode_kernel_name(&res, name, sizeof name);
        check_str("resolved name", name, r.resolved_name);
        /* Resolution is idempotent; the other family is rejected. */
        rocke_wmma_fmha_decode_spec_t again;
        check_int("re-resolve",
                  rocke_wmma_fmha_decode_resolve(&res, r.arch, &again, reason, sizeof reason),
                  0);
        name[0] = '\0';
        check_int("re-resolved kernel_name",
                  rocke_wmma_fmha_decode_kernel_name(&again, name, sizeof name),
                  0);
        check_str("re-resolved name", name, r.resolved_name);
        const char* other
            = strcmp(r.arch, "gfx11-generic") == 0 ? "gfx12-generic" : "gfx11-generic";
        check_int("other family rejected",
                  rocke_wmma_fmha_decode_resolve(&res, other, &again, reason, sizeof reason),
                  ROCKE_ERR_VALUE);
    }
    struct Valid
    {
        const char* arch;
        int head_size;
        int ok;
    };
    static const Valid valid[] = {
        {"gfx11-generic", 256, 0},
        {"gfx11-generic", 128, 0},
        {"gfx11-generic", 64, 1},
        {"gfx12-generic", 256, 1},
        {"gfx12-generic", 128, 0},
        {"gfx12-generic", 64, 1},
        {"gfx950", 64, 0},
    };
    for(const Valid& v : valid)
    {
        rocke_wmma_fmha_decode_spec_t s = rocke_wmma_fmha_decode_spec_default();
        char reason[256];
        s.head_size = v.head_size;
        s.num_waves = 8;
        check_int("is_valid_spec nw8",
                  rocke_wmma_fmha_decode_is_valid_spec(&s, v.arch, reason, sizeof reason),
                  v.ok);
    }
    rocke_wmma_fmha_decode_spec_t s = rocke_wmma_fmha_decode_spec_default();
    char name[128];
    s.layout = "ragged";
    rocke_wmma_fmha_decode_merge_kernel_name(&s, name, sizeof name);
    check_str("merge name packed", name, "rocke_wmma_fmha_decode_merge_fp16_packed");
    s.head_size = 64;
    s.dv_split = 4;
    char reason[256];
    check_int("dv_split 4 at d64 rejected",
              rocke_wmma_fmha_decode_is_valid_spec(&s, "gfx12-generic", reason, sizeof reason),
              0);
}

int main(void)
{
    test_bucket();
    test_kv_range();
    test_heads();
    test_num_splits();
    test_grids_workspace();
    test_names_lds_validity();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("wmma_fmha_decode host helpers: all checks passed\n");
    return 0;
}
