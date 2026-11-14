// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test of the C entries of the MX GEMM (rocke/instance_mx_gemm.h),
 * run by CTest without the Python binding: rocke_mx_gemm_grid, which no Python
 * caller reaches, and the size refusals of rocke_mx_gemm_is_valid_spec, whose
 * both-engine pytest skips without the binding. rocke_mx_gemm_grid must give the empty
 * grid (0, 0, 1) for a spec whose workgroup tile is unset, as
 * rocke_mx_gemm_spec_default leaves it, and for a NULL spec, so a launch fails
 * instead of reading unset dimensions; for a set tile it gives
 * (ceil(N / block_tile_n), ceil(M / block_tile_m), 1), also for M and N at
 * INT_MAX. rocke_mx_gemm_is_valid_spec must refuse, with the Python engine's
 * text, specs whose sizes leave the int and long long ranges: a workgroup of
 * 2^31 threads, one of 2^38 threads (its wave count wraps to 0 in int) and
 * M = N = 1600000000, refused for its offset into A. That last case pins the
 * text only: no spec passes the A and B checks yet overflows C's offset in
 * long long, so the overflow it once caused shows only under UBSan.
 */

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "rocke/instance_mx_gemm.h"

static int failures = 0;

static void expect_grid(
    const char* what, const rocke_mx_gemm_spec_t* spec, int want_x, int want_y, int want_z)
{
    /* Sentinels, so a grid entry the function leaves unwritten is caught. */
    int got_x = -1;
    int got_y = -1;
    int got_z = -1;
    rocke_mx_gemm_grid(spec, &got_x, &got_y, &got_z);
    if(got_x != want_x || got_y != want_y || got_z != want_z)
    {
        fprintf(stderr,
                "FAIL %s: grid (%d, %d, %d), expected (%d, %d, %d)\n",
                what,
                got_x,
                got_y,
                got_z,
                want_x,
                want_y,
                want_z);
        ++failures;
    }
}

static void expect_refusal(const char* what,
                           int M,
                           int N,
                           int K,
                           int block_tile_m,
                           int block_tile_n,
                           const char* want_reason)
{
    rocke_mx_gemm_spec_t spec = rocke_mx_gemm_spec_default();
    spec.M = M;
    spec.N = N;
    spec.K = K;
    spec.block_tile_m = block_tile_m;
    spec.block_tile_n = block_tile_n;
    char reason[256] = "";
    const bool valid = rocke_mx_gemm_is_valid_spec(&spec, "gfx950", reason, sizeof(reason));
    if(valid || strcmp(reason, want_reason) != 0)
    {
        fprintf(stderr,
                "FAIL %s: valid=%d reason \"%s\", expected a refusal \"%s\"\n",
                what,
                (int)valid,
                reason,
                want_reason);
        ++failures;
    }
}

int main(void)
{
    rocke_mx_gemm_spec_t spec = rocke_mx_gemm_spec_default();
    spec.M = 100;
    spec.N = 300;
    spec.K = 64;
    expect_grid("unset workgroup tile", &spec, 0, 0, 1);
    expect_grid("NULL spec", NULL, 0, 0, 1);

    spec.block_tile_m = 32;
    expect_grid("unset block_tile_n", &spec, 0, 0, 1);

    spec.block_tile_n = 128;
    expect_grid("32 x 128 tile", &spec, 3, 4, 1);

    spec.M = INT_MAX;
    spec.N = INT_MAX;
    spec.block_tile_m = 32;
    spec.block_tile_n = 32;
    expect_grid("M and N at INT_MAX", &spec, 1 << 26, 1 << 26, 1);

    expect_refusal("workgroup of 2^31 threads",
                   64,
                   64,
                   64,
                   1 << 30,
                   32,
                   "block_size 2147483648 > 1024 (hardware cap) on gfx950");
    expect_refusal("workgroup of 2^38 threads",
                   64,
                   64,
                   64,
                   1 << 21,
                   1 << 21,
                   "block_size 274877906944 > 1024 (hardware cap) on gfx950");
    expect_refusal("A refused before C's offset overflows long long",
                   1600000000,
                   1600000000,
                   32,
                   32,
                   32,
                   "the largest byte offset into A is 51199999999, expected at most 2147483647: "
                   "the kernel computes every buffer offset in i32");

    if(failures == 0)
    {
        printf("rocke_mx_gemm_host: OK\n");
    }
    return failures == 0 ? 0 : 1;
}
