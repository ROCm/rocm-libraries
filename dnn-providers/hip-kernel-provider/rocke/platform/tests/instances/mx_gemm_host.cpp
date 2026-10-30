// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test of the C entries of the MX GEMM that no Python caller
 * reaches (rocke/instance_mx_gemm.h): rocke_mx_gemm_grid must give the empty
 * grid (0, 0, 1) for a spec whose workgroup tile is unset, as
 * rocke_mx_gemm_spec_default leaves it, and for a NULL spec, so a launch fails
 * instead of reading unset dimensions; for a set tile it gives
 * (ceil(N / block_tile_n), ceil(M / block_tile_m), 1).
 */

#include <stdio.h>

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

    if(failures == 0)
    {
        printf("rocke_mx_gemm_host: OK\n");
    }
    return failures == 0 ? 0 : 1;
}
