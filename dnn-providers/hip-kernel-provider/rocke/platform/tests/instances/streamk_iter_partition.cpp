// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/instances/streamk_iter_partition.cpp -- C parity for
 * tests/instances/test_streamk_iter_partition.py.
 *
 * The iteration-balanced stream-K partition sizes the launch grid, which the
 * byte-identity gate never sees, so a C/Python divergence here would surface
 * as a silently wrong launch rather than an IR diff. The reference vectors are
 * the Python test's, verbatim; the sweep checks the same invariants the fixup
 * relies on.
 */
#include <cstdio>

#include "rocke/helper_rocke.helpers.streamk.h"

namespace
{

int g_failures = 0;

#define CHECK(cond)                                                           \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
        {                                                                     \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                     \
        }                                                                     \
    } while(0)

struct Vector
{
    int m, n, ipt, w;
    bool persistent;
    /* flags_bytes is for n_waves = 1; ipsc is clamped to 1 when all-DP. */
    int sk_tiles, sk_ctas, ipsc, extra, dp_tiles, grid, partners, rounds, flags_bytes;
};

const Vector k_vectors[] = {
    {4, 4, 8, 16, false, 0, 0, 1, 0, 16, 16, 0, 0, 0},
    {2, 3, 10, 8, false, 6, 8, 7, 4, 0, 8, 2, 2, 256},
    {5, 7, 12, 16, false, 19, 16, 14, 4, 16, 32, 1, 1, 256},
    {1, 3, 2, 8, false, 0, 0, 1, 0, 3, 3, 0, 0, 0},
    {5, 7, 12, 16, true, 19, 16, 14, 4, 16, 16, 1, 1, 256},
    {3, 3, 64, 4, false, 5, 4, 80, 0, 4, 8, 1, 1, 256},
    {7, 5, 3, 16, false, 19, 16, 3, 9, 16, 32, 1, 1, 256},
};

rocke_streamk_iter_partition_t make(int m, int n, int ipt, int w, bool persistent)
{
    rocke_status_t st = ROCKE_OK;
    rocke_streamk_iter_partition_t p
        = rocke_streamk_iter_partition_make(m, n, ipt, w, persistent, &st);
    CHECK(st == ROCKE_OK);
    return p;
}

void check_vectors()
{
    for(const Vector& v : k_vectors)
    {
        const rocke_streamk_iter_partition_t p = make(v.m, v.n, v.ipt, v.w, v.persistent);
        const rocke_streamk_iter_plan_t r = rocke_streamk_iter_plan(&p);
        CHECK(r.sk_tiles == v.sk_tiles);
        CHECK(r.sk_ctas == v.sk_ctas);
        CHECK(r.iters_per_sk_cta == v.ipsc);
        CHECK(r.extra_iters == v.extra);
        CHECK(r.dp_tiles == v.dp_tiles);
        CHECK(r.grid_size == v.grid);
        CHECK(r.max_linear_partners == v.partners);
        CHECK(r.tree_rounds == v.rounds);
        CHECK(rocke_streamk_flags_bytes(&r, 1, NULL) == (size_t)v.flags_bytes);
    }
    const rocke_streamk_iter_partition_t p = make(5, 7, 12, 16, false);
    const int starts[] = {192, 207, 222, 237, 252};
    const int ends[] = {207, 222, 237, 252, 266};
    for(int c = 0; c < 5; ++c)
    {
        CHECK(rocke_streamk_start_iter(&p, c) == starts[c]);
        CHECK(rocke_streamk_end_iter(&p, c) == ends[c]);
    }
}

void check_flags_bytes_per_wave()
{
    /* One i32 flag per (stream-K CTA, wave), padded to 256 bytes. */
    struct
    {
        int m, n, ipt, w, n_waves;
        size_t want;
    } const cases[] = {
        {5, 7, 12, 16, 4, 256},
        {5, 7, 12, 16, 8, 512},
        {2, 3, 10, 8, 9, 512},
        {4, 4, 8, 16, 4, 0},
    };
    for(const auto& c : cases)
    {
        const rocke_streamk_iter_partition_t p = make(c.m, c.n, c.ipt, c.w, false);
        const rocke_streamk_iter_plan_t r = rocke_streamk_iter_plan(&p);
        rocke_status_t st = ROCKE_ERR_VALUE;
        CHECK(rocke_streamk_flags_bytes(&r, c.n_waves, &st) == c.want);
        CHECK(st == ROCKE_OK);
    }
    const rocke_streamk_iter_partition_t p = make(5, 7, 12, 16, false);
    const rocke_streamk_iter_plan_t r = rocke_streamk_iter_plan(&p);
    rocke_status_t st = ROCKE_OK;
    CHECK(rocke_streamk_flags_bytes(&r, 0, &st) == 0);
    CHECK(st == ROCKE_ERR_VALUE);
}

void check_rejects_non_positive()
{
    rocke_status_t st = ROCKE_OK;
    (void)rocke_streamk_iter_partition_make(0, 2, 4, 8, false, &st);
    CHECK(st == ROCKE_ERR_VALUE);
    st = ROCKE_OK;
    (void)rocke_streamk_iter_partition_make(2, 2, 4, 0, false, &st);
    CHECK(st == ROCKE_ERR_VALUE);
}

void check_invariants(int m, int n, int ipt, int w)
{
    const rocke_streamk_iter_partition_t p = make(m, n, ipt, w, false);
    const rocke_streamk_iter_plan_t r = rocke_streamk_iter_plan(&p);
    CHECK(r.dp_tiles + r.sk_tiles == r.num_tiles);
    CHECK(r.iters_per_sk_cta >= 1); /* a kernel divisor even when all-DP */
    if(!r.sk_ctas)
        return;
    int cursor = r.total_dp_iters;
    int prev_first_tile = -1;
    int run = 0; /* contributors to the current tile */
    for(int c = 0; c < r.sk_ctas; ++c)
    {
        const int lo = rocke_streamk_start_iter(&p, c);
        const int hi = rocke_streamk_end_iter(&p, c);
        CHECK(lo == cursor);
        CHECK(hi > lo);
        cursor = hi;
        const int first = lo / ipt;
        const int last = (hi - 1) / ipt;
        /* Contributors to `first` are this CTA plus the run carried from the
         * previous CTA when it ended inside the same tile. */
        run = (first == prev_first_tile) ? run + 1 : 1;
        CHECK(run - 1 <= r.max_linear_partners);
        CHECK(run <= (1 << r.tree_rounds));
        if(last != first)
        {
            run = 1;
            prev_first_tile = last;
        }
        else
            prev_first_tile = first;
    }
    CHECK(cursor == r.total_dp_iters + r.total_sk_iters);
}

} // namespace

int main(void)
{
    check_vectors();
    check_flags_bytes_per_wave();
    check_rejects_non_positive();
    const int ipts[] = {1, 2, 3, 5, 8, 13};
    const int ws[] = {1, 2, 3, 4, 7, 8, 16};
    for(int m = 1; m < 7; ++m)
        for(int n = 1; n < 5; ++n)
            for(int ipt : ipts)
                for(int w : ws)
                    check_invariants(m, n, ipt, w);
    if(g_failures)
    {
        fprintf(stderr, "streamk_iter_partition: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("streamk_iter_partition: OK\n");
    return 0;
}
