// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// rocke_attn_bwd_ring_slot_schedule is a C entry point, so it refuses a ring
// path other than "vgpr" / "dma" (NULL included) and a depth below 1 instead
// of reading them as the register path. The legal cases keep Python's
// ring_slot_schedule values.

#include <cstdio>
#include <cstring>

#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"
#include "rocke/ir.h"

static int g_failures = 0;

static void expect_ok(int depth, const char* path, int read, int fill, int ahead)
{
    rocke_attn_bwd_err_t e{};
    rocke_attn_bwd_ring_slots_t s{};
    if(rocke_attn_bwd_ring_slot_schedule(depth, path, &s, &e) != ROCKE_OK)
    {
        std::fprintf(stderr, "FAIL: depth %d path %s refused: %s\n", depth, path, e.msg);
        ++g_failures;
        return;
    }
    if(s.read != read || s.fill != fill || s.ahead != ahead)
    {
        std::fprintf(stderr,
                     "FAIL: depth %d path %s: got (%d, %d, %d), want (%d, %d, %d)\n",
                     depth,
                     path,
                     s.read,
                     s.fill,
                     s.ahead,
                     read,
                     fill,
                     ahead);
        ++g_failures;
    }
}

static void expect_refused(int depth, const char* path, const char* msg)
{
    rocke_attn_bwd_err_t e{};
    rocke_attn_bwd_ring_slots_t s{};
    if(rocke_attn_bwd_ring_slot_schedule(depth, path, &s, &e) != ROCKE_ERR_VALUE
       || std::strcmp(e.msg, msg) != 0)
    {
        std::fprintf(stderr,
                     "FAIL: depth %d path %s: want ValueError \"%s\", got \"%s\"\n",
                     depth,
                     path ? path : "NULL",
                     msg,
                     e.msg);
        ++g_failures;
    }
}

int main()
{
    expect_ok(2, "vgpr", 0, 1, 1);
    expect_ok(3, "vgpr", 0, 1, 2);
    expect_ok(1, "dma", 0, 0, 0);
    expect_ok(3, "dma", 0, 2, 2);
    expect_refused(1, "vgpr", "ring_depth 1 on the register path is not a ring");
    expect_refused(2, NULL, "ring path must be one of ('vgpr', 'dma'), not None");
    expect_refused(2, "DMA", "ring path must be one of ('vgpr', 'dma'), not 'DMA'");
    expect_refused(2, "", "ring path must be one of ('vgpr', 'dma'), not ''");
    expect_refused(0, "dma", "ring_depth must be >= 1, not 0");
    expect_refused(-1, "vgpr", "ring_depth must be >= 1, not -1");
    if(g_failures)
        return 1;
    std::puts("OK: ring slot schedule");
    return 0;
}
