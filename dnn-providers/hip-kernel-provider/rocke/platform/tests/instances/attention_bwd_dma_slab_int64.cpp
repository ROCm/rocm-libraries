// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// DmaSlabLayout sizes and offsets are 64-bit, as Python's ints are.
//
// rows * row_bytes passes 2^31 well before rows and cols do. In 32-bit int
// math, slab_offset(rows=2^20, cols=2^16, row=2^20-1, col=2^16-1) returned
// ok with -57 (signed overflow) where Python's
// DmaSlabLayout(2**20, 2**16, 2).offset(2**20 - 1, 2**16 - 1) // 2 is
// 68719476679. The expected values below are Python's.

#include <cstdint>
#include <cstdio>

#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"
#include "rocke/ir.h"

static int g_failures = 0;

static void expect_eq(long long got, long long want, const char* what)
{
    if(got != want)
    {
        std::fprintf(stderr, "FAIL: %s: got %lld, want %lld\n", what, got, want);
        ++g_failures;
    }
}

static void layout(int rows, int cols, int row, int col, long long size, long long offset)
{
    rocke_attn_bwd_err_t e{};
    rocke_attn_bwd_dma_slab_layout_t l;
    if(rocke_attn_bwd_dma_slab_layout_init(&l, rows, cols, 2, &e) != ROCKE_OK)
    {
        std::fprintf(stderr, "FAIL: init %dx%d: %s\n", rows, cols, e.msg);
        ++g_failures;
        return;
    }
    expect_eq(rocke_attn_bwd_dma_slab_layout_size(&l), size, "size");
    expect_eq(rocke_attn_bwd_dma_slab_layout_offset(&l, row, col, &e), offset, "offset");
}

int main()
{
    // a real tile: unchanged
    layout(64, 64, 63, 63, 8192, 8078);
    // past 2^31 bytes
    layout(1 << 20, 1 << 16, (1 << 20) - 1, (1 << 16) - 1, 137438953472LL, 137438953358LL);

    // the builder helper the reviewer reproduced with
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "dma_slab_int64") != ROCKE_OK)
    {
        std::fprintf(stderr, "FAIL: builder init\n");
        return 1;
    }
    int64_t off = 0;
    bool ok = rocke_attn_bwd_slab_offset(&b, 1 << 20, 1 << 16, (1 << 20) - 1, (1 << 16) - 1, &off);
    expect_eq(ok, 1, "slab_offset ok");
    expect_eq(off, 68719476679LL, "slab_offset");
    rocke_ir_builder_free(&b);

    if(g_failures)
    {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("OK: DmaSlabLayout sizes and offsets match Python past 2^31\n");
    return 0;
}
