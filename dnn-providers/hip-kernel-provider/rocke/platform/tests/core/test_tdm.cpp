// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_tdm.cpp -- host unit test for the C++ TDM descriptor
 * packing (rocke/tdm.h), case for case with tests/core/test_tdm.py.
 *
 * A misplaced bit in the descriptor produces silent garbage or a hang, never a
 * compile error, so the packing is pinned here field by field. The IR-emitting
 * half (rocke_tdm_build_descriptor_2d) is byte-compared against Python by the
 * gfx1250_lowering parity family instead.
 *
 * Plain executable: returns non-zero if any check fails.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "rocke/tdm.h"

static int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
        {                                                                     \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_failures;                                                     \
        }                                                                     \
    } while(0)

static uint32_t field(const uint32_t* words, int word, int lsb, int width)
{
    return (uint32_t)(((uint64_t)words[word] >> lsb) & ((1ull << width) - 1));
}

static bool pad_ok(int64_t interval_bytes, int64_t pad_bytes, int want_interval, int want_amount)
{
    int interval = -1;
    int amount = -1;
    if(rocke_tdm_encode_padding(interval_bytes, pad_bytes, &interval, &amount, NULL, 0)
       != ROCKE_OK)
        return false;
    return interval == want_interval && amount == want_amount;
}

static bool pad_rejected(int64_t interval_bytes, int64_t pad_bytes)
{
    char reason[256];
    reason[0] = 0;
    return rocke_tdm_encode_padding(interval_bytes, pad_bytes, NULL, NULL, reason, sizeof reason)
               == ROCKE_ERR_VALUE
           && reason[0] != 0;
}

static void test_padding(void)
{
    /* pad_interval = log2(interval_bytes / 4) - 1 */
    CHECK(pad_ok(8, 4, 0, 0), "interval 8B");
    CHECK(pad_ok(16, 4, 1, 0), "interval 16B");
    CHECK(pad_ok(32, 4, 2, 0), "interval 32B");
    CHECK(pad_ok(64, 4, 3, 0), "interval 64B");
    CHECK(pad_ok(1024, 4, 7, 0), "interval 1024B");
    /* pad_amount = pad_bytes / 4 - 1 */
    CHECK(pad_ok(64, 16, 3, 3), "pad 16B");
    CHECK(pad_ok(64, 32, 3, 7), "pad 32B (bf16 block_k32 pad16)");
    CHECK(pad_ok(64, 512, 3, 127), "pad 512B");
    CHECK(pad_ok(1024, 512, ROCKE_TDM_PAD_INTERVAL_MAX, ROCKE_TDM_PAD_AMOUNT_MAX),
          "documented maxima");

    CHECK(pad_rejected(64, 2), "pad not a dword multiple");
    CHECK(pad_rejected(12, 16), "interval not a power of two");
    CHECK(pad_rejected(2048, 16), "interval overflows 3 bits");
    CHECK(pad_rejected(64, 1024), "amount overflows 7 bits");
    CHECK(pad_rejected(64, 0), "zero pad");

    int en = -1, iv = -1, am = -1;
    CHECK(rocke_tdm_padding_for_tile(2, 32, 0, &en, &iv, &am, NULL, 0) == ROCKE_OK && en == 0
              && iv == 0 && am == 0,
          "pad_elems 0 disables padding");
    CHECK(rocke_tdm_padding_for_tile(2, 32, 16, &en, &iv, &am, NULL, 0) == ROCKE_OK && en == 1
              && iv == 3 && am == 7,
          "bf16 row 32 pad 16");
}

static void test_group0(void)
{
    uint32_t w[4];
    CHECK(rocke_tdm_pack_group0(0, 0, 0, 0, w, NULL, 0) == ROCKE_OK, "pack group0");
    CHECK(field(w, 0, 0, 2) == 1, "count");
    CHECK(field(w, 3, 30, 2) == 2, "type");
    CHECK(field(w, 0, 31, 1) == 0, "gather_mode");

    CHECK(rocke_tdm_pack_group0(0x1234, 0x1F00112233ll, 0, 0, w, NULL, 0) == ROCKE_OK,
          "pack group0 addresses");
    CHECK(w[1] == 0x1234, "lds_addr");
    CHECK(w[2] == 0x00112233, "global_addr_lo");
    CHECK(field(w, 3, 0, 25) == 0x1F, "global_addr_hi");

    CHECK(rocke_tdm_pack_group0(0, 0xFF00000000ll, 0, 0, w, NULL, 0) == ROCKE_OK,
          "pack group0 word3");
    CHECK(field(w, 3, 0, 25) == 0xFF, "global_addr_hi shares word 3");
    CHECK(field(w, 3, 30, 2) == 2, "type shares word 3");
}

static rocke_tdm_group1_2d_args_t base_group1(void)
{
    rocke_tdm_group1_2d_args_t a;
    memset(&a, 0, sizeof(a));
    a.elem_bytes = 2;
    a.tensor_dim0 = 4096;
    a.tensor_dim1 = 1024;
    a.tile_dim0 = 32;
    a.tile_dim1 = 128;
    a.dim0_stride = 4096;
    a.dim1_stride = 1;
    return a;
}

static void test_group1(void)
{
    uint32_t w[8];
    rocke_tdm_group1_2d_args_t a;
    char reason[256];
    static const int64_t k_elem[] = {1, 2, 4, 8};
    static const int k_code[] = {0, 1, 2, 3};

    for(int i = 0; i < 4; ++i)
    {
        int code = -1;
        CHECK(rocke_tdm_data_size_code(k_elem[i], &code, NULL, 0) == ROCKE_OK
                  && code == k_code[i],
              "data_size code");
        a = base_group1();
        a.elem_bytes = k_elem[i];
        CHECK(rocke_tdm_pack_group1_2d(&a, w, NULL, 0) == ROCKE_OK
                  && (int)field(w, 0, 16, 2) == k_code[i],
              "data_size field");
    }
    CHECK(rocke_tdm_data_size_code(3, NULL, NULL, 0) == ROCKE_ERR_VALUE, "data_size 3B rejected");

    a = base_group1();
    a.tensor_dim0 = 0x12345678;
    a.tensor_dim1 = 0x9ABCDEF0ll;
    CHECK(rocke_tdm_pack_group1_2d(&a, w, NULL, 0) == ROCKE_OK, "pack extents");
    CHECK(field(w, 1, 16, 16) == 0x5678, "tensor_dim0_lo");
    CHECK(field(w, 2, 0, 16) == 0x1234, "tensor_dim0_hi");
    CHECK(field(w, 2, 16, 16) == 0xDEF0, "tensor_dim1_lo");
    CHECK(field(w, 3, 0, 16) == 0x9ABC, "tensor_dim1_hi");

    a = base_group1();
    a.tile_dim0 = 64;
    a.tile_dim1 = 256;
    a.dim0_stride = 8192;
    CHECK(rocke_tdm_pack_group1_2d(&a, w, NULL, 0) == ROCKE_OK, "pack tile dims");
    CHECK(field(w, 3, 16, 16) == 64, "tile_dim0");
    CHECK(field(w, 4, 0, 16) == 256, "tile_dim1");
    CHECK(field(w, 4, 16, 16) == 0, "tile_dim2 unused at rank 2");
    CHECK(w[5] == 8192, "row pitch in stride slot 0");

    a = base_group1();
    a.pad_enable = 1;
    a.pad_interval = 3;
    a.pad_amount = 7;
    CHECK(rocke_tdm_pack_group1_2d(&a, w, NULL, 0) == ROCKE_OK, "pack padding");
    CHECK(field(w, 0, 20, 1) == 1, "pad_enable");
    CHECK(field(w, 0, 22, 3) == 3, "pad_interval");
    CHECK(field(w, 0, 25, 7) == 7, "pad_amount");

    a = base_group1();
    CHECK(rocke_tdm_pack_group1_2d(&a, w, NULL, 0) == ROCKE_OK, "pack defaults");
    CHECK(field(w, 0, 20, 1) == 0, "pad off by default");
    CHECK(field(w, 0, 18, 1) == 0, "atomic_barrier_enable off");
    CHECK(field(w, 0, 19, 1) == 0, "iterate_enable off");
    CHECK(field(w, 0, 21, 1) == 0, "early_timeout off");

    a = base_group1();
    a.tile_dim0 = 1 << 16;
    reason[0] = 0;
    CHECK(rocke_tdm_pack_group1_2d(&a, w, reason, sizeof reason) == ROCKE_ERR_VALUE,
          "tile_dim0 overflow rejected");
    CHECK(strcmp(reason, "TDM field 'tile_dim0' is 16 bits, cannot hold 65536 (0x10000)") == 0,
          "tile_dim0 overflow message matches Python");

    a = base_group1();
    a.dim1_stride = 1 << 17;
    CHECK(rocke_tdm_pack_group1_2d(&a, w, NULL, 0) == ROCKE_ERR_VALUE,
          "dim1_stride overflow rejected");
}

int main(void)
{
    test_padding();
    test_group0();
    test_group1();
    if(g_failures)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("test_tdm: all checks passed\n");
    return 0;
}
