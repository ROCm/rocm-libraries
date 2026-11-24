// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_attention_arch_caps.cpp -- the C++ attention capability
 * table (rocke/helper_rocke.helpers.attention_arch_caps.h) pinned to the same
 * expected values as tests/helpers/test_attention_arch_caps.py, so a C++-only
 * build still checks it. The Python test also compares the two engines field
 * by field when the rocke_engine binding is built.
 */
#include <cstdio>
#include <cstring>
#include <string>

extern "C" {
#include "rocke/helper_rocke.helpers.attention_arch_caps.h"
}

namespace
{

int failures = 0;

void expect(bool ok, const std::string& what)
{
    if(!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Row
{
    const char* arch;
    int wave;
    const char* path;
    int n_dma;
    int dma[3];
    bool tr64;
    int agprs;
    bool wide_k;
};

const Row k_rows[] = {
    {"gfx942", 64, "mfma", 1, {1}, false, 256, false},
    {"gfx950", 64, "mfma", 3, {1, 3, 4}, true, 256, true},
    {"gfx1151", 32, "wmma", 0, {}, false, 0, false},
    {"gfx1201", 32, "wmma", 0, {}, false, 0, false},
    {"gfx1250", 32, "wmma", 0, {}, false, 0, true},
};

void check_rows()
{
    for(const Row& r : k_rows)
    {
        const std::string a = r.arch;
        rocke_attn_arch_caps_t c;
        expect(rocke_attn_arch_caps(r.arch, &c, nullptr, 0), a + ": row");
        expect(c.wave_size == r.wave, a + ": wave_size");
        expect(std::strcmp(c.matrix_path, r.path) == 0, a + ": matrix_path");
        expect(c.n_lds_dma_dwords == r.n_dma, a + ": LDS-DMA width count");
        for(int i = 0; i < r.n_dma && i < c.n_lds_dma_dwords; ++i)
            expect(c.lds_dma_dwords[i] == r.dma[i], a + ": LDS-DMA width");
        expect(rocke_attn_caps_max_lds_dma_dwords(&c) == (r.n_dma ? r.dma[r.n_dma - 1] : 0),
               a + ": max LDS-DMA width");
        expect(rocke_attn_max_lds_dma_dwords(r.arch) == rocke_attn_caps_max_lds_dma_dwords(&c),
               a + ": rocke_attn_max_lds_dma_dwords");
        expect(rocke_attn_caps_has_tr_read_b64(&c) == r.tr64, a + ": ds_read_tr16_b64");
        expect(c.agprs == r.agprs && c.has_agprs == (r.agprs > 0), a + ": AGPRs");
        expect(c.native_fp32_atomic_add, a + ": native fp32 atomic add");
        expect(rocke_attn_wide_k_atom_available(r.arch, "bf16") == r.wide_k, a + ": wide-K atom");
    }
}

void check_messages()
{
    char err[256];
    expect(!rocke_attn_arch_caps("gfx90a", nullptr, err, sizeof(err))
               && std::strcmp(err,
                              "no attention capability row for 'gfx90a'; known: ('gfx942', "
                              "'gfx950', 'gfx1151', 'gfx1201', 'gfx1250')")
                      == 0,
           "unknown arch message");
    expect(!rocke_attn_check_lds_dma_width("gfx942", 4, err, sizeof(err))
               && std::strcmp(err,
                              "LDS-DMA width of 4 dword(s) is not legal on gfx942; legal "
                              "widths: (1,)")
                      == 0,
           "gfx942 width-4 message");
    expect(!rocke_attn_check_lds_dma_width("gfx1151", 1, err, sizeof(err))
               && std::strcmp(err, "gfx1151 has no global-to-LDS DMA path") == 0,
           "gfx1151 no-DMA message");
    expect(rocke_attn_check_lds_dma_width("gfx950", 3, err, sizeof(err)), "gfx950 width 3");
    expect(!rocke_attn_check_transpose_read("gfx942", 64, err, sizeof(err))
               && std::strcmp(err, "gfx942 has no ds_read_tr16_b64") == 0,
           "gfx942 transpose-read message");

    int v = 0;
    expect(rocke_attn_asm_tail_nop_min("gfx942", "bf16", 16, &v, err, sizeof(err)) && v == 6,
           "gfx942 s_nop floor 6");
    expect(rocke_attn_asm_tail_nop_min("gfx950", "fp16", 32, &v, err, sizeof(err)) && v == 7,
           "gfx950 s_nop floor 7");
    expect(!rocke_attn_check_asm_tail_nop("gfx942", "bf16", 16, 5, err, sizeof(err))
               && std::strcmp(err,
                              "asm tail s_nop 5 is not legal on gfx942 for the 16x16x16 bf16 "
                              "atom; legal: 6..15 (ISA MFMA result wait states)")
                      == 0,
           "s_nop floor message");
    const char* mn = nullptr;
    expect(!rocke_attn_mfma_asm_mnemonic("gfx942", "bf16", 32, &mn, err, sizeof(err))
               && std::strcmp(err, "the 16x16x32 MFMA atom is gfx950-only") == 0,
           "gfx942 K=32 message");
    expect(rocke_attn_mfma_asm_mnemonic("gfx950", "bf16", 32, &mn, err, sizeof(err))
               && std::strcmp(mn, "v_mfma_f32_16x16x32_bf16") == 0,
           "gfx950 K=32 mnemonic");
}

} // namespace

int main()
{
    check_rows();
    check_messages();
    if(failures)
        std::fprintf(stderr, "%d failure(s)\n", failures);
    else
        std::printf("attention arch caps: OK\n");
    return failures ? 1 : 0;
}
