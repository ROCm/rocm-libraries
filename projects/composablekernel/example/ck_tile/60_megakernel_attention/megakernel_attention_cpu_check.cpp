// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only check of DecodeAttentionPhase's work decomposition (no GPU, no HIP runtime calls):
// for sliding and global layers, bf16 and fp8 KV, one to four requests, it runs the phase's own
// Locate / IsMasked / ChooseNumSplits in a float emulation of the split + combine algorithm and
// compares with a double reference. Also asserts every visible (row, key) is visited once.
// Exit code: 0 pass, 1 fail.

#include <cstdio>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/attention/decode_attention_phase.hpp"

#include "attention_reference.hpp"

namespace {

using index_t = ck_tile::index_t;

template <index_t kPage, bool kRunningMax>
long check(attn_ref::Config cfg, index_t grid, index_t splits = 0)
{
    using Phase = ck_tile::
        DecodeAttentionPhase<ck_tile::bf16_t, ck_tile::AttentionOutputStore, kPage, kRunningMax>;
    cfg.page_size = kPage;
    const auto D  = attn_ref::make_data(cfg);

    typename Phase::Args a{};
    a.num_requests = cfg.num_requests;
    a.q_len        = cfg.q_len;
    a.num_kv_heads = cfg.kv_heads;
    a.window       = cfg.window;
    a.scale        = float(attn_ref::kScale);
    a.k_scale      = 1.f; // the emulation reads dequantised K / V
    a.v_scale      = 1.f;
    a.num_splits   = splits > 0
                         ? splits
                         : Phase::ChooseNumSplits(Phase::NumTiles(a),
                                                Phase::MaxSpanKeys(cfg.ctx, cfg.q_len, cfg.window),
                                                grid);
    std::printf("B %d q_len %d ctx %d window %d %s page %d rmax %d grid %d: %d tiles x %d splits\n",
                cfg.num_requests,
                cfg.q_len,
                cfg.ctx,
                cfg.window,
                cfg.fp8 ? "fp8" : "bf16",
                kPage,
                int(kRunningMax),
                grid,
                Phase::NumTiles(a),
                a.num_splits);

    const auto ref = attn_ref::reference(D);
    const auto emu = attn_ref::emulate<Phase>(D, a);
    if(emu.empty())
        return 1;
    return attn_ref::compare("emulation", emu, ref, 1e-4, 1e-3);
}

} // namespace

int main()
{
    long bad = 0;
    attn_ref::Config c;

    // Sliding layer, one DFlash2 verify window, 96-workgroup partition.
    c.ctx    = 4096;
    c.window = 2048;
    bad += check<16, false>(c, 96);
    bad += check<16, true>(c, 96);
    c.fp8 = true;
    bad += check<16, false>(c, 96);
    c.fp8 = false;
    // Four requests, and a context shorter than the window.
    c.num_requests = 4;
    c.ctx          = 1500;
    bad += check<64, false>(c, 96);
    // 8 rows; 24 rows (a full and a partial q tile); forced odd split count.
    c.num_requests = 1;
    c.q_len        = 8;
    c.ctx          = 3000;
    bad += check<16, false>(c, 96);
    c.num_requests = 2;
    c.q_len        = 24;
    bad += check<16, false>(c, 96, 7);
    // Global layer at 64K (the heuristic picks grid / tiles splits).
    c.num_requests = 1;
    c.q_len        = 16;
    c.ctx          = 65536;
    c.window       = 0;
    bad += check<16, false>(c, 96);
    c.fp8 = true;
    bad += check<64, true>(c, 96);

    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
