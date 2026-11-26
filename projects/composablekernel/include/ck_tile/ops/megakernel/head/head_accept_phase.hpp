// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/head/head_common.hpp"

namespace ck_tile {

/// @brief Greedy finalize and DFlash2 accept as a megakernel phase: one work item, run after the
/// argmax keys are complete (after_all on HeadLogitsPhase, or on the cross-rank combine).
///
/// Decodes row_keys into the argmax token (and its softcapped logit) of each of the M rows, then
/// clears the keys for the next launch (clear_keys). Rows are num_seqs = M / window verify windows
/// of `window` rows each; with drafts, sequence s accepts the longest prefix of its window - 1
/// drafts that equals the argmax (head_dflash_accept) and emits it plus one bonus token.
///
/// TP: row_keys hold global token ids, so a cross-rank argmax (item 2) is an elementwise u64 max
/// of every rank's keys; this phase then reads the combined keys. keys_out exports the local keys
/// for that exchange. Out of scope: tree verification (drafts as a token tree, accept the
/// longest matching root path) and sampled verification (see head.hpp).
/// NumWaves_ is the workgroup's wave count; the thread count follows the compile pass's wave size
/// (the host pass sees 64-lane waves, gfx1250 runs 32), so the type is the same in both passes.
template <index_t NumWaves_>
struct HeadAcceptPhase
{
    static constexpr index_t kMaxRows = 256;

    struct Args
    {
        uint64_t* row_keys;    // [M] argmax keys
        index_t M;             //
        index_t window;        // rows per verify window; M % window == 0
        bool clear_keys;       // zero row_keys after reading (the last reader of a step must)
        const int32_t* drafts; // optional [num_seqs, window - 1]; nullptr = greedy only
        int32_t* tokens;       // [M] argmax token of each row
        float* max_logit;      // optional [M] softcapped logit of that token
        uint64_t* keys_out;    // optional [M] copy of the keys (cross-rank exchange)
        int32_t* accepted;     // [num_seqs, window] accepted drafts + bonus (with drafts)
        int32_t* num_accepted; // [num_seqs] tokens emitted per sequence, 1..window (with drafts)
    };

    using barrier_pipeline = void;

    static constexpr index_t kBlockSize = NumWaves_ * get_warp_size();
    static constexpr index_t kSlotRows  = kMaxRows;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kMaxRows * sizeof(int32_t);
    }

    CK_TILE_HOST_DEVICE static index_t NumWork(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t) { return 1; }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t) { return 0; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t)
    {
        return make_tuple(0, a.M);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        return a.row_keys != nullptr && a.tokens != nullptr && a.M > 0 && a.M <= kMaxRows &&
               a.window > 0 && a.M % a.window == 0 &&
               (a.drafts == nullptr || (a.accepted != nullptr && a.num_accepted != nullptr));
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t, void* smem, const Token&)
    {
        int32_t* s_tok    = static_cast<int32_t*>(smem);
        const index_t tid = static_cast<index_t>(threadIdx.x);

        for(index_t r = tid; r < a.M; r += kBlockSize)
        {
            const uint64_t key = a.row_keys[r];
            const int32_t t    = head_key_token(key);
            s_tok[r]           = t;
            a.tokens[r]        = t;
            if(a.max_logit != nullptr)
                a.max_logit[r] = head_key_value(key);
            if(a.keys_out != nullptr)
                a.keys_out[r] = key;
            if(a.clear_keys)
                a.row_keys[r] = 0;
        }
        if(a.drafts == nullptr)
            return;
        block_sync_lds();

        const index_t w = a.window;
        for(index_t s = tid; s < a.M / w; s += kBlockSize)
        {
            a.num_accepted[s] =
                head_dflash_accept(s_tok + s * w, a.drafts + s * (w - 1), w, a.accepted + s * w);
        }
    }
};

} // namespace ck_tile
