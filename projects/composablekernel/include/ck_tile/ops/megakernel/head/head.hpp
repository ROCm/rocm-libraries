// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/megakernel/head/head_accept_phase.hpp"
#include "ck_tile/ops/megakernel/head/head_common.hpp"
#include "ck_tile/ops/megakernel/head/head_logits_phase.hpp"
#include "ck_tile/ops/megakernel/head/head_norm_phase.hpp"

/// @file
/// The target model's head for DFlash2 verify, as three megakernel stages:
/// @code
///   megakernel_stage<HeadNormPhase<W>>                              // xn = RMSNorm(x, 1e-5, w)
///   megakernel_stage<HeadLogitsPhase<Gemm>, after_rows<norm>>       // softcap + argmax keys
///   megakernel_stage<HeadAcceptPhase<W>, after_all<logits>>         // tokens, DFlash2 accept
/// @endcode
/// Under TP the lm_head is vocab-sharded (202048 / 8 = 25256 rows per rank); each rank runs the
/// first two stages on its shard with vocab_offset = rank * 25256, and the cross-rank argmax
/// (item 2) is a u64 max over the ranks' row_keys before HeadAcceptPhase.
///
/// Sampled mode (not implemented; design): temperature / top-k / top-p need the whole softcapped
/// row, so HeadLogitsPhase writes args.logits (fp32 [M, N]) and also needs per-row softmax
/// statistics. Plan: per tile, the row max and sum(exp((v - max) / T)) go to a [tiles, M] partial
/// buffer (online-softmax merge in a finalize stage, cross-rank via item 2); top-k (k <= 64 or
/// so) as per-tile top-k candidates (ordered u32 keys, wave-level selection) merged in one
/// workgroup; top-p over those candidates after the k cut; then one uniform draw per row from a
/// counter-based RNG (Philox keyed by seed, step, row) so every rank draws the same. Verify with
/// sampling uses the DFlash2 / speculative-sampling rejection rule, which also needs the draft
/// probabilities; this is a separate accept phase.
