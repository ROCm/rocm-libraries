// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// =============================================================================
// AITER Provenance
//
// Commit:  9522048dc10de20ba9dcda1c0a3f640867e7a586
// Sources:
//   - aiter/csrc/include/mha_fwd.h        (lines 266-350: fmha_fwd_v3_args)
//   - aiter/csrc/include/aiter_hip_common.h (lines 44-54: p2, p3 padding structs)
//
// Adaptations:
//   - Replaced AITER padding types p2/p3 with SgprPad2/SgprPad3 (see SgprPadding.hpp)
//   - Removed all AITER #include directives
//   - Added static_assert on sizeof to catch ABI drift
//   - Added field documentation from cross-reference with CK fmha_fwd_args
//     (composablekernel/example/ck_tile/01_fmha/fmha_fwd.hpp)
// =============================================================================

#pragma once

#include "SgprPadding.hpp"

#include <cstdint>

namespace asm_sdpa_engine
{

// NOLINTBEGIN(readability-identifier-naming)

// Packed kernel argument struct for the AITER Flash Attention v3 forward kernel.
// This struct is passed directly to the GPU via HIP_LAUNCH_PARAM_BUFFER_POINTER
// and must match the binary layout expected by the pre-compiled .co kernel.
//
// Tensor layout assumed: [Batch, Head, SeqLen, HeadDim]  (BHSD)
//
// Stride fields use element counts (not bytes).  For the POC the naming follows
// AITER's convention:
//   s_<tensor>Seqs  = stride between consecutive sequence positions (= HeadDim)
//   s_<tensor>Hs    = stride between consecutive heads
//   s_<tensor>Bs    = stride between consecutive batches
//
// Each non-padding field carries a "co:<name>" tag in its comment, recording
// the corresponding argument name in the .co kernel's AMDHSA metadata.
struct __attribute__((packed)) fmha_fwd_v3_args
{
    // ---- Output pointers ---------------------------------------------------
    void* ptr_o; // co:R  Attention output  [B, H_q, S_q, D_v]
    SgprPad2 _p0;
    // ---- Input pointers ----------------------------------------------------
    const void* ptr_q; // co:Q  Query             [B, H_q, S_q, D_qk]
    SgprPad2 _p1;
    const void* ptr_k; // co:K  Key               [B, H_kv, S_kv, D_qk]
    SgprPad2 _p2;
    const void* ptr_v; // co:V  Value             [B, H_kv, S_kv, D_v]
    SgprPad2 _p3;
    void* ptr_lse; // co:LSE  Log-sum-exp stats [B, H_q, S_q, 1]
    SgprPad2 _p4;

    // ---- Attention scale ---------------------------------------------------
    float scalar; // co:scalar  Typically 1 / sqrt(D_qk)
    SgprPad3 _p5;

    // ---- Q tensor dimensions and strides -----------------------------------
    uint32_t s_seq_len; // co:seq_len  Query sequence length  (S_q)
    SgprPad3 _p6;
    uint32_t s_Seqs; // co:Seqs  Q stride: sequence dim
    SgprPad3 _p7;
    uint32_t s_Ts; // co:Ts  Q stride: row
    SgprPad3 _p8;
    uint32_t s_Hs; // co:Hs  Q stride: head dim
    SgprPad3 _p9;
    uint32_t s_Bs; // co:Bs  Q stride: batch dim
    SgprPad3 _p10;

    // ---- GQA ratio ---------------------------------------------------------
    uint32_t s_gqa; // co:gqa  Grouped-Query Attention ratio (H_q / H_kv)
    SgprPad3 _p11;

    // ---- K tensor strides --------------------------------------------------
    uint32_t s_k_Seqs; // co:k_Seqs  K stride: sequence dim
    SgprPad3 _p12;
    uint32_t s_k_Hs; // co:k_Hs  K stride: head dim
    SgprPad3 _p13;
    uint32_t s_k_Bs; // co:k_Bs  K stride: batch dim
    SgprPad3 _p14;

    // ---- Options and flags -------------------------------------------------
    uint32_t s_opt; // co:msk_opt  Options bitmask (e.g., rounding mode)
    SgprPad3 _p15;
    uint32_t s_lse; // co:lse  Whether to compute LSE output (0 or 1)
    SgprPad3 _p16;

    // ---- KV sequence and head dimensions -----------------------------------
    uint32_t s_kv_seq_len; // co:kv_seq_len  KV sequence length (S_kv)
    SgprPad3 _p17;
    uint32_t s_qk_head_dim; // co:qk_head_dim  QK head dimension  (D_qk)
    SgprPad3 _p18;
    uint32_t s_v_head_dim; // co:v_head_dim  V head dimension   (D_v)
    SgprPad3 _p19;
    uint32_t s_q_head_num; // co:q_head_num  Number of Q heads  (H_q)
    SgprPad3 _p20;

    // ---- V tensor strides --------------------------------------------------
    uint32_t s_v_Seqs; // co:v_Seqs  V stride: sequence dim
    SgprPad3 _p21;
    uint32_t s_v_Hs; // co:v_Hs  V stride: head dim
    SgprPad3 _p22;
    uint32_t s_v_Bs; // co:v_Bs  V stride: batch dim
    SgprPad3 _p23;

    // ---- O tensor strides --------------------------------------------------
    uint32_t s_o_Seqs; // co:r_Seqs  O stride: sequence dim
    SgprPad3 _p24;
    uint32_t s_o_Hs; // co:r_Hs  O stride: head dim
    SgprPad3 _p25;
    uint32_t s_o_Bs; // co:r_Bs  O stride: batch dim
    SgprPad3 _p26;

    // ---- Variable-length sequence pointers (nullptr for batch mode) --------
    const void* ptr_qseq; // co:ptr_qseq  Cumulative Q sequence starts (group mode)
    SgprPad2 _p27;
    const void* ptr_kseq; // co:ptr_kseq  Cumulative K sequence starts (group mode)
    SgprPad2 _p28;

    // ---- LSE stride --------------------------------------------------------
    uint32_t s_lse_Hs; // co:lse_Hs  LSE stride: head dim
    SgprPad3 _p29;

    // ---- Padding sequence pointers (nullptr for batch mode) ----------------
    const void* ptr_qseq_padding; // co:ptr_qseq_padding  Q padded seq starts (group mode)
    SgprPad2 _p30;
    const void* ptr_kseq_padding; // co:ptr_kseq_padding  K padded seq starts (group mode)
    SgprPad2 _p31;

    // ---- FP8 descale pointers (nullptr for BF16/FP16) ----------------------
    const void* ptr_q_descale; // co:ptr_q_descale  Q descale tensor
    SgprPad2 _p32;
    const void* ptr_k_descale; // co:ptr_k_descale  K descale tensor
    SgprPad2 _p33;
    const void* ptr_v_descale; // co:ptr_v_descale  V descale tensor
    SgprPad2 _p34;

    // ---- FP8 descale strides -----------------------------------------------
    uint32_t s_descale_q_Bs; // co:descale_q_Bs  Q descale batch stride
    SgprPad3 _p35;
    uint32_t s_descale_q_Hs; // co:descale_q_Hs  Q descale head stride
    SgprPad3 _p36;
    uint32_t s_descale_k_Bs; // co:descale_k_Bs  K descale batch stride
    SgprPad3 _p37;
    uint32_t s_descale_k_Hs; // co:descale_k_Hs  K descale head stride
    SgprPad3 _p38;
    uint32_t s_descale_v_Bs; // co:descale_v_Bs  V descale batch stride
    SgprPad3 _p39;
    uint32_t s_descale_v_Hs; // co:descale_v_Hs  V descale head stride
    SgprPad3 _p40;
};

// Compile-time ABI verification: the struct size must match AITER's definition
// exactly.  Any mismatch indicates the kernel ABI has changed and the struct
// must be updated.
static_assert(sizeof(fmha_fwd_v3_args) == 656,
              "fmha_fwd_v3_args size mismatch with AITER ABI "
              "(commit 9522048dc10de20ba9dcda1c0a3f640867e7a586)");

// NOLINTEND(readability-identifier-naming)

} // namespace asm_sdpa_engine
