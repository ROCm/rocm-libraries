// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// =============================================================================
// AITER Provenance
//
// Commit:  17d4a33b6f9535e820353ebc6217769efc3766d6
// Sources:
//   - aiter/csrc/include/mha_bwd.h           (lines 149-305: backward kernel args)
//   - aiter/csrc/include/aiter_hip_common.h  (lines 48-62: p2, p3 padding structs)
//
// Adaptations:
//   - Replaced AITER padding types p2/p3 with SgprPad2/SgprPad3 (see SgprPadding.hpp)
//   - Removed all AITER #include directives
//   - Added static_assert on sizeof to catch ABI drift
//   - Added detailed field documentation for all three backward kernels
//   - Organized into 3-kernel pipeline: odo → dqdkdv → dq_convert
// =============================================================================

#pragma once

#include "SgprPadding.hpp"

#include <cstddef>
#include <cstdint>

namespace asm_sdpa_engine
{

// NOLINTBEGIN(readability-identifier-naming)

// Field naming note:
// The backward kernel arg structs use the naming convention from AITER's mha_bwd.h
// (e.g. Hs_q, BAs_q, Seqs_q) which differs from the forward kernel args (e.g.
// s_Hs, s_Bs, s_Seqs from mha_fwd.h).  This is intentional: each set of field
// names is kept as close as possible to the corresponding co:<name> AMDHSA
// metadata tags so that future AITER updates can be reconciled by a direct
// textual diff.  The forward and backward AITER sources themselves use different
// naming conventions; we preserve that distinction here.

// =============================================================================
// Kernel 1: ODO (O ⊙ dO) Pre-Processing Kernel
// =============================================================================
// Computes D[b,h,i] = sum_j( O[b,h,i,j] * dO[b,h,i,j] ) reduction along head
// dimension.  This D buffer is used by the main backward kernel to compute
// attention gradients.
//
// Inputs:  O [B, H_q, S_q, D_v], dO [B, H_q, S_q, D_v]
// Outputs: D [B, H_q, S_q] (FP32 reduction result)
//
// Tensor layout: [Batch, Head, SeqLen, HeadDim]  (BHSD)
// Stride naming: Hs = head stride, BAs = batch stride, Seqs = sequence stride
struct __attribute__((packed)) fmha_bwd_odo_args
{
    // ---- Input: forward output O -------------------------------------------
    const void* ptr_o; // co:ptr_o  Output from forward pass [B, H_q, S_q, D_v]
    SgprPad2 _p0;

    // ---- Input: upstream gradient dO ---------------------------------------
    const void* ptr_do; // co:ptr_do  Gradient w.r.t. O [B, H_q, S_q, D_v]
    SgprPad2 _p1;

    // ---- Output: D reduction buffer ----------------------------------------
    void* ptr_d; // co:ptr_d  Output D buffer [B, H_q, S_q] (FP32)
    SgprPad2 _p2;

    // ---- O tensor strides (in bytes) ----------------------------------------
    uint32_t Hs_o; // co:Hs_o  Head stride for O tensor
    SgprPad3 _p3;
    uint32_t BAs_o; // co:BAs_o  Batch stride for O tensor
    SgprPad3 _p4;
    uint32_t Seqs_o; // co:Seqs_o  Sequence stride for O tensor
    SgprPad3 _p5;

    // ---- dO tensor strides (in bytes) --------------------------------------
    uint32_t Hs_do; // co:Hs_do  Head stride for dO tensor
    SgprPad3 _p6;
    uint32_t BAs_do; // co:BAs_do  Batch stride for dO tensor
    SgprPad3 _p7;
    uint32_t Seqs_do; // co:Seqs_do  Sequence stride for dO tensor
    SgprPad3 _p8;

    // ---- D buffer strides (in bytes, FP32) ---------------------------------
    uint32_t Hs_d; // co:Hs_d  Head stride for D buffer
    SgprPad3 _p9;
    uint32_t BAs_d; // co:BAs_d  Batch stride for D buffer
    SgprPad3 _p10;
    uint32_t Seqs_d; // co:Seqs_d  Sequence stride for D (always 4 bytes)
    SgprPad3 _p11;

    // ---- Dimensions --------------------------------------------------------
    uint32_t seqlen_q; // co:seqlen_q  Query sequence length
    SgprPad3 _p12;
    uint32_t head_dim; // co:head_dim  Head dimension (D_v)
    SgprPad3 _p13;

    // ---- Group mode sequence pointers (nullptr for batch mode) -------------
    const void* ptr_qseq; // co:ptr_qseq  Cumulative Q sequence starts
    SgprPad2 _p14;
    const void* ptr_qseq_padded; // co:ptr_qseq_padded  Q padded seq starts
    SgprPad2 _p15;
};

// =============================================================================
// Kernel 2: DQDKDV Main Backward Kernel
// =============================================================================
// Computes gradients dQ, dK, dV from inputs Q, K, V, dO, LSE, D.
//
// Inputs:  Q, K, V, dO, LSE [B,H,S] (FP32), D [B,H,S] (FP32 from odo kernel)
// Outputs: dQ (FP32 if a32, else BF16), dK [B,H_kv,S_k,D], dV [B,H_kv,S_k,D_v]
//
// For a32 accumulator variant: dQ output is in FP32 and written to dq_acc
// workspace buffer. A subsequent dq_convert kernel converts it to BF16.
//
// Tensor layout: [Batch, Head, SeqLen, HeadDim]  (BHSD)
struct __attribute__((packed)) fmha_bwd_dqdkdv_args
{
    // ---- Outputs -----------------------------------------------------------
    void* ptr_dq; // co:ptr_dq  Gradient w.r.t. Q (dq_acc if a32, else dQ)
    SgprPad2 _p0;
    void* ptr_dk; // co:ptr_dk  Gradient w.r.t. K [B, H_kv, S_k, D_qk]
    SgprPad2 _p1;
    void* ptr_dv; // co:ptr_dv  Gradient w.r.t. V [B, H_kv, S_k, D_v]
    SgprPad2 _p2;

    // ---- Inputs: forward pass tensors --------------------------------------
    const void* ptr_q; // co:ptr_q  Query [B, H_q, S_q, D_qk]
    SgprPad2 _p3;
    const void* ptr_k; // co:ptr_k  Key [B, H_kv, S_k, D_qk]
    SgprPad2 _p4;
    const void* ptr_v; // co:ptr_v  Value [B, H_kv, S_k, D_v]
    SgprPad2 _p5;

    // ---- Input: upstream gradient ------------------------------------------
    const void* ptr_do; // co:ptr_do  Gradient w.r.t. O [B, H_q, S_q, D_v]
    SgprPad2 _p6;

    // ---- Input: forward pass LSE stats -------------------------------------
    const void* ptr_lse; // co:ptr_lse  LogSumExp from forward [B, H_q, S_q]
    SgprPad2 _p7;

    // ---- Input: D buffer from odo kernel -----------------------------------
    const void* ptr_d; // co:ptr_d  D reduction buffer [B, H_q, S_q] (FP32)
    SgprPad2 _p8;

    // ---- Attention scale ---------------------------------------------------
    float scalar; // co:scalar  Attention scale (1 / sqrt(D_qk))
    SgprPad3 _p9;
    float log2e; // co:log2e  Constant: log2(e) ≈ 1.44269504
    SgprPad3 _p10;

    // ---- Q dimensions ------------------------------------------------------
    uint32_t seqlen_q; // co:seqlen_q  Query sequence length
    SgprPad3 _p11;
    uint32_t Ts; // co:Ts  Tile size × Seqs_k × 2 (bytes)
    SgprPad3 _p12;

    // ---- Q tensor strides (in bytes) ---------------------------------------
    uint32_t Hs_q; // co:Hs_q  Q head stride
    SgprPad3 _p13;
    uint32_t BAs_q; // co:BAs_q  Q batch stride
    SgprPad3 _p14;
    uint32_t Seqs_q; // co:Seqs_q  Q sequence stride
    SgprPad3 _p15;

    // ---- GQA ratio ---------------------------------------------------------
    uint32_t ratio; // co:ratio  Grouped-Query Attention ratio (H_q / H_kv)
    SgprPad3 _p16;

    // ---- K tensor strides (in bytes) ---------------------------------------
    uint32_t Hs_k; // co:Hs_k  K head stride
    SgprPad3 _p17;
    uint32_t BAs_k; // co:BAs_k  K batch stride
    SgprPad3 _p18;
    uint32_t Seqs_k; // co:Seqs_k  K sequence stride
    SgprPad3 _p19;

    // ---- dK tensor strides (in bytes) --------------------------------------
    uint32_t Seqs_dk; // co:Seqs_dk  dK sequence stride
    SgprPad3 _p20;

    // ---- K/V dimensions ----------------------------------------------------
    uint32_t seqlen_k; // co:seqlen_k  Key/Value sequence length
    SgprPad3 _p21;
    uint32_t head_dim_q; // co:head_dim_q  QK head dimension (D_qk)
    SgprPad3 _p22;
    uint32_t head_dim_v; // co:head_dim_v  V head dimension (D_v)
    SgprPad3 _p23;
    uint32_t nhead_q; // co:nhead_q  Number of Q heads (H_q)
    SgprPad3 _p24;

    // ---- V tensor strides (in bytes) ---------------------------------------
    uint32_t Hs_v; // co:Hs_v  V head stride
    SgprPad3 _p25;
    uint32_t BAs_v; // co:BAs_v  V batch stride
    SgprPad3 _p26;
    uint32_t Seqs_v; // co:Seqs_v  V sequence stride
    SgprPad3 _p27;

    // ---- dO tensor strides (in bytes) --------------------------------------
    uint32_t Hs_do; // co:Hs_do  dO head stride
    SgprPad3 _p28;
    uint32_t BAs_do; // co:BAs_do  dO batch stride
    SgprPad3 _p29;
    uint32_t Seqs_do; // co:Seqs_do  dO sequence stride
    SgprPad3 _p30;

    // ---- dK tensor strides (in bytes) --------------------------------------
    uint32_t Hs_dk; // co:Hs_dk  dK head stride
    SgprPad3 _p31;
    uint32_t BAs_dk; // co:BAs_dk  dK batch stride
    SgprPad3 _p32;

    // ---- dV tensor strides (in bytes) --------------------------------------
    uint32_t Hs_dv; // co:Hs_dv  dV head stride
    SgprPad3 _p33;
    uint32_t BAs_dv; // co:BAs_dv  dV batch stride
    SgprPad3 _p34;
    uint32_t Seqs_dv; // co:Seqs_dv  dV sequence stride
    SgprPad3 _p35;

    // ---- LSE tensor strides (in bytes) -------------------------------------
    uint32_t Hs_lsed; // co:Hs_lsed  LSE head stride (group mode)
    SgprPad3 _p36;

    // ---- Group mode sequence pointers (nullptr for batch mode) -------------
    const void* ptr_qseq; // co:ptr_qseq  Q cumulative seq starts
    SgprPad2 _p37;
    const void* ptr_kseq; // co:ptr_kseq  K cumulative seq starts
    SgprPad2 _p38;
    const void* ptr_qseq_padded; // co:ptr_qseq_padded  Q padded seq starts
    SgprPad2 _p39;
    const void* ptr_kseq_padded; // co:ptr_kseq_padded  K padded seq starts
    SgprPad2 _p40;

    // ---- dq_acc buffer configuration ---------------------------------------
    uint32_t max_seqlen_dq; // co:max_seqlen_dq  Max Q seqlen (for a16)
    SgprPad3 _p41;

    // ---- Window attention mask coordinates ---------------------------------
    int32_t mask_x; // co:mask_x  Window mask X coordinate (-1 if disabled)
    SgprPad3 _p42;
    int32_t mask_y; // co:mask_y  Window mask Y coordinate (-1 if disabled)
    SgprPad3 _p43;
};

// =============================================================================
// Kernel 3: DQ_CONVERT Post-Processing Kernel
// =============================================================================
// Converts dQ from FP32 accumulator (dq_acc) to BF16 output format.
// Only used with a32 accumulator kernels.  The a16 accumulator kernels write
// dQ directly in BF16 and skip this step.
//
// Inputs:  dq_acc [B, H_q, S_q, D_qk] (FP32 from dqdkdv kernel)
// Outputs: dQ [B, H_q, S_q, D_qk] (BF16)
//
// Tensor layout: [Batch, Head, SeqLen, HeadDim]  (BHSD)
struct __attribute__((packed)) fmha_bwd_post_kernel_args
{
    // ---- Input: FP32 dq accumulator buffer ---------------------------------
    void* ptr_dq_acc; // co:ptr_dq_acc  dQ in FP32 [B, H_q, S_q, D_qk]
    SgprPad2 _p0;

    // ---- Output: BF16 dQ ---------------------------------------------------
    void* ptr_dq; // co:ptr_dq  dQ in BF16 [B, H_q, S_q, D_qk]
    SgprPad2 _p1;

    // ---- dq_acc tensor strides (in bytes, FP32) ----------------------------
    uint32_t Hs_dq_acc; // co:Hs_dq_acc  dq_acc head stride
    SgprPad3 _p2;
    uint32_t BAs_dq_acc; // co:BAs_dq_acc  dq_acc batch stride
    SgprPad3 _p3;
    uint32_t Seqs_dq_acc; // co:Seqs_dq_acc  dq_acc sequence stride
    SgprPad3 _p4;

    // ---- dQ tensor strides (in bytes, BF16) --------------------------------
    uint32_t Hs_dq; // co:Hs_dq  dQ head stride
    SgprPad3 _p5;
    uint32_t BAs_dq; // co:BAs_dq  dQ batch stride
    SgprPad3 _p6;
    uint32_t Seqs_dq; // co:Seqs_dq  dQ sequence stride
    SgprPad3 _p7;

    // ---- Dimensions --------------------------------------------------------
    uint32_t seqlen_q; // co:seqlen_q  Query sequence length
    SgprPad3 _p8;
    uint32_t head_dim; // co:head_dim  Head dimension (D_qk)
    SgprPad3 _p9;

    // ---- Group mode sequence pointers (nullptr for batch mode) -------------
    const void* ptr_qseq; // co:ptr_qseq  Q cumulative seq starts
    SgprPad2 _p10;
    const void* ptr_qseq_padded; // co:ptr_qseq_padded  Q padded seq starts
    SgprPad2 _p11;
};

// Compile-time ABI verification: the struct sizes must match AITER's
// definitions exactly.  Any mismatch indicates the kernel ABI has changed and
// the structs must be updated.
static_assert(sizeof(fmha_bwd_dqdkdv_args) == 704,
              "fmha_bwd_dqdkdv_args size mismatch with AITER ABI");

static_assert(sizeof(fmha_bwd_odo_args) == 256, "fmha_bwd_odo_args size mismatch with AITER ABI");

static_assert(sizeof(fmha_bwd_post_kernel_args) == 192,
              "fmha_bwd_post_kernel_args size mismatch with AITER ABI");

// Per-field offset checks for the most load-bearing kernarg fields.  sizeof
// pins total layout, but the kernel reads each field by SGPR index — a
// toolchain change that honoured `__attribute__((packed))` differently could
// keep the total size while shifting individual offsets and silently corrupt
// the dispatch.  Offsets below come from AITER's mha_bwd.h hex annotations
// (commit 9522048).
static_assert(offsetof(fmha_bwd_dqdkdv_args, ptr_dq) == 0x00,
              "fmha_bwd_dqdkdv_args::ptr_dq offset drift vs AITER ABI");
static_assert(offsetof(fmha_bwd_dqdkdv_args, seqlen_q) == 0xb0,
              "fmha_bwd_dqdkdv_args::seqlen_q offset drift vs AITER ABI");
static_assert(offsetof(fmha_bwd_dqdkdv_args, Ts) == 0xc0,
              "fmha_bwd_dqdkdv_args::Ts offset drift vs AITER ABI");
static_assert(offsetof(fmha_bwd_post_kernel_args, Hs_dq_acc) == 0x20,
              "fmha_bwd_post_kernel_args::Hs_dq_acc offset drift vs AITER ABI");

// NOLINTEND(readability-identifier-naming)

} // namespace asm_sdpa_engine
