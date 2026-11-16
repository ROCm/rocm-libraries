// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include "engines/asm_sdpa_engine/asm/SdpaBwdKernelArgs.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdArgsBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdParams.hpp"

using asm_sdpa_engine::AccumulatorType;
using asm_sdpa_engine::buildDqdkdvArgs;
using asm_sdpa_engine::buildMhaBwdArgs;
using asm_sdpa_engine::buildOdoArgs;
using asm_sdpa_engine::buildPostArgs;
using asm_sdpa_engine::fmha_bwd_dqdkdv_args;
using asm_sdpa_engine::fmha_bwd_odo_args;
using asm_sdpa_engine::fmha_bwd_post_kernel_args;
using asm_sdpa_engine::MhaBwdArgs;
using asm_sdpa_engine::SdpaBwdParams;

// =============================================================================
// Tests for fmha_bwd_odo_args (Kernel 1: O ⊙ dO pre-processing)
// =============================================================================

TEST(TestSdpaBwdOdoKernelArgs, TotalSizeMatches)
{
    EXPECT_EQ(sizeof(fmha_bwd_odo_args), 256u);
}

TEST(TestSdpaBwdOdoKernelArgs, PointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, ptr_o), 0u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, ptr_do), 16u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, ptr_d), 32u);
}

TEST(TestSdpaBwdOdoKernelArgs, OStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, Hs_o), 48u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, BAs_o), 64u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, Seqs_o), 80u);
}

TEST(TestSdpaBwdOdoKernelArgs, DOStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, Hs_do), 96u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, BAs_do), 112u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, Seqs_do), 128u);
}

TEST(TestSdpaBwdOdoKernelArgs, DBufferStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, Hs_d), 144u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, BAs_d), 160u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, Seqs_d), 176u);
}

TEST(TestSdpaBwdOdoKernelArgs, DimensionFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, seqlen_q), 192u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, head_dim), 208u);
}

TEST(TestSdpaBwdOdoKernelArgs, SequencePointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, ptr_qseq), 224u);
    EXPECT_EQ(offsetof(fmha_bwd_odo_args, ptr_qseq_padded), 240u);
}

// =============================================================================
// Tests for fmha_bwd_dqdkdv_args (Kernel 2: Main backward computation)
// =============================================================================

TEST(TestSdpaBwdDqdkdvKernelArgs, TotalSizeMatches)
{
    EXPECT_EQ(sizeof(fmha_bwd_dqdkdv_args), 704u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, OutputPointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_dq), 0u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_dk), 16u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_dv), 32u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, InputPointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_q), 48u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_k), 64u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_v), 80u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_do), 96u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_lse), 112u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_d), 128u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, ScalarFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, scalar), 144u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, log2e), 160u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, QDimensionsAndStrideOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, seqlen_q), 176u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Ts), 192u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Hs_q), 208u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, BAs_q), 224u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Seqs_q), 240u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, GqaAndKStrideOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ratio), 256u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Hs_k), 272u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, BAs_k), 288u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Seqs_k), 304u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Seqs_dk), 320u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, KVDimensionOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, seqlen_k), 336u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, head_dim_q), 352u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, head_dim_v), 368u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, nhead_q), 384u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, VStrideOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Hs_v), 400u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, BAs_v), 416u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Seqs_v), 432u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, DOStrideOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Hs_do), 448u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, BAs_do), 464u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Seqs_do), 480u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, DKDVStrideOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Hs_dk), 496u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, BAs_dk), 512u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Hs_dv), 528u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, BAs_dv), 544u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Seqs_dv), 560u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, LseStrideOffset)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, Hs_lsed), 576u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, SequencePointerOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_qseq), 592u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_kseq), 608u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_qseq_padded), 624u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, ptr_kseq_padded), 640u);
}

TEST(TestSdpaBwdDqdkdvKernelArgs, MaxSeqlenAndMaskOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, max_seqlen_dq), 656u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, mask_x), 672u);
    EXPECT_EQ(offsetof(fmha_bwd_dqdkdv_args, mask_y), 688u);
}

// =============================================================================
// Tests for fmha_bwd_post_kernel_args (Kernel 3: dQ convert FP32→BF16)
// =============================================================================

TEST(TestSdpaBwdPostKernelArgs, TotalSizeMatches)
{
    EXPECT_EQ(sizeof(fmha_bwd_post_kernel_args), 192u);
}

TEST(TestSdpaBwdPostKernelArgs, PointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, ptr_dq_acc), 0u);
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, ptr_dq), 16u);
}

TEST(TestSdpaBwdPostKernelArgs, DqAccStrideOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, Hs_dq_acc), 32u);
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, BAs_dq_acc), 48u);
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, Seqs_dq_acc), 64u);
}

TEST(TestSdpaBwdPostKernelArgs, DqStrideOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, Hs_dq), 80u);
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, BAs_dq), 96u);
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, Seqs_dq), 112u);
}

// The gfx950 post kernels load the dq_acc head and batch strides as 64-bit, so
// those fields, and the dQ ones beside them, are 64-bit (AITER v0.1.29
// mha_bwd.h); the sequence strides stay 32-bit.
TEST(TestSdpaBwdPostKernelArgs, HeadAndBatchStridesAre64Bit)
{
    EXPECT_TRUE((std::is_same_v<decltype(fmha_bwd_post_kernel_args::Hs_dq_acc), uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(fmha_bwd_post_kernel_args::BAs_dq_acc), uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(fmha_bwd_post_kernel_args::Hs_dq), uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(fmha_bwd_post_kernel_args::BAs_dq), uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(fmha_bwd_post_kernel_args::Seqs_dq_acc), uint32_t>));
    EXPECT_TRUE((std::is_same_v<decltype(fmha_bwd_post_kernel_args::Seqs_dq), uint32_t>));
}

TEST(TestSdpaBwdPostKernelArgs, DimensionFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, seqlen_q), 128u);
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, head_dim), 144u);
}

TEST(TestSdpaBwdPostKernelArgs, SequencePointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, ptr_qseq), 160u);
    EXPECT_EQ(offsetof(fmha_bwd_post_kernel_args, ptr_qseq_padded), 176u);
}

// =============================================================================
// Stage argument builders (SdpaBwdArgsBuilder.hpp): pinned values
// =============================================================================
//
// Each case builds the stage arguments as SdpaBwdPlan::execute does and pins
// every field. makeArgsParams() gives every tensor and axis a distinct element
// stride (not a real layout), so a field filled from the wrong tensor or axis,
// or scaled by the wrong element size, changes the value. B, H, S_q and S_kv
// differ too, and S_q = 200 is not a multiple of 16, so the A16 dq_acc row
// count, ceil16(S_q) = 208, differs from S_q.

namespace
{

// Host addresses standing in for the device buffers; only their identity is read.
struct BwdBuffers
{
    char q;
    char k;
    char v;
    char o;
    char dOut;
    char lse;
    char dq;
    char dk;
    char dv;
    char d;
    char dqAcc;
};

struct BwdStageArgs
{
    fmha_bwd_odo_args odo;
    fmha_bwd_dqdkdv_args dqdkdv;
    fmha_bwd_post_kernel_args post;
};

constexpr float K_ARGS_SCALE = 0.125F;

// Element strides: the hundreds digit names the tensor (Q 1, K 2, V 3, O 4,
// dO 5, dQ 6, dK 7, dV 8, stats and D 9), the units digit the axis (1 sequence,
// 2 head, 3 batch). Byte strides are twice these, or four times for stats and D.
SdpaBwdParams makeArgsParams(const char* arch,
                             AccumulatorType accType,
                             unsigned int headDimQk,
                             unsigned int tsKv)
{
    SdpaBwdParams p{};
    p.batchSize = 2U;
    p.numHeadsQ = 8U;
    p.numHeadsKv = 2U;
    p.seqLenQ = 200U;
    p.seqLenKv = 300U;
    p.headDimQk = headDimQk;
    p.headDimV = 128U;
    p.qStrideSeq = 101U;
    p.qStrideHead = 102U;
    p.qStrideBatch = 103U;
    p.kStrideSeq = 201U;
    p.kStrideHead = 202U;
    p.kStrideBatch = 203U;
    p.vStrideSeq = 301U;
    p.vStrideHead = 302U;
    p.vStrideBatch = 303U;
    p.oStrideSeq = 401U;
    p.oStrideHead = 402U;
    p.oStrideBatch = 403U;
    p.doStrideSeq = 501U;
    p.doStrideHead = 502U;
    p.doStrideBatch = 503U;
    p.dqStrideSeq = 601U;
    p.dqStrideHead = 602U;
    p.dqStrideBatch = 603U;
    p.dkStrideSeq = 701U;
    p.dkStrideHead = 702U;
    p.dkStrideBatch = 703U;
    p.dvStrideSeq = 801U;
    p.dvStrideHead = 802U;
    p.dvStrideBatch = 803U;
    p.statsStrideHead = 902U;
    p.statsStrideBatch = 903U;
    p.dqdkdvTiles.ts = tsKv;
    p.accumulatorType = accType;
    p.archString = arch;
    return p;
}

// Builds the stage arguments as SdpaBwdPlan::execute does: a dq_acc exists only
// when a post stage reads it.
BwdStageArgs buildStageArgs(const SdpaBwdParams& p, BwdBuffers& b, bool needPostProcessing)
{
    const MhaBwdArgs a = buildMhaBwdArgs(p,
                                         K_ARGS_SCALE,
                                         &b.q,
                                         &b.k,
                                         &b.v,
                                         &b.o,
                                         &b.dOut,
                                         &b.lse,
                                         &b.dq,
                                         &b.dk,
                                         &b.dv,
                                         &b.d,
                                         needPostProcessing ? &b.dqAcc : nullptr,
                                         needPostProcessing);
    return {buildOdoArgs(a), buildDqdkdvArgs(a, p.dqdkdvTiles.ts, p), buildPostArgs(a)};
}

void expectOdoArgs(const fmha_bwd_odo_args& odo, const BwdBuffers& b, uint32_t headDimQk)
{
    EXPECT_EQ(odo.ptr_o, &b.o);
    EXPECT_EQ(odo.ptr_do, &b.dOut);
    EXPECT_EQ(odo.ptr_d, &b.d);
    EXPECT_EQ(odo.Hs_o, 804U);
    EXPECT_EQ(odo.BAs_o, 806U);
    EXPECT_EQ(odo.Seqs_o, 802U);
    EXPECT_EQ(odo.Hs_do, 1004U);
    EXPECT_EQ(odo.BAs_do, 1006U);
    EXPECT_EQ(odo.Seqs_do, 1002U);
    EXPECT_EQ(odo.Hs_d, 3608U);
    EXPECT_EQ(odo.BAs_d, 3612U);
    EXPECT_EQ(odo.Seqs_d, 4U);
    EXPECT_EQ(odo.seqlen_q, 200U);
    EXPECT_EQ(odo.head_dim, headDimQk);
    EXPECT_EQ(odo.ptr_qseq, nullptr);
    EXPECT_EQ(odo.ptr_qseq_padded, nullptr);
}

// The dqdkdv fields every case shares; each test pins ptr_dq, Ts, the head
// dims, max_seqlen_dq and the mask coordinates.
void expectDqdkdvSharedArgs(const fmha_bwd_dqdkdv_args& dqdkdv, const BwdBuffers& b)
{
    EXPECT_EQ(dqdkdv.ptr_dk, &b.dk);
    EXPECT_EQ(dqdkdv.ptr_dv, &b.dv);
    EXPECT_EQ(dqdkdv.ptr_q, &b.q);
    EXPECT_EQ(dqdkdv.ptr_k, &b.k);
    EXPECT_EQ(dqdkdv.ptr_v, &b.v);
    EXPECT_EQ(dqdkdv.ptr_do, &b.dOut);
    EXPECT_EQ(dqdkdv.ptr_lse, &b.lse);
    EXPECT_EQ(dqdkdv.ptr_d, &b.d);
    EXPECT_EQ(dqdkdv.scalar, K_ARGS_SCALE);
    EXPECT_EQ(dqdkdv.log2e, 1.44269504089F);
    EXPECT_EQ(dqdkdv.ratio, 4U);
    EXPECT_EQ(dqdkdv.seqlen_q, 200U);
    EXPECT_EQ(dqdkdv.seqlen_k, 300U);
    EXPECT_EQ(dqdkdv.nhead_q, 8U);
    EXPECT_EQ(dqdkdv.Hs_q, 204U);
    EXPECT_EQ(dqdkdv.BAs_q, 206U);
    EXPECT_EQ(dqdkdv.Seqs_q, 202U);
    EXPECT_EQ(dqdkdv.Hs_k, 404U);
    EXPECT_EQ(dqdkdv.BAs_k, 406U);
    EXPECT_EQ(dqdkdv.Seqs_k, 402U);
    EXPECT_EQ(dqdkdv.Hs_v, 604U);
    EXPECT_EQ(dqdkdv.BAs_v, 606U);
    EXPECT_EQ(dqdkdv.Seqs_v, 602U);
    EXPECT_EQ(dqdkdv.Hs_do, 1004U);
    EXPECT_EQ(dqdkdv.BAs_do, 1006U);
    EXPECT_EQ(dqdkdv.Seqs_do, 1002U);
    EXPECT_EQ(dqdkdv.Hs_dk, 1404U);
    EXPECT_EQ(dqdkdv.BAs_dk, 1406U);
    EXPECT_EQ(dqdkdv.Seqs_dk, 1402U);
    EXPECT_EQ(dqdkdv.Hs_dv, 1604U);
    EXPECT_EQ(dqdkdv.BAs_dv, 1606U);
    EXPECT_EQ(dqdkdv.Seqs_dv, 1602U);
    EXPECT_EQ(dqdkdv.Hs_lsed, 3608U);
    EXPECT_EQ(dqdkdv.ptr_qseq, nullptr);
    EXPECT_EQ(dqdkdv.ptr_kseq, nullptr);
    EXPECT_EQ(dqdkdv.ptr_qseq_padded, nullptr);
    EXPECT_EQ(dqdkdv.ptr_kseq_padded, nullptr);
}

// The post fields every case shares; each test pins the dq_acc strides.
void expectPostSharedArgs(const fmha_bwd_post_kernel_args& post,
                          const BwdBuffers& b,
                          uint32_t headDimQk)
{
    EXPECT_EQ(post.ptr_dq_acc, &b.dqAcc);
    EXPECT_EQ(post.ptr_dq, &b.dq);
    EXPECT_EQ(post.Hs_dq, 1204U);
    EXPECT_EQ(post.BAs_dq, 1206U);
    EXPECT_EQ(post.Seqs_dq, 1202U);
    EXPECT_EQ(post.seqlen_q, 200U);
    EXPECT_EQ(post.head_dim, headDimQk);
    EXPECT_EQ(post.ptr_qseq, nullptr);
    EXPECT_EQ(post.ptr_qseq_padded, nullptr);
}

} // namespace

// gfx942 A32, bottom-right sliding window: DQDKDV accumulates into an FP32
// dq_acc that DQ_CONVERT reads, and takes computeMaskCoordinates() output.
TEST(TestSdpaBwdArgsBuilder, Gfx942A32WindowArgs)
{
    SdpaBwdParams p = makeArgsParams("gfx942", AccumulatorType::A32, 128U, 192U);
    p.maskOrdinal = 3;
    p.windowLeft = 63;
    p.windowRight = 7;
    p.topLeftAlignment = false;
    BwdBuffers b{};
    const BwdStageArgs args = buildStageArgs(p, b, true);

    expectOdoArgs(args.odo, b, 128U);

    expectDqdkdvSharedArgs(args.dqdkdv, b);
    EXPECT_EQ(args.dqdkdv.ptr_dq, &b.dqAcc);
    EXPECT_EQ(args.dqdkdv.Ts, 77184U); // ts 192 * K row 402 bytes
    EXPECT_EQ(args.dqdkdv.head_dim_q, 128U);
    EXPECT_EQ(args.dqdkdv.head_dim_v, 128U);
    EXPECT_EQ(args.dqdkdv.max_seqlen_dq, 200U);
    // Bottom-right offsets S_kv - S_q = 100: {1 + 63 - 100, 1 + 7 + 100}
    EXPECT_EQ(args.dqdkdv.mask_y, -36);
    EXPECT_EQ(args.dqdkdv.mask_x, 108);

    // FP32 dq_acc [B = 2, H = 3, 200, 128]
    expectPostSharedArgs(args.post, b, 128U);
    EXPECT_EQ(args.post.Hs_dq_acc, 102400U);
    EXPECT_EQ(args.post.BAs_dq_acc, 819200U);
    EXPECT_EQ(args.post.Seqs_dq_acc, 512U);
}

// gfx950 A32, bottom-right sliding window: the window kernel takes the
// FlashAttention window sizes as they are.
TEST(TestSdpaBwdArgsBuilder, Gfx950A32WindowArgs)
{
    SdpaBwdParams p = makeArgsParams("gfx950", AccumulatorType::A32, 128U, 256U);
    p.maskOrdinal = asm_sdpa_engine::BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW;
    p.windowLeft = 63;
    p.windowRight = 7;
    p.topLeftAlignment = false;
    BwdBuffers b{};
    const BwdStageArgs args = buildStageArgs(p, b, true);

    expectOdoArgs(args.odo, b, 128U);

    expectDqdkdvSharedArgs(args.dqdkdv, b);
    EXPECT_EQ(args.dqdkdv.ptr_dq, &b.dqAcc);
    EXPECT_EQ(args.dqdkdv.Ts, 102912U); // ts 256 * K row 402 bytes
    EXPECT_EQ(args.dqdkdv.head_dim_q, 128U);
    EXPECT_EQ(args.dqdkdv.head_dim_v, 128U);
    EXPECT_EQ(args.dqdkdv.max_seqlen_dq, 200U);
    EXPECT_EQ(args.dqdkdv.mask_y, 63);
    EXPECT_EQ(args.dqdkdv.mask_x, 7);

    // FP32 dq_acc [B = 2, H = 3, 200, 128]
    expectPostSharedArgs(args.post, b, 128U);
    EXPECT_EQ(args.post.Hs_dq_acc, 102400U);
    EXPECT_EQ(args.post.BAs_dq_acc, 819200U);
    EXPECT_EQ(args.post.Seqs_dq_acc, 512U);
}

// gfx942 A16 has no post stage: DQDKDV writes dQ directly, with A16's
// row count, ceil16(S_q).
TEST(TestSdpaBwdArgsBuilder, Gfx942A16Args)
{
    const SdpaBwdParams p = makeArgsParams("gfx942", AccumulatorType::A16, 128U, 192U);
    BwdBuffers b{};
    const BwdStageArgs args = buildStageArgs(p, b, false);

    expectOdoArgs(args.odo, b, 128U);

    expectDqdkdvSharedArgs(args.dqdkdv, b);
    EXPECT_EQ(args.dqdkdv.ptr_dq, &b.dq);
    EXPECT_EQ(args.dqdkdv.Ts, 77184U);
    EXPECT_EQ(args.dqdkdv.head_dim_q, 128U);
    EXPECT_EQ(args.dqdkdv.head_dim_v, 128U);
    EXPECT_EQ(args.dqdkdv.max_seqlen_dq, 208U);
    EXPECT_EQ(args.dqdkdv.mask_y, 0);
    EXPECT_EQ(args.dqdkdv.mask_x, 0);
}

// gfx950 A16, D_qk = 192, D_v = 128: DQDKDV accumulates into a 2-byte dq_acc
// of ceil16(S_q) rows of 192 that DQ_SHUFFLE reads.
TEST(TestSdpaBwdArgsBuilder, Gfx950A16DqShuffleArgs)
{
    const SdpaBwdParams p = makeArgsParams("gfx950", AccumulatorType::A16, 192U, 192U);
    BwdBuffers b{};
    const BwdStageArgs args = buildStageArgs(p, b, true);

    expectOdoArgs(args.odo, b, 192U);

    expectDqdkdvSharedArgs(args.dqdkdv, b);
    EXPECT_EQ(args.dqdkdv.ptr_dq, &b.dqAcc);
    EXPECT_EQ(args.dqdkdv.Ts, 77184U);
    EXPECT_EQ(args.dqdkdv.head_dim_q, 192U);
    EXPECT_EQ(args.dqdkdv.head_dim_v, 128U);
    EXPECT_EQ(args.dqdkdv.max_seqlen_dq, 208U);
    EXPECT_EQ(args.dqdkdv.mask_y, 0);
    EXPECT_EQ(args.dqdkdv.mask_x, 0);

    // 2-byte dq_acc [B = 2, H = 3, 208, 192]
    expectPostSharedArgs(args.post, b, 192U);
    EXPECT_EQ(args.post.Hs_dq_acc, 79872U);
    EXPECT_EQ(args.post.BAs_dq_acc, 638976U);
    EXPECT_EQ(args.post.Seqs_dq_acc, 384U);
}
