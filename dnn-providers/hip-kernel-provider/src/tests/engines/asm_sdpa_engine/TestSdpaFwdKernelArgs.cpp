// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <unordered_map>

#include <gtest/gtest.h>

#include "engines/asm_sdpa_engine/asm/SdpaFwdKernelArgs.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdArgsBuilder.hpp"

using asm_sdpa_engine::buildFwdKernelArgs;
using asm_sdpa_engine::fmha_fwd_v3_args;
using asm_sdpa_engine::SdpaFwdParams;

// Verify the packed struct has the exact binary layout expected by the
// pre-compiled ASM kernel.  Each field must sit at the byte offset that
// the GPU kernel reads from.

TEST(TestSdpaFwdKernelArgs, TotalSizeMatches)
{
    EXPECT_EQ(sizeof(fmha_fwd_v3_args), 656u);
}

TEST(TestSdpaFwdKernelArgs, PointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_o), 0u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_q), 16u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_k), 32u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_v), 48u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_lse), 64u);
}

TEST(TestSdpaFwdKernelArgs, ScalarFieldOffset)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, scalar), 80u);
}

TEST(TestSdpaFwdKernelArgs, QueryStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_seq_len), 96u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_Seqs), 112u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_Ts), 128u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_Hs), 144u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_Bs), 160u);
}

TEST(TestSdpaFwdKernelArgs, GqaAndKeyStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_gqa), 176u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_k_Seqs), 192u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_k_Hs), 208u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_k_Bs), 224u);
}

TEST(TestSdpaFwdKernelArgs, OptionAndFlagFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_opt), 240u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_lse), 256u);
}

TEST(TestSdpaFwdKernelArgs, DimensionFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_kv_seq_len), 272u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_qk_head_dim), 288u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_v_head_dim), 304u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_q_head_num), 320u);
}

TEST(TestSdpaFwdKernelArgs, ValueStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_v_Seqs), 336u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_v_Hs), 352u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_v_Bs), 368u);
}

TEST(TestSdpaFwdKernelArgs, OutputStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_o_Seqs), 384u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_o_Hs), 400u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_o_Bs), 416u);
}

TEST(TestSdpaFwdKernelArgs, SequencePointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_qseq), 432u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_kseq), 448u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_lse_Hs), 464u);
}

TEST(TestSdpaFwdKernelArgs, PaddingSequencePointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_qseq_padding), 480u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_kseq_padding), 496u);
}

TEST(TestSdpaFwdKernelArgs, DescalePointerFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_q_descale), 512u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_k_descale), 528u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, ptr_v_descale), 544u);
}

TEST(TestSdpaFwdKernelArgs, DescaleStrideFieldOffsets)
{
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_descale_q_Bs), 560u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_descale_q_Hs), 576u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_descale_k_Bs), 592u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_descale_k_Hs), 608u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_descale_v_Bs), 624u);
    EXPECT_EQ(offsetof(fmha_fwd_v3_args, s_descale_v_Hs), 640u);
}

// ---------------------------------------------------------------------------
// buildFwdKernelArgs: pure kernel-argument construction (no GPU required).
// ---------------------------------------------------------------------------

namespace
{
// Params with distinct strides so byte-scaling is observable; batch mode, no mask.
SdpaFwdParams makeBaseParams()
{
    SdpaFwdParams params{};
    params.qUid = 1;
    params.kUid = 2;
    params.vUid = 3;
    params.oUid = 4;
    params.batchSize = 2;
    params.numHeadsQ = 8;
    params.numHeadsKv = 8;
    params.seqLenQ = 256;
    params.seqLenKv = 128;
    params.headDimQk = 128;
    params.headDimV = 128;
    params.qStrideSeq = 128;
    params.qStrideRow = 128;
    params.qStrideHead = 1000;
    params.qStrideBatch = 2000;
    params.kStrideSeq = 128;
    params.kStrideHead = 1100;
    params.kStrideBatch = 2100;
    params.vStrideSeq = 128;
    params.vStrideHead = 1200;
    params.vStrideBatch = 2200;
    params.oStrideSeq = 128;
    params.oStrideHead = 1300;
    params.oStrideBatch = 2300;
    params.tileSizeQo = 256;
    // attnScale (a ScalarOperand) is resolved at execute, not by buildFwdKernelArgs,
    // so it is left default-constructed here; these tests never read args.scalar.
    params.archString = "gfx942";
    params.maskType = asm_sdpa_engine::plan_utils::MaskType::NO_MASK;
    return params;
}
} // namespace

TEST(TestSdpaFwdKernelArgs, BuildArgsBf16UsesTwoByteStridesAndNoDescale)
{
    SdpaFwdParams params = makeBaseParams();
    params.inBytesPerElement = 2;

    int qBuf = 0;
    int kBuf = 0;
    int vBuf = 0;
    int oBuf = 0;
    const std::unordered_map<int64_t, void*> uidToPtrMap{
        {1, &qBuf}, {2, &kBuf}, {3, &vBuf}, {4, &oBuf}};

    const fmha_fwd_v3_args args = buildFwdKernelArgs(params, uidToPtrMap);

    EXPECT_EQ(args.ptr_q, &qBuf);
    EXPECT_EQ(args.ptr_o, &oBuf);
    // Input strides scaled by 2 bytes.
    EXPECT_EQ(args.s_Seqs, params.qStrideSeq * 2u);
    EXPECT_EQ(args.s_Hs, params.qStrideHead * 2u);
    EXPECT_EQ(args.s_k_Bs, params.kStrideBatch * 2u);
    EXPECT_EQ(args.s_v_Hs, params.vStrideHead * 2u);
    // Output strides scaled by 2 bytes (BF16).
    EXPECT_EQ(args.s_o_Seqs, params.oStrideSeq * 2u);
    // No descales on the bf16 path.
    EXPECT_EQ(args.ptr_q_descale, nullptr);
    EXPECT_EQ(args.ptr_k_descale, nullptr);
    EXPECT_EQ(args.ptr_v_descale, nullptr);
}

TEST(TestSdpaFwdKernelArgs, BuildArgsFp8UsesOneByteInputStridesAndWiresDescales)
{
    SdpaFwdParams params = makeBaseParams();
    params.inBytesPerElement = 1;
    params.qDescaleUid = 5;
    params.kDescaleUid = 6;
    params.vDescaleUid = 7;

    int qBuf = 0;
    int kBuf = 0;
    int vBuf = 0;
    int oBuf = 0;
    int dqBuf = 0;
    int dkBuf = 0;
    int dvBuf = 0;
    const std::unordered_map<int64_t, void*> uidToPtrMap{
        {1, &qBuf}, {2, &kBuf}, {3, &vBuf}, {4, &oBuf}, {5, &dqBuf}, {6, &dkBuf}, {7, &dvBuf}};

    const fmha_fwd_v3_args args = buildFwdKernelArgs(params, uidToPtrMap);

    // FP8 input strides scaled by 1 byte.
    EXPECT_EQ(args.s_Seqs, params.qStrideSeq * 1u);
    EXPECT_EQ(args.s_k_Seqs, params.kStrideSeq * 1u);
    EXPECT_EQ(args.s_v_Hs, params.vStrideHead * 1u);
    // Output remains BF16 (2 bytes) even for fp8 inputs.
    EXPECT_EQ(args.s_o_Seqs, params.oStrideSeq * 2u);
    EXPECT_EQ(args.s_o_Hs, params.oStrideHead * 2u);
    // Descale pointers wired from the variant pack.
    EXPECT_EQ(args.ptr_q_descale, &dqBuf);
    EXPECT_EQ(args.ptr_k_descale, &dkBuf);
    EXPECT_EQ(args.ptr_v_descale, &dvBuf);
    // Per-tensor descales carry zero strides.
    EXPECT_EQ(args.s_descale_q_Bs, 0u);
    EXPECT_EQ(args.s_descale_q_Hs, 0u);
    EXPECT_EQ(args.s_descale_v_Hs, 0u);
}
