// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>

#include <gtest/gtest.h>

#include "engines/asm_sdpa_engine/asm/SdpaBwdKernelArgs.hpp"

using asm_sdpa_engine::fmha_bwd_dqdkdv_args;
using asm_sdpa_engine::fmha_bwd_odo_args;
using asm_sdpa_engine::fmha_bwd_post_kernel_args;

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
