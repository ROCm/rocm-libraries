// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>

#include <gtest/gtest.h>

#include "engines/asm_sdpa_engine/asm/SdpaFwdKernelArgs.hpp"

using asm_sdpa_engine::fmha_fwd_v3_args;

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
