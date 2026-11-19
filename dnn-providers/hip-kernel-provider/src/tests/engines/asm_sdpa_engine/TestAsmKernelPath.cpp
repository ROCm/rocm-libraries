// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/asm_sdpa_engine/asm/AsmKernelPath.hpp"

#include <gtest/gtest.h>

namespace asm_sdpa_engine::asm_kernels
{
namespace
{

TEST(TestAsmKernelPath, GetAsmKernelTocKeyStripsArchPrefix)
{
    EXPECT_EQ(getAsmKernelTocKey("gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co"),
              "fmha_v3_bwd/bwd_hd128_odo_bf16.co");
}

TEST(TestAsmKernelPath, GetAsmKernelTocKeyStripsArchPrefixWithVariant)
{
    EXPECT_EQ(getAsmKernelTocKey("gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co"),
              "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co");
}

TEST(TestAsmKernelPath, GetAsmKernelTocKeyStripsGfx950Prefix)
{
    EXPECT_EQ(getAsmKernelTocKey("gfx950/fmha_v3_fwd/fwd_hd128_bf16.co"),
              "fmha_v3_fwd/fwd_hd128_bf16.co");
}

TEST(TestAsmKernelPath, GetAsmKernelTocKeyNoSlashReturnsInput)
{
    EXPECT_EQ(getAsmKernelTocKey("kernel_name_only.co"), "kernel_name_only.co");
}

TEST(TestAsmKernelPath, GetAsmKernelTocKeyEmptyStringReturnsEmpty)
{
    EXPECT_EQ(getAsmKernelTocKey(""), "");
}

// gfx942 keys pick the build for the device family; other archs have only one.
TEST(TestAsmKernelPath, GetFwdKernelTocKeySelectsDeviceBuild)
{
    EXPECT_EQ(getFwdKernelTocKey("gfx942/fmha_v3_fwd/fwd_hd128_bf16_rtne.co", "gfx942", false),
              "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co");
    EXPECT_EQ(getFwdKernelTocKey("gfx942/fmha_v3_fwd/fwd_hd128_bf16_rtne.co", "gfx942", true),
              "fmha_v3_fwd/MI308/fwd_hd128_bf16_rtne.co");
    EXPECT_EQ(getFwdKernelTocKey("gfx950/fmha_v3_fwd/fwd_hd128_bf16.co", "gfx950", true),
              "fmha_v3_fwd/fwd_hd128_bf16.co");
}

} // namespace
} // namespace asm_sdpa_engine::asm_kernels
