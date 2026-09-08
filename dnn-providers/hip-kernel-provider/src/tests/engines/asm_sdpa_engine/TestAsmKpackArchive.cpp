// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/asm_sdpa_engine/asm/AsmKernelPath.hpp"
#include "engines/asm_sdpa_engine/asm/AsmKpackArchive.hpp"

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace asm_sdpa_engine::asm_kernels
{
namespace
{

// =============================================================================
// Error mapping
// =============================================================================

TEST(TestAsmKpackArchive, GetKernelThrowsOnMissingArchive)
{
    // Opening a non-existent archive (bogus arch) should throw
    // HipdnnPluginException from kpack_open failure.
    EXPECT_THROW(
        {
            auto& archive = AsmKpackArchive::instance();
            archive.getKernel("nonexistent/kernel.co", "gfx_bogus_arch_999");
        },
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// =============================================================================
// TOC key derivation correctness
// =============================================================================
// These tests validate that the forward builder's MI300/MI308 variant insertion
// followed by getAsmKernelTocKey produces the correct TOC keys that match the
// packer's index.

TEST(TestAsmKpackArchive, FwdTocKeyMi300Variant)
{
    // Forward builder: insert MI300 variant then strip arch prefix
    std::string coName = "gfx942/fmha_v3_fwd/fwd_hd128_bf16_rtne.co";
    auto pos = coName.rfind('/');
    coName = coName.substr(0, pos + 1) + "MI300/" + coName.substr(pos + 1);
    EXPECT_EQ(getAsmKernelTocKey(coName), "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co");
}

TEST(TestAsmKpackArchive, FwdTocKeyMi308Variant)
{
    std::string coName = "gfx942/fmha_v3_fwd/fwd_hd128_bf16_rtne.co";
    auto pos = coName.rfind('/');
    coName = coName.substr(0, pos + 1) + "MI308/" + coName.substr(pos + 1);
    EXPECT_EQ(getAsmKernelTocKey(coName), "fmha_v3_fwd/MI308/fwd_hd128_bf16_rtne.co");
}

TEST(TestAsmKpackArchive, BwdTocKeyNoVariant)
{
    // Backward builder: no variant insertion, just strip arch prefix
    EXPECT_EQ(getAsmKernelTocKey("gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co"),
              "fmha_v3_bwd/bwd_hd128_odo_bf16.co");
}

TEST(TestAsmKpackArchive, Gfx950TocKeyNoVariant)
{
    // gfx950: no MI300/MI308 variant insertion
    EXPECT_EQ(getAsmKernelTocKey("gfx950/fmha_v3_fwd/fwd_hd128_bf16.co"),
              "fmha_v3_fwd/fwd_hd128_bf16.co");
}

TEST(TestAsmKpackArchive, TocKeyNoSlashReturnsUnchanged)
{
    EXPECT_EQ(getAsmKernelTocKey("kernel.co"), "kernel.co");
}

TEST(TestAsmKpackArchive, TocKeyNonGfxPrefixReturnsUnchanged)
{
    // A path whose first component is not a gfx arch should be returned as-is.
    EXPECT_EQ(getAsmKernelTocKey("fmha_v3_fwd/fwd_hd128_bf16.co"),
              "fmha_v3_fwd/fwd_hd128_bf16.co");
}

} // namespace
} // namespace asm_sdpa_engine::asm_kernels
