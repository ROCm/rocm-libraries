// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Real kpack archive loading tests. These open the build-tree .kpack archives
// and extract known TOC keys to verify end-to-end packing + loading for each
// supported architecture and kernel variant.
//
// The tests use the kpack C API directly (not AsmKpackArchive singleton) so
// they are independent of the plugin directory resolution and can run as unit
// tests against the build-tree archives.

#include <rocm_kpack/kpack.h>

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#ifndef ASM_KPACK_TEST_DIR
#error "ASM_KPACK_TEST_DIR must be defined (set via CMake compile definition)"
#endif

namespace asm_sdpa_engine::asm_kernels
{
namespace
{

/// Helper: open an archive, extract a kernel by TOC key + arch, verify non-empty.
void verifyKernelExtraction(const std::string& archivePath,
                            const std::string& tocKey,
                            const std::string& arch)
{
    SCOPED_TRACE("archive=" + archivePath + " tocKey=" + tocKey + " arch=" + arch);

    ASSERT_TRUE(std::filesystem::exists(archivePath))
        << "Archive not found: " << archivePath;

    kpack_archive_t archive = nullptr;
    kpack_error_t err = kpack_open(archivePath.c_str(), &archive);
    ASSERT_EQ(err, KPACK_SUCCESS) << "kpack_open failed for " << archivePath;

    void* data = nullptr;
    size_t size = 0;
    err = kpack_get_kernel(archive, tocKey.c_str(), arch.c_str(), &data, &size);
    EXPECT_EQ(err, KPACK_SUCCESS)
        << "kpack_get_kernel failed for tocKey='" << tocKey << "' arch='" << arch << "'";
    EXPECT_NE(data, nullptr);
    EXPECT_GT(size, 0u);

    if(data)
    {
        kpack_free_kernel(data);
    }
    kpack_close(archive);
}

std::string kpackPath(const std::string& arch)
{
    return std::string(ASM_KPACK_TEST_DIR) + "/hip_kernel_provider_sdpa_" + arch + ".kpack";
}

// =============================================================================
// gfx942 forward kernels — MI300 and MI308 variants
// =============================================================================

TEST(TestAsmKpackLoading, Gfx942FwdMi300Hd128)
{
    verifyKernelExtraction(
        kpackPath("gfx942"), "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co", "gfx942");
}

TEST(TestAsmKpackLoading, Gfx942FwdMi308Hd128)
{
    verifyKernelExtraction(
        kpackPath("gfx942"), "fmha_v3_fwd/MI308/fwd_hd128_bf16_rtne.co", "gfx942");
}

TEST(TestAsmKpackLoading, Gfx942FwdMi300CausalHd128)
{
    verifyKernelExtraction(
        kpackPath("gfx942"), "fmha_v3_fwd/MI300/fwd_hd128_bf16_causal_rtne.co", "gfx942");
}

TEST(TestAsmKpackLoading, Gfx942FwdMi308CausalHd128)
{
    verifyKernelExtraction(
        kpackPath("gfx942"), "fmha_v3_fwd/MI308/fwd_hd128_bf16_causal_rtne.co", "gfx942");
}

// =============================================================================
// gfx942 backward kernels — no MI300/MI308 variant
// =============================================================================

TEST(TestAsmKpackLoading, Gfx942BwdHd128Odo)
{
    verifyKernelExtraction(
        kpackPath("gfx942"), "fmha_v3_bwd/bwd_hd128_odo_bf16.co", "gfx942");
}

TEST(TestAsmKpackLoading, Gfx942BwdHd64Odo)
{
    verifyKernelExtraction(
        kpackPath("gfx942"), "fmha_v3_bwd/bwd_hd64_odo_bf16.co", "gfx942");
}

// =============================================================================
// gfx950 forward kernels — no MI300/MI308 variants
// =============================================================================

TEST(TestAsmKpackLoading, Gfx950FwdHd128)
{
    verifyKernelExtraction(
        kpackPath("gfx950"), "fmha_v3_fwd/fwd_hd128_bf16.co", "gfx950");
}

// =============================================================================
// gfx950 backward kernels
// =============================================================================

TEST(TestAsmKpackLoading, Gfx950BwdHd64Odo)
{
    verifyKernelExtraction(
        kpackPath("gfx950"), "fmha_v3_bwd/bwd_hd64_odo_bf16.co", "gfx950");
}

// =============================================================================
// Negative: valid archive, missing TOC key
// =============================================================================

TEST(TestAsmKpackLoading, MissingTocKeyReturnsError)
{
    auto path = kpackPath("gfx942");
    if(!std::filesystem::exists(path))
    {
        GTEST_SKIP() << "gfx942 archive not found at " << path;
    }

    kpack_archive_t archive = nullptr;
    kpack_error_t err = kpack_open(path.c_str(), &archive);
    ASSERT_EQ(err, KPACK_SUCCESS);

    void* data = nullptr;
    size_t size = 0;
    err = kpack_get_kernel(archive, "nonexistent/bogus_kernel.co", "gfx942", &data, &size);
    EXPECT_NE(err, KPACK_SUCCESS) << "Expected failure for missing TOC key";
    EXPECT_EQ(data, nullptr);

    kpack_close(archive);
}

} // namespace
} // namespace asm_sdpa_engine::asm_kernels
