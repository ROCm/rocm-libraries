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
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "asm_fmha_v3_fwd_configs.hpp"
#include "engines/asm_sdpa_engine/asm/AsmKernelPath.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdLaunchParams.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaKernelUtils.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

#ifndef ASM_KPACK_TEST_DIR
#error "ASM_KPACK_TEST_DIR must be defined (set via CMake compile definition)"
#endif

#ifndef ASM_KPACK_INSTALL_DIR
#define ASM_KPACK_INSTALL_DIR ""
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

    if(!std::filesystem::exists(archivePath))
    {
        GTEST_SKIP() << "Archive not found (arch may not be built in this artifact): "
                     << archivePath;
    }

    kpack_archive_t archive = nullptr;
    kpack_error_t err = kpack_open(archivePath.c_str(), &archive);
    ASSERT_EQ(err, KPACK_SUCCESS) << "kpack_open failed for " << archivePath;

    void* data = nullptr;
    size_t size = 0;
    err = kpack_get_kernel(archive, tocKey.c_str(), arch.c_str(), &data, &size);
    EXPECT_EQ(err, KPACK_SUCCESS) << "kpack_get_kernel failed for tocKey='" << tocKey << "' arch='"
                                  << arch << "'";
    EXPECT_NE(data, nullptr);
    EXPECT_GT(size, 0u);

    if(data != nullptr)
    {
        kpack_free_kernel(data);
    }
    kpack_close(archive);
}

/// Return the directory containing the per-arch kpack archives. Prefers the
/// build-tree path; falls back to an exe-relative installed layout path so that
/// TheRock CI (which runs from an install prefix) can locate the archives too.
std::string kpackDir()
{
    // Build-tree path (works for local developer builds)
    if(std::error_code ec; std::filesystem::is_directory(ASM_KPACK_TEST_DIR, ec))
    {
        return ASM_KPACK_TEST_DIR;
    }
    // Installed layout: resolve relative to the test executable
    //   <prefix>/bin/<exe> → <prefix>/<ASM_KPACK_INSTALL_DIR>
    if(const std::string_view installDir{ASM_KPACK_INSTALL_DIR}; !installDir.empty())
    {
        auto candidate = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / ".."
                         / std::string(installDir);
        std::error_code ec;
        if(std::filesystem::is_directory(candidate, ec))
        {
            return candidate.string();
        }
    }
    // Return build-tree path so the assertion reports the expected location
    return ASM_KPACK_TEST_DIR;
}

std::string kpackPath(const std::string& arch)
{
    return kpackDir() + "/" + arch + "/hip_kernel_provider_sdpa_" + arch + ".kpack";
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
    verifyKernelExtraction(kpackPath("gfx942"), "fmha_v3_bwd/bwd_hd128_odo_bf16.co", "gfx942");
}

TEST(TestAsmKpackLoading, Gfx942BwdHd64Odo)
{
    verifyKernelExtraction(kpackPath("gfx942"), "fmha_v3_bwd/bwd_hd64_odo_bf16.co", "gfx942");
}

// =============================================================================
// gfx950 forward kernels — no MI300/MI308 variants
// =============================================================================

TEST(TestAsmKpackLoading, Gfx950FwdHd128)
{
    verifyKernelExtraction(kpackPath("gfx950"), "fmha_v3_fwd/fwd_hd128_bf16.co", "gfx950");
}

// =============================================================================
// gfx950 backward kernels
// =============================================================================

TEST(TestAsmKpackLoading, Gfx950BwdHd64Odo)
{
    verifyKernelExtraction(kpackPath("gfx950"), "fmha_v3_bwd/bwd_hd64_odo_bf16.co", "gfx950");
}

// =============================================================================
// Forward launch limits — every config row for the current device
// =============================================================================
//
// Loads each forward kernel the config table offers for this device and checks
// that the block size the plan launches with is within the kernel's own
// max-threads-per-block limit. A row whose kernel cannot take that block passes
// the table lookup but fails at hipModuleLaunchKernel, which is how the gfx950
// hd192x128 kernel (limit 256) was once launched with 512 threads.

/// TOC keys for one forward config row. gfx942 ships MI300 and MI308 builds of
/// each kernel and the builder picks one by PCI chip ID, so both are checked.
std::vector<std::string> fwdTocKeys(const fmha_v3_fwdConfig& config)
{
    const auto tocKey = getAsmKernelTocKey(config.co_name);
    if(config.arch != "gfx942")
    {
        return {tocKey};
    }
    const auto pos = tocKey.rfind('/');
    return {tocKey.substr(0, pos + 1) + "MI300/" + tocKey.substr(pos + 1),
            tocKey.substr(0, pos + 1) + "MI308/" + tocKey.substr(pos + 1)};
}

TEST(TestGpuSdpaFwdLaunchLimits, EveryConfigRowFitsKernelMaxThreads)
{
    SKIP_IF_NO_DEVICES();

    const auto arch = hip_kernel_provider_common::getDeviceString(nullptr);
    const auto path = kpackPath(arch);
    if(!std::filesystem::exists(path))
    {
        GTEST_SKIP() << "No kpack archive for " << arch << " at " << path;
    }

    kpack_archive_t rawArchive = nullptr;
    ASSERT_EQ(kpack_open(path.c_str(), &rawArchive), KPACK_SUCCESS) << path;
    const auto archive
        = std::unique_ptr<std::remove_pointer_t<kpack_archive_t>, decltype(&kpack_close)>(
            rawArchive, &kpack_close);

    size_t checked = 0;
    for(const auto& [name, config] : cfg_fmha_fwd)
    {
        if(config.arch != arch)
        {
            continue;
        }

        SdpaFwdParams params{};
        params.batchSize = 1U;
        params.numHeadsQ = 1U;
        params.seqLenQ = static_cast<unsigned int>(config.ts_qo);
        params.headDimQk = static_cast<unsigned int>(config.hdim_q);
        params.headDimV = static_cast<unsigned int>(config.hdim_v);
        params.tileSizeQo = static_cast<unsigned int>(config.ts_qo);
        params.archString = arch;
        params.maskType = static_cast<plan_utils::MaskType>(config.mask);
        const auto blockDimX = computeFwdLaunchParams(params).blockDimX;

        for(const auto& tocKey : fwdTocKeys(config))
        {
            SCOPED_TRACE("tocKey=" + tocKey + " kernel=" + config.knl_name);

            void* rawData = nullptr;
            size_t size = 0;
            ASSERT_EQ(
                kpack_get_kernel(archive.get(), tocKey.c_str(), arch.c_str(), &rawData, &size),
                KPACK_SUCCESS);
            const auto data
                = std::unique_ptr<void, decltype(&kpack_free_kernel)>(rawData, &kpack_free_kernel);

            const auto kernel
                = loadKernelModuleFromMemory(data.get(), size, config.knl_name.c_str());
            ASSERT_TRUE(kernel.has_value());

            const auto maxThreads = getMaxThreadsPerBlock(kernel->function());
            ASSERT_TRUE(maxThreads.has_value());
            EXPECT_LE(blockDimX, *maxThreads);
            ++checked;
        }
    }

    if(checked == 0)
    {
        GTEST_SKIP() << "No forward config rows for " << arch;
    }
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
