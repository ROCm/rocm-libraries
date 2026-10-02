// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <gtest/gtest.h>

#include <filesystem>

// Define AITER_ASM_DIR / AITER_ASM_SUBDIR for the test if not already defined
#ifndef AITER_ASM_DIR
#define AITER_ASM_DIR "/test/default/asm/dir"
#endif
#ifndef AITER_ASM_SUBDIR
#define AITER_ASM_SUBDIR "hkp_test_asm_kernels_beside_module"
#endif

#include "engines/asm_sdpa_engine/asm/AsmKernelPath.hpp"

namespace asm_sdpa_engine::asm_kernels
{
namespace
{

TEST(TestAsmKernelPath, GetAsmKernelDirReturnsCompileTimeDefault)
{
    // Ensure the env var is unset for this test
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");

    const std::string dir = getAsmKernelDir();
    EXPECT_EQ(dir, AITER_ASM_DIR);
}

/// The tree beside the loaded module wins over the configure-time path, so an install that
/// was moved (or unpacked from a package built under another prefix) finds its own .co
/// files instead of failing every launch with hipErrorFileNotFound. In this test binary the
/// "module" is the executable; the subdir name is test-only, so nothing else creates it.
TEST(TestAsmKernelPath, GetAsmKernelDirPrefersTheTreeBesideTheLoadedModule)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");

    const auto besideModule = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
                                  reinterpret_cast<const void*>(&getAsmKernelDir))
                              / AITER_ASM_SUBDIR;
    std::filesystem::create_directories(besideModule);

    const std::string dir = getAsmKernelDir();
    // The env override still beats it.
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_AITER_ASM_DIR", "/custom/kernel/path");
    const std::string overridden = getAsmKernelDir();
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");

    std::filesystem::remove_all(besideModule);

    EXPECT_EQ(dir, besideModule.string());
    EXPECT_EQ(overridden, "/custom/kernel/path");
}

TEST(TestAsmKernelPath, GetAsmKernelDirReturnsEnvVarWhenSet)
{
    const char* customDir = "/custom/kernel/path";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_AITER_ASM_DIR", customDir);

    const std::string dir = getAsmKernelDir();
    EXPECT_EQ(dir, customDir);

    // Clean up
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");
}

TEST(TestAsmKernelPath, GetAsmKernelDirIgnoresEmptyEnvVar)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_AITER_ASM_DIR", "");

    const std::string dir = getAsmKernelDir();
    EXPECT_EQ(dir, AITER_ASM_DIR);

    // Clean up
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");
}

TEST(TestAsmKernelPath, GetAsmKernelPathAppendsFilename)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");

    const std::string path = getAsmKernelPath("kernel.co");
    const std::string expected = std::string(AITER_ASM_DIR) + "/kernel.co";
    EXPECT_EQ(path, expected);
}

TEST(TestAsmKernelPath, GetAsmKernelPathUsesEnvVarDir)
{
    const char* customDir = "/env/kernels";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_AITER_ASM_DIR", customDir);

    const std::string path = getAsmKernelPath("gfx942/test.co");
    EXPECT_EQ(path, "/env/kernels/gfx942/test.co");

    // Clean up
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");
}

} // namespace
} // namespace asm_sdpa_engine::asm_kernels
