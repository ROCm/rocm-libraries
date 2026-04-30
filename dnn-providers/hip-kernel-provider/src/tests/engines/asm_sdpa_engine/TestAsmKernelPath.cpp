// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <gtest/gtest.h>

// Define AITER_ASM_DIR for the test if not already defined
#ifndef AITER_ASM_DIR
#define AITER_ASM_DIR "/test/default/asm/dir"
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

    std::string dir = getAsmKernelDir();
    EXPECT_EQ(dir, AITER_ASM_DIR);
}

TEST(TestAsmKernelPath, GetAsmKernelDirReturnsEnvVarWhenSet)
{
    const char* customDir = "/custom/kernel/path";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_AITER_ASM_DIR", customDir);

    std::string dir = getAsmKernelDir();
    EXPECT_EQ(dir, customDir);

    // Clean up
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");
}

TEST(TestAsmKernelPath, GetAsmKernelDirIgnoresEmptyEnvVar)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_AITER_ASM_DIR", "");

    std::string dir = getAsmKernelDir();
    EXPECT_EQ(dir, AITER_ASM_DIR);

    // Clean up
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");
}

TEST(TestAsmKernelPath, GetAsmKernelPathAppendsFilename)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");

    std::string path = getAsmKernelPath("kernel.co");
    std::string expected = std::string(AITER_ASM_DIR) + "/kernel.co";
    EXPECT_EQ(path, expected);
}

TEST(TestAsmKernelPath, GetAsmKernelPathUsesEnvVarDir)
{
    const char* customDir = "/env/kernels";
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_AITER_ASM_DIR", customDir);

    std::string path = getAsmKernelPath("gfx942/test.co");
    EXPECT_EQ(path, "/env/kernels/gfx942/test.co");

    // Clean up
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_AITER_ASM_DIR");
}

} // namespace
} // namespace asm_sdpa_engine::asm_kernels
