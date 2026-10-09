// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Pins where the ASM SDPA engine looks for its .kpack archives (ALMIOPEN-2766):
// HIPDNN_ASM_SDPA_KPACK_DIR when it names a directory, else beside the loaded
// module. There is deliberately no configure-time third step -- a baked install
// prefix is what made relocated installs fail with hipError 301.

#include "engines/asm_sdpa_engine/asm/AsmKpackArchive.hpp"
#include "engines/asm_sdpa_engine/asm/PluginModuleDir.hpp"

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

namespace asm_sdpa_engine::asm_kernels
{
namespace
{

using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

constexpr const char* SCRATCH_LABEL = "asmkpackpath";

/// What step 2 should answer, computed independently of the resolver: the directory
/// of the module that contains the resolver's own code.
std::filesystem::path moduleRelativeRoot()
{
    return hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
               reinterpret_cast<const void*>(&asmSdpaKpackRoot))
           / "arch_content" / "asm_sdpa";
}

/// The override has to beat the module-relative path, or there is no way to point a
/// shipped plugin at a different set of archives.
TEST(TestAsmKpackPath, PrefersEnvOverrideNamingADirectory)
{
    const ScopedDirectory existing = claimScratchDirectory(SCRATCH_LABEL);
    const ScopedEnvironmentVariableSetter override(ASM_SDPA_KPACK_DIR_ENV,
                                                   existing.path().string());

    EXPECT_EQ(asmSdpaKpackRoot(), existing.path());
}

/// A stale override is ignored, not obeyed: a path left over from another machine or
/// install must not shadow archives sitting next to the plugin.
TEST(TestAsmKpackPath, IgnoresEnvOverrideThatDoesNotExist)
{
    const ScopedEnvironmentVariableSetter stale(ASM_SDPA_KPACK_DIR_ENV, "/nowhere/in/particular");

    EXPECT_EQ(asmSdpaKpackRoot(), moduleRelativeRoot());
}

/// Pointing the override at an archive instead of its directory is the likely typo.
TEST(TestAsmKpackPath, IgnoresEnvOverrideNamingAFile)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const auto file = scratch.path() / "hip_kernel_provider_sdpa_gfx942.kpack";
    std::ofstream(file).put('\0');
    ASSERT_TRUE(std::filesystem::is_regular_file(file));
    const ScopedEnvironmentVariableSetter override(ASM_SDPA_KPACK_DIR_ENV, file.string());

    EXPECT_EQ(asmSdpaKpackRoot(), moduleRelativeRoot());
}

/// With no override the root follows the loaded module, wherever it was loaded from --
/// the property a relocated install depends on.
TEST(TestAsmKpackPath, FallsBackToTheLoadedModuleDirectory)
{
    const ScopedEnvironmentVariableSetter unset(ASM_SDPA_KPACK_DIR_ENV, "");

    const auto resolved = asmSdpaKpackRoot();

    EXPECT_TRUE(resolved.is_absolute()) << resolved;
    EXPECT_EQ(resolved, moduleRelativeRoot());
    EXPECT_EQ(resolved, currentPluginDirectory() / "arch_content" / "asm_sdpa");
}

/// The per-arch layout must match what the build copies and installs:
/// <root>/<arch>/hip_kernel_provider_sdpa_<arch>.kpack.
TEST(TestAsmKpackPath, ArchivePathAppendsArchDirectoryAndFileName)
{
    const ScopedDirectory existing = claimScratchDirectory(SCRATCH_LABEL);
    const ScopedEnvironmentVariableSetter override(ASM_SDPA_KPACK_DIR_ENV,
                                                   existing.path().string());

    EXPECT_EQ(asmSdpaKpackArchivePath("gfx942"),
              existing.path() / "gfx942" / "hip_kernel_provider_sdpa_gfx942.kpack");
    EXPECT_EQ(asmSdpaKpackArchivePath("gfx950"),
              existing.path() / "gfx950" / "hip_kernel_provider_sdpa_gfx950.kpack");
}

/// AsmKpackArchive must open what the resolver names, not a path of its own. The
/// scratch root is empty, so the open fails and the error names the attempted path.
TEST(TestAsmKpackPath, ArchiveOpenUsesTheResolvedPath)
{
    const ScopedDirectory existing = claimScratchDirectory(SCRATCH_LABEL);
    const ScopedEnvironmentVariableSetter override(ASM_SDPA_KPACK_DIR_ENV,
                                                   existing.path().string());
    // An arch no other test opens: the singleton caches successfully opened archives.
    const std::string arch = "gfx_kpack_path_test";

    try
    {
        AsmKpackArchive::instance().getKernel("fmha_v3_fwd/missing.co", arch);
        FAIL() << "expected kpack_open to fail for an empty archive root";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_NE(std::string(error.what()).find(asmSdpaKpackArchivePath(arch).string()),
                  std::string::npos)
            << error.what();
    }
}

} // namespace
} // namespace asm_sdpa_engine::asm_kernels
