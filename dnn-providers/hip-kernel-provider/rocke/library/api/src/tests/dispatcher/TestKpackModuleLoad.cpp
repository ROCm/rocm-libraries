// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Proves that rocke_client carries a direct DT_NEEDED on librocm_kpack.so.0:
// KpackModuleLoader.cpp (compiled into rocke_client_impl, and therefore into
// the rocke_client SHARED plugin) calls kpack_open/kpack_get_kernel, which
// forces the dynamic-linker entry even under --as-needed + --exclude-libs=ALL.
//
// Fixture layout (committed; copied beside the test binary by CMake POST_BUILD):
//   <testModuleDir>/arch_content/rocke/gfx942/rocke_client_gfx942.kpack
// The kpack contains a real gfx942 HSACO (ELF64, AMD GPU, rocke_test_probe),
// compiled cross-arch with amdclang++ --offload-arch=gfx942 + unbundled, and
// packed with zstd compression by the Python rocm_kpack library.
//
// Host tier (always runs):
//   Resolves the fixture at arch_content/rocke/gfx942/rocke_client_gfx942.kpack
//   relative to testModuleDir(). Calls kpack_open + kpack_get_kernel with the
//   well-known toc_key and arch "gfx942" (fixture-fixed, independent of the
//   running device). Asserts KPACK_SUCCESS, size > 0, ELF magic \x7fELF.
//   Skips if the fixture is absent (e.g., stripped install).
//
// GPU tier (device-gated):
//   Queries the running device arch (gcnArchName, colon-stripped). Looks for
//   arch_content/rocke/<arch>/rocke_client_<arch>.kpack. On gfx942 runners this
//   matches the committed fixture and does a real hipModuleLoadData + GetFunction.
//   Other arches skip (no fixture shipped for them). Also skips when no device.

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include <hip/hip_runtime_api.h>
#include <rocm_kpack/kpack.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "dispatcher/KpackModuleLoader.hpp"

namespace rocke_client::dispatcher
{
namespace
{

// ---- testModuleDir -------------------------------------------------------
//
// Return the directory that contains the running test binary.
// Mirrors hipdnn_backend::platform_utilities::getCurrentModuleDirectory().
// rocke intentionally does not link the backend (pending rocKE platform-utils
// split); keep this helper in sync with PlatformUtils.linux.cpp /
// PlatformUtils.windows.cpp.
std::filesystem::path testModuleDir()
{
#ifdef _WIN32
    HMODULE handle = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&testModuleDir),
                          &handle)
       == TRUE)
    {
        char buf[MAX_PATH];
        const DWORD len = GetModuleFileNameA(handle, buf, MAX_PATH);
        if(len > 0 && len < MAX_PATH)
        {
            return std::filesystem::weakly_canonical(
                std::filesystem::path(std::string(buf, len)).parent_path());
        }
    }
    return {};
#else
    Dl_info info{};
    if(dladdr(reinterpret_cast<const void*>(&testModuleDir), &info) != 0
       && info.dli_fname != nullptr && info.dli_fname[0] != '\0')
    {
        return std::filesystem::weakly_canonical(
            std::filesystem::absolute(std::filesystem::path(info.dli_fname).parent_path()));
    }
    return {};
#endif
}

// ---- Helpers ---------------------------------------------------------------

// Return the bare GFX arch string for device 0 (e.g. "gfx942"), or "" on
// failure. Mirrors the deviceArch() helper in RockeClientDispatcher.cpp.
std::string archForDevice(int deviceId)
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, deviceId) != hipSuccess)
    {
        return {};
    }
    std::string arch{props.gcnArchName};
    const auto colon = arch.find(':'); // strip "gfx942:sramecc+:xnack-"
    if(colon != std::string::npos)
    {
        arch.resize(colon);
    }
    return arch;
}

// Build the expected kpack path for an arch under the module-relative tree:
//   <testModuleDir>/arch_content/rocke/<arch>/rocke_client_<arch>.kpack
std::filesystem::path kpackFixturePath(const std::string& arch)
{
    return testModuleDir() / "arch_content" / "rocke" / arch / ("rocke_client_" + arch + ".kpack");
}

// toc_key packed into every fixture archive (producer: rocke_kpack_pack.py
// formula rocke/{op}/{family}/{name}).
constexpr const char* TOC_KEY = "rocke/sdpa_fwd/fmha_fwd_mfma/test_kpack_module_loader";

// HIP kernel symbol compiled into the gfx942 fixture HSACO.
constexpr const char* KERNEL_SYMBOL = "rocke_test_probe";

// ---- Host tier -------------------------------------------------------------

TEST(TestKpackModuleLoad, HostTierOpensArchiveAndExtractsKernelBytes)
{
    // The host tier uses the committed gfx942 fixture: the arch is fixture-fixed
    // and independent of the running device.
    static constexpr const char* FIXTURE_ARCH = "gfx942";
    const std::filesystem::path kpackPath = kpackFixturePath(FIXTURE_ARCH);

    if(!std::filesystem::exists(kpackPath))
    {
        GTEST_SKIP() << "Fixture not found at " << kpackPath
                     << "; ensure the data tree was copied by the POST_BUILD step";
    }

    const std::string kpackPathStr = kpackPath.string();

    kpack_archive_t archive = nullptr;
    ASSERT_EQ(kpack_open(kpackPathStr.c_str(), &archive), KPACK_SUCCESS)
        << "kpack_open failed for " << kpackPathStr;

    void* kernelData = nullptr;
    size_t kernelSize = 0;
    const kpack_error_t rc
        = kpack_get_kernel(archive, TOC_KEY, FIXTURE_ARCH, &kernelData, &kernelSize);

    EXPECT_EQ(rc, KPACK_SUCCESS) << "kpack_get_kernel must return KPACK_SUCCESS";
    EXPECT_GT(kernelSize, 0u) << "extracted kernel size must be non-zero";

    if(rc == KPACK_SUCCESS && kernelData != nullptr)
    {
        const auto* bytes = static_cast<const unsigned char*>(kernelData);
        // Assert ELF magic: \x7f 'E' 'L' 'F'
        EXPECT_EQ(bytes[0], 0x7Fu) << "byte[0] must be 0x7F (ELF magic)";
        EXPECT_EQ(bytes[1], static_cast<unsigned char>('E')) << "byte[1] must be 'E'";
        EXPECT_EQ(bytes[2], static_cast<unsigned char>('L')) << "byte[2] must be 'L'";
        EXPECT_EQ(bytes[3], static_cast<unsigned char>('F')) << "byte[3] must be 'F'";
        kpack_free_kernel(kernelData);
    }

    kpack_close(archive);
}

// ---- GPU tier (real device + matching fixture) ----------------------------

TEST(TestKpackModuleLoad, GpuTierLoadsRealModuleFromKpack)
{
    // 1. Require a HIP device
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "No HIP device available; skipping GPU tier";
    }

    const std::string arch = archForDevice(0);
    if(arch.empty())
    {
        GTEST_SKIP() << "Could not determine device arch from hipGetDeviceProperties";
    }

    // 2. Locate the arch-specific fixture
    const std::filesystem::path kpackPath = kpackFixturePath(arch);
    if(!std::filesystem::exists(kpackPath))
    {
        GTEST_SKIP() << "No fixture for arch '" << arch << "' at " << kpackPath
                     << "; only gfx942 fixture is committed";
    }

    // 3. Load: kpack_open -> kpack_get_kernel -> hipModuleLoadData ->
    //    hipModuleGetFunction (all via KpackModuleLoader, which also carries
    //    the DT_NEEDED reference into rocke_client).
    const KpackLoadResult result
        = loadKernelFromKpack(kpackPath.string(), TOC_KEY, arch, KERNEL_SYMBOL);

    ASSERT_EQ(result.kpackError, KPACK_SUCCESS)
        << "kpack extraction failed with kpack_error_t " << result.kpackError;

    ASSERT_EQ(result.hipError, hipSuccess) << "hipModuleLoad/GetFunction failed with hipError_t "
                                           << result.hipError << " for symbol=" << KERNEL_SYMBOL;

    ASSERT_NE(result.module, nullptr) << "hipModule_t must be valid on success";
    ASSERT_NE(result.fn, nullptr) << "hipFunction_t must be valid on success";

    // Caller owns the module on success; unload it.
    EXPECT_EQ(hipModuleUnload(result.module), hipSuccess);
}

} // namespace
} // namespace rocke_client::dispatcher
