// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Proves that rocke_client carries a direct DT_NEEDED on librocm_kpack.so.0:
// KpackModuleLoader.cpp (compiled into rocke_client_impl, and therefore into
// the rocke_client SHARED plugin) calls kpack_open/kpack_get_kernel, which
// forces the dynamic-linker entry even under --as-needed + --exclude-libs=ALL.
//
// Host tier (always):
//   Opens the committed synthetic .kpack fixture, extracts kernel bytes via
//   kpack_open + kpack_get_kernel, and asserts KPACK_SUCCESS, non-zero size,
//   and ELF magic (\x7fELF) in the first four bytes.
//   Fixture: api/src/tests/data/rocke_client_gfx942.kpack
//   toc_key: rocke/sdpa_fwd/fmha_fwd_mfma/test_kpack_module_loader, arch gfx942
//
// GPU tier (device-gated, real-bundle):
//   Detects the running device arch (gcnArchName, colon-stripped).
//   Looks up the per-arch bundle directory via env var ROCKE_CLIENT_BUNDLE_DIR
//   (or the CMake-baked default ROCKE_CLIENT_BUNDLE_DIR_DEFAULT if defined).
//   Reads rocke_client_<arch>.json, extracts entries[0].toc_key + symbol and
//   the kpack filename, then calls loadKernelFromKpack and asserts that both
//   hipModule_t and hipFunction_t are valid.
//   Skips cleanly when: no device, no bundle dir configured, or no manifest
//   for the running arch (don't fail — the bundle may not be installed).

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <string>

#include <hip/hip_runtime_api.h>
#include <rocm_kpack/kpack.h>

#include "dispatcher/KpackModuleLoader.hpp"

// Provided by CMake (api/src/tests/CMakeLists.txt):
//   ROCKE_KPACK_TEST_FIXTURE      — absolute path to the committed fake fixture
//   ROCKE_CLIENT_BUNDLE_DIR_DEFAULT — optional build-tree bundle dir for GPU tier
#ifndef ROCKE_KPACK_TEST_FIXTURE
#define ROCKE_KPACK_TEST_FIXTURE ""
#endif

namespace rocke_client::dispatcher
{
namespace
{

// ---- Helpers ----------------------------------------------------------------

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

// Resolve the per-arch bundle directory.
// Priority: env var ROCKE_CLIENT_BUNDLE_DIR > CMake-baked default.
std::string resolveBundleDir()
{
    const char* env = std::getenv("ROCKE_CLIENT_BUNDLE_DIR"); // NOLINT(concurrency-mt-unsafe)
    if(env != nullptr && env[0] != '\0')
    {
        return std::string{env};
    }
#ifdef ROCKE_CLIENT_BUNDLE_DIR_DEFAULT
    return std::string{ROCKE_CLIENT_BUNDLE_DIR_DEFAULT};
#else
    return {};
#endif
}

// Bundle manifest fields extracted from rocke_client_<arch>.json.
struct BundleEntry
{
    std::string kpackFilename; // value of "kpack" field in the manifest
    std::string tocKey; // entries[0].toc_key
    std::string symbol; // entries[0].symbol
};

// Minimal regex-based extractor for the well-known rocke.aot.bundle/v1 schema.
// Not a general JSON parser: uses the predictable field order and quoting of
// the manifest produced by rocke_kpack_pack.py / _write_manifest().
std::optional<BundleEntry> parseManifest(const std::string& json)
{
    // Each regex matches the first occurrence of its field in the document.
    static const std::regex s_kpackRe{R"lit("kpack"\s*:\s*"([^"]+)")lit"};
    static const std::regex s_tocKeyRe{R"lit("toc_key"\s*:\s*"([^"]+)")lit"};
    static const std::regex s_symbolRe{R"lit("symbol"\s*:\s*"([^"]+)")lit"};

    std::smatch m;
    BundleEntry entry;

    if(!std::regex_search(json, m, s_kpackRe))
    {
        return std::nullopt;
    }
    entry.kpackFilename = m[1].str();

    if(!std::regex_search(json, m, s_tocKeyRe))
    {
        return std::nullopt;
    }
    entry.tocKey = m[1].str();

    if(!std::regex_search(json, m, s_symbolRe))
    {
        return std::nullopt;
    }
    entry.symbol = m[1].str();

    return entry;
}

// ---- Host tier --------------------------------------------------------------

TEST(TestKpackModuleLoad, HostTierOpensArchiveAndExtractsKernelBytes)
{
    const std::string kpackPath{ROCKE_KPACK_TEST_FIXTURE};
    if(kpackPath.empty())
    {
        GTEST_SKIP() << "ROCKE_KPACK_TEST_FIXTURE not configured";
    }

    // toc_key and arch baked into the committed fixture
    static constexpr const char* TOC_KEY = "rocke/sdpa_fwd/fmha_fwd_mfma/test_kpack_module_loader";
    static constexpr const char* ARCH = "gfx942";

    kpack_archive_t archive = nullptr;
    ASSERT_EQ(kpack_open(kpackPath.c_str(), &archive), KPACK_SUCCESS)
        << "kpack_open failed for fixture: " << kpackPath;

    void* kernelData = nullptr;
    size_t kernelSize = 0;
    const kpack_error_t rc = kpack_get_kernel(archive, TOC_KEY, ARCH, &kernelData, &kernelSize);

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

// ---- GPU tier (real AOT bundle) --------------------------------------------

TEST(TestKpackModuleLoad, GpuTierLoadsRealModuleFromBundleManifest)
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

    // 2. Locate the per-arch bundle directory
    const std::string bundleDir = resolveBundleDir();
    if(bundleDir.empty())
    {
        GTEST_SKIP() << "ROCKE_CLIENT_BUNDLE_DIR not set and no CMake default; "
                        "set it to the arch_content/rocke/ root to enable GPU tier";
    }

    // 3. Locate the manifest and kpack for this arch
    const std::string manifestPath = bundleDir + "/rocke_client_" + arch + ".json";
    std::ifstream manifestFile{manifestPath};
    if(!manifestFile.is_open())
    {
        GTEST_SKIP() << "No bundle manifest for arch '" << arch << "' at " << manifestPath
                     << "; skipping GPU tier";
    }

    const std::string manifestJson{std::istreambuf_iterator<char>{manifestFile},
                                   std::istreambuf_iterator<char>{}};

    // 4. Parse first entry (toc_key, symbol, kpack filename)
    const std::optional<BundleEntry> entry = parseManifest(manifestJson);
    if(!entry)
    {
        GTEST_SKIP() << "Could not parse toc_key/symbol from " << manifestPath;
    }

    const std::string kpackPath = bundleDir + "/" + entry->kpackFilename;

    // 5. Load via KpackModuleLoader: kpack_open → kpack_get_kernel →
    //    hipModuleLoadData → hipModuleGetFunction.
    const KpackLoadResult result
        = loadKernelFromKpack(kpackPath, entry->tocKey, arch, entry->symbol);

    ASSERT_EQ(result.kpackError, KPACK_SUCCESS)
        << "kpack extraction failed with kpack_error_t " << result.kpackError
        << " for toc_key=" << entry->tocKey << " arch=" << arch;

    ASSERT_EQ(result.hipError, hipSuccess) << "hipModuleLoad/GetFunction failed with hipError_t "
                                           << result.hipError << " for symbol=" << entry->symbol;

    ASSERT_NE(result.module, nullptr) << "hipModule_t must be valid on success";
    ASSERT_NE(result.fn, nullptr) << "hipFunction_t must be valid on success";

    // Caller owns the module on success; unload it.
    EXPECT_EQ(hipModuleUnload(result.module), hipSuccess);
}

} // namespace
} // namespace rocke_client::dispatcher
