// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU numeric-parity test for the AOT catalog engine's RMS-norm path on gfx950
// (CDNA4). Structurally identical to the gfx1151 test -- the rmsnorm ABI
// (X, Gamma, Y, M, N, eps) carries no arch-specific arguments, so only the arch
// token and the suite name differ. It drives the substrate directly (Catalog load
// -> candidate selection -> module load -> LaunchAbi pack/grid via
// CatalogPlan::execute) and compares against a CPU reference.
//
// This family exists mainly to prove the gfx950 authoring chain end to end
// (folder auto-discovery, GPU_TARGETS gating, the producer under the rocKE pyenv,
// family.json load, selection, launch) on a cheap kernel before the dense SDPA
// family depends on all of it.
//
//   rms[m] = sqrt(sum_n(X[m,n]^2) / N + eps);  Y[m,n] = X[m,n] / rms[m] * Gamma[n]

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "core/Handle.hpp"
#include "engines/aot_catalog_engine/AotCatalogTestSupport.hpp"
#include "launch/LaunchAbi.hpp"
#include "launch/ModuleLoader.hpp"
#include "plans/CatalogPlan.hpp"

namespace
{

using namespace aot_catalog_engine;

// Baked in by CMake: the build-tree copy of the catalog (<arch>/<family>/...).
const std::string CATALOG_DIR = aotResolveTestCatalogDir();

// This family's .co are built for gfx950 only; the test is meaningful only there.
constexpr const char* ARCH = "gfx950";

bool gpuIsArch(const std::string& arch)
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        return false;
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return false;
    }
    // gcnArchName looks like "gfx950:sramecc+:xnack-"; match the leading token.
    const std::string name(props.gcnArchName);
    return name.rfind(arch, 0) == 0;
}

// Deterministic small inputs kept well within f16 range so the N-length
// sum-of-squares stays accurate enough for a tight tolerance.
float xVal(size_t m, size_t n)
{
    return (static_cast<float>((m * 13u + n * 7u) % 7u) - 3.0f) * 0.1f;
}
float gammaVal(size_t n)
{
    return static_cast<float>((n * 5u + 3u) % 5u) * 0.25f;
}

} // namespace

TEST(TestAotCatalogRmsNormNumericParityGfx950, RmsNorm2dF16MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }

    // 1. Load the catalog and select the rmsnorm kernel for an f16 M/N problem.
    //    N must equal the value baked into a shipped static kernel (or fall to a
    //    runtime-N _dyn kernel, which also accepts it).
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    constexpr size_t M = 8;
    constexpr size_t N = 2048;
    constexpr float EPS = 1e-5f;

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("rmsnorm", problem);
    ASSERT_FALSE(candidates.empty()) << "no rmsnorm candidate for the f16 N=2048 problem";
    const catalog::KernelEntry& kernel = *candidates.front().kernel;

    // 2. Load the module for the selected kernel.
    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    // 3. Build the launch bindings by hand (the RmsNormAdapter builds these from a
    //    graph; here we assign the uids ourselves and match them in the device
    //    buffer table below). This is exactly the (X,Gamma,Y,M,N,eps) ABI.
    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("X", 1);
    bindings.pointerUids.emplace("Gamma", 2);
    bindings.pointerUids.emplace("Y", 3);
    bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(M)});
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(N)});
    bindings.scalars.emplace("eps", catalog::ScalarValue{EPS});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("M", static_cast<int64_t>(M));
    gridSymbols.emplace("N", static_cast<int64_t>(N));

    const auto workspaceBytes
        = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           workspaceBytes,
                           kernel.symbol);

    // 4. Host inputs: X[M,N] row-major, Gamma[N], f16 (== _Float16). Reference is
    //    per-row RMS norm computed in float.
    std::vector<_Float16> hostX(M * N);
    std::vector<_Float16> hostGamma(N);
    std::vector<_Float16> hostY(M * N, static_cast<_Float16>(0.0f));
    std::vector<float> reference(M * N, 0.0f);

    for(size_t n = 0; n < N; ++n)
    {
        hostGamma[n] = static_cast<_Float16>(gammaVal(n));
    }
    for(size_t m = 0; m < M; ++m)
    {
        for(size_t n = 0; n < N; ++n)
        {
            hostX[m * N + n] = static_cast<_Float16>(xVal(m, n));
        }
    }
    for(size_t m = 0; m < M; ++m)
    {
        float sumSquares = 0.0f;
        for(size_t n = 0; n < N; ++n)
        {
            // Read back the f16-rounded value so the reference sees the same
            // inputs the kernel does.
            const auto x = static_cast<float>(hostX[m * N + n]);
            sumSquares += x * x;
        }
        const float invRms = 1.0f / std::sqrt(sumSquares / static_cast<float>(N) + EPS);
        for(size_t n = 0; n < N; ++n)
        {
            const auto x = static_cast<float>(hostX[m * N + n]);
            const auto g = static_cast<float>(hostGamma[n]);
            reference[m * N + n] = x * invRms * g;
        }
    }

    // 5. Device buffers + execute through the plan.
    void* deviceX = nullptr;
    void* deviceGamma = nullptr;
    void* deviceY = nullptr;
    ASSERT_EQ(hipMalloc(&deviceX, hostX.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceGamma, hostGamma.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceY, hostY.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceX, hostX.data(), hostX.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceGamma,
                        hostGamma.data(),
                        hostGamma.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(deviceY, 0, hostY.size() * sizeof(_Float16)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers = {{
        {1, deviceX},
        {2, deviceGamma},
        {3, deviceY},
    }};

    ASSERT_NO_THROW(
        plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(hostY.data(), deviceY, hostY.size() * sizeof(_Float16), hipMemcpyDeviceToHost),
        hipSuccess);

    // 6. Compare. f16 has ~3 decimal digits; scale tolerance with magnitude.
    for(size_t m = 0; m < M; ++m)
    {
        for(size_t n = 0; n < N; ++n)
        {
            const auto got = static_cast<float>(hostY[m * N + n]);
            const float want = reference[m * N + n];
            const float tol = std::max(2e-2f, 3e-2f * std::fabs(want));
            ASSERT_NEAR(got, want, tol) << "mismatch at (" << m << "," << n << ")";
        }
    }

    (void)hipFree(deviceX);
    (void)hipFree(deviceGamma);
    (void)hipFree(deviceY);
    (void)hipStreamDestroy(stream);
}
