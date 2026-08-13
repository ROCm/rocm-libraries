// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU numeric-parity test for the AOT catalog engine's LayerNorm path -- the
// third op wired into the catalog (after GEMM and RMSNorm). Like the RMSNorm
// parity test it drives the substrate directly (Catalog load -> candidate
// selection -> module load -> LaunchAbi pack/grid via CatalogPlan::execute)
// against the shipped gfx1151 rocKE layernorm2d .co and compares to a CPU
// reference.
//
// LayerNorm here matches hipDNN's standard LayerNorm semantics exactly:
//   mean[m]    = sum_n(X[m,n]) / N
//   var[m]     = sum_n((X[m,n] - mean[m])^2) / N          (population/biased)
//   inv_std[m] = 1 / sqrt(var[m] + eps)
//   Y[m,n]     = (X[m,n] - mean[m]) * inv_std[m] * Gamma[n] + Beta[n]
// The structural diff vs RMSNorm is the row-mean subtraction and the per-column
// Beta (bias) add, so the ABI carries one extra pointer (Beta).

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
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

// The shipped .co is gfx1151-only; the test is meaningful only on that arch.
constexpr const char* ARCH = "gfx1151";

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
    // gcnArchName looks like "gfx1151:sramecc+:xnack-"; match the leading token.
    const std::string name(props.gcnArchName);
    return name.rfind(arch, 0) == 0;
}

// Deterministic small inputs. X is intentionally given a large per-row DC offset
// (the +2.0f bias) on top of the small oscillation so mean subtraction is
// actually exercised: a reference that forgot to subtract the mean would be far
// outside tolerance here, unlike RMSNorm where the mean is irrelevant.
float xVal(size_t m, size_t n)
{
    return 2.0f + (static_cast<float>((m * 13u + n * 7u) % 7u) - 3.0f) * 0.1f;
}
float gammaVal(size_t n)
{
    return static_cast<float>((n * 5u + 3u) % 5u) * 0.25f;
}
float betaVal(size_t n)
{
    return (static_cast<float>((n * 3u + 1u) % 5u) - 2.0f) * 0.1f;
}

} // namespace

TEST(TestAotCatalogLayerNormNumericParity, LayerNorm2dF16MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }

    // 1. Load the catalog and select the layernorm kernel for an f16 M/N problem.
    //    N must equal the value baked into a shipped kernel (exact-match
    //    applicability, like RMSNorm).
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
        = cat.candidatesFor("layernorm", problem);
    ASSERT_FALSE(candidates.empty()) << "no layernorm candidate for the f16 N=2048 problem";
    const catalog::KernelEntry& kernel = *candidates.front().kernel;

    // 2. Load the module for the selected kernel.
    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    // 3. Build the launch bindings by hand (the LayerNormAdapter builds these from
    //    a graph; here we assign the uids ourselves and match them in the device
    //    buffer table below). This is exactly the (X,Gamma,Beta,Y,M,N,eps) ABI,
    //    with epsilon baked as an f32 scalar just as the adapter bakes it.
    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("X", 1);
    bindings.pointerUids.emplace("Gamma", 2);
    bindings.pointerUids.emplace("Beta", 3);
    bindings.pointerUids.emplace("Y", 4);
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

    // 4. Host inputs: X[M,N] row-major, Gamma[N], Beta[N], f16 (== _Float16).
    //    Reference is per-row LayerNorm computed in float.
    std::vector<_Float16> hostX(M * N);
    std::vector<_Float16> hostGamma(N);
    std::vector<_Float16> hostBeta(N);
    std::vector<_Float16> hostY(M * N, static_cast<_Float16>(0.0f));
    std::vector<float> reference(M * N, 0.0f);

    for(size_t n = 0; n < N; ++n)
    {
        hostGamma[n] = static_cast<_Float16>(gammaVal(n));
        hostBeta[n] = static_cast<_Float16>(betaVal(n));
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
        // Read back the f16-rounded values so the reference sees the same inputs
        // the kernel does.
        float sum = 0.0f;
        for(size_t n = 0; n < N; ++n)
        {
            sum += static_cast<float>(hostX[m * N + n]);
        }
        const float mean = sum / static_cast<float>(N);
        float sumSqDev = 0.0f;
        for(size_t n = 0; n < N; ++n)
        {
            const float d = static_cast<float>(hostX[m * N + n]) - mean;
            sumSqDev += d * d;
        }
        const float invStd = 1.0f / std::sqrt(sumSqDev / static_cast<float>(N) + EPS);
        for(size_t n = 0; n < N; ++n)
        {
            const auto x = static_cast<float>(hostX[m * N + n]);
            const auto g = static_cast<float>(hostGamma[n]);
            const auto bt = static_cast<float>(hostBeta[n]);
            reference[m * N + n] = (x - mean) * invStd * g + bt;
        }
    }

    // 5. Device buffers + execute through the plan.
    void* deviceX = nullptr;
    void* deviceGamma = nullptr;
    void* deviceBeta = nullptr;
    void* deviceY = nullptr;
    ASSERT_EQ(hipMalloc(&deviceX, hostX.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceGamma, hostGamma.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceBeta, hostBeta.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceY, hostY.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceX, hostX.data(), hostX.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceGamma,
                        hostGamma.data(),
                        hostGamma.size() * sizeof(_Float16),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(
            deviceBeta, hostBeta.data(), hostBeta.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(deviceY, 0, hostY.size() * sizeof(_Float16)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers = {{
        {1, deviceX},
        {2, deviceGamma},
        {3, deviceBeta},
        {4, deviceY},
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
    (void)hipFree(deviceBeta);
    (void)hipFree(deviceY);
    (void)hipStreamDestroy(stream);
}
