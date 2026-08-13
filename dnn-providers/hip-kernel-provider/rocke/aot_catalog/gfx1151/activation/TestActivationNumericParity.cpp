// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU numeric-parity test for the AOT catalog engine's elementwise-activation
// path -- the fourth op wired into the catalog. Like the RMSNorm/LayerNorm
// parity tests it drives the substrate directly (Catalog load -> candidate
// selection -> module load -> LaunchAbi pack/grid via CatalogPlan::execute)
// against the shipped gfx1151 rocKE elementwise .co and compares to a CPU
// reference.
//
// The two activations match hipDNN's PointwiseMode semantics exactly:
//   silu(x)      = x * sigmoid(x)                    (SWISH_FWD, beta == 1)
//   gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))  (GELU_APPROX_TANH_FWD)
// numel is deliberately NOT a multiple of the block*vec slab so the kernel's
// per-element scalar tail path is exercised alongside the vectorised fast path.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// Non-slab-aligned so both the vectorised fast path and the scalar tail run.
constexpr size_t NUMEL = 4099;

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
    const std::string name(props.gcnArchName);
    return name.rfind(arch, 0) == 0;
}

// Spread across [-3, 3] so both activation tails (saturating negative, ~linear
// positive) are covered.
float xVal(size_t i)
{
    return (static_cast<float>(i % 13u) - 6.0f) * 0.5f;
}

float siluRef(float x)
{
    return x / (1.0f + std::exp(-x));
}
float geluTanhRef(float x)
{
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// Drive one activation token end-to-end and compare against `ref`.
void runActivationParity(const std::string& activation, const std::function<float(float)>& ref)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("activation", catalog::ShapeValue{activation});
    problem.emplace("numel", catalog::ShapeValue{static_cast<int64_t>(NUMEL)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("pointwise", problem);
    ASSERT_FALSE(candidates.empty()) << "no pointwise candidate for f16 " << activation;
    const catalog::KernelEntry& kernel = *candidates.front().kernel;

    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    // (A, C, N) ABI: A in, C out, N the flat element count. Assign the uids here
    // and match them in the device buffer table below.
    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", 1);
    bindings.pointerUids.emplace("C", 2);
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(NUMEL)});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("numel", static_cast<int64_t>(NUMEL));

    // KernelEntry carries a workspace expression (data-driven sizing); evaluate
    // it against the grid symbols before moving them into the plan.
    const auto workspaceBytes
        = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           workspaceBytes,
                           kernel.symbol);

    std::vector<_Float16> hostA(NUMEL);
    std::vector<_Float16> hostC(NUMEL, static_cast<_Float16>(0.0f));
    std::vector<float> reference(NUMEL, 0.0f);
    for(size_t i = 0; i < NUMEL; ++i)
    {
        hostA[i] = static_cast<_Float16>(xVal(i));
        // Read back the f16-rounded input so the reference sees the kernel's input.
        reference[i] = ref(static_cast<float>(hostA[i]));
    }

    void* deviceA = nullptr;
    void* deviceC = nullptr;
    ASSERT_EQ(hipMalloc(&deviceA, hostA.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceC, hostC.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(deviceC, 0, hostC.size() * sizeof(_Float16)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::array<hipdnnPluginDeviceBuffer_t, 2> buffers = {{
        {1, deviceA},
        {2, deviceC},
    }};

    ASSERT_NO_THROW(
        plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(hostC.data(), deviceC, hostC.size() * sizeof(_Float16), hipMemcpyDeviceToHost),
        hipSuccess);

    for(size_t i = 0; i < NUMEL; ++i)
    {
        const auto got = static_cast<float>(hostC[i]);
        const float want = reference[i];
        const float tol = std::max(2e-2f, 3e-2f * std::fabs(want));
        ASSERT_NEAR(got, want, tol)
            << activation << " mismatch at " << i << " (x=" << static_cast<float>(hostA[i]) << ")";
    }

    (void)hipFree(deviceA);
    (void)hipFree(deviceC);
    (void)hipStreamDestroy(stream);
}

} // namespace

TEST(TestAotCatalogActivationNumericParity, SiluF16MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    runActivationParity("silu", siluRef);
}

TEST(TestAotCatalogActivationNumericParity, GeluTanhF16MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    runActivationParity("gelu_tanh", geluTanhRef);
}
