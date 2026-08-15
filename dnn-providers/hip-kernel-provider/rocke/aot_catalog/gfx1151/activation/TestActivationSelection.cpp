// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU test for the AOT catalog engine's selection over the multi-kernel
// activation family. For one (dtype, activation) each perf variant
// (block_size/vec) decodes to the same {dtype,activation,numel} problem, so they
// are all candidates for one plan. This test drives the substrate directly and
// checks that
//   1. the decoded "activation" token discriminates silu vs gelu_tanh (a silu
//      problem never selects a gelu_tanh kernel and vice versa),
//   2. multiple perf variants match one (dtype, activation),
//   3. the tuner records a valid winner and output is correct,
//   4. every shipped variant is individually numerically correct,
//   5. numel {min:1} lets one .co serve any element count (the scalar tail
//      handles the ragged remainder), across a slab-aligned and a ragged numel.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "catalog/TuneCache.hpp"
#include "core/Handle.hpp"
#include "engines/aot_catalog_engine/AotCatalogTestSupport.hpp"
#include "launch/LaunchAbi.hpp"
#include "launch/ModuleLoader.hpp"
#include "plans/CatalogPlan.hpp"

namespace
{

using namespace aot_catalog_engine;
namespace fs = std::filesystem;

const std::string CATALOG_DIR = aotResolveTestCatalogDir();
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
    const std::string name(props.gcnArchName);
    return name.rfind(arch, 0) == 0;
}

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
const std::function<float(float)>& refFor(const std::string& activation)
{
    static const std::function<float(float)> s_silu = siluRef;
    static const std::function<float(float)> s_gelu = geluTanhRef;
    return activation == "silu" ? s_silu : s_gelu;
}

// Build the (A,C,N) launch bindings + grid symbols the ActivationAdapter would
// produce -- identical for every perf variant. Returns nullopt on module-load
// failure (a value-returning helper cannot host ASSERT_*).
std::optional<PlanCandidate> makeCandidate(const catalog::KernelEntry& kernel, size_t numel)
{
    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    if(!module.has_value())
    {
        return std::nullopt;
    }
    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", 1);
    bindings.pointerUids.emplace("C", 2);
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(numel)});
    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("numel", static_cast<int64_t>(numel));
    const auto workspaceBytes
        = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));
    return PlanCandidate{std::move(*module),
                         kernel.launch,
                         std::move(bindings),
                         std::move(gridSymbols),
                         workspaceBytes,
                         kernel.symbol};
}

struct ShapeOutcome
{
    std::vector<std::string> symbols;
    std::string winner;
};

// Full end-to-end for one (activation, numel): load all candidates, run a
// multi-candidate tuned plan on real device buffers, and EXPECT numerical
// correctness. Returns the candidate symbols + the tuned winner.
ShapeOutcome runAndCheck(const std::string& activation, size_t numel, const std::string& cachePath)
{
    ShapeOutcome outcome;

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    EXPECT_FALSE(cat.empty());

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("activation", catalog::ShapeValue{activation});
    problem.emplace("numel", catalog::ShapeValue{static_cast<int64_t>(numel)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("pointwise", problem);
    if(candidates.empty())
    {
        return outcome; // caller asserts on the (empty) symbol list
    }

    std::vector<PlanCandidate> planCandidates;
    planCandidates.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        std::optional<PlanCandidate> built = makeCandidate(*candidate.kernel, numel);
        if(!built.has_value())
        {
            ADD_FAILURE() << "failed to build candidate for " << candidate.kernel->symbol;
            return outcome;
        }
        outcome.symbols.push_back(candidate.kernel->symbol);
        planCandidates.push_back(std::move(*built));
    }

    std::vector<_Float16> hostA(numel);
    std::vector<_Float16> hostC(numel, static_cast<_Float16>(0.0f));
    std::vector<float> ref(numel, 0.0f);
    const std::function<float(float)>& fn = refFor(activation);
    for(size_t i = 0; i < numel; ++i)
    {
        hostA[i] = static_cast<_Float16>(xVal(i));
        ref[i] = fn(static_cast<float>(hostA[i]));
    }

    void* deviceA = nullptr;
    void* deviceC = nullptr;
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipMalloc(&deviceA, numel * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceC, numel * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(hipMemcpy(deviceA, hostA.data(), numel * sizeof(_Float16), hipMemcpyHostToDevice),
              hipSuccess);
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::string key = catalog::problemKey(candidates.front().family->name, problem);
    catalog::TuneCache cache(cachePath);
    const CatalogPlan plan(std::move(planCandidates), &cache, key);

    const std::array<hipdnnPluginDeviceBuffer_t, 2> buffers = {{
        {1, deviceA},
        {2, deviceC},
    }};
    EXPECT_EQ(hipMemset(deviceC, 0, numel * sizeof(_Float16)), hipSuccess);
    EXPECT_NO_THROW(plan.execute(handle, buffers.data(), 2, nullptr));
    EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(hipMemcpy(hostC.data(), deviceC, numel * sizeof(_Float16), hipMemcpyDeviceToHost),
              hipSuccess);

    size_t mismatches = 0;
    std::string firstMismatch;
    for(size_t i = 0; i < numel; ++i)
    {
        const auto got = static_cast<float>(hostC[i]);
        const float want = ref[i];
        const float tol = std::max(2e-2f, 3e-2f * std::fabs(want));
        if(std::fabs(got - want) > tol)
        {
            if(mismatches == 0)
            {
                firstMismatch = std::to_string(i) + " got=" + std::to_string(got)
                                + " want=" + std::to_string(want);
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << activation << " numel=" << numel << " first mismatch "
                              << firstMismatch;

    const std::optional<std::string> winner = cache.lookup(key);
    if(winner.has_value())
    {
        outcome.winner = *winner;
    }

    (void)hipFree(deviceA);
    (void)hipFree(deviceC);
    (void)hipStreamDestroy(stream);
    std::error_code ec;
    fs::remove(cachePath, ec);
    fs::remove(cachePath + ".tmp", ec);
    return outcome;
}

} // namespace

// The decoded "activation" token discriminates the two ops: a silu problem
// selects only silu kernels, a gelu_tanh problem only gelu_tanh kernels.
TEST(TestActivationSelection, DiscriminatesActivationToken)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    for(const char* activation : {"silu", "gelu_tanh"})
    {
        catalog::ProblemShape problem;
        problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
        problem.emplace("activation", catalog::ShapeValue{std::string(activation)});
        problem.emplace("numel", catalog::ShapeValue{static_cast<int64_t>(4096)});

        const std::vector<catalog::Catalog::Candidate> candidates
            = cat.candidatesFor("pointwise", problem);
        ASSERT_FALSE(candidates.empty()) << "no candidate for " << activation;

        const std::string other = std::string(activation) == "silu" ? "gelu_tanh" : "silu";
        for(const catalog::Catalog::Candidate& candidate : candidates)
        {
            const std::string& sym = candidate.kernel->symbol;
            EXPECT_NE(sym.find(activation), std::string::npos)
                << "candidate '" << sym << "' should be a " << activation << " kernel";
            // gelu_tanh contains no "silu"; silu's symbol must not contain "gelu".
            if(std::string(activation) == "silu")
            {
                EXPECT_EQ(sym.find("gelu"), std::string::npos)
                    << "silu problem selected a gelu kernel: " << sym;
            }
            else
            {
                EXPECT_EQ(sym.find("silu"), std::string::npos)
                    << "gelu_tanh problem selected a silu kernel: " << sym;
            }
            (void)other;
        }
    }
}

// Multiple perf variants match one (dtype, activation), the tuner records a
// valid winner, and output is correct (slab-aligned numel).
TEST(TestActivationSelection, MultipleCandidatesTuneAndCache)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    if(catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH).empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }
    const std::string cachePath
        = (fs::temp_directory_path() / "hipdnn_aot_activation_silu_4096.json").string();
    const ShapeOutcome o = runAndCheck("silu", 4096, cachePath);

    EXPECT_GE(o.symbols.size(), 2u) << "expected multiple silu perf variants";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for silu";
    EXPECT_NE(std::find(o.symbols.begin(), o.symbols.end(), o.winner), o.symbols.end())
        << "winner '" << o.winner << "' is not a candidate";
}

// numel {min:1} serves any element count: a ragged numel (scalar tail) is
// correct for both activations.
TEST(TestActivationSelection, RaggedNumelIsCorrect)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    if(catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH).empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }
    for(const char* activation : {"silu", "gelu_tanh"})
    {
        const std::string cachePath
            = (fs::temp_directory_path()
               / (std::string("hipdnn_aot_activation_ragged_") + activation + ".json"))
                  .string();
        const ShapeOutcome o = runAndCheck(activation, 4099, cachePath);
        ASSERT_FALSE(o.symbols.empty()) << "no candidate matched ragged numel for " << activation;
        EXPECT_FALSE(o.winner.empty()) << "tuning recorded no winner for ragged " << activation;
    }
}

// Every shipped perf variant is individually numerically correct (single-
// candidate plan per symbol, exercising the no-tuning launch path).
TEST(TestActivationSelection, EachVariantIsCorrect)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    const size_t numel = 4099;
    for(const char* activation : {"silu", "gelu_tanh"})
    {
        catalog::ProblemShape problem;
        problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
        problem.emplace("activation", catalog::ShapeValue{std::string(activation)});
        problem.emplace("numel", catalog::ShapeValue{static_cast<int64_t>(numel)});

        const std::vector<catalog::Catalog::Candidate> candidates
            = cat.candidatesFor("pointwise", problem);
        ASSERT_FALSE(candidates.empty()) << "no candidate for " << activation;

        const std::function<float(float)>& fn = refFor(activation);

        std::vector<_Float16> hostA(numel);
        std::vector<float> ref(numel, 0.0f);
        for(size_t i = 0; i < numel; ++i)
        {
            hostA[i] = static_cast<_Float16>(xVal(i));
            ref[i] = fn(static_cast<float>(hostA[i]));
        }

        for(const catalog::Catalog::Candidate& candidate : candidates)
        {
            const catalog::KernelEntry& kernel = *candidate.kernel;
            std::optional<PlanCandidate> built = makeCandidate(kernel, numel);
            ASSERT_TRUE(built.has_value()) << "failed to build candidate for " << kernel.symbol;
            PlanCandidate pc = std::move(*built);
            const CatalogPlan plan(std::move(pc.module),
                                   pc.launch,
                                   std::move(pc.bindings),
                                   std::move(pc.gridSymbols),
                                   pc.workspaceBytes,
                                   pc.symbol);
            SCOPED_TRACE(kernel.symbol);

            std::vector<_Float16> hostC(numel, static_cast<_Float16>(0.0f));
            void* deviceA = nullptr;
            void* deviceC = nullptr;
            hipStream_t stream = nullptr;
            ASSERT_EQ(hipMalloc(&deviceA, numel * sizeof(_Float16)), hipSuccess);
            ASSERT_EQ(hipMalloc(&deviceC, numel * sizeof(_Float16)), hipSuccess);
            ASSERT_EQ(
                hipMemcpy(deviceA, hostA.data(), numel * sizeof(_Float16), hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(hipMemset(deviceC, 0, numel * sizeof(_Float16)), hipSuccess);
            ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
            Handle handle;
            handle.setStream(stream);

            const std::array<hipdnnPluginDeviceBuffer_t, 2> buffers = {{
                {1, deviceA},
                {2, deviceC},
            }};
            ASSERT_NO_THROW(plan.execute(handle, buffers.data(), 2, nullptr));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            ASSERT_EQ(
                hipMemcpy(hostC.data(), deviceC, numel * sizeof(_Float16), hipMemcpyDeviceToHost),
                hipSuccess);

            for(size_t i = 0; i < numel; ++i)
            {
                const auto got = static_cast<float>(hostC[i]);
                const float want = ref[i];
                const float tol = std::max(2e-2f, 3e-2f * std::fabs(want));
                ASSERT_NEAR(got, want, tol) << kernel.symbol << " mismatch at " << i;
            }

            (void)hipFree(deviceA);
            (void)hipFree(deviceC);
            (void)hipStreamDestroy(stream);
        }
    }
}
