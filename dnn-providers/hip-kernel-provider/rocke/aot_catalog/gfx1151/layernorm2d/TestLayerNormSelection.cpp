// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU test for the AOT catalog engine's Phase 2 measure-and-cache selection over
// the multi-kernel layernorm2d family. Each N tier ships several perf-only
// (block_size/vec) variants that all decode to the same {dtype,M,N} problem, so
// they are all candidates for one plan. This test drives the substrate directly
// (like the parity tests): it builds a multi-candidate CatalogPlan + a TuneCache
// pointed at a unique temp file and checks that
//   1. the first execute tunes, records a winner in the cache, and is correct,
//   2. a second execute hits the cache and stays correct,
//   3. every shipped variant is individually numerically correct.
//
// Like rmsnorm2d, layernorm2d ships BOTH static-N specializations (exact
// N==<n> constraint, tuned per shape) AND runtime-N '_dyn_' kernels (N read
// from the i32 arg, matching any N that is a multiple of vec). So a listed N
// tier (2048, 4096) draws both static and dynamic candidates, while an
// unlisted multiple-of-vec N (Flux 3072, SD3.5 2432) is served by the dynamic
// kernels alone -- the runtime-N parity the RuntimeN tests below assert.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

// Baked in by CMake: the build-tree copy of the catalog (<arch>/<family>/...).
const std::string CATALOG_DIR = aotResolveTestCatalogDir();
constexpr const char* ARCH = "gfx1151";

constexpr size_t M = 8;
constexpr size_t N = 2048;
constexpr float EPS = 1e-5f;

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

// Large per-row DC offset (+2.0f) so mean subtraction is genuinely exercised.
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

// Build the (X,Gamma,Beta,Y,M,N,eps) launch bindings + grid symbols the
// LayerNormAdapter would produce -- identical for every perf variant of the
// family. Returns nullopt when the module fails to load (a value-returning
// helper cannot host ASSERT_*; callers ASSERT_TRUE(has_value()) at the call site).
std::optional<PlanCandidate> makeCandidate(const catalog::KernelEntry& kernel)
{
    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    if(!module.has_value())
    {
        return std::nullopt;
    }

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
    return PlanCandidate{std::move(*module),
                         kernel.launch,
                         std::move(bindings),
                         std::move(gridSymbols),
                         workspaceBytes,
                         kernel.symbol};
}

// Per-row LayerNorm reference (population variance):
//   mean=sum/N; var=sum((x-mean)^2)/N; y=(x-mean)/sqrt(var+eps)*gamma+beta.
std::vector<float> reference(const std::vector<_Float16>& hostX,
                             const std::vector<_Float16>& hostGamma,
                             const std::vector<_Float16>& hostBeta)
{
    std::vector<float> ref(M * N, 0.0f);
    for(size_t m = 0; m < M; ++m)
    {
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
            ref[m * N + n] = (x - mean) * invStd * g + bt;
        }
    }
    return ref;
}

void expectMatches(const std::vector<_Float16>& hostY, const std::vector<float>& ref)
{
    for(size_t m = 0; m < M; ++m)
    {
        for(size_t n = 0; n < N; ++n)
        {
            const auto got = static_cast<float>(hostY[m * N + n]);
            const float want = ref[m * N + n];
            const float tol = std::max(2e-2f, 3e-2f * std::fabs(want));
            ASSERT_NEAR(got, want, tol) << "mismatch at (" << m << "," << n << ")";
        }
    }
}

// Fixture holding the device buffers + a stream; hosts X/Gamma/Beta and CPU ref.
class TestLayerNormSelection : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!gpuIsArch(ARCH))
        {
            GTEST_SKIP() << "no " << ARCH << " GPU present";
        }

        _hostX.resize(M * N);
        _hostGamma.resize(N);
        _hostBeta.resize(N);
        _hostY.assign(M * N, static_cast<_Float16>(0.0f));
        for(size_t n = 0; n < N; ++n)
        {
            _hostGamma[n] = static_cast<_Float16>(gammaVal(n));
            _hostBeta[n] = static_cast<_Float16>(betaVal(n));
        }
        for(size_t m = 0; m < M; ++m)
        {
            for(size_t n = 0; n < N; ++n)
            {
                _hostX[m * N + n] = static_cast<_Float16>(xVal(m, n));
            }
        }
        _ref = reference(_hostX, _hostGamma, _hostBeta);

        ASSERT_EQ(hipMalloc(&_deviceX, M * N * sizeof(_Float16)), hipSuccess);
        ASSERT_EQ(hipMalloc(&_deviceGamma, N * sizeof(_Float16)), hipSuccess);
        ASSERT_EQ(hipMalloc(&_deviceBeta, N * sizeof(_Float16)), hipSuccess);
        ASSERT_EQ(hipMalloc(&_deviceY, M * N * sizeof(_Float16)), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(_deviceX, _hostX.data(), M * N * sizeof(_Float16), hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(_deviceGamma, _hostGamma.data(), N * sizeof(_Float16), hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(_deviceBeta, _hostBeta.data(), N * sizeof(_Float16), hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        _handle.setStream(_stream);
    }

    void TearDown() override
    {
        if(_deviceX != nullptr)
        {
            (void)hipFree(_deviceX);
        }
        if(_deviceGamma != nullptr)
        {
            (void)hipFree(_deviceGamma);
        }
        if(_deviceBeta != nullptr)
        {
            (void)hipFree(_deviceBeta);
        }
        if(_deviceY != nullptr)
        {
            (void)hipFree(_deviceY);
        }
        if(_stream != nullptr)
        {
            (void)hipStreamDestroy(_stream);
        }
        if(!_cachePath.empty())
        {
            std::error_code ec;
            fs::remove(_cachePath, ec);
            fs::remove(_cachePath + ".tmp", ec);
        }
    }

    // Run `plan`, copy Y back into _hostY, and assert it matches the CPU ref.
    void runAndCheck(const CatalogPlan& plan)
    {
        const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers = {{
            {1, _deviceX},
            {2, _deviceGamma},
            {3, _deviceBeta},
            {4, _deviceY},
        }};
        ASSERT_EQ(hipMemset(_deviceY, 0, M * N * sizeof(_Float16)), hipSuccess);
        ASSERT_NO_THROW(plan.execute(_handle, buffers.data(), 4, nullptr));
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(_hostY.data(), _deviceY, M * N * sizeof(_Float16), hipMemcpyDeviceToHost),
            hipSuccess);
        expectMatches(_hostY, _ref);
    }

    std::string uniqueCachePath(const std::string& tag)
    {
        _cachePath
            = (fs::temp_directory_path() / ("hipdnn_aot_layernorm_sel_" + tag + ".json")).string();
        std::error_code ec;
        fs::remove(_cachePath, ec);
        fs::remove(_cachePath + ".tmp", ec);
        return _cachePath;
    }

    std::vector<_Float16> _hostX;
    std::vector<_Float16> _hostGamma;
    std::vector<_Float16> _hostBeta;
    std::vector<_Float16> _hostY;
    std::vector<float> _ref;
    void* _deviceX = nullptr;
    void* _deviceGamma = nullptr;
    void* _deviceBeta = nullptr;
    void* _deviceY = nullptr;
    hipStream_t _stream = nullptr;
    Handle _handle;
    std::string _cachePath;
};

// ---- bf16 coverage (diffusion transformers run bf16) ----------------------
//
// The fixture above is f16 (_Float16); bf16 needs its own host storage. We
// represent bf16 on the host as the raw 16-bit pattern (bfloat16 == the top 16
// bits of an IEEE f32), so the device buffers are byte-for-byte what the kernel
// reads -- no HIP bf16 header needed.

uint16_t floatToBf16(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t roundingBias = ((bits >> 16) & 1u) + 0x7fffu; // round to nearest even
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16ToFloat(uint16_t bpat)
{
    const uint32_t bits = static_cast<uint32_t>(bpat) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

struct ShapeOutcome
{
    std::vector<std::string> symbols; // candidate symbols, in catalog order
    std::string winner; // tuned winner (empty if single/none)
};

// bf16 end-to-end for one N: load the bf16 family candidates, run a
// multi-candidate tuned plan on real device buffers, and EXPECT numerical
// correctness at a bf16-appropriate tolerance. Uses only EXPECT_* (a
// value-returning helper cannot host ASSERT_*).
ShapeOutcome runShapeAndCheckBf16(size_t cols, const std::string& cachePath)
{
    ShapeOutcome outcome;
    const size_t rows = M;

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    EXPECT_FALSE(cat.empty());

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("bf16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(rows)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(cols)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("layernorm", problem);
    if(candidates.empty())
    {
        return outcome; // caller asserts on the (empty) symbol list
    }

    std::vector<PlanCandidate> planCandidates;
    planCandidates.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        const catalog::KernelEntry& kernel = *candidate.kernel;
        std::optional<launch::HipModuleGuard> module
            = launch::loadKernelModule(kernel.coPath, kernel.symbol);
        if(!module.has_value())
        {
            ADD_FAILURE() << "failed to load candidate " << kernel.symbol;
            return outcome;
        }
        catalog::LaunchBindings bindings;
        bindings.pointerUids.emplace("X", 1);
        bindings.pointerUids.emplace("Gamma", 2);
        bindings.pointerUids.emplace("Beta", 3);
        bindings.pointerUids.emplace("Y", 4);
        bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(rows)});
        bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(cols)});
        bindings.scalars.emplace("eps", catalog::ScalarValue{EPS});
        launch::SymbolTable gridSymbols;
        gridSymbols.emplace("M", static_cast<int64_t>(rows));
        gridSymbols.emplace("N", static_cast<int64_t>(cols));
        outcome.symbols.push_back(kernel.symbol);
        const auto workspaceBytes
            = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));
        planCandidates.push_back(PlanCandidate{std::move(*module),
                                               kernel.launch,
                                               std::move(bindings),
                                               std::move(gridSymbols),
                                               workspaceBytes,
                                               kernel.symbol});
    }

    // Host inputs (bf16 bit patterns) + f32 CPU reference from the same values.
    std::vector<uint16_t> hostX(rows * cols);
    std::vector<uint16_t> hostGamma(cols);
    std::vector<uint16_t> hostBeta(cols);
    std::vector<uint16_t> hostY(rows * cols, 0);
    for(size_t n = 0; n < cols; ++n)
    {
        hostGamma[n] = floatToBf16(gammaVal(n));
        hostBeta[n] = floatToBf16(betaVal(n));
    }
    for(size_t m = 0; m < rows; ++m)
    {
        for(size_t n = 0; n < cols; ++n)
        {
            hostX[m * cols + n] = floatToBf16(xVal(m, n));
        }
    }
    std::vector<float> ref(rows * cols, 0.0f);
    for(size_t m = 0; m < rows; ++m)
    {
        float sum = 0.0f;
        for(size_t n = 0; n < cols; ++n)
        {
            sum += bf16ToFloat(hostX[m * cols + n]);
        }
        const float mean = sum / static_cast<float>(cols);
        float sumSqDev = 0.0f;
        for(size_t n = 0; n < cols; ++n)
        {
            const float d = bf16ToFloat(hostX[m * cols + n]) - mean;
            sumSqDev += d * d;
        }
        const float invStd = 1.0f / std::sqrt(sumSqDev / static_cast<float>(cols) + EPS);
        for(size_t n = 0; n < cols; ++n)
        {
            const float x = bf16ToFloat(hostX[m * cols + n]);
            const float g = bf16ToFloat(hostGamma[n]);
            const float bt = bf16ToFloat(hostBeta[n]);
            ref[m * cols + n] = (x - mean) * invStd * g + bt;
        }
    }

    void* deviceX = nullptr;
    void* deviceGamma = nullptr;
    void* deviceBeta = nullptr;
    void* deviceY = nullptr;
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipMalloc(&deviceX, rows * cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceGamma, cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceBeta, cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceY, rows * cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceX, hostX.data(), rows * cols * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceGamma, hostGamma.data(), cols * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceBeta, hostBeta.data(), cols * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::string key = catalog::problemKey(candidates.front().family->name, problem);
    catalog::TuneCache cache(cachePath);
    const CatalogPlan plan(std::move(planCandidates), &cache, key);

    const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers = {{
        {1, deviceX},
        {2, deviceGamma},
        {3, deviceBeta},
        {4, deviceY},
    }};
    EXPECT_EQ(hipMemset(deviceY, 0, rows * cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_NO_THROW(plan.execute(handle, buffers.data(), 4, nullptr));
    EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(hostY.data(), deviceY, rows * cols * sizeof(uint16_t), hipMemcpyDeviceToHost),
        hipSuccess);

    // Correctness at bf16 tolerance (7-bit mantissa -> looser than f16).
    size_t mismatches = 0;
    std::string firstMismatch;
    for(size_t m = 0; m < rows; ++m)
    {
        for(size_t n = 0; n < cols; ++n)
        {
            const float got = bf16ToFloat(hostY[m * cols + n]);
            const float want = ref[m * cols + n];
            const float tol = std::max(5e-2f, 5e-2f * std::fabs(want));
            if(std::fabs(got - want) > tol)
            {
                if(mismatches == 0)
                {
                    firstMismatch = "(" + std::to_string(m) + "," + std::to_string(n) + ") got="
                                    + std::to_string(got) + " want=" + std::to_string(want);
                }
                ++mismatches;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << "bf16 N=" << cols << " first mismatch " << firstMismatch;

    const std::optional<std::string> winner = cache.lookup(key);
    if(winner.has_value())
    {
        outcome.winner = *winner;
    }

    (void)hipFree(deviceX);
    (void)hipFree(deviceGamma);
    (void)hipFree(deviceBeta);
    (void)hipFree(deviceY);
    (void)hipStreamDestroy(stream);
    std::error_code ec;
    fs::remove(cachePath, ec);
    fs::remove(cachePath + ".tmp", ec);

    return outcome;
}

// f16 analog of runShapeAndCheckBf16: load the f16 family candidates for one
// runtime column count, run a multi-candidate tuned plan on real device buffers,
// and EXPECT numerical correctness. Uses a free arbitrary `cols` (unlike the
// fixture, which is pinned to N=2048) so it can exercise runtime-N shapes the
// static specializations don't cover (Flux 3072, SD3.5 2432). Uses only EXPECT_*
// (a value-returning helper cannot host ASSERT_*).
ShapeOutcome runShapeAndCheckF16(size_t cols, const std::string& cachePath)
{
    ShapeOutcome outcome;
    const size_t rows = M;

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    EXPECT_FALSE(cat.empty());

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(rows)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(cols)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("layernorm", problem);
    if(candidates.empty())
    {
        return outcome; // caller asserts on the (empty) symbol list
    }

    std::vector<_Float16> hostX(rows * cols);
    std::vector<_Float16> hostGamma(cols);
    std::vector<_Float16> hostBeta(cols);
    std::vector<_Float16> hostY(rows * cols, static_cast<_Float16>(0.0f));
    for(size_t n = 0; n < cols; ++n)
    {
        hostGamma[n] = static_cast<_Float16>(gammaVal(n));
        hostBeta[n] = static_cast<_Float16>(betaVal(n));
    }
    for(size_t m = 0; m < rows; ++m)
    {
        for(size_t n = 0; n < cols; ++n)
        {
            hostX[m * cols + n] = static_cast<_Float16>(xVal(m, n));
        }
    }
    // f32 CPU reference (population variance) from the same host values.
    std::vector<float> ref(rows * cols, 0.0f);
    for(size_t m = 0; m < rows; ++m)
    {
        float sum = 0.0f;
        for(size_t n = 0; n < cols; ++n)
        {
            sum += static_cast<float>(hostX[m * cols + n]);
        }
        const float mean = sum / static_cast<float>(cols);
        float sumSqDev = 0.0f;
        for(size_t n = 0; n < cols; ++n)
        {
            const float d = static_cast<float>(hostX[m * cols + n]) - mean;
            sumSqDev += d * d;
        }
        const float invStd = 1.0f / std::sqrt(sumSqDev / static_cast<float>(cols) + EPS);
        for(size_t n = 0; n < cols; ++n)
        {
            const auto x = static_cast<float>(hostX[m * cols + n]);
            const auto g = static_cast<float>(hostGamma[n]);
            const auto bt = static_cast<float>(hostBeta[n]);
            ref[m * cols + n] = (x - mean) * invStd * g + bt;
        }
    }

    std::vector<PlanCandidate> planCandidates;
    planCandidates.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        const catalog::KernelEntry& kernel = *candidate.kernel;
        std::optional<launch::HipModuleGuard> module
            = launch::loadKernelModule(kernel.coPath, kernel.symbol);
        if(!module.has_value())
        {
            ADD_FAILURE() << "failed to load candidate " << kernel.symbol;
            return outcome;
        }
        catalog::LaunchBindings bindings;
        bindings.pointerUids.emplace("X", 1);
        bindings.pointerUids.emplace("Gamma", 2);
        bindings.pointerUids.emplace("Beta", 3);
        bindings.pointerUids.emplace("Y", 4);
        bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(rows)});
        bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(cols)});
        bindings.scalars.emplace("eps", catalog::ScalarValue{EPS});
        launch::SymbolTable gridSymbols;
        gridSymbols.emplace("M", static_cast<int64_t>(rows));
        gridSymbols.emplace("N", static_cast<int64_t>(cols));
        outcome.symbols.push_back(kernel.symbol);
        const auto workspaceBytes
            = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));
        planCandidates.push_back(PlanCandidate{std::move(*module),
                                               kernel.launch,
                                               std::move(bindings),
                                               std::move(gridSymbols),
                                               workspaceBytes,
                                               kernel.symbol});
    }

    void* deviceX = nullptr;
    void* deviceGamma = nullptr;
    void* deviceBeta = nullptr;
    void* deviceY = nullptr;
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipMalloc(&deviceX, rows * cols * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceGamma, cols * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceBeta, cols * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceY, rows * cols * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceX, hostX.data(), rows * cols * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceGamma, hostGamma.data(), cols * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceBeta, hostBeta.data(), cols * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::string key = catalog::problemKey(candidates.front().family->name, problem);
    catalog::TuneCache cache(cachePath);
    const CatalogPlan plan(std::move(planCandidates), &cache, key);

    const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers = {{
        {1, deviceX},
        {2, deviceGamma},
        {3, deviceBeta},
        {4, deviceY},
    }};
    EXPECT_EQ(hipMemset(deviceY, 0, rows * cols * sizeof(_Float16)), hipSuccess);
    EXPECT_NO_THROW(plan.execute(handle, buffers.data(), 4, nullptr));
    EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(hostY.data(), deviceY, rows * cols * sizeof(_Float16), hipMemcpyDeviceToHost),
        hipSuccess);

    size_t mismatches = 0;
    std::string firstMismatch;
    for(size_t m = 0; m < rows; ++m)
    {
        for(size_t n = 0; n < cols; ++n)
        {
            const auto got = static_cast<float>(hostY[m * cols + n]);
            const float want = ref[m * cols + n];
            const float tol = std::max(2e-2f, 3e-2f * std::fabs(want));
            if(std::fabs(got - want) > tol)
            {
                if(mismatches == 0)
                {
                    firstMismatch = "(" + std::to_string(m) + "," + std::to_string(n) + ") got="
                                    + std::to_string(got) + " want=" + std::to_string(want);
                }
                ++mismatches;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << "f16 N=" << cols << " first mismatch " << firstMismatch;

    const std::optional<std::string> winner = cache.lookup(key);
    if(winner.has_value())
    {
        outcome.winner = *winner;
    }

    (void)hipFree(deviceX);
    (void)hipFree(deviceGamma);
    (void)hipFree(deviceBeta);
    (void)hipFree(deviceY);
    (void)hipStreamDestroy(stream);
    std::error_code ec;
    fs::remove(cachePath, ec);
    fs::remove(cachePath + ".tmp", ec);

    return outcome;
}

bool hasRuntimeN(const std::vector<std::string>& symbols)
{
    return std::any_of(symbols.begin(), symbols.end(), [](const std::string& s) {
        return s.find("_dyn_") != std::string::npos;
    });
}

bool allRuntimeN(const std::vector<std::string>& symbols)
{
    return std::all_of(symbols.begin(), symbols.end(), [](const std::string& s) {
        return s.find("_dyn_") != std::string::npos;
    });
}

} // namespace

// The N=2048 f16 family exposes multiple perf variants that all match one problem.
TEST_F(TestLayerNormSelection, MultipleCandidatesForOneProblem)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("layernorm", problem);
    // Four static N=2048 specializations (b256/v4, b512/v4, b128/v8, b64/v8)
    // PLUS the two runtime-N _dyn_ kernels (b256/v4, b128/v8) that also match
    // N=2048 -> at least six candidates for the one problem.
    EXPECT_GE(candidates.size(), 4u) << "expected the N=2048 static specializations";
    std::vector<std::string> symbols;
    symbols.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        symbols.push_back(candidate.kernel->symbol);
    }
    EXPECT_TRUE(hasRuntimeN(symbols))
        << "N=2048 should also draw the runtime-N _dyn_ catch-all kernels";
}

// First execute tunes (records a winner among the candidates) and is correct;
// the second execute hits the cache and stays correct.
TEST_F(TestLayerNormSelection, TunesRecordsAndCaches)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("layernorm", problem);
    ASSERT_GE(candidates.size(), 2u);

    std::vector<std::string> symbols;
    std::vector<PlanCandidate> planCandidates;
    planCandidates.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        std::optional<PlanCandidate> built = makeCandidate(*candidate.kernel);
        ASSERT_TRUE(built.has_value())
            << "failed to build candidate for " << candidate.kernel->symbol;
        symbols.push_back(candidate.kernel->symbol);
        planCandidates.push_back(std::move(*built));
    }

    const std::string key = catalog::problemKey(candidates.front().family->name, problem);
    catalog::TuneCache cache(uniqueCachePath("TunesRecordsAndCaches"));
    ASSERT_FALSE(cache.lookup(key).has_value());

    const CatalogPlan plan(std::move(planCandidates), &cache, key);

    // First execute: tunes on the real buffers, launches the winner last.
    runAndCheck(plan);

    // A winner was recorded, and it is one of the family's candidate symbols.
    const std::optional<std::string> winner = cache.lookup(key);
    ASSERT_TRUE(winner.has_value()) << "tuning did not record a winner for " << key;
    EXPECT_NE(std::find(symbols.begin(), symbols.end(), *winner), symbols.end())
        << "cached winner '" << *winner << "' is not a family candidate";

    // Second execute: cache hit -> straight to the winner, still correct.
    runAndCheck(plan);
    EXPECT_EQ(cache.lookup(key), winner);
}

// Every shipped perf variant is individually numerically correct (each built as
// a single-candidate plan, exercising the no-tuning launch path per symbol).
TEST_F(TestLayerNormSelection, EachVariantIsCorrect)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("layernorm", problem);
    ASSERT_GE(candidates.size(), 2u);

    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        const catalog::KernelEntry& kernel = *candidate.kernel;
        std::optional<PlanCandidate> built = makeCandidate(kernel);
        ASSERT_TRUE(built.has_value()) << "failed to build candidate for " << kernel.symbol;
        PlanCandidate pc = std::move(*built);
        const CatalogPlan plan(std::move(pc.module),
                               pc.launch,
                               std::move(pc.bindings),
                               std::move(pc.gridSymbols),
                               pc.workspaceBytes,
                               pc.symbol);
        SCOPED_TRACE(kernel.symbol);
        runAndCheck(plan);
    }
}

// --- bf16 coverage --------------------------------------------------------
//
// bf16 diffusion transformers normalize over N=2048 (block/QK) and N=4096 (text
// projection); these gate wave32 bf16 correctness end-to-end. Free (not fixture)
// functions because they drive their own bf16 host storage.

TEST(TestLayerNormBf16, N2048CompetesWithSpecializations)
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
        = (fs::temp_directory_path() / "hipdnn_aot_layernorm_bf16_2048.json").string();
    const ShapeOutcome o = runShapeAndCheckBf16(2048, cachePath);

    EXPECT_GE(o.symbols.size(), 4u) << "N=2048 bf16 should offer static specializations";
    EXPECT_TRUE(hasRuntimeN(o.symbols))
        << "N=2048 bf16 should also draw the runtime-N _dyn_ catch-all kernels";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=2048 bf16";
    EXPECT_NE(std::find(o.symbols.begin(), o.symbols.end(), o.winner), o.symbols.end())
        << "winner '" << o.winner << "' is not a candidate";
}

TEST(TestLayerNormBf16, N4096CompetesWithSpecializations)
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
        = (fs::temp_directory_path() / "hipdnn_aot_layernorm_bf16_4096.json").string();
    const ShapeOutcome o = runShapeAndCheckBf16(4096, cachePath);

    EXPECT_GE(o.symbols.size(), 2u) << "N=4096 bf16 should offer static specializations";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=4096 bf16";
}

// --- runtime-N (_dyn_) parity with rmsnorm2d ------------------------------
//
// These assert the LayerNorm family now has the same "static specific-shapes-
// for-perf + runtime-N general catch-all" combo RMSNorm has. On a LISTED N tier
// (2048) the dynamic kernels compete alongside the static specializations; on an
// UNLISTED multiple-of-vec N (Flux 3072, SD3.5 2432) the dynamic kernels are the
// sole match, and must still be selected and numerically correct.

// f16, listed tier: static + dynamic both compete for N=2048.
TEST(TestLayerNormRuntimeN, N2048CompetesWithSpecializations)
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
        = (fs::temp_directory_path() / "hipdnn_aot_layernorm_dyn_2048.json").string();
    const ShapeOutcome o = runShapeAndCheckF16(2048, cachePath);

    EXPECT_GE(o.symbols.size(), 4u) << "N=2048 should offer static specializations";
    EXPECT_TRUE(hasRuntimeN(o.symbols))
        << "N=2048 should also draw the runtime-N _dyn_ catch-all kernels";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=2048";
    EXPECT_NE(std::find(o.symbols.begin(), o.symbols.end(), o.winner), o.symbols.end())
        << "winner '" << o.winner << "' is not a candidate";
}

// f16, unlisted tier: only the runtime-N kernels match Flux's N=3072, and the
// tuned result is correct.
TEST(TestLayerNormRuntimeN, N3072RuntimeNOnly)
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
        = (fs::temp_directory_path() / "hipdnn_aot_layernorm_dyn_3072.json").string();
    const ShapeOutcome o = runShapeAndCheckF16(3072, cachePath);

    ASSERT_FALSE(o.symbols.empty()) << "N=3072 should be served by the runtime-N kernels";
    EXPECT_TRUE(allRuntimeN(o.symbols))
        << "only runtime-N _dyn_ kernels should match the unlisted N=3072";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=3072";
}

// f16, unlisted tier: SD3.5's N=2432 (multiple of 8, not a static tier).
TEST(TestLayerNormRuntimeN, N2432RuntimeNOnly)
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
        = (fs::temp_directory_path() / "hipdnn_aot_layernorm_dyn_2432.json").string();
    const ShapeOutcome o = runShapeAndCheckF16(2432, cachePath);

    ASSERT_FALSE(o.symbols.empty()) << "N=2432 should be served by the runtime-N kernels";
    EXPECT_TRUE(allRuntimeN(o.symbols))
        << "only runtime-N _dyn_ kernels should match the unlisted N=2432";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=2432";
}

// bf16, unlisted tier: the diffusion dtype path through the runtime-N kernels.
TEST(TestLayerNormBf16RuntimeN, N3072RuntimeNOnly)
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
        = (fs::temp_directory_path() / "hipdnn_aot_layernorm_bf16_dyn_3072.json").string();
    const ShapeOutcome o = runShapeAndCheckBf16(3072, cachePath);

    ASSERT_FALSE(o.symbols.empty()) << "N=3072 bf16 should be served by the runtime-N kernels";
    EXPECT_TRUE(allRuntimeN(o.symbols))
        << "only runtime-N _dyn_ kernels should match the unlisted bf16 N=3072";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for bf16 N=3072";
}
