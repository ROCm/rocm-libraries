// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU test for the AOT catalog engine's Phase 2 measure-and-cache selection over
// the multi-kernel rmsnorm2d f16 family. The N=2048 family ships several
// perf-only (block_size/vec) variants that all decode to the same {dtype,M,N}
// problem, so they are all candidates for one plan. This test drives the
// substrate directly (like the parity tests): it builds a multi-candidate
// CatalogPlan + a TuneCache pointed at a unique temp file and checks that
//   1. the first execute tunes, records a winner in the cache, and is correct,
//   2. a second execute hits the cache and stays correct,
//   3. every shipped variant is individually numerically correct.

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

float xVal(size_t m, size_t n)
{
    return (static_cast<float>((m * 13u + n * 7u) % 7u) - 3.0f) * 0.1f;
}
float gammaVal(size_t n)
{
    return static_cast<float>((n * 5u + 3u) % 5u) * 0.25f;
}

// Build the (X,Gamma,Y,M,N,eps) launch bindings + grid symbols the RmsNormAdapter
// would produce -- identical for every perf variant of the family.
// Returns nullopt when the module fails to load. A value-returning helper cannot
// host ASSERT_* (it expands to `return;`), so the module check must be a fatal
// assertion at the (void) call site -- otherwise an EXPECT here would record the
// failure but let execution fall through to `*module`, dereferencing an empty
// optional. Callers ASSERT_TRUE(has_value()) before using the result.
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
    bindings.pointerUids.emplace("Y", 3);
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

// Per-row RMS-norm reference: rms=sqrt(mean(x^2)+eps); y=x/rms*gamma.
std::vector<float> reference(const std::vector<_Float16>& hostX,
                             const std::vector<_Float16>& hostGamma)
{
    std::vector<float> ref(M * N, 0.0f);
    for(size_t m = 0; m < M; ++m)
    {
        float sumSquares = 0.0f;
        for(size_t n = 0; n < N; ++n)
        {
            const auto x = static_cast<float>(hostX[m * N + n]);
            sumSquares += x * x;
        }
        const float invRms = 1.0f / std::sqrt(sumSquares / static_cast<float>(N) + EPS);
        for(size_t n = 0; n < N; ++n)
        {
            const auto x = static_cast<float>(hostX[m * N + n]);
            const auto g = static_cast<float>(hostGamma[n]);
            ref[m * N + n] = x * invRms * g;
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

// Fixture holding the device buffers + a stream; hosts X/Gamma and the CPU ref.
class TestRmsNormSelection : public ::testing::Test
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
        _hostY.assign(M * N, static_cast<_Float16>(0.0f));
        for(size_t n = 0; n < N; ++n)
        {
            _hostGamma[n] = static_cast<_Float16>(gammaVal(n));
        }
        for(size_t m = 0; m < M; ++m)
        {
            for(size_t n = 0; n < N; ++n)
            {
                _hostX[m * N + n] = static_cast<_Float16>(xVal(m, n));
            }
        }
        _ref = reference(_hostX, _hostGamma);

        ASSERT_EQ(hipMalloc(&_deviceX, M * N * sizeof(_Float16)), hipSuccess);
        ASSERT_EQ(hipMalloc(&_deviceGamma, N * sizeof(_Float16)), hipSuccess);
        ASSERT_EQ(hipMalloc(&_deviceY, M * N * sizeof(_Float16)), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(_deviceX, _hostX.data(), M * N * sizeof(_Float16), hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(_deviceGamma, _hostGamma.data(), N * sizeof(_Float16), hipMemcpyHostToDevice),
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
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers = {{
            {1, _deviceX},
            {2, _deviceGamma},
            {3, _deviceY},
        }};
        ASSERT_EQ(hipMemset(_deviceY, 0, M * N * sizeof(_Float16)), hipSuccess);
        ASSERT_NO_THROW(plan.execute(_handle, buffers.data(), 3, nullptr));
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(_hostY.data(), _deviceY, M * N * sizeof(_Float16), hipMemcpyDeviceToHost),
            hipSuccess);
        expectMatches(_hostY, _ref);
    }

    std::string uniqueCachePath(const std::string& tag)
    {
        _cachePath
            = (fs::temp_directory_path() / ("hipdnn_aot_rmsnorm_sel_" + tag + ".json")).string();
        std::error_code ec;
        fs::remove(_cachePath, ec);
        fs::remove(_cachePath + ".tmp", ec);
        return _cachePath;
    }

    std::vector<_Float16> _hostX;
    std::vector<_Float16> _hostGamma;
    std::vector<_Float16> _hostY;
    std::vector<float> _ref;
    void* _deviceX = nullptr;
    void* _deviceGamma = nullptr;
    void* _deviceY = nullptr;
    hipStream_t _stream = nullptr;
    Handle _handle;
    std::string _cachePath;
};

// ---- Generic (arbitrary M,N) helpers for the runtime-N variants ----
//
// The fixture above bakes M/N as compile-time constants (it predates the
// runtime-N kernels). These free helpers drive the same substrate for an
// arbitrary (rows, cols) so the runtime-N tests can hit unlisted shapes
// (Flux 3072, SD3.5 2432) that the static specializations don't cover. They
// use only EXPECT_* because a value-returning helper can't host ASSERT_*.

std::vector<float> referenceForShape(size_t rows,
                                     size_t cols,
                                     const std::vector<_Float16>& hostX,
                                     const std::vector<_Float16>& hostGamma,
                                     float eps)
{
    std::vector<float> ref(rows * cols, 0.0f);
    for(size_t m = 0; m < rows; ++m)
    {
        float sumSquares = 0.0f;
        for(size_t n = 0; n < cols; ++n)
        {
            const auto x = static_cast<float>(hostX[m * cols + n]);
            sumSquares += x * x;
        }
        const float invRms = 1.0f / std::sqrt(sumSquares / static_cast<float>(cols) + eps);
        for(size_t n = 0; n < cols; ++n)
        {
            const auto x = static_cast<float>(hostX[m * cols + n]);
            const auto g = static_cast<float>(hostGamma[n]);
            ref[m * cols + n] = x * invRms * g;
        }
    }
    return ref;
}

// Returns nullopt on module-load failure (see makeCandidate). The callers here
// return ShapeOutcome (non-void), so they cannot ASSERT_*; on nullopt they
// ADD_FAILURE and bail with the partial outcome, which the test-level ASSERTs
// then flag via the (empty) symbol list.
std::optional<PlanCandidate>
    makeCandidateForShape(const catalog::KernelEntry& kernel, size_t rows, size_t cols, float eps)
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
    bindings.pointerUids.emplace("Y", 3);
    bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(rows)});
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(cols)});
    bindings.scalars.emplace("eps", catalog::ScalarValue{eps});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("M", static_cast<int64_t>(rows));
    gridSymbols.emplace("N", static_cast<int64_t>(cols));

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
    std::vector<std::string> symbols; // candidate symbols, in catalog order
    std::string winner; // tuned winner (empty if single/none)
};

// Full end-to-end for one (rows, cols): load candidates, allocate real device
// buffers, run a multi-candidate tuned plan, and EXPECT numerical correctness.
// Returns the candidate symbols + the tuned winner; removes its cache file.
ShapeOutcome runShapeAndCheck(size_t rows, size_t cols, float eps, const std::string& cachePath)
{
    ShapeOutcome outcome;

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    EXPECT_FALSE(cat.empty());

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(rows)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(cols)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("rmsnorm", problem);
    if(candidates.empty())
    {
        return outcome; // caller asserts on the (empty) symbol list
    }

    std::vector<PlanCandidate> planCandidates;
    planCandidates.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        std::optional<PlanCandidate> built
            = makeCandidateForShape(*candidate.kernel, rows, cols, eps);
        if(!built.has_value())
        {
            ADD_FAILURE() << "failed to build candidate for " << candidate.kernel->symbol;
            return outcome; // bail with the partial outcome; caller asserts on it
        }
        outcome.symbols.push_back(candidate.kernel->symbol);
        planCandidates.push_back(std::move(*built));
    }

    // Host inputs + CPU reference.
    std::vector<_Float16> hostX(rows * cols);
    std::vector<_Float16> hostGamma(cols);
    std::vector<_Float16> hostY(rows * cols, static_cast<_Float16>(0.0f));
    for(size_t n = 0; n < cols; ++n)
    {
        hostGamma[n] = static_cast<_Float16>(gammaVal(n));
    }
    for(size_t m = 0; m < rows; ++m)
    {
        for(size_t n = 0; n < cols; ++n)
        {
            hostX[m * cols + n] = static_cast<_Float16>(xVal(m, n));
        }
    }
    const std::vector<float> ref = referenceForShape(rows, cols, hostX, hostGamma, eps);

    // Device buffers + stream.
    void* deviceX = nullptr;
    void* deviceGamma = nullptr;
    void* deviceY = nullptr;
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipMalloc(&deviceX, rows * cols * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceGamma, cols * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceY, rows * cols * sizeof(_Float16)), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceX, hostX.data(), rows * cols * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceGamma, hostGamma.data(), cols * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::string key = catalog::problemKey(candidates.front().family->name, problem);
    catalog::TuneCache cache(cachePath);
    const CatalogPlan plan(std::move(planCandidates), &cache, key);

    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers = {{
        {1, deviceX},
        {2, deviceGamma},
        {3, deviceY},
    }};
    EXPECT_EQ(hipMemset(deviceY, 0, rows * cols * sizeof(_Float16)), hipSuccess);
    EXPECT_NO_THROW(plan.execute(handle, buffers.data(), 3, nullptr));
    EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(hostY.data(), deviceY, rows * cols * sizeof(_Float16), hipMemcpyDeviceToHost),
        hipSuccess);

    // Correctness (single aggregate EXPECT to avoid per-element spam).
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
    EXPECT_EQ(mismatches, 0u) << "M=" << rows << " N=" << cols << " first mismatch "
                              << firstMismatch;

    const std::optional<std::string> winner = cache.lookup(key);
    if(winner.has_value())
    {
        outcome.winner = *winner;
    }

    (void)hipFree(deviceX);
    (void)hipFree(deviceGamma);
    (void)hipFree(deviceY);
    (void)hipStreamDestroy(stream);
    std::error_code ec;
    fs::remove(cachePath, ec);
    fs::remove(cachePath + ".tmp", ec);

    return outcome;
}

// ---- bf16 variant (LTX-Video runs bf16) ----------------------------------
//
// The helpers above are f16 (_Float16); bf16 needs its own host storage. We
// represent bf16 on the host as the raw 16-bit pattern (bfloat16 == the top 16
// bits of an IEEE f32), so the device buffers are just uint16_t byte-for-byte
// what the kernel reads -- no HIP bf16 header needed. wave32 correctness of the
// bf16 reduction is the top risk of the bf16 work; these tests are its gate.

uint16_t floatToBf16(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t roundingBias = ((bits >> 16) & 1u) + 0x7fffu; // round to nearest even
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16ToFloat(uint16_t b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// bf16 analog of runShapeAndCheck: loads the bf16 family candidates for one
// (rows, cols), runs a multi-candidate tuned plan on real device buffers, and
// EXPECTs numerical correctness at a bf16-appropriate tolerance.
ShapeOutcome runShapeAndCheckBf16(size_t rows, size_t cols, float eps, const std::string& cachePath)
{
    ShapeOutcome outcome;

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    EXPECT_FALSE(cat.empty());

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("bf16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(rows)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(cols)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("rmsnorm", problem);
    if(candidates.empty())
    {
        return outcome; // caller asserts on the (empty) symbol list
    }

    std::vector<PlanCandidate> planCandidates;
    planCandidates.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        std::optional<PlanCandidate> built
            = makeCandidateForShape(*candidate.kernel, rows, cols, eps);
        if(!built.has_value())
        {
            ADD_FAILURE() << "failed to build candidate for " << candidate.kernel->symbol;
            return outcome; // bail with the partial outcome; caller asserts on it
        }
        outcome.symbols.push_back(candidate.kernel->symbol);
        planCandidates.push_back(std::move(*built));
    }

    // Host inputs (bf16 bit patterns) + f32 CPU reference from the same values.
    std::vector<uint16_t> hostX(rows * cols);
    std::vector<uint16_t> hostGamma(cols);
    std::vector<uint16_t> hostY(rows * cols, 0);
    for(size_t n = 0; n < cols; ++n)
    {
        hostGamma[n] = floatToBf16(gammaVal(n));
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
        float sumSquares = 0.0f;
        for(size_t n = 0; n < cols; ++n)
        {
            const float x = bf16ToFloat(hostX[m * cols + n]);
            sumSquares += x * x;
        }
        const float invRms = 1.0f / std::sqrt(sumSquares / static_cast<float>(cols) + eps);
        for(size_t n = 0; n < cols; ++n)
        {
            const float x = bf16ToFloat(hostX[m * cols + n]);
            const float g = bf16ToFloat(hostGamma[n]);
            ref[m * cols + n] = x * invRms * g;
        }
    }

    void* deviceX = nullptr;
    void* deviceGamma = nullptr;
    void* deviceY = nullptr;
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipMalloc(&deviceX, rows * cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceGamma, cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(hipMalloc(&deviceY, rows * cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceX, hostX.data(), rows * cols * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(
        hipMemcpy(deviceGamma, hostGamma.data(), cols * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::string key = catalog::problemKey(candidates.front().family->name, problem);
    catalog::TuneCache cache(cachePath);
    const CatalogPlan plan(std::move(planCandidates), &cache, key);

    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers = {{
        {1, deviceX},
        {2, deviceGamma},
        {3, deviceY},
    }};
    EXPECT_EQ(hipMemset(deviceY, 0, rows * cols * sizeof(uint16_t)), hipSuccess);
    EXPECT_NO_THROW(plan.execute(handle, buffers.data(), 3, nullptr));
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
    EXPECT_EQ(mismatches, 0u) << "bf16 M=" << rows << " N=" << cols << " first mismatch "
                              << firstMismatch;

    const std::optional<std::string> winner = cache.lookup(key);
    if(winner.has_value())
    {
        outcome.winner = *winner;
    }

    (void)hipFree(deviceX);
    (void)hipFree(deviceGamma);
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
    return !symbols.empty()
           && std::all_of(symbols.begin(), symbols.end(), [](const std::string& s) {
                  return s.find("_dyn_") != std::string::npos;
              });
}

} // namespace

// The N=2048 family exposes multiple perf variants that all match one problem.
TEST_F(TestRmsNormSelection, MultipleCandidatesForOneProblem)
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
        = cat.candidatesFor("rmsnorm", problem);
    // Four static N=2048 specializations + the two runtime-N variants
    // (multiple_of 4 / 8, both satisfied by 2048) all decode to this problem.
    EXPECT_GE(candidates.size(), 5u)
        << "expected the N=2048 static specializations + runtime-N variants";
    const bool hasDyn = std::any_of(
        candidates.begin(), candidates.end(), [](const catalog::Catalog::Candidate& c) {
            return c.kernel->symbol.find("_dyn_") != std::string::npos;
        });
    EXPECT_TRUE(hasDyn) << "runtime-N kernel should be a candidate for N=2048";
}

// First execute tunes (records a winner among the candidates) and is correct;
// the second execute hits the cache and stays correct.
TEST_F(TestRmsNormSelection, TunesRecordsAndCaches)
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
        = cat.candidatesFor("rmsnorm", problem);
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
TEST_F(TestRmsNormSelection, EachVariantIsCorrect)
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
        = cat.candidatesFor("rmsnorm", problem);
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

// --- Runtime-N variant coverage (Phase 3) --------------------------------
//
// The two _dyn_ kernels read N as a runtime arg and carry a multiple_of N
// rule, so they match many shapes from one binary. These tests are free
// (not fixture) functions because they drive arbitrary (M,N) via the generic
// helpers above; each skips when no gfx1151 GPU is present.

// On the listed N=2048 the runtime-N kernels join the static specializations
// as candidates, the tuner records a valid winner, and output is correct.
TEST(TestRmsNormRuntimeN, N2048CompetesWithSpecializations)
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
        = (fs::temp_directory_path() / "hipdnn_aot_rmsnorm_dyn_2048.json").string();
    const ShapeOutcome o = runShapeAndCheck(8, 2048, EPS, cachePath);

    EXPECT_GE(o.symbols.size(), 5u)
        << "N=2048 should offer static specializations + runtime-N variants";
    EXPECT_TRUE(hasRuntimeN(o.symbols)) << "runtime-N kernel missing from N=2048 candidates";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=2048";
    EXPECT_NE(std::find(o.symbols.begin(), o.symbols.end(), o.winner), o.symbols.end())
        << "winner '" << o.winner << "' is not a candidate";
}

// Flux hidden size 3072 is unlisted by the static family, so ONLY the
// runtime-N kernels match -- and they are numerically correct.
TEST(TestRmsNormRuntimeN, N3072RuntimeNOnly)
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
        = (fs::temp_directory_path() / "hipdnn_aot_rmsnorm_dyn_3072.json").string();
    const ShapeOutcome o = runShapeAndCheck(8, 3072, EPS, cachePath);

    ASSERT_FALSE(o.symbols.empty()) << "no candidate matched the unlisted N=3072";
    EXPECT_TRUE(allRuntimeN(o.symbols))
        << "only runtime-N kernels should match the unlisted N=3072";
    EXPECT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=3072";
}

// SD3.5 hidden size 2432 (also unlisted, also a multiple of 8): runtime-N is
// again the sole match and correct. Covers a second real-workload shape that
// no static specialization ships.
TEST(TestRmsNormRuntimeN, N2432RuntimeNOnly)
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
        = (fs::temp_directory_path() / "hipdnn_aot_rmsnorm_dyn_2432.json").string();
    const ShapeOutcome o = runShapeAndCheck(8, 2432, EPS, cachePath);

    ASSERT_FALSE(o.symbols.empty()) << "no candidate matched the unlisted N=2432";
    EXPECT_TRUE(allRuntimeN(o.symbols))
        << "only runtime-N kernels should match the unlisted N=2432";
    EXPECT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=2432";
}

// --- bf16 coverage (LTX-Video) --------------------------------------------
//
// LTX-Video runs bf16 and normalizes over N=2048 (block/QK norms) and N=4096
// (text projection). These gate wave32 bf16 correctness end-to-end through the
// same substrate the f16 tests use.

// N=2048 bf16: the four static specializations + two runtime-N variants all
// match, the tuner records a valid winner, and output is correct.
TEST(TestRmsNormBf16, N2048CompetesWithSpecializations)
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
        = (fs::temp_directory_path() / "hipdnn_aot_rmsnorm_bf16_2048.json").string();
    const ShapeOutcome o = runShapeAndCheckBf16(8, 2048, EPS, cachePath);

    EXPECT_GE(o.symbols.size(), 5u)
        << "N=2048 bf16 should offer static specializations + runtime-N variants";
    EXPECT_TRUE(hasRuntimeN(o.symbols)) << "runtime-N kernel missing from N=2048 bf16 candidates";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=2048 bf16";
    EXPECT_NE(std::find(o.symbols.begin(), o.symbols.end(), o.winner), o.symbols.end())
        << "winner '" << o.winner << "' is not a candidate";
}

// N=4096 bf16 (LTX text-projection hidden size): static specializations +
// runtime-N variants match, and output is correct.
TEST(TestRmsNormBf16, N4096CompetesWithSpecializations)
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
        = (fs::temp_directory_path() / "hipdnn_aot_rmsnorm_bf16_4096.json").string();
    const ShapeOutcome o = runShapeAndCheckBf16(8, 4096, EPS, cachePath);

    EXPECT_GE(o.symbols.size(), 3u)
        << "N=4096 bf16 should offer static specializations + runtime-N variants";
    EXPECT_TRUE(hasRuntimeN(o.symbols)) << "runtime-N kernel missing from N=4096 bf16 candidates";
    ASSERT_FALSE(o.winner.empty()) << "tuning recorded no winner for N=4096 bf16";
}

// An unlisted bf16 N=3072: only the runtime-N kernels match, and correct.
TEST(TestRmsNormBf16, N3072RuntimeNOnly)
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
        = (fs::temp_directory_path() / "hipdnn_aot_rmsnorm_bf16_3072.json").string();
    const ShapeOutcome o = runShapeAndCheckBf16(8, 3072, EPS, cachePath);

    ASSERT_FALSE(o.symbols.empty()) << "no bf16 candidate matched the unlisted N=3072";
    EXPECT_TRUE(allRuntimeN(o.symbols))
        << "only runtime-N kernels should match the unlisted bf16 N=3072";
    EXPECT_FALSE(o.winner.empty()) << "tuning recorded no winner for bf16 N=3072";
}
