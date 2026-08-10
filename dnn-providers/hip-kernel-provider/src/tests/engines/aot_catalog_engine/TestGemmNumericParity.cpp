// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU numeric-parity test for the AOT catalog engine's GEMM path. This drives
// the *substrate* directly (Catalog load -> candidate selection -> module load
// -> LaunchAbi pack/grid via CatalogPlan::execute) against the shipped gfx1151
// wmma_gemm f16 .co, and compares to a CPU reference.
//
// It deliberately bypasses the hipDNN frontend: the shipped kernel is RCR
// (C[M,N] = A[M,K] * B[N,K]^T, i.e. nn.Linear's y = x @ W^T), whereas the
// frontend's CpuReferenceGraphExecutor computes standard C = A @ B. So the
// frontend harness cannot validate this kernel; a direct substrate test can,
// and it exercises exactly the catalog/selection/launch-ABI code we forked.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "core/Handle.hpp"
#include "launch/LaunchAbi.hpp"
#include "launch/ModuleLoader.hpp"
#include "plans/CatalogPlan.hpp"

namespace
{

using namespace aot_catalog_engine;

// Baked in by CMake: the build-tree copy of the catalog (<arch>/<family>/...).
constexpr const char* kCatalogDir = AOT_CATALOG_TEST_DIR;

// The shipped .co is gfx1151-only; the test is meaningful only on that arch.
constexpr const char* kArch = "gfx1151";

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

// Deterministic small inputs kept well within f16 range so the K-length sum
// stays exact enough for a tight tolerance.
float aVal(size_t i, size_t k)
{
    return static_cast<float>((i * 7u + k * 3u) % 5u) * 0.25f;
}
float bVal(size_t j, size_t k)
{
    return static_cast<float>((j * 5u + k * 2u) % 4u) * 0.25f;
}

// bf16 host storage: bfloat16 == the top 16 bits of an IEEE f32, so the device
// buffers are just uint16_t byte-for-byte what the kernel reads -- no HIP bf16
// header needed. wave32 correctness of the bf16 WMMA path is the top risk of the
// bf16 GEMM work; the WmmaGemmBf16 test below is its gate. (Same helpers as
// TestRmsNormSelection.cpp's bf16 gate.)
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

// Multiple matmul families can match one problem (the reference gemm_wmma_* is
// mult-of-16, the tiled gemm_wmma_universal_* is mult-of-64), so candidates.front()
// is ambiguous. The universal-GEMM tests below pick their kernel explicitly by the
// "ugemm" symbol prefix the tiled builder emits.
const catalog::KernelEntry*
    findCandidateBySymbol(const std::vector<catalog::Catalog::Candidate>& candidates,
                          const std::string& needle)
{
    for(const auto& cand : candidates)
    {
        if(cand.kernel->symbol.find(needle) != std::string::npos)
        {
            return cand.kernel;
        }
    }
    return nullptr;
}

} // namespace

TEST(TestAotCatalogGemmNumericParity, WmmaGemmF16MatchesReference)
{
    if(!gpuIsArch(kArch))
    {
        GTEST_SKIP() << "no " << kArch << " GPU present";
    }

    // 1. Load the catalog and select the GEMM kernel for an f16 M/N/K problem.
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(kCatalogDir, kArch);
    if(cat.empty())
    {
        GTEST_SKIP() << "empty AOT catalog at " << kCatalogDir
                     << "; build with -DROCKE_PYTHON_DIR to populate it (see the engine README)";
    }

    constexpr size_t M = 64;
    constexpr size_t N = 64;
    constexpr size_t K = 64;

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});
    problem.emplace("K", catalog::ShapeValue{static_cast<int64_t>(K)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("matmul", problem);
    ASSERT_FALSE(candidates.empty()) << "no matmul candidate for the f16 problem";
    const catalog::KernelEntry& kernel = *candidates.front().kernel;

    // 2. Load the module for the selected kernel.
    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    // 3. Build the launch bindings by hand (the GemmAdapter builds these from a
    //    graph; here we assign the uids ourselves and match them in the device
    //    buffer table below). This is exactly the (A,B,C,M,N,K) ABI.
    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", 1);
    bindings.pointerUids.emplace("B", 2);
    bindings.pointerUids.emplace("C", 3);
    bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(M)});
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(N)});
    bindings.scalars.emplace("K", catalog::ScalarValue{static_cast<int64_t>(K)});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("M", static_cast<int64_t>(M));
    gridSymbols.emplace("N", static_cast<int64_t>(N));
    gridSymbols.emplace("K", static_cast<int64_t>(K));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           kernel.workspaceBytes,
                           kernel.symbol);

    // 4. Host inputs: A[M,K], B[N,K] row-major, f16 (== _Float16). Reference is
    //    the RCR product C[M,N] = A * B^T computed in float.
    std::vector<_Float16> hostA(M * K);
    std::vector<_Float16> hostB(N * K);
    std::vector<_Float16> hostC(M * N, static_cast<_Float16>(0.0f));
    std::vector<float> reference(M * N, 0.0f);

    for(size_t i = 0; i < M; ++i)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostA[i * K + k] = static_cast<_Float16>(aVal(i, k));
        }
    }
    for(size_t j = 0; j < N; ++j)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostB[j * K + k] = static_cast<_Float16>(bVal(j, k));
        }
    }
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(size_t k = 0; k < K; ++k)
            {
                sum += aVal(i, k) * bVal(j, k);
            }
            reference[i * N + j] = sum;
        }
    }

    // 5. Device buffers + execute through the plan.
    void* deviceA = nullptr;
    void* deviceB = nullptr;
    void* deviceC = nullptr;
    ASSERT_EQ(hipMalloc(&deviceA, hostA.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceB, hostB.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceC, hostC.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(deviceC, 0, hostC.size() * sizeof(_Float16)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const hipdnnPluginDeviceBuffer_t buffers[3] = {
        {1, deviceA},
        {2, deviceB},
        {3, deviceC},
    };

    ASSERT_NO_THROW(plan.execute(handle, buffers, 3, nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(hostC.data(), deviceC, hostC.size() * sizeof(_Float16), hipMemcpyDeviceToHost),
        hipSuccess);

    // 6. Compare. f16 has ~3 decimal digits; scale tolerance with magnitude.
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            const float got = static_cast<float>(hostC[i * N + j]);
            const float want = reference[i * N + j];
            const float tol = std::max(1e-2f, 2e-2f * std::fabs(want));
            ASSERT_NEAR(got, want, tol) << "mismatch at (" << i << "," << j << ")";
        }
    }

    (void)hipFree(deviceA);
    (void)hipFree(deviceB);
    (void)hipFree(deviceC);
    (void)hipStreamDestroy(stream);
}

// Tiled (LDS-staged, register-blocked) universal-GEMM f16 parity. Same substrate
// path as the reference test, but selects the ugemm_* kernel explicitly and runs
// at a tile-crossing size (M=N=128, K=64 -> a 2x2 grid of 64x64 tiles, 2 K-steps
// of 32). Non-square-in-blocks + multi-block is the gate on the grid_order NM
// contract (block_id.x -> N, block_id.y -> M) that is INVERTED vs the reference
// kernel: get it backwards and this test computes a transposed/garbage C.
TEST(TestAotCatalogGemmNumericParity, WmmaUniversalGemmF16MatchesReference)
{
    if(!gpuIsArch(kArch))
    {
        GTEST_SKIP() << "no " << kArch << " GPU present";
    }

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(kCatalogDir, kArch);
    if(cat.empty())
    {
        GTEST_SKIP() << "empty AOT catalog at " << kCatalogDir
                     << "; build with -DROCKE_PYTHON_DIR to populate it (see the engine README)";
    }

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 64;

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});
    problem.emplace("K", catalog::ShapeValue{static_cast<int64_t>(K)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("matmul", problem);
    ASSERT_FALSE(candidates.empty()) << "no matmul candidate for the f16 problem";
    const catalog::KernelEntry* kernelPtr = findCandidateBySymbol(candidates, "ugemm");
    ASSERT_NE(kernelPtr, nullptr) << "no tiled ugemm_* f16 candidate for M=N=128,K=64";
    const catalog::KernelEntry& kernel = *kernelPtr;

    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", 1);
    bindings.pointerUids.emplace("B", 2);
    bindings.pointerUids.emplace("C", 3);
    bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(M)});
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(N)});
    bindings.scalars.emplace("K", catalog::ScalarValue{static_cast<int64_t>(K)});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("M", static_cast<int64_t>(M));
    gridSymbols.emplace("N", static_cast<int64_t>(N));
    gridSymbols.emplace("K", static_cast<int64_t>(K));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           kernel.workspaceBytes,
                           kernel.symbol);

    std::vector<_Float16> hostA(M * K);
    std::vector<_Float16> hostB(N * K);
    std::vector<_Float16> hostC(M * N, static_cast<_Float16>(0.0f));
    std::vector<float> reference(M * N, 0.0f);

    for(size_t i = 0; i < M; ++i)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostA[i * K + k] = static_cast<_Float16>(aVal(i, k));
        }
    }
    for(size_t j = 0; j < N; ++j)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostB[j * K + k] = static_cast<_Float16>(bVal(j, k));
        }
    }
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(size_t k = 0; k < K; ++k)
            {
                sum += aVal(i, k) * bVal(j, k);
            }
            reference[i * N + j] = sum;
        }
    }

    void* deviceA = nullptr;
    void* deviceB = nullptr;
    void* deviceC = nullptr;
    ASSERT_EQ(hipMalloc(&deviceA, hostA.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceB, hostB.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceC, hostC.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(_Float16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(deviceC, 0, hostC.size() * sizeof(_Float16)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const hipdnnPluginDeviceBuffer_t buffers[3] = {
        {1, deviceA},
        {2, deviceB},
        {3, deviceC},
    };

    ASSERT_NO_THROW(plan.execute(handle, buffers, 3, nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(hostC.data(), deviceC, hostC.size() * sizeof(_Float16), hipMemcpyDeviceToHost),
        hipSuccess);

    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            const float got = static_cast<float>(hostC[i * N + j]);
            const float want = reference[i * N + j];
            const float tol = std::max(1e-2f, 2e-2f * std::fabs(want));
            ASSERT_NEAR(got, want, tol) << "mismatch at (" << i << "," << j << ")";
        }
    }

    (void)hipFree(deviceA);
    (void)hipFree(deviceB);
    (void)hipFree(deviceC);
    (void)hipStreamDestroy(stream);
}

// Tiled universal-GEMM bf16 parity -- bf16 sibling of the tiled f16 test above,
// same tile-crossing M=N=128, K=64 and same ugemm_* selection.
TEST(TestAotCatalogGemmNumericParity, WmmaUniversalGemmBf16MatchesReference)
{
    if(!gpuIsArch(kArch))
    {
        GTEST_SKIP() << "no " << kArch << " GPU present";
    }

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(kCatalogDir, kArch);
    if(cat.empty())
    {
        GTEST_SKIP() << "empty AOT catalog at " << kCatalogDir
                     << "; build with -DROCKE_PYTHON_DIR to populate it (see the engine README)";
    }

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 64;

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("bf16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});
    problem.emplace("K", catalog::ShapeValue{static_cast<int64_t>(K)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("matmul", problem);
    ASSERT_FALSE(candidates.empty()) << "no matmul candidate for the bf16 problem";
    const catalog::KernelEntry* kernelPtr = findCandidateBySymbol(candidates, "ugemm");
    ASSERT_NE(kernelPtr, nullptr) << "no tiled ugemm_* bf16 candidate for M=N=128,K=64";
    const catalog::KernelEntry& kernel = *kernelPtr;

    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", 1);
    bindings.pointerUids.emplace("B", 2);
    bindings.pointerUids.emplace("C", 3);
    bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(M)});
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(N)});
    bindings.scalars.emplace("K", catalog::ScalarValue{static_cast<int64_t>(K)});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("M", static_cast<int64_t>(M));
    gridSymbols.emplace("N", static_cast<int64_t>(N));
    gridSymbols.emplace("K", static_cast<int64_t>(K));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           kernel.workspaceBytes,
                           kernel.symbol);

    std::vector<uint16_t> hostA(M * K);
    std::vector<uint16_t> hostB(N * K);
    std::vector<uint16_t> hostC(M * N, 0);
    std::vector<float> reference(M * N, 0.0f);

    for(size_t i = 0; i < M; ++i)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostA[i * K + k] = floatToBf16(aVal(i, k));
        }
    }
    for(size_t j = 0; j < N; ++j)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostB[j * K + k] = floatToBf16(bVal(j, k));
        }
    }
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(size_t k = 0; k < K; ++k)
            {
                sum += bf16ToFloat(hostA[i * K + k]) * bf16ToFloat(hostB[j * K + k]);
            }
            reference[i * N + j] = sum;
        }
    }

    void* deviceA = nullptr;
    void* deviceB = nullptr;
    void* deviceC = nullptr;
    ASSERT_EQ(hipMalloc(&deviceA, hostA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceB, hostB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceC, hostC.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(deviceC, 0, hostC.size() * sizeof(uint16_t)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const hipdnnPluginDeviceBuffer_t buffers[3] = {
        {1, deviceA},
        {2, deviceB},
        {3, deviceC},
    };

    ASSERT_NO_THROW(plan.execute(handle, buffers, 3, nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(hostC.data(), deviceC, hostC.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        hipSuccess);

    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            const float got = bf16ToFloat(hostC[i * N + j]);
            const float want = reference[i * N + j];
            const float tol = std::max(5e-2f, 4e-2f * std::fabs(want));
            ASSERT_NEAR(got, want, tol) << "mismatch at (" << i << "," << j << ")";
        }
    }

    (void)hipFree(deviceA);
    (void)hipFree(deviceB);
    (void)hipFree(deviceC);
    (void)hipStreamDestroy(stream);
}

// bf16 analog of the f16 test: same substrate path (Catalog -> candidate ->
// module load -> LaunchAbi pack/grid) against the shipped gfx1151 wmma_gemm bf16
// .co, added self-serve as a sibling family (dtype constraint bf16) with NO
// engine change. RCR C[M,N] = A[M,K] * B[N,K]^T, bf16 in / f32 acc / bf16 out.
TEST(TestAotCatalogGemmNumericParity, WmmaGemmBf16MatchesReference)
{
    if(!gpuIsArch(kArch))
    {
        GTEST_SKIP() << "no " << kArch << " GPU present";
    }

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(kCatalogDir, kArch);
    if(cat.empty())
    {
        GTEST_SKIP() << "empty AOT catalog at " << kCatalogDir
                     << "; build with -DROCKE_PYTHON_DIR to populate it (see the engine README)";
    }

    constexpr size_t M = 64;
    constexpr size_t N = 64;
    constexpr size_t K = 64;

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("bf16")});
    problem.emplace("M", catalog::ShapeValue{static_cast<int64_t>(M)});
    problem.emplace("N", catalog::ShapeValue{static_cast<int64_t>(N)});
    problem.emplace("K", catalog::ShapeValue{static_cast<int64_t>(K)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("matmul", problem);
    ASSERT_FALSE(candidates.empty()) << "no matmul candidate for the bf16 problem";
    const catalog::KernelEntry& kernel = *candidates.front().kernel;

    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", 1);
    bindings.pointerUids.emplace("B", 2);
    bindings.pointerUids.emplace("C", 3);
    bindings.scalars.emplace("M", catalog::ScalarValue{static_cast<int64_t>(M)});
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(N)});
    bindings.scalars.emplace("K", catalog::ScalarValue{static_cast<int64_t>(K)});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("M", static_cast<int64_t>(M));
    gridSymbols.emplace("N", static_cast<int64_t>(N));
    gridSymbols.emplace("K", static_cast<int64_t>(K));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           kernel.workspaceBytes,
                           kernel.symbol);

    // Host inputs as bf16 bit patterns; f32 CPU reference from the same values.
    std::vector<uint16_t> hostA(M * K);
    std::vector<uint16_t> hostB(N * K);
    std::vector<uint16_t> hostC(M * N, 0);
    std::vector<float> reference(M * N, 0.0f);

    for(size_t i = 0; i < M; ++i)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostA[i * K + k] = floatToBf16(aVal(i, k));
        }
    }
    for(size_t j = 0; j < N; ++j)
    {
        for(size_t k = 0; k < K; ++k)
        {
            hostB[j * K + k] = floatToBf16(bVal(j, k));
        }
    }
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(size_t k = 0; k < K; ++k)
            {
                sum += bf16ToFloat(hostA[i * K + k]) * bf16ToFloat(hostB[j * K + k]);
            }
            reference[i * N + j] = sum;
        }
    }

    void* deviceA = nullptr;
    void* deviceB = nullptr;
    void* deviceC = nullptr;
    ASSERT_EQ(hipMalloc(&deviceA, hostA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceB, hostB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceC, hostC.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(deviceC, 0, hostC.size() * sizeof(uint16_t)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const hipdnnPluginDeviceBuffer_t buffers[3] = {
        {1, deviceA},
        {2, deviceB},
        {3, deviceC},
    };

    ASSERT_NO_THROW(plan.execute(handle, buffers, 3, nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(hostC.data(), deviceC, hostC.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        hipSuccess);

    // bf16 has a 7-bit mantissa (~2 decimal digits) -> looser tolerance than f16.
    for(size_t i = 0; i < M; ++i)
    {
        for(size_t j = 0; j < N; ++j)
        {
            const float got = bf16ToFloat(hostC[i * N + j]);
            const float want = reference[i * N + j];
            const float tol = std::max(5e-2f, 4e-2f * std::fabs(want));
            ASSERT_NEAR(got, want, tol) << "mismatch at (" << i << "," << j << ")";
        }
    }

    (void)hipFree(deviceA);
    (void)hipFree(deviceB);
    (void)hipFree(deviceC);
    (void)hipStreamDestroy(stream);
}
