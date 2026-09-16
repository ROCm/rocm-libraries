// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for ROCM-31016 (https://github.com/ROCm/hipBLASLt/issues/2299):
// bf16 hipblasLtMatmul silently drops stores to D once the output byte offset
// crosses the buffer-store SRD's fixed num_records ceiling on gfx950/gfx942.
//
// Root cause: allocPostLoopSrd() (Tensile/KernelWriterAssembly.py) programs the
// post-loop store SRD for C/D with num_records fixed to the BufferOOB sentinel
// (0xfffff000, ~4 GiB - 4 KiB). That field is a 32-bit cap on every store a
// BufferStore=True kernel issues, independent of how far the SRD base itself
// is re-based per workgroup; once D's true byte extent reaches the sentinel,
// stores at or past it are silently dropped by the hardware buffer-store
// instruction rather than faulting, so the caller gets a partially-written D
// with no error.
//
// Fix (this PR): repair the existing BufferStoreOffsetLimitCheck problem
// predicate (ContractionProblemPredicates.hpp) so it checks the D tensor's
// real worst-case byte extent (strides()[1] * sizes()[1] * elementBytes())
// against BufferOOB, instead of the old formula's std::min(value, sizes()[1])
// cap -- "value" is MacroTile1, so the old check only ever validated one
// output tile's worth of offset and silently accepted any problem whose full
// extent exceeded BufferOOB while a single tile's did not. CompoundPredicates
// (Tensile/Contractions.py) already auto-attaches this predicate to every
// solution whose kernel has BufferStore=True, so the fix is confined to the
// predicate's evaluation formula; no library-export or dispatch-layer changes
// are needed.
//
// This test exercises the real dispatch path end to end
// (hipblasLtMatmulAlgoGetHeuristic + hipblasLtMatmul), not just the predicate
// in isolation (see tensilelite/tests/Predicates_test.cpp for the direct
// unit-test coverage of the formula itself), and uses the exact shape family
// (bf16 NN, large M, N=256, K=48) that this PR's reproduction session
// confirmed silently drops stores on real gfx950 (MI350X) hardware pre-fix --
// see the PR description's Test Result section for the getAllAlgos sweep that
// found dozens of MT16x16x*-tile BufferStore=True kernels leaving ~99.9996% of
// a 16 GiB D unwritten at this shape.
//
// Two outcomes are accepted for the oversized-D case, per the guard being a
// solution-*selection* change rather than a new safe-kernel family:
//   (a) the heuristic finds zero algorithms (HIPBLAS_STATUS_SUCCESS status,
//       foundAlgoCount == 0) -- confirmed to be what happens on gfx950 today
//       for the narrow logic subsets this environment could scope a device
//       build to, since every BufferStore=True solution examined there has
//       nothing to fall back to; or
//   (b) an algorithm is found and D is bit-exact correct (a BufferStore=False
//       solution, or a future BufferStore=True one the guard has no reason to
//       reject, was selected).
// The one unacceptable outcome -- an algorithm is found and reported success,
// but D is wrong or partially unwritten -- is exactly the silent-corruption
// bug this predicate exists to prevent, and is what FAILs pre-fix: this is
// the outcome the PR's reproduction session hit repeatedly against an
// unpatched hipBLASLt build at this shape on real gfx950 hardware.
//
// gfx942 also has the affected allocPostLoopSrd code path, but this test was
// only validated against gfx950 hardware in the reproduction session backing
// this PR (see the PR description's Test Result section), so it is scoped to
// gfx950 rather than claimed for gfx942 too.

#include <gtest/gtest.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace
{
    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // gcnArchName carries feature suffixes (e.g. "gfx950:sramecc+:xnack-") that
    // a plain "==" against "gfx950" would miss.
    std::string gpuArchFamily()
    {
        hipDeviceProp_t props{};
        int             device = 0;
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&props, device) != hipSuccess)
            return "unknown";
        std::string arch = props.gcnArchName;
        return arch.substr(0, arch.find(':'));
    }

    __global__ void fillConstant(__hip_bfloat16* p, size_t n, float v)
    {
        for(size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
            i += (size_t)gridDim.x * blockDim.x)
            p[i] = __float2bfloat16(v);
    }

    // Counts D elements that are still NaN (never written by the kernel) and,
    // separately, elements that were written but do not equal the expected
    // bit-exact value (a wrong, rather than silently-dropped, store).
    __global__ void scanAgainstExpected(const __hip_bfloat16* d,
                                         size_t                n,
                                         float                 expected,
                                         unsigned long long*   unwrittenCount,
                                         unsigned long long*   wrongCount)
    {
        for(size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
            i += (size_t)gridDim.x * blockDim.x)
        {
            float v = __bfloat162float(d[i]);
            if(isnan(v))
                atomicAdd(unwrittenCount, 1ull);
            else if(v != expected)
                atomicAdd(wrongCount, 1ull);
        }
    }

    struct MatmulOutcome
    {
        hipblasStatus_t    heuristicStatus = HIPBLAS_STATUS_INTERNAL_ERROR;
        int                foundAlgoCount  = 0;
        bool               ran             = false;
        // True only if the post-run verification scan itself completed
        // cleanly (kernel launch + both hipMemcpy's succeeded). Callers must
        // check this before trusting unwrittenCount/wrongCount == 0 -- a
        // verification-side failure must not be misread as "nothing wrong".
        bool               verified        = false;
        unsigned long long unwrittenCount  = 0;
        unsigned long long wrongCount      = 0;
        size_t             totalElements   = 0;
    };

    // Runs `cleanups` in reverse (LIFO) order when it goes out of scope, so
    // every early return below still frees exactly the resources that were
    // successfully allocated up to that point -- no leaks on any path.
    struct ScopeGuard
    {
        std::vector<std::function<void()>>& cleanups;
        ~ScopeGuard()
        {
            for(auto it = cleanups.rbegin(); it != cleanups.rend(); ++it)
                (*it)();
        }
    };

    // bf16 NN matmul, A and B filled with 1.0 everywhere, alpha=1, beta=0, so
    // the mathematically-correct D is uniformly K (exactly representable in
    // bf16 as long as K itself is, which every K used below satisfies). D is
    // pre-filled with NaN so an element left NaN after the call proves the
    // kernel never wrote it (the ROCM-31016 signature), distinct from a
    // wrong-but-written value.
    MatmulOutcome runConstantMatmul(int64_t M, int64_t N, int64_t K)
    {
        MatmulOutcome outcome;
        outcome.totalElements = (size_t)M * (size_t)N;

        std::vector<std::function<void()>> cleanups;
        ScopeGuard                         guard{cleanups};

        __hip_bfloat16* A = nullptr;
        if(hipMalloc(&A, (size_t)M * K * sizeof(*A)) != hipSuccess)
            return outcome;
        cleanups.push_back([A] { static_cast<void>(hipFree(A)); });

        __hip_bfloat16* B = nullptr;
        if(hipMalloc(&B, (size_t)K * N * sizeof(*B)) != hipSuccess)
            return outcome;
        cleanups.push_back([B] { static_cast<void>(hipFree(B)); });

        __hip_bfloat16* D = nullptr;
        if(hipMalloc(&D, (size_t)M * N * sizeof(*D)) != hipSuccess)
            return outcome;
        cleanups.push_back([D] { static_cast<void>(hipFree(D)); });

        fillConstant<<<2048, 256>>>(A, (size_t)M * K, 1.0f);
        fillConstant<<<2048, 256>>>(B, (size_t)K * N, 1.0f);
        fillConstant<<<4096, 256>>>(D, (size_t)M * N, std::nanf(""));
        if(hipDeviceSynchronize() != hipSuccess)
            return outcome;

        hipblasLtHandle_t handle;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return outcome;
        cleanups.push_back([handle] { static_cast<void>(hipblasLtDestroy(handle)); });

        hipblasLtMatrixLayout_t la;
        if(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, M, K, M) != HIPBLAS_STATUS_SUCCESS)
            return outcome;
        cleanups.push_back([la] { static_cast<void>(hipblasLtMatrixLayoutDestroy(la)); });

        hipblasLtMatrixLayout_t lb;
        if(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, N, K) != HIPBLAS_STATUS_SUCCESS)
            return outcome;
        cleanups.push_back([lb] { static_cast<void>(hipblasLtMatrixLayoutDestroy(lb)); });

        hipblasLtMatrixLayout_t ld;
        if(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, M, N, M) != HIPBLAS_STATUS_SUCCESS)
            return outcome;
        cleanups.push_back([ld] { static_cast<void>(hipblasLtMatrixLayoutDestroy(ld)); });

        hipblasLtMatmulDesc_t desc;
        if(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
           != HIPBLAS_STATUS_SUCCESS)
            return outcome;
        cleanups.push_back([desc] { static_cast<void>(hipblasLtMatmulDescDestroy(desc)); });

        hipblasOperation_t opN = HIPBLAS_OP_N;
        if(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN))
           != HIPBLAS_STATUS_SUCCESS)
            return outcome;
        if(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
           != HIPBLAS_STATUS_SUCCESS)
            return outcome;

        void*  workspace = nullptr;
        size_t wsize      = 128ull << 20;
        if(hipMalloc(&workspace, wsize) != hipSuccess)
            return outcome;
        cleanups.push_back([workspace] { static_cast<void>(hipFree(workspace)); });

        hipblasLtMatmulPreference_t pref;
        if(hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
            return outcome;
        cleanups.push_back(
            [pref] { static_cast<void>(hipblasLtMatmulPreferenceDestroy(pref)); });
        if(hipblasLtMatmulPreferenceSetAttribute(
               pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsize, sizeof(wsize))
           != HIPBLAS_STATUS_SUCCESS)
            return outcome;

        hipblasLtMatmulHeuristicResult_t heuristic[1];
        int                              found = 0;
        outcome.heuristicStatus                = hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, la, lb, ld, ld, pref, 1, heuristic, &found);
        outcome.foundAlgoCount = found;

        if(found > 0 && outcome.heuristicStatus == HIPBLAS_STATUS_SUCCESS)
        {
            float alpha = 1.0f, beta = 0.0f;
            hipblasStatus_t runStatus
                = hipblasLtMatmul(handle, desc, &alpha, A, la, B, lb, &beta, D, ld, D, ld,
                                  &heuristic[0].algo, workspace, wsize, nullptr);
            outcome.ran = (runStatus == HIPBLAS_STATUS_SUCCESS
                           && hipDeviceSynchronize() == hipSuccess);
        }

        if(outcome.ran)
        {
            unsigned long long *unwrittenCount = nullptr, *wrongCount = nullptr;
            bool                verifyOk       = hipMalloc(&unwrittenCount, sizeof(*unwrittenCount))
                                    == hipSuccess
                             && hipMalloc(&wrongCount, sizeof(*wrongCount)) == hipSuccess;
            if(verifyOk)
                verifyOk = hipMemset(unwrittenCount, 0, sizeof(*unwrittenCount)) == hipSuccess
                           && hipMemset(wrongCount, 0, sizeof(*wrongCount)) == hipSuccess;
            if(verifyOk)
            {
                scanAgainstExpected<<<4096, 256>>>(
                    D, (size_t)M * N, (float)K, unwrittenCount, wrongCount);
                verifyOk = hipMemcpy(&outcome.unwrittenCount, unwrittenCount,
                                      sizeof(outcome.unwrittenCount), hipMemcpyDeviceToHost)
                               == hipSuccess
                           && hipMemcpy(&outcome.wrongCount, wrongCount,
                                        sizeof(outcome.wrongCount), hipMemcpyDeviceToHost)
                                  == hipSuccess;
            }
            outcome.verified = verifyOk;
            if(unwrittenCount)
                static_cast<void>(hipFree(unwrittenCount));
            if(wrongCount)
                static_cast<void>(hipFree(wrongCount));
        }

        return outcome;
    }

#define SKIP_UNLESS_GFX950()                                                       \
    do                                                                             \
    {                                                                              \
        if(!gpuAvailable())                                                        \
            GTEST_SKIP() << "no GPU available";                                    \
        if(gpuArchFamily() != "gfx950")                                            \
            GTEST_SKIP() << "ROCM-31016 was only reproduced/validated on gfx950, " \
                            "not "                                                 \
                         << gpuArchFamily();                                       \
    } while(0)
}

// Sanity check: a D well under the BufferOOB ceiling (~4 GiB - 4 KiB) must
// dispatch to a real algorithm and be bit-exact correct with nothing left
// unwritten. If this ever fails, the case below proves nothing about the
// guard specifically -- it would just be evidence the whole GEMM path is
// broken.
TEST(BufferStoreOffsetGuard_pre_checkin, SafeOutputDispatchesAndIsExact)
{
    SKIP_UNLESS_GFX950();

    auto outcome = runConstantMatmul(/*M=*/1024, /*N=*/1024, /*K=*/1024);
    ASSERT_GT(outcome.foundAlgoCount, 0) << "no algorithm found for an ordinary, "
                                            "well within-range problem";
    ASSERT_TRUE(outcome.ran);
    ASSERT_TRUE(outcome.verified) << "post-run verification scan itself failed";
    EXPECT_EQ(outcome.unwrittenCount, 0u);
    EXPECT_EQ(outcome.wrongCount, 0u);
}

// The ROCM-31016 reproducer: bf16 NN, M large enough (with N=256, K=48) that
// D's true byte extent (M * N * 2 bytes) is 16 GiB, well past the 0xfffff000
// BufferOOB ceiling. This is the exact shape family (M % 256 != 0, N=256,
// K=48) this PR's reproduction session used to confirm the silent-store-drop
// defect on real gfx950 hardware pre-fix.
TEST(BufferStoreOffsetGuard_pre_checkin, OversizedOutputGuardedAgainstSilentCorruption_ROCM31016)
{
    SKIP_UNLESS_GFX950();

    constexpr int64_t M = 33554560, N = 256, K = 48;
    static_assert(M % 256 != 0, "matches the shape family from the PR's repro session");
    static_assert((uint64_t)M * N * 2 > 0xfffff000ull,
                  "D's true byte extent must exceed the BufferOOB sentinel");

    auto outcome = runConstantMatmul(M, N, K);

    if(outcome.foundAlgoCount == 0)
    {
        // Outcome (a): the guard correctly finds no safe solution. Confirmed a
        // clean, non-silent failure: HIPBLAS_STATUS_SUCCESS status, 0
        // algorithms returned, not a crash and not a wrong answer.
        EXPECT_EQ(outcome.heuristicStatus, HIPBLAS_STATUS_SUCCESS);
        SUCCEED() << "heuristic correctly reported no supported algorithm for "
                     "an output past the BufferOOB ceiling";
        return;
    }

    // Outcome (b): an algorithm was found (a BufferStore=False solution, or a
    // future tuning addition the guard has no reason to reject). It must be
    // fully and correctly written -- this is the only alternative to (a) that
    // is not the ROCM-31016 corruption bug itself.
    ASSERT_TRUE(outcome.ran) << "an algorithm was found but failed to run";
    ASSERT_TRUE(outcome.verified) << "post-run verification scan itself failed";
    EXPECT_EQ(outcome.unwrittenCount, 0u)
        << outcome.unwrittenCount << " of " << outcome.totalElements
        << " D elements were never written -- this IS the ROCM-31016 silent "
           "store-drop bug (an unsafe BufferStore=True solution was dispatched "
           "against an output past the SRD's num_records ceiling)";
    EXPECT_EQ(outcome.wrongCount, 0u);
}
