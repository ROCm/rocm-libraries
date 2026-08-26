// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Numeric tests for the uniform summation order mode
// (HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT).
//
// Every case builds a column-major fp32 NN GEMM whose A holds one identical
// K-vector in every row, so D is mathematically row-invariant and any bitwise
// difference between two rows of D is purely an artifact of the reduction order
// the kernel used. Rows are compared bit for bit, never with a tolerance, and a
// case only claims the guarantee for an algorithm shown, in that same run, to
// produce differing rows with the mode off; otherwise it skips with a
// diagnostic rather than reporting a green result it did not earn.
//
// The suite name carries the "pre_checkin" token on purpose:
// clients/tests/test_categories.yaml selects by loose substring on the category
// token the YAML data layer prepends to parameterized test names, and a plain
// gtest suite has no such token, so it would be invisible to every ctest preset.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

// hipblaslt-test links roc::tensilelite-host (through hipblaslt-clients-common),
// so the solution metadata the launch gate reasons from is available here even
// though libhipblaslt.so exports none of it. caching_library_gtest.cpp relies on
// the same property; see its header comment for why white-box coverage lives in
// this binary rather than in the tensilelite-tests suite, which CI never builds.
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace
{
    constexpr size_t kWorkspaceBytes = 256ull * 1024 * 1024;

    // Without a cap the candidate loop runs for minutes without adding
    // coverage. Split so no single source of candidates can consume the whole
    // budget, and weighted towards the sweep because only a minority of the
    // candidates that witness the bug are repaired by the StaggerU clamp rather
    // than refused by the launch gate.
    constexpr size_t kMaxHeuristicCandidates = 4;
    constexpr size_t kMaxSweepCandidates     = 30;

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    std::string gpuArchName()
    {
        hipDeviceProp_t props{};
        int             device = 0;
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&props, device) != hipSuccess)
            return "unknown";
        return props.gcnArchName;
    }

    using ContractionLibrary
        = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    // The two fields the launch gate reasons from when it decides whether the
    // StaggerU clamp can reach a kernel.
    struct SolutionStagger
    {
        bool   clampCanReach = false;
        size_t staggerU      = 0;
    };

    // rocblaslt_find_library_relative_path has hidden visibility, so its probe
    // order for the Tensile library root is mirrored rather than called.
    std::vector<std::filesystem::path> libraryRootCandidates()
    {
        std::vector<std::filesystem::path> roots;

        if(const char* env = std::getenv("HIPBLASLT_TENSILE_LIBPATH"))
            roots.emplace_back(env);

        auto appendRootsFromBinary = [&](const std::filesystem::path& binary) {
            std::error_code       ec;
            std::filesystem::path resolved = std::filesystem::weakly_canonical(binary, ec);
            if(ec)
                resolved = binary;
            const std::filesystem::path libDir = resolved.parent_path();
            roots.push_back(libDir / "hipblaslt" / "library");
            roots.push_back(libDir.parent_path() / "Tensile" / "library");
            roots.push_back(libDir / "library");
        };

#ifdef _WIN32
        // hipblaslt-test links the DLL, so the loaded module is the runtime
        // probe root. Taking &hipblasLtCreate would hit this binary's import
        // thunk instead of libhipblaslt.dll.
        HMODULE hModule = GetModuleHandleA("libhipblaslt.dll");
        if(hModule == nullptr)
            hModule = GetModuleHandleA("hipblaslt.dll");
        if(hModule != nullptr)
        {
            std::string raw(MAX_PATH, '\0');
            for(;;)
            {
                DWORD n = GetModuleFileNameA(
                    hModule, raw.data(), static_cast<DWORD>(raw.size()));
                if(n == 0)
                    break;
                if(n < raw.size())
                {
                    raw.resize(n);
                    appendRootsFromBinary(raw);
                    break;
                }
                raw.assign(raw.size() * 2, '\0');
            }
        }
#else
        // Taking &hipblasLtCreate here would yield this binary's PLT stub, which
        // dladdr resolves to the test executable rather than to the library.
        // dlsym returns the definition itself, so its directory is the one the
        // runtime probes from.
        Dl_info info{};
        void*   entry = dlsym(RTLD_DEFAULT, "hipblasLtCreate");
        if(entry != nullptr && dladdr(entry, &info) != 0 && info.dli_fname != nullptr)
            appendRootsFromBinary(info.dli_fname);
#endif

        return roots;
    }

    // The lazy master library the runtime loads, opened here for its metadata
    // alone: getSolutionByIndex pulls in a placeholder's .dat on demand and
    // never touches the code objects. Empty return leaves reason populated with
    // every path tried.
    std::shared_ptr<ContractionLibrary> masterLibrary(std::string& reason)
    {
        static std::string                         failure;
        static std::shared_ptr<ContractionLibrary> library = [] {
            // gcnArchName carries feature suffixes ("gfx942:sramecc+:xnack-")
            // that never appear in the library filename.
            const std::string arch      = gpuArchName();
            const std::string processor = arch.substr(0, arch.find(':'));

            std::string tried;
            for(const auto& root : libraryRootCandidates())
            {
                // Always the logical single-extension name: the loader resolves
                // the shipped ".dat.zlib" by appending the suffix itself.
                const std::filesystem::path logical
                    = root / processor / ("TensileLibrary_lazy_" + processor + ".dat");
                tried += (tried.empty() ? "" : ", ") + logical.string();

                if(!std::filesystem::exists(logical)
                   && !std::filesystem::exists(logical.string() + ".zlib"))
                    continue;

                auto loaded
                    = TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(
                        logical.string());
                auto master = std::dynamic_pointer_cast<ContractionLibrary>(loaded);
                if(master && master->initLibraryMapping(logical.string()))
                    return master;
            }

            failure = "no Tensile library found for " + processor + "; tried " + tried;
            return std::shared_ptr<ContractionLibrary>();
        }();

        reason = failure;
        return library;
    }

    bool staggerMetadata(const hipblasLtMatmulHeuristicResult_t& candidate,
                         SolutionStagger&                        out,
                         std::string&                            reason)
    {
        auto library = masterLibrary(reason);
        if(!library)
            return false;

        static std::shared_ptr<TensileLite::Hardware> hardware
            = TensileLite::hip::GetCurrentDevice();
        if(!hardware)
        {
            reason = "could not describe the current device to TensileLite";
            return false;
        }

        hipblasLtMatmulAlgo_t algo  = candidate.algo;
        const int             index = hipblaslt_ext::getIndexFromAlgo(algo);

        auto solution = library->getSolutionByIndex(*hardware, index);
        if(!solution)
        {
            reason = "solution index " + std::to_string(index)
                     + " is enumerable but absent from the loaded library";
            return false;
        }

        out.clampCanReach = solution->internalArgsSupport.staggerU;
        out.staggerU      = solution->sizeMapping.staggerU;
        return true;
    }

    // Wide dynamic range, mixed sign, non-dyadic. The rand_int fill hipBLASLt
    // uses by default produces fp32 values in [-2,2], which sum exactly, so
    // every summation order would agree and these tests would pass against a
    // completely broken implementation. Magnitudes land in [2^-12, 2^13): with
    // K up to 12288 the largest possible accumulator stays far below FLT_MAX and
    // the smallest product far above the smallest normal, so neither overflow
    // nor denormal flush can occur.
    float wideRangeSample(std::mt19937& rng)
    {
        std::uniform_int_distribution<int>    exponentOf{-12, 12};
        std::uniform_real_distribution<float> mantissaOf{0.0f, 1.0f};
        std::uniform_int_distribution<int>    signOf{0, 1};

        const int   exponent = exponentOf(rng);
        const float mantissa = 1.0f + mantissaOf(rng);
        const float sign     = signOf(rng) ? 1.0f : -1.0f;
        return sign * std::ldexp(mantissa, exponent);
    }

    struct Problem
    {
        int64_t m;
        int64_t n;
        int64_t k;
    };

    // Owns every device and host resource for one problem size and can replay
    // the same GEMM with an arbitrary algorithm and mode setting.
    class RowUniformityHarness
    {
    public:
        explicit RowUniformityHarness(const Problem& problem)
            : m_problem(problem)
        {
        }

        ~RowUniformityHarness()
        {
            if(m_desc)
                hipblasLtMatmulDescDestroy(m_desc);
            if(m_layoutD)
                hipblasLtMatrixLayoutDestroy(m_layoutD);
            if(m_layoutB)
                hipblasLtMatrixLayoutDestroy(m_layoutB);
            if(m_layoutA)
                hipblasLtMatrixLayoutDestroy(m_layoutA);
            static_cast<void>(hipFree(m_deviceWorkspace));
            static_cast<void>(hipFree(m_deviceD));
            static_cast<void>(hipFree(m_deviceB));
            static_cast<void>(hipFree(m_deviceA));
            if(m_stream)
                static_cast<void>(hipStreamDestroy(m_stream));
            if(m_handle)
                hipblasLtDestroy(m_handle);
        }

        RowUniformityHarness(const RowUniformityHarness&)            = delete;
        RowUniformityHarness& operator=(const RowUniformityHarness&) = delete;

        // Returns false with a populated skipReason when the environment cannot
        // host the problem; other failures are reported through gtest directly
        // and leave skipReason empty.
        bool setUp(std::string& skipReason)
        {
            const int64_t m = m_problem.m;
            const int64_t n = m_problem.n;
            const int64_t k = m_problem.k;

            if(hipblasLtCreate(&m_handle) != HIPBLAS_STATUS_SUCCESS
               || hipStreamCreate(&m_stream) != hipSuccess)
            {
                skipReason = "hipblasLt handle or stream creation failed";
                return false;
            }

            if(hipMalloc(&m_deviceA, static_cast<size_t>(m * k) * sizeof(float)) != hipSuccess
               || hipMalloc(&m_deviceB, static_cast<size_t>(k * n) * sizeof(float)) != hipSuccess
               || hipMalloc(&m_deviceD, static_cast<size_t>(m * n) * sizeof(float)) != hipSuccess
               || hipMalloc(&m_deviceWorkspace, kWorkspaceBytes) != hipSuccess)
            {
                skipReason = "hipMalloc failed: not enough device memory for " + std::to_string(m)
                             + "x" + std::to_string(n) + "x" + std::to_string(k)
                             + " plus workspace";
                return false;
            }

            if(!uploadOperands(skipReason))
                return false;

            m_hostD.resize(static_cast<size_t>(m * n));

            if(hipblasLtMatrixLayoutCreate(&m_layoutA, HIP_R_32F, m, k, m) != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatrixLayoutCreate(&m_layoutB, HIP_R_32F, k, n, k)
                      != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatrixLayoutCreate(&m_layoutD, HIP_R_32F, m, n, m)
                      != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatmulDescCreate(&m_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                      != HIPBLAS_STATUS_SUCCESS)
            {
                ADD_FAILURE() << "hipblasLt layout or matmul descriptor creation failed";
                return false;
            }

            const int32_t opN = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(
                m_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
            hipblasLtMatmulDescSetAttribute(
                m_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

            return true;
        }

        // The fill only means something if fp32 summation of this data actually
        // depends on the order; otherwise every assertion below is vacuous.
        bool referenceOrderMatters() const
        {
            float forward = 0.0f;
            for(int64_t idx = 0; idx < m_problem.k; ++idx)
                forward += m_aVector[static_cast<size_t>(idx)] * m_hostB[static_cast<size_t>(idx)];

            float reverse = 0.0f;
            for(int64_t idx = m_problem.k - 1; idx >= 0; --idx)
                reverse += m_aVector[static_cast<size_t>(idx)] * m_hostB[static_cast<size_t>(idx)];

            return std::memcmp(&forward, &reverse, sizeof(float)) != 0;
        }

        std::vector<hipblasLtMatmulHeuristicResult_t> candidateAlgos(int& enumeratedCount)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> supported;

            hipblasLtMatmulPreference_t preference     = nullptr;
            uint64_t                    workspaceBytes = kWorkspaceBytes;
            hipblasLtMatmulPreferenceCreate(&preference);
            hipblasLtMatmulPreferenceSetAttribute(preference,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspaceBytes,
                                                  sizeof(workspaceBytes));

            // The default heuristic picks go first: the regression shapes were
            // measured as non-uniform on exactly that path.
            hipblasLtMatmulHeuristicResult_t heuristic[4]{};
            int                              heuristicCount = 0;
            hipblasLtMatmulAlgoGetHeuristic(m_handle,
                                            m_desc,
                                            m_layoutA,
                                            m_layoutB,
                                            m_layoutD,
                                            m_layoutD,
                                            preference,
                                            4,
                                            heuristic,
                                            &heuristicCount);
            hipblasLtMatmulPreferenceDestroy(preference);

            const std::vector<hipblasLtMatmulHeuristicResult_t> defaults(
                heuristic, heuristic + heuristicCount);
            appendSupported(defaults, 0, 1, kMaxHeuristicCandidates, supported);

            std::vector<hipblasLtMatmulHeuristicResult_t> enumerated;
            hipblaslt_ext::getAllAlgos(m_handle,
                                       hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_N,
                                       HIP_R_32F,
                                       HIP_R_32F,
                                       HIP_R_32F,
                                       HIP_R_32F,
                                       HIPBLAS_COMPUTE_32F,
                                       enumerated);
            enumeratedCount = static_cast<int>(enumerated.size());

            // getAllAlgos walks a std::set of shared_ptr, so the order follows
            // heap addresses and varies from process to process. Sorting on the
            // solution index -- a value baked into the library -- is what makes
            // the sweep below pick the same candidates on every run.
            std::sort(
                enumerated.begin(),
                enumerated.end(),
                [](hipblasLtMatmulHeuristicResult_t lhs, hipblasLtMatmulHeuristicResult_t rhs) {
                    return hipblaslt_ext::getIndexFromAlgo(lhs.algo)
                           < hipblaslt_ext::getIndexFromAlgo(rhs.algo);
                });

            // Neighbouring entries are near-identical kernels that behave the
            // same way, so the list is swept twice -- once from the front, once
            // with a coarse stride -- each with its own share of the budget:
            // whether a shape has a non-uniform algorithm at all turned out to
            // depend on which part of the list the candidates came from.
            const size_t stride = std::max<size_t>(1, enumerated.size() / 256);
            appendSupported(enumerated, 0, 1, kMaxSweepCandidates, supported);
            appendSupported(enumerated, stride, stride, kMaxSweepCandidates, supported);

            return supported;
        }

        std::string solutionName(const hipblasLtMatmulHeuristicResult_t& candidate)
        {
            hipblasLtMatmulAlgo_t algo = candidate.algo;
            return hipblaslt_ext::getSolutionNameFromAlgo(m_handle, algo);
        }

        // Runs the GEMM with the mode either off or on and, on success, leaves
        // the result in the host copy inspected by firstNonUniformRowOfLastRun.
        hipblasStatus_t run(const hipblasLtMatmulHeuristicResult_t& candidate, bool uniformMode)
        {
            const int32_t mode      = uniformMode ? 1 : 0;
            const auto    setStatus = hipblasLtMatmulDescSetAttribute(
                m_desc, HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT, &mode, sizeof(mode));
            // Distinguished from a matmul-level refusal, which the caller is
            // allowed to accept: setting a legal mode value must always work.
            EXPECT_EQ(setStatus, HIPBLAS_STATUS_SUCCESS)
                << "Setting UNIFORM_SUMMATION_ORDER_EXT to " << mode << " must succeed";
            if(setStatus != HIPBLAS_STATUS_SUCCESS)
                return setStatus;

            static_cast<void>(
                hipMemsetAsync(m_deviceD, 0, m_hostD.size() * sizeof(float), m_stream));

            const float           alpha = 1.0f;
            const float           beta  = 0.0f;
            hipblasLtMatmulAlgo_t algo  = candidate.algo;

            const auto status = hipblasLtMatmul(m_handle,
                                                m_desc,
                                                &alpha,
                                                m_deviceA,
                                                m_layoutA,
                                                m_deviceB,
                                                m_layoutB,
                                                &beta,
                                                m_deviceD,
                                                m_layoutD,
                                                m_deviceD,
                                                m_layoutD,
                                                &algo,
                                                m_deviceWorkspace,
                                                candidate.workspaceSize,
                                                m_stream);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;

            if(hipStreamSynchronize(m_stream) != hipSuccess)
                return HIPBLAS_STATUS_EXECUTION_FAILED;

            if(hipMemcpy(
                   m_hostD.data(), m_deviceD, m_hostD.size() * sizeof(float), hipMemcpyDeviceToHost)
               != hipSuccess)
                return HIPBLAS_STATUS_EXECUTION_FAILED;

            return HIPBLAS_STATUS_SUCCESS;
        }

        // Index of the first row of the last run that differs bitwise from row
        // 0, or -1 when every row is identical. A row of a column-major D is
        // strided, but "every row equals row 0" is the same statement as "every
        // column is constant", and a column is contiguous, so one memcmp of a
        // column against itself shifted by a single element settles the whole
        // column at memcmp speed.
        int64_t firstNonUniformRowOfLastRun() const
        {
            const int64_t m = m_problem.m;
            for(int64_t col = 0; col < m_problem.n; ++col)
            {
                const float* base = m_hostD.data() + col * m;
                if(std::memcmp(base, base + 1, static_cast<size_t>(m - 1) * sizeof(float)) == 0)
                    continue;

                uint32_t row0 = 0;
                std::memcpy(&row0, base, sizeof(row0));
                for(int64_t row = 1; row < m; ++row)
                {
                    uint32_t bits = 0;
                    std::memcpy(&bits, base + row, sizeof(bits));
                    if(bits != row0)
                        return row;
                }
            }
            return -1;
        }

    private:
        // Records the workspace this algorithm needs, or says why the library
        // cannot run it on this problem.
        bool runnable(hipblasLtMatmulHeuristicResult_t& candidate, std::string& reason)
        {
            const float alpha    = 1.0f;
            const float beta     = 0.0f;
            size_t      required = 0;
            if(hipblaslt_ext::matmulIsAlgoSupported(m_handle,
                                                    m_desc,
                                                    &alpha,
                                                    m_layoutA,
                                                    m_layoutB,
                                                    &beta,
                                                    m_layoutD,
                                                    m_layoutD,
                                                    candidate.algo,
                                                    required)
               != HIPBLAS_STATUS_SUCCESS)
            {
                reason = "does not support this problem";
                return false;
            }
            if(required > kWorkspaceBytes)
            {
                reason = "needs " + std::to_string(required) + " workspace bytes, more than the "
                         + std::to_string(kWorkspaceBytes) + " this test allocates";
                return false;
            }

            candidate.workspaceSize = required;
            return true;
        }

        // Walks pool from first in steps of stride and moves at most budget
        // runnable entries into out.
        void appendSupported(const std::vector<hipblasLtMatmulHeuristicResult_t>& pool,
                             size_t                                               first,
                             size_t                                               stride,
                             size_t                                               budget,
                             std::vector<hipblasLtMatmulHeuristicResult_t>&       out)
        {
            std::string unused;
            size_t      taken = 0;
            for(size_t idx = first; idx < pool.size() && taken < budget; idx += stride)
            {
                hipblasLtMatmulHeuristicResult_t result = pool[idx];
                if(!runnable(result, unused))
                    continue;

                out.push_back(result);
                ++taken;
            }
        }

        bool uploadOperands(std::string& skipReason)
        {
            const int64_t m = m_problem.m;
            const int64_t n = m_problem.n;
            const int64_t k = m_problem.k;

            std::mt19937 rng(0x52755f31u ^ static_cast<uint32_t>(m * 31 + n * 17 + k));

            m_aVector.resize(static_cast<size_t>(k));
            for(auto& value : m_aVector)
                value = wideRangeSample(rng);

            m_hostB.resize(static_cast<size_t>(k * n));
            for(auto& value : m_hostB)
                value = wideRangeSample(rng);

            if(hipMemcpy(
                   m_deviceB, m_hostB.data(), m_hostB.size() * sizeof(float), hipMemcpyHostToDevice)
               != hipSuccess)
            {
                skipReason = "hipMemcpy of B failed";
                return false;
            }

            // Column k of a column-major A is M copies of a[k], so A goes up one
            // column at a time and never needs an M*K host buffer.
            std::vector<float> column(static_cast<size_t>(m));
            for(int64_t idx = 0; idx < k; ++idx)
            {
                std::fill(column.begin(), column.end(), m_aVector[static_cast<size_t>(idx)]);
                if(hipMemcpy(static_cast<float*>(m_deviceA) + idx * m,
                             column.data(),
                             column.size() * sizeof(float),
                             hipMemcpyHostToDevice)
                   != hipSuccess)
                {
                    skipReason = "hipMemcpy of A failed";
                    return false;
                }
            }
            return true;
        }

        Problem                 m_problem;
        hipblasLtHandle_t       m_handle          = nullptr;
        hipStream_t             m_stream          = nullptr;
        hipblasLtMatrixLayout_t m_layoutA         = nullptr;
        hipblasLtMatrixLayout_t m_layoutB         = nullptr;
        hipblasLtMatrixLayout_t m_layoutD         = nullptr;
        hipblasLtMatmulDesc_t   m_desc            = nullptr;
        void*                   m_deviceA         = nullptr;
        void*                   m_deviceB         = nullptr;
        void*                   m_deviceD         = nullptr;
        void*                   m_deviceWorkspace = nullptr;
        std::vector<float>      m_aVector;
        std::vector<float>      m_hostB;
        std::vector<float>      m_hostD;
    };

    // A fixture rather than bare TEST() only so the shared drivers can reach
    // RecordProperty, which gtest exposes to Test subclasses alone.
    class RowUniformity_pre_checkin : public ::testing::Test
    {
    protected:
        // Brings the harness up, or skips (no GPU, no memory) or fails (setUp
        // error, degenerate fill). Callers must stop on IsSkipped/HasFailure:
        // a skip or an ADD_FAILURE inside a helper does not return for them,
        // and the harness is not usable afterwards.
        void prepare(RowUniformityHarness& harness)
        {
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";

            std::string skipReason;
            if(!harness.setUp(skipReason))
            {
                if(skipReason.empty())
                    return;
                GTEST_SKIP() << skipReason;
            }

            ASSERT_TRUE(harness.referenceOrderMatters())
                << "Forward and reverse fp32 reference dot products are bitwise equal, so the "
                   "fill is degenerate and this test cannot detect a summation-order change";
        }

        void checkRowUniformity(const Problem& problem)
        {
            RowUniformityHarness harness(problem);
            prepare(harness);
            if(IsSkipped() || HasFailure())
                return;

            int        enumeratedCount = 0;
            const auto candidates      = harness.candidateAlgos(enumeratedCount);

            int witnesses = 0;
            int honored   = 0;
            int rejected  = 0;

            for(const auto& candidate : candidates)
            {
                if(harness.run(candidate, /*uniformMode=*/false) != HIPBLAS_STATUS_SUCCESS)
                    continue;
                if(harness.firstNonUniformRowOfLastRun() < 0)
                    continue;

                ++witnesses;

                const auto status = harness.run(candidate, /*uniformMode=*/true);
                if(status == HIPBLAS_STATUS_INVALID_VALUE)
                {
                    // The mode may refuse a configuration it cannot make
                    // uniform, as long as it refuses cleanly.
                    ++rejected;
                    continue;
                }

                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS)
                    << "Uniform summation order must either honor the request or reject it with "
                       "HIPBLAS_STATUS_INVALID_VALUE";

                const int64_t badRow = harness.firstNonUniformRowOfLastRun();
                EXPECT_EQ(badRow, -1)
                    << "Row " << badRow
                    << " of D differs bitwise from row 0 with uniform summation order enabled";
                if(badRow < 0)
                    ++honored;
            }

            RecordProperty("algos_enumerated", enumeratedCount);
            RecordProperty("candidates_tried", static_cast<int>(candidates.size()));
            RecordProperty("witnesses", witnesses);
            RecordProperty("honored", honored);
            RecordProperty("rejected", rejected);

            if(witnesses == 0)
                GTEST_SKIP() << "No algorithm produced non-uniform rows with the mode off, so "
                                "this run cannot witness the guarantee. arch="
                             << gpuArchName() << " problem=" << problem.m << "x" << problem.n << "x"
                             << problem.k << " algos_enumerated=" << enumeratedCount
                             << " candidates_tried=" << candidates.size();
        }

        // checkRowUniformity accepts a clean refusal for every candidate, so it
        // can report green having exercised only the launch gate's rejection
        // path and never the repair the mode exists to perform. This driver
        // selects candidates by the same metadata the gate reasons from -- a
        // kernel that staggers, whose StaggerU the clamp can reach -- and then
        // asserts the outcome. Selecting on metadata rather than on the observed
        // result is what stops the assertion being circular, and it replaces an
        // earlier pair of pinned solution indices that could not survive either
        // a retune or a change of architecture.
        void checkClampRepairedSolutions(const Problem& problem)
        {
            RowUniformityHarness harness(problem);
            prepare(harness);
            if(IsSkipped() || HasFailure())
                return;

            std::string libraryReason;
            if(!masterLibrary(libraryReason))
                GTEST_SKIP() << libraryReason;

            int        enumeratedCount = 0;
            const auto candidates      = harness.candidateAlgos(enumeratedCount);

            int         staggering  = 0;
            int         reachable   = 0;
            int         unreachable = 0;
            int         honored     = 0;
            int         repaired    = 0;
            int         unresolved  = 0;
            std::string lastUnresolved;

            for(const auto& candidate : candidates)
            {
                SolutionStagger stagger;
                std::string     reason;
                if(!staggerMetadata(candidate, stagger, reason))
                {
                    // Not a failure on its own: the runtime may have resolved a
                    // different library than the probe above found, and an index
                    // it never loaded says nothing about the mode. The tally is
                    // what distinguishes that from an odd solution or two.
                    ++unresolved;
                    lastUnresolved = reason;
                    continue;
                }

                if(stagger.staggerU == 0)
                    continue;

                ++staggering;
                const std::string name = harness.solutionName(candidate);

                if(!stagger.clampCanReach)
                {
                    // The clamp writes a field this kernel never reads, so the
                    // gate has to refuse it: admitting it would leave the
                    // compiled-in rotation running.
                    ++unreachable;
                    EXPECT_EQ(harness.run(candidate, /*uniformMode=*/true),
                              HIPBLAS_STATUS_INVALID_VALUE)
                        << name << " declares StaggerU " << stagger.staggerU
                        << " with no runtime StaggerU argument, so uniform summation order must "
                           "refuse it rather than run it unclamped";
                    continue;
                }

                ++reachable;

                if(harness.run(candidate, /*uniformMode=*/false) != HIPBLAS_STATUS_SUCCESS)
                    continue;

                // Staggering metadata does not guarantee this shape actually
                // reduces in a differing order, and a solution already uniform
                // with the mode off would satisfy the assertion below without
                // the clamp having repaired anything.
                const bool brokenAtBaseline = harness.firstNonUniformRowOfLastRun() >= 0;

                const auto status = harness.run(candidate, /*uniformMode=*/true);
                // Reaching the clamp settles StaggerU, but the gate refuses on
                // other grounds too -- an uneven Stream-K partition, atomic
                // accumulation -- so a clean refusal stays admissible here.
                if(status == HIPBLAS_STATUS_INVALID_VALUE)
                    continue;

                ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS)
                    << "Uniform summation order must either honor " << name
                    << " or reject it with HIPBLAS_STATUS_INVALID_VALUE";

                ++honored;
                if(brokenAtBaseline)
                    ++repaired;

                const int64_t badRow = harness.firstNonUniformRowOfLastRun();
                EXPECT_EQ(badRow, -1)
                    << "Row " << badRow << " of D differs bitwise from row 0 for " << name
                    << ", a staggering solution the clamp reaches, with uniform summation order "
                       "enabled";
            }

            RecordProperty("algos_enumerated", enumeratedCount);
            RecordProperty("candidates_tried", static_cast<int>(candidates.size()));
            RecordProperty("staggering", staggering);
            RecordProperty("clamp_reachable", reachable);
            RecordProperty("clamp_unreachable", unreachable);
            RecordProperty("honored", honored);
            RecordProperty("repaired", repaired);
            RecordProperty("unresolved", unresolved);

            // Every candidate unresolved means the library loaded here is not
            // the one the runtime enumerated from, so no conclusion below is
            // about the shipped solutions.
            ASSERT_LT(static_cast<size_t>(unresolved), candidates.size())
                << "No enumerated solution was found in the library this test loaded, so the "
                   "probe resolved a different library than the runtime uses. Last reason: "
                << lastUnresolved;

            // Skipping rather than passing is the point: with no solution that
            // was non-uniform without the mode and honored with it, nothing
            // above witnessed a repair, and this case must not be mistaken for
            // coverage of one.
            if(repaired == 0)
                GTEST_SKIP() << "No staggering solution the clamp reaches was both non-uniform "
                                "with the mode off and honored with it on, so this run cannot "
                                "witness a repair. arch="
                             << gpuArchName() << " problem=" << problem.m << "x" << problem.n << "x"
                             << problem.k << " algos_enumerated=" << enumeratedCount
                             << " candidates_tried=" << candidates.size()
                             << " staggering=" << staggering << " clamp_reachable=" << reachable
                             << " clamp_unreachable=" << unreachable << " honored=" << honored;
        }
    };

    // Measured non-uniform on gfx950 with the default heuristic path.
    TEST_F(RowUniformity_pre_checkin, Regression_6144x5120x8192)
    {
        checkRowUniformity({6144, 5120, 8192});
    }

    TEST_F(RowUniformity_pre_checkin, Regression_3072x3072x12288)
    {
        checkRowUniformity({3072, 3072, 12288});
    }

    // Uniform at baseline, so these skip unless an enumerated algorithm breaks
    // uniformity; they are controls for the two regression shapes above.
    TEST_F(RowUniformity_pre_checkin, Control_4096x4096x8192)
    {
        checkRowUniformity({4096, 4096, 8192});
    }

    TEST_F(RowUniformity_pre_checkin, Control_8192x8192x8192)
    {
        checkRowUniformity({8192, 8192, 8192});
    }

    // Covers the repair itself rather than the refusal: every enumerated
    // solution whose metadata says it staggers is checked, against the outcome
    // that metadata implies.
    TEST_F(RowUniformity_pre_checkin, ClampRepaired_6144x5120x8192)
    {
        checkClampRepairedSolutions({6144, 5120, 8192});
    }

} // namespace
