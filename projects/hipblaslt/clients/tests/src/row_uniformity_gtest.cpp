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
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

// hipblaslt-test links roc::tensilelite-host (through hipblaslt-clients-common),
// so the solution metadata the launch gate reasons from is available here even
// though libhipblaslt.so exports none of it. caching_library_gtest.cpp relies on
// the same property; see its header comment for why white-box coverage lives in
// this binary rather than in the tensilelite-tests suite, which CI never builds.
#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#include <origami/streamk.hpp>

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

    // The TensileLite view of the current device, shared by every white-box
    // helper below. Null when there is no GPU or it cannot be described.
    std::shared_ptr<TensileLite::Hardware> currentHardware()
    {
        static std::shared_ptr<TensileLite::Hardware> hardware
            = TensileLite::hip::GetCurrentDevice();
        return hardware;
    }

    // The solution record behind an enumerated algorithm, or null with a
    // populated reason. This is metadata only: getSolutionByIndex pulls in a
    // placeholder's .dat on demand and never touches the code objects.
    std::shared_ptr<TensileLite::ContractionSolution> solutionForAlgo(
        const hipblasLtMatmulHeuristicResult_t& candidate, std::string& reason)
    {
        auto library = masterLibrary(reason);
        if(!library)
            return nullptr;

        auto hardware = currentHardware();
        if(!hardware)
        {
            reason = "could not describe the current device to TensileLite";
            return nullptr;
        }

        hipblasLtMatmulAlgo_t algo  = candidate.algo;
        const int             index = hipblaslt_ext::getIndexFromAlgo(algo);

        auto solution = library->getSolutionByIndex(*hardware, index);
        if(!solution)
            reason = "solution index " + std::to_string(index)
                     + " is enumerable but absent from the loaded library";
        return solution;
    }

    bool staggerMetadata(const hipblasLtMatmulHeuristicResult_t& candidate,
                         SolutionStagger&                        out,
                         std::string&                            reason)
    {
        auto solution = solutionForAlgo(candidate, reason);
        if(!solution)
            return false;

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

    // The A/B storage type is a parameter because the only shipped shape known
    // to reach the newly admitted StreamK split regime through a public API
    // attribute is a bf16-in / f32-out entry; every pre-existing case stays on
    // the fp32 default, for which the conversions below are the identity.
    struct Problem
    {
        int64_t     m;
        int64_t     n;
        int64_t     k;
        hipDataType abType = HIP_R_32F;
        // HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET; 0 leaves the attribute unset.
        int32_t smCountTarget = 0;
        // When true, request HIPBLASLT_EPILOGUE_BIAS with a constant f32 bias
        // vector of length m. Needed to select the gfx942 GridBased Bias exact
        // size that reaches the newly admitted Stream-K split (report 09).
        bool useBias = false;
    };

    size_t elementSizeOf(hipDataType type)
    {
        return type == HIP_R_32F ? sizeof(float) : sizeof(uint16_t);
    }

    // Rounds values in place to what the device will actually hold, then
    // returns the byte image to upload. Rounding in place matters: the host
    // reference below must see the same numbers the kernel does, or its
    // order-sensitivity self-check describes a different problem.
    //
    // HIP_R_16BF uses hip_bfloat16. hipBLASLt leaves mxDataGenerator off on
    // Windows, so this harness cannot depend on DGen headers.
    std::vector<uint8_t> encodeOperand(std::vector<float>& values, hipDataType type)
    {
        std::vector<uint8_t> bytes(values.size() * elementSizeOf(type));
        if(type == HIP_R_32F)
        {
            std::memcpy(bytes.data(), values.data(), bytes.size());
            return bytes;
        }

        auto* out = reinterpret_cast<uint16_t*>(bytes.data());
        for(size_t idx = 0; idx < values.size(); ++idx)
        {
            const hip_bfloat16 packed{values[idx]};
            out[idx]     = packed.data;
            values[idx]  = static_cast<float>(packed);
        }
        return bytes;
    }

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
            static_cast<void>(hipFree(m_deviceBias));
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

            const size_t abBytes = elementSizeOf(m_problem.abType);
            if(hipMalloc(&m_deviceA, static_cast<size_t>(m * k) * abBytes) != hipSuccess
               || hipMalloc(&m_deviceB, static_cast<size_t>(k * n) * abBytes) != hipSuccess
               || hipMalloc(&m_deviceD, static_cast<size_t>(m * n) * sizeof(float)) != hipSuccess
               || hipMalloc(&m_deviceWorkspace, kWorkspaceBytes) != hipSuccess)
            {
                skipReason = "hipMalloc failed: not enough device memory for " + std::to_string(m)
                             + "x" + std::to_string(n) + "x" + std::to_string(k)
                             + " plus workspace";
                return false;
            }

            if(m_problem.useBias
               && hipMalloc(&m_deviceBias, static_cast<size_t>(m) * sizeof(float)) != hipSuccess)
            {
                skipReason = "hipMalloc failed for bias vector of length " + std::to_string(m);
                return false;
            }

            if(!uploadOperands(skipReason))
                return false;

            m_hostD.resize(static_cast<size_t>(m * n));

            if(hipblasLtMatrixLayoutCreate(&m_layoutA, m_problem.abType, m, k, m)
                   != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatrixLayoutCreate(&m_layoutB, m_problem.abType, k, n, k)
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

            // The CU budget the StreamK grid selector reasons from. This is the
            // whole lever the clause-2 case below uses to reach its regime, and
            // it participates only in the problem hash and in grid selection,
            // never in a selection predicate.
            if(m_problem.smCountTarget > 0
               && hipblasLtMatmulDescSetAttribute(m_desc,
                                                  HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                  &m_problem.smCountTarget,
                                                  sizeof(m_problem.smCountTarget))
                      != HIPBLAS_STATUS_SUCCESS)
            {
                ADD_FAILURE() << "Setting SM_COUNT_TARGET must succeed";
                return false;
            }

            if(m_problem.useBias)
            {
                // Constant bias across rows: identical A rows + identical bias
                // keeps the row-uniformity assertion meaningful.
                std::vector<float> bias(static_cast<size_t>(m), 0.125f);
                if(hipMemcpy(m_deviceBias,
                             bias.data(),
                             bias.size() * sizeof(float),
                             hipMemcpyHostToDevice)
                   != hipSuccess)
                {
                    skipReason = "hipMemcpy of bias failed";
                    return false;
                }

                const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
                const hipDataType         biasType = HIP_R_32F;
                if(hipblasLtMatmulDescSetAttribute(m_desc,
                                                   HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue,
                                                   sizeof(epilogue))
                       != HIPBLAS_STATUS_SUCCESS
                   || hipblasLtMatmulDescSetAttribute(m_desc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                      &biasType,
                                                      sizeof(biasType))
                          != HIPBLAS_STATUS_SUCCESS
                   || hipblasLtMatmulDescSetAttribute(m_desc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &m_deviceBias,
                                                      sizeof(m_deviceBias))
                          != HIPBLAS_STATUS_SUCCESS)
                {
                    ADD_FAILURE() << "Setting bias epilogue attributes must succeed";
                    return false;
                }
            }

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

        std::vector<hipblasLtMatmulHeuristicResult_t> candidateAlgos(
            int& enumeratedCount, size_t sweepBudget = kMaxSweepCandidates)
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
                                       m_problem.abType,
                                       m_problem.abType,
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
            appendSupported(enumerated, 0, 1, sweepBudget, supported);
            appendSupported(enumerated, stride, stride, sweepBudget, supported);

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

            const std::vector<uint8_t> bBytes = encodeOperand(m_hostB, m_problem.abType);
            if(hipMemcpy(m_deviceB, bBytes.data(), bBytes.size(), hipMemcpyHostToDevice)
               != hipSuccess)
            {
                skipReason = "hipMemcpy of B failed";
                return false;
            }

            // Column k of a column-major A is M copies of a[k], so A goes up one
            // column at a time and never needs an M*K host buffer.
            const size_t         abBytes = elementSizeOf(m_problem.abType);
            std::vector<float>   column(static_cast<size_t>(m));
            std::vector<uint8_t> aColumn;
            for(int64_t idx = 0; idx < k; ++idx)
            {
                std::fill(column.begin(), column.end(), m_aVector[static_cast<size_t>(idx)]);
                aColumn = encodeOperand(column, m_problem.abType);
                // encodeOperand rounds in place, so the reference vector picks
                // up the value the device will hold.
                m_aVector[static_cast<size_t>(idx)] = column[0];
                if(hipMemcpy(static_cast<uint8_t*>(m_deviceA) + static_cast<size_t>(idx * m) * abBytes,
                             aColumn.data(),
                             aColumn.size(),
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
        void*                   m_deviceBias      = nullptr;
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
        // selects candidates by the same metadata the gate reasons from -- any
        // kernel that declares a stagger, whose StaggerU the host-side clamp
        // therefore has to settle -- and then asserts the outcome. Selecting on
        // metadata rather than on the observed result is what stops the
        // assertion being circular, and it replaces an earlier pair of pinned
        // solution indices that could not survive either a retune or a change
        // of architecture.
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

                // SupportCustomStaggerU: False does not mean the kernel found
                // its stagger somewhere the clamp cannot reach. It means only
                // that the host leaves the packed StaggerU field at 0, and a
                // generated kernel reads StaggerU from that field and nowhere
                // else. So these are admitted like any other staggering
                // solution and held to the same bitwise row-uniformity
                // assertion below, which is what would catch a stagger the
                // clamp really had failed to reach. Only frozen hand-written
                // custom kernels can carry one, and the gate refuses those.
                if(stagger.clampCanReach)
                    ++reachable;
                else
                    ++unreachable;

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
                    << ", a staggering solution, with uniform summation order enabled";
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
                GTEST_SKIP() << "No staggering solution was both non-uniform with the mode off "
                                "and honored with it on, so this run cannot "
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

    // =======================================================================
    // StreamK split: the packing mirror and the row-uniformity predicate.
    //
    // Pure integer arithmetic over (tiles, itersPerTile, grid, skFullTiles,
    // forceDPOnly), so these need no GPU and run on every architecture. They
    // call the same two functions generateSingleCall() packs from and
    // checkUniformSummationOrder() decides from, so a drift between the gate
    // and the packer would fail here rather than silently weakening the
    // guarantee.
    // =======================================================================

    struct SplitCase
    {
        const char* name;
        size_t      tiles;
        size_t      itersPerTile;
        size_t      grid;
        int         skFullTiles;
        bool        forceDPOnly;
        uint32_t    skTiles;
        uint32_t    skItersPerWG;
        uint32_t    extraIters;
        bool        rowUniform;
        bool        perTileExtraIters = false;
    };

    const SplitCase kSplitCases[] = {
        // Newly admitted: the grid is an exact multiple of the tile count and
        // the multiplier divides the iterations per tile, so every tile is cut
        // into equal chunks at identical offsets.
        {"MultipleGridHalfTile", 100, 128, 200, 1, false, 100, 64, 0, true},
        {"MultipleGridQuarterTile", 16, 32, 64, 1, false, 16, 8, 0, true},

        // GridEqualsTiles: every WG writes one whole SK tile, no DP region.
        // GridDividesTiles is mixed two-tile DP+SK (skTiles == grid < tiles).
        // gfx950 does not store the SK half when workspace is skipped
        // (tiles % grid == 0), so the predicate refuses it.
        {"GridEqualsTiles", 256, 64, 256, 1, false, 256, 64, 0, true},
        {"GridDividesTiles", 512, 64, 256, 1, false, 256, 64, 0, false},

        // extraIters == 0 and skTiles == tiles, but the chunk length does not
        // divide the tile length, so some tiles are split and others are not.
        {"ChunkStraddlesTiles", 288, 64, 192, 1, false, 288, 96, 0, false},
        {"ChunkLongerThanTile", 300, 64, 256, 1, false, 300, 75, 0, false},

        // Chunks come in two lengths under the global first-E mapping.
        {"RaggedChunks", 100, 100, 300, 1, false, 100, 33, 100, false},
        {"MultipleGridRaggedChunks", 16, 30, 64, 1, false, 16, 7, 32, false},

        // T=4, I=17, g=8 = T*F with F=2 and F does not divide I. Without the
        // capability the global mapping is non-uniform; with it, every tile
        // gets the same (9,8) fold signature.
        {"PerTileExtraIters_T4I17G8_NoCap", 4, 17, 8, 1, false, 4, 8, 4, false, false},
        {"PerTileExtraIters_T4I17G8_Cap", 4, 17, 8, 1, false, 4, 8, 4, true, true},

        // Same all-partial remainder shape as MultipleGridRaggedChunks, but the
        // capability admits it.
        {"MultipleGridRaggedChunks_Cap", 16, 30, 64, 1, false, 16, 7, 32, true, true},

        // The three grids that matter around the StreamKFlagElements (2048)
        // bound, for tiles=100 I=256. 2048 is what clamping a selected grid
        // down to the bound produces, and it is not a multiple of the tile
        // count, so the packer lands on ragged chunks and the predicate refuses
        // it -- which is why the bound is a constraint on grid selection rather
        // than a correction applied after it. 1600 is the largest uniform grid
        // at or below the bound and is accepted. The all-full grid is accepted
        // even above the bound: tiles % grid == 0 leaves no partial tiles, so
        // the launch never takes the flag protocol.
        {"FlagClampedGridNotRowUniform", 100, 256, 2048, 1, false, 100, 12, 1024, false},
        {"FlagBoundedGridRowUniform", 100, 256, 1600, 1, false, 100, 16, 0, true},
        {"AllFullGridAboveFlagBound", 4096, 16, 4096, 1, false, 4096, 16, 0, true},

        // TENSILE_STREAMK_FULL_TILES=0 is the only way to reach a non-empty
        // data-parallel region alongside equally cut StreamK tiles. Those two
        // populations have different fold signatures, so skTiles == tiles is
        // what refuses it -- extraIters is 0 here and the chunk length does
        // divide the tile length.
        {"FullTilesZeroMixedRegions", 320, 64, 256, 0, false, 64, 16, 0, false},

        // StreamKForceDPOnly keeps every tile whole. Accepted today, and the
        // predicate must not regress it: skItersPerWG is 0, so both of the
        // other accept clauses are false.
        {"ForceDataParallelOnly", 300, 64, 256, 1, true, 0, 0, 0, true},

        // Grouped GEMM calls the gate with a default-constructed
        // StreamKSettings. The helper must not divide by zero; the gate
        // refuses the launch on grid == 0 before ever consulting the split.
        {"ZeroGrid", 300, 64, 0, 1, false, 0, 0, 0, true},
    };

    class RowUniformityStreamKSplit_pre_checkin : public ::testing::TestWithParam<SplitCase>
    {
    };

    TEST_P(RowUniformityStreamKSplit_pre_checkin, MirrorsPackingAndDecidesUniformity)
    {
        const SplitCase& c = GetParam();

        const TensileLite::StreamKStaticSplit split = TensileLite::streamKStaticSplit(
            c.tiles, c.itersPerTile, c.grid, c.skFullTiles, c.forceDPOnly);

        EXPECT_EQ(split.skTiles, c.skTiles) << "packed skTiles";
        EXPECT_EQ(split.skItersPerWG, c.skItersPerWG) << "packed SKItersPerWG";
        EXPECT_EQ(split.extraIters, c.extraIters)
            << "skTiles*itersPerTile - SKItersPerWG*skGrid, the leftover the kernel recomputes";

        EXPECT_EQ(TensileLite::streamKStaticSplitRowUniform(
                      split, c.tiles, c.itersPerTile, c.grid, c.perTileExtraIters),
                  c.rowUniform)
            << "tiles=" << c.tiles << " itersPerTile=" << c.itersPerTile << " grid=" << c.grid
            << " skFullTiles=" << c.skFullTiles << " forceDPOnly=" << c.forceDPOnly
            << " perTileExtraIters=" << c.perTileExtraIters
            << " -> skTiles=" << split.skTiles << " skItersPerWG=" << split.skItersPerWG
            << " extraIters=" << split.extraIters;
    }

    INSTANTIATE_TEST_SUITE_P(RowUniformity,
                             RowUniformityStreamKSplit_pre_checkin,
                             ::testing::ValuesIn(kSplitCases),
                             [](const ::testing::TestParamInfo<SplitCase>& info) {
                                 return std::string(info.param.name);
                             });

    // Host mirror of StreamK.py skAssignItersPerTile for T=4, I=17, g=8
    // (F=2 does not divide I; each tile gets a (9,8) fold).
    TEST(RowUniformityPerTileExtraIters_pre_checkin, PerTileStartEndMatchesFormula)
    {
        constexpr size_t tiles        = 4;
        constexpr size_t itersPerTile = 17;
        constexpr size_t skGrid       = 8;
        // Expected WG owners for iterations 0..67 under the per-tile column.
        const int expectedWg[68] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7};

        for(size_t w = 0; w < skGrid; ++w)
        {
            const auto range = TensileLite::streamKWorkgroupIterRange(
                w, tiles, itersPerTile, skGrid, /*perTileExtraIters=*/true);
            for(size_t it = range.start; it < range.end; ++it)
            {
                ASSERT_LT(it, size_t{68});
                EXPECT_EQ(expectedWg[it], static_cast<int>(w))
                    << "iter " << it << " start=" << range.start << " end=" << range.end;
            }
        }

        // Without the capability, the same shape stays on the global first-E
        // mapping and is not row-uniform.
        const auto split
            = TensileLite::streamKStaticSplit(tiles, itersPerTile, skGrid, 1, false);
        EXPECT_FALSE(TensileLite::streamKStaticSplitRowUniform(
            split, tiles, itersPerTile, skGrid, /*perTileExtraIters=*/false));
        EXPECT_TRUE(TensileLite::streamKStaticSplitRowUniform(
            split, tiles, itersPerTile, skGrid, /*perTileExtraIters=*/true));

        // When I % F == 0 the per-tile mapping must match the global E==0 path.
        constexpr size_t evenI = 16;
        for(size_t w = 0; w < skGrid; ++w)
        {
            const auto perTile = TensileLite::streamKWorkgroupIterRange(
                w, tiles, evenI, skGrid, true);
            const auto global = TensileLite::streamKWorkgroupIterRange(
                w, tiles, evenI, skGrid, false);
            EXPECT_EQ(perTile.start, global.start) << "w=" << w;
            EXPECT_EQ(perTile.end, global.end) << "w=" << w;
        }
    }

    // =======================================================================
    // StreamK configurations that cannot be shown row-uniform.
    //
    // None of these is present in the tuned logic, so no problem shape can
    // produce them; they are exercised against a synthesised solution instead.
    // The surface used is uniformSummationOrderSupported(), which now resolves
    // StreamK / GSU / StaggerU the same way solve() does and admits only
    // launches the gate would accept (except a missing Synchronizer pointer).
    // Static obstacles still share one implementation with the gate. The
    // gate's split arithmetic is covered directly by the split cases above;
    // the tests below pin that selection drops the same configurations.
    // =======================================================================

    // A mock device rather than the real one: nothing below depends on the
    // hardware, and this keeps the cases running where there is no GPU.
    // skDynamicGrid is forced off so StreamK resolution stays on the
    // non-analytical path (plain AMDGPU has no origami hardware). The
    // AMDGPU constructor reads TENSILE_STREAMK_DYNAMIC_GRID (default 6 =
    // k_split_aware), so leaving the field at its post-construction value
    // would take getSKReduction into HipAMDGPU::analyticalHardware and abort.
    TensileLite::AMDGPU probeHardware()
    {
        auto hardware = TensileLite::AMDGPU(
            TensileLite::AMDGPU::Processor::gfx950, 256, "row_uniformity_probe");
        hardware.skDynamicGrid = 0;
        return hardware;
    }

    // A minimal StreamK=3 solution that is accepted as it stands, so each case
    // below changes exactly one field and attributes the outcome to it.
    // GlobalAccumulation 4 (PartialsBuffer) and a runtime StaggerU keep the
    // predicate's other clauses satisfied.
    //
    // workGroupMapping and workGroupMappingXCC must stay off the origami
    // defaults (0 and -1): calculateAutoWGM, called from the shared launch
    // obstacle helper, would otherwise dereference HipAMDGPU::analyticalHardware
    // on this mock.
    std::shared_ptr<TensileLite::ContractionSolution> probeSolution()
    {
        auto solution                            = std::make_shared<TensileLite::ContractionSolution>();
        solution->kernelName                     = "row_uniformity_probe_kernel";
        solution->sizeMapping.streamK            = 3;
        solution->sizeMapping.streamKAtomic      = 0;
        solution->sizeMapping.streamKForceDPOnly = 0;
        solution->sizeMapping.macroTile          = TensileLite::dim3(128, 128, 1);
        solution->sizeMapping.workGroupSize      = TensileLite::dim3(256, 1, 1);
        solution->sizeMapping.threadTile         = TensileLite::dim3(1, 1, 1);
        solution->sizeMapping.depthU             = 256;
        solution->sizeMapping.matrixInstruction  = {32, 32, 128, 1};
        solution->sizeMapping.LocalSplitU        = 1;
        solution->sizeMapping.packBatchDims      = 0;
        solution->sizeMapping.workGroupMapping   = 1;
        solution->sizeMapping.workGroupMappingXCC = 0;
        solution->sizeMapping.staggerU           = 0;
        solution->sizeMapping.globalSplitU       = 1;
        solution->sizeMapping.globalAccumulation = 4;
        solution->internalArgsSupport.staggerU   = true;
        solution->problemType.mxScaleFormat      = 1;
        return solution;
    }

    TensileLite::ContractionProblemGemm probeProblem()
    {
        auto problem = TensileLite::ContractionProblemGemm::GEMM(
            false, false, 1024, 1024, 1024, 1024, 1024, 1024, 0.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setParams().setUniformSummationOrder(true);
        return problem;
    }

    bool admitsUniformSummationOrder(const TensileLite::ContractionSolution& solution,
                                     const TensileLite::Hardware&            hardware)
    {
        return solution.uniformSummationOrderSupported(probeProblem(), hardware);
    }

    TEST(RowUniformityStreamKRejection_pre_checkin, AdmitsTheProbeConfiguration)
    {
        const auto hardware = probeHardware();
        auto       solution = probeSolution();
        auto       problem  = probeProblem();

        ASSERT_EQ(hardware.skDynamicGrid, 0)
            << "probe AMDGPU must keep StreamK on the non-analytical path; "
               "plain AMDGPU has no origami hardware";
        ASSERT_NE(solution->sizeMapping.workGroupMapping, 0)
            << "workGroupMapping==0 would take the origami WGM path on this mock";
        ASSERT_NE(solution->sizeMapping.workGroupMappingXCC, -1)
            << "workGroupMappingXCC==-1 would take the origami WGMXCC path on this mock";

        // Selection now calls getSKReduction / getSKGrid. Those must not abort
        // on a plain AMDGPU (no analyticalHardware).
        EXPECT_EQ(solution->getSKReduction(problem, hardware), origami::reduction_t::tree);
        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_GT(tiles, 0u);
        const size_t grid = solution->getSKGrid(problem, hardware, tiles, origami::reduction_t::tree);
        ASSERT_GT(grid, 0u);
        const size_t iters
            = std::max(size_t{1}, problem.getItersPerTile(solution->sizeMapping));
        const auto split = TensileLite::streamKStaticSplit(
            tiles, iters, grid, hardware.skFullTiles, solution->sizeMapping.streamKForceDPOnly != 0);
        EXPECT_TRUE(TensileLite::streamKStaticSplitRowUniform(
            split, tiles, iters, grid, solution->internalArgsSupport.perTileExtraIters))
            << "the unmodified probe must be launch-legal on the static split "
               "(tiles="
            << tiles << " grid=" << grid << " skTiles=" << split.skTiles
            << " skItersPerWG=" << split.skItersPerWG << " extraIters=" << split.extraIters
            << "); the Synchronizer pointer is the one clause selection skips";

        EXPECT_TRUE(solution->uniformSummationOrderSupported(problem, hardware))
            << "The unmodified probe must be admitted, or the cases below cannot attribute a "
               "rejection to the single field they change";
    }

    // Rejection A. UseInitialStridesCD turns off optSingleColVgpr and
    // optSharedColVgpr, so the StreamK partials store stops being
    // coordinate-agnostic.
    TEST(RowUniformityStreamKRejection_pre_checkin, InitialStridesCD)
    {
        const auto hardware                       = probeHardware();
        auto       solution                       = probeSolution();
        solution->problemType.useInitialStridesCD = true;

        EXPECT_FALSE(admitsUniformSummationOrder(*solution, hardware))
            << "StreamK with UseInitialStridesCD must be refused";
    }

    // Rejection B. A packed index set in either output dimension has the same
    // effect on the store addressing. The public GEMM API always presents one
    // free index per operand, so PackBatchDims is the reachable half.
    TEST(RowUniformityStreamKRejection_pre_checkin, PackedC0IndexSet)
    {
        const auto hardware                  = probeHardware();
        auto       solution                  = probeSolution();
        solution->sizeMapping.packBatchDims = 0x1;

        EXPECT_FALSE(admitsUniformSummationOrder(*solution, hardware))
            << "StreamK with a packed C0 index set must be refused";
    }

    TEST(RowUniformityStreamKRejection_pre_checkin, PackedC1IndexSet)
    {
        const auto hardware                  = probeHardware();
        auto       solution                  = probeSolution();
        solution->sizeMapping.packBatchDims = 0x2;

        EXPECT_FALSE(admitsUniformSummationOrder(*solution, hardware))
            << "StreamK with a packed C1 index set must be refused";
    }

    // Rejection C. WorkGroup[2] > 1 with LocalSplitU == 1 is exactly
    // WaveSplitK, whose redundant-lane store mask does not cover the WS
    // partials store StreamK writes.
    TEST(RowUniformityStreamKRejection_pre_checkin, WaveSplitK)
    {
        const auto hardware                  = probeHardware();
        auto       solution                  = probeSolution();
        solution->sizeMapping.workGroupSize = TensileLite::dim3(256, 1, 16);
        solution->sizeMapping.LocalSplitU   = 1;

        EXPECT_FALSE(admitsUniformSummationOrder(*solution, hardware))
            << "StreamK with WaveSplitK must be refused";
    }

    // The same WorkGroup[2] with LocalSplitU carrying it is the ordinary
    // LocalSplitU configuration that 7,614 shipped solutions use, and it must
    // stay accepted: the two are mutually exclusive projections of one slot.
    TEST(RowUniformityStreamKRejection_pre_checkin, LocalSplitUIsNotWaveSplitK)
    {
        const auto hardware                  = probeHardware();
        auto       solution                  = probeSolution();
        solution->sizeMapping.workGroupSize = TensileLite::dim3(256, 1, 16);
        solution->sizeMapping.LocalSplitU   = 16;

        EXPECT_TRUE(admitsUniformSummationOrder(*solution, hardware))
            << "StreamK with LocalSplitU > 1 must stay admitted";
    }

    // WaveSplitK without StreamK never reaches a WS partials store, so the
    // guard must not touch it: all 22 shipped WaveSplitK solutions are
    // StreamK 0, and an unconditional form would refuse them for nothing.
    TEST(RowUniformityStreamKRejection_pre_checkin, WaveSplitKWithoutStreamK)
    {
        const auto hardware                  = probeHardware();
        auto       solution                  = probeSolution();
        solution->sizeMapping.streamK       = 0;
        solution->sizeMapping.workGroupSize = TensileLite::dim3(256, 1, 16);
        solution->sizeMapping.LocalSplitU   = 1;

        EXPECT_TRUE(admitsUniformSummationOrder(*solution, hardware))
            << "WaveSplitK without StreamK must stay admitted";
    }

    TEST(RowUniformityStreamKRejection_pre_checkin, HandwrittenCustomKernel)
    {
        const auto hardware              = probeHardware();
        auto       solution              = probeSolution();
        solution->customKernel.name      = "DummyCustomKernel";
        solution->customKernel.generated = false;

        EXPECT_FALSE(admitsUniformSummationOrder(*solution, hardware))
            << "A handwritten custom kernel must be refused under uniform summation order";
    }

    // SupportCustomStaggerU: False does not mean the kernel found its stagger
    // somewhere the clamp cannot reach. It means only that the host declines to
    // write the packed StaggerU field, which leaves it 0, and a generated
    // kernel reads StaggerU from that field and from nowhere else -- the
    // declared value survives code generation solely as an SGPR-pool sizing
    // input. So a generated kernel that declares a non-zero StaggerU with no
    // runtime StaggerU argument is reached by the clamp like any other, and
    // must stay admitted rather than being refused for nothing.
    TEST(RowUniformityStreamKRejection_pre_checkin, GeneratedKernelWithCompiledInStaggerU)
    {
        const auto hardware                    = probeHardware();
        auto       solution                    = probeSolution();
        solution->sizeMapping.streamK          = 0;
        solution->internalArgsSupport.staggerU = false;
        solution->sizeMapping.staggerU         = 16;
        auto problem                           = probeProblem();

        // The admission below is only meaningful because the clamp really does
        // settle this launch at StaggerU 0: it is the resolved value, not the
        // declared one, that decides whether the summation order is uniform.
        const int32_t autoWGM
            = std::get<0>(solution->calculateAutoWGM(problem, &hardware, /*skgrid=*/0));
        EXPECT_EQ(std::get<1>(solution->calculateAutoStaggerU(problem, &hardware, 0, autoWGM)), 0u)
            << "uniform summation order clamps resolved StaggerU to 0";

        EXPECT_TRUE(solution->uniformSummationOrderSupported(problem, hardware))
            << "a generated kernel takes StaggerU from the packed argument, so the clamp "
               "reaches it and a declared StaggerU must not refuse it";
    }

    // The other half of the same rule: frozen hand-written assembly can bake a
    // literal rotation into its main loop where no host-side clamp reaches, so
    // a compiled-in StaggerU there must still be refused.
    //
    // The refusal this pins is the outcome, not one particular clause. The
    // CustomKernel clause at the top of the predicate already refuses every
    // hand-written custom kernel outright, so it is what fires here; the
    // narrower CompiledInStaggerU clause sits behind it as defense in depth,
    // for the day that first clause is relaxed to admit individually vetted
    // custom kernels. Both are private, so the public predicate cannot say
    // which one objected -- what matters is that this configuration never
    // becomes admissible.
    TEST(RowUniformityStreamKRejection_pre_checkin, HandwrittenCustomKernelWithCompiledInStaggerU)
    {
        const auto hardware                    = probeHardware();
        auto       solution                    = probeSolution();
        solution->sizeMapping.streamK          = 0;
        solution->internalArgsSupport.staggerU = false;
        solution->sizeMapping.staggerU         = 16;
        solution->customKernel.name            = "DummyCustomKernel";
        solution->customKernel.generated       = false;
        auto problem                           = probeProblem();

        EXPECT_FALSE(solution->uniformSummationOrderSupported(problem, hardware))
            << "a hand-written custom kernel with a compiled-in StaggerU must be refused";
    }

    // The old selection filter skipped GSU when AdaptiveGemmGSUA was set, so a
    // kernel the launch gate would refuse (SingleBuffer GSU>1 that does not
    // adapt up to MultipleBuffer) could still win the heuristic. Selection
    // now resolves accumulation the same way solve() does.
    TEST(RowUniformityStreamKRejection_pre_checkin, AdaptiveGsuThatLaunchWouldReject)
    {
        const auto hardware                      = probeHardware();
        auto       solution                      = probeSolution();
        solution->sizeMapping.streamK            = 0;
        solution->sizeMapping.adaptiveGemmGSUA   = 1;
        solution->sizeMapping.globalAccumulation = 0;
        solution->sizeMapping.globalSplitU       = 4;
        auto problem                             = probeProblem();

        // 1024x1024x1024 / MT 128x128 / DepthU 256 → 64 tiles, 4 iters/tile,
        // GSU 4. AdaptiveGemmGSUA upgrades to MultipleBuffer only when
        // itersPerTile >= 64 or GSU >= 64, so this shape stays SingleBuffer.
        const uint32_t gsu = solution->calculateAutoGSU(problem, &hardware);
        ASSERT_EQ(gsu, 4u);
        ASSERT_EQ(problem.getAccumulation(hardware, solution->sizeMapping, gsu), 0u)
            << "this shape must stay on SingleBuffer so the test is about the old adaptive "
               "skip, not about an MB upgrade the launch gate would accept";

        EXPECT_FALSE(solution->uniformSummationOrderSupported(problem, hardware))
            << "Adaptive GSU that still resolves to SingleBuffer with GSU>1 must be refused "
               "at selection, not at the launch gate";
    }

    // SK4 does not take the static two-tile snap. On this mock device the
    // grid is the CU count (256) and the tile count is 64, so the launch
    // gate's divisibility check fails. Selection must drop it rather than
    // return it as a USO winner.
    TEST(RowUniformityStreamKRejection_pre_checkin, DynamicStreamKGridDoesNotDivideTiles)
    {
        const auto hardware           = probeHardware();
        auto       solution           = probeSolution();
        solution->sizeMapping.streamK = 4;
        auto problem                  = probeProblem();

        ASSERT_EQ(hardware.skDynamicGrid, 0);
        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, 64u) << "1024x1024 with MT 128x128";
        const size_t grid
            = solution->getSKGrid(problem, hardware, tiles, origami::reduction_t::tree);
        ASSERT_EQ(grid, static_cast<size_t>(hardware.computeUnitCount))
            << "SK4 does not take the static two-tile USO snap; mock grid is the CU count";
        ASSERT_NE(tiles % grid, 0u)
            << "the launch gate refuses the dynamic-queue path when tiles % grid != 0";

        EXPECT_FALSE(solution->uniformSummationOrderSupported(problem, hardware))
            << "A dynamic-queue StreamK launch whose grid does not divide the tile count "
               "must be refused at selection";
    }

    // The dynamic-queue path takes no USO grid snap, so nothing there selects a
    // grid under the flag bound. When the clamp fires on that path it rewrites
    // the grid to StreamKFlagElements, which need not divide the tile count --
    // and the gate must then refuse the launch rather than admit a split whose
    // rows would depend on the schedule. This is the fail-closed half of the
    // bound, and the reason the snap folds the bound in rather than relying on
    // the clamp to repair a grid after the fact.
    TEST(RowUniformityStreamKRejection_pre_checkin, FlagClampedDynamicGridFailsClosed)
    {
        auto hardware                 = probeHardware();
        hardware.skFixedGrid          = 4096;
        auto solution                 = probeSolution();
        solution->sizeMapping.streamK = 4;

        auto problem = TensileLite::ContractionProblemGemm::GEMM(
            false, false, 1280, 1280, 1024, 1280, 1280, 1280, 0.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setParams().setUniformSummationOrder(true);

        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, 100u) << "1280x1280 with MT 128x128";
        ASSERT_GT(static_cast<size_t>(hardware.skFixedGrid),
                  static_cast<size_t>(TensileLite::StreamKFlagElements))
            << "the requested grid must be above the bound so the clamp is what fires";

        const size_t grid
            = solution->getSKGrid(problem, hardware, tiles, origami::reduction_t::tree);
        ASSERT_EQ(grid, static_cast<size_t>(TensileLite::StreamKFlagElements))
            << "SK4 takes no snap, so the flag clamp is the only thing that moves the grid";
        ASSERT_NE(tiles % grid, 0u) << "the clamped grid does not divide the tile count";

        EXPECT_FALSE(solution->uniformSummationOrderSupported(problem, hardware))
            << "a grid the flag clamp moved off a tile multiple must be refused at selection, "
               "not launched";

        // Control: the refusal is the uniform-summation-order gate acting on
        // that grid, not some unrelated property of this mock solution.
        auto problemOff = problem;
        problemOff.setParams().setUniformSummationOrder(false);
        EXPECT_TRUE(solution->uniformSummationOrderSupported(problemOff, hardware))
            << "with the mode off the same configuration must still be selectable";
    }

    TEST(RowUniformityStreamKRejection_pre_checkin, FilterIsIdleWhenUniformSummationOrderIsOff)
    {
        const auto hardware           = probeHardware();
        auto       solution           = probeSolution();
        solution->sizeMapping.streamK = 4;
        auto problemOn                = probeProblem();
        auto problemOff               = probeProblem();
        problemOff.setParams().setUniformSummationOrder(false);

        EXPECT_FALSE(solution->uniformSummationOrderSupported(problemOn, hardware))
            << "SK4 on this mock is not launch-legal under USO; the idle-filter assertion "
               "below is only meaningful against that contrast";
        EXPECT_TRUE(solution->uniformSummationOrderSupported(problemOff, hardware))
            << "The selection filter must not drop kernels when uniform summation order is off";
    }

    TEST(RowUniformityStreamKRejection_pre_checkin, GroupedGemmStreamKGridIsZero)
    {
        const auto hardware = probeHardware();
        auto       solution = probeSolution();
        auto       problem  = probeProblem();
        problem.setGroupedGemm(true);

        EXPECT_FALSE(solution->uniformSummationOrderSupported(problem, hardware))
            << "Grouped GEMM packs skGrid == 0; a StreamK kernel must not win USO selection";
    }

    TEST(RowUniformityStreamKRejection_pre_checkin, GroupedGemmNonStreamKStillAdmitted)
    {
        const auto hardware           = probeHardware();
        auto       solution           = probeSolution();
        solution->sizeMapping.streamK = 0;
        auto problem                  = probeProblem();
        problem.setGroupedGemm(true);

        EXPECT_TRUE(solution->uniformSummationOrderSupported(problem, hardware))
            << "Grouped GEMM without StreamK must stay selectable when the rest of the "
               "launch is row-uniform";
    }

    // Rejection D. The MX cases are a conjunction over a guard, and the
    // non-rejections are as much the point as the rejection: a kernel-name
    // substring test for MX would refuse 2,065 shipped f32 StreamK solutions
    // whose _MX_ token comes from F32XdlMathOp=XFloat32, not from block
    // scaling.
    struct MXCase
    {
        const char* name;
        int         mxBlock;
        size_t      depthU;
        bool        rejected;
    };

    const MXCase kMXCases[] = {
        // Every shipped MX StreamK solution sits at DepthU 256 or 512.
        {"MXDepthU256", 32, 256, false},
        {"MXDepthU512", 32, 512, false},
        // Reachable through the generator's own validation for MX fp8, where
        // duUnit is numSubIterK(1) * MatrixInstK(128) * LocalSplitU(1) = 128.
        {"MXDepthU128", 32, 128, true},
        // The guard must scope to MX problems and nothing else.
        {"NonMXDepthU128", 0, 128, false},
        {"NonMXDepthU16", 0, 16, false},
    };

    class RowUniformityStreamKMX_pre_checkin : public ::testing::TestWithParam<MXCase>
    {
    };

    TEST_P(RowUniformityStreamKMX_pre_checkin, GranuleAlignment)
    {
        const MXCase& c        = GetParam();
        const auto    hardware = probeHardware();

        auto solution                    = probeSolution();
        solution->problemType.mxBlockA   = c.mxBlock;
        solution->problemType.mxBlockB   = c.mxBlock;
        solution->sizeMapping.depthU     = c.depthU;

        EXPECT_EQ(!admitsUniformSummationOrder(*solution, hardware), c.rejected)
            << "mxBlock=" << c.mxBlock << " depthU=" << c.depthU;
    }

    INSTANTIATE_TEST_SUITE_P(RowUniformity,
                             RowUniformityStreamKMX_pre_checkin,
                             ::testing::ValuesIn(kMXCases),
                             [](const ::testing::TestParamInfo<MXCase>& info) {
                                 return std::string(info.param.name);
                             });

    // The granule is derived for HostPreSwizzle, block size 32 and
    // MatrixInstK 128. Outside that envelope 256 is the wrong number rather
    // than a violated one, so the envelope is pinned too.
    TEST(RowUniformityStreamKRejection_pre_checkin, MXScaleFormatEnvelope)
    {
        const auto hardware                  = probeHardware();
        auto       solution                  = probeSolution();
        solution->problemType.mxBlockA      = 32;
        solution->problemType.mxBlockB      = 32;
        solution->problemType.mxScaleFormat = 2; // InMemorySwizzle

        EXPECT_FALSE(admitsUniformSummationOrder(*solution, hardware))
            << "An unaudited MX scale layout must be refused";
    }

    TEST(RowUniformityStreamKRejection_pre_checkin, MXMatrixInstKEnvelope)
    {
        const auto hardware                       = probeHardware();
        auto       solution                       = probeSolution();
        solution->problemType.mxBlockA           = 32;
        solution->problemType.mxBlockB           = 32;
        solution->sizeMapping.matrixInstruction = {32, 32, 64, 1};

        EXPECT_FALSE(admitsUniformSummationOrder(*solution, hardware))
            << "An MX MatrixInstK the granule derivation does not cover must be refused";
    }

    // =======================================================================
    // The newly admitted StreamK split regime, on hardware.
    // =======================================================================

    // What the host resolves for one (solution, problem) pair, recomputed with
    // the same functions solve() uses rather than with a model of them.
    struct StreamKResolution
    {
        bool                            streamK       = false;
        bool                            staticPacking = false;
        bool                            tree          = false;
        size_t                          tiles         = 0;
        size_t                          itersPerTile  = 0;
        size_t                          grid          = 0;
        TensileLite::StreamKStaticSplit split;
        bool                            rowUniform = false;
        // Row-uniform and refused by the pre-existing tiles % grid == 0 test,
        // i.e. exactly the regime this change adds.
        bool newlyAdmitted     = false;
        bool perTileExtraIters = false;
    };

    // The TensileLite problem matching what the runtime builds for the public
    // NN column-major GEMM the harness runs. workspaceSize is set explicitly:
    // the StreamK grid selector clamps the grid against it, so leaving it at 0
    // would collapse the grid to the tile count and quietly move every case
    // into the regime the gate already accepted.
    TensileLite::ContractionProblemGemm tensileProblemFor(const Problem& problem,
                                                         size_t         workspaceBytes)
    {
        const rocisa::DataType ab = problem.abType == HIP_R_32F ? rocisa::DataType::Float
                                                                : rocisa::DataType::BFloat16;
        const size_t           m  = static_cast<size_t>(problem.m);
        const size_t           n  = static_cast<size_t>(problem.n);
        const size_t           k  = static_cast<size_t>(problem.k);

        auto tensile = TensileLite::ContractionProblemGemm::GEMM_Strides(false,
                                                                        false,
                                                                        ab,
                                                                        ab,
                                                                        rocisa::DataType::Float,
                                                                        rocisa::DataType::Float,
                                                                        m,
                                                                        n,
                                                                        k,
                                                                        1,
                                                                        m,
                                                                        m * k,
                                                                        k,
                                                                        k * n,
                                                                        m,
                                                                        m * n,
                                                                        m,
                                                                        m * n,
                                                                        0.0);
        tensile.setComputeInputTypeA(ab);
        tensile.setComputeInputTypeB(ab);
        tensile.setAlphaType(rocisa::DataType::Float);
        tensile.setBetaType(rocisa::DataType::Float);
        tensile.setWorkspaceSize(workspaceBytes);
        tensile.setParams().setUniformSummationOrder(true);
        tensile.setParams().setSmCountTarget(problem.smCountTarget);
        if(problem.useBias)
        {
            // useBias=1: bias length along M (matches EPILOGUE_BIAS on NN GEMM).
            tensile.setUseBias(1);
            tensile.setBias(rocisa::DataType::Float,
                            static_cast<size_t>(problem.m),
                            /*stride=*/1);
        }
        return tensile;
    }

    StreamKResolution resolveStreamK(const TensileLite::ContractionSolution&    solution,
                                     const TensileLite::ContractionProblemGemm& tensile,
                                     const TensileLite::Hardware&               hardware)
    {
        StreamKResolution out;
        if(solution.sizeMapping.streamK == 0)
            return out;

        out.streamK = true;

        const bool effectiveDynamic = solution.sizeMapping.streamK == 5
                                          ? solution.streamK5EffectiveDynamic(tensile, hardware)
                                          : false;
        out.staticPacking           = solution.sizeMapping.streamK == 3
                                      || (solution.sizeMapping.streamK == 5 && !effectiveDynamic);

        const origami::reduction_t reduction
            = effectiveDynamic ? origami::reduction_t::tree
                               : solution.getSKReduction(tensile, hardware);
        out.tree = reduction == origami::reduction_t::tree;

        out.tiles = tensile.getNumTiles(solution.sizeMapping, 1);
        out.itersPerTile
            = std::max(size_t{1}, tensile.getItersPerTile(solution.sizeMapping));
        out.grid = solution.getSKGrid(tensile, hardware, out.tiles, reduction);

        if(!out.staticPacking || !out.tree || out.grid == 0)
            return out;

        auto const* amdgpu = dynamic_cast<TensileLite::AMDGPU const*>(&hardware);
        out.split          = TensileLite::streamKStaticSplit(
            out.tiles,
            out.itersPerTile,
            out.grid,
            amdgpu != nullptr ? amdgpu->skFullTiles : 1,
            solution.sizeMapping.streamKForceDPOnly != 0);
        out.perTileExtraIters = solution.internalArgsSupport.perTileExtraIters;
        out.rowUniform        = TensileLite::streamKStaticSplitRowUniform(
            out.split, out.tiles, out.itersPerTile, out.grid, out.perTileExtraIters);
        out.newlyAdmitted
            = out.rowUniform && out.split.skTiles != 0 && out.tiles % out.grid != 0;
        return out;
    }

    // A tall, narrow shape with a long K, which is what puts the tile count
    // below the CU count and lets the grid selector cut every tile into the
    // same number of chunks: grid = F * tiles for an integer F >= 2. When F
    // divides the iterations per tile, that is an even K-split (extraIters ==
    // 0). When it does not, leftover K-iters (extraIters != 0) stay
    // tile-symmetric if the kernel redistributes extras within each tile;
    // splits inside one tile may still differ (I % F != 0). Both are
    // row-uniform, and both were refused before this change because the grid
    // does not divide the tile count.
    //
    // Written to discover rather than to pin. Solution indices, the
    // tile_fractions vector and select_reduction's thresholds are all library
    // details that a retune can move, so the case enumerates candidates,
    // resolves each one, and asserts on whichever lands in the new regime --
    // in both directions, since a candidate that resolves to parallel
    // reduction must still be refused.
    class RowUniformityClauseTwo_pre_checkin : public ::testing::Test
    {
    protected:
        void checkNewlyAdmittedSplit(const Problem& problem)
        {
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";

            std::string libraryReason;
            if(!masterLibrary(libraryReason))
                GTEST_SKIP() << libraryReason;

            auto hardware = currentHardware();
            if(!hardware)
                GTEST_SKIP() << "could not describe the current device to TensileLite";

            RowUniformityHarness harness(problem);
            std::string          skipReason;
            if(!harness.setUp(skipReason))
            {
                if(skipReason.empty())
                    return;
                GTEST_SKIP() << skipReason;
            }

            const auto tensile = tensileProblemFor(problem, kWorkspaceBytes);

            int        enumeratedCount = 0;
            const auto candidates      = harness.candidateAlgos(enumeratedCount, 256);

                int         streamKCandidates   = 0;
                int         admitted            = 0;
                int         admittedParallel    = 0;
                int         refusedParallel     = 0;
                std::string firstAdmitted;

                for(const auto& candidate : candidates)
                {
                    std::string reason;
                    auto        solution = solutionForAlgo(candidate, reason);
                    if(!solution)
                        continue;

                    const StreamKResolution resolved
                        = resolveStreamK(*solution, tensile, *hardware);
                    if(!resolved.streamK)
                        continue;
                    ++streamKCandidates;

                    const std::string name = harness.solutionName(candidate);

                    // Parallel reduction: admit when the shared helper would
                    // (F = grid/tiles >= 2 and grid % tiles == 0 under static
                    // two-tile packing). Otherwise still refuse.
                    if(resolved.staticPacking && !resolved.tree)
                    {
                        const bool admitParallel
                            = resolved.tiles != 0 && resolved.grid % resolved.tiles == 0
                              && (resolved.grid / resolved.tiles) >= 2;
                        if(admitParallel)
                        {
                            EXPECT_EQ(harness.run(candidate, /*uniformMode=*/true),
                                      HIPBLAS_STATUS_SUCCESS)
                                << name
                                << " resolves to row-uniform parallel reduction and must be "
                                   "honored";
                            EXPECT_EQ(harness.firstNonUniformRowOfLastRun(), -1)
                                << name << " parallel launch must keep identical D rows";
                            ++admittedParallel;
                            continue;
                        }

                        EXPECT_EQ(harness.run(candidate, /*uniformMode=*/true),
                                  HIPBLAS_STATUS_INVALID_VALUE)
                            << name
                            << " resolves to parallel reduction that is not row-uniform and "
                               "must still be refused";
                        ++refusedParallel;
                        continue;
                    }

                    if(!resolved.newlyAdmitted)
                        continue;

                    // The split is only one of the gate's clauses. Consulting the
                    // selection-time predicate for the rest -- it now resolves
                    // StreamK / GSU / StaggerU the same way solve() does -- is
                    // what makes the success assertion below an assertion about
                    // a launch-legal winner, not a kernel the gate would still
                    // refuse.
                    if(!solution->uniformSummationOrderSupported(tensile, *hardware))
                        continue;

                    // Pin the resolution, not just the outcome: asserting only
                    // success would let a future threshold change degrade this case
                    // into a duplicate of the pre-existing one with no signal.
                    //
                    // Clause 2 is grid = F * tiles with F >= 2 and skTiles == tiles.
                    // That admits two K-splits:
                    //   even: extraIters == 0 and I % skItersPerWG == 0 (F divides I)
                    //   leftover: extraIters != 0, admitted only when the kernel
                    //     redistributes extras within each tile. Intra-tile chunk
                    //     lengths may still differ when I % F != 0; that is the
                    //     leftover regime, not the even-split one.
                    EXPECT_EQ(resolved.grid % resolved.tiles, 0u)
                        << name << ": clause 2 requires the grid to be a multiple of the tile count";
                    EXPECT_NE(resolved.grid, resolved.tiles)
                        << name << ": a grid equal to the tile count is the pre-existing regime";
                    EXPECT_EQ(resolved.split.skTiles, resolved.tiles);
                    EXPECT_NE(resolved.split.skItersPerWG, resolved.itersPerTile);
                    ASSERT_NE(resolved.split.skItersPerWG, 0u);

                    const bool leftoverSplit = resolved.split.extraIters != 0u;
                    if(leftoverSplit)
                    {
                        EXPECT_TRUE(resolved.perTileExtraIters)
                            << name << ": leftover extraIters=" << resolved.split.extraIters
                            << " (itersPerTile%skItersPerWG="
                            << (resolved.itersPerTile % resolved.split.skItersPerWG)
                            << ") is admitted only with per-tile extra-iters; do not "
                               "classify it as the even K-split";
                    }
                    else
                    {
                        EXPECT_EQ(resolved.split.extraIters, 0u);
                        EXPECT_EQ(resolved.itersPerTile % resolved.split.skItersPerWG, 0u);
                    }

                    if(admitted == 0)
                    {
                        firstAdmitted = name;
                        RecordProperty("tiles", static_cast<int>(resolved.tiles));
                        RecordProperty("iters_per_tile", static_cast<int>(resolved.itersPerTile));
                        RecordProperty("sk_grid", static_cast<int>(resolved.grid));
                        RecordProperty("grid_over_tiles",
                                       static_cast<int>(resolved.grid / resolved.tiles));
                        RecordProperty("sk_tiles", static_cast<int>(resolved.split.skTiles));
                        RecordProperty("sk_iters_per_wg",
                                       static_cast<int>(resolved.split.skItersPerWG));
                        RecordProperty("extra_iters",
                                       static_cast<int>(resolved.split.extraIters));
                        RecordProperty("leftover_split", leftoverSplit ? 1 : 0);
                    }
                    ++admitted;

                    EXPECT_EQ(harness.run(candidate, /*uniformMode=*/true), HIPBLAS_STATUS_SUCCESS)
                        << name << " has a row-uniform StreamK split (tiles=" << resolved.tiles
                        << " itersPerTile=" << resolved.itersPerTile << " grid=" << resolved.grid
                        << " skTiles=" << resolved.split.skTiles
                        << " skItersPerWG=" << resolved.split.skItersPerWG
                        << " extraIters=" << resolved.split.extraIters
                        << (leftoverSplit ? ", leftover per-tile extra-iters" : ", even K-split")
                        << ") and must be admitted rather than refused";

                    const int64_t badRow = harness.firstNonUniformRowOfLastRun();
                    EXPECT_EQ(badRow, -1)
                        << "Row " << badRow << " of D differs bitwise from row 0 for " << name;
                }

                RecordProperty("arch", gpuArchName());
                RecordProperty("algos_enumerated", enumeratedCount);
                RecordProperty("candidates_tried", static_cast<int>(candidates.size()));
                RecordProperty("streamk_candidates", streamKCandidates);
                RecordProperty("newly_admitted", admitted);
                RecordProperty("admitted_parallel", admittedParallel);
                RecordProperty("refused_parallel", refusedParallel);

                if(admitted == 0 && admittedParallel == 0 && refusedParallel == 0)
                {
                    const std::string arch      = gpuArchName();
                    const std::string processor = arch.substr(0, arch.find(':'));
                    const std::string detail
                        = std::string("No candidate resolved into either the newly admitted "
                                      "StreamK split regime, an admitted parallel reduction, or "
                                      "a refused parallel reduction, so this run witnessed "
                                      "neither. arch=")
                          + arch + " problem=" + std::to_string(problem.m) + "x"
                          + std::to_string(problem.n) + "x" + std::to_string(problem.k)
                          + " smCountTarget=" + std::to_string(problem.smCountTarget)
                          + " useBias=" + (problem.useBias ? "1" : "0")
                          + " algos_enumerated=" + std::to_string(enumeratedCount)
                          + " candidates_tried=" + std::to_string(candidates.size())
                          + " streamk_candidates=" + std::to_string(streamKCandidates);
                    // No Stream-K kernels in this device library for this
                    // problem: skip rather than fail-closed. Math CI gfx942 and
                    // TheRock gfx94X enumerate algos but streamk_candidates==0
                    // even with Bias; the GridBased exact-size SK entry from
                    // report 09 is not in those libraries.
                    if(streamKCandidates == 0)
                        GTEST_SKIP() << detail;
                    // Fail-closed only when Stream-K candidates exist and still
                    // none of the ClauseTwo outcomes fired (gate/steering bug):
                    //   gfx950 + no-bias  (Math CI / local MI355X)
                    //   gfx942 + Bias     (only if this library actually has SK)
                    // Elsewhere skip: gfx1201 and other empty-SK arches are
                    // already covered by streamKCandidates==0 above.
                    const bool expectWitness
                        = (processor.rfind("gfx950", 0) == 0 && !problem.useBias)
                          || (processor.rfind("gfx942", 0) == 0 && problem.useBias);
                    if(expectWitness)
                        FAIL() << detail;
                    GTEST_SKIP() << detail;
                }

                RecordProperty("first_admitted_solution", firstAdmitted);
            }
        };

    // An 80-CU budget through the public HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET
    // attribute. This is the configuration report 09 identified as reaching
    // the new regime on a stock MI300X, where the tuned entry is a bf16-in /
    // f32-out exact-size match.
    TEST_F(RowUniformityClauseTwo_pre_checkin, SplitRegime_4096x32x10240_SmCount80)
    {
        checkNewlyAdmittedSplit({4096, 32, 10240, HIP_R_16BF, 80});
    }

    // The same shape with the budget left unset, so the grid selector sees
    // every CU. Which candidates land in which regime changes completely, and
    // both directions must still hold.
    TEST_F(RowUniformityClauseTwo_pre_checkin, SplitRegime_4096x32x10240_AllCUs)
    {
        checkNewlyAdmittedSplit({4096, 32, 10240, HIP_R_16BF, 0});
    }

    // Bias epilogue of the same shape. Libraries that ship a Stream-K Bias
    // kernel (some GridBased gfx942 packs) still fail-closed when candidates
    // exist but none resolve. Math CI gfx942 / TheRock gfx94X ship zero
    // Stream-K solutions here, so checkNewlyAdmittedSplit skips instead.
    TEST_F(RowUniformityClauseTwo_pre_checkin, SplitRegime_4096x32x10240_SmCount80_Bias)
    {
        Problem problem{4096, 32, 10240, HIP_R_16BF, 80, /*useBias=*/true};
        checkNewlyAdmittedSplit(problem);
    }

    // Same Bias shape with the full CU count: on a stock MI300X (~304 CU) the
    // resolution flips to parallel and must still be refused under uniform summation order.
    TEST_F(RowUniformityClauseTwo_pre_checkin, SplitRegime_4096x32x10240_AllCUs_Bias)
    {
        Problem problem{4096, 32, 10240, HIP_R_16BF, 0, /*useBias=*/true};
        checkNewlyAdmittedSplit(problem);
    }

    // =======================================================================
    // Uniform-summation-order Stream-K grid steering (parallel admission + mixed-split snap)
    //
    // Host-only: synthesised SK3 solution + gfx950 analytical HipAMDGPU, same
    // pattern as tensilelite CuCount_test (which CI does not build).
    // =======================================================================

    TensileLite::hip::HipAMDGPU uniformitySteeringDevice()
    {
        using arch_t = origami::hardware_t::architecture_t;
        auto hw      = std::make_shared<origami::hardware_t>(
            arch_t::gfx950,
            /*N_CU=*/256,
            /*L2=*/163840,
            /*rf_capacity=*/262144,
            /*NUM_XCD=*/8,
            1.0,
            1.0,
            1.0,
            4000000,
            1.2,
            1,
            std::make_tuple(0.0, 0.008, 0.0));

        TensileLite::hip::HipAMDGPU device;
        device.processor          = TensileLite::AMDGPU::Processor::gfx950;
        device.computeUnitCount   = 256;
        device.deviceName         = "row_uniformity_grid_steering";
        device.analyticalHardware = hw;
        device.skDynamicGrid
            = static_cast<int>(origami::grid_selection_t::k_split_aware);
        device.skFixedGrid      = 0;
        device.skMaxCUs         = 0;
        device.skGridMultiplier = 1;
        return device;
    }

    std::shared_ptr<TensileLite::ContractionSolution> uniformitySteeringSolution()
    {
        auto solution = probeSolution();
        solution->sizeMapping.depthU                = 64;
        solution->sizeMapping.matrixInstruction     = {16, 16, 32, 1};
        solution->sizeMapping.CUOccupancy           = 1;
        solution->sizeMapping.workspaceSizePerElemC = 4;
        solution->internalArgsSupport.perTileExtraIters = false;
        return solution;
    }

    TensileLite::ContractionProblemGemm uniformityGemm(size_t m, size_t n, size_t k)
    {
        auto problem = TensileLite::ContractionProblemGemm::GEMM(
            false, false, m, n, k, m, n, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setWorkspaceSize(32ull << 20);
        return problem;
    }

    TEST(RowUniformityGridSteering_pre_checkin, KeepParallelUnderUniformSummationOrderWhenEligible)
    {
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        auto problem  = uniformityGemm(512, 512, 8192);
        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);

        EXPECT_EQ(solution->getSKReduction(problem, device), origami::reduction_t::parallel);

        problem.setParams().setUniformSummationOrder(true);
        EXPECT_EQ(solution->getSKReduction(problem, device), origami::reduction_t::parallel)
            << "uniform summation order must keep parallel when SK3 static, non-atomic, and obstacles clear";

        const size_t grid
            = solution->getSKGrid(problem, device, tiles, origami::reduction_t::parallel);
        TensileLite::StreamKSettings sk;
        sk.reduction = origami::reduction_t::parallel;
        sk.grid      = grid;
        EXPECT_TRUE(TensileLite::streamKParallelReductionRowUniform(
            sk, solution->sizeMapping.streamKAtomic, /*staticTwoTilePacking=*/true, tiles))
            << "gate helper must admit the resolved parallel settings (grid=" << grid
            << " tiles=" << tiles << ")";
        // Empirical note: report 15 measured UNIFORM for this parallel-window
        // class (512x512x8192) under mode-off; mode-on now keeps that reduction
        // when the admission helper holds, so the bitwise row guarantee rides
        // the same tile-symmetric PartialIdx mapping.
    }

    TEST(RowUniformityGridSteering_pre_checkin, ForceTreeUnderUniformSummationOrderWhenIneligible)
    {
        auto solution = uniformitySteeringSolution();
        solution->sizeMapping.streamKAtomic = 1;
        auto device  = uniformitySteeringDevice();
        auto problem = uniformityGemm(512, 512, 8192);

        EXPECT_EQ(solution->getSKReduction(problem, device), origami::reduction_t::parallel);

        problem.setParams().setUniformSummationOrder(true);
        EXPECT_EQ(solution->getSKReduction(problem, device), origami::reduction_t::tree)
            << "uniform summation order must force tree when parallel is ineligible (StreamKAtomic=1)";
    }

    TEST(RowUniformityGridSteering_pre_checkin, ForceTreeUnderUniformSummationOrderCustomKernel)
    {
        auto solution = uniformitySteeringSolution();
        solution->customKernel.name      = "DummyCustomKernel";
        solution->customKernel.generated = false;
        auto device  = uniformitySteeringDevice();
        auto problem = uniformityGemm(512, 512, 8192);

        EXPECT_EQ(solution->getSKReduction(problem, device), origami::reduction_t::tree);
        problem.setParams().setUniformSummationOrder(true);
        EXPECT_EQ(solution->getSKReduction(problem, device), origami::reduction_t::tree);
    }

    TEST(RowUniformityGridSteering_pre_checkin, NonDivisorGridBelowTilesSnapsToTiles)
    {
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 10;

        auto         problem = uniformityGemm(512, 512, 1024);
        const size_t tiles   = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, 16u);

        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree), 10u);

        problem.setParams().setUniformSummationOrder(true);
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree), tiles);
    }

    TEST(RowUniformityGridSteering_pre_checkin, DivisorGridBelowTilesSnapsToTiles)
    {
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 8;

        auto         problem = uniformityGemm(512, 512, 1024);
        const size_t tiles   = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, 16u);

        problem.setParams().setUniformSummationOrder(true);
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree), tiles)
            << "mixed GridDividesTiles (g0 | T, g0 < T) must snap up to all-full";
    }

    TEST(RowUniformityGridSteering_pre_checkin, AllPartialNonDivisorFRequiresCapability)
    {
        // T=4, I=17 (K=1088, DepthU=64), g0=8=T*F with F=2 and 2 does not divide 17.
        // Without perTileExtraIters: snap to T. With capability: keep T*F.
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 8;

        auto         problem = uniformityGemm(256, 256, 1088);
        const size_t tiles   = problem.getNumTiles(solution->sizeMapping, 1);
        const size_t I
            = std::max(size_t{1}, problem.getItersPerTile(solution->sizeMapping));
        ASSERT_EQ(tiles, 4u);
        ASSERT_EQ(I, 17u);
        // The whole point of the case: F=2 must NOT divide I, otherwise the
        // capability bit is not what decides the outcome.
        ASSERT_EQ(I % 2, 1u);

        // Mode off: fixed grid unchanged.
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree), 8u);

        problem.setParams().setUniformSummationOrder(true);
        solution->internalArgsSupport.perTileExtraIters = false;
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree), tiles)
            << "Without perTileExtraIters, F that does not divide I must snap down to T";

        solution->internalArgsSupport.perTileExtraIters = true;
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree), 8u)
            << "With perTileExtraIters, F need not divide I; keep T*F when workspace fits";
    }

    TEST(RowUniformityGridSteering_pre_checkin, ParallelSnapSkipsFIRequirement)
    {
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 8;

        auto         problem = uniformityGemm(256, 256, 1088);
        const size_t tiles   = problem.getNumTiles(solution->sizeMapping, 1);
        const size_t I
            = std::max(size_t{1}, problem.getItersPerTile(solution->sizeMapping));
        ASSERT_EQ(tiles, 4u);
        ASSERT_EQ(I, 17u);
        ASSERT_EQ(I % 2, 1u);

        problem.setParams().setUniformSummationOrder(true);
        solution->internalArgsSupport.perTileExtraIters = false;
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree), tiles)
            << "Tree without perTileExtraIters must still require F | I";
        EXPECT_EQ(
            solution->getSKGrid(problem, device, tiles, origami::reduction_t::parallel), 8u)
            << "Parallel snap must skip F | I and keep T*F when workspace fits";
    }

    TEST(RowUniformityGridSteering_pre_checkin, ModeOffLeavesNaturalGridAndReduction)
    {
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skFixedGrid      = 0;
        device.skMaxCUs         = 0;
        device.skGridMultiplier = 1;

        auto         problem = uniformityGemm(512, 512, 8192);
        const size_t tiles   = problem.getNumTiles(solution->sizeMapping, 1);

        EXPECT_FALSE(problem.getParams().uniformSummationOrder());

        const auto   redOff  = solution->getSKReduction(problem, device);
        const size_t gridOff = solution->getSKGrid(problem, device, tiles, redOff);

        // Flag stays false (default): a second call must land on the same
        // reduction AND the same grid, i.e. the steering is inert when off.
        EXPECT_EQ(solution->getSKReduction(problem, device), redOff);
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, redOff), gridOff);

        // Confirm the off path above was actually parallel so the comparison is
        // meaningful. USO-on admits parallel for this eligible shape.
        EXPECT_EQ(redOff, origami::reduction_t::parallel);
    }

    // MinItersPerCU (ContractionSolution.cpp, mirrors origami::streamk) is the
    // floor on iterations per Stream-K workgroup that the F-star search
    // enforces via `if((I / F) < MinItersPerCU) continue;`. Every other case
    // here sits at or above the floor, so lowering the constant would go
    // unnoticed. Bracket it from both sides with parallel reduction, which
    // skips the F | I requirement and therefore isolates this one condition:
    //   I=16, F=2 -> I/F == 8 must be ACCEPTED (fails if the floor rises to 9)
    //   I=15, F=2 -> I/F == 7 must be REJECTED (fails if the floor drops to 7)
    TEST(RowUniformityGridSteering_pre_checkin, MinItersPerCUBoundaryIsEight)
    {
        auto device          = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 8; // g0 = 8 = T * 2 with T = 4

        // K=1024, DepthU=64 -> I = 16, so I / F = 8 sits exactly on the floor.
        {
            auto         solution = uniformitySteeringSolution();
            auto         problem  = uniformityGemm(256, 256, 1024);
            const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
            const size_t I
                = std::max(size_t{1}, problem.getItersPerTile(solution->sizeMapping));
            ASSERT_EQ(tiles, 4u);
            ASSERT_EQ(I, 16u);
            ASSERT_EQ(I / 2, 8u);

            problem.setParams().setUniformSummationOrder(true);
            EXPECT_EQ(
                solution->getSKGrid(problem, device, tiles, origami::reduction_t::parallel), 8u)
                << "I/F == MinItersPerCU (8) is admissible; F=2 must be kept";
        }

        // K=960, DepthU=64 -> I = 15, so I / F = 7 is one below the floor.
        {
            auto         solution = uniformitySteeringSolution();
            auto         problem  = uniformityGemm(256, 256, 960);
            const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
            const size_t I
                = std::max(size_t{1}, problem.getItersPerTile(solution->sizeMapping));
            ASSERT_EQ(tiles, 4u);
            ASSERT_EQ(I, 15u);
            ASSERT_EQ(I / 2, 7u);

            problem.setParams().setUniformSummationOrder(true);
            EXPECT_EQ(
                solution->getSKGrid(problem, device, tiles, origami::reduction_t::parallel), tiles)
                << "I/F == 7 is below MinItersPerCU (8); F=2 must be refused and "
                   "the grid must snap down to all-full (T)";
        }
    }

    // StreamKFlagElements bounds the grid of any launch that takes the flag
    // protocol. The F-star search carries that bound as one of its
    // admissibility conditions, so the grid it selects already satisfies it.
    //
    // The three tests below pin the three regimes:
    //   * the bound binds and a smaller uniform grid exists -- take it
    //   * the bound binds and no uniform grid at or below it exists -- fall
    //     back to all-full, which is exempt because it has no partial tiles
    //   * the launch never reaches the flags -- the bound must not constrain it
    //
    // Selecting under the bound and clamping after it are not the same thing:
    // clamping rewrites tiles*F to StreamKFlagElements, which is not in general
    // a multiple of the tile count, and the resulting ragged split is then
    // refused outright (kSplitCases/FlagClampedGridNotRowUniform).

    TEST(RowUniformityGridSteering_pre_checkin, FlagRegionBoundBindsInsideTheSnap)
    {
        // T=100 (1280x1280, MT 128x128), I=256 (K=16384, DepthU=64),
        // g0 = 3200 = T*32. F=32 and F=16 both divide I and both clear
        // MinItersPerCU, so the flag bound is the only thing separating them.
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 3200;

        auto problem = uniformityGemm(1280, 1280, 16384);
        // Large enough that partialTileSize() never rejects a candidate here;
        // the bound, not the workspace, has to be what decides.
        problem.setWorkspaceSize(1ull << 30);

        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        const size_t I
            = std::max(size_t{1}, problem.getItersPerTile(solution->sizeMapping));
        ASSERT_EQ(tiles, 100u);
        ASSERT_EQ(I, 256u);
        ASSERT_EQ(I % 32, 0u) << "F=32 must be rejected by the flag bound alone";
        ASSERT_GT(3200u, static_cast<size_t>(TensileLite::StreamKFlagElements));
        ASSERT_LE(solution->partialTileSize(3200), problem.workspaceSize());

        // With the mode off the F-star search never runs, so nothing narrows g0
        // to a uniform grid. The post-selection clamp is untouched and still has
        // the last write, and it fires on exactly this shape (g0 exceeds the
        // region and tiles % g0 != 0), so the launch is handed the bound itself.
        // 2048 is not a multiple of the tile count -- it is the ragged split the
        // clamped case at the end of this test re-derives, and the shape
        // selecting under the bound exists to avoid.
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree),
                  static_cast<size_t>(TensileLite::StreamKFlagElements))
            << "mode off: no snap, so the fixed grid is cut back only by the "
               "post-selection flag clamp";

        problem.setParams().setUniformSummationOrder(true);
        const size_t grid
            = solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree);
        EXPECT_EQ(grid, 1600u)
            << "F must step down to the largest value keeping T*F inside the flag region";
        EXPECT_LE(grid, static_cast<size_t>(TensileLite::StreamKFlagElements));
        EXPECT_EQ(grid % tiles, 0u) << "the selected grid must stay a multiple of the tiles";

        // The grid is admissible on its own terms, not merely small enough.
        const TensileLite::StreamKStaticSplit split = TensileLite::streamKStaticSplit(
            tiles, I, grid, /*skFullTiles=*/1, /*forceDPOnly=*/false);
        EXPECT_TRUE(TensileLite::streamKStaticSplitRowUniform(
            split, tiles, I, grid, solution->internalArgsSupport.perTileExtraIters))
            << "grid=" << grid << " must pack row-uniformly";

        // What clamping after selection produces instead -- the mode-off grid above.
        const TensileLite::StreamKStaticSplit clamped = TensileLite::streamKStaticSplit(
            tiles, I, TensileLite::StreamKFlagElements, 1, false);
        EXPECT_FALSE(TensileLite::streamKStaticSplitRowUniform(
            clamped, tiles, I, TensileLite::StreamKFlagElements,
            solution->internalArgsSupport.perTileExtraIters))
            << "the clamped grid is not row-uniform; selecting under the bound is what "
               "keeps this shape admissible";
    }

    // The bound cannot always be met by a grid that splits tiles, because the
    // tile count itself can exceed it. That is not a fail-closed case: F=1
    // leaves tiles % grid == 0, so there are no partial tiles and no flag
    // protocol, and the grid is legal at any size. The all-full grid is
    // therefore always available as the floor of the search.
    TEST(RowUniformityGridSteering_pre_checkin, AllFullGridExemptFromFlagRegionBound)
    {
        // T=4096 (8192x8192, MT 128x128) is already twice the bound.
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 8192; // g0 = T*2

        auto problem = uniformityGemm(8192, 8192, 1024);
        problem.setWorkspaceSize(1ull << 30);

        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        const size_t I
            = std::max(size_t{1}, problem.getItersPerTile(solution->sizeMapping));
        ASSERT_EQ(tiles, 4096u);
        ASSERT_EQ(I, 16u);
        ASSERT_GT(tiles, static_cast<size_t>(TensileLite::StreamKFlagElements))
            << "the point of the case: no grid that splits tiles can fit the bound";
        ASSERT_EQ(I % 2, 0u);
        ASSERT_GE(I / 2, 8u);
        ASSERT_LE(solution->partialTileSize(8192), problem.workspaceSize())
            << "F=2 must be rejected by the flag bound, not by the workspace";

        problem.setParams().setUniformSummationOrder(true);
        const size_t grid
            = solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree);
        EXPECT_EQ(grid, tiles)
            << "with every F >= 2 outside the flag region, the search must land on all-full";
        EXPECT_EQ(tiles % grid, 0u)
            << "all-full leaves no partial tiles, so the launch takes no flag region and the "
               "bound does not apply -- the grid may legally exceed it";

        const TensileLite::StreamKStaticSplit split = TensileLite::streamKStaticSplit(
            tiles, I, grid, /*skFullTiles=*/1, /*forceDPOnly=*/false);
        EXPECT_TRUE(TensileLite::streamKStaticSplitRowUniform(
            split, tiles, I, grid, solution->internalArgsSupport.perTileExtraIters))
            << "the all-full grid must remain admissible above the bound";
    }

    // Parallel reduction is passed Flags == nullptr and skips the flag protocol,
    // so it is one of the cases the clamp deliberately leaves alone. The search
    // is conditioned on the same predicates and must leave it alone too;
    // otherwise the bound would shrink grids that have no flag region to overrun.
    TEST(RowUniformityGridSteering_pre_checkin, FlagRegionBoundDoesNotBindParallelReduction)
    {
        auto solution = uniformitySteeringSolution();
        auto device   = uniformitySteeringDevice();
        device.skDynamicGrid = 0;
        device.skFixedGrid   = 3200;

        auto problem = uniformityGemm(1280, 1280, 16384);
        problem.setWorkspaceSize(1ull << 30);

        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, 100u);
        ASSERT_GT(3200u, static_cast<size_t>(TensileLite::StreamKFlagElements));

        problem.setParams().setUniformSummationOrder(true);
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree),
                  1600u)
            << "tree takes the flag region, so the bound binds";
        EXPECT_EQ(
            solution->getSKGrid(problem, device, tiles, origami::reduction_t::parallel), 3200u)
            << "parallel never reaches the flags; the bound must not shrink its grid";
    }

} // namespace
