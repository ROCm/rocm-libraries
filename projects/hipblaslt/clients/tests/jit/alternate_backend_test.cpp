// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-backend.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt-jit.hpp>

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jit = hipblaslt_ext::experimental::jit;
namespace abi = hipblaslt_ext::experimental::jit::detail;

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }
    void hip(hipError_t status, const char* expression)
    {
        require(status == hipSuccess, std::string(expression) + ": " + hipGetErrorString(status));
    }
    void blas(hipblasStatus_t status, const char* expression)
    {
        require(status == HIPBLAS_STATUS_SUCCESS,
                std::string(expression) + ": status " + std::to_string(status));
    }
#define HIP(expression) hip((expression), #expression)
#define BLAS(expression) blas((expression), #expression)

    struct Counts
    {
        int                   compile = 0, support = 0, prepare = 0, run = 0;
        int                   loaded = 0, unloaded = 0;
        float                 compileAlpha = 0, compileBeta = 0;
        std::function<void()> onSupport;
    };

    struct Module
    {
        hipModule_t             module{};
        std::shared_ptr<Counts> counts;
        Module(const std::string& path, std::shared_ptr<Counts> c)
            : counts(std::move(c))
        {
            HIP(hipModuleLoad(&module, path.c_str()));
            ++counts->loaded;
        }
        ~Module()
        {
            if(module)
                hipModuleUnload(module);
            ++counts->unloaded;
        }
    };

    hipblasStatus_t reject(jit::Diagnostics&  diagnostics,
                           const std::string& message,
                           hipblasStatus_t    status = HIPBLAS_STATUS_NOT_SUPPORTED)
    {
        diagnostics.backend = "alternate-native";
        diagnostics.message = message;
        return status;
    }

    // These arguments are copied by value during prepare; no descriptor, request,
    // or temporary host scalar is dereferenced during run.
    struct Arguments
    {
        const float *A{}, *B{}, *C{};
        float *      D{}, *workspace{};
        int          m{}, n{}, k{}, lda{}, ldb{}, ldc{}, ldd{};
        float        alpha{}, beta{};
    };

    struct Launch final : abi::PreparedLaunch
    {
        std::shared_ptr<Module> module;
        std::shared_ptr<Counts> counts;
        hipFunction_t           initialize{}, gemm{};
        Arguments               args;

        hipblasStatus_t run(hipStream_t stream, hipEvent_t start, hipEvent_t stop) const override
        {
            ++counts->run;
            Arguments a          = args;
            void*     initArgs[] = {&a.C, &a.workspace, &a.m, &a.n, &a.ldc, &a.beta};
            void*     gemmArgs[] = {
                &a.A, &a.B, &a.workspace, &a.D, &a.m, &a.n, &a.k, &a.lda, &a.ldb, &a.ldd, &a.alpha};
            const unsigned int blocks = (a.m * a.n + 63) / 64;
            if(start && hipEventRecord(start, stream) != hipSuccess)
                return HIPBLAS_STATUS_EXECUTION_FAILED;
            if(hipModuleLaunchKernel(
                   initialize, blocks, 1, 1, 64, 1, 1, 0, stream, initArgs, nullptr)
               != hipSuccess)
                return HIPBLAS_STATUS_EXECUTION_FAILED;
            if(hipModuleLaunchKernel(gemm, blocks, 1, 1, 64, 1, 1, 0, stream, gemmArgs, nullptr)
               != hipSuccess)
                return HIPBLAS_STATUS_EXECUTION_FAILED;
            if(stop && hipEventRecord(stop, stream) != hipSuccess)
                return HIPBLAS_STATUS_EXECUTION_FAILED;
            return HIPBLAS_STATUS_SUCCESS;
        }
    };

    struct Bundle final : abi::KernelBundle
    {
        std::shared_ptr<Module> module;
        std::shared_ptr<Counts> counts;
        bool                    missingHelper;
        Bundle(std::shared_ptr<Module> m, std::shared_ptr<Counts> c, bool missing)
            : module(std::move(m))
            , counts(std::move(c))
            , missingHelper(missing)
        {
        }
        std::string_view operationKind() const noexcept override
        {
            return abi::GemmRequest::operation;
        }
        std::string name() const override
        {
            return "alternate-native-f32-nn";
        }
        std::string kernelNames() const override
        {
            return "alternate_initialize;alternate_gemm";
        }

        hipblasStatus_t support(const abi::OperationRequest& request,
                                size_t                       limit,
                                size_t&                      bytes,
                                jit::Diagnostics&            diagnostics) const override
        {
            ++counts->support;
            if(counts->onSupport)
                counts->onSupport();
            bytes            = 0;
            const auto* gemm = dynamic_cast<const abi::GemmRequest*>(&request);
            if(!gemm)
                return reject(diagnostics, "This bundle accepts GEMM requests");
            const auto& p = gemm->problem;
            if(p.trans_a != HIPBLAS_OP_N || p.trans_b != HIPBLAS_OP_N || p.m == 0 || p.n == 0
               || p.k == 0 || p.m > 32 || p.n > 32 || p.k > 32 || p.batch_count != 1
               || p.grouped_gemm || !p.strided_batch || p.a_type != HIP_R_32F
               || p.b_type != HIP_R_32F || p.c_type != HIP_R_32F || p.d_type != HIP_R_32F
               || p.compute_type != rocblaslt_compute_f32
               || (p.scale_type != HIP_R_32F && p.scale_type != HIPBLASLT_DATATYPE_INVALID)
               || p.row_stride_a != 1 || p.row_stride_b != 1 || p.row_stride_c != 1
               || p.row_stride_d != 1 || p.col_stride_a < p.m || p.col_stride_b < p.k
               || p.col_stride_c < p.m || p.col_stride_d < p.m || p.col_stride_a > 128
               || p.col_stride_b > 128 || p.col_stride_c > 128 || p.col_stride_d > 128
               || p.batch_offset_a || p.batch_offset_b || p.batch_offset_c || p.batch_offset_d
               || p.bias || p.E || p.amaxD || p.gradient || p.swizzleA || p.swizzleB || p.scaleA
               || p.scaleB || p.scaleC || p.scaleD || p.scaleE || p.scaleAlphaVec
               || static_cast<int>(p.epilogue) != HIPBLASLT_EPILOGUE_DEFAULT)
                return reject(diagnostics,
                              "Only bounded batch-one FP32 column-major NN GEMM is supported");
            if(!p.A || !p.B || !p.C || !p.D || !p.alpha || !p.beta)
                return reject(
                    diagnostics, "Missing buffer or scalar", HIPBLAS_STATUS_INVALID_VALUE);
            bytes = p.m * p.n * sizeof(float);
            if(bytes > limit)
                return reject(diagnostics, "Insufficient workspace", HIPBLAS_STATUS_INVALID_VALUE);
            return HIPBLAS_STATUS_SUCCESS;
        }

        hipblasStatus_t prepare(const abi::OperationRequest&                request,
                                const abi::ExecutionContext&                execution,
                                std::shared_ptr<const abi::PreparedLaunch>& output,
                                jit::Diagnostics& diagnostics) const override
        {
            ++counts->prepare;
            output.reset();
            size_t bytes  = 0;
            auto   status = support(request, execution.workspaceBytes, bytes, diagnostics);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
            if(!execution.workspace)
                return reject(diagnostics, "Null workspace", HIPBLAS_STATUS_INVALID_VALUE);
            auto launch    = std::make_shared<Launch>();
            launch->module = module;
            launch->counts = counts;
            // Resolve BOTH symbols before returning any launch object or touching GPU memory.
            const auto helperStatus = hipModuleGetFunction(
                &launch->initialize,
                module->module,
                missingHelper ? "deliberately_absent_helper" : "alternate_initialize");
            const auto mainStatus
                = hipModuleGetFunction(&launch->gemm, module->module, "alternate_gemm");
            if(helperStatus != hipSuccess || mainStatus != hipSuccess)
                return reject(diagnostics,
                              "A required native symbol is missing",
                              HIPBLAS_STATUS_EXECUTION_FAILED);
            const auto& p = dynamic_cast<const abi::GemmRequest&>(request).problem;
            auto&       a = launch->args;
            a.A           = static_cast<const float*>(p.A);
            a.B           = static_cast<const float*>(p.B);
            a.C           = static_cast<const float*>(p.C);
            a.D           = static_cast<float*>(p.D);
            a.workspace   = static_cast<float*>(execution.workspace);
            a.m           = p.m;
            a.n           = p.n;
            a.k           = p.k;
            a.lda         = p.col_stride_a;
            a.ldb         = p.col_stride_b;
            a.ldc         = p.col_stride_c;
            a.ldd         = p.col_stride_d;
            std::memcpy(&a.alpha, p.alpha, sizeof(float));
            std::memcpy(&a.beta, p.beta, sizeof(float));
            output = std::move(launch);
            return HIPBLAS_STATUS_SUCCESS;
        }
    };

    // A second operation kind proves the generic compiler path does not require a
    // GEMM request. This operation has no public execution adapter and submits no work.
    struct ProbeRequest final : abi::OperationRequest
    {
        std::string_view kind() const noexcept override
        {
            return "test.alternate.probe.v1";
        }
    };
    struct ProbeBundle final : abi::KernelBundle
    {
        std::string_view operationKind() const noexcept override
        {
            return "test.alternate.probe.v1";
        }
        std::string name() const override
        {
            return "alternate-probe";
        }
        std::string kernelNames() const override
        {
            return {};
        }
        hipblasStatus_t support(const abi::OperationRequest& request,
                                size_t,
                                size_t&           bytes,
                                jit::Diagnostics& diagnostics) const override
        {
            bytes = 0;
            return dynamic_cast<const ProbeRequest*>(&request)
                       ? HIPBLAS_STATUS_SUCCESS
                       : reject(diagnostics, "Probe request required");
        }
        hipblasStatus_t prepare(const abi::OperationRequest&,
                                const abi::ExecutionContext&,
                                std::shared_ptr<const abi::PreparedLaunch>& output,
                                jit::Diagnostics& diagnostics) const override
        {
            output.reset();
            return reject(diagnostics, "Probe has no execution adapter");
        }
    };

    enum class Mode
    {
        Native,
        MissingHelper,
        Unsupported,
        Throwing
    };
    struct Backend final : abi::BackendImplementation
    {
        std::string             path;
        std::shared_ptr<Counts> counts;
        Mode                    mode;
        Backend(std::string p, std::shared_ptr<Counts> c, Mode m)
            : path(std::move(p))
            , counts(std::move(c))
            , mode(m)
        {
        }
        std::string_view name() const noexcept override
        {
            return "alternate-native";
        }
        hipblasStatus_t compile(const abi::OperationRequest& request,
                                const abi::Target&           target,
                                size_t,
                                std::shared_ptr<const abi::KernelBundle>& output,
                                jit::Diagnostics& diagnostics) const override
        {
            ++counts->compile;
            output.reset();
            if(mode == Mode::Throwing)
                throw std::runtime_error("deliberate alternate compile exception");
            if(mode == Mode::Unsupported)
                return reject(diagnostics, "deliberate alternate rejection");
            if(dynamic_cast<const ProbeRequest*>(&request))
            {
                output = std::make_shared<ProbeBundle>();
                return HIPBLAS_STATUS_SUCCESS;
            }
            const auto* gemm = dynamic_cast<const abi::GemmRequest*>(&request);
            if(!gemm)
                return reject(diagnostics, "Unrecognized operation request");
            int current = -1;
            HIP(hipGetDevice(&current));
            require(current == target.device, "Generic compile passed a different device");
            std::memcpy(&counts->compileAlpha, gemm->problem.alpha, sizeof(float));
            std::memcpy(&counts->compileBeta, gemm->problem.beta, sizeof(float));
            output = std::make_shared<Bundle>(
                std::make_shared<Module>(path, counts), counts, mode == Mode::MissingHelper);
            return HIPBLAS_STATUS_SUCCESS;
        }
    };

    jit::Backend backend(const std::string&             path,
                         const std::shared_ptr<Counts>& counts,
                         Mode                           mode = Mode::Native)
    {
        return abi::BackendAccess::make(std::make_shared<Backend>(path, counts, mode));
    }

    struct DeviceFloats
    {
        float* pointer{};
        size_t count;
        explicit DeviceFloats(size_t n)
            : count(n)
        {
            HIP(hipMalloc(&pointer, n * sizeof(float)));
        }
        ~DeviceFloats()
        {
            if(pointer)
                hipFree(pointer);
        }
        DeviceFloats(const DeviceFloats&) = delete;
        void put(const std::vector<float>& values)
        {
            require(values.size() == count, "Wrong upload size");
            HIP(hipMemcpy(pointer, values.data(), count * sizeof(float), hipMemcpyHostToDevice));
        }
        std::vector<float> get() const
        {
            std::vector<float> result(count);
            HIP(hipMemcpy(result.data(), pointer, count * sizeof(float), hipMemcpyDeviceToHost));
            return result;
        }
        void fill(float value)
        {
            put(std::vector<float>(count, value));
        }
    };

    struct Problem
    {
        static constexpr int    m = 7, n = 5, k = 3;
        static constexpr int    lda = 9, ldb = 6, ldc = 11, ldd = 13;
        static constexpr float  sentinel = -913.25f;
        hipblasLtHandle_t       handle{};
        hipblasLtMatmulDesc_t   desc{};
        hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
        hipStream_t             stream{};
        hipEvent_t              start{}, stop{};
        DeviceFloats       A{lda * k}, A2{lda * k}, B{ldb * n}, C{ldc * n}, D{ldd * n}, W{m * n};
        std::vector<float> a, a2, b, c;

        Problem()
            : a(lda * k, sentinel)
            , a2(lda * k, sentinel)
            , b(ldb * n, sentinel)
            , c(ldc * n, sentinel)
        {
            BLAS(hipblasLtCreate(&handle));
            BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_32F, m, k, lda));
            BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_32F, k, n, ldb));
            BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, m, n, ldc));
            BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_32F, m, n, ldd));
            HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            HIP(hipEventCreate(&start));
            HIP(hipEventCreate(&stop));
            for(int p = 0; p < k; ++p)
                for(int row = 0; row < m; ++row)
                {
                    a[row + p * lda]  = (row - 2 * p - 3) * 0.25f;
                    a2[row + p * lda] = (2 * row + p - 6) * 0.125f;
                }
            for(int col = 0; col < n; ++col)
                for(int p = 0; p < k; ++p)
                    b[p + col * ldb] = (3 * p - col - 1) * 0.25f;
            for(int col = 0; col < n; ++col)
                for(int row = 0; row < m; ++row)
                    c[row + col * ldc] = (row - 2 * col + 1) * 0.5f;
            A.put(a);
            A2.put(a2);
            B.put(b);
            C.put(c);
            reset();
        }
        ~Problem()
        {
            hipEventDestroy(start);
            hipEventDestroy(stop);
            hipStreamDestroy(stream);
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatrixLayoutDestroy(ld);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }
        void reset()
        {
            D.fill(sentinel);
            W.fill(sentinel);
        }
        jit::Request request(float& alpha, float& beta)
        {
            jit::Request     result;
            jit::Diagnostics d;
            BLAS(jit::makeGemmRequest(handle,
                                      desc,
                                      &alpha,
                                      A.pointer,
                                      la,
                                      B.pointer,
                                      lb,
                                      &beta,
                                      C.pointer,
                                      lc,
                                      D.pointer,
                                      ld,
                                      result,
                                      d));
            return result;
        }
        hipblasStatus_t call(const hipblasLtMatmulAlgo_t& algo,
                             float                        alpha,
                             float                        beta,
                             bool                         changedA       = false,
                             size_t                       workspaceBytes = m * n * sizeof(float))
        {
            return hipblasLtMatmul(handle,
                                   desc,
                                   &alpha,
                                   changedA ? A2.pointer : A.pointer,
                                   la,
                                   B.pointer,
                                   lb,
                                   &beta,
                                   C.pointer,
                                   lc,
                                   D.pointer,
                                   ld,
                                   &algo,
                                   W.pointer,
                                   workspaceBytes,
                                   stream);
        }
        void verify(float alpha, float beta, bool changedA = false)
        {
            HIP(hipStreamSynchronize(stream));
            const auto  actual    = D.get();
            const auto  workspace = W.get();
            const auto& sourceA   = changedA ? a2 : a;
            for(int col = 0; col < n; ++col)
                for(int row = 0; row < ldd; ++row)
                {
                    double expected = sentinel;
                    if(row < m)
                    {
                        double sum = 0;
                        for(int p = 0; p < k; ++p)
                            sum += double(sourceA[row + p * lda]) * b[p + col * ldb];
                        expected = alpha * sum + beta * c[row + col * ldc];
                        require(workspace[row + col * m] == beta * c[row + col * ldc],
                                "Helper did not initialize workspace from C's actual leading "
                                "dimension");
                    }
                    require(std::isfinite(actual[row + col * ldd])
                                && std::abs(actual[row + col * ldd] - expected) < 1e-5,
                            "Incorrect output or overwritten D padding at (" + std::to_string(row)
                                + ", " + std::to_string(col) + ")");
                }
            require(C.get() == c && B.get() == b && A.get() == a && A2.get() == a2,
                    "An input matrix was modified");
        }
        void untouched()
        {
            HIP(hipStreamSynchronize(stream));
            require(D.get() == std::vector<float>(ldd * n, sentinel), "Failure changed D");
            require(W.get() == std::vector<float>(m * n, sentinel), "Failure changed workspace");
        }
    };

    void test(const std::string& path)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        Problem          p;
        constexpr size_t bytes    = Problem::m * Problem::n * sizeof(float);
        auto             counts   = std::make_shared<Counts>();
        auto             provider = backend(path, counts);
        float            alpha = 1.25f, beta = -0.5f;
        auto             request = p.request(alpha, beta);
        alpha                    = 81;
        beta                     = 92; // makeGemmRequest must own the original host scalars.
        jit::Solution    solution;
        jit::Diagnostics diagnostics;
        const auto       initial
            = jit::getJitAlgo(device, request, provider, bytes, solution, diagnostics);
        require(initial == HIPBLAS_STATUS_SUCCESS,
                "Initial compile: " + diagnostics.message + " (status " + std::to_string(initial)
                    + ")");
        require(counts->compileAlpha == 1.25f && counts->compileBeta == -0.5f,
                "makeGemmRequest retained the caller's mutable scalar pointers");
        hipblasLtMatmulHeuristicResult_t first{}, second{};
        BLAS(jit::getGemmAlgo(solution, first, diagnostics));
        BLAS(jit::getGemmAlgo(solution, second, diagnostics));
        require(first.workspaceSize == bytes, "Incorrect workspace size");
        require(std::memcmp(&first.algo, &second.algo, sizeof(first.algo)) == 0,
                "Repeated adaptation changed algorithm identity");
        auto                                       copiedAlgo = first.algo;
        std::weak_ptr<const abi::CompiledSolution> retained   = abi::SolutionAccess::get(solution);
        provider                                              = {};
        request                                               = {};
        solution                                              = {};
        require(!retained.expired() && counts->unloaded == 0,
                "Adapted algorithm lost its module after public owners were destroyed");
        require(hipblaslt_ext::getSolutionNameFromAlgo(p.handle, copiedAlgo)
                    == "alternate-native-f32-nn",
                "Solution name lookup did not reach the alternate backend");
        require(hipblaslt_ext::getKernelNameFromAlgo(p.handle, copiedAlgo)
                    == "alternate_initialize;alternate_gemm",
                "Kernel name lookup did not reach the alternate backend");
        counts->onSupport = [&] {
            require(hipblaslt_ext::getKernelNameFromAlgo(p.handle, copiedAlgo)
                        == "alternate_initialize;alternate_gemm",
                    "Provider callback could not reenter registry name lookup");
        };
        BLAS(p.call(copiedAlgo, 1.25f, -0.5f));
        counts->onSupport = {};
        p.verify(1.25f, -0.5f);
        std::cout
            << "PASS C GEMM, independent module, helper workspace, padded layouts, owned scalars\n";

        p.reset();
        BLAS(p.call(copiedAlgo, -0.75f, 0.25f, true));
        p.verify(-0.75f, 0.25f, true);
        require(counts->compile == 1, "Changing execution arguments recompiled the bundle");
        std::cout
            << "PASS copied algorithm, owner destruction, changed pointer/scalars, bundle reuse\n";

        // Prebuilt Tensile handles have 64 Stream-K flag slots. This provider never
        // uses them; crossing that count must not introduce a backend restriction.
        struct Streams
        {
            Problem&                 problem;
            hipStream_t              original;
            std::vector<hipStream_t> values;
            explicit Streams(Problem& p)
                : problem(p)
                , original(p.stream)
            {
            }
            ~Streams()
            {
                problem.stream = original;
                for(auto stream : values)
                    hipStreamDestroy(stream);
            }
        } streams(p);
        for(int i = 0; i < 65; ++i)
        {
            hipStream_t stream;
            HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            streams.values.push_back(stream);
            p.stream = stream;
            BLAS(p.call(copiedAlgo, 1.25f, -0.5f));
            HIP(hipStreamSynchronize(stream));
        }
        p.verify(1.25f, -0.5f);
        p.stream = streams.original;
        std::cout << "PASS alternate backend does not consume Tensile Stream-K stream slots\n";

        p.reset();
        const int preparesBefore = counts->prepare, runsBefore = counts->run;
        require(p.call(copiedAlgo, 1, 1, false, bytes - 1) == HIPBLAS_STATUS_INVALID_VALUE,
                "Insufficient workspace returned an unexpected status");
        p.untouched();
        require(counts->prepare == preparesBefore && counts->run == runsBefore,
                "Insufficient workspace reached preparation or execution");
        std::cout << "PASS insufficient workspace rejected before GPU submission\n";

        hipblaslt_ext::Gemm gemm(p.handle,
                                 HIPBLAS_OP_N,
                                 HIPBLAS_OP_N,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIPBLAS_COMPUTE_32F);
        {
            hipblaslt_ext::GemmInputs inputs;
            inputs.setA(p.A.pointer);
            inputs.setB(p.B.pointer);
            inputs.setC(p.C.pointer);
            inputs.setD(p.D.pointer);
            float a = 0.75f, b = -0.25f;
            inputs.setAlpha(&a);
            inputs.setBeta(&b);
            hipblaslt_ext::GemmEpilogue    epilogue;
            hipblaslt_ext::GemmProblemType type(HIPBLAS_OP_N,
                                                HIPBLAS_OP_N,
                                                HIP_R_32F,
                                                HIP_R_32F,
                                                HIP_R_32F,
                                                HIP_R_32F,
                                                HIPBLAS_COMPUTE_32F);
            BLAS(gemm.setProblem(p.m,
                                 p.n,
                                 p.k,
                                 1,
                                 p.lda,
                                 p.ldb,
                                 p.ldc,
                                 p.ldd,
                                 p.lda * p.k,
                                 p.ldb * p.n,
                                 p.ldc * p.n,
                                 p.ldd * p.n,
                                 epilogue,
                                 inputs,
                                 type));
            a = 91;
            b = 92;
        }
        size_t dimensionWorkspace = 0;
        BLAS(gemm.isAlgoSupported(copiedAlgo, dimensionWorkspace));
        require(dimensionWorkspace == bytes, "Dimension-based workspace mismatch");
        gemm.setMaxWorkspaceBytes(bytes);
        BLAS(gemm.initialize(copiedAlgo, p.W.pointer, false, p.stream));
        BLAS(gemm.run(p.stream));
        p.verify(0.75f, -0.25f);
        std::cout << "PASS dimension-based C++ GEMM retains host scalar values\n";
        {
            float localAlpha = 0.5f, localBeta = -1.25f;
            BLAS(gemm.setProblem(p.desc,
                                 &localAlpha,
                                 p.A2.pointer,
                                 p.la,
                                 p.B.pointer,
                                 p.lb,
                                 &localBeta,
                                 p.C.pointer,
                                 p.lc,
                                 p.D.pointer,
                                 p.ld));
            localAlpha = -999;
            localBeta  = 999;
        }
        size_t needed = 0;
        BLAS(gemm.isAlgoSupported(copiedAlgo, needed));
        require(needed == bytes, "C++ support returned an incorrect workspace size");
        gemm.setMaxWorkspaceBytes(bytes);
        BLAS(gemm.initialize(copiedAlgo, p.W.pointer, false, p.stream));
        BLAS(gemm.run(p.stream, p.start, p.stop));
        HIP(hipEventSynchronize(p.stop));
        float elapsed = -1;
        HIP(hipEventElapsedTime(&elapsed, p.start, p.stop));
        require(elapsed >= 0, "Run did not record the supplied events");
        p.verify(0.5f, -1.25f, true);
        require(gemm.getSolutionName() == "alternate-native-f32-nn", "C++ name dispatch failed");
        require(counts->compile == 1, "C++ execution recompiled the bundle");
        std::cout << "PASS C++ setProblem/support/initialize/run, scalar lifetime, events, name "
                     "dispatch\n";

        alpha              = 1;
        beta               = 1;
        request            = p.request(alpha, beta);
        auto failureCounts = std::make_shared<Counts>();
        auto missing       = backend(path, failureCounts, Mode::MissingHelper);
        BLAS(jit::getJitAlgo(device, request, missing, bytes, solution, diagnostics));
        BLAS(jit::getGemmAlgo(solution, first, diagnostics));
        p.reset();
        require(p.call(first.algo, 1, 1) == HIPBLAS_STATUS_EXECUTION_FAILED,
                "Missing helper returned an unexpected status");
        p.untouched();
        require(failureCounts->prepare == 1 && failureCounts->run == 0,
                "Missing helper reached execution");
        require(gemm.initialize(first.algo, p.W.pointer, false, p.stream)
                    == HIPBLAS_STATUS_EXECUTION_FAILED,
                "C++ missing helper returned an unexpected status");
        p.untouched();
        BLAS(gemm.run(p.stream));
        p.verify(0.5f, -1.25f, true);
        require(failureCounts->run == 0, "Failed preparation replaced the previous launch");
        BLAS(gemm.setProblem(p.desc,
                             &alpha,
                             p.A.pointer,
                             p.la,
                             p.B.pointer,
                             p.lb,
                             &beta,
                             p.C.pointer,
                             p.lc,
                             p.D.pointer,
                             p.ld));
        require(gemm.run(p.stream) == HIPBLAS_STATUS_NOT_INITIALIZED,
                "setProblem retained a launch bound to previous buffers/scalars");
        std::cout << "PASS missing helper rejected before workspace initialization or GEMM\n";

        provider         = backend(path, counts);
        const auto probe = abi::RequestAccess::make(std::make_shared<ProbeRequest>());
        BLAS(jit::getJitAlgo(device, probe, provider, 0, solution, diagnostics));
        require(jit::getGemmAlgo(solution, first, diagnostics) == HIPBLAS_STATUS_NOT_SUPPORTED
                    && first.state == HIPBLAS_STATUS_NOT_SUPPORTED,
                "A non-GEMM operation was adapted to GEMM");
        std::cout
            << "PASS non-GEMM operation compiled through generic API; GEMM adapter rejects it\n";

        for(const auto mode : {Mode::Unsupported, Mode::Throwing})
        {
            BLAS(jit::getJitAlgo(device, probe, provider, 0, solution, diagnostics));
            const auto status = jit::getJitAlgo(
                device, request, backend(path, counts, mode), bytes, solution, diagnostics);
            require(status
                        == (mode == Mode::Unsupported ? HIPBLAS_STATUS_NOT_SUPPORTED
                                                      : HIPBLAS_STATUS_INTERNAL_ERROR),
                    "Compile rejection/exception returned an unexpected status");
            require(!abi::SolutionAccess::get(solution), "Compile failure left a stale solution");
            require(diagnostics.backend == "alternate-native"
                        && diagnostics.message.find("deliberate alternate") != std::string::npos,
                    "Compile failure lost its backend diagnostic");
        }
        std::cout << "PASS rejected/throwing backend clears solution and preserves diagnostics\n";

        auto releasedCounts = std::make_shared<Counts>();
        {
            auto          temporaryBackend = backend(path, releasedCounts);
            jit::Solution temporarySolution;
            BLAS(jit::getJitAlgo(
                device, request, temporaryBackend, bytes, temporarySolution, diagnostics));
            require(releasedCounts->loaded == 1 && releasedCounts->unloaded == 0,
                    "Unadapted module was not owned by its solution");
        }
        require(releasedCounts->unloaded == 1, "Unadapted solution retained its module");
        require(!retained.expired(), "Adapted solution did not retain its module");
        std::cout
            << "PASS unadapted module released; adapted module retained for copied algorithms\n";
        std::cout << "ALL ALTERNATE BACKEND CHECKS PASSED\n";
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr
            << "Usage: alternate_backend_test /absolute/path/alternate_backend_kernels.hsaco\n";
        return 2;
    }
    try
    {
        test(argv[1]);
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
