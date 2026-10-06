// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-mode.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt_secure_env.hpp"
#include "hipblaslt-jit-replay.hpp"
#include "rocblaslt.h"
#include "rocblaslt_arch_revision.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace
{
    // A null stream and the legacy stream are not capturing. Querying the null
    // stream while another stream is capturing is an error, so only this stream
    // is checked.
    bool streamIsCapturing(hipStream_t stream)
    {
        if(stream == nullptr || stream == hipStreamLegacy)
            return false;
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        if(hipStreamIsCapturing(stream, &status) != hipSuccess)
            return false;
        return status != hipStreamCaptureStatusNone;
    }

    std::string captureSkipMessage(size_t m, size_t n, size_t k)
    {
        return "JIT generation skipped during stream capture for GEMM M=" + std::to_string(m)
               + " N=" + std::to_string(n) + " K=" + std::to_string(k);
    }
}

namespace hipblaslt_jit
{
    Status DeviceTarget::make(int device, DeviceTarget& target)
    {
        target        = {};
        target.device = device;
        const auto error = hipGetDeviceProperties(&target.properties, device);
        if(error != hipSuccess)
            return {Status::Code::Failed,
                    Stage::Configure,
                    std::string("Cannot query HIP device properties: ") + hipGetErrorString(error)};
        target.targetId      = target.properties.gcnArchName;
        target.isa           = target.targetId.substr(0, target.targetId.find(':'));
        target.libraryArch   = rocblaslt_revisioned_arch_name(target.isa,
                                                            target.properties.asicRevision);
        target.wavefrontSize = target.properties.warpSize;
        target.cuCount       = target.properties.multiProcessorCount;
        target.hardware      = TensileLite::hip::GetDevice(target.properties, device);
        return {};
    }

    JitLibrary& JitLibrary::process()
    {
        static auto* library = new JitLibrary(defaultRoot(), [](int device, CacheKey& key) {
            DeviceTarget target;
            auto         status = DeviceTarget::make(device, target);
            if(status.ok())
                key = CacheKey::make(target, BackendInfo{}, jitCodeObjectVersion);
            return status;
        });
        return *library;
    }

    namespace
    {
        class LibraryStore final : public SolutionStore
        {
        public:
            LibraryStore(JitLibrary& library, BackendInfo backend, int codeObjectVersion)
                : m_library(library)
                , m_backend(std::move(backend))
                , m_codeObjectVersion(codeObjectVersion)
            {
            }

            Status lookup(const OperationRequest&         request,
                          const DeviceTarget&             target,
                          size_t                          count,
                          size_t                          maxWorkspaceBytes,
                          const std::vector<std::string>& excludeKernels,
                          std::vector<int32_t>&           indices) const override
            {
                indices.clear();
                const auto* gemm = gemmOf(request);
                if(!gemm || !target.hardware)
                    return unsupported(Stage::Lookup, target);
                auto problem = lowerForJit(*gemm);
                problem.setWorkspaceSize(maxWorkspaceBytes);
                return m_library.lookup(CacheKey::make(target, m_backend, m_codeObjectVersion),
                                        target.device,
                                        problem,
                                        *target.hardware,
                                        count,
                                        excludeKernels,
                                        indices);
            }

            Status publish(const OperationRequest&   request,
                           const DeviceTarget&       target,
                           const SupportedSolutions& solutions,
                           std::vector<int32_t>&     indices) const override
            {
                indices.clear();
                const auto* gemm = gemmOf(request);
                if(!gemm || !target.hardware)
                    return unsupported(Stage::Publish, target);
                return m_library.publish(CacheKey::make(target, m_backend, m_codeObjectVersion),
                                         target.device,
                                         lowerForJit(*gemm),
                                         solutions,
                                         indices);
            }

        private:
            using GemmRequest = hipblaslt_ext::experimental::jit::detail::GemmRequest;

            static const GemmRequest* gemmOf(const OperationRequest& request)
            {
                return request.kind() == GemmRequest::operation
                           ? dynamic_cast<const GemmRequest*>(&request)
                           : nullptr;
            }
            static Status unsupported(Stage stage, const DeviceTarget& target)
            {
                return {Status::Code::NotSupported,
                        stage,
                        target.hardware ? "The JIT solution library holds GEMM solutions"
                                        : "The device has no TensileLite hardware description"};
            }

            JitLibrary& m_library;
            BackendInfo m_backend;
            int         m_codeObjectVersion;
        };
    }

    std::shared_ptr<const SolutionStore>
        makeLibraryStore(JitLibrary& library, const BackendInfo& backend, int codeObjectVersion)
    {
        return std::make_shared<const LibraryStore>(library, backend, codeObjectVersion);
    }
}

namespace hipblaslt_ext::experimental
{
    namespace jit::detail
    {
        namespace
        {
            uint64_t processId()
            {
#ifdef _WIN32
                return _getpid();
#else
                return getpid();
#endif
            }
            struct Registry
            {
                std::mutex                                                            mutex;
                std::unordered_map<uint64_t, std::shared_ptr<const CompiledSolution>> entries;
                std::unordered_map<const CompiledSolution*, uint64_t>                 tokens;
                std::mt19937_64 random{std::random_device{}()};
            };
            Registry& registry()
            {
                // Copied algorithms and captured graphs can outlive their creator.
                static auto* instance = new Registry;
                return *instance;
            }

            hipblasStatus_t toHipStatus(hipblaslt_jit::Status::Code code)
            {
                using Code = hipblaslt_jit::Status::Code;
                switch(code)
                {
                case Code::Success:
                    return HIPBLAS_STATUS_SUCCESS;
                case Code::NotSupported:
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                case Code::TargetMismatch:
                    return HIPBLAS_STATUS_ARCH_MISMATCH;
                default:
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                }
            }

            // support() plus the workspace bound supportJit applies to a stored algorithm.
            hipblasStatus_t acceptBundle(const KernelBundle&     bundle,
                                         const OperationRequest& request,
                                         size_t                  workspaceLimit,
                                         size_t&                 workspaceBytes,
                                         Diagnostics&            diagnostics)
            {
                workspaceBytes = 0;
                size_t     required = 0;
                const auto status = bundle.support(request, workspaceLimit, required, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                if(required > workspaceLimit)
                    return HIPBLAS_STATUS_INVALID_VALUE;
                workspaceBytes = required;
                return HIPBLAS_STATUS_SUCCESS;
            }
        }

        uint64_t registerBundle(std::shared_ptr<const CompiledSolution> bundle)
        {
            auto&                       r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            auto                        existing = r.tokens.find(bundle.get());
            if(existing != r.tokens.end())
                return existing->second;
            uint64_t token;
            do
                token = r.random() & ((uint64_t{1} << 56) - 1);
            while(token == 0 || r.entries.count(token));
            r.entries.emplace(token, bundle);
            try
            {
                r.tokens.emplace(bundle.get(), token);
            }
            catch(...)
            {
                r.entries.erase(token);
                throw;
            }
            return token;
        }

        std::shared_ptr<const CompiledSolution> resolveJitAlgo(const rocblaslt_matmul_algo& algo,
                                                               int                          device)
        {
            if(!experimental::detail::isJitAlgo(algo))
                return {};
            int index;
            std::memcpy(&index, algo.data, sizeof(index));
            uint64_t token = 0;
            std::memcpy(&token, algo.data_pad, sizeof(algo.data_pad));
            std::shared_ptr<const CompiledSolution> entry;
            {
                auto&                       r = registry();
                std::lock_guard<std::mutex> lock(r.mutex);
                auto                        found = r.entries.find(token);
                if(algo.fallback || found == r.entries.end()
                   || index != found->second->bundle->solutionIndex())
                    throw std::invalid_argument("Unknown process-local JIT algorithm");
                entry = found->second;
            }
            int current = -1;
            if(entry->process != processId() || entry->target.device != device
               || hipGetDevice(&current) != hipSuccess || current != device)
                throw std::invalid_argument("JIT algorithm belongs to another process or device");
            return entry;
        }

        rocblaslt_status toRocStatus(hipblasStatus_t status)
        {
            switch(status)
            {
            case HIPBLAS_STATUS_SUCCESS:
                return rocblaslt_status_success;
            case HIPBLAS_STATUS_NOT_INITIALIZED:
                return rocblaslt_status_not_initialized;
            case HIPBLAS_STATUS_ALLOC_FAILED:
                return rocblaslt_status_memory_error;
            case HIPBLAS_STATUS_INVALID_VALUE:
                return rocblaslt_status_invalid_value;
            case HIPBLAS_STATUS_ARCH_MISMATCH:
                return rocblaslt_status_arch_mismatch;
            case HIPBLAS_STATUS_NOT_SUPPORTED:
                return rocblaslt_status_not_supported;
            case HIPBLAS_STATUS_EXECUTION_FAILED:
                return rocblaslt_status_execution_failed;
            default:
                return rocblaslt_status_internal_error;
            }
        }

        template <class F>
        rocblaslt_status invoke(F&& f)
        {
            try
            {
                return toRocStatus(f());
            }
            catch(const std::bad_alloc&)
            {
                return rocblaslt_status_memory_error;
            }
            catch(const std::invalid_argument&)
            {
                return rocblaslt_status_invalid_value;
            }
            catch(...)
            {
                return rocblaslt_status_internal_error;
            }
        }

        struct PreparedQuery
        {
            std::shared_ptr<const hipblaslt_jit::Jit> jit;
            std::shared_ptr<const OperationRequest>   operation;
            hipblaslt_jit::DeviceTarget               target;
        };

        // The device and request prologue shared by getJitAlgo and getLibraryAlgos.
        hipblasStatus_t prepareQuery(int                device,
                                     const Request&     request,
                                     const Backend&     backend,
                                     PreparedQuery&     prepared,
                                     Diagnostics&       diagnostics)
        {
            prepared.jit       = BackendAccess::get(backend);
            prepared.operation = RequestAccess::get(request);
            if(!prepared.operation || !prepared.jit)
            {
                diagnostics.message = "A valid request and backend are required";
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            diagnostics.backend = prepared.jit->components().backend->info().name;
            int current         = -1;
            if(hipGetDevice(&current) != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            if(current != device)
            {
                diagnostics.message = "Compile on the requested HIP device";
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            auto status = hipblaslt_jit::DeviceTarget::make(device, prepared.target);
            if(!status.ok())
            {
                diagnostics.message = std::move(status.message);
                return toHipStatus(status.code);
            }
            return HIPBLAS_STATUS_SUCCESS;
        }

        size_t publishedWorkspace(const hipblaslt_jit::DeviceTarget& target,
                                  const OperationRequest&            request,
                                  int32_t                            index)
        {
            const auto* gemm = dynamic_cast<const GemmRequest*>(&request);
            if(!gemm || !target.hardware || !hipblaslt_jit::isJitIndex(index))
                return 0;
            hipblaslt_jit::Status why;
            auto                  solution = hipblaslt_jit::JitLibrary::process().solutionByIndex(
                target.device, *target.hardware, index, why);
            if(!solution)
                return 0;
            auto problem = hipblaslt_jit::lowerForJit(*gemm);
            return solution->requiredWorkspaceSize(problem, *target.hardware);
        }

        // Look up the JIT solution library, then publish anything still missing.
        // A capturing stream may return a hit and does not start a build.
        hipblasStatus_t
            lookupThenPublish(const std::shared_ptr<const hipblaslt_jit::Jit>& jit,
                              const std::shared_ptr<const OperationRequest>&   operation,
                              const hipblaslt_jit::DeviceTarget&               target,
                              size_t                                           count,
                              size_t                                           workspaceLimit,
                              const std::vector<std::string>&                  excludeKernels,
                              std::vector<int32_t>&                            indices,
                              Diagnostics&                                     diagnostics)
        {
            indices.clear();
            if(!jit || !operation)
            {
                diagnostics.message = "A valid request and backend are required";
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            auto components = jit->components();
            if(!components.store)
                components.store = hipblaslt_jit::makeLibraryStore(
                    hipblaslt_jit::JitLibrary::process(),
                    components.backend->info(),
                    hipblaslt_jit::jitCodeObjectVersion);
            auto status = components.store->lookup(
                *operation, target, count, workspaceLimit, excludeKernels, indices);
            const auto  found = indices.size();
            const auto* gemm  = dynamic_cast<const GemmRequest*>(operation.get());
            const bool  capturing = gemm && streamIsCapturing(gemm->problem.stream);
            if(status.ok() && found < count && capturing)
            {
                if(indices.empty())
                {
                    diagnostics.message = captureSkipMessage(
                        gemm->problem.m, gemm->problem.n, gemm->problem.k);
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
            }
            else if(status.ok() && found < count)
            {
                std::vector<std::string> exclude = excludeKernels;
                auto&                    library = hipblaslt_jit::JitLibrary::process();
                for(auto index : indices)
                {
                    hipblaslt_jit::Status why;
                    if(auto solution
                       = library.solutionByIndex(target.device, *target.hardware, index, why))
                        if(!hipblaslt_jit::excludedKernel(exclude, solution->kernelName))
                            exclude.push_back(solution->kernelName);
                }
                const hipblaslt_jit::Jit generator(std::move(components));
                auto outcome = generator.generate(
                    *operation, target, count - found, workspaceLimit, exclude);
                for(auto index : outcome.indices)
                    if(std::find(indices.begin(), indices.end(), index) == indices.end())
                        indices.push_back(index);
                if(outcome.indices.empty())
                    status = outcome.failures.empty()
                                 ? hipblaslt_jit::Status{hipblaslt_jit::Status::Code::Failed,
                                                         hipblaslt_jit::Stage::Publish,
                                                         "Nothing was published"}
                                 : std::move(outcome.failures.front());
                else
                    diagnostics.message = std::move(outcome.summary);
            }
            if(!status.ok())
            {
                diagnostics.message = std::move(status.message);
                if(indices.empty())
                    return toHipStatus(status.code);
            }
            else if(diagnostics.message.empty())
                diagnostics.message = std::to_string(found) + " of " + std::to_string(indices.size())
                                      + " solutions came from the JIT solution library";
            return HIPBLAS_STATUS_SUCCESS;
        }

        rocblaslt_status supportJit(rocblaslt_handle             handle,
                                    const rocblaslt_matmul_algo& algo,
                                    const GemmRequest&           request,
                                    size_t&                      workspaceBytes)
        {
            workspaceBytes = 0;
            return invoke([&] {
                auto        entry = resolveJitAlgo(algo, handle->device);
                Diagnostics diagnostics;
                return acceptBundle(
                    *entry->bundle, request, algo.max_workspace_bytes, workspaceBytes, diagnostics);
            });
        }
    }

    namespace jit
    {
        hipblasStatus_t makeGemmRequest(hipblasLtHandle_t       handle,
                                        hipblasLtMatmulDesc_t   desc,
                                        const void*             alpha,
                                        const void*             A,
                                        hipblasLtMatrixLayout_t matA,
                                        const void*             B,
                                        hipblasLtMatrixLayout_t matB,
                                        const void*             beta,
                                        const void*             C,
                                        hipblasLtMatrixLayout_t matC,
                                        void*                   D,
                                        hipblasLtMatrixLayout_t matD,
                                        Request&                request,
                                        Diagnostics&            diagnostics)
        {
            request     = {};
            diagnostics = {};
            try
            {
                std::shared_ptr<const detail::GemmRequest> operation;
                auto                                       status = RocBlasLtStatusToHIPStatus(
                    detail::createGemmRequest(reinterpret_cast<rocblaslt_handle>(handle),
                                              reinterpret_cast<rocblaslt_matmul_desc>(desc),
                                              alpha,
                                              A,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matA),
                                              B,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matB),
                                              beta,
                                              C,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matC),
                                              D,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matD),
                                              operation));
                if(status != HIPBLAS_STATUS_SUCCESS)
                {
                    diagnostics.message = "Invalid GEMM descriptors";
                    return status;
                }
                if(!operation || !operation->problem.m || !operation->problem.n)
                {
                    diagnostics.message = "Empty output needs no GEMM algorithm";
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
                request = detail::RequestAccess::make(std::move(operation));
                return HIPBLAS_STATUS_SUCCESS;
            }
            catch(const std::bad_alloc&)
            {
                return HIPBLAS_STATUS_ALLOC_FAILED;
            }
            catch(const std::exception& e)
            {
                diagnostics.message = e.what();
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
        }

        hipblasStatus_t getJitAlgo(int            device,
                                   const Request& request,
                                   const Backend& backend,
                                   size_t         workspaceLimit,
                                   Solution&      solution,
                                   Diagnostics&   diagnostics)
        {
            solution    = {};
            diagnostics = {};
            try
            {
                detail::PreparedQuery prepared;
                auto                  status = detail::prepareQuery(
                    device, request, backend, prepared, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                std::vector<int32_t> indices;
                status = detail::lookupThenPublish(prepared.jit,
                                                   prepared.operation,
                                                   prepared.target,
                                                   1,
                                                   workspaceLimit,
                                                   {},
                                                   indices,
                                                   diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                if(indices.size() != 1 || !hipblaslt_jit::isJitIndex(indices.front()))
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                auto compiled            = std::make_shared<detail::CompiledSolution>();
                compiled->target         = std::move(prepared.target);
                compiled->request        = std::move(prepared.operation);
                compiled->jit            = std::move(prepared.jit);
                compiled->process        = detail::processId();
                compiled->workspaceLimit = workspaceLimit;
                compiled->libraryIndex   = indices.front();
                compiled->workspaceBytes = detail::publishedWorkspace(
                    compiled->target, *compiled->request, compiled->libraryIndex);
                solution = detail::SolutionAccess::make(std::move(compiled));
                return HIPBLAS_STATUS_SUCCESS;
            }
            catch(const std::bad_alloc&)
            {
                return HIPBLAS_STATUS_ALLOC_FAILED;
            }
            catch(const std::exception& e)
            {
                diagnostics.message = e.what();
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
            catch(...)
            {
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
        }

        hipblasStatus_t getLibraryAlgos(int                   device,
                                        const Request&        request,
                                        const Backend&        backend,
                                        size_t                count,
                                        size_t                workspaceLimit,
                                        std::vector<int32_t>& indices,
                                        Diagnostics&          diagnostics)
        {
            indices.clear();
            diagnostics = {};
            try
            {
                detail::PreparedQuery prepared;
                auto                  status = detail::prepareQuery(
                    device, request, backend, prepared, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                return detail::lookupThenPublish(prepared.jit,
                                                 prepared.operation,
                                                 prepared.target,
                                                 count,
                                                 workspaceLimit,
                                                 {},
                                                 indices,
                                                 diagnostics);
            }
            catch(const std::bad_alloc&)
            {
                return HIPBLAS_STATUS_ALLOC_FAILED;
            }
            catch(const std::exception& e)
            {
                diagnostics.message = e.what();
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
            catch(...)
            {
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
        }

        hipblasStatus_t getGemmAlgo(const Solution&                   solution,
                                    hipblasLtMatmulHeuristicResult_t& result,
                                    Diagnostics&                      diagnostics)
        {
            result       = {};
            result.state = HIPBLAS_STATUS_INVALID_VALUE;
            diagnostics  = {};
            auto finish  = [&](hipblasStatus_t status) {
                result.state = status;
                return status;
            };
            try
            {
                auto compiled = detail::SolutionAccess::get(solution);
                if(!compiled || !hipblaslt_jit::isJitIndex(compiled->libraryIndex))
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                diagnostics.backend = compiled->jit->components().backend->info().name;
                int device          = -1;
                if(compiled->process != detail::processId() || hipGetDevice(&device) != hipSuccess
                   || device != compiled->target.device)
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                const int32_t         index = compiled->libraryIndex;
                rocblaslt_matmul_algo algo{};
                std::memcpy(algo.data, &index, sizeof(index));
                algo.max_workspace_bytes = compiled->workspaceLimit;
                std::memcpy(&result.algo, &algo, sizeof(algo));
                result.workspaceSize = compiled->workspaceBytes;
                return finish(HIPBLAS_STATUS_SUCCESS);
            }
            catch(const std::bad_alloc&)
            {
                return finish(HIPBLAS_STATUS_ALLOC_FAILED);
            }
            catch(...)
            {
                return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
            }
        }
    }
}

namespace hipblaslt_jit
{
#ifdef HIPBLASLT_JIT_HIPKITTENS
    std::shared_ptr<const Jit> makeHipKittensJit();
#endif

    namespace
    {
        // The process-wide Jit: replay backend, comgr builder, TensileLite loader.
        // Null when HIPBLASLT_JIT_TEST_REPLAY names no bundle. The store on that
        // Jit publishes every solution the replay backend builds.
        std::shared_ptr<const Jit> replayProcess()
        {
            static std::once_flag                 once;
            static std::shared_ptr<const Jit>     jit;
            std::call_once(once, [] {
                const char* value = rocblaslt_secure_getenv("HIPBLASLT_JIT_TEST_REPLAY");
                if(!value || !*value)
                    return;
                hipblaslt_ext::experimental::jit::replay::Options options;
                std::istringstream                                paths(value);
                std::string                                       path;
                while(paths >> path)
                    options.replay.push_back(std::move(path));
                if(options.replay.empty())
                    return;
                hipblaslt_ext::experimental::jit::Backend     backend;
                hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
                const auto status = hipblaslt_ext::experimental::jit::replay::createBackend(
                    options, backend, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                {
                    std::cerr << "hipblaslt warning: HIPBLASLT_JIT_TEST_REPLAY could not be read: "
                              << diagnostics.message << std::endl;
                    return;
                }
                jit = hipblaslt_ext::experimental::jit::detail::BackendAccess::get(backend);
            });
            return jit;
        }

        // Replay when HIPBLASLT_JIT_TEST_REPLAY names bundles. Otherwise the
        // HipKittens backend of a HipKittens build. Both publish through the
        // shared lookup.
        std::shared_ptr<const Jit> processJit()
        {
            if(auto replay = replayProcess())
                return replay;
#ifdef HIPBLASLT_JIT_HIPKITTENS
            static std::once_flag             once;
            static std::shared_ptr<const Jit> hipkittens;
            std::call_once(once, [] { hipkittens = makeHipKittensJit(); });
            return hipkittens;
#else
            return nullptr;
#endif
        }
    }

    bool jitHeuristicLibrary()
    {
        return static_cast<bool>(processJit());
    }

    void warnJitHeuristicUnavailable()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            const char* value = rocblaslt_secure_getenv("HIPBLASLT_JIT");
            std::cerr << "hipblaslt warning: HIPBLASLT_JIT=" << (value ? value : "");
            if(mode() == Mode::Forced)
                std::cerr << " has no JIT library; heuristic queries return no solutions"
                          << std::endl;
            else
                std::cerr << " has no JIT library; heuristic queries are unchanged" << std::endl;
        });
    }

    int appendJitHeuristic(rocblaslt_handle                   handle,
                           const RocblasltContractionProblem& problem,
                           size_t                             workspaceLimit,
                           const std::vector<std::string>&    excludeKernels,
                           rocblaslt_matmul_heuristic_result* results,
                           int                                room)
    {
        if(room <= 0 || results == nullptr || handle == nullptr)
            return 0;
        const auto jit = processJit();
        if(!jit)
        {
            warnJitHeuristicUnavailable();
            return 0;
        }
        try
        {
            const bool capturing = streamIsCapturing(problem.stream);
            int        current   = handle->device;
            if(!capturing
               && (hipGetDevice(&current) != hipSuccess || current != handle->device))
                return 0;
            using GemmRequest = hipblaslt_ext::experimental::jit::detail::GemmRequest;
            auto owned        = std::make_shared<const GemmRequest>(problem);
            DeviceTarget target;
            if(!DeviceTarget::make(handle->device, target).ok() || !target.hardware)
                return 0;

            namespace detail = hipblaslt_ext::experimental::jit::detail;
            std::vector<int32_t> indices;
            hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
            const auto           status = detail::lookupThenPublish(jit,
                                                          owned,
                                                          target,
                                                          static_cast<size_t>(room),
                                                          workspaceLimit,
                                                          excludeKernels,
                                                          indices,
                                                          diagnostics);
            if(status != HIPBLAS_STATUS_SUCCESS && indices.empty())
            {
                if(!diagnostics.message.empty())
                    std::cerr << (capturing ? "hipblaslt error: " : "hipblaslt warning: JIT heuristic ")
                              << diagnostics.message << std::endl;
                return 0;
            }

            int written = 0;
            for(auto index : indices)
            {
                if(written == room || !isJitIndex(index))
                    break;
                auto& result = results[written];
                std::memset(&result, 0, sizeof(result));
                auto* stored = reinterpret_cast<int32_t*>(result.algo.data);
                *stored      = index;
                result.algo.max_workspace_bytes = workspaceLimit;
                result.workspaceSize = detail::publishedWorkspace(target, *owned, index);
                result.state         = rocblaslt_status_success;
                result.wavesCount    = 1.0f;
                ++written;
            }
            return written;
        }
        catch(const std::exception& error)
        {
            std::cerr << "hipblaslt warning: JIT heuristic " << error.what() << std::endl;
            return 0;
        }
        catch(...)
        {
            return 0;
        }
    }
}
