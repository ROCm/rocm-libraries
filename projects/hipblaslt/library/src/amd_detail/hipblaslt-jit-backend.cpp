// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt.h"
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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
                if(index != 0 || algo.fallback || found == r.entries.end())
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

        rocblaslt_status supportJit(rocblaslt_handle             handle,
                                    const rocblaslt_matmul_algo& algo,
                                    const GemmRequest&           request,
                                    size_t&                      workspaceBytes)
        {
            workspaceBytes = 0;
            return invoke([&] {
                auto        entry = resolveJitAlgo(algo, handle->device);
                Diagnostics diagnostics;
                size_t      required = 0;
                auto        status   = entry->bundle->support(
                    request, algo.max_workspace_bytes, required, diagnostics);
                if(status == HIPBLAS_STATUS_SUCCESS)
                {
                    if(required > algo.max_workspace_bytes)
                        return HIPBLAS_STATUS_INVALID_VALUE;
                    workspaceBytes = required;
                }
                return status;
            });
        }

        rocblaslt_status prepareJit(rocblaslt_handle                       handle,
                                    const rocblaslt_matmul_algo&           algo,
                                    const GemmRequest&                     request,
                                    void*                                  workspace,
                                    size_t                                 workspaceBytes,
                                    hipStream_t                            stream,
                                    std::shared_ptr<const PreparedLaunch>& launch)
        {
            launch.reset();
            return invoke([&] {
                auto        entry = resolveJitAlgo(algo, handle->device);
                Diagnostics diagnostics;
                size_t      required = 0;
                auto        status
                    = entry->bundle->support(request,
                                             std::min(workspaceBytes, algo.max_workspace_bytes),
                                             required,
                                             diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                if(required > workspaceBytes || required > algo.max_workspace_bytes
                   || (required && !workspace))
                    return HIPBLAS_STATUS_INVALID_VALUE;
                std::shared_ptr<const PreparedLaunch> candidate;
                status = entry->bundle->prepare(request,
                                                {reinterpret_cast<hipblasLtHandle_t>(handle),
                                                 workspace,
                                                 workspaceBytes,
                                                 stream},
                                                candidate,
                                                diagnostics);
                if(status == HIPBLAS_STATUS_SUCCESS)
                {
                    if(!candidate)
                        return HIPBLAS_STATUS_INTERNAL_ERROR;
                    launch = std::move(candidate);
                }
                return status;
            });
        }

        rocblaslt_status runJit(rocblaslt_handle             handle,
                                const rocblaslt_matmul_algo& algo,
                                const PreparedLaunch&        launch,
                                hipStream_t                  stream,
                                hipEvent_t                   start,
                                hipEvent_t                   stop)
        {
            return invoke([&] {
                auto entry = resolveJitAlgo(algo, handle->device);
                return launch.run(stream, start, stop);
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
                auto implementation = detail::BackendAccess::get(backend);
                auto operation      = detail::RequestAccess::get(request);
                if(!operation || !implementation)
                {
                    diagnostics.message = "A valid request and backend are required";
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
                diagnostics.backend = implementation->name();
                detail::Target target;
                if(hipGetDevice(&target.device) != hipSuccess)
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                if(target.device != device)
                {
                    diagnostics.message = "Compile on the requested HIP device";
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
                if(hipGetDeviceProperties(&target.properties, device) != hipSuccess)
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                std::shared_ptr<const detail::KernelBundle> bundle;
                auto                                        status = implementation->compile(
                    *operation, target, workspaceLimit, bundle, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                if(!bundle)
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                if(bundle->operationKind() != operation->kind())
                {
                    diagnostics.message = "Bundle does not implement the requested operation";
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
                size_t required = 0;
                status = bundle->support(*operation, workspaceLimit, required, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                if(required > workspaceLimit)
                    return HIPBLAS_STATUS_INVALID_VALUE;
                auto compiled            = std::make_shared<detail::CompiledSolution>();
                compiled->target         = target;
                compiled->request        = std::move(operation);
                compiled->backend        = std::move(implementation);
                compiled->bundle         = std::move(bundle);
                compiled->process        = detail::processId();
                compiled->workspaceLimit = workspaceLimit;
                compiled->workspaceBytes = required;
                solution                 = detail::SolutionAccess::make(std::move(compiled));
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
                if(!compiled)
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                diagnostics.backend = compiled->backend->name();
                int device          = -1;
                if(compiled->process != detail::processId() || hipGetDevice(&device) != hipSuccess
                   || device != compiled->target.device)
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                if(compiled->bundle->operationKind() != detail::GemmRequest::operation)
                    return finish(HIPBLAS_STATUS_NOT_SUPPORTED);
                const auto            token = detail::registerBundle(compiled);
                rocblaslt_matmul_algo algo{};
                std::memcpy(algo.data + sizeof(int32_t),
                            &experimental::detail::jitAlgoTag,
                            sizeof(experimental::detail::jitAlgoTag));
                std::memcpy(algo.data_pad, &token, sizeof(algo.data_pad));
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
