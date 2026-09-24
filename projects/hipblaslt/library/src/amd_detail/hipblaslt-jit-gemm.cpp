// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-tensilelite-internal.hpp"
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
    namespace jit::tensilelite::detail
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
                std::mutex                                                  mutex;
                std::unordered_map<uint64_t, std::shared_ptr<const Bundle>> entries;
                std::unordered_map<const Bundle*, uint64_t>                 tokens;
                std::mt19937_64 random{std::random_device{}()};
            };
            Registry& registry()
            {
                // Copied algorithms and captured graphs can outlive their creator.
                static auto* instance = new Registry;
                return *instance;
            }
        }

        uint64_t registerBundle(std::shared_ptr<const Bundle> bundle)
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

        std::shared_ptr<const Bundle> resolveJitAlgo(const rocblaslt_matmul_algo& algo, int device)
        {
            if(!experimental::detail::isJitAlgo(algo))
                return {};
            int index;
            std::memcpy(&index, algo.data, sizeof(index));
            uint64_t token = 0;
            std::memcpy(&token, algo.data_pad, sizeof(algo.data_pad));
            std::shared_ptr<const Bundle> entry;
            {
                auto&                       r = registry();
                std::lock_guard<std::mutex> lock(r.mutex);
                auto                        found = r.entries.find(token);
                if(index != 0 || algo.fallback || found == r.entries.end())
                    throw std::invalid_argument("Unknown process-local JIT algorithm");
                entry = found->second;
            }
            int current = -1;
            if(entry->process != processId() || entry->device != device
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
                                    const GemmProblem&           request,
                                    size_t&                      workspaceBytes)
        {
            workspaceBytes = 0;
            return invoke([&] {
                auto        entry = resolveJitAlgo(algo, handle->device);
                Diagnostics diagnostics;
                size_t      required = 0;
                auto        status
                    = entry->support(request, algo.max_workspace_bytes, required, diagnostics);
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
                                    const GemmProblem&                     request,
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
                auto        status   = entry->support(request,
                                             std::min(workspaceBytes, algo.max_workspace_bytes),
                                             required,
                                             diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                if(required > workspaceBytes || required > algo.max_workspace_bytes
                   || (required && !workspace))
                    return HIPBLAS_STATUS_INVALID_VALUE;
                std::shared_ptr<const PreparedLaunch> candidate;
                status = entry->prepare(request,
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

    namespace jit::tensilelite
    {
        hipblasStatus_t getGemmAlgo(hipblasLtHandle_t                 handle,
                                    hipblasLtMatmulDesc_t             desc,
                                    const void*                       alpha,
                                    const void*                       A,
                                    hipblasLtMatrixLayout_t           layoutA,
                                    const void*                       B,
                                    hipblasLtMatrixLayout_t           layoutB,
                                    const void*                       beta,
                                    const void*                       C,
                                    hipblasLtMatrixLayout_t           layoutC,
                                    void*                             D,
                                    hipblasLtMatrixLayout_t           layoutD,
                                    const Options&                    options,
                                    size_t                            maxWorkspaceBytes,
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
                if(options.pythonExecutable.empty() || options.tensileSourceDirectory.empty()
                   || options.configPath.empty() || options.outputPath.empty())
                {
                    diagnostics.message
                        = "TensileLite requires Python, source, explicit YAML and output paths";
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                }
                std::shared_ptr<const detail::GemmProblem> problem;
                auto                                       status = RocBlasLtStatusToHIPStatus(
                    detail::createGemmProblem(reinterpret_cast<rocblaslt_handle>(handle),
                                              reinterpret_cast<rocblaslt_matmul_desc>(desc),
                                              alpha,
                                              A,
                                              reinterpret_cast<rocblaslt_matrix_layout>(layoutA),
                                              B,
                                              reinterpret_cast<rocblaslt_matrix_layout>(layoutB),
                                              beta,
                                              C,
                                              reinterpret_cast<rocblaslt_matrix_layout>(layoutC),
                                              D,
                                              reinterpret_cast<rocblaslt_matrix_layout>(layoutD),
                                              problem));
                if(status != HIPBLAS_STATUS_SUCCESS)
                {
                    diagnostics.message = "Invalid GEMM descriptors";
                    return finish(status);
                }
                if(!problem || !problem->problem.m || !problem->problem.n)
                {
                    diagnostics.message = "Empty output needs no GEMM algorithm";
                    return finish(HIPBLAS_STATUS_NOT_SUPPORTED);
                }
                int device = -1;
                if(hipGetDevice(&device) != hipSuccess)
                    return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
                if(device != reinterpret_cast<rocblaslt_handle>(handle)->device)
                {
                    diagnostics.message = "Compile on the handle's HIP device";
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                }
                hipDeviceProp_t properties{};
                if(hipGetDeviceProperties(&properties, device) != hipSuccess)
                    return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
                std::shared_ptr<detail::Bundle> bundle;
                status = detail::generateBundle(options, properties, device, bundle, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return finish(status);
                size_t required = 0;
                status = bundle->support(*problem, maxWorkspaceBytes, required, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return finish(status);
                bundle->process             = detail::processId();
                const auto            token = detail::registerBundle(bundle);
                rocblaslt_matmul_algo algo{};
                std::memcpy(algo.data + sizeof(int32_t),
                            &experimental::detail::jitAlgoTag,
                            sizeof(experimental::detail::jitAlgoTag));
                std::memcpy(algo.data_pad, &token, sizeof(algo.data_pad));
                algo.max_workspace_bytes = maxWorkspaceBytes;
                std::memcpy(&result.algo, &algo, sizeof(algo));
                result.workspaceSize = required;
                return finish(HIPBLAS_STATUS_SUCCESS);
            }
            catch(const std::bad_alloc&)
            {
                return finish(HIPBLAS_STATUS_ALLOC_FAILED);
            }
            catch(const std::exception& error)
            {
                diagnostics.message = error.what();
                return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
            }
            catch(...)
            {
                return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
            }
        }
    }
}
