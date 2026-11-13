// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-mode.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt.h"
#include "rocblaslt_secure_env.hpp"
#include "utility.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#ifndef _WIN32
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
}

namespace hipblaslt_jit
{
    Status DeviceTarget::make(int device, DeviceTarget& target)
    {
        target           = {};
        target.device    = device;
        const auto error = hipGetDeviceProperties(&target.properties, device);
        if(error != hipSuccess)
            return {Status::Code::Failed,
                    Stage::Configure,
                    std::string("Cannot query HIP device properties: ") + hipGetErrorString(error)};
        const auto parsed = Target::fromTargetId(target.properties.gcnArchName);
        if(!parsed)
            return {Status::Code::NotSupported,
                    Stage::Configure,
                    std::string("Unknown target ID ") + target.properties.gcnArchName};
        static_cast<Target&>(target) = *parsed;
        target.wavefrontSize         = target.properties.warpSize;
        target.cuCount               = target.properties.multiProcessorCount;
        target.hardware              = TensileLite::hip::GetDevice(target.properties, device);
        return {};
    }

    JitLibrary& processLibrary()
    {
        // The library path is read here, at the process edge, and passed in.
#ifdef _WIN32
        std::error_code error;
        const char*     user = std::getenv("USERNAME");
        auto            root = std::filesystem::temp_directory_path(error)
                    / ("hipblaslt-jit-" + std::string(user && *user ? user : "user"));
#else
        auto root = std::filesystem::path("/tmp") / ("hipblaslt-jit-" + std::to_string(geteuid()));
#endif
        if(const char* path = rocblaslt_secure_getenv("HIPBLASLT_JIT_LIBRARY_PATH"); path && *path)
        {
            std::error_code error;
            auto absolute = std::filesystem::absolute(std::filesystem::u8path(path), error);
            root          = error ? std::filesystem::u8path(path) : absolute;
        }
        static auto* library = new JitLibrary(std::move(root), [](int device, CacheKey& key) {
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

            Status publish(
                const GemmRequest&                                                    request,
                const DeviceTarget&                                                   target,
                const Artifact&                                                       executable,
                const std::vector<std::shared_ptr<TensileLite::ContractionSolution>>& solutions,
                std::vector<int32_t>& indices) const override
            {
                indices.clear();
                if(!target.hardware)
                    return unsupported(Stage::Publish, target);
                return m_library.publish(CacheKey::make(target, m_backend, m_codeObjectVersion),
                                         target.device,
                                         RocblasltToTensileContractionProblemGemm(request),
                                         executable,
                                         solutions,
                                         indices);
            }

        private:
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
        }

        struct PreparedQuery
        {
            std::shared_ptr<const hipblaslt_jit::Jit> jit;
            std::shared_ptr<const GemmRequest>        operation;
            hipblaslt_jit::DeviceTarget               target;
        };

        // The device and request checks of getJitAlgo.
        hipblasStatus_t prepareQuery(int                                              device,
                                     const std::shared_ptr<const GemmRequest>&        request,
                                     const std::shared_ptr<const hipblaslt_jit::Jit>& jit,
                                     PreparedQuery&                                   prepared,
                                     Diagnostics&                                     diagnostics)
        {
            prepared.jit       = jit;
            prepared.operation = request;
            if(!prepared.operation || !prepared.jit)
            {
                diagnostics.message = "A valid request and backend are required";
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            int current = -1;
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

        struct Published
        {
            size_t workspace = 0;
            // How many times the launch fills the device. One means the grid
            // occupies every compute unit; a smaller grid is below one. The tile
            // grid is the solution's, so a small problem is not reported as full
            // occupancy.
            float waves = 0.f;
        };

        // Zero for an index the JIT solution library does not resolve.
        Published published(const hipblaslt_jit::DeviceTarget& target,
                            const GemmRequest&                 request,
                            int32_t                            index)
        {
            Published result;
            if(!target.hardware || !TensileLite::isJitKernelIndex(index))
                return result;
            hipblaslt_jit::Status why;
            auto                  solution = hipblaslt_jit::processLibrary().solutionByIndex(
                target.device, *target.hardware, index, why);
            if(!solution)
                return result;
            auto problem     = hipblaslt_jit::RocblasltToTensileContractionProblemGemm(request);
            result.workspace = solution->requiredWorkspaceSize(problem, *target.hardware);
            if(target.cuCount > 0)
            {
                // HIP's dim3 and TensileLite::dim3 are unrelated types.
                TensileLite::dim3 workGroupSize{};
                TensileLite::dim3 numWorkGroups{};
                solution->calculateGrid(workGroupSize, numWorkGroups, problem);
                const auto groups = static_cast<unsigned long long>(numWorkGroups.x)
                                    * numWorkGroups.y * numWorkGroups.z;
                result.waves = static_cast<float>(groups) / static_cast<float>(target.cuCount);
            }
            return result;
        }

        // Look up JitLibrary::findTopSolutions, which publishes on a miss.
        // Heuristic problems carry no stream, so this cannot see a capture.
        hipblasStatus_t lookupThenPublish(const std::shared_ptr<const hipblaslt_jit::Jit>& jit,
                                          const std::shared_ptr<const GemmRequest>& operation,
                                          const hipblaslt_jit::DeviceTarget&        target,
                                          size_t                                    count,
                                          size_t                                    workspaceLimit,
                                          const std::vector<std::string>&           excludeKernels,
                                          std::vector<int32_t>&                     indices,
                                          Diagnostics&                              diagnostics)
        {
            indices.clear();
            if(!jit || !operation || !target.hardware)
            {
                diagnostics.message = "A valid request and backend are required";
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            const bool capturing = streamIsCapturing(operation->problem.stream);
            auto  problem = hipblaslt_jit::RocblasltToTensileContractionProblemGemm(*operation);
            auto& library = hipblaslt_jit::processLibrary();
            library.beginQuery(
                !capturing,
                target,
                jit->components().backend->info(),
                workspaceLimit,
                excludeKernels,
                [jit, operation, &target, workspaceLimit](size_t                          count,
                                                          const std::vector<std::string>& exclude,
                                                          std::vector<int32_t>&           produced,
                                                          hipblaslt_jit::Status&          failure) {
                    try
                    {
                        auto outcome
                            = jit->generate(*operation, target, count, workspaceLimit, exclude);
                        produced = std::move(outcome.indices);
                        if(produced.empty())
                            failure
                                = outcome.failures.empty()
                                      ? hipblaslt_jit::Status{hipblaslt_jit::Status::Code::Failed,
                                                              hipblaslt_jit::Stage::Publish,
                                                              "Nothing was published"}
                                      : std::move(outcome.failures.front());
                    }
                    catch(const std::exception& error)
                    {
                        failure = {hipblaslt_jit::Status::Code::Failed,
                                   hipblaslt_jit::Stage::Generate,
                                   error.what()};
                    }
                });
            struct Guard
            {
                hipblaslt_jit::JitLibrary& library;
                ~Guard()
                {
                    library.endQuery();
                }
            } guard{library};
            auto solutions
                = library.findTopSolutions(problem, *target.hardware, static_cast<int>(count));
            const auto status = library.queryStatus();
            for(const auto& solution : solutions)
                if(solution && TensileLite::isJitKernelIndex(solution->index))
                    indices.push_back(solution->index);
            if(capturing && indices.empty() && status.ok())
            {
                diagnostics.message = "JIT generation skipped during stream capture for GEMM M="
                                      + std::to_string(operation->problem.m)
                                      + " N=" + std::to_string(operation->problem.n)
                                      + " K=" + std::to_string(operation->problem.k);
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            }
            if(!status.ok() && indices.empty())
            {
                diagnostics.message = status.message;
                return toHipStatus(status.code);
            }
            diagnostics.message = std::to_string(indices.size()) + " solutions";
            return HIPBLAS_STATUS_SUCCESS;
        }
    }

    namespace jit
    {
        hipblasStatus_t makeGemmRequest(hipblasLtHandle_t                           handle,
                                        hipblasLtMatmulDesc_t                       desc,
                                        const void*                                 alpha,
                                        const void*                                 A,
                                        hipblasLtMatrixLayout_t                     matA,
                                        const void*                                 B,
                                        hipblasLtMatrixLayout_t                     matB,
                                        const void*                                 beta,
                                        const void*                                 C,
                                        hipblasLtMatrixLayout_t                     matC,
                                        void*                                       D,
                                        hipblasLtMatrixLayout_t                     matD,
                                        std::shared_ptr<const detail::GemmRequest>& request,
                                        Diagnostics&                                diagnostics)
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
                request = std::move(operation);
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

        hipblasStatus_t getJitAlgo(int                                               device,
                                   const std::shared_ptr<const detail::GemmRequest>& request,
                                   const std::shared_ptr<const hipblaslt_jit::Jit>&  jit,
                                   size_t                                            workspaceLimit,
                                   hipblasLtMatmulHeuristicResult_t&                 result,
                                   Diagnostics&                                      diagnostics)
        {
            result      = {};
            diagnostics = {};
            auto finish = [&](hipblasStatus_t status) {
                result.state = status;
                return status;
            };
            try
            {
                detail::PreparedQuery prepared;
                auto status = detail::prepareQuery(device, request, jit, prepared, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return finish(status);
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
                    return finish(status);
                if(indices.size() != 1 || !TensileLite::isJitKernelIndex(indices.front()))
                    return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
                const int32_t         index = indices.front();
                rocblaslt_matmul_algo algo{};
                std::memcpy(algo.data, &index, sizeof(index));
                algo.max_workspace_bytes = workspaceLimit;
                std::memcpy(&result.algo, &algo, sizeof(algo));
                const auto published
                    = detail::published(prepared.target, *prepared.operation, index);
                result.workspaceSize = published.workspace;
                result.wavesCount    = published.waves;
                return finish(HIPBLAS_STATUS_SUCCESS);
            }
            catch(const std::bad_alloc&)
            {
                return finish(HIPBLAS_STATUS_ALLOC_FAILED);
            }
            catch(const std::exception& e)
            {
                diagnostics.message = e.what();
                return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
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

    namespace
    {
        struct HeuristicBackends
        {
            std::mutex                              mutex;
            std::vector<std::shared_ptr<const Jit>> added;
        };
        HeuristicBackends& heuristicBackends()
        {
            static auto* backends = new HeuristicBackends;
            return *backends;
        }

        // The backends a query asks, in order, for the slots still open.
        std::vector<std::shared_ptr<const Jit>> processBackends()
        {
            auto&                       backends = heuristicBackends();
            std::lock_guard<std::mutex> lock(backends.mutex);
            return backends.added;
        }
    }

    bool jitHeuristicLibrary()
    {
        return !processBackends().empty();
    }

    void warnJitHeuristicUnavailable()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            std::cerr << "hipblaslt warning: HIPBLASLT_JIT="
                      << (mode() == Mode::Forced ? "2" : "1");
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
        const auto backends = processBackends();
        if(backends.empty())
        {
            warnJitHeuristicUnavailable();
            return 0;
        }
        try
        {
            const bool capturing = streamIsCapturing(problem.stream);
            int        current   = handle->device;
            if(!capturing && (hipGetDevice(&current) != hipSuccess || current != handle->device))
                return 0;
            using GemmRequest  = hipblaslt_ext::experimental::jit::detail::GemmRequest;
            auto         owned = std::make_shared<const GemmRequest>(problem);
            DeviceTarget target;
            if(!DeviceTarget::make(handle->device, target).ok() || !target.hardware)
                return 0;

            namespace detail = hipblaslt_ext::experimental::jit::detail;
            int written      = 0;
            for(const auto& jit : backends)
            {
                if(written == room)
                    break;
                std::vector<int32_t>                          indices;
                hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
                const auto status = detail::lookupThenPublish(jit,
                                                              owned,
                                                              target,
                                                              static_cast<size_t>(room - written),
                                                              workspaceLimit,
                                                              excludeKernels,
                                                              indices,
                                                              diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS && indices.empty())
                {
                    if(!diagnostics.message.empty())
                    {
                        if(capturing)
                            log_error(__func__,
                                      std::string("JIT heuristic ") + diagnostics.message);
                        else
                            log_info(__func__, std::string("JIT heuristic ") + diagnostics.message);
                    }
                    continue;
                }
                for(auto index : indices)
                {
                    if(written == room || !TensileLite::isJitKernelIndex(index))
                        break;
                    auto& result = results[written];
                    std::memset(&result, 0, sizeof(result));
                    auto* stored                    = reinterpret_cast<int32_t*>(result.algo.data);
                    *stored                         = index;
                    result.algo.max_workspace_bytes = workspaceLimit;
                    const auto published            = detail::published(target, *owned, index);
                    result.workspaceSize            = published.workspace;
                    result.state                    = rocblaslt_status_success;
                    result.wavesCount               = published.waves;
                    ++written;
                }
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

void hipblaslt_ext::experimental::jit::addHeuristicBackend(
    std::shared_ptr<const hipblaslt_jit::Jit> jit)
{
    if(!jit)
        return;
    auto&                       backends = hipblaslt_jit::heuristicBackends();
    std::lock_guard<std::mutex> lock(backends.mutex);
    backends.added.push_back(std::move(jit));
}
