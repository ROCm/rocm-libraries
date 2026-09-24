// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_internal.hpp"
#include "rocblaslt.h"
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
#include <hipblaslt/hipblaslt-jit.hpp>

namespace hipblaslt_ext::experimental::jit::tensilelite
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
        jit::Diagnostics sharedDiagnostics;
        auto             finish = [&](hipblasStatus_t status) {
            result.state        = status;
            diagnostics.message = std::move(sharedDiagnostics.message);
            return status;
        };
        try
        {
            // This entry point always requires an explicit recipe, including
            // when a generic TensileLite backend supports predicted recipes.
            if(options.pythonExecutable.empty() || options.tensileSourceDirectory.empty()
               || options.configPath.empty() || options.outputPath.empty())
            {
                sharedDiagnostics.message
                    = "TensileLite requires Python, source, explicit YAML and output paths";
                return finish(HIPBLAS_STATUS_INVALID_VALUE);
            }
            jit::Request request;
            auto         status = jit::makeGemmRequest(handle,
                                               desc,
                                               alpha,
                                               A,
                                               layoutA,
                                               B,
                                               layoutB,
                                               beta,
                                               C,
                                               layoutC,
                                               D,
                                               layoutD,
                                               request,
                                               sharedDiagnostics);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return finish(status);
            int device = -1;
            if(hipGetDevice(&device) != hipSuccess)
                return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
            if(device != reinterpret_cast<rocblaslt_handle>(handle)->device)
            {
                sharedDiagnostics.message = "Compile on the handle's HIP device";
                return finish(HIPBLAS_STATUS_INVALID_VALUE);
            }
            jit::Backend backend;
            status = createBackend(options, backend, sharedDiagnostics);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return finish(status);
            jit::Solution solution;
            status = jit::getJitAlgo(
                device, request, backend, maxWorkspaceBytes, solution, sharedDiagnostics);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return finish(status);
            return finish(jit::getGemmAlgo(solution, result, sharedDiagnostics));
        }
        catch(const std::bad_alloc&)
        {
            return finish(HIPBLAS_STATUS_ALLOC_FAILED);
        }
        catch(const std::exception& error)
        {
            sharedDiagnostics.message = error.what();
            return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        catch(...)
        {
            return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
        }
    }
}
