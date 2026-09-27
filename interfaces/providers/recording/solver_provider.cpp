// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <new>

#include "recording.h"
#include "rocm/interfaces/solver.h"

namespace {
struct Context {
    rocm_solver_context_options options;
};

rocblas_status create_context(const rocm_solver_context_options* options, void** result) {
    if (!options || !result) return rocblas_status_invalid_pointer;
    if (!options->blas.context || !options->blas.vector_execute || !options->blas.matmul_execute)
        return rocblas_status_invalid_handle;
    auto* context = new (std::nothrow) Context{*options};
    if (!context) return rocblas_status_memory_error;
    *result = context;
    rocm::interfaces::recording::trace(options->host, "solver", "create_context", nullptr, 0);
    return rocblas_status_success;
}

void destroy_context(void* opaque) {
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "solver", "destroy_context", nullptr,
                                       0);
    delete context;
}

rocblas_status query_workspace(void* opaque, const rocm_solver_request* request, size_t* size) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request || !size) return rocblas_status_invalid_pointer;
    if (request->m < 0 || request->n < 0) return rocblas_status_invalid_size;
    *size = static_cast<size_t>(request->m) * static_cast<size_t>(request->n) * 4;
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "solver", "query_workspace", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

rocblas_status execute(void* opaque, const rocm_solver_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    if (request->m < 0 || request->n < 0) return rocblas_status_invalid_size;
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "solver", "execute", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

const rocm_solver_provider_v1 table = {
    rocm::interfaces::recording::header(sizeof(rocm_solver_provider_v1)),
    create_context,
    destroy_context,
    query_workspace,
    execute,
};
}  // namespace

extern "C" ROCM_INTERFACES_EXPORT rocm_interfaces_status ROCM_INTERFACES_CALL
rocm_interfaces_provider_query_v1(const rocm_interfaces_provider_request* request,
                                  rocm_interfaces_provider_response* response) {
    return rocm::interfaces::recording::query(request, response, ROCM_INTERFACES_DOMAIN_SOLVER,
                                              "recording-solver", &table);
}
