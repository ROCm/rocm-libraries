// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <new>

#include "recording.h"
#include "rocm/interfaces/blas.h"

namespace {
struct Context {
    rocm_blas_context_options options;
};

rocblas_status create_context(const rocm_blas_context_options* options, void** result) {
    if (!options || !result) return rocblas_status_invalid_pointer;
    auto* context = new (std::nothrow) Context{*options};
    if (!context) return rocblas_status_memory_error;
    *result = context;
    rocm::interfaces::recording::trace(options->host, "blas", "create_context", options,
                                       sizeof(*options));
    return rocblas_status_success;
}

void destroy_context(void* opaque) {
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blas", "destroy_context", nullptr,
                                       0);
    delete context;
}

rocblas_status vector_execute(void* opaque, const rocm_blas_vector_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blas", "vector_execute", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

rocblas_status matmul_execute(void* opaque, const rocm_blas_matmul_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blas", "matmul_execute", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

const rocm_blas_provider_v1 table = {
    rocm::interfaces::recording::header(sizeof(rocm_blas_provider_v1)),
    create_context,
    destroy_context,
    vector_execute,
    matmul_execute,
};
}  // namespace

extern "C" ROCM_INTERFACES_EXPORT rocm_interfaces_status ROCM_INTERFACES_CALL
rocm_interfaces_provider_query_v1(const rocm_interfaces_provider_request* request,
                                  rocm_interfaces_provider_response* response) {
    return rocm::interfaces::recording::query(request, response, ROCM_INTERFACES_DOMAIN_BLAS,
                                              "recording-blas-legacy", &table);
}
