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
    rocm::interfaces::recording::trace(options->host, "blaslt", "create_context", options,
                                       sizeof(*options));
    return rocblas_status_success;
}

void destroy_context(void* opaque) {
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blaslt", "destroy_context", nullptr,
                                       0);
    delete context;
}

rocblas_status heuristic(void* opaque, const rocm_blas_matmul_request* request,
                         rocm_blaslt_heuristic_result* results, size_t capacity, size_t* count) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request || !count || (capacity && !results)) return rocblas_status_invalid_pointer;
    *count = capacity ? 1 : 0;
    if (capacity) {
        results[0] = {
            rocm::interfaces::recording::header(sizeof(results[0])), {0x4c4547414359, 1}, 8192, 0};
    }
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blaslt", "heuristic", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

rocblas_status matmul(void* opaque, const rocm_blas_matmul_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blaslt", "matmul", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

const rocm_blaslt_provider_v1 table = {
    rocm::interfaces::recording::header(sizeof(rocm_blaslt_provider_v1)),
    create_context,
    destroy_context,
    heuristic,
    matmul,
};
}  // namespace

extern "C" ROCM_INTERFACES_EXPORT rocm_interfaces_status ROCM_INTERFACES_CALL
rocm_interfaces_provider_query_v1(const rocm_interfaces_provider_request* request,
                                  rocm_interfaces_provider_response* response) {
    return rocm::interfaces::recording::query(request, response, ROCM_INTERFACES_DOMAIN_BLASLT,
                                              "recording-blaslt-legacy", &table);
}
