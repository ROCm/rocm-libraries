// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <new>

#include "recording.h"
#include "rocm/interfaces/blas.h"

namespace {
struct Context {
    rocm_blas_context_options options;
};

struct LtContext {
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

rocblas_status create_lt_context(const rocm_blas_context_options* options, void** result) {
    if (!options || !result) return rocblas_status_invalid_pointer;
    auto* context = new (std::nothrow) LtContext{*options};
    if (!context) return rocblas_status_memory_error;
    *result = context;
    rocm::interfaces::recording::trace(options->host, "blaslt", "create_context", options,
                                       sizeof(*options));
    return rocblas_status_success;
}

void destroy_lt_context(void* opaque) {
    auto* context = static_cast<LtContext*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blaslt", "destroy_context", nullptr,
                                       0);
    delete context;
}

rocblas_status vector_execute(void* opaque, const rocm_blas_vector_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    if (request->x.length < 0 || request->y.length < 0 || request->batch_count < 0)
        return rocblas_status_invalid_size;
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blas", "vector_execute", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

rocblas_status matmul_execute(void* opaque, const rocm_blas_matmul_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    if (request->a.rows < 0 || request->b.columns < 0 || request->batch_count < 0)
        return rocblas_status_invalid_size;
    auto* context = static_cast<Context*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blas", "matmul_execute", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

rocblas_status lt_heuristic(void* opaque, const rocm_blas_matmul_request* request,
                            rocm_blaslt_heuristic_result* results, size_t capacity, size_t* count) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request || !count || (capacity && !results)) return rocblas_status_invalid_pointer;
    *count = capacity ? 1 : 0;
    if (capacity) {
        results[0] = {
            rocm::interfaces::recording::header(sizeof(results[0])), {0x524f434d, 1}, 4096, 0};
    }
    auto* context = static_cast<LtContext*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blaslt", "heuristic", request,
                                       sizeof(*request));
    return rocblas_status_success;
}

rocblas_status lt_matmul(void* opaque, const rocm_blas_matmul_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    auto* context = static_cast<LtContext*>(opaque);
    rocm::interfaces::recording::trace(context->options.host, "blaslt", "matmul", request,
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

const rocm_blaslt_provider_v1 lt_table = {
    rocm::interfaces::recording::header(sizeof(rocm_blaslt_provider_v1)),
    create_lt_context,
    destroy_lt_context,
    lt_heuristic,
    lt_matmul,
};

}  // namespace

extern "C" ROCM_INTERFACES_EXPORT rocm_interfaces_status ROCM_INTERFACES_CALL
rocm_interfaces_provider_query_v1(const rocm_interfaces_provider_request* request,
                                  rocm_interfaces_provider_response* response) {
    if (request && request->domain == ROCM_INTERFACES_DOMAIN_BLASLT) {
        return rocm::interfaces::recording::query(request, response, ROCM_INTERFACES_DOMAIN_BLASLT,
                                                  "recording-blas", &lt_table);
    }
    return rocm::interfaces::recording::query(request, response, ROCM_INTERFACES_DOMAIN_BLAS,
                                              "recording-blas", &table);
}
