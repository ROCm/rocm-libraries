// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <new>

#include "recording.h"
#include "rocm/interfaces/experimental/blas_narrow_v2.h"

namespace {
struct Context {
    rocm_blas_v2_context_options options;
};
rocblas_status create_context(const rocm_blas_v2_context_options* o, void** out) {
    if (!o || !out) return rocblas_status_invalid_pointer;
    *out = new (std::nothrow) Context{*o};
    return *out ? rocblas_status_success : rocblas_status_memory_error;
}
void destroy_context(void* p) {
    delete static_cast<Context*>(p);
}
template <class T>
rocblas_status record(void* p, const T* r, const char* operation) {
    if (!p) return rocblas_status_invalid_handle;
    if (!r) return rocblas_status_invalid_pointer;
    auto* c = static_cast<Context*>(p);
    rocm::interfaces::recording::trace(c->options.host, "blas_v2", operation, r, sizeof(*r));
    return rocblas_status_success;
}
#define CALLBACK(name, Type)                      \
    rocblas_status name(void* p, const Type* r) { \
        return record(p, r, #name);               \
    }
CALLBACK(vector_transform, rocm_blas_v2_vector_transform_request)
CALLBACK(vector_reduce, rocm_blas_v2_vector_reduce_request)
CALLBACK(vector_rotate, rocm_blas_v2_rotation_request)
CALLBACK(matrix_vector, rocm_blas_v2_matrix_vector_request)
CALLBACK(rank_update, rocm_blas_v2_rank_update_request)
CALLBACK(structured_matrix, rocm_blas_v2_structured_matrix_request)
CALLBACK(triangular_matrix, rocm_blas_v2_triangular_matrix_request)
CALLBACK(matrix_transform, rocm_blas_v2_matrix_transform_request)
#undef CALLBACK
rocblas_status matmul_query(void* p, const rocm_blas_v2_matmul_request* r, rocm_blas_v2_solution*,
                            size_t, size_t* count) {
    if (count) *count = 0;
    return record(p, r, "matmul_query");
}
rocblas_status matmul(void* p, const rocm_blas_v2_matmul_request* r, rocm_blas_v2_matmul_result*) {
    return record(p, r, "matmul");
}
const rocm_blas_v2_provider table = {rocm::interfaces::recording::header(sizeof(table)),
                                     create_context,
                                     destroy_context,
                                     vector_transform,
                                     vector_reduce,
                                     vector_rotate,
                                     matrix_vector,
                                     rank_update,
                                     matmul_query,
                                     matmul,
                                     structured_matrix,
                                     triangular_matrix,
                                     matrix_transform};
}  // namespace
extern "C" ROCM_INTERFACES_EXPORT rocm_interfaces_status ROCM_INTERFACES_CALL
rocm_interfaces_provider_query_v1(const rocm_interfaces_provider_request* request,
                                  rocm_interfaces_provider_response* response) {
    return rocm::interfaces::recording::query(request, response, ROCM_INTERFACES_DOMAIN_BLAS_V2,
                                              "recording-blas-narrow-v2", &table);
}
