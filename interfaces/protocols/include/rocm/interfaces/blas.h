// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_BLAS_H_
#define ROCM_INTERFACES_BLAS_H_

#include <rocblas/rocblas.h>

#include "rocm/interfaces/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rocm_blas_scalar_location {
    ROCM_BLAS_SCALAR_HOST = 1,
    ROCM_BLAS_SCALAR_DEVICE = 2
} rocm_blas_scalar_location;

typedef enum rocm_blas_batch_kind {
    ROCM_BLAS_BATCH_SINGLE = 0,
    ROCM_BLAS_BATCH_POINTER_ARRAY = 1,
    ROCM_BLAS_BATCH_STRIDED = 2,
    ROCM_BLAS_BATCH_GROUPED = 3
} rocm_blas_batch_kind;

typedef enum rocm_blas_vector_opcode {
    ROCM_BLAS_VECTOR_AXPY = 1,
    ROCM_BLAS_VECTOR_DOT = 2
} rocm_blas_vector_opcode;

typedef struct rocm_blas_scalar {
    rocm_interfaces_abi_header header;
    rocblas_datatype type;
    rocm_blas_scalar_location location;
    const void* value;
} rocm_blas_scalar;

typedef struct rocm_blas_vector {
    rocm_interfaces_abi_header header;
    rocblas_datatype type;
    void* data;
    int64_t length;
    int64_t increment;
    int64_t stride;
} rocm_blas_vector;

typedef struct rocm_blas_matrix {
    rocm_interfaces_abi_header header;
    rocblas_datatype type;
    void* data;
    int64_t rows;
    int64_t columns;
    int64_t leading_dimension;
    int64_t batch_stride;
} rocm_blas_matrix;

typedef struct rocm_blas_context_options {
    rocm_interfaces_abi_header header;
    const rocm_interfaces_host_services* host;
    rocm_interfaces_device_key device;
    void* stream;
    uint32_t pointer_mode;
    uint32_t math_mode;
    uint32_t atomics_mode;
    uint32_t numerics_mode;
} rocm_blas_context_options;

typedef struct rocm_blas_vector_request {
    rocm_interfaces_abi_header header;
    rocm_blas_vector_opcode opcode;
    uint32_t index_width;
    rocm_blas_batch_kind batch_kind;
    int64_t batch_count;
    rocm_blas_scalar alpha;
    rocm_blas_vector x;
    rocm_blas_vector y;
    void* result;
} rocm_blas_vector_request;

typedef struct rocm_blas_matmul_request {
    rocm_interfaces_abi_header header;
    uint32_t index_width;
    rocm_blas_batch_kind batch_kind;
    int64_t batch_count;
    rocblas_operation operation_a;
    rocblas_operation operation_b;
    rocblas_datatype compute_type;
    rocm_blas_scalar alpha;
    rocm_blas_scalar beta;
    rocm_blas_matrix a;
    rocm_blas_matrix b;
    rocm_blas_matrix c;
    rocm_blas_matrix d;
    void* algorithm_token;
    void* workspace;
    size_t workspace_size;
} rocm_blas_matmul_request;

typedef struct rocm_blas_provider_v1 {
    rocm_interfaces_abi_header header;
    rocblas_status(ROCM_INTERFACES_CALL* create_context)(const rocm_blas_context_options*, void**);
    void(ROCM_INTERFACES_CALL* destroy_context)(void*);
    rocblas_status(ROCM_INTERFACES_CALL* vector_execute)(void*, const rocm_blas_vector_request*);
    rocblas_status(ROCM_INTERFACES_CALL* matmul_execute)(void*, const rocm_blas_matmul_request*);
} rocm_blas_provider_v1;

typedef struct rocm_blaslt_heuristic_result {
    rocm_interfaces_abi_header header;
    uint64_t algorithm_token[2];
    size_t workspace_size;
    uint64_t capability_mask;
} rocm_blaslt_heuristic_result;

typedef struct rocm_blaslt_provider_v1 {
    rocm_interfaces_abi_header header;
    rocblas_status(ROCM_INTERFACES_CALL* create_context)(const rocm_blas_context_options*, void**);
    void(ROCM_INTERFACES_CALL* destroy_context)(void*);
    rocblas_status(ROCM_INTERFACES_CALL* heuristic)(void* blaslt_context,
                                                    const rocm_blas_matmul_request*,
                                                    rocm_blaslt_heuristic_result* results,
                                                    size_t result_capacity, size_t* result_count);
    rocblas_status(ROCM_INTERFACES_CALL* matmul)(void* blaslt_context,
                                                 const rocm_blas_matmul_request*);
} rocm_blaslt_provider_v1;

#ifdef __cplusplus
}
#endif
#endif
