// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_SOLVER_H_
#define ROCM_INTERFACES_SOLVER_H_

#include "rocm/interfaces/blas.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rocm_solver_operation {
    ROCM_SOLVER_GETRF = 1,
    ROCM_SOLVER_GETRS = 2,
    ROCM_SOLVER_GEQRF = 3
} rocm_solver_operation;

typedef struct rocm_solver_request {
    rocm_interfaces_abi_header header;
    rocm_solver_operation operation;
    uint32_t index_width;
    rocblas_datatype data_type;
    int64_t m;
    int64_t n;
    int64_t nrhs;
    rocm_blas_matrix a;
    rocm_blas_matrix b;
    void* pivots;
    void* tau;
    void* info;
    void* workspace;
    size_t workspace_size;
} rocm_solver_request;

typedef struct rocm_solver_blas_services_v1 {
    rocm_interfaces_abi_header header;
    void* context;
    rocblas_status(ROCM_INTERFACES_CALL* vector_execute)(void*, const rocm_blas_vector_request*);
    rocblas_status(ROCM_INTERFACES_CALL* matmul_execute)(void*, const rocm_blas_matmul_request*);
} rocm_solver_blas_services_v1;

typedef struct rocm_solver_context_options {
    rocm_interfaces_abi_header header;
    const rocm_interfaces_host_services* host;
    rocm_interfaces_device_key device;
    void* stream;
    rocm_solver_blas_services_v1 blas;
} rocm_solver_context_options;

typedef struct rocm_solver_provider_v1 {
    rocm_interfaces_abi_header header;
    rocblas_status(ROCM_INTERFACES_CALL* create_context)(const rocm_solver_context_options*,
                                                         void** solver_context);
    void(ROCM_INTERFACES_CALL* destroy_context)(void* solver_context);
    rocblas_status(ROCM_INTERFACES_CALL* query_workspace)(void* solver_context,
                                                          const rocm_solver_request*,
                                                          size_t* workspace_size);
    rocblas_status(ROCM_INTERFACES_CALL* execute)(void* solver_context, const rocm_solver_request*);
} rocm_solver_provider_v1;

#ifdef __cplusplus
}
#endif
#endif
