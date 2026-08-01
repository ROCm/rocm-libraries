// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_EXPERIMENTAL_BLAS_NARROW_V2_H_
#define ROCM_INTERFACES_EXPERIMENTAL_BLAS_NARROW_V2_H_

#include "rocm/interfaces/blas.h"

#ifdef __cplusplus
extern "C" {
#endif

// This is an evaluable protocol proposal, not an adopted provider ABI. Values
// are append-only once a generation is adopted. Public rocBLAS enums and status
// values are intentionally reused rather than mirrored.

typedef uint32_t rocm_blas_v2_index_width;
enum { ROCM_BLAS_V2_INDEX_32 = 1, ROCM_BLAS_V2_INDEX_64 = 2 };

typedef uint32_t rocm_blas_v2_storage_kind;
enum {
    ROCM_BLAS_V2_STORAGE_DENSE = 1,
    ROCM_BLAS_V2_STORAGE_BANDED = 2,
    ROCM_BLAS_V2_STORAGE_PACKED = 3
};

typedef uint32_t rocm_blas_v2_matrix_kind;
enum {
    ROCM_BLAS_V2_MATRIX_GENERAL = 1,
    ROCM_BLAS_V2_MATRIX_SYMMETRIC = 2,
    ROCM_BLAS_V2_MATRIX_HERMITIAN = 3,
    ROCM_BLAS_V2_MATRIX_TRIANGULAR = 4
};

typedef uint32_t rocm_blas_v2_vector_transform_op;
enum {
    ROCM_BLAS_V2_VECTOR_SCALE = 1,
    ROCM_BLAS_V2_VECTOR_COPY = 2,
    ROCM_BLAS_V2_VECTOR_SWAP = 3,
    ROCM_BLAS_V2_VECTOR_AXPY = 4
};

typedef uint32_t rocm_blas_v2_vector_reduce_op;
enum {
    ROCM_BLAS_V2_REDUCE_DOT = 1,
    ROCM_BLAS_V2_REDUCE_DOT_CONJUGATE_X = 2,
    ROCM_BLAS_V2_REDUCE_NORM_2 = 3,
    ROCM_BLAS_V2_REDUCE_ABSOLUTE_SUM = 4,
    ROCM_BLAS_V2_REDUCE_ABSOLUTE_MAX_INDEX = 5,
    ROCM_BLAS_V2_REDUCE_ABSOLUTE_MIN_INDEX = 6
};

typedef uint32_t rocm_blas_v2_rotation_op;
enum {
    ROCM_BLAS_V2_ROTATE = 1,
    ROCM_BLAS_V2_ROTATION_PARAMETERS = 2,
    ROCM_BLAS_V2_ROTATE_MODIFIED = 3,
    ROCM_BLAS_V2_MODIFIED_ROTATION_PARAMETERS = 4
};

typedef uint32_t rocm_blas_v2_matrix_vector_op;
enum {
    ROCM_BLAS_V2_MATRIX_VECTOR_MULTIPLY = 1,
    ROCM_BLAS_V2_TRIANGULAR_VECTOR_MULTIPLY = 2,
    ROCM_BLAS_V2_TRIANGULAR_VECTOR_SOLVE = 3
};

typedef uint32_t rocm_blas_v2_rank_update_op;
enum {
    ROCM_BLAS_V2_RANK_ONE = 1,
    ROCM_BLAS_V2_RANK_ONE_CONJUGATE_Y = 2,
    ROCM_BLAS_V2_RANK_TWO = 3
};

typedef uint32_t rocm_blas_v2_structured_matrix_op;
enum {
    ROCM_BLAS_V2_STRUCTURED_MATMUL = 1,
    ROCM_BLAS_V2_STRUCTURED_RANK_K = 2,
    ROCM_BLAS_V2_STRUCTURED_RANK_2K = 3,
    ROCM_BLAS_V2_STRUCTURED_RANK_K_EXTENDED = 4
};

typedef uint32_t rocm_blas_v2_triangular_matrix_op;
enum {
    ROCM_BLAS_V2_TRIANGULAR_MATMUL = 1,
    ROCM_BLAS_V2_TRIANGULAR_SOLVE = 2,
    ROCM_BLAS_V2_TRIANGULAR_INVERSE = 3
};

typedef uint32_t rocm_blas_v2_matrix_transform_op;
enum { ROCM_BLAS_V2_MATRIX_ADD = 1, ROCM_BLAS_V2_DIAGONAL_MATRIX_MULTIPLY = 2 };

typedef uint32_t rocm_blas_v2_batch_kind;
enum {
    ROCM_BLAS_V2_BATCH_SINGLE = 1,
    ROCM_BLAS_V2_BATCH_POINTER_ARRAY = 2,
    ROCM_BLAS_V2_BATCH_STRIDED = 3,
    ROCM_BLAS_V2_BATCH_GROUPED = 4
};

typedef struct rocm_blas_v2_scalar {
    rocm_interfaces_abi_header header;
    rocblas_datatype data_type;
    rocblas_pointer_mode location;
    const void* value;
} rocm_blas_v2_scalar;

typedef struct rocm_blas_v2_execution {
    rocm_interfaces_abi_header header;
    void* stream;
    rocm_blas_v2_index_width index_width;
    rocm_blas_v2_batch_kind batch_kind;
    int64_t batch_count;
    uint64_t behavior_flags;
} rocm_blas_v2_execution;

typedef struct rocm_blas_v2_memory {
    rocm_interfaces_abi_header header;
    void* base;
    const void* const* pointer_array;
    int64_t element_offset;
    int64_t batch_stride;
} rocm_blas_v2_memory;

typedef struct rocm_blas_v2_vector {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_memory memory;
    rocblas_datatype data_type;
    int64_t length;
    int64_t increment;
} rocm_blas_v2_vector;

typedef struct rocm_blas_v2_matrix {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_memory memory;
    rocblas_datatype data_type;
    rocm_blas_v2_storage_kind storage;
    rocm_blas_v2_matrix_kind kind;
    rocblas_fill fill;
    rocblas_diagonal diagonal;
    int64_t rows;
    int64_t columns;
    int64_t leading_dimension;
    int64_t lower_bandwidth;
    int64_t upper_bandwidth;
} rocm_blas_v2_matrix;

typedef struct rocm_blas_v2_vector_transform_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_vector_transform_op operation;
    rocblas_datatype compute_type;
    rocm_blas_v2_scalar alpha;
    rocm_blas_v2_vector x;
    rocm_blas_v2_vector y;
} rocm_blas_v2_vector_transform_request;

typedef struct rocm_blas_v2_vector_reduce_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_vector_reduce_op operation;
    rocblas_datatype compute_type;
    rocm_blas_v2_vector x;
    rocm_blas_v2_vector y;
    rocblas_datatype result_type;
    rocblas_pointer_mode result_location;
    void* result;
} rocm_blas_v2_vector_reduce_request;

typedef struct rocm_blas_v2_rotation_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_rotation_op operation;
    rocblas_datatype compute_type;
    rocm_blas_v2_vector x;
    rocm_blas_v2_vector y;
    rocm_blas_v2_scalar parameters[5];
    // ROTM/ROTMG use a public, datatype-dependent five-element parameter
    // block. Keeping the block intact preserves its in/out and device-location
    // semantics without baking five host loads into the loader.
    rocm_blas_v2_memory parameter_block;
    rocblas_datatype parameter_type;
} rocm_blas_v2_rotation_request;

typedef struct rocm_blas_v2_matrix_vector_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_matrix_vector_op operation;
    rocblas_operation transpose;
    rocm_blas_v2_scalar alpha;
    rocm_blas_v2_scalar beta;
    rocm_blas_v2_matrix matrix;
    rocm_blas_v2_vector x;
    rocm_blas_v2_vector y;
} rocm_blas_v2_matrix_vector_request;

typedef struct rocm_blas_v2_rank_update_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_rank_update_op operation;
    rocm_blas_v2_scalar alpha;
    rocm_blas_v2_vector x;
    rocm_blas_v2_vector y;
    rocm_blas_v2_matrix matrix;
} rocm_blas_v2_rank_update_request;

typedef struct rocm_blas_v2_matmul_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocblas_operation operation_a;
    rocblas_operation operation_b;
    // Used by GEMMT; rocblas_fill_full means an ordinary GEMM.
    rocblas_fill output_fill;
    rocblas_datatype compute_type;
    rocm_blas_v2_scalar alpha;
    rocm_blas_v2_scalar beta;
    rocm_blas_v2_matrix a;
    rocm_blas_v2_matrix b;
    rocm_blas_v2_matrix c;
    rocm_blas_v2_matrix d;
    rocblas_gemm_algo public_algorithm;
    int32_t public_solution_index;
    uint32_t public_flags;
    uint64_t provider_algorithm_token[4];
    void* workspace;
    size_t workspace_size;
} rocm_blas_v2_matmul_request;

typedef uint32_t rocm_blas_v2_solution_outcome;
enum {
    ROCM_BLAS_V2_SOLUTION_EXECUTED = 1,
    ROCM_BLAS_V2_SOLUTION_NOT_FOUND = 2,
    ROCM_BLAS_V2_SOLUTION_REJECTED = 3
};

typedef struct rocm_blas_v2_solution {
    rocm_interfaces_abi_header header;
    uint64_t provider_algorithm_token[4];
    int32_t public_solution_index;
    uint32_t reserved;
    size_t workspace_size;
    uint64_t capability_mask;
} rocm_blas_v2_solution;

typedef struct rocm_blas_v2_matmul_result {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_solution_outcome outcome;
    size_t workspace_size;
} rocm_blas_v2_matmul_result;

typedef struct rocm_blas_v2_structured_matrix_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_structured_matrix_op operation;
    rocblas_datatype compute_type;
    rocblas_side side;
    rocblas_operation operation_a;
    rocblas_operation operation_b;
    rocm_blas_v2_scalar alpha;
    rocm_blas_v2_scalar beta;
    rocm_blas_v2_matrix a;
    rocm_blas_v2_matrix b;
    rocm_blas_v2_matrix c;
} rocm_blas_v2_structured_matrix_request;

typedef struct rocm_blas_v2_triangular_matrix_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_triangular_matrix_op operation;
    rocblas_datatype compute_type;
    rocblas_side side;
    rocblas_operation transpose;
    rocm_blas_v2_scalar alpha;
    rocm_blas_v2_matrix a;
    rocm_blas_v2_matrix b;
    rocm_blas_v2_matrix d;
    // Optional caller-supplied packed inverse used by the TRSM_EX spellings.
    // It is input storage, not provider workspace.
    rocm_blas_v2_memory inverse_a;
    int64_t inverse_a_size;
    void* workspace;
    size_t workspace_size;
} rocm_blas_v2_triangular_matrix_request;

typedef struct rocm_blas_v2_matrix_transform_request {
    rocm_interfaces_abi_header header;
    rocm_blas_v2_execution execution;
    rocm_blas_v2_matrix_transform_op operation;
    rocblas_datatype compute_type;
    uint32_t has_public_extended_operation;
    rocblas_geam_ex_operation public_extended_operation;
    int64_t auxiliary_dimension;
    rocblas_side side;
    rocblas_operation operation_a;
    rocblas_operation operation_b;
    rocm_blas_v2_scalar alpha;
    rocm_blas_v2_scalar beta;
    rocm_blas_v2_matrix a;
    rocm_blas_v2_matrix b;
    rocm_blas_v2_vector diagonal;
    rocm_blas_v2_matrix c;
    rocm_blas_v2_matrix d;
} rocm_blas_v2_matrix_transform_request;

typedef struct rocm_blas_v2_context_options {
    rocm_interfaces_abi_header header;
    const rocm_interfaces_host_services* host;
    rocm_interfaces_device_key device;
    uint64_t capability_requirements;
} rocm_blas_v2_context_options;

typedef struct rocm_blas_v2_provider {
    rocm_interfaces_abi_header header;
    rocblas_status(ROCM_INTERFACES_CALL* create_context)(const rocm_blas_v2_context_options*,
                                                         void**);
    void(ROCM_INTERFACES_CALL* destroy_context)(void*);
    rocblas_status(ROCM_INTERFACES_CALL* vector_transform)(
        void*, const rocm_blas_v2_vector_transform_request*);
    rocblas_status(ROCM_INTERFACES_CALL* vector_reduce)(void*,
                                                        const rocm_blas_v2_vector_reduce_request*);
    rocblas_status(ROCM_INTERFACES_CALL* vector_rotate)(void*,
                                                        const rocm_blas_v2_rotation_request*);
    rocblas_status(ROCM_INTERFACES_CALL* matrix_vector)(void*,
                                                        const rocm_blas_v2_matrix_vector_request*);
    rocblas_status(ROCM_INTERFACES_CALL* rank_update)(void*,
                                                      const rocm_blas_v2_rank_update_request*);
    rocblas_status(ROCM_INTERFACES_CALL* matmul_query)(void*, const rocm_blas_v2_matmul_request*,
                                                       rocm_blas_v2_solution*, size_t, size_t*);
    rocblas_status(ROCM_INTERFACES_CALL* matmul)(void*, const rocm_blas_v2_matmul_request*,
                                                 rocm_blas_v2_matmul_result*);
    rocblas_status(ROCM_INTERFACES_CALL* structured_matrix)(
        void*, const rocm_blas_v2_structured_matrix_request*);
    rocblas_status(ROCM_INTERFACES_CALL* triangular_matrix)(
        void*, const rocm_blas_v2_triangular_matrix_request*);
    rocblas_status(ROCM_INTERFACES_CALL* matrix_transform)(
        void*, const rocm_blas_v2_matrix_transform_request*);
} rocm_blas_v2_provider;

#ifdef __cplusplus
}
#endif
#endif
