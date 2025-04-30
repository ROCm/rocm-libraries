/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief hipsolver-types.h defines data types used by hipsolver
 */

#ifndef HIPSOLVER_TYPES_H
#define HIPSOLVER_TYPES_H

#if defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__)
#include <cusolver_common.h>
#endif

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hip/library_types.h>

typedef void* hipsolverHandle_t;

typedef void* hipsolverGesvdjInfo_t;
typedef void* hipsolverSyevjInfo_t;

typedef enum
{
    HIPSOLVER_STATUS_SUCCESS           = 0, // Function succeeds
    HIPSOLVER_STATUS_NOT_INITIALIZED   = 1, // hipSOLVER library not initialized
    HIPSOLVER_STATUS_ALLOC_FAILED      = 2, // resource allocation failed
    HIPSOLVER_STATUS_INVALID_VALUE     = 3, // unsupported numerical value was passed to function
    HIPSOLVER_STATUS_MAPPING_ERROR     = 4, // access to GPU memory space failed
    HIPSOLVER_STATUS_EXECUTION_FAILED  = 5, // GPU program failed to execute
    HIPSOLVER_STATUS_INTERNAL_ERROR    = 6, // an internal hipSOLVER operation failed
    HIPSOLVER_STATUS_NOT_SUPPORTED     = 7, // function not implemented
    HIPSOLVER_STATUS_ARCH_MISMATCH     = 8,
    HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9, // hipSOLVER handle is null pointer
    HIPSOLVER_STATUS_INVALID_ENUM      = 10, // unsupported enum value was passed to function
    HIPSOLVER_STATUS_UNKNOWN           = 11, // back-end returned an unsupported status code
    HIPSOLVER_STATUS_ZERO_PIVOT        = 12,
    HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 13,
} hipsolverStatus_t;

#ifndef HIPBLAS_OPERATION_DECLARED
#define HIPBLAS_OPERATION_DECLARED
/*! \brief Used to specify whether the matrix is to be transposed or not. */
typedef enum
{
    HIPBLAS_OP_N = 111, /**<  Operate with the matrix. */
    HIPBLAS_OP_T = 112, /**<  Operate with the transpose of the matrix. */
    HIPBLAS_OP_C = 113 /**< Operate with the conjugate transpose of the matrix. */
} hipblasOperation_t;

#elif __cplusplus >= 201103L
static_assert(HIPBLAS_OP_N == 111, "Inconsistent declaration of HIPBLAS_OP_N");
static_assert(HIPBLAS_OP_T == 112, "Inconsistent declaration of HIPBLAS_OP_T");
static_assert(HIPBLAS_OP_C == 113, "Inconsistent declaration of HIPBLAS_OP_C");
#endif // HIPBLAS_OPERATION_DECLARED

#ifndef HIPBLAS_FILL_MODE_DECLARED
#define HIPBLAS_FILL_MODE_DECLARED
/*! \brief Used by the Hermitian, symmetric and triangular matrix routines to specify whether the upper or lower triangle is being referenced. */
typedef enum
{
    HIPBLAS_FILL_MODE_UPPER = 121, /**<  Upper triangle */
    HIPBLAS_FILL_MODE_LOWER = 122, /**<  Lower triangle */
    HIPBLAS_FILL_MODE_FULL  = 123
} hipblasFillMode_t;

#elif __cplusplus >= 201103L
static_assert(HIPBLAS_FILL_MODE_UPPER == 121,
              "Inconsistent declaration of HIPBLAS_FILL_MODE_UPPER");
static_assert(HIPBLAS_FILL_MODE_LOWER == 122,
              "Inconsistent declaration of HIPBLAS_FILL_MODE_LOWER");
static_assert(HIPBLAS_FILL_MODE_FULL == 123, "Inconsistent declaration of HIPBLAS_FILL_MODE_FULL");
#endif // HIPBLAS_FILL_MODE_DECLARED

#ifndef HIPBLAS_DIAG_TYPE_DECLARED
#define HIPBLAS_DIAG_TYPE_DECLARED
/*! \brief It is used by the triangular matrix routines to specify whether the matrix is unit triangular.*/
typedef enum
{
    HIPBLAS_DIAG_NON_UNIT = 131, /**<  Non-unit triangular. */
    HIPBLAS_DIAG_UNIT     = 132 /**<  Unit triangular. */
} hipblasDiagType_t;

#elif __cplusplus >= 201103L
static_assert(HIPBLAS_DIAG_NON_UNIT == 131, "Inconsistent declaration of HIPBLAS_DIAG_NON_UNIT");
static_assert(HIPBLAS_DIAG_UNIT == 132, "Inconsistent declaration of HIPBLAS_DIAG_UNIT");
#endif // HIPBLAS_DIAG_TYPE_DECLARED

#ifndef HIPBLAS_SIDE_MODE_DECLARED
#define HIPBLAS_SIDE_MODE_DECLARED
/*! \brief Indicates the side matrix A is located relative to matrix B during multiplication. */
typedef enum
{
    HIPBLAS_SIDE_LEFT
    = 141, /**< Multiply general matrix by symmetric, Hermitian or triangular matrix on the left. */
    HIPBLAS_SIDE_RIGHT
    = 142, /**< Multiply general matrix by symmetric, Hermitian or triangular matrix on the right. */
    HIPBLAS_SIDE_BOTH = 143
} hipblasSideMode_t;

#elif __cplusplus >= 201103L
static_assert(HIPBLAS_SIDE_LEFT == 141, "Inconsistent declaration of HIPBLAS_SIDE_LEFT");
static_assert(HIPBLAS_SIDE_RIGHT == 142, "Inconsistent declaration of HIPBLAS_SIDE_RIGHT");
static_assert(HIPBLAS_SIDE_BOTH == 143, "Inconsistent declaration of HIPBLAS_SIDE_BOTH");
#endif // HIPBLAS_SIDE_MODE_DECLARED

typedef enum
{
    HIPSOLVER_EIG_MODE_NOVECTOR = 201,
    HIPSOLVER_EIG_MODE_VECTOR   = 202,
} hipsolverEigMode_t;

typedef enum
{
    HIPSOLVER_EIG_TYPE_1 = 211,
    HIPSOLVER_EIG_TYPE_2 = 212,
    HIPSOLVER_EIG_TYPE_3 = 213,
} hipsolverEigType_t;

typedef enum
{
    HIPSOLVER_EIG_RANGE_ALL = 221,
    HIPSOLVER_EIG_RANGE_V   = 222,
    HIPSOLVER_EIG_RANGE_I   = 223,
} hipsolverEigRange_t;

typedef enum
{
    HIPSOLVER_DETERMINISTIC_RESULTS           = 241,
    HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS = 242,
} hipsolverDeterministicMode_t;

// Aliases for hipBLAS enums

/*! \brief Alias of hipblasOperation_t. HIPSOLVER_OP_N, HIPSOLVER_OP_T, and HIPSOLVER_OP_C
 *  are provided as equivalents to HIPBLAS_OP_N, HIPBLAS_OP_T, and HIPBLAS_OP_C.
 ********************************************************************************/
typedef hipblasOperation_t hipsolverOperation_t;
#define HIPSOLVER_OP_N HIPBLAS_OP_N
#define HIPSOLVER_OP_T HIPBLAS_OP_T
#define HIPSOLVER_OP_C HIPBLAS_OP_C

/*! \brief Alias of hipblasFillMode_t. HIPSOLVER_FILL_MODE_UPPER and HIPSOLVER_FILL_MODE_LOWER
 *  are provided as equivalents to HIPBLAS_FILL_MODE_UPPER and HIPBLAS_FILL_MODE_LOWER.
 ********************************************************************************/
typedef hipblasFillMode_t hipsolverFillMode_t;
#define HIPSOLVER_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define HIPSOLVER_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER

/*! \brief Alias of hipblasSideMode_t. HIPSOLVER_SIDE_LEFT and HIPSOLVER_SIDE_RIGHT
 *  are provided as equivalents to HIPBLAS_SIDE_LEFT and HIPBLAS_SIDE_RIGHT.
 ********************************************************************************/
typedef hipblasSideMode_t hipsolverSideMode_t;
#define HIPSOLVER_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define HIPSOLVER_SIDE_RIGHT HIPBLAS_SIDE_RIGHT

#endif // HIPSOLVER_TYPES_H
