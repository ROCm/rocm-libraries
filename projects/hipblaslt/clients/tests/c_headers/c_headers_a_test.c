// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Must build as C, not C++: the public headers wrap their API in extern "C",
// and C consumers include them directly. This file links with c_headers_b_test.c,
// which includes the same headers, so an object defined with external linkage
// in a header fails the link. The binary is built, never run.

#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-local-kernel.h>
#include <hipblaslt/hipblaslt.h>

#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(hipblasLtBfloat16) == sizeof(uint16_t),
               "hipblasLtBfloat16 must stay a 16-bit type in C");

int hipblaslt_c_headers_b(int dataType);

int main(void)
{
    hipblasLtMatmulHeuristicResult_t result;
    memset(&result, 0, sizeof(result));
    result.state = HIPBLAS_STATUS_SUCCESS;

    hipblasLtBfloat16 bf16;
    bf16.data = 0;

    hipDataType          invalidType    = HIPBLASLT_DATATYPE_INVALID;
    hipblasComputeType_t invalidCompute = HIPBLASLT_COMPUTE_TYPE_INVALID;
    hipblasOperation_t   invalidOp      = HIPBLASLT_OPERATION_INVALID;

    (void)invalidCompute;
    (void)invalidOp;

    hipblasStatus_t (*create)(hipblasLtHandle_t*) = hipblasLtCreate;
    void (*setFlush)(bool)                        = hipblasltSetFlushValue;

    if(!create || !setFlush || result.state != HIPBLAS_STATUS_SUCCESS || bf16.data != 0)
        return 1;

    return hipblaslt_c_headers_b((int)invalidType);
}
