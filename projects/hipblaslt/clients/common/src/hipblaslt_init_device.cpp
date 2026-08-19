// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_init.hpp"
#include "hipblaslt_test.hpp"

#include <hip/hip_runtime.h>

#include <optional>
#include <stdexcept>

namespace
{
    hipblaslt::host_validation::MatrixRole matrixRole(ABC_dims role)
    {
        using hipblaslt::host_validation::MatrixRole;

        switch(role)
        {
        case ABC_dims::A:
            return MatrixRole::A;
        case ABC_dims::B:
            return MatrixRole::B;
        case ABC_dims::C:
            return MatrixRole::C;
        }
        throw std::invalid_argument("Unsupported hipBLASLt matrix role.");
    }
} // namespace

void hipblaslt_init_device(
    ABC_dims                                                   abc,
    hipblaslt_initialization                                   init,
    bool                                                       is_nan,
    void*                                                      destination,
    size_t                                                     rows,
    size_t                                                     columns,
    size_t                                                     leadingDimension,
    hipDataType                                                type,
    size_t                                                     batchStride,
    size_t                                                     batchCount,
    bool                                                       positiveOnly,
    std::optional<hipblaslt::host_validation::OneSpecialValue> oneSpecialValue)
{
    using hipblaslt::host_validation::MatrixInitialization;
    using roc::host_validation::Tensor;

    MatrixInitialization initialization;
    initialization.role             = matrixRole(abc);
    initialization.initialization   = init;
    initialization.forceNaN         = is_nan;
    initialization.type             = type;
    initialization.rows             = rows;
    initialization.columns          = columns;
    initialization.leadingDimension = leadingDimension;
    initialization.batchStride      = batchStride;
    initialization.batchCount       = batchCount;
    initialization.oneSpecialValue  = oneSpecialValue;
    initialization.positiveOnly     = positiveOnly;

    Tensor     matrix  = hipblaslt::host_validation::generateMatrix(initialization);
    const auto storage = matrix.storage();
    if(!storage.empty())
    {
        if(destination == nullptr)
            throw std::invalid_argument("hipBLASLt device initialization destination is null.");
        CHECK_HIP_ERROR(
            hipMemcpy(destination, storage.data(), storage.size(), hipMemcpyHostToDevice));
    }
}
