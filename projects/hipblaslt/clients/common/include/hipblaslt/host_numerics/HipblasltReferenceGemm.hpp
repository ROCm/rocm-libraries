// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt reference GEMM adapter.

#include "datatype_interface.hpp"

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/client/MatmulPreparation.hpp>
#include <roc/host_numerics/gemm.hpp>

#include <cstddef>
#include <optional>
#include <utility>

namespace hipblaslt::host_numerics
{
    struct MatmulReferenceInputs
    {
        MatmulReferenceInputs(const roc::host_numerics::Tensor& aTensor,
                              const roc::host_numerics::Tensor& bTensor,
                              const roc::host_numerics::Tensor& cTensor,
                              const roc::host_numerics::Tensor& dTensor)
            : a(aTensor)
            , b(bTensor)
            , c(cTensor)
            , d(dTensor)
        {
        }

        roc::host_numerics::Tensor                a;
        roc::host_numerics::Tensor                b;
        roc::host_numerics::Tensor                c;
        roc::host_numerics::Tensor                d;
        std::optional<roc::host_numerics::Tensor> alphaVector;
        std::optional<roc::host_numerics::Tensor> scaleA;
        std::optional<roc::host_numerics::Tensor> scaleB;
        std::optional<roc::host_numerics::Tensor> scaleC;
        std::optional<roc::host_numerics::Tensor> scaleD;
    };

    roc::host_numerics::Layout referenceBatchLayout(const hipblaslt::client::MatmulMatrix& matrix,
                                                    size_t                                 rows,
                                                    size_t                                 columns,
                                                    hipblasOperation_t operation,
                                                    size_t             batch,
                                                    bool               separateBatchStorage);

    void referenceMatmulGemm(const hipblaslt::client::MatmulProblem&         problem,
                             const hipblaslt::client::MatmulDataTypes&       dataTypes,
                             const hipblaslt::client::PreparedMatmulProblem& preparation,
                             const MatmulReferenceInputs&                    inputs,
                             hipblaslt_scaling_format                        scaleAMode,
                             hipblaslt_scaling_format                        scaleBMode);
} // namespace hipblaslt::host_numerics
