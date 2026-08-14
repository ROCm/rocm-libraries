// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private adapter from hipBLASLt storage descriptors to the
// product-independent host-validation epilogue API.

#include <cstdint>
#include <hipblaslt/hipblaslt.h>
#include <roc/host_validation/validation.hpp>

namespace roc::host_validation::hipblaslt_adapter
{
    struct EpilogueArguments
    {
        int64_t               rows                  = 0;
        int64_t               columns               = 0;
        int64_t               leadingDimension      = 0;
        const void*           input                 = nullptr;
        void*                 output                = nullptr;
        void*                 rawOutput             = nullptr;
        void*                 amax                  = nullptr;
        bool                  accumulateAmax        = true;
        void*                 auxiliary             = nullptr;
        hipDataType           auxiliaryType         = HIP_R_32F;
        // Null scale pointers select the multiplicative identity.
        const void*           outputScale           = nullptr;
        const void*           auxiliaryScale        = nullptr;
        const void*           bias                  = nullptr;
        hipDataType           biasType              = HIP_R_32F;
        MatrixAxis            biasAxis              = MatrixAxis::Row;
        Activation            activation            = Activation::None;
        ActivationApplication activationApplication = ActivationApplication::Forward;
        double                activationParameter0  = 0.0;
        double                activationParameter1  = 0.0;
        hipDataType           outputType            = HIP_R_32F;
        hipDataType           computeType           = HIP_R_32F;
    };

    EpilogueRunInfo referenceEpilogue(const EpilogueArguments& arguments);
} // namespace roc::host_validation::hipblaslt_adapter
