// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Product-private TensileLite adapter.

// TensileLite adapter over host-validation-owned MX generation.

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "DataInitialization.hpp"
#include <hipblaslt/host_validation/Types.hpp>
#include <hip/hip_runtime.h>
#include <mxDataGen.hpp>
#include <roc/host_validation/adapters/tensilelite/HostValidationBridge.hpp>
#include <stdexcept>

namespace TensileLite
{
    namespace Client
    {
        namespace detail
        {
            // ----------------------------------------------------------------
            //  Maps Tensile MX *scale* element type to hipDataType for
            //  generateMXInput (mxDataGen).
            // ----------------------------------------------------------------
            inline hipDataType hipMxScaleTypeForDataGenerator(rocisa::DataType mxType)
            {
                return hipblaslt::host_validation::hipDataTypeForScalarType(
                    toHostValidationMxScaleType(mxType));
            }
            // ----------------------------------------------------------------
            //  MX *data*-element dtype mapper. generateMXInput() takes a
            //  hipDataType for the data tensor too:
            //    Float4  -> HIP_R_4F_E2M1   (OCP E2M1, 2 elems / byte)
            //    Float6  -> HIP_R_6F_E2M3   (OCP E2M3, 4 elems / 3 bytes)
            //    BFloat6 -> HIP_R_6F_E3M2   (OCP E3M2, 4 elems / 3 bytes)
            //    Float8  -> HIP_R_8F_E4M3   (OCP E4M3, 1 elem / byte)
            //    BFloat8 -> HIP_R_8F_E5M2   (OCP E5M2, 1 elem / byte)
            // ----------------------------------------------------------------
            inline hipDataType hipMxDataTypeForDataGenerator(rocisa::DataType dataType)
            {
                const auto scalarType = toHostValidationScalarType(dataType);
                switch(scalarType)
                {
                case roc::host_validation::ScalarType::Float4E2M1:
                case roc::host_validation::ScalarType::Float6E2M3:
                case roc::host_validation::ScalarType::Float6E3M2:
                case roc::host_validation::ScalarType::Float8E4M3:
                case roc::host_validation::ScalarType::Float8E5M2:
                    return hipblaslt::host_validation::hipDataTypeForScalarType(scalarType);
                default:
                    throw std::invalid_argument(
                        "initializeMXData: unsupported MX data element type for generateMXInput");
                }
            }
        } // namespace detail
    } // namespace Client
} // namespace TensileLite
#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
