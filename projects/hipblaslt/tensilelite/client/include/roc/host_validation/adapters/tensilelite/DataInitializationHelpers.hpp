// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Product-private TensileLite adapter.

// TensileLite adapter over host-validation-owned MX generation.

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "DataInitialization.hpp"
#include <hip/hip_runtime.h>
#include <mxDataGen.hpp>
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
                switch(mxType)
                {
                case rocisa::DataType::Float8:
                    return HIP_R_8F_E4M3;
                case rocisa::DataType::E5M3:
                    return static_cast<hipDataType>(HIP_R_8F_E5M3_EXT);
                case rocisa::DataType::E8:
                case rocisa::DataType::None:
                    return HIP_R_8F_UE8M0;
                default:
                    throw std::runtime_error(
                        "initializeMXData: unsupported MX scale element type for generateMXInput");
                }
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
                switch(dataType)
                {
                case rocisa::DataType::Float4:
                    return static_cast<hipDataType>(HIP_R_4F_E2M1);
                case rocisa::DataType::Float6:
                    return static_cast<hipDataType>(HIP_R_6F_E2M3);
                case rocisa::DataType::BFloat6:
                    return static_cast<hipDataType>(HIP_R_6F_E3M2);
                case rocisa::DataType::Float8:
                    return HIP_R_8F_E4M3;
                case rocisa::DataType::BFloat8:
                    return HIP_R_8F_E5M2;
                default:
                    throw std::runtime_error(
                        "initializeMXData: unsupported MX data element type for generateMXInput");
                }
            }
        } // namespace detail
    } // namespace Client
} // namespace TensileLite
#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
