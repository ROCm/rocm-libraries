// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "DataInitialization.hpp"
#include <mxDataGen.hpp>                          // hipDataType, generateMXInput
#include <mxDataGenerator/dataTypeInfo.hpp>       // DGen::toFloat / toFloatPacked
#include <mxDataGenerator/ocp_e2m1_mxfp4.hpp>     // DGen::ocp_e2m1_mxfp4
#include <mxDataGenerator/ocp_e2m3_mxfp6.hpp>     // DGen::ocp_e2m3_mxfp6
#include <mxDataGenerator/ocp_e3m2_mxfp6.hpp>     // DGen::ocp_e3m2_mxfp6
#include <mxDataGenerator/ocp_e4m3_mxfp8.hpp>     // DGen::ocp_e4m3_mxfp8
#include <mxDataGenerator/ocp_e5m2_mxfp8.hpp>     // DGen::ocp_e5m2_mxfp8
#include <hip/hip_runtime.h>                      // HIP_R_*

#include <algorithm>
#include <cmath>                                  // std::ldexp
#include <cstdint>
#include <cstring>                                // std::memset
#include <limits>
#include <stdexcept>
#include <string>
#include <fstream>

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
            // ----------------------------------------------------------------
            //  decodeE8M0  : matches mxScaleElementAsFloat(rocisa::DataType::E8,..)
            //                used by Reference.cpp - keep them in sync.
            //  decodeMXElement : thin dispatch over DGen::toFloatPacked / toFloat
            //                for supported MX element types.
            // ----------------------------------------------------------------
            inline float decodeE8M0(uint8_t b)
            {
                if(b == 0x00) return 0.0f;
                if(b == 0xFF) return std::numeric_limits<float>::quiet_NaN();
                return std::ldexp(1.0f, static_cast<int>(b) - 127);
            }
            inline float decodeMXElement(rocisa::DataType dataType,
                                         uint8_t const*   scalePtr,
                                         uint8_t const*   dataPtr,
                                         size_t           scaleIndex,
                                         size_t           elemIndex)
            {
                switch(dataType)
                {
                case rocisa::DataType::Float4:  // OCP E2M1, packed 2/byte
                    return DGen::toFloatPacked<DGen::ocp_e2m1_mxfp4>(
                        scalePtr, dataPtr, scaleIndex, elemIndex);
                case rocisa::DataType::Float6:  // OCP E2M3, packed 4/3 bytes
                    return DGen::toFloatPacked<DGen::ocp_e2m3_mxfp6>(
                        scalePtr, dataPtr, scaleIndex, elemIndex);
                case rocisa::DataType::BFloat6: // OCP E3M2, packed 4/3 bytes
                    return DGen::toFloatPacked<DGen::ocp_e3m2_mxfp6>(
                        scalePtr, dataPtr, scaleIndex, elemIndex);
                case rocisa::DataType::Float8:  // OCP E4M3
                    return DGen::toFloat<DGen::ocp_e4m3_mxfp8>(
                        scalePtr, dataPtr, scaleIndex, elemIndex);
                case rocisa::DataType::BFloat8: // OCP E5M2
                    return DGen::toFloat<DGen::ocp_e5m2_mxfp8>(
                        scalePtr, dataPtr, scaleIndex, elemIndex);
                default:
                    return std::numeric_limits<float>::quiet_NaN();
                }
            }
            // ----------------------------------------------------------------
            //  Forensic dump (debug-only). Writes four text files for one MX
            //  side. See file comment in DataInitialization.cpp for layout.
            //
            // Usage:
            //      dumpMXSideForDebug("dbg_A",
            //                         tensorA.dataType(),
            //                         pristineA.cpuInput.valid.get(),
            //                         problem.a(),
            //                         pristineE8A.cpuInput.valid.get(),
            //                         problem.mxsa(),
            //                         problem.mxBlockA());
            // ----------------------------------------------------------------
            inline void dumpMXSideForDebug(std::string const&      prefix,
                                           rocisa::DataType        dataType,
                                           void const*             dataPtr,
                                           TensorDescriptor const& tensor,
                                           void const*             scalePtr,
                                           TensorDescriptor const& scaleTensor,
                                           size_t                  mxBlock)
            {
                assert(tensor.sizes().size() <= 2
                       || (tensor.sizes().size() == 3 && tensor.sizes()[2] == 1));
                assert(scaleTensor.sizes().size() <= 2
                       || (scaleTensor.sizes().size() == 3 && scaleTensor.sizes()[2] == 1));
                auto const* dp = static_cast<uint8_t const*>(dataPtr);
                auto const* sp = static_cast<uint8_t const*>(scalePtr);
                size_t const dataBytes  = tensor.totalAllocatedBytes();
                size_t const scaleBytes = scaleTensor.totalAllocatedBytes();
                {   // (1) raw data bytes
                    std::ofstream f(prefix + "_data_bytes.txt");
                    f << std::hex << std::setfill('0');
                    for(size_t i = 0; i < dataBytes; ++i)
                        f << "0x" << std::setw(8) << i << "  0x" << std::setw(2)
                          << static_cast<unsigned>(dp[i]) << '\n';
                }
                {   // (2) raw scale bytes
                    std::ofstream f(prefix + "_scale_bytes.txt");
                    f << std::hex << std::setfill('0');
                    for(size_t i = 0; i < scaleBytes; ++i)
                        f << "0x" << std::setw(8) << i << "  0x" << std::setw(2)
                          << static_cast<unsigned>(sp[i]) << '\n';
                }
                {   // (3) scale decoded as float
                    std::ofstream f(prefix + "_scale_float.txt");
                    f << std::scientific << std::setprecision(9);
                    for(size_t i = 0; i < scaleBytes; ++i)
                        f << std::dec << i << ' ' << decodeE8M0(sp[i]) << '\n';
                }
                {   // (4) data as decoded float matrix
                    std::ofstream f(prefix + "_data_float.txt");
                    f << std::scientific << std::setprecision(9);
                    size_t const rows  = tensor.sizes()[0];
                    size_t const cols  = tensor.sizes()[1];
                    size_t const dStr0 = tensor.strides()[0];
                    size_t const dStr1 = tensor.strides()[1];
                    size_t const sStr0 = scaleTensor.strides()[0];
                    size_t const sStr1 = scaleTensor.strides()[1];
                    for(size_t r = 0; r < rows; ++r)
                    {
                        for(size_t c = 0; c < cols; ++c)
                        {
                            size_t const elemIdx = r * dStr0 + c * dStr1;
                            size_t const sR      = mxBlock ? (r / mxBlock) : 0;
                            size_t const sC      = c;
                            size_t const sIdx    = sR * sStr0 + sC * sStr1;
                            f << decodeMXElement(dataType, sp, dp, sIdx, elemIdx);
                            f << (c + 1 == cols ? '\n' : ' ');
                        }
                    }
                }
            }
        } // namespace detail
    } // namespace Client
} // namespace TensileLite
#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
