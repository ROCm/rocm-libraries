// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "DataInitialization.hpp"
#include <mxDataGen.hpp>                          // hipDataType, generateMXInput
#include <mxDataGenerator/dataTypeInfo.hpp>       // DGen::toFloat / toFloatPacked
#include <mxDataGenerator/ocp_e2m1_mxfp4.hpp>     // DGen::ocp_e2m1_mxfp4
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
            //    Float8  -> HIP_R_8F_E4M3   (OCP E4M3, 1 elem / byte)
            //    BFloat8 -> HIP_R_8F_E5M2   (OCP E5M2, 1 elem / byte)
            // ----------------------------------------------------------------
            inline hipDataType hipMxDataTypeForDataGenerator(rocisa::DataType dataType)
            {
                switch(dataType)
                {
                case rocisa::DataType::Float4:
                    return static_cast<hipDataType>(HIP_R_4F_E2M1);
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
            //  randomFP4DataAndFixScales:
            //      scale[]  = scaleByte (default 0x7F = E8M0 bias 127 -> 2^0 = 1.0)
            //      data[i]  = high|low nibbles drawn from the OCP E2M1 set
            //                 {0x0,0x1,0x2, 0x8,0x9,0xA} so |value| <= 1.0.
            // Usage:
            //   randomFP4DataAndFixScales(pristineData.cpuInput.valid.get(),
            //                             dataTensor.totalAllocatedBytes(),
            //                             pristineE8.cpuInput.valid.get(),
            //                             scaleTensor.totalAllocatedBytes());
            // ----------------------------------------------------------------
            inline void randomFP4DataAndFixScales(void*   data,
                                                  size_t  numDataBytes,
                                                  void*   scale,
                                                  size_t  numScaleBytes,
                                                  uint8_t scaleByte = 0x7F)
            {
                std::memset(scale, scaleByte, numScaleBytes);
                static constexpr uint8_t kFP4Nibbles[6] = {0x0, 0x1, 0x2, 0x8, 0x9, 0xA};
                auto* bytes = static_cast<uint8_t*>(data);
                for(size_t i = 0; i < numDataBytes; ++i)
                {
                    uint8_t hi = kFP4Nibbles[getThreadLocalRandInt() % 6];
                    uint8_t lo = kFP4Nibbles[getThreadLocalRandInt() % 6];
                    bytes[i]   = static_cast<uint8_t>((hi << 4) | lo);
                }
            }
            // ----------------------------------------------------------------
            //  randomFP8DataAndFixScales:
            //      scale[]  = scaleByte
            //      data[i]  = sign(1b) | mag, mag uniform in [0, maxMag]
            //                 maxMag = 0x38 for E4M3 (= +1.0)
            //                 maxMag = 0x3C for E5M2 (= +1.0)
            //
            // Usage:
            //   randomFP8DataAndFixScales(dataTensor.dataType(),
            //                             pristineData.cpuInput.valid.get(),
            //                             dataTensor.totalAllocatedBytes(),
            //                             pristineE8.cpuInput.valid.get(),
            //                             scaleTensor.totalAllocatedBytes());
            // ----------------------------------------------------------------
            inline void randomFP8DataAndFixScales(rocisa::DataType dataType,
                                                  void*            data,
                                                  size_t           numDataBytes,
                                                  void*            scale,
                                                  size_t           numScaleBytes,
                                                  uint8_t          scaleByte = 0x7F)
            {
                std::memset(scale, scaleByte, numScaleBytes);
                uint8_t maxMagByte;
                if(dataType == rocisa::DataType::Float8)        // OCP E4M3
                    maxMagByte = 0x38;                          // exp=7,  m=0 -> +1.0
                else if(dataType == rocisa::DataType::BFloat8)  // OCP E5M2
                    maxMagByte = 0x3C;                          // exp=15, m=0 -> +1.0
                else
                    throw std::runtime_error(
                        "randomFP8DataAndFixScales: unsupported FP8 data type");
                int const numMagValues = static_cast<int>(maxMagByte) + 1;
                auto*     bytes        = static_cast<uint8_t*>(data);
                for(size_t i = 0; i < numDataBytes; ++i)
                {
                    uint8_t mag  = static_cast<uint8_t>(getThreadLocalRandInt() % numMagValues);
                    uint8_t sign = static_cast<uint8_t>((getThreadLocalRandInt() & 1) << 7);
                    bytes[i]     = static_cast<uint8_t>(sign | mag);
                }
            }
            // ----------------------------------------------------------------
            //  fixBytes / fixDataAndScaleBytes: brute "memset-everywhere" knobs.
            //  Useful for hand-checking the kernel against a single known byte
            //  (e.g. 0x30 in E4M3 == 2^-7 = 0.0078125).
            //
            // Usage:
            //
            //   fixBytes(pristineData.cpuInput.valid.get(),
            //            dataTensor.totalAllocatedBytes(),
            //            0x38);                       // every elem == +1.0
            //
            //
            //   fixDataAndScaleBytes(pristineData.cpuInput.valid.get(),
            //                        dataTensor.totalAllocatedBytes(),
            //                        pristineE8.cpuInput.valid.get(),
            //                        scaleTensor.totalAllocatedBytes(),
            //                        0x7E,            // scale = 2^-1 = 0.5
            //                        0x30);           // payload = 2^-7
            // ----------------------------------------------------------------
            inline void fixBytes(void* data, size_t numBytes, uint8_t fillByte = 0x30)
            {
                std::memset(data, fillByte, numBytes);
            }
            inline void fixDataAndScaleBytes(void*   data,
                                             size_t  numDataBytes,
                                             void*   scale,
                                             size_t  numScaleBytes,
                                             uint8_t scaleByte    = 0x7E,
                                             uint8_t dataFillByte = 0x30)
            {
                std::memset(scale, scaleByte, numScaleBytes);
                fixBytes(data, numDataBytes, dataFillByte);
            }
            // ----------------------------------------------------------------
            //  identityFP8DataAndFixScales:
            //      scale[]               = scaleByte (default 0x7F = scale 1.0)
            //      data[]                = 0x00
            //      data[i, i] (diagonal) = +1.0 byte for the dtype
            //          E4M3 (Float8)  -> 0x38  (exp=7,  m=0 -> 2^0 = 1.0)
            //          E5M2 (BFloat8) -> 0x3C  (exp=15, m=0 -> 2^0 = 1.0)
            //
            //  Per-batch independent identity is written when the data tensor
            //  is 3D. FP8 packing == 1, so the TensorDescriptor strides are
            //  byte offsets and totalAllocatedBytes() == totalAllocatedElements().
            //
            // Usage:
            //    identityFP8DataAndFixScales(tensorA.dataType(),
            //                                pristineA.cpuInput.valid.get(),
            //                                problem.a(),
            //                                pristineE8A.cpuInput.valid.get(),
            //                                problem.mxsa().totalAllocatedBytes());
            // ----------------------------------------------------------------
            inline void identityFP8DataAndFixScales(rocisa::DataType        dataType,
                                                    void*                   data,
                                                    TensorDescriptor const& dataTensor,
                                                    void*                   scale,
                                                    size_t                  numScaleBytes,
                                                    uint8_t                 scaleByte = 0x7F)
            {
                std::memset(scale, scaleByte, numScaleBytes);
                std::memset(data,  0x00,      dataTensor.totalAllocatedBytes());
                uint8_t oneByte;
                if(dataType == rocisa::DataType::Float8)
                    oneByte = 0x38;
                else if(dataType == rocisa::DataType::BFloat8)
                    oneByte = 0x3C;
                else
                    throw std::runtime_error(
                        "identityFP8DataAndFixScales: unsupported FP8 data type");
                auto const& sizes       = dataTensor.sizes();
                auto const& strides     = dataTensor.strides();
                size_t      rows        = sizes[0];
                size_t      cols        = sizes[1];
                size_t      strideRow   = strides[0];
                size_t      strideCol   = strides[1];
                size_t      batchCount  = sizes.size()   > 2 ? sizes[2]   : 1;
                size_t      batchStride = strides.size() > 2 ? strides[2] : 0;
                size_t      diag        = std::min(rows, cols);
                auto* bytes = static_cast<uint8_t*>(data);
                for(size_t b = 0; b < batchCount; ++b)
                {
                    uint8_t* batchPtr = bytes + b * batchStride;
                    for(size_t i = 0; i < diag; ++i)
                        batchPtr[i * strideRow + i * strideCol] = oneByte;
                }
            }
            // ----------------------------------------------------------------
            //  decodeE8M0  : matches mxScaleElementAsFloat(rocisa::DataType::E8,..)
            //                used by Reference.cpp - keep them in sync.
            //  decodeMXElement : thin dispatch over DGen::toFloatPacked / toFloat
            //                for the three supported element types.
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
