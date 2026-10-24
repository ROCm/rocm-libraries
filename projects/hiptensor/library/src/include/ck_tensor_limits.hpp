/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

#include <hiptensor/hiptensor.h>

#include "data_types.hpp"
#include "logger.hpp"

// Size limits of the tensors the Composable Kernel (CK) kernels are relied on to address
// correctly. Tensors beyond them have been read and written at wrong offsets without any error,
// or have made the kernels fail, so hiptensorCreatePlan refuses them with
// HIPTENSOR_STATUS_NOT_SUPPORTED. The limits are conservative: some tensors beyond them give
// correct results, but which ones depends on the operation, shape and layout.
namespace hiptensor
{
    // Tensors whose addressed range spans 2^31 bytes or more have given wrong results without an
    // error in permutations, elementwise operations and reductions, and have made contractions
    // fail during plan creation or execution. Complex tensors are split
    // into separate real and imaginary planes before they reach the kernels, so for them the
    // limit applies to one plane.
    constexpr int64_t ckMaxTensorBytes = int64_t(1) << 31;

    // f64 elementwise operations whose inputs are not laid out like the output wrote zeros for
    // tensors of 1 GiB, and were correct at 512 MiB. Until that is root-caused, such tensors must
    // span fewer than 512 MiB, and the same limit is applied to permutations.
    constexpr int64_t ckMaxF64PermutedTensorBytes = int64_t(1) << 29;

    // Bytes of one element as the kernels see it: one plane of a complex element.
    inline int64_t ckElementBytes(hiptensorDataType_t type)
    {
        auto bytes = static_cast<int64_t>(hiptensorDataTypeSize(type));
        return (type == HIPTENSOR_C_32F || type == HIPTENSOR_C_64F) ? bytes / 2 : bytes;
    }

    // Bytes from the lowest to the highest element a tensor addresses, inclusive, computed from
    // its lengths and strides with 64-bit arithmetic, so padded strides count. Zero for a tensor
    // without elements. A range of 2^31 elements or more is reported as exactly 2^31 elements,
    // which keeps every intermediate value in range for any lengths and strides.
    inline int64_t ckTensorSpanBytes(hiptensorTensorDescriptor const& desc)
    {
        constexpr int64_t maxElements  = int64_t(1) << 31;
        auto const        elementBytes = ckElementBytes(desc.mType);

        auto const& lengths = desc.mLengths;
        if(std::find(lengths.cbegin(), lengths.cend(), std::size_t(0)) != lengths.cend())
        {
            return 0;
        }

        // Offsets of the lowest and highest addressed element relative to element 0. Strides
        // are passed to the API as int64_t, so a negative stride lowers the lowest offset.
        int64_t lowest  = 0;
        int64_t highest = 0;
        for(std::size_t i = 0; i < lengths.size() && i < desc.mStrides.size(); i++)
        {
            auto const stride = static_cast<int64_t>(desc.mStrides[i]);
            if(lengths[i] == 1 || stride == 0)
            {
                continue;
            }
            if(lengths[i] - 1 >= static_cast<std::size_t>(maxElements) || stride >= maxElements
               || stride <= -maxElements)
            {
                return maxElements * elementBytes;
            }

            // |reach| < 2^62, and highest - lowest < 2^31 before this step.
            auto const reach = static_cast<int64_t>(lengths[i] - 1) * stride;
            (reach > 0 ? highest : lowest) += reach;
            if(highest - lowest >= maxElements)
            {
                return maxElements * elementBytes;
            }
        }
        return (highest - lowest + 1) * elementBytes;
    }

    // Describes why the kernels cannot address tensor `name`, or returns an empty string.
    inline std::string ckTensorSizeViolation(hiptensorTensorDescriptor const& desc,
                                             char const*                      name)
    {
        auto const bytes = ckTensorSpanBytes(desc);
        if(bytes < ckMaxTensorBytes)
        {
            return {};
        }

        bool const isComplex = desc.mType == HIPTENSOR_C_32F || desc.mType == HIPTENSOR_C_64F;
        char       msg[256];
        snprintf(msg,
                 sizeof(msg),
                 "Tensor %s spans at least %lld bytes%s; at most %lld bytes are supported",
                 name,
                 static_cast<long long>(bytes),
                 isComplex ? " per real or imaginary plane" : "",
                 static_cast<long long>(ckMaxTensorBytes - 1));
        return msg;
    }

    // Whether an elementwise input addresses its elements at the output's offsets: it carries
    // every output mode of extent > 1 with the output's stride.
    inline bool hasElementwiseOutputLayout(hiptensorTensorDescriptor const& in,
                                           std::vector<int32_t> const&      modeIn,
                                           hiptensorTensorDescriptor const& out,
                                           std::vector<int32_t> const&      modeOut)
    {
        for(std::size_t i = 0; i < modeOut.size(); i++)
        {
            if(i >= out.mLengths.size() || i >= out.mStrides.size() || out.mLengths[i] <= 1)
            {
                continue;
            }
            auto it = std::find(modeIn.cbegin(), modeIn.cend(), modeOut[i]);
            if(it == modeIn.cend())
            {
                return false;
            }
            auto j = static_cast<std::size_t>(std::distance(modeIn.cbegin(), it));
            if(j >= in.mStrides.size() || in.mStrides[j] != out.mStrides[i])
            {
                return false;
            }
        }
        return true;
    }

    // Describes the first tensor of an operation that the kernels cannot address, or returns an
    // empty string when they can address all of them.
    inline std::string ckTensorLimitViolation(hiptensorOperationDescriptor const& desc)
    {
        struct Operand
        {
            char const*                      name;
            hiptensorTensorDescriptor const* tensor;
            std::vector<int32_t> const*      modes;
        };
        Operand const operands[] = {{"A", desc.mDescA, &desc.mModeA},
                                    {"B", desc.mDescB, &desc.mModeB},
                                    {"C", desc.mDescC, &desc.mModeC},
                                    {"D", desc.mDescD, &desc.mModeD},
                                    {"E", desc.mDescE, &desc.mModeE}};

        for(auto const& operand : operands)
        {
            if(operand.tensor == nullptr)
            {
                continue;
            }
            if(auto violation = ckTensorSizeViolation(*operand.tensor, operand.name);
               !violation.empty())
            {
                return violation;
            }
        }

        // f64 permutations and elementwise operations with any input laid out differently from
        // the output, including a broadcast input.
        bool const isElementwise = desc.mOperationType == HIPTENSOR_PERMUTATION
                                   || desc.mOperationType == HIPTENSOR_ELEMENTWISE_BINARY
                                   || desc.mOperationType == HIPTENSOR_ELEMENTWISE_TRINARY;
        if(!isElementwise)
        {
            return {};
        }

        auto const& output
            = desc.mOperationType == HIPTENSOR_PERMUTATION ? operands[1] : operands[3];
        if(output.tensor == nullptr)
        {
            return {};
        }

        bool hasF64   = false;
        bool permuted = false;
        for(auto const& operand : operands)
        {
            if(operand.tensor == nullptr)
            {
                continue;
            }
            hasF64 |= operand.tensor->mType == HIPTENSOR_R_64F;
            if(&operand != &output)
            {
                permuted |= !hasElementwiseOutputLayout(
                    *operand.tensor, *operand.modes, *output.tensor, *output.modes);
            }
        }
        if(!hasF64 || !permuted)
        {
            return {};
        }

        for(auto const& operand : operands)
        {
            if(operand.tensor == nullptr)
            {
                continue;
            }
            auto const bytes = ckTensorSpanBytes(*operand.tensor);
            if(bytes >= ckMaxF64PermutedTensorBytes)
            {
                char msg[256];
                snprintf(msg,
                         sizeof(msg),
                         "Tensor %s spans at least %lld bytes; f64 elementwise operations whose "
                         "inputs are laid out differently from the output support at most %lld "
                         "bytes per tensor",
                         operand.name,
                         static_cast<long long>(bytes),
                         static_cast<long long>(ckMaxF64PermutedTensorBytes - 1));
                return msg;
            }
        }
        return {};
    }

    inline bool exceedsCkTensorLimits(hiptensorOperationDescriptor const& desc)
    {
        return !ckTensorLimitViolation(desc).empty();
    }

    // Logs a violation found by the functions above and returns HIPTENSOR_STATUS_NOT_SUPPORTED,
    // or returns HIPTENSOR_STATUS_SUCCESS for an empty violation.
    inline hiptensorStatus_t reportCkTensorLimitViolation(std::string const& violation,
                                                          char const*        apiFuncName)
    {
        if(violation.empty())
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }

        auto const errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        char       msg[512];
        snprintf(msg,
                 sizeof(msg),
                 "Unsupported Tensor Size Error : %s (%s)",
                 violation.c_str(),
                 hiptensorGetErrorString(errorCode));
        Logger::instance()->logError(apiFuncName, msg);
        return errorCode;
    }

    // HIPTENSOR_STATUS_NOT_SUPPORTED, logged, when the kernels cannot address a tensor of `desc`.
    inline hiptensorStatus_t checkCkTensorLimits(hiptensorOperationDescriptor const& desc,
                                                 char const*                         apiFuncName)
    {
        return reportCkTensorLimitViolation(ckTensorLimitViolation(desc), apiFuncName);
    }

} // namespace hiptensor
