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

#ifndef HIPTENSOR_BACKEND_PROBLEM_HPP
#define HIPTENSOR_BACKEND_PROBLEM_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <hiptensor/hiptensor_types.h>

struct hiptensorOperationDescriptor;

namespace hiptensor
{
    class HipDevice;

    namespace backend
    {
        // Largest number of modes of one tensor that the backend interface carries. A backend
        // declines the problems it cannot serve; the interface never truncates them.
        inline constexpr int32_t kMaxModes = 64;

        enum class OpKind : uint8_t
        {
            Contraction,
            ContractionTrinary,
            Permute,
            ElementwiseBinary,
            ElementwiseTrinary,
            Reduction,
        };

        char const* opKindName(OpKind kind);

        // One tensor of a problem as the caller described it: modes in the caller's order and
        // strides in elements, all 64-bit.
        struct TensorArg
        {
            bool                 present = false;
            hiptensorDataType_t  type    = HIPTENSOR_R_32F;
            std::vector<int64_t> extents;
            std::vector<int64_t> strides;
            std::vector<int32_t> modes;
            hiptensorOperator_t  unaryOp        = HIPTENSOR_OP_IDENTITY;
            uint32_t             alignmentBytes = 0;

            int32_t numModes() const
            {
                return static_cast<int32_t>(extents.size());
            }

            // Number of elements spanned in memory: 1 + sum((extent - 1) * stride), 0 when the
            // tensor is absent.
            int64_t spanElements() const;
        };

        // The device a problem is planned for. Every field takes part in the problem key.
        struct DeviceKey
        {
            std::string archName; // full name with target features, e.g. "gfx942:sramecc+:xnack-"
            std::string baseArch; // without the target features, e.g. "gfx942"
            int32_t     cuCount  = 0;
            int32_t     waveSize = 0;
        };

        // The validated user problem of a plan. Tensor slots follow the operation descriptor:
        // A, B, C, D, E. The output is the tensor in slot outputSlot():
        //   Contraction            A, B, C (optional) -> D
        //   ContractionTrinary     A, B, C, D (optional) -> E
        //   Permute                A -> B
        //   ElementwiseBinary      A, C -> D
        //   ElementwiseTrinary     A, B, C -> D
        //   Reduction              A, C -> D
        struct Problem
        {
            OpKind                       kind = OpKind::Contraction;
            std::array<TensorArg, 5>     t;
            hiptensorOperator_t          opAB       = HIPTENSOR_OP_UNKNOWN;
            hiptensorOperator_t          opABC      = HIPTENSOR_OP_UNKNOWN;
            hiptensorOperator_t          opReduce   = HIPTENSOR_OP_UNKNOWN;
            hiptensorComputeDescriptor_t compute    = HIPTENSOR_COMPUTE_DESC_32F;
            hiptensorDataType_t          scalarType = HIPTENSOR_R_32F;
            DeviceKey                    device;

            int outputSlot() const;

            TensorArg const& output() const
            {
                return t[outputSlot()];
            }

            // Fixed-layout little-endian bytes of every field above.
            std::vector<uint8_t> keyBytes() const;

            // FNV-1a of keyBytes(), for logs.
            uint64_t digest() const;
        };

        // Builds the problem of an operation descriptor for a device. Extents and strides that
        // do not fit in int64_t, or tensors with more than kMaxModes modes, are
        // HIPTENSOR_STATUS_NOT_SUPPORTED; an unknown operation type is
        // HIPTENSOR_STATUS_INVALID_VALUE.
        hiptensorStatus_t makeProblem(hiptensorOperationDescriptor const& desc,
                                      HipDevice const&                    device,
                                      Problem&                            problem);

        // The device key of a device.
        DeviceKey makeDeviceKey(HipDevice const& device);
    } // namespace backend
} // namespace hiptensor

#endif // HIPTENSOR_BACKEND_PROBLEM_HPP
