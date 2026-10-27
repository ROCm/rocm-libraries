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

#include <limits>

#include "data_types.hpp"
#include "hip_device.hpp"
#include "problem.hpp"

namespace hiptensor
{
    namespace backend
    {
        namespace
        {
            constexpr uint64_t fnvOffsetBasis = 14695981039346656037ull;
            constexpr uint64_t fnvPrime       = 1099511628211ull;

            class KeyWriter
            {
            public:
                void u8(uint8_t value)
                {
                    mBytes.push_back(value);
                }

                void u32(uint32_t value)
                {
                    for(int i = 0; i < 4; i++)
                    {
                        mBytes.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFu));
                    }
                }

                void i64(int64_t value)
                {
                    auto bits = static_cast<uint64_t>(value);
                    for(int i = 0; i < 8; i++)
                    {
                        mBytes.push_back(static_cast<uint8_t>((bits >> (8 * i)) & 0xFFu));
                    }
                }

                void str(std::string const& value)
                {
                    u32(static_cast<uint32_t>(value.size()));
                    mBytes.insert(mBytes.end(), value.begin(), value.end());
                }

                std::vector<uint8_t> take()
                {
                    return std::move(mBytes);
                }

            private:
                std::vector<uint8_t> mBytes;
            };

            // Converts the lengths or strides of a descriptor; false if a value does not fit.
            bool toInt64(std::vector<std::size_t> const& in, std::vector<int64_t>& out)
            {
                out.clear();
                out.reserve(in.size());
                for(auto value : in)
                {
                    if(value > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
                    {
                        return false;
                    }
                    out.push_back(static_cast<int64_t>(value));
                }
                return true;
            }

            hiptensorStatus_t makeTensor(hiptensorTensorDescriptor const* desc,
                                         std::vector<int32_t> const&      modes,
                                         hiptensorOperator_t              unaryOp,
                                         TensorArg&                       tensor)
            {
                tensor = TensorArg{};
                if(desc == nullptr)
                {
                    return HIPTENSOR_STATUS_SUCCESS;
                }

                if(desc->mLengths.size() > static_cast<std::size_t>(kMaxModes)
                   || !toInt64(desc->mLengths, tensor.extents)
                   || !toInt64(desc->mStrides, tensor.strides))
                {
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                tensor.present        = true;
                tensor.type           = desc->mType;
                tensor.modes          = modes;
                tensor.unaryOp        = unaryOp;
                tensor.alignmentBytes = desc->mAlignmentRequirement;
                return HIPTENSOR_STATUS_SUCCESS;
            }

            std::string baseArchOf(std::string const& archName)
            {
                return archName.substr(0, archName.find(':'));
            }
        } // namespace

        char const* opKindName(OpKind kind)
        {
            switch(kind)
            {
            case OpKind::Contraction:
                return "contraction";
            case OpKind::ContractionTrinary:
                return "trinary contraction";
            case OpKind::Permute:
                return "permutation";
            case OpKind::ElementwiseBinary:
                return "elementwise binary";
            case OpKind::ElementwiseTrinary:
                return "elementwise trinary";
            case OpKind::Reduction:
                return "reduction";
            }
            return "unknown";
        }

        int64_t TensorArg::spanElements() const
        {
            if(!present)
            {
                return 0;
            }
            int64_t span = 1;
            for(std::size_t i = 0; i < extents.size(); i++)
            {
                int64_t const stride = i < strides.size() ? strides[i] : 0;
                span += (extents[i] - 1) * stride;
            }
            return span;
        }

        int Problem::outputSlot() const
        {
            switch(kind)
            {
            case OpKind::Permute:
                return 1;
            case OpKind::ContractionTrinary:
                return 4;
            default:
                return 3;
            }
        }

        std::vector<uint8_t> Problem::keyBytes() const
        {
            KeyWriter writer;
            writer.u8(static_cast<uint8_t>(kind));
            for(auto const& tensor : t)
            {
                writer.u8(tensor.present ? 1u : 0u);
                if(!tensor.present)
                {
                    continue;
                }
                writer.u32(static_cast<uint32_t>(tensor.type));
                writer.u32(static_cast<uint32_t>(tensor.unaryOp));
                writer.u32(tensor.alignmentBytes);
                writer.u32(static_cast<uint32_t>(tensor.extents.size()));
                for(auto extent : tensor.extents)
                {
                    writer.i64(extent);
                }
                writer.u32(static_cast<uint32_t>(tensor.strides.size()));
                for(auto stride : tensor.strides)
                {
                    writer.i64(stride);
                }
                writer.u32(static_cast<uint32_t>(tensor.modes.size()));
                for(auto mode : tensor.modes)
                {
                    writer.u32(static_cast<uint32_t>(mode));
                }
            }
            writer.u32(static_cast<uint32_t>(opAB));
            writer.u32(static_cast<uint32_t>(opABC));
            writer.u32(static_cast<uint32_t>(opReduce));
            writer.u32(static_cast<uint32_t>(compute));
            writer.u32(static_cast<uint32_t>(scalarType));
            writer.str(device.archName);
            writer.str(device.baseArch);
            writer.u32(static_cast<uint32_t>(device.cuCount));
            writer.u32(static_cast<uint32_t>(device.waveSize));
            return writer.take();
        }

        uint64_t Problem::digest() const
        {
            uint64_t hash = fnvOffsetBasis;
            for(auto byte : keyBytes())
            {
                hash = (hash ^ byte) * fnvPrime;
            }
            return hash;
        }

        hiptensorStatus_t makeProblem(hiptensorOperationDescriptor const& desc,
                                      HipDevice const&                    device,
                                      Problem&                            problem)
        {
            problem = Problem{};
            switch(desc.mOperationType)
            {
            case HIPTENSOR_CONTRACTION:
                problem.kind = OpKind::Contraction;
                break;
            case HIPTENSOR_CONTRACTION_TRINARY:
                problem.kind = OpKind::ContractionTrinary;
                break;
            case HIPTENSOR_PERMUTATION:
                problem.kind = OpKind::Permute;
                break;
            case HIPTENSOR_ELEMENTWISE_BINARY:
                problem.kind = OpKind::ElementwiseBinary;
                problem.opAB = desc.mOpAC;
                break;
            case HIPTENSOR_ELEMENTWISE_TRINARY:
                problem.kind  = OpKind::ElementwiseTrinary;
                problem.opAB  = desc.mOpAC;
                problem.opABC = desc.mOpABC;
                break;
            case HIPTENSOR_REDUCTION:
                problem.kind     = OpKind::Reduction;
                problem.opReduce = desc.mOpAC;
                break;
            default:
                return HIPTENSOR_STATUS_INVALID_VALUE;
            }

            hiptensorTensorDescriptor const* descs[]
                = {desc.mDescA, desc.mDescB, desc.mDescC, desc.mDescD, desc.mDescE};
            std::vector<int32_t> const* modes[]
                = {&desc.mModeA, &desc.mModeB, &desc.mModeC, &desc.mModeD, &desc.mModeE};
            hiptensorOperator_t const ops[]
                = {desc.mOpA, desc.mOpB, desc.mOpC, desc.mOpD, HIPTENSOR_OP_IDENTITY};
            for(int i = 0; i < 5; i++)
            {
                if(auto status = makeTensor(descs[i], *modes[i], ops[i], problem.t[i]);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
            }
            if(!problem.output().present)
            {
                return HIPTENSOR_STATUS_INVALID_VALUE;
            }

            problem.compute    = desc.mDescCompute;
            problem.scalarType = desc.mScalarType;
            problem.device     = makeDeviceKey(device);
            return HIPTENSOR_STATUS_SUCCESS;
        }

        DeviceKey makeDeviceKey(HipDevice const& device)
        {
            DeviceKey key;
            key.archName = device.getArchName();
            key.baseArch = baseArchOf(key.archName);
            key.cuCount  = device.getCuCount();
            key.waveSize = device.warpSize();
            return key;
        }
    } // namespace backend
} // namespace hiptensor
