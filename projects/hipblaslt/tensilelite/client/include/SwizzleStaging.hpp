// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Host-side staging of a swizzled operand (SwizzleTensorA/B). Plain host code on top of
// TensorDataManipulation: it does not depend on mxDataGenerator.
#pragma once

#include "TensorDataManipulation.hpp"

#include <Tensile/DataTypes.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        namespace detail
        {
            inline size_t roundUpTo(size_t x, size_t unit)
            {
                return (x + unit - 1) / unit * unit;
            }

            // ---- swizzled operand layout ----
            inline void calculateKforSwizzling(rocisa::DataType datatype,
                                               size_t&          MiK,
                                               size_t&          MiKv,
                                               size_t&          PackK)
            {
                // The same K block on every architecture. gfx1250's kernel reads 32-byte K blocks and
                // gfx942's 64-byte ones, but flattening (kO, kM) gives the 16-byte K block index in both,
                // so the staged bytes only depend on the padded K (see swizzledPaddedKUnit).
                switch(datatype)
                {
                case rocisa::DataType::Float:
                    MiK  = 4;
                    MiKv = 1;
                    break;
                case rocisa::DataType::Double:
                    MiK  = 4;
                    MiKv = 1;
                    break;
                case rocisa::DataType::XFloat32:
                    MiK  = 8;
                    MiKv = 2;
                    break;
                case rocisa::DataType::Half:
                case rocisa::DataType::BFloat16:
                    MiK  = 16;
                    MiKv = 4;
                    break;
                case rocisa::DataType::Int8:
                    MiK  = 32;
                    MiKv = 8;
                    break;
                case rocisa::DataType::Float8_fnuz:
                case rocisa::DataType::BFloat8_fnuz:
                case rocisa::DataType::Float8BFloat8_fnuz:
                case rocisa::DataType::BFloat8Float8_fnuz:
                case rocisa::DataType::Float8:
                case rocisa::DataType::BFloat8:
                case rocisa::DataType::Float8BFloat8:
                case rocisa::DataType::BFloat8Float8:
                case rocisa::DataType::E8:
                case rocisa::DataType::E5M3:
                    // BK = MiK * PackK = 64 elements (64 B); PackK = 16/MiKv/1 = 2.
                    MiK  = 32;
                    MiKv = 8;
                    break;
                case rocisa::DataType::Float4:
                    MiK  = 32;
                    MiKv = 16;
                    break;
                default:
                    throw std::runtime_error("unsupported datatype for swizzling");
                }

                PackK = 16 / MiKv / rocisa::GetElementSize(datatype);
            }

            // Whether the K of a swizzled tensor is padded to 64 bytes rather than to its swizzle K block.
            // The kernel's K stride is 64 B for every swizzle path except the gfx950 subtile swizzle of
            // F4 (UseSubtileImpl; MXF4 -> bf16, TN), whose pre-shuffled block stride (swizzleGeometry
            // layoutK = miInput * 16 / (miKv * bpe), 64 F4 elements for wave64) is 32 B. 64 B is also what
            // the public orders say (COL16_4R32 for F4: 128 K elements; 4R16 F8: 64; 4R8 F16: 32).
            // gfx1250 does not take the subtile swizzle (Solution rejects it), so no wave-size term here.
            inline bool swizzledKPadsTo64Bytes(rocisa::DataType dtype, bool subtileSwizzle)
            {
                return !(subtileSwizzle && dtype == rocisa::DataType::Float4);
            }

            // K elements a swizzled tensor's rows are padded to: its swizzle K block, or 64 bytes when
            // padTo64Bytes is set. For the 1- and 2-byte types the swizzle K block is already 64 B, so the
            // flag only matters for F4 (swizzle K block 32 B, see swizzledKPadsTo64Bytes).
            inline size_t swizzledPaddedKUnit(size_t swizzleK, float elementBytes, bool padTo64Bytes)
            {
                if(!padTo64Bytes)
                    return swizzleK;
                const size_t padK = static_cast<size_t>(64.0f / elementBytes);
                return std::max(swizzleK, padK);
            }

            inline size_t getSwizzledTensorNumAllocatedElements(const TensorDescriptor& desc,
                                                                size_t                  miM_N,
                                                                size_t                  miK,
                                                                size_t                  packK,
                                                                bool                    padTo64Bytes)
            {
                // TODO: currently [0][1] = k, (m or n) is based on TN, need to make this generic in the future
                const auto k         = desc.sizes()[0];
                const auto m_n       = desc.sizes()[1];
                const auto b         = desc.sizes()[2];
                const auto swizzleK
                    = swizzledPaddedKUnit(miK * packK, desc.elementBytes(), padTo64Bytes);
                const auto paddedM_N = roundUpTo(m_n, miM_N);
                const auto paddedK   = roundUpTo(k, swizzleK);
                const auto perBatch  = paddedM_N * paddedK;
                // Batches sit at the problem's batch stride (see copySwizzledToGPUBuffer). The same
                // allocation also holds the unswizzled source, which follows the descriptor's own
                // strides (e.g. a leading dimension > K), so never go below that either.
                const size_t swizzled
                    = b <= 1 ? perBatch
                             : std::max<size_t>(perBatch * b, desc.strides()[2] * (b - 1) + perBatch);
                return std::max<size_t>(swizzled, desc.totalAllocatedElements());
            }

            // ----------------------------------------------------------------
            //  Host-side staging of a swizzled operand (SwizzleTensorA/B): per batch
            //  the [free, K] source is padded to [roundUp(free, MI), roundUp(K, padUnitK)]
            //  and permuted to [nO, kO, kM, nI, kI]; batches sit at the source batch stride.
            // ----------------------------------------------------------------
            struct SwizzleStageGeometry
            {
                size_t                        miM_N          = 16;
                size_t                        unrolledSize   = 0; // K, in elements
                size_t                        tiledSize      = 0; // free dimension (M or N)
                bool                          isSubByte      = false;
                float                         effectiveElem  = 0; // bytes per staged "element"
                size_t                        effUnrolled    = 0;
                size_t                        effMiK         = 0;
                size_t                        effMiKv        = 0;
                size_t                        effPackK       = 0;
                size_t                        padUnitK       = 0;
                size_t                        numBatch       = 1;
                size_t                        srcLd          = 0;
                size_t                        batchStride    = 0;
                size_t                        perBatchNative = 0;
                size_t                        totalNative    = 0; // elements of the staged buffer
                ::Tensor::Manipulation::Shape paddedShape;
            };

            inline SwizzleStageGeometry swizzleStageGeometry(const TensorDescriptor& desc,
                                                             bool                    padTo64Bytes)
            {
                SwizzleStageGeometry g;
                size_t               MiK = 0, MiKv = 0, PackK = 0;
                calculateKforSwizzling(desc.dataType(), MiK, MiKv, PackK);
                g.unrolledSize = desc.sizes()[0];
                g.tiledSize    = desc.sizes()[1];

                // Sub-byte types (e.g. FP4 = 0.5 bytes/elem) need special handling:
                // The swizzle reshape/permute operates on byte-granularity tensors, so
                // we convert element counts to byte counts and treat each byte as one
                // "element" for the reshape dimensions.
                g.isSubByte     = (desc.elementBytes() < 1.0f);
                g.effectiveElem = g.isSubByte ? 1.0f : desc.elementBytes();
                g.effUnrolled   = g.isSubByte
                                      ? multiplyElementSize(g.unrolledSize, desc.elementBytes())
                                      : g.unrolledSize;
                g.effMiK  = g.isSubByte ? size_t(MiK * PackK * desc.elementBytes()) : MiK;
                g.effMiKv = g.isSubByte ? size_t(MiKv * PackK * desc.elementBytes()) : MiKv;
                g.effPackK = g.isSubByte ? size_t(1) : PackK;

                // K is padded to a whole number of swizzle K blocks (see swizzledPaddedKUnit);
                // the effective element is one byte for sub-byte types.
                g.padUnitK
                    = swizzledPaddedKUnit(g.effMiK * g.effPackK, g.effectiveElem, padTo64Bytes);
                g.paddedShape = {roundUpTo(g.tiledSize, g.miM_N),
                                 roundUpTo(g.effUnrolled, g.padUnitK)};
                // Batched: each batch is swizzled on its own and placed at the problem's
                // batch stride, which the kernel also uses to step between batches. The
                // source keeps the descriptor's leading dimension; the swizzled layout is
                // dense (paddedK row length), so the batch stride must hold one padded batch.
                g.numBatch    = desc.dimensions() > 2 ? desc.sizes()[2] : size_t(1);
                g.srcLd       = desc.strides()[1];
                g.batchStride = desc.dimensions() > 2 ? desc.strides()[2] : size_t(0);
                const size_t perBatchEff = g.paddedShape[0] * g.paddedShape[1];
                g.perBatchNative         = g.isSubByte
                                               ? size_t(perBatchEff / desc.elementBytes())
                                               : perBatchEff;
                if(g.numBatch > 1 && g.batchStride < g.perBatchNative)
                {
                    std::stringstream ss;
                    ss << "Swizzled tensor " << desc.getName() << ": batch stride "
                       << g.batchStride << " is smaller than one padded swizzled batch ("
                       << g.perBatchNative << " elements = alignUp(" << g.tiledSize << ", "
                       << g.miM_N << ") x alignUp(" << g.unrolledSize << ", "
                       << swizzledPaddedKUnit(MiK * PackK, desc.elementBytes(), padTo64Bytes) << "))";
                    throw std::runtime_error(ss.str());
                }
                if(g.isSubByte && g.srcLd != g.unrolledSize)
                    throw std::runtime_error(
                        "Swizzled sub-byte tensor requires a packed leading dimension");
                g.totalNative = (g.numBatch - 1) * g.batchStride + g.perBatchNative;
                return g;
            }

            // Stages `src` (the descriptor's tensor, host memory) into the swizzled layout.
            inline ::Tensor::Manipulation::Tensor
                stageSwizzledTensor(const TensorDescriptor&   desc,
                                    const void*               src,
                                    const SwizzleStageGeometry& g)
            {
                using Tensor = ::Tensor::Manipulation::Tensor;
                const size_t totalEff = g.isSubByte
                                            ? multiplyElementSize(g.totalNative, desc.elementBytes())
                                            : g.totalNative;
                Tensor staged({totalEff}, g.effectiveElem);
                memset(staged.as<void>(), 0, staged.getNumBytes());
                const size_t rowBytes = multiplyElementSize(g.unrolledSize, desc.elementBytes());
                for(size_t b = 0; b < g.numBatch; ++b)
                {
                    auto        tmpTensor = Tensor({g.tiledSize, g.effUnrolled}, g.effectiveElem);
                    const char* bsrc      = static_cast<const char*>(src)
                                       + multiplyElementSize(b * g.batchStride, desc.elementBytes());
                    if(g.srcLd == g.unrolledSize)
                        memcpy(tmpTensor.as<void>(), bsrc, tmpTensor.getNumBytes());
                    else
                        for(size_t r = 0; r < g.tiledSize; ++r)
                            memcpy(tmpTensor.as<char>() + r * rowBytes,
                                   bsrc + multiplyElementSize(r * g.srcLd, desc.elementBytes()),
                                   rowBytes);
                    uint64_t padVal{};
                    auto     paddedTensor = ::Tensor::Manipulation::pad(
                        tmpTensor, g.paddedShape, &padVal, g.effectiveElem);
                    paddedTensor.reshape({g.paddedShape[0] / g.miM_N,
                                          g.miM_N,
                                          g.paddedShape[1] / (g.effMiK * g.effPackK),
                                          g.effMiK / g.effMiKv,
                                          g.effMiKv * g.effPackK});
                    Tensor permuted = permute(paddedTensor, {0, 2, 3, 1, 4});
                    memcpy(staged.as<char>()
                               + multiplyElementSize(b * g.batchStride, desc.elementBytes()),
                           permuted.as<void>(),
                           permuted.getNumBytes());
                }
                return staged;
            }

        } // namespace detail
    } // namespace Client
} // namespace TensileLite
