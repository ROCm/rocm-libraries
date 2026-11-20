// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "DataInitialization.hpp"
#include "TensorDataManipulation.hpp"
#include <mxDataGen.hpp>                          // hipDataType, generateMXInput
#include <mxDataGenerator/PreSwizzle.hpp>         // DGen::preSwizzleScalesGFX950 / GFX1250
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
#include <sstream>
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
            //  downsampleMXScaleFreeDim
            //
            // A 2D MX scaling tile shares one scale across mxBlockFree free-dim
            // elements, but DGen::ScaleBlockLayout only blocks one axis and always
            // emits one scale per free element. Collapse those groups here.
            //
            // The representative is the group maximum: the data bytes were
            // quantized against the scales being dropped, so a max can only shrink
            // a product, never overflow it to inf. Byte-wise max is exact for E8
            // (UE8M0, byte == unsigned exponent); other encodings take the first
            // element. Reference and kernel both read what this writes, so they
            // agree either way -- only value conditioning is lost, not consistency.
            // ----------------------------------------------------------------
            inline void downsampleMXScaleFreeDim(uint8_t*       dst,
                                                 uint8_t const* src,
                                                 size_t         compactFreeDim,
                                                 size_t         kBlocks,
                                                 size_t         mxBlockFree,
                                                 size_t         elemBytes,
                                                 bool           kFast,
                                                 bool           unsignedExponentScale)
            {
                if(mxBlockFree <= 1 || compactFreeDim == 0 || kBlocks == 0)
                    return;

                size_t const tiledFreeDim = (compactFreeDim + mxBlockFree - 1) / mxBlockFree;

                // kFast (bound dim at index 0 -> TN A / NT B) lays the buffer out as
                // [free][kBlock], so a free step skips a whole row of K-blocks.
                // Otherwise K-blocks are the slow axis.
                size_t const srcFreeStride = kFast ? kBlocks : 1;
                size_t const srcKStride    = kFast ? 1 : compactFreeDim;
                size_t const dstFreeStride = kFast ? kBlocks : 1;
                size_t const dstKStride    = kFast ? 1 : tiledFreeDim;

                bool const useMax = unsignedExponentScale && elemBytes == 1;

                for(size_t g = 0; g < tiledFreeDim; g++)
                {
                    size_t const fBegin = g * mxBlockFree;
                    size_t const fEnd   = std::min(fBegin + mxBlockFree, compactFreeDim);
                    for(size_t kb = 0; kb < kBlocks; kb++)
                    {
                        uint8_t const* best
                            = src + (fBegin * srcFreeStride + kb * srcKStride) * elemBytes;
                        if(useMax)
                        {
                            for(size_t f = fBegin + 1; f < fEnd; f++)
                            {
                                uint8_t const* cand
                                    = src + (f * srcFreeStride + kb * srcKStride) * elemBytes;
                                if(*cand > *best)
                                    best = cand;
                            }
                        }
                        std::memcpy(dst + (g * dstFreeStride + kb * dstKStride) * elemBytes,
                                    best,
                                    elemBytes);
                    }
                }
            }
            // ----------------------------------------------------------------
            //  varyMXScaleFreeTiles
            //
            // Random-like data with a mirrored scale init quantizes every block
            // against the same exponent, so after downsampleMXScaleFreeDim every
            // 2D tile holds 0x7F and a kernel that reads the wrong scale row still
            // validates. Offset each E8 by a small exponent (-2..2) that differs
            // between neighbouring free tiles and K-blocks. The data bytes are
            // untouched; kernel and reference read the same scales, so products
            // only move by a power of two. 0x00 / 0xFF (zero / NaN) are kept.
            // Layout matches downsampleMXScaleFreeDim's dst (compact K-blocks).
            // ----------------------------------------------------------------
            inline void varyMXScaleFreeTiles(uint8_t* scale,
                                             size_t   tiledFreeDim,
                                             size_t   kBlocks,
                                             bool     kFast)
            {
                size_t const freeStride = kFast ? kBlocks : 1;
                size_t const kStride    = kFast ? 1 : tiledFreeDim;
                for(size_t g = 0; g < tiledFreeDim; g++)
                {
                    for(size_t kb = 0; kb < kBlocks; kb++)
                    {
                        uint8_t& b = scale[g * freeStride + kb * kStride];
                        if(b == 0x00 || b == 0xFF)
                            continue;
                        // Steps by 1 (mod 5) per free tile, so adjacent rows differ.
                        int const delta = static_cast<int>((g + 2 * kb) % 5) - 2;
                        b = static_cast<uint8_t>(std::clamp(static_cast<int>(b) + delta, 1, 254));
                    }
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
            // ----------------------------------------------------------------
            //  naturalMXScaleFromCanonical : inverse of restrideMXScaleBufferKFast.
            //                Recovers the scales generateMXInput emits for one
            //                batch (unpadded, pre-swizzle) from the canonical copy.
            //  swizzleNaturalMXScale : the swizzle generateMXInput applies to
            //                those scales for `layout`; rows/cols are the data
            //                tensor's sizes()[0] / sizes()[1].
            // ----------------------------------------------------------------
            inline std::vector<uint8_t> naturalMXScaleFromCanonical(uint8_t const* canonical,
                                                                    size_t         naturalBytes,
                                                                    bool           kFast,
                                                                    size_t         compactFree,
                                                                    size_t         compactKBlocks,
                                                                    size_t         paddedKBlocks,
                                                                    size_t         elemBytes)
            {
                std::vector<uint8_t> natural(naturalBytes);
                if(kFast && compactKBlocks != paddedKBlocks)
                {
                    size_t const compactRow = compactKBlocks * elemBytes;
                    size_t const paddedRow  = paddedKBlocks * elemBytes;
                    for(size_t f = 0; f < compactFree; f++)
                        std::memcpy(
                            natural.data() + f * compactRow, canonical + f * paddedRow, compactRow);
                }
                else
                {
                    std::memcpy(natural.data(), canonical, naturalBytes);
                }
                return natural;
            }
            inline std::vector<uint8_t> swizzleNaturalMXScale(std::vector<uint8_t> const& natural,
                                                              MXScaleLayout               layout,
                                                              size_t                      rows,
                                                              size_t                      cols,
                                                              size_t                      mxBlock,
                                                              bool                        kIsRows)
            {
                if(layout == MXScaleLayout::GFX950)
                {
                    size_t const kExtent = kIsRows ? rows : cols;
                    size_t const kBlocks = mxBlock > 0 ? (kExtent + mxBlock - 1) / mxBlock : 0;
                    if(!kIsRows && mxBlock > 0 && kExtent % mxBlock != 0)
                        throw std::runtime_error(
                            "GFX950 scales with K along the columns require K to be a multiple of "
                            + std::to_string(mxBlock) + ", got " + std::to_string(kExtent));
                    size_t const mnExtent = (kBlocks > 0 && natural.size() % kBlocks == 0)
                                                ? natural.size() / kBlocks
                                                : (kIsRows ? cols : rows);
                    if(mnExtent * kBlocks != natural.size())
                        throw std::runtime_error("GFX950 scales do not form an MN by K block grid: "
                                                 + std::to_string(mnExtent) + " x "
                                                 + std::to_string(kBlocks) + " covers "
                                                 + std::to_string(mnExtent * kBlocks) + " of "
                                                 + std::to_string(natural.size()) + " scales");
                    if(kIsRows)
                        return DGen::preSwizzleScalesGFX950(natural, {mnExtent, kBlocks});
                    // The swizzle reads [MN][K block]; a free-dim contiguous operand's
                    // natural scales are [K block][MN].
                    std::vector<uint8_t> mnMajor(natural.size());
                    for(size_t kb = 0; kb < kBlocks; ++kb)
                        for(size_t mn = 0; mn < mnExtent; ++mn)
                            mnMajor[mn * kBlocks + kb] = natural[kb * mnExtent + mn];
                    return DGen::preSwizzleScalesGFX950(mnMajor, {mnExtent, kBlocks});
                }
                if(layout == MXScaleLayout::GFX1250 && mxBlock > 0)
                    return DGen::preSwizzleScalesGFX1250(
                        natural, cols, (rows + mxBlock - 1) / mxBlock, mxBlock);
                return natural;
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

            // ---- swizzled operand layout ----
            inline void calculateKforSwizzling(rocisa::DataType datatype,
                                               size_t&          MiK,
                                               size_t&          MiKv,
                                               size_t&          PackK,
                                               bool             isGfx1250 = false)
            {
                // gfx1250 WMMA_V3 swizzle (TDM path) uses a 32-byte K block: one 16-byte ds_load_b128
                // per lane-half x 2 lane-halves. In elements that is BK = 32/bpe (F8/I8: 32, F16/BF16: 16),
                // i.e. MiK = BK/PackK with PackK = 2 below. gfx942 MFMA keeps the wider blocks.
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
                    MiK  = isGfx1250 ? 8 : 16;
                    MiKv = 4;
                    break;
                case rocisa::DataType::Int8:
                    MiK  = isGfx1250 ? 16 : 32;
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
                    // gfx1250 F8 WMMA_V3 (16x16x128) swizzle uses BK=32 (MiK=16); gfx942 MFMA uses BK=64
                    // (MiK=32). PackK = 16/MiKv/1 = 2 for both, so gfx1250 gives BK=MiK*PackK=32.
                    MiK  = isGfx1250 ? 16 : 32;
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

            // K elements a swizzled tensor's rows are padded to. gfx942 pads to its swizzle K block;
            // gfx1250 swizzles in smaller K blocks (32 B) but pads to 64 B as well. Flattening the K block
            // indices gives the same bytes for either block size, so with the same padding the layout is
            // identical on both architectures and one hipblasLtOrder_t (COL16_4R*, K in 64 B) covers both.
            inline size_t swizzledPaddedKUnit(size_t swizzleK, float elementBytes, bool isGfx1250)
            {
                if(!isGfx1250)
                    return swizzleK;
                const size_t padK = static_cast<size_t>(64.0f / elementBytes);
                return std::max(swizzleK, padK);
            }

            inline size_t getSwizzledTensorNumAllocatedElements(const TensorDescriptor& desc,
                                                                size_t                  miM_N,
                                                                size_t                  miK,
                                                                size_t                  packK,
                                                                bool                    isGfx1250)
            {
                // TODO: currently [0][1] = k, (m or n) is based on TN, need to make this generic in the future
                const auto k         = desc.sizes()[0];
                const auto m_n       = desc.sizes()[1];
                const auto b         = desc.sizes()[2];
                const auto swizzleK
                    = swizzledPaddedKUnit(miK * packK, desc.elementBytes(), isGfx1250);
                const auto paddedM_N = DGen::roundUp(m_n, miM_N);
                const auto paddedK   = DGen::roundUp(k, swizzleK);
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
                                                             bool                    isGfx1250)
            {
                SwizzleStageGeometry g;
                size_t               MiK = 0, MiKv = 0, PackK = 0;
                calculateKforSwizzling(desc.dataType(), MiK, MiKv, PackK, isGfx1250);
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
                    = swizzledPaddedKUnit(g.effMiK * g.effPackK, g.effectiveElem, isGfx1250);
                g.paddedShape = {DGen::roundUp(g.tiledSize, g.miM_N),
                                 DGen::roundUp(g.effUnrolled, g.padUnitK)};
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
                       << swizzledPaddedKUnit(MiK * PackK, desc.elementBytes(), isGfx1250) << "))";
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
#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
