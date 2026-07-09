// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "mxDataGen.hpp"
#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/PreSwizzle.hpp>
#include <mxDataGenerator/dataTypeInfo.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace
{
    constexpr double FP4E2M1Max = 6.0;

    inline std::pair<int, int> randIntRangeFor(hipDataType dataType)
    {
        switch(static_cast<int>(dataType))
        {
        case static_cast<int>(HIP_R_4F_E2M1):
            return {-4, 4};
        case static_cast<int>(HIP_R_6F_E2M3):
            return {-7, 7};
        case static_cast<int>(HIP_R_6F_E3M2):
            return {-28, 28};
        case static_cast<int>(HIP_R_8F_E4M3):
        case static_cast<int>(HIP_R_8F_E5M2):
        default:
            return {1, 10};
        }
    }

    inline double normDistStdDevFor(hipDataType dataType)
    {
        switch(static_cast<int>(dataType))
        {
        case static_cast<int>(HIP_R_4F_E2M1):
            return 5.0;
        default:
            return 1.0;
        }
    }
} // namespace

namespace
{
    using namespace DGen;

    void applyInitMethodString(DataGeneratorOptions&  opt,
                               std::string_view const initMethod,
                               hipDataType            dataType,
                               float                  min_val,
                               float                  max_val)
    {
        opt.min         = initMethod == "uniform_01" ? 0. : (initMethod == "hpl" ? -.5 : min_val);
        opt.max         = initMethod == "uniform_01" ? 1. : (initMethod == "hpl" ? .5 : max_val);
        opt.forceDenorm = false;

        if(initMethod == "Sequential")
            opt.initMode = DataInitMode(Sequential{});
        else if(initMethod == "RowIndex")
            opt.initMode = DataInitMode(RowIndex{});
        else if(initMethod == "ColIndex")
            opt.initMode = DataInitMode(ColIndex{});
        else if(initMethod == "Checkerboard")
            opt.initMode = DataInitMode(Checkerboard{});
        else if(initMethod == "ScaledDiagonal")
            opt.initMode = DataInitMode(ScaledDiagonal{});
        else if(initMethod == "Identity")
            opt.initMode = DataInitMode(Identity{});
        else if(initMethod == "Ones")
            opt.initMode = DataInitMode(Ones{});
        else if(initMethod == "Zeros" || initMethod == "zero")
            opt.initMode = DataInitMode(Zeros{});
        else if(initMethod == "Twos")
            opt.initMode = DataInitMode(Twos{});
        else if(initMethod == "NegOnes")
            opt.initMode = DataInitMode(NegOnes{});
        else if(initMethod == "MaxVals")
            opt.initMode = DataInitMode(MaxVals{});
        else if(initMethod == "DenormMins")
            opt.initMode = DataInitMode(DenormMins{});
        else if(initMethod == "DenormMaxs")
            opt.initMode = DataInitMode(DenormMaxs{});
        else if(initMethod == "NaNs")
            opt.initMode = DataInitMode(NaNs{});
        else if(initMethod == "Infs")
            opt.initMode = DataInitMode(Infs{});
        else if(initMethod == "Bounded" || initMethod == "uniform_01" || initMethod == "hpl")
            opt.initMode = DataInitMode(Bounded{});
        else if(initMethod == "uniform_low_precision")
        {
            opt.min      = -FP4E2M1Max;
            opt.max      = FP4E2M1Max;
            opt.initMode = DataInitMode(Bounded{});
        }
        else if(initMethod == "TrigonometricFromFloat" || initMethod == "trig_float")
            opt.initMode = DataInitMode(TrigonometricFromFloat{});
        else if(initMethod == "norm_dist")
            opt.initMode = DataInitMode(NormalFromFloat{0.0, normDistStdDevFor(dataType)});
        else if(initMethod == "rand_int")
        {
            auto const range = randIntRangeFor(dataType);
            opt.initMode     = DataInitMode(RandInt{range.first, range.second});
        }
        else
            throw std::runtime_error(
                std::string("generateMXInput: unsupported initMethod '")
                + std::string(initMethod)
                + "'. Supported methods: Bounded/uniform_01, hpl, "
                  "uniform_low_precision, "
                  "TrigonometricFromFloat/trig_float, norm_dist, rand_int, "
                  "Sequential, RowIndex, ColIndex, Checkerboard, ScaledDiagonal, "
                  "Identity, Ones, Zeros/zero, Twos, NegOnes, MaxVals, "
                  "DenormMins, DenormMaxs, NaNs, Infs.");
    }

    void applyScaleInitMethodString(DataGeneratorOptions&  opt,
                                    std::string_view const scaleInitMethod,
                                    hipDataType            dataType)
    {
        if(scaleInitMethod.empty())
            return;

        DataGeneratorOptions scaleOpt;
        applyInitMethodString(scaleOpt, scaleInitMethod, dataType, -1.0f, 1.0f);
        if(!canDecoupleScaleInit(opt.initMode, scaleOpt.initMode))
            return;
        opt.scaleInitMode = scaleOpt.initMode;
    }
} // namespace

template <typename DT>
std::vector<uint8_t> unpackData(std::vector<uint8_t> const& packedBytes, size_t elementCount)
{
    static_assert(std::is_same_v<DT, DGen::ocp_e2m1_mxfp4>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e5m3>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e4m3>
                  || std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                  || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>);

    if constexpr(std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                 || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        std::vector<uint8_t> unpackedDataBytes(elementCount);
        for(size_t i = 0; i < elementCount; ++i)
        {
            size_t const bitOffset = i * 6;
            size_t const byteIndex = bitOffset / 8;
            size_t const bitIndex  = bitOffset % 8;

            uint16_t word = 0;
            if(byteIndex < packedBytes.size())
                word |= static_cast<uint16_t>(packedBytes[byteIndex]);
            if(byteIndex + 1 < packedBytes.size())
                word |= static_cast<uint16_t>(packedBytes[byteIndex + 1]) << 8;

            unpackedDataBytes[i] = static_cast<uint8_t>((word >> bitIndex) & 0x3F);
        }
        return unpackedDataBytes;
    }
    else
    {
        std::vector<uint8_t> unpackedDataBytes(elementCount);
        for(size_t i = 0; i < elementCount; ++i)
        {
            size_t const  byteIndex = i / 2;
            uint8_t const b = (byteIndex < packedBytes.size()) ? packedBytes[byteIndex] : 0;
            unpackedDataBytes[i]
                = static_cast<uint8_t>((i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F));
        }
        return unpackedDataBytes;
    }
}

template <typename DT>
void packData(std::vector<uint8_t> const& dataBytes, uint8_t* packedData)
{
    static_assert(std::is_same_v<DT, DGen::ocp_e2m1_mxfp4>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e5m3>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e4m3>
                  || std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                  || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>);

    if constexpr(std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                 || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        size_t const elementCount = dataBytes.size();
        size_t const packedSize   = (elementCount * 6 + 7) / 8;
        std::memset(packedData, 0, packedSize);

        for(size_t i = 0; i < elementCount; ++i)
        {
            uint16_t const v = static_cast<uint16_t>(dataBytes[i] & 0x3F);
            size_t const   bitOffset = i * 6;
            size_t const   byteIndex = bitOffset / 8;
            size_t const   bitIndex  = bitOffset % 8;

            if(byteIndex >= packedSize)
                break;

            uint16_t word = static_cast<uint16_t>(packedData[byteIndex]);
            if(byteIndex + 1 < packedSize)
                word |= static_cast<uint16_t>(packedData[byteIndex + 1]) << 8;

            uint16_t const mask = static_cast<uint16_t>(0x3F) << bitIndex;
            word                = static_cast<uint16_t>((word & ~mask) | (v << bitIndex));

            packedData[byteIndex] = static_cast<uint8_t>(word & 0xFF);
            if(byteIndex + 1 < packedSize)
                packedData[byteIndex + 1] = static_cast<uint8_t>((word >> 8) & 0xFF);
        }
    }
    else
    {
        size_t const elementCount = dataBytes.size();
        size_t const packedSize   = (elementCount + 1) / 2;
        std::memset(packedData, 0, packedSize);

        for(size_t i = 0; i < elementCount; ++i)
        {
            size_t const  byteIndex = i / 2;
            uint8_t const v         = static_cast<uint8_t>(dataBytes[i] & 0x0F);

            if(i % 2 == 0)
                packedData[byteIndex] = static_cast<uint8_t>((packedData[byteIndex] & 0xF0) | v);
            else
                packedData[byteIndex]
                    = static_cast<uint8_t>((packedData[byteIndex] & 0x0F) | (v << 4));
        }
    }
}

template <typename DT>
std::vector<float> getAlignedFloat(std::vector<uint8_t>&              dataBytes,
                                   std::vector<uint8_t> const&        scaleBytes,
                                   std::array<DGen::index_t, 2> const sizes,
                                   int                                elementsPerMXBlock,
                                   bool                               isMatrixA)
{
    std::vector<float>   refFloat(sizes[0] * sizes[1], 0.0);
    std::vector<uint8_t> alignedDataBytes(dataBytes.size());

    if(isMatrixA)
    {
        int M = sizes[0];
        int K = sizes[1];

#pragma omp parallel for
        for(size_t mk = 0; mk < M * K; ++mk)
        {
            auto m        = mk % M;
            auto k        = mk / M;
            auto scale_id = (k / elementsPerMXBlock) * M + m;

            auto data_id         = scale_id * elementsPerMXBlock + k % elementsPerMXBlock;
            alignedDataBytes[mk] = dataBytes[data_id];
            refFloat[mk]
                = DGen::toFloat<DT>(scaleBytes.data(), dataBytes.data(), scale_id, data_id);
        }
        std::swap(dataBytes, alignedDataBytes);
    }
    else
    {
        int N = sizes[0];
        int K = sizes[1];

#pragma omp parallel for
        for(size_t kn = 0; kn < K * N; ++kn)
        {
            auto k        = kn / N;
            auto n        = kn % N;
            auto scale_id = (k / elementsPerMXBlock) * N + n;

            auto data_id         = scale_id * elementsPerMXBlock + k % elementsPerMXBlock;
            alignedDataBytes[kn] = dataBytes[data_id];
            refFloat[kn]
                = DGen::toFloat<DT>(scaleBytes.data(), dataBytes.data(), scale_id, data_id);
        }
        std::swap(dataBytes, alignedDataBytes);
    }
    return refFloat;
}

template <typename T, typename DT>
std::vector<float> generateData(T                           dgen,
                                void*                       data,
                                void*                       scale,
                                std::vector<DGen::index_t>  sizes,
                                std::vector<DGen::index_t>  strides,
                                uint32_t                    seed,
                                DGen::DataGeneratorOptions& opt,
                                int                         elementsPerMXBlock,
                                bool                        isTranspose,
                                bool                        isMatrixA,
                                MXScaleLayout               scaleLayout)
{
    using namespace DGen;

    dgen.setSeed(seed);
    dgen.generate(sizes, strides, opt);

    std::vector<uint8_t> dataBytes = dgen.getDataBytes();
    std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));

    std::vector<uint8_t> scaleBytes = dgen.getScaleBytes();

    size_t const scaleRows
        = (elementsPerMXBlock > 0) ? static_cast<size_t>(sizes[0]) / static_cast<size_t>(elementsPerMXBlock) : 0;
    size_t const scaleCols = static_cast<size_t>(sizes[1]);

    switch(scaleLayout)
    {
    case MXScaleLayout::GFX950:
        scaleBytes = DGen::preSwizzleScalesGFX950(scaleBytes, {scaleCols, scaleRows});
        break;
    case MXScaleLayout::GFX1250:
        if(elementsPerMXBlock > 0)
        {
            scaleBytes
                = DGen::preSwizzleScalesGFX1250(scaleBytes,
                                                /*slowDim=*/scaleCols,
                                                /*fastDim=*/scaleRows,
                                                /*mxBlock=*/static_cast<size_t>(
                                                    elementsPerMXBlock));
        }
        break;
    case MXScaleLayout::None:
        break;
    }

    std::memcpy(scale, scaleBytes.data(), scaleBytes.size() * sizeof(uint8_t));

    if((isMatrixA && isTranspose) || (!isMatrixA && !isTranspose))
    {
        return dgen.getReferenceFloat();
    }

    if constexpr(std::is_same_v<DT, DGen::ocp_e5m2_mxfp8>
                 || std::is_same_v<DT, DGen::ocp_e4m3_mxfp8>)
    {
        auto ret = getAlignedFloat<DT>(
            dataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));
        return ret;
    }
    else if constexpr(std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                      || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        size_t const elementCount = static_cast<size_t>(sizes[0]) * static_cast<size_t>(sizes[1]);
        auto         unpackedDataBytes = unpackData<DT>(dataBytes, elementCount);
        auto ret               = getAlignedFloat<DT>(
            unpackedDataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
        return ret;
    }
    else if constexpr(std::is_same_v<DT, DGen::ocp_e2m1_mxfp4>
                      || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e5m3>
                      || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e4m3>)
    {
        size_t const elementCount = static_cast<size_t>(sizes[0]) * static_cast<size_t>(sizes[1]);
        auto         unpackedDataBytes = unpackData<DT>(dataBytes, elementCount);
        auto ret               = getAlignedFloat<DT>(
            unpackedDataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
        return ret;
    }
    else
    {
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
}

std::vector<float> generateMXInput(hipDataType            dataType,
                                   hipDataType            scaleType,
                                   void*                  data,
                                   void*                  scale,
                                   uint64_t               row,
                                   uint64_t               col,
                                   uint64_t               stride,
                                   bool                   isTranspose,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   bool                   isMatrixA,
                                   MXScaleLayout          scaleLayout,
                                   std::string_view const initMethod,
                                   float                  min_val,
                                   float                  max_val,
                                   std::string_view const scaleInitMethod)
{
    using namespace DGen;

    DataGeneratorOptions opt;
    opt.blockScaling = scaleBlockRowSize * scaleBlockColSize;
    applyInitMethodString(opt, initMethod, dataType, min_val, max_val);
    applyScaleInitMethodString(opt, scaleInitMethod, dataType);

    const uint32_t seed = 1713573849;

    std::vector<index_t> sizes = {row, col};
    std::vector<index_t> strides;

    strides.push_back(1);
    strides.push_back(stride);

    auto const elementsPerMXBlock = scaleBlockRowSize * scaleBlockColSize;

    if(dataType == HIP_R_8F_E5M2)
    {
        DGen::DataGenerator<DGen::ocp_e5m2_mxfp8> dgen;
        return generateData<decltype(dgen), DGen::ocp_e5m2_mxfp8>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  scaleLayout);
    }
    else if(dataType == HIP_R_8F_E4M3)
    {
        DGen::DataGenerator<DGen::ocp_e4m3_mxfp8> dgen;
        return generateData<decltype(dgen), DGen::ocp_e4m3_mxfp8>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  scaleLayout);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_6F_E2M3)
    {
        DGen::DataGenerator<DGen::ocp_e2m3_mxfp6> dgen;
        return generateData<decltype(dgen), DGen::ocp_e2m3_mxfp6>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  scaleLayout);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_6F_E3M2)
    {
        DGen::DataGenerator<DGen::ocp_e3m2_mxfp6> dgen;
        return generateData<decltype(dgen), DGen::ocp_e3m2_mxfp6>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  scaleLayout);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_4F_E2M1)
    {
        if(scaleType == HIP_R_8F_E4M3)
        {
            DGen::DataGenerator<DGen::ocp_e2m1_mxfp4_e4m3> dgen;
            return generateData<decltype(dgen), DGen::ocp_e2m1_mxfp4_e4m3>(dgen,
                                                                          data,
                                                                          scale,
                                                                          sizes,
                                                                          strides,
                                                                          seed,
                                                                          opt,
                                                                          elementsPerMXBlock,
                                                                          isTranspose,
                                                                          isMatrixA,
                                                                          scaleLayout);
        }
        else if(scaleType == static_cast<hipDataType>(HIP_R_8F_E5M3_EXT))
        {
            DGen::DataGenerator<DGen::ocp_e2m1_mxfp4_e5m3> dgen;
            return generateData<decltype(dgen), DGen::ocp_e2m1_mxfp4_e5m3>(dgen,
                                                                          data,
                                                                          scale,
                                                                          sizes,
                                                                          strides,
                                                                          seed,
                                                                          opt,
                                                                          elementsPerMXBlock,
                                                                          isTranspose,
                                                                          isMatrixA,
                                                                          scaleLayout);
        }
        else
        {
            DGen::DataGenerator<DGen::ocp_e2m1_mxfp4> dgen;
            return generateData<decltype(dgen), DGen::ocp_e2m1_mxfp4>(dgen,
                                                                      data,
                                                                      scale,
                                                                      sizes,
                                                                      strides,
                                                                      seed,
                                                                      opt,
                                                                      elementsPerMXBlock,
                                                                      isTranspose,
                                                                      isMatrixA,
                                                                      scaleLayout);
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
}

MXScaleLayout mxScaleLayoutForArchName(std::string_view archName)
{
    if(archName.find("gfx950") != std::string_view::npos)
        return MXScaleLayout::GFX950;
    if(archName.find("gfx1250") != std::string_view::npos)
        return MXScaleLayout::GFX1250;
    return MXScaleLayout::None;
}
