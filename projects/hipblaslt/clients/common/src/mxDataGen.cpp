/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "mxDataGen.hpp"
#include <DataGenerator.hpp>
#include <cblas.h>

template <typename DT>
std::vector<uint8_t> unpackData(std::vector<uint8_t> const& dataBytes)
{
    // Only F4 and F6 need to unpack data.
    static_assert(
        std::is_same_v<
            DT,
            DGen::
                ocp_e2m1_mxfp4> || std::is_same_v<DT, DGen::ocp_e3m2_mxfp6> || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>);

    if constexpr(std::is_same_v<DT,
                                DGen::ocp_e3m2_mxfp6> || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        std::vector<uint8_t> unpackedDataBytes(dataBytes.size() * 8 / 6);
#pragma omp parallel for
        for(int i = 0; i < dataBytes.size(); i++)
        {
            int const f6_id = (i * 6) / 8;
            uint8_t   value = 0;
            switch(i % 4)
            {
            case 0:
                value = (dataBytes[f6_id] & 0x3F);
                break;
            case 1:
                value = ((dataBytes[f6_id] & 0xC0) >> 6) | ((dataBytes[f6_id + 1] & 0xF) << 2);
                break;
            case 2:
                value = ((dataBytes[f6_id] & 0xF0) >> 4) | ((dataBytes[f6_id + 1] & 0x3) << 4);
                break;
            case 3:
                value = ((dataBytes[f6_id] & 0xFC) >> 2);
                break;
            }
            unpackedDataBytes[i] = value;
        }
        return unpackedDataBytes;
    }
    else
    {
        std::vector<uint8_t> unpackedDataBytes(dataBytes.size() * 2);
#pragma omp parallel for
        for(int i = 0; i < dataBytes.size(); i++)
        {
            unpackedDataBytes[i * 2]     = (dataBytes[i] & 0x0F);
            unpackedDataBytes[i * 2 + 1] = (dataBytes[i] >> 4);
        }
        return unpackedDataBytes;
    }
}

template <typename DT>
void packData(std::vector<uint8_t> const& dataBytes, uint8_t* packedData)
{
    // Only F4 and F6 need to unpack data.
    static_assert(
        std::is_same_v<
            DT,
            DGen::
                ocp_e2m1_mxfp4> || std::is_same_v<DT, DGen::ocp_e3m2_mxfp6> || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>);

    if constexpr(std::is_same_v<DT,
                                DGen::ocp_e3m2_mxfp6> || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        auto const total = dataBytes.size() * 6 / 8;
#pragma omp parallel for
        for(int i = 0; i < total; i += 3)
        {
            auto const f8_id = i * 8 / 6;

            packedData[i] = (dataBytes[f8_id] & 0x3F);
            packedData[i] |= ((dataBytes[f8_id + 1] & 0x03) << 6);

            packedData[i + 1] = ((dataBytes[f8_id + 1] & 0xFC) >> 2);
            packedData[i + 1] |= ((dataBytes[f8_id + 2] & 0x03) << 6);

            packedData[i + 2] = ((dataBytes[f8_id + 2] & 0xFC) >> 6);
            packedData[i + 2] |= ((dataBytes[f8_id + 3] & 0x3F) << 2);
        }
    }
    else
    {
#pragma omp parallel for
        for(int i = 0; i < dataBytes.size() / 2; i++)
        {
            packedData[i] = (dataBytes[2 * i] & 0x0F);
            packedData[i] |= (dataBytes[2 * i + 1] << 4);
        }
    }
}

/**
 * @brief Align data with scale and return reference floats
 *
 * mxDataGenerator returns data and scale in which every consecutive
 * 32 data share a scale (i.e., data 0-31 use scale 0, data 32-63 use
 * scale 1, etc.). But when doing matrix multiplication with non-transpose
 * matrix A or transpose matrix B, the data and scale are accessed in a
 * different order (see the example in comment below). This function
 * re-arranges the data to let the data use the correct scale.
 * Note, the passed-in dataBytes will be changed due to the rearrangement.
 *
 * @return float values of generated MX type data aligned with scale
 */
template <typename DT>
std::vector<float> getAlignedFloat(std::vector<uint8_t>&       dataBytes,
                                   std::vector<uint8_t> const& scaleBytes,
                                   std::array<DGen::index_t, 2> const    sizes,
                                   int                         elementsPerMXBlock,
                                   bool                        isMatrixA)
{
    std::vector<float>   refFloat(sizes[0] * sizes[1], 0.0);
    std::vector<uint8_t> alignedDataBytes(dataBytes.size());

    if(isMatrixA) // non-transpose
    {
        int M = sizes[0];
        int K = sizes[1];

        // For example, assume matrix A is 128x128 and elementsPerMXBlock is 32.
        // Before aligned,
        //
        //  mk     m     k       scale ID
        //  0      0     0           0
        //  1      1     0           1     (data at index 1 use scale 1 not 0)
        //  2      2     0           2
        //            ...
        //  127   127    0          127
        //
        //  128    0     1           0
        //  129    1     1           1
        //            ...
        //  255   127    1          127
        //            ...
        //
        // To align data with scale,
        //
        //  mk     m     k       scale ID      data id
        //  0      0     0           0            0
        //  1      1     0           1           32
        //  2      2     0           2           64
        //            ...
        // 127    127    0          127        4064 (127 x 32)
        //
        // We move data at index 32 to index 1 (because the index 1
        // is using scale 1), data at index 64 to index 2, and so on.

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
    else // transpose matrixB
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
                                bool                        isMatrixA)
{
    dgen.setSeed(seed);
    dgen.generate(sizes, strides, opt);

    std::vector<uint8_t> dataBytes = dgen.getDataBytes();
    std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));

    std::vector<uint8_t> scaleBytes = dgen.getScaleBytes();
    std::memcpy(scale, scaleBytes.data(), scaleBytes.size() * sizeof(uint8_t));

    if((isMatrixA && isTranspose) || (!isMatrixA && !isTranspose))
    {
        // For (1) transposed matrixA and (2) non-transposed matrixB,
        // return the reference float directly since they are aligned already.
        return dgen.getReferenceFloat();
    }

    // For types smaller than 8-bit, mxDataGenerator returns packed data (i.e., two FP4 will be
    // stored in a uint8_t), so unpacking the data is required before converting them to float
    if constexpr(std::is_same_v<DT,
                                DGen::ocp_e5m2_mxfp8> || std::is_same_v<DT, DGen::ocp_e4m3_mxfp8>)
    {
        auto ret = getAlignedFloat<DT>(
            dataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));
        return ret;
    }
    else if constexpr(std::is_same_v<
                          DT,
                          DGen::ocp_e3m2_mxfp6> || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        auto unpackedDataBytes = unpackData<DT>(dataBytes);
        auto ret               = getAlignedFloat<DT>(
            unpackedDataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        // GPU expects the data are packed
        packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
        return ret;
    }
    else if constexpr(std::is_same_v<DT, DGen::ocp_e2m1_mxfp4>)
    {
        auto unpackedDataBytes = unpackData<DT>(dataBytes);
        auto ret               = getAlignedFloat<DT>(
            unpackedDataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        // GPU expects the data are packed
        packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
        return ret;
    }
    else
    {
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
}

#ifdef HIPBLASLT_USE_ROCROLLER
/**
 * @brief Generate random data for OCP (MX) F8/F6/F4 types
 *
 * The generated data consist of data part and scale part,
 * and the corresponding float values (combine data and scale)
 * will be returned.
 *
 * @return float values of generated MX type data
 */
std::vector<float> generateMXInput(hipDataType            dataType,
                                   void*                  data,
                                   void*                  scale,
                                   DGen::index_t          rowSize,
                                   DGen::index_t          colSize,
                                   DGen::index_t          stride,
                                   bool                   isTranspose,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   bool                   isMatrixA,
                                   std::string_view const initMethod,
                                   float                  min_val,
                                   float                  max_val)
{
    using namespace DGen;

    DataGeneratorOptions opt;
    opt.min          = min_val;
    opt.max          = max_val;
    opt.blockScaling = scaleBlockRowSize * scaleBlockColSize;
    opt.pattern      = initMethod == "Bounded" ? Bounded : Trigonometric;

    const uint32_t seed = 1713573849;

    std::vector<index_t> sizes = {rowSize, colSize};
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
                                                                  isMatrixA);
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
                                                                  isMatrixA);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_6F_E2M3_EXT)
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
                                                                  isMatrixA);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_6F_E3M2_EXT)
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
                                                                  isMatrixA);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_4F_E2M1_EXT)
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
                                                                  isMatrixA);
    }
    else
    {
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
}
#endif
