/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <Tensile/hip/HipHardware.hpp>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mx_tile128x4.hpp"

namespace
{
    struct MatrixTransformIO
    {
        MatrixTransformIO()                 = default;
        virtual ~MatrixTransformIO()        = default;
        virtual void*  getBuf(size_t i)     = 0;
        virtual size_t elemNumBytes() const = 0;

    private:
        MatrixTransformIO(const MatrixTransformIO&)            = delete;
        MatrixTransformIO(MatrixTransformIO&&)                 = delete;
        MatrixTransformIO& operator=(const MatrixTransformIO&) = delete;
        MatrixTransformIO& operator=(MatrixTransformIO&&)      = delete;
    };

    template <typename DType>
    struct TypedMatrixTransformIO : public MatrixTransformIO
    {
        TypedMatrixTransformIO(int64_t m, int64_t n, int64_t b)
        {
            constexpr std::size_t alignment = 2 * 1024 * 1024;
            const auto            bufSize   = m * n * b * sizeof(DType);
            const auto            res       = bufSize % alignment;
            const auto            allocSize = bufSize + (res ? (alignment - res) : 0);
            // ASSERT_* cannot be used in constructors (generates illegal
            // return-void). Use hard abort on allocation failure instead.
            auto err = hipMalloc(&this->aBase, allocSize);
            if(err != hipSuccess)
            {
                fprintf(stderr, "hipMalloc failed: %s at %s:%d\n",
                        hipGetErrorString(err), __FILE__, __LINE__);
                abort();
            }
            err = hipMalloc(&this->bBase, allocSize);
            if(err != hipSuccess)
            {
                fprintf(stderr, "hipMalloc failed: %s at %s:%d\n",
                        hipGetErrorString(err), __FILE__, __LINE__);
                abort();
            }
            err = hipMalloc(&this->cBase, allocSize);
            if(err != hipSuccess)
            {
                fprintf(stderr, "hipMalloc failed: %s at %s:%d\n",
                        hipGetErrorString(err), __FILE__, __LINE__);
                abort();
            }
            this->a = reinterpret_cast<DType*>(aBase + (allocSize - bufSize));
            this->b = reinterpret_cast<DType*>(bBase + (allocSize - bufSize));
            this->c = reinterpret_cast<DType*>(cBase + (allocSize - bufSize));
            init(this->a, m * n * b);
            init(this->b, m * n * b);
        }

        ~TypedMatrixTransformIO() override
        {
            auto err = hipFree(aBase);
            err      = hipFree(bBase);
            err      = hipFree(cBase);
            aBase    = nullptr;
            bBase    = nullptr;
            cBase    = nullptr;
            if(err != hipSuccess)
            {
                fprintf(stderr, "hipFree failed: %s at %s:%d\n",
                        hipGetErrorString(err), __FILE__, __LINE__);
                abort();
            }
        }

        void* getBuf(size_t i) override
        {
            void* buf[] = {a, b, c};
            return buf[i];
        }

        size_t elemNumBytes() const override
        {
            return sizeof(DType);
        }

    private:
        void init(DType* buf, size_t len)
        {
            srand(time(nullptr));
            std::vector<DType> ref(len);

            for(auto& i : ref)
            {
                i = DType(rand() % 7 - 3);
            }

            auto err = hipMemcpy(buf, ref.data(), len * sizeof(DType), hipMemcpyHostToDevice);
            ASSERT_EQ(err, hipSuccess);
        }

    private:
        DType* a{};
        DType* b{};
        DType* c{};
        char*  aBase{};
        char*  bBase{};
        char*  cBase{};
    };

    using MatrixTransformIOPtr = std::unique_ptr<MatrixTransformIO>;
    MatrixTransformIOPtr
        makeMatrixTransformIOPtr(hipDataType datatype, int64_t m, int64_t n, int64_t b)
    {
        if(datatype == HIP_R_32F)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtFloat>>(m, n, b);
        }
        else if(datatype == HIP_R_16F)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtHalf>>(m, n, b);
        }
        else if(datatype == HIP_R_16BF)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtBfloat16>>(m, n, b);
        }
        else if(datatype == HIP_R_8I)
        {
            return std::make_unique<TypedMatrixTransformIO<int8_t>>(m, n, b);
        }
        else if(datatype == HIP_R_32I)
        {
            return std::make_unique<TypedMatrixTransformIO<int32_t>>(m, n, b);
        }
        return nullptr;
    }

    template <bool RowMaj>
    int64_t getLeadingDimSize(int64_t numRows, int64_t numCols)
    {
        return RowMaj ? numCols : numRows;
    }

    template <bool RowMaj>
    uint32_t getOffset(uint32_t row, uint32_t col, uint32_t ld)
    {
        if constexpr(RowMaj)
        {
            return ld * row + col;
        }
        else
        {
            return ld * col + row;
        }
    }

    template <typename DType, typename ScaleType, bool RowMajA, bool RowMajB, bool RowMajC>
    void cpuTransform(DType*       c,
                      const DType* a,
                      const DType* b,
                      ScaleType    alpha,
                      ScaleType    beta,
                      bool         transA,
                      bool         transB,
                      uint32_t     m,
                      uint32_t     n,
                      uint32_t     ldA,
                      uint32_t     ldB,
                      uint32_t     ldC,
                      uint32_t     batchSize,
                      uint32_t     batchStride)
    {
        for(uint32_t k = 0; k < batchSize; ++k)
        {
            const int64_t batchOffset = k * int64_t(batchStride);

            for(uint32_t i = 0; i < m; ++i)
            {
                for(uint32_t j = 0; j < n; ++j)
                {
                    const auto offsetA
                        = transA ? getOffset<RowMajA>(j, i, ldA) : getOffset<RowMajA>(i, j, ldA);
                    const auto offsetB
                        = transB ? getOffset<RowMajB>(j, i, ldB) : getOffset<RowMajB>(i, j, ldB);
                    const auto offsetC = getOffset<RowMajC>(i, j, ldC);
                    c[batchOffset + offsetC]
                        = a[batchOffset + offsetA] * alpha + b[batchOffset + offsetB] * beta;
                }
            }
        }
    }

    template <typename DType>
    void validation(void*    c,
                    void*    a,
                    void*    b,
                    float    alpha,
                    float    beta,
                    uint32_t m,
                    uint32_t n,
                    uint32_t ldA,
                    uint32_t ldB,
                    uint32_t ldC,
                    uint32_t batchSize,
                    uint32_t batchStride,
                    bool     rowMajA,
                    bool     rowMajB,
                    bool     rowMajC,
                    bool     transA,
                    bool     transB)
    {
        using std::begin;
        using std::end;
        std::vector<float> hC(m * n * batchSize, 0);
        std::vector<float> hA(m * n * batchSize, 0);
        std::vector<float> hB(m * n * batchSize, 0);
        std::vector<float> cpuRef(m * n * batchSize, 0);
        std::vector<DType> dA(m * n * batchSize);
        std::vector<DType> dB(m * n * batchSize);
        std::vector<DType> dC(m * n * batchSize);
        auto               err = hipSuccess;

        if(a)
        {
            err = hipMemcpyDtoH(dA.data(), a, m * n * batchSize * sizeof(DType));
        }

        if(b)
        {
            err = hipMemcpyDtoH(dB.data(), b, m * n * batchSize * sizeof(DType));
        }

        err = hipMemcpyDtoH(dC.data(), c, m * n * batchSize * sizeof(DType));

        ASSERT_EQ(err, hipSuccess);

        std::transform(begin(dC), end(dC), begin(hC), [](auto i) { return float(i); });

        if(a)
        {
            std::transform(begin(dA), end(dA), begin(hA), [](auto i) { return float(i); });
        }

        if(b)
        {
            std::transform(begin(dB), end(dB), begin(hB), [](auto i) { return float(i); });
        }

        if(rowMajA && rowMajB && rowMajC)
        {
            cpuTransform<float, float, true, true, true>(cpuRef.data(),
                                                         hA.data(),
                                                         hB.data(),
                                                         alpha,
                                                         beta,
                                                         transA,
                                                         transB,
                                                         m,
                                                         n,
                                                         ldA,
                                                         ldB,
                                                         ldC,
                                                         batchSize,
                                                         batchStride);
        }
        else if(!rowMajA && rowMajB && rowMajC)
        {
            cpuTransform<float, float, false, true, true>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(rowMajA && !rowMajB && rowMajC)
        {
            cpuTransform<float, float, true, false, true>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(rowMajA && rowMajB && !rowMajC)
        {
            cpuTransform<float, float, true, true, false>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(!rowMajA && !rowMajB && rowMajC)
        {
            cpuTransform<float, float, false, false, true>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(!rowMajA && rowMajB && !rowMajC)
        {
            cpuTransform<float, float, false, true, false>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(rowMajA && !rowMajB && !rowMajC)
        {
            cpuTransform<float, float, true, false, false>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(!rowMajA && !rowMajB && !rowMajC)
        {
            cpuTransform<float, float, false, false, false>(cpuRef.data(),
                                                            hA.data(),
                                                            hB.data(),
                                                            alpha,
                                                            beta,
                                                            transA,
                                                            transB,
                                                            m,
                                                            n,
                                                            ldA,
                                                            ldB,
                                                            ldC,
                                                            batchSize,
                                                            batchStride);
        }

        ASSERT_THAT(hC, ::testing::Pointwise(::testing::FloatNear(1e-5), cpuRef));
    }
}

class MatrixTransformTest : public ::testing::TestWithParam<std::tuple<int64_t,
                                                                       int64_t,
                                                                       hipDataType,
                                                                       hipDataType,
                                                                       hipblasOperation_t,
                                                                       hipblasOperation_t,
                                                                       hipblasLtOrder_t,
                                                                       hipblasLtOrder_t,
                                                                       hipblasLtOrder_t>>
{
};

TEST_P(MatrixTransformTest, Basic)
{
    int64_t                     m             = std::get<0>(GetParam());
    int64_t                     n             = std::get<1>(GetParam());
    int32_t                     batchSize     = 1;
    auto                        datatype      = std::get<2>(GetParam());
    auto                        scaleDatatype = std::get<3>(GetParam());
    auto                        opA           = std::get<4>(GetParam());
    auto                        opB           = std::get<5>(GetParam());
    auto                        orderA        = std::get<6>(GetParam());
    auto                        orderB        = std::get<7>(GetParam());
    auto                        orderC        = std::get<8>(GetParam());
    float                       alpha         = 1;
    float                       beta          = 1;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);

    if(datatype == HIP_R_32F)
    {
        validation<float>(dC,
                          dA,
                          dB,
                          alpha,
                          beta,
                          m,
                          n,
                          ldA,
                          ldB,
                          ldC,
                          batchSize,
                          batchStride,
                          rowMajA,
                          rowMajB,
                          rowMajC,
                          transA,
                          transB);
    }
    else if(datatype == HIP_R_16F)
    {
        validation<hipblasLtHalf>(dC,
                                  dA,
                                  dB,
                                  alpha,
                                  beta,
                                  m,
                                  n,
                                  ldA,
                                  ldB,
                                  ldC,
                                  batchSize,
                                  batchStride,
                                  rowMajA,
                                  rowMajB,
                                  rowMajC,
                                  transA,
                                  transB);
    }
    else if(datatype == HIP_R_16BF)
    {
        validation<hipblasLtBfloat16>(dC,
                                      dA,
                                      dB,
                                      alpha,
                                      beta,
                                      m,
                                      n,
                                      ldA,
                                      ldB,
                                      ldC,
                                      batchSize,
                                      batchStride,
                                      rowMajA,
                                      rowMajB,
                                      rowMajC,
                                      transA,
                                      transB);
    }
    else if(datatype == HIP_R_8I)
    {
        validation<int8_t>(dC,
                           dA,
                           dB,
                           alpha,
                           beta,
                           m,
                           n,
                           ldA,
                           ldB,
                           ldC,
                           batchSize,
                           batchStride,
                           rowMajA,
                           rowMajB,
                           rowMajC,
                           transA,
                           transB);
    }
    else if(datatype == HIP_R_32I)
    {
        validation<int32_t>(dC,
                            dA,
                            dB,
                            alpha,
                            beta,
                            m,
                            n,
                            ldA,
                            ldB,
                            ldC,
                            batchSize,
                            batchStride,
                            rowMajA,
                            rowMajB,
                            rowMajC,
                            transA,
                            transB);
    }

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
}

TEST(MatrixTransformTest, InvalidConfigurations)
{
    int64_t                     m             = 1024;
    int64_t                     n             = 1024;
    int32_t                     batchSize     = 1;
    auto                        datatype      = HIP_R_32F;
    auto                        scaleDatatype = HIP_R_32F;
    auto                        opA           = HIPBLAS_OP_N;
    auto                        opB           = HIPBLAS_OP_N;
    auto                        orderA        = HIPBLASLT_ORDER_ROW;
    auto                        orderB        = HIPBLASLT_ORDER_ROW;
    auto                        orderC        = HIPBLASLT_ORDER_COL;
    float                       alpha         = 1;
    float                       beta          = 1;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_NOT_INITIALIZED);

    hipblasLtErr = hipblasLtCreate(&handle);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, nullptr, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, nullptr, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, nullptr, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, nullptr, dC, nullptr, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, nullptr, &beta, dB, nullptr, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, nullptr, &beta, dB, layoutB, dC, nullptr, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_INVALID_VALUE);

    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
}

TEST(MatrixTransformTest, NullA)
{
    int64_t                     m             = 1024;
    int64_t                     n             = 1024;
    int32_t                     batchSize     = 1;
    auto                        datatype      = HIP_R_32F;
    auto                        scaleDatatype = HIP_R_32F;
    auto                        opA           = HIPBLAS_OP_N;
    auto                        opB           = HIPBLAS_OP_N;
    auto                        orderA        = HIPBLASLT_ORDER_ROW;
    auto                        orderB        = HIPBLASLT_ORDER_ROW;
    auto                        orderC        = HIPBLASLT_ORDER_COL;
    float                       alpha         = 0;
    float                       beta          = 1;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, nullptr, nullptr, nullptr, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);
    validation<float>(dC,
                      nullptr,
                      dB,
                      alpha,
                      beta,
                      m,
                      n,
                      0,
                      ldB,
                      ldC,
                      batchSize,
                      batchStride,
                      rowMajA,
                      rowMajB,
                      rowMajC,
                      transA,
                      transB);
}

TEST(MatrixTransformTest, NullB)
{
    int64_t                     m             = 1024;
    int64_t                     n             = 1024;
    int32_t                     batchSize     = 1;
    auto                        datatype      = HIP_R_32F;
    auto                        scaleDatatype = HIP_R_32F;
    auto                        opA           = HIPBLAS_OP_N;
    auto                        opB           = HIPBLAS_OP_N;
    auto                        orderA        = HIPBLASLT_ORDER_ROW;
    auto                        orderB        = HIPBLASLT_ORDER_ROW;
    auto                        orderC        = HIPBLASLT_ORDER_COL;
    float                       alpha         = 1;
    float                       beta          = 0;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, nullptr, nullptr, nullptr, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);
    validation<float>(dC,
                      dA,
                      nullptr,
                      alpha,
                      beta,
                      m,
                      n,
                      ldA,
                      0,
                      ldC,
                      batchSize,
                      batchStride,
                      rowMajA,
                      rowMajB,
                      rowMajC,
                      transA,
                      transB);
}

TEST(MatrixTransformTest, ScalarsOnDevice)
{
    int64_t m             = 1024;
    int64_t n             = 1024;
    int32_t batchSize     = 1;
    auto    datatype      = HIP_R_32F;
    auto    scaleDatatype = HIP_R_32F;
    auto    opA           = HIPBLAS_OP_N;
    auto    opB           = HIPBLAS_OP_N;
    auto    orderA        = HIPBLASLT_ORDER_ROW;
    auto    orderB        = HIPBLASLT_ORDER_ROW;
    auto    orderC        = HIPBLASLT_ORDER_COL;
    float   alpha         = 1;
    float   beta          = 1;
    float*  deviceAlpha{};
    float*  deviceBeta{};
    auto    hipErr                          = hipMalloc(&deviceAlpha, sizeof(deviceAlpha));
    hipErr                                  = hipMalloc(&deviceBeta, sizeof(deviceBeta));
    hipErr                                  = hipMemcpyHtoD(deviceAlpha, &alpha, sizeof(alpha));
    hipErr                                  = hipMemcpyHtoD(deviceBeta, &beta, sizeof(beta));
    int64_t                     batchStride = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_N ? m : n;
    shapeA.second = opA == HIPBLAS_OP_N ? n : m;
    shapeB.first  = opB == HIPBLAS_OP_N ? m : n;
    shapeB.second = opB == HIPBLAS_OP_N ? n : m;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_DEVICE;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, deviceAlpha, dA, layoutA, deviceBeta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    hipErr       = hipFree(deviceAlpha);
    hipErr       = hipFree(deviceBeta);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA != HIPBLAS_OP_N);
    auto transB  = (opB != HIPBLAS_OP_N);
    validation<float>(dC,
                      dA,
                      dB,
                      alpha,
                      beta,
                      m,
                      n,
                      ldA,
                      ldB,
                      ldC,
                      batchSize,
                      batchStride,
                      rowMajA,
                      rowMajB,
                      rowMajC,
                      transA,
                      transB);
}

TEST(MatrixTransformTest, MultipleDevices)
{
    int numDevices{};
    int curDevice{};
    auto hipErr = hipGetDeviceCount(&numDevices);
    EXPECT_EQ(hipErr, hipSuccess);
    hipErr = hipGetDevice(&curDevice);
    EXPECT_EQ(hipErr, hipSuccess);
    // acquire at most 2 devices
    numDevices = std::min<int>(numDevices, 2);

    for (int deviceId = 0; deviceId < numDevices; ++deviceId)
    {
        hipErr = hipSetDevice(deviceId);
        EXPECT_EQ(hipErr, hipSuccess);
        int64_t                     m             = 1024;
        int64_t                     n             = 1024;
        int32_t                     batchSize     = 1;
        auto                        datatype      = HIP_R_32F;
        auto                        scaleDatatype = HIP_R_32F;
        auto                        opA           = HIPBLAS_OP_N;
        auto                        opB           = HIPBLAS_OP_N;
        auto                        orderA        = HIPBLASLT_ORDER_ROW;
        auto                        orderB        = HIPBLASLT_ORDER_ROW;
        auto                        orderC        = HIPBLASLT_ORDER_COL;
        float                       alpha         = 1;
        float                       beta          = 1;
        int64_t                     batchStride   = m * n;
        std::pair<int64_t, int64_t> shapeA;
        std::pair<int64_t, int64_t> shapeB;
        shapeA.first  = opA == HIPBLAS_OP_T ? n : m;
        shapeA.second = opA == HIPBLAS_OP_T ? m : n;
        shapeB.first  = opB == HIPBLAS_OP_T ? n : m;
        shapeB.second = opB == HIPBLAS_OP_T ? m : n;
        uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                            ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                            : getLeadingDimSize<false>(shapeA.first, shapeA.second);
        uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                            ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                            : getLeadingDimSize<false>(shapeB.first, shapeB.second);
        uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                        : getLeadingDimSize<false>(m, n);

        auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
        void* dA     = inputs->getBuf(0);
        void* dB     = inputs->getBuf(1);
        void* dC     = inputs->getBuf(2);

        hipblasLtMatrixTransformDesc_t desc;
        auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
        hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
        hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
            desc,
            hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
            &pMode,
            sizeof(pMode));

        ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

        hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
            desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
        hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
            desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
        hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
        hipblasLtErr
            = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
        hipblasLtErr
            = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
        hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutA,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
            &orderA,
            sizeof(orderA));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutB,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
            &orderB,
            sizeof(orderB));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutC,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
            &orderC,
            sizeof(orderC));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutA,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batchSize,
            sizeof(batchSize));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutB,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batchSize,
            sizeof(batchSize));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutC,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batchSize,
            sizeof(batchSize));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutA,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batchStride,
            sizeof(batchStride));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutB,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batchStride,
            sizeof(batchStride));
        hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
            layoutC,
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batchStride,
            sizeof(batchStride));
        hipblasLtHandle_t handle{};
        hipblasLtErr = hipblasLtCreate(&handle);
        hipblasLtErr = hipblasLtMatrixTransform(
            handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
        ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
        auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
        auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
        auto transA  = (opA == HIPBLAS_OP_T);
        auto transB  = (opB == HIPBLAS_OP_T);

        if(datatype == HIP_R_32F)
        {
            validation<float>(dC,
                            dA,
                            dB,
                            alpha,
                            beta,
                            m,
                            n,
                            ldA,
                            ldB,
                            ldC,
                            batchSize,
                            batchStride,
                            rowMajA,
                            rowMajB,
                            rowMajC,
                            transA,
                            transB);
        }
        else if(datatype == HIP_R_16F)
        {
            validation<hipblasLtHalf>(dC,
                                    dA,
                                    dB,
                                    alpha,
                                    beta,
                                    m,
                                    n,
                                    ldA,
                                    ldB,
                                    ldC,
                                    batchSize,
                                    batchStride,
                                    rowMajA,
                                    rowMajB,
                                    rowMajC,
                                    transA,
                                    transB);
        }
        else if(datatype == HIP_R_16BF)
        {
            validation<hipblasLtBfloat16>(dC,
                                        dA,
                                        dB,
                                        alpha,
                                        beta,
                                        m,
                                        n,
                                        ldA,
                                        ldB,
                                        ldC,
                                        batchSize,
                                        batchStride,
                                        rowMajA,
                                        rowMajB,
                                        rowMajC,
                                        transA,
                                        transB);
        }
        else if(datatype == HIP_R_8I)
        {
            validation<int8_t>(dC,
                            dA,
                            dB,
                            alpha,
                            beta,
                            m,
                            n,
                            ldA,
                            ldB,
                            ldC,
                            batchSize,
                            batchStride,
                            rowMajA,
                            rowMajB,
                            rowMajC,
                            transA,
                            transB);
        }
        else if(datatype == HIP_R_32I)
        {
            validation<int32_t>(dC,
                                dA,
                                dB,
                                alpha,
                                beta,
                                m,
                                n,
                                ldA,
                                ldB,
                                ldC,
                                batchSize,
                                batchStride,
                                rowMajA,
                                rowMajB,
                                rowMajC,
                                transA,
                                transB);
        }

        hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
        hipblasLtErr = hipblasLtDestroy(handle);
        hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    }

    hipErr = hipSetDevice(curDevice);
    EXPECT_EQ(hipErr, hipSuccess);
}

TEST(MatrixTransformTest, Mxfp6Pack)
{
    // The rocKE packers are built for 4096 M (N) indices of K = 65536 and nothing else.
    constexpr uint64_t rows       = 4096;
    constexpr uint64_t k          = 65536;
    constexpr uint64_t kBlocks    = k / mx_tile128x4::kGroupK;
    constexpr size_t   dataBytes  = rows * k * 6 / 8;
    constexpr size_t   scaleBytes = rows * kBlocks;
    constexpr auto     e2m3       = static_cast<hipDataType>(HIP_R_6F_E2M3);
    constexpr auto     e3m2       = static_cast<hipDataType>(HIP_R_6F_E3M2);

    int             deviceId{};
    hipDeviceProp_t props{};
    ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&props, deviceId), hipSuccess);
    const bool gfx950 = std::string(props.gcnArchName).rfind("gfx950", 0) == 0;

    std::mt19937         rng(20261125);
    std::vector<uint8_t> data(dataBytes);
    std::vector<uint8_t> scales(scaleBytes);
    std::generate(data.begin(), data.end(), [&] { return static_cast<uint8_t>(rng()); });
    std::generate(scales.begin(), scales.end(), [&] { return static_cast<uint8_t>(rng()); });

    void* dA{};
    void* dC{};
    ASSERT_EQ(hipMalloc(&dA, dataBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dC, dataBytes), hipSuccess);

    hipblasLtHandle_t              handle{};
    hipblasLtMatrixTransformDesc_t desc{};
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixTransformDescCreate(&desc, HIP_R_32F), HIPBLAS_STATUS_SUCCESS);

    struct PackLayout
    {
        hipDataType      type;
        uint64_t         m;
        uint64_t         n;
        int64_t          ld;
        int32_t          batch;
        hipblasLtOrder_t order;
    };
    // The only A layouts the packers read, and the C layouts they write.
    const PackLayout colData{e2m3, k, rows, k, 1, HIPBLASLT_ORDER_COL};
    const PackLayout colScales{HIP_R_8F_UE8M0, kBlocks, rows, kBlocks, 1, HIPBLASLT_ORDER_COL};
    PackLayout       p16Data = colData;
    p16Data.order            = HIPBLASLT_ORDER_COL256_R128_P16_EXT;

    auto setScaleMode = [&](int32_t mode) {
        return hipblasLtMatrixTransformDescSetAttribute(
            desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_C_SCALE_MODE_EXT, &mode, sizeof(mode));
    };
    auto setOpA = [&](hipblasOperation_t opA) {
        return hipblasLtMatrixTransformDescSetAttribute(
            desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    };
    auto createLayout = [](const PackLayout& l) {
        hipblasLtMatrixLayout_t layout{};
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layout, l.type, l.m, l.n, l.ld),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      layout, HIPBLASLT_MATRIX_LAYOUT_ORDER, &l.order, sizeof(l.order)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &l.batch, sizeof(l.batch)),
                  HIPBLAS_STATUS_SUCCESS);
        return layout;
    };
    // Transforms A into dC; B, when set, is passed with A's layout.
    auto transform = [&](const PackLayout& a, const PackLayout& c, const void* A, const void* B) {
        hipblasLtMatrixLayout_t layoutA = createLayout(a);
        hipblasLtMatrixLayout_t layoutB = B ? createLayout(a) : nullptr;
        hipblasLtMatrixLayout_t layoutC = createLayout(c);
        const auto              status  = hipblasLtMatrixTransform(
            handle, desc, nullptr, A, layoutA, nullptr, B, layoutB, dC, layoutC, nullptr);
        EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layoutA), HIPBLAS_STATUS_SUCCESS);
        if(layoutB)
        {
            EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layoutB), HIPBLAS_STATUS_SUCCESS);
        }
        EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layoutC), HIPBLAS_STATUS_SUCCESS);
        return status;
    };
    auto pack
        = [&](const PackLayout& a, const PackLayout& c) { return transform(a, c, dA, nullptr); };
    auto edit = [](PackLayout l, auto change) {
        change(l);
        return l;
    };
    auto firstDifference = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        return static_cast<size_t>(std::mismatch(a.begin(), a.end(), b.begin()).first - a.begin());
    };

    // The packers bounds-check nothing, so every layout but the two above must be refused. These
    // checks run before the device check.
    struct Reject
    {
        const char* what;
        PackLayout  a;
        PackLayout  c;
    };
    const auto halfK = [](PackLayout& l) {
        l.m /= 2;
        l.ld /= 2;
    };
    const auto halfN     = [](PackLayout& l) { l.n = rows / 2; };
    const auto padLd     = [](PackLayout& l) { l.ld += 128; };
    const auto batch2    = [](PackLayout& l) { l.batch = 2; };
    const auto rowOrder  = [](PackLayout& l) { l.order = HIPBLASLT_ORDER_ROW; };
    const auto p16Order  = [](PackLayout& l) { l.order = HIPBLASLT_ORDER_COL256_R128_P16_EXT; };
    const auto e3m2Type  = [](PackLayout& l) { l.type = e3m2; };
    const auto int8Type  = [](PackLayout& l) { l.type = HIP_R_8I; };
    const auto fp8Type   = [](PackLayout& l) { l.type = HIP_R_8F_E4M3; };
    const auto withOrder = [&](hipblasLtOrder_t order) {
        return edit(p16Data, [order](PackLayout& l) { l.order = order; });
    };
    const std::vector<Reject> dataRejects = {
        {"K != 65536", edit(colData, halfK), edit(p16Data, halfK)},
        {"n != 4096", edit(colData, halfN), edit(p16Data, halfN)},
        {"ld != m", edit(colData, padLd), edit(p16Data, padLd)},
        {"A batch_count != 1", edit(colData, batch2), p16Data},
        {"C batch_count != 1", colData, edit(p16Data, batch2)},
        {"C type differs", colData, edit(p16Data, e3m2Type)},
        {"C m differs", colData, edit(p16Data, [](PackLayout& l) { l.m = k / 2; })},
        {"C n differs", colData, edit(p16Data, halfN)},
        {"C ld differs", colData, edit(p16Data, padLd)},
        {"A ROW", edit(colData, rowOrder), p16Data},
        {"A tiled", p16Data, p16Data},
        {"C order 104", colData, withOrder(HIPBLASLT_ORDER_COL256_R128_EXT)},
        {"C order 105", colData, withOrder(HIPBLASLT_ORDER_COL256_R128_P48_EXT)},
        {"C order 107", colData, withOrder(HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT)},
        {"C order 108", colData, withOrder(HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT)},
        {"not FP6", edit(colData, fp8Type), edit(p16Data, fp8Type)},
    };
    const std::vector<Reject> scaleRejects = {
        {"K blocks != 2048", edit(colScales, halfK), edit(colScales, halfK)},
        {"n != 4096", edit(colScales, halfN), edit(colScales, halfN)},
        {"ld != m", edit(colScales, padLd), edit(colScales, padLd)},
        {"A batch_count != 1", edit(colScales, batch2), colScales},
        {"C batch_count != 1", colScales, edit(colScales, batch2)},
        {"not UE8M0", edit(colScales, int8Type), edit(colScales, int8Type)},
        {"C type differs", colScales, edit(colScales, int8Type)},
        {"C not COL", colScales, edit(colScales, p16Order)},
        {"A ROW", edit(colScales, rowOrder), colScales},
        {"FP6 data", colData, p16Data},
        {"FP6 COL", colData, colData},
    };

    for(const auto& r : dataRejects)
        EXPECT_EQ(pack(r.a, r.c), HIPBLAS_STATUS_INVALID_VALUE) << "data: " << r.what;
    EXPECT_EQ(transform(colData, p16Data, nullptr, nullptr), HIPBLAS_STATUS_INVALID_VALUE)
        << "A NULL";
    EXPECT_EQ(transform(colData, p16Data, dA, dA), HIPBLAS_STATUS_INVALID_VALUE) << "B not NULL";
    ASSERT_EQ(setOpA(HIPBLAS_OP_T), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(pack(colData, p16Data), HIPBLAS_STATUS_INVALID_VALUE) << "data: opA T";
    ASSERT_EQ(setOpA(HIPBLAS_OP_N), HIPBLAS_STATUS_SUCCESS);

    ASSERT_EQ(setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT),
              HIPBLAS_STATUS_SUCCESS);
    for(const auto& r : scaleRejects)
        EXPECT_EQ(pack(r.a, r.c), HIPBLAS_STATUS_INVALID_VALUE) << "scales: " << r.what;
    EXPECT_EQ(transform(colScales, colScales, dA, dA), HIPBLAS_STATUS_INVALID_VALUE)
        << "scales: B not NULL";
    ASSERT_EQ(setOpA(HIPBLAS_OP_T), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(pack(colScales, colScales), HIPBLAS_STATUS_INVALID_VALUE) << "scales: opA T";
    ASSERT_EQ(setOpA(HIPBLAS_OP_N), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F), HIPBLAS_STATUS_SUCCESS);

    if(!gfx950)
    {
        EXPECT_EQ(pack(colData, p16Data), HIPBLAS_STATUS_ARCH_MISMATCH);
    }
    else
    {
        std::vector<uint8_t> expected(dataBytes);
        std::vector<uint8_t> packed(dataBytes);
        mx_tile128x4::tileFp6Data(
            expected.data(), data.data(), rows, k, mx_tile128x4::TileOrder::P16);
        ASSERT_EQ(hipMemcpy(dA, data.data(), dataBytes, hipMemcpyHostToDevice), hipSuccess);
        for(auto type : {e2m3, e3m2})
        {
            const auto setType = [type](PackLayout& l) { l.type = type; };
            ASSERT_EQ(hipMemset(dC, 0, dataBytes), hipSuccess);
            ASSERT_EQ(pack(edit(colData, setType), edit(p16Data, setType)), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            ASSERT_EQ(hipMemcpy(packed.data(), dC, dataBytes, hipMemcpyDeviceToHost), hipSuccess);
            EXPECT_EQ(firstDifference(packed, expected), dataBytes) << "first differing byte";
        }

        // dC is as large as the data, so the bytes after the scales show any write past them.
        constexpr uint8_t untouched = 0x5a;
        ASSERT_EQ(setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipMemcpy(dA, scales.data(), scaleBytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, untouched, dataBytes), hipSuccess);
        ASSERT_EQ(pack(colScales, colScales), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(packed.data(), dC, dataBytes, hipMemcpyDeviceToHost), hipSuccess);
        expected = mx_tile128x4::shuffleScales(scales, rows, kBlocks);
        expected.resize(dataBytes, untouched);
        EXPECT_EQ(firstDifference(packed, expected), dataBytes)
            << "first differing byte; the scales end at " << scaleBytes;
    }

    EXPECT_EQ(hipblasLtMatrixTransformDescDestroy(desc), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipFree(dA), hipSuccess);
    EXPECT_EQ(hipFree(dC), hipSuccess);
}

// Test support in libhipblaslt: replaces /sys/module/amdgpu/version; NULL restores it.
extern "C" void hipblaslt_page24_driver_for_test(const char* version);

namespace
{
    // A host copy of place_operand (with p24) and place_scales_b (with p24) from the rocKE MXFP6
    // gfx950 v8 package's tools/aid_runtime.hpp at phase 0, the layout the placed GEMM reads.
    // `split` is rocKE's operand A, hipBLASLt's B; the unsplit operand and the scales are
    // hipBLASLt's A.
    constexpr int64_t kAidBlock = 65536;
    constexpr int64_t kAidPage  = int64_t(2) << 20;

    std::vector<uint8_t> aidPage24Operand(
        const std::vector<uint8_t>& plain, bool split, int64_t rows, int64_t nk, int rowoff)
    {
        const int64_t        nr = rows / 256 / (split ? 2 : 1);
        std::vector<uint8_t> out(static_cast<size_t>((2 * 32 * nr + 2) * kAidPage), 0);
        for(int64_t r = 0; r < nr; ++r)
            for(int64_t s = 0; s < nk; ++s)
            {
                const int64_t sh = rowoff ? (r & (8 / rowoff - 1)) * rowoff * kAidBlock : 0;
                const int64_t q  = 32 * r + s / 24;
                const int64_t o
                    = (2 * q + ((q >> 6) & 1)) * kAidPage + (s % 24) * kAidBlock + sh;
                std::memcpy(out.data() + o, plain.data() + (r * nk + s) * kAidBlock, kAidBlock);
            }
        return out;
    }

    std::vector<uint8_t> aidPlaceOperand(
        const std::vector<uint8_t>& img, bool split, int64_t rows, int64_t k, int rowoff)
    {
        const int64_t        nk = k / 128, nt = rows / 256;
        std::vector<uint8_t> out(
            static_cast<size_t>((split ? rows / 256 / 2 : rows / 256) * nk * kAidBlock), 0);
        for(int64_t m = 0; m < nt; ++m)
            for(int64_t s = 0; s < nk; ++s)
            {
                const int64_t  blk = (split ? (m & 7) : m) * nk + s;
                const uint8_t* src = img.data() + (m * nk + s) * 24576;
                for(int h = 0; h < 2; ++h)
                {
                    if(split && h != static_cast<int>((m >> 3) & 1))
                        continue;
                    for(int w = 0; w < 4; ++w)
                        for(int i = 0; i < 2; ++i)
                            std::memcpy(out.data() + blk * kAidBlock + w * 16384 + h * 8192
                                            + i * 4096,
                                        src + w * 6144 + i * 3072,
                                        3072);
                }
            }
        return aidPage24Operand(out, split, rows, nk, rowoff);
    }

    std::vector<uint8_t> aidPlaceScales(const std::vector<uint8_t>& sc, int64_t k)
    {
        const int64_t        nk = k / 128;
        std::vector<uint8_t> out(static_cast<size_t>(nk * 32768), 0);
        for(int64_t s = 0; s < nk; ++s)
            for(int g = 0; g < 2; ++g)
                for(int h = 0; h < 2; ++h)
                    std::memcpy(out.data() + s * 32768 + g * 16384 + h * 8192,
                                sc.data() + s * 16384 + g * 8192,
                                8192);
        std::vector<uint8_t> o(static_cast<size_t>((2 * ((nk + 23) / 24) + 2) * kAidPage), 0);
        for(int64_t s = 0; s < nk; ++s)
            std::memcpy(o.data() + 2 * (s / 24) * kAidPage + (s % 24) * 32768,
                        out.data() + s * 32768,
                        32768);
        return o;
    }

    // Whether this machine's amdgpu has the library's driver rule: 6.19 or newer.
    bool page24DriverRuleHolds()
    {
        std::ifstream in("/sys/module/amdgpu/version");
        std::string   version;
        std::getline(in, version);
        return TensileLite::hip::page24RuleHolds(version);
    }

    // Restores the library's near-die driver hook when a test ends, however it ends.
    struct Page24HooksReset
    {
        ~Page24HooksReset()
        {
            hipblaslt_page24_driver_for_test(nullptr);
        }
    };

    // The near-die placements of one MXFP6 operand of 4096 rows and K = 65536, the only shape the
    // kernels are built for, through hipblasLtMatrixTransform.
    struct Page24Harness
    {
        static constexpr uint64_t rows        = 4096;
        static constexpr uint64_t k           = 65536;
        static constexpr uint64_t kBlocks     = k / 32;
        static constexpr size_t   dataBytes   = rows * k * 6 / 8;
        static constexpr size_t   scaleBytes  = rows * kBlocks;
        static constexpr size_t   splitBytes  = size_t(514) << 21;
        static constexpr size_t   dupBytes    = size_t(1026) << 21;
        static constexpr size_t   placedScale = size_t(46) << 21;
        static constexpr int      rowoff      = 1;

        struct Layout
        {
            hipDataType      type;
            uint64_t         m;
            uint64_t         n;
            int64_t          ld;
            hipblasLtOrder_t order;
            int32_t          batch = 1;
        };
        static constexpr auto e2m3 = static_cast<hipDataType>(HIP_R_6F_E2M3);
        static constexpr auto e3m2 = static_cast<hipDataType>(HIP_R_6F_E3M2);
        const Layout          p48Data{e2m3, k, rows, k, HIPBLASLT_ORDER_COL256_R128_P48_EXT};
        const Layout          splitData{
            e2m3, k, rows, k, HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT}; // B
        const Layout dupData{e2m3, k, rows, k, HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT}; // A
        const Layout colScales{HIP_R_8F_UE8M0, kBlocks, rows, kBlocks, HIPBLASLT_ORDER_COL};

        int                            deviceId{};
        bool                           gfx950{};
        hipblasLtHandle_t              handle{};
        hipblasLtMatrixTransformDesc_t desc{};
        void*                          dA{};
        void*                          dC{};
        size_t                         cBytes{};
        // The source images: 48-byte tiles (order 105) and 128x4 scales (scale mode 1018). Their
        // content is opaque to the placement, so random bytes stand in for both.
        std::vector<uint8_t> image;
        std::vector<uint8_t> scales;
        std::vector<uint8_t> placed;

        // Only gfx950 runs the kernels, so only gfx950 gets buffers of their size; elsewhere the
        // calls stop at the device check, before C is written.
        void init()
        {
            hipDeviceProp_t props{};
            ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
            ASSERT_EQ(hipGetDeviceProperties(&props, deviceId), hipSuccess);
            gfx950 = std::string(props.gcnArchName).rfind("gfx950", 0) == 0;
            cBytes = gfx950 ? dupBytes : 4096;
            ASSERT_EQ(hipMalloc(&dA, gfx950 ? dataBytes : 4096), hipSuccess);
            ASSERT_EQ(hipMalloc(&dC, cBytes), hipSuccess);
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixTransformDescCreate(&desc, HIP_R_32F), HIPBLAS_STATUS_SUCCESS);
            if(!gfx950)
                return;
            std::mt19937 rng(20261126);
            image.resize(dataBytes);
            scales.resize(scaleBytes);
            for(auto* v : {&image, &scales})
                for(size_t i = 0; i < v->size(); i += 4)
                {
                    const uint32_t x = rng();
                    std::memcpy(v->data() + i, &x, 4);
                }
        }

        ~Page24Harness()
        {
            if(desc)
                EXPECT_EQ(hipblasLtMatrixTransformDescDestroy(desc), HIPBLAS_STATUS_SUCCESS);
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipFree(dA), hipSuccess);
            EXPECT_EQ(hipFree(dC), hipSuccess);
        }

        hipblasStatus_t setScaleMode(int32_t mode)
        {
            return hipblasLtMatrixTransformDescSetAttribute(
                desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_C_SCALE_MODE_EXT, &mode, sizeof(mode));
        }

        hipblasStatus_t setOpA(hipblasOperation_t opA)
        {
            return hipblasLtMatrixTransformDescSetAttribute(
                desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
        }

        // Loads the data image or the scales into A, for a data or a scale placement.
        void loadA(const std::vector<uint8_t>& src)
        {
            ASSERT_EQ(hipMemcpy(dA, src.data(), src.size(), hipMemcpyHostToDevice), hipSuccess);
        }

        static hipblasLtMatrixLayout_t createLayout(const Layout& l)
        {
            hipblasLtMatrixLayout_t layout{};
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layout, l.type, l.m, l.n, l.ld),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                          layout, HIPBLASLT_MATRIX_LAYOUT_ORDER, &l.order, sizeof(l.order)),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                          layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &l.batch, sizeof(l.batch)),
                      HIPBLAS_STATUS_SUCCESS);
            return layout;
        }

        // Transforms `A` (dA by default) into dC; B, when set, is passed with A's layout.
        hipblasStatus_t
            transform(const Layout& a, const Layout& c, const void* A, const void* B = nullptr)
        {
            hipblasLtMatrixLayout_t layoutA = createLayout(a);
            hipblasLtMatrixLayout_t layoutB = B ? createLayout(a) : nullptr;
            hipblasLtMatrixLayout_t layoutC = createLayout(c);
            const auto              status  = hipblasLtMatrixTransform(
                handle, desc, nullptr, A, layoutA, nullptr, B, layoutB, dC, layoutC, nullptr);
            EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layoutA), HIPBLAS_STATUS_SUCCESS);
            if(layoutB)
            {
                EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layoutB), HIPBLAS_STATUS_SUCCESS);
            }
            EXPECT_EQ(hipblasLtMatrixLayoutDestroy(layoutC), HIPBLAS_STATUS_SUCCESS);
            return status;
        }

        hipblasStatus_t transform(const Layout& a, const Layout& c)
        {
            return transform(a, c, dA);
        }

        // The whole of C against the host copy: C starts zeroed, and the bytes past `expected`
        // and those it does not place stay zero.
        void check(const std::string&          what,
                   const Layout&               a,
                   const Layout&               c,
                   const std::vector<uint8_t>& expected)
        {
            SCOPED_TRACE(what);
            ASSERT_LE(expected.size(), cBytes);
            ASSERT_EQ(hipMemset(dC, 0, cBytes), hipSuccess);
            ASSERT_EQ(transform(a, c), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            placed.resize(cBytes);
            ASSERT_EQ(hipMemcpy(placed.data(), dC, cBytes, hipMemcpyDeviceToHost), hipSuccess);
            const auto first = static_cast<size_t>(
                std::mismatch(expected.begin(), expected.end(), placed.begin()).first
                - expected.begin());
            EXPECT_EQ(first, expected.size()) << "first differing byte of " << expected.size();
            EXPECT_TRUE(std::all_of(
                placed.begin() + expected.size(), placed.end(), [](uint8_t b) { return b == 0; }))
                << "a write past the " << expected.size() << " bytes of the layout";
        }

        // A call that must fail with `status` and leave all of C as it was.
        void checkUntouched(const std::string& what,
                            const Layout&      a,
                            const Layout&      c,
                            hipblasStatus_t    status)
        {
            SCOPED_TRACE(what);
            constexpr uint8_t untouched = 0x5a;
            ASSERT_EQ(hipMemset(dC, untouched, cBytes), hipSuccess);
            EXPECT_EQ(transform(a, c), status);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            placed.resize(cBytes);
            ASSERT_EQ(hipMemcpy(placed.data(), dC, cBytes, hipMemcpyDeviceToHost), hipSuccess);
            EXPECT_TRUE(std::all_of(placed.begin(), placed.end(), [](uint8_t b) {
                return b == untouched;
            })) << "C was written";
        }

        // Every placement against the host copy of aid_runtime.hpp.
        void checkAll(const std::string& how)
        {
            const std::string at = ", " + how;
            ASSERT_EQ(setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F),
                      HIPBLAS_STATUS_SUCCESS);
            loadA(image);
            {
                const auto expected = aidPlaceOperand(image, true, rows, k, rowoff);
                ASSERT_EQ(expected.size(), splitBytes);
                check("B, order 108" + at, p48Data, splitData, expected);
                Layout a = p48Data, c = splitData;
                a.type = c.type = e3m2;
                check("B, order 108, E3M2" + at, a, c, expected);
            }
            {
                const auto expected = aidPlaceOperand(image, false, rows, k, rowoff);
                ASSERT_EQ(expected.size(), dupBytes);
                check("A, order 107" + at, p48Data, dupData, expected);
            }
            ASSERT_EQ(setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT),
                      HIPBLAS_STATUS_SUCCESS);
            loadA(scales);
            {
                const auto expected = aidPlaceScales(scales, k);
                ASSERT_EQ(expected.size(), placedScale);
                check("A scales, scale mode 1019" + at, colScales, colScales, expected);
            }
            ASSERT_EQ(setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F),
                      HIPBLAS_STATUS_SUCCESS);
        }
    };
}

TEST(MatrixTransformTest, Mxfp6Page24)
{
    Page24HooksReset reset;
    Page24Harness    h;
    h.init();
    if(HasFatalFailure())
        return;
    using Layout = Page24Harness::Layout;
    auto edit    = [](Layout l, auto change) {
        change(l);
        return l;
    };

    // Shapes and layouts other than the three placements are refused before the device check.
    const auto halfK = [](Layout& l) {
        l.m /= 2;
        l.ld /= 2;
    };
    const auto halfN  = [](Layout& l) { l.n = Page24Harness::rows / 2; };
    const auto padLd  = [](Layout& l) { l.ld += 128; };
    const auto batch2 = [](Layout& l) { l.batch = 2; };
    const auto colOrd = [](Layout& l) { l.order = HIPBLASLT_ORDER_COL; };
    const auto e3m2Ty = [](Layout& l) { l.type = Page24Harness::e3m2; };
    const auto fp8Ty  = [](Layout& l) { l.type = HIP_R_8F_E4M3; };
    const auto ord    = [](hipblasLtOrder_t o) { return [o](Layout& l) { l.order = o; }; };
    struct Reject
    {
        const char* what;
        Layout      a;
        Layout      c;
    };
    const std::vector<Reject> dataRejects = {
        {"K != 65536", edit(h.p48Data, halfK), edit(h.splitData, halfK)},
        {"n != 4096", edit(h.p48Data, halfN), edit(h.dupData, halfN)},
        {"ld != m", edit(h.p48Data, padLd), edit(h.dupData, padLd)},
        {"A batch_count != 1", edit(h.p48Data, batch2), h.splitData},
        {"C batch_count != 1", h.p48Data, edit(h.dupData, batch2)},
        {"C type differs", h.p48Data, edit(h.splitData, e3m2Ty)},
        {"A in COL", edit(h.p48Data, colOrd), h.splitData},
        {"A in order 104", edit(h.p48Data, ord(HIPBLASLT_ORDER_COL256_R128_EXT)), h.dupData},
        {"A in order 106", edit(h.p48Data, ord(HIPBLASLT_ORDER_COL256_R128_P16_EXT)), h.dupData},
        {"A already placed", h.splitData, h.splitData},
        {"C in order 105", h.p48Data, h.p48Data},
        {"not FP6", edit(h.p48Data, fp8Ty), edit(h.dupData, fp8Ty)},
    };
    for(const auto& r : dataRejects)
        EXPECT_EQ(h.transform(r.a, r.c), HIPBLAS_STATUS_INVALID_VALUE) << "data: " << r.what;
    EXPECT_EQ(h.transform(h.p48Data, h.splitData, nullptr), HIPBLAS_STATUS_INVALID_VALUE)
        << "data: A NULL";
    EXPECT_EQ(h.transform(h.p48Data, h.dupData, h.dA, h.dA), HIPBLAS_STATUS_INVALID_VALUE)
        << "data: B not NULL";
    ASSERT_EQ(h.setOpA(HIPBLAS_OP_T), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(h.transform(h.p48Data, h.splitData), HIPBLAS_STATUS_INVALID_VALUE) << "data: opA T";
    ASSERT_EQ(h.setOpA(HIPBLAS_OP_N), HIPBLAS_STATUS_SUCCESS);

    ASSERT_EQ(h.setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT),
              HIPBLAS_STATUS_SUCCESS);
    const std::vector<Reject> scaleRejects = {
        {"K blocks != 2048", edit(h.colScales, halfK), edit(h.colScales, halfK)},
        {"n != 4096", edit(h.colScales, halfN), edit(h.colScales, halfN)},
        {"ld != m", edit(h.colScales, padLd), edit(h.colScales, padLd)},
        {"A batch_count != 1", edit(h.colScales, batch2), h.colScales},
        {"C batch_count != 1", h.colScales, edit(h.colScales, batch2)},
        {"C not COL", h.colScales, edit(h.colScales, ord(h.dupData.order))},
        {"FP6 data", h.p48Data, h.dupData},
    };
    for(const auto& r : scaleRejects)
        EXPECT_EQ(h.transform(r.a, r.c), HIPBLAS_STATUS_INVALID_VALUE) << "scales: " << r.what;
    EXPECT_EQ(h.transform(h.colScales, h.colScales, nullptr), HIPBLAS_STATUS_INVALID_VALUE)
        << "scales: A NULL";
    ASSERT_EQ(h.setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F), HIPBLAS_STATUS_SUCCESS);

    if(!h.gfx950)
    {
        EXPECT_EQ(h.transform(h.p48Data, h.splitData), HIPBLAS_STATUS_ARCH_MISMATCH);
        return;
    }

    hipblaslt_page24_driver_for_test("6.19.14.31400000");
    h.checkAll("amdgpu 6.19");
    if(HasFatalFailure())
        return;

    // The public path: this machine's driver has the rule, or every placement fails without
    // writing C.
    hipblaslt_page24_driver_for_test(nullptr);
    if(page24DriverRuleHolds())
    {
        h.checkAll("this driver");
    }
    else
    {
        h.checkUntouched("B, order 108", h.p48Data, h.splitData, HIPBLAS_STATUS_NOT_SUPPORTED);
        h.checkUntouched("A, order 107", h.p48Data, h.dupData, HIPBLAS_STATUS_NOT_SUPPORTED);
        ASSERT_EQ(h.setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT),
                  HIPBLAS_STATUS_SUCCESS);
        h.checkUntouched(
            "A scales, scale mode 1019", h.colScales, h.colScales, HIPBLAS_STATUS_NOT_SUPPORTED);
    }
}

// The driver rule on versions this machine does not run: amdgpu 6.19 and newer have it; an
// older version launches nothing.
TEST(MatrixTransformTest, Mxfp6Page24DriverRule)
{
    Page24HooksReset reset;
    Page24Harness    h;
    h.init();
    if(HasFatalFailure())
        return;

    const std::vector<const char*> rejected = {"6.16", "6.18", "6.16.13", "6", ""};
    const std::vector<const char*> accepted
        = {"6.19", "6.20", "7.0", "7.1", "7.10", "6.19.14.31400000", "7.1.9.31600000"};

    if(!h.gfx950)
    {
        for(const auto* v : {"6.16", "6.19"})
        {
            hipblaslt_page24_driver_for_test(v);
            EXPECT_EQ(h.transform(h.p48Data, h.splitData), HIPBLAS_STATUS_ARCH_MISMATCH) << v;
        }
        return;
    }

    for(const auto* v : rejected)
    {
        hipblaslt_page24_driver_for_test(v);
        const std::string at = std::string(", amdgpu '") + v + "'";
        ASSERT_EQ(h.setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F), HIPBLAS_STATUS_SUCCESS);
        h.checkUntouched("B, order 108" + at, h.p48Data, h.splitData, HIPBLAS_STATUS_NOT_SUPPORTED);
        h.checkUntouched("A, order 107" + at, h.p48Data, h.dupData, HIPBLAS_STATUS_NOT_SUPPORTED);
        ASSERT_EQ(h.setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT),
                  HIPBLAS_STATUS_SUCCESS);
        h.checkUntouched("A scales" + at, h.colScales, h.colScales, HIPBLAS_STATUS_NOT_SUPPORTED);
    }

    ASSERT_EQ(h.setScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT),
              HIPBLAS_STATUS_SUCCESS);
    h.loadA(h.scales);
    const auto expected = aidPlaceScales(h.scales, Page24Harness::k);
    for(const auto* v : accepted)
    {
        hipblaslt_page24_driver_for_test(v);
        h.check(std::string("A scales, amdgpu '") + v + "'", h.colScales, h.colScales, expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    MatrixTransformTest,
    ::testing::Combine(::testing::ValuesIn({int64_t(1), int64_t(127), int64_t(1024)}),
                       ::testing::ValuesIn({int64_t(1), int64_t(127), int64_t(1024)}),
                       ::testing::ValuesIn({HIP_R_32F, HIP_R_16F, HIP_R_16BF, HIP_R_8I, HIP_R_32I}),
                       ::testing::ValuesIn({HIP_R_32F}),
                       ::testing::ValuesIn({HIPBLAS_OP_N, HIPBLAS_OP_T}),
                       ::testing::ValuesIn({HIPBLAS_OP_N, HIPBLAS_OP_T}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL})));
