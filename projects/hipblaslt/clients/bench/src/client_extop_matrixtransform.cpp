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
#include <cstddef>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt_datatype2string.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>
#include <hipblaslt/host_numerics/MatrixTransformReference.hpp>
#include <vector>

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
    TypedMatrixTransformIO(int64_t m, int64_t n, int64_t b, hipblaslt_initialization initMethod)
    {
        auto        hipErr = hipMalloc(&this->a, m * n * b * sizeof(DType));
        hipErr = hipMalloc(&this->b, m * n * b * sizeof(DType));
        hipErr = hipMalloc(&this->c, m * n * b * sizeof(DType));
        init(this->a, m * n * b, initMethod);
        init(this->b, m * n * b, initMethod);
    }

    ~TypedMatrixTransformIO() override
    {
        auto        hipErr = hipFree(a);
        hipErr = hipFree(b);
        hipErr = hipFree(c);
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
    void init(DType* buf, size_t len, hipblaslt_initialization initMethod)
    {
        std::vector<DType> ref(len);
        hipblaslt::host_numerics::initialize(
            ref.data(), ref.size(), initMethod);

        auto err = hipMemcpy(buf, ref.data(), len * sizeof(DType), hipMemcpyHostToDevice);
    }

private:
    DType* a{};
    DType* b{};
    DType* c{};
};

using MatrixTransformIOPtr = std::unique_ptr<MatrixTransformIO>;
MatrixTransformIOPtr makeMatrixTransformIOPtr(
    hipDataType datatype, int64_t m, int64_t n, int64_t b, hipblaslt_initialization init)
{
    if(datatype == HIP_R_32F)
    {
        return std::make_unique<TypedMatrixTransformIO<hipblasLtFloat>>(m, n, b, init);
    }
    else if(datatype == HIP_R_16F)
    {
        return std::make_unique<TypedMatrixTransformIO<hipblasLtHalf>>(m, n, b, init);
    }
    else if(datatype == HIP_R_16BF)
    {
        return std::make_unique<TypedMatrixTransformIO<hipblasLtBfloat16>>(m, n, b, init);
    }
    else if(datatype == HIP_R_8I)
    {
        return std::make_unique<TypedMatrixTransformIO<int8_t>>(m, n, b, init);
    }
    return nullptr;
}

hipDataType str2Datatype(const std::string& typeStr)
{
    if(typeStr == "fp32")
    {
        return HIP_R_32F;
    }
    else if(typeStr == "fp16")
    {
        return HIP_R_16F;
    }
    else if(typeStr == "bf16")
    {
        return HIP_R_16BF;
    }
    else if(typeStr == "i8")
    {
        return HIP_R_8I;
    }
    else if(typeStr == "i32")
    {
        return HIP_R_32I;
    }

    return HIPBLASLT_DATATYPE_INVALID;
}

static int parseArguments(int                       argc,
                          char*                     argv[],
                          hipDataType&              datatype,
                          hipDataType&              scaleDatatype,
                          int64_t&                  m,
                          int64_t&                  n,
                          float&                    alpha,
                          float&                    beta,
                          bool&                     transA,
                          bool&                     transB,
                          uint32_t&                 ldA,
                          uint32_t&                 ldB,
                          uint32_t&                 ldC,
                          bool&                     rowMajA,
                          bool&                     rowMajB,
                          bool&                     rowMajC,
                          int32_t&                  batchSize,
                          int64_t&                  batchStride,
                          bool&                     runValidation,
                          hipblaslt_initialization& init)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if(arg == "--alpha" || arg == "-alpha")
                {
                    alpha = atof(argv[++i]);
                }
                else if(arg == "--beta" || arg == "-beta")
                {
                    beta = atof(argv[++i]);
                }
                else if(arg == "--trans_a")
                {
                    transA = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--trans_b")
                {
                    transB = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--ld_a")
                {
                    ldA = atoi(argv[++i]);
                }
                else if(arg == "--ld_b")
                {
                    ldB = atoi(argv[++i]);
                }
                else if(arg == "--ld_c")
                {
                    ldC = atoi(argv[++i]);
                }
                else if(arg == "--batch_size")
                {
                    batchSize = atoi(argv[++i]);
                }
                else if(arg == "--batch_stride")
                {
                    batchStride = atoi(argv[++i]);
                }
                else if(arg == "--datatype")
                {
                    datatype = str2Datatype(argv[++i]);
                }
                else if(arg == "--scale_datatype")
                {
                    scaleDatatype = str2Datatype(argv[++i]);
                }
                else if(arg == "--row_maj_a")
                {
                    rowMajA = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--row_maj_b")
                {
                    rowMajB = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--row_maj_c")
                {
                    rowMajC = (atoi(argv[++i]) > 0);
                }
                else if(arg == "--validation" || arg == "-V")
                {
                    runValidation = true;
                }
                else if(arg == "--initialization" || arg == "--init")
                {
                    const std::string initStr{argv[++i]};

                    if(initStr != "rand_int" && initStr != "trig_float" && initStr != "hpl" && initStr != "special" && initStr != "zero")
                    {
                        std::cerr << "Invalid initialization type: " << initStr << '\n';
                        return EXIT_FAILURE;
                    }

                    init = string2hipblaslt_initialization(initStr);
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    return EXIT_SUCCESS;
}

template <bool RowMaj>
int64_t getLeadingDimSize(int64_t numRows, int64_t numCols)
{
    return RowMaj ? numCols : numRows;
}

void validation(hipDataType datatype,
                void*       c,
                void*       a,
                void*       b,
                float       alpha,
                float       beta,
                uint32_t    m,
                uint32_t    n,
                uint32_t    ldA,
                uint32_t    ldB,
                uint32_t    ldC,
                uint32_t    batchSize,
                uint32_t    batchStride,
                bool        rowMajA,
                bool        rowMajB,
                bool        rowMajC,
                bool        transA,
                bool        transB)
{
    const auto   scalarType   = hipblaslt::host_numerics::scalarType(datatype);
    const size_t elementBytes = roc::host_numerics::scalarTypeInfo(scalarType).storageBits / 8;
    const size_t storageBytes = size_t(m) * n * batchSize * elementBytes;
    std::vector<std::byte> hA(storageBytes);
    std::vector<std::byte> hB(storageBytes);
    std::vector<std::byte> hC(storageBytes);
    auto                   hipErr = hipMemcpyDtoH(hA.data(), a, storageBytes);
    hipErr                        = hipMemcpyDtoH(hB.data(), b, storageBytes);
    hipErr                        = hipMemcpyDtoH(hC.data(), c, storageBytes);

    hipblaslt::host_numerics::MatrixTransformReferenceArguments arguments;
    arguments.observed               = hC.data();
    arguments.observedStorageBytes   = storageBytes;
    arguments.a                      = hA.data();
    arguments.aStorageBytes          = storageBytes;
    arguments.b                      = hB.data();
    arguments.bStorageBytes          = storageBytes;
    arguments.type                   = datatype;
    arguments.rows                   = m;
    arguments.columns                = n;
    arguments.batchCount             = batchSize;
    arguments.leadingDimensionA      = ldA;
    arguments.leadingDimensionB      = ldB;
    arguments.leadingDimensionOutput = ldC;
    arguments.batchStride            = batchStride;
    arguments.rowMajorA              = rowMajA;
    arguments.rowMajorB              = rowMajB;
    arguments.rowMajorOutput         = rowMajC;
    arguments.transposeA             = transA;
    arguments.transposeB             = transB;
    arguments.alpha                  = alpha;
    arguments.beta                   = beta;

    const auto comparison = hipblaslt::host_numerics::referenceMatrixTransform(arguments);
    if(!comparison.passed())
    {
        hipblaslt::host_numerics::reportMatrixTransformMismatches(std::cerr, comparison);
        std::cerr << '\n';
    }
}

int main(int argc, char** argv)
{
    int64_t                  m         = 2048;
    int64_t                  n         = 2048;
    int32_t                  batchSize = 1;
    float                    alpha     = 1;
    float                    beta      = 1;
    auto                     transA    = false;
    auto                     transB    = false;
    auto                     rowMajA   = false;
    auto                     rowMajB   = false;
    auto                     rowMajC   = false;
    int64_t                  batchStride{};
    uint32_t                 ldA{};
    uint32_t                 ldB{};
    uint32_t                 ldC{};
    bool                     runValidation{};
    hipblaslt_initialization init{hipblaslt_initialization::hpl};
    hipDataType              datatype{HIP_R_32F};
    hipDataType              scaleDatatype{HIP_R_32F};
    parseArguments(argc,
                   argv,
                   datatype,
                   scaleDatatype,
                   m,
                   n,
                   alpha,
                   beta,
                   transA,
                   transB,
                   ldA,
                   ldB,
                   ldC,
                   rowMajA,
                   rowMajB,
                   rowMajC,
                   batchSize,
                   batchStride,
                   runValidation,
                   init);

    if(!ldA || !ldB || !ldC)
    {
        ldA = rowMajA ? getLeadingDimSize<true>(m, n) : getLeadingDimSize<false>(m, n);
        ldB = rowMajB ? getLeadingDimSize<true>(m, n) : getLeadingDimSize<false>(m, n);
        ldC = rowMajC ? getLeadingDimSize<true>(m, n) : getLeadingDimSize<false>(m, n);
    }

    if(!batchStride)
    {
        batchStride = m * n;
    }

    hipblasLtOrder_t orderA = rowMajA ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    hipblasLtOrder_t orderB = rowMajB ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    hipblasLtOrder_t orderC = rowMajC ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    auto             tA     = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    auto             tB     = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize, init);
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

    if(hipblasLtErr)
    {
        return EXIT_FAILURE;
    }

    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = transA ? n : m;
    shapeA.second = transA ? m : n;
    shapeB.first  = transB ? n : m;
    shapeB.second = transB ? m : n;

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &tA, sizeof(tA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &tB, sizeof(tB));
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
    //warmup
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);

    if(hipblasLtErr)
    {
        std::cerr << "Unable to launch hipblasLtMatrixTransform\n";
        return EXIT_FAILURE;
    }

    auto err = hipStreamSynchronize(nullptr);

    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    err = hipEventCreate(&stop);
    err = hipEventRecord(start);

    constexpr int numRuns{50};

    for(int i = 0; i < numRuns; ++i)
    {
        hipblasLtErr = hipblasLtMatrixTransform(
            handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    }

    err = hipEventRecord(stop);
    err = hipStreamSynchronize(nullptr);
    float dur{};
    float refDur{};
    err               = hipEventElapsedTime(&dur, start, stop);
    const auto avgDur = dur / numRuns;
    std::cout << "hipblasLtMatrixTransform elapsed time: " << std::to_string(avgDur) << " ms\n";
    std::cout << "Throughput: "
              << 3. * m * n * batchSize * inputs->elemNumBytes() / std::pow(1024, 4) / avgDur * 1e3
              << " TB/s\n";

    if(runValidation)
    {
        validation(datatype,
                   dC,
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

releaseResource:
    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
    return EXIT_SUCCESS;
}
