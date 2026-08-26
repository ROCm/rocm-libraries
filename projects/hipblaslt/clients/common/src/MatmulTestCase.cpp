// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/client/MatmulTestCase.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace hipblaslt::client
{
    namespace
    {
        hipblasOperation_t normalizeOperation(char operation, const char* name)
        {
            switch(operation)
            {
            case 'N':
            case 'n':
                return HIPBLAS_OP_N;
            case 'T':
            case 't':
                return HIPBLAS_OP_T;
            case 'C':
            case 'c':
                return HIPBLAS_OP_C;
            default:
                throw std::invalid_argument(std::string("Invalid ") + name + " operation.");
            }
        }

        size_t normalizeExtent(int64_t value, const char* name)
        {
            if(value < 0)
                throw std::invalid_argument(std::string(name) + " must be non-negative.");
            return static_cast<size_t>(value);
        }

        ptrdiff_t normalizeStride(int64_t value, const char* name)
        {
            if(value < 0)
                throw std::invalid_argument(std::string(name) + " must be non-negative.");
            if(static_cast<uint64_t>(value)
               > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()))
                throw std::overflow_error(std::string(name) + " exceeds ptrdiff_t.");
            return static_cast<ptrdiff_t>(value);
        }

        int64_t checkedProduct(int64_t left, int64_t right, const char* name)
        {
            if(left < 0 || right < 0)
                throw std::invalid_argument(std::string(name) + " operands must be non-negative.");
            if(left != 0 && right > std::numeric_limits<int64_t>::max() / left)
                throw std::overflow_error(std::string(name) + " overflow.");
            return left * right;
        }

        MatmulMatrix normalizeMatrix(hipDataType type,
                                     int64_t     rows,
                                     int64_t     columns,
                                     int64_t     leadingDimension,
                                     int64_t     batchStride,
                                     int32_t     batchCount)
        {
            using roc::host_validation::Layout;
            using roc::host_validation::Shape;

            return {
                type,
                hipblaslt::host_validation::scalarType(type),
                Layout(Shape{normalizeExtent(rows, "matrix rows"),
                             normalizeExtent(columns, "matrix columns"),
                             static_cast<size_t>(batchCount)},
                       {1,
                        normalizeStride(leadingDimension, "matrix leading dimension"),
                        normalizeStride(batchStride, "matrix batch stride")}),
            };
        }
    } // namespace

    std::vector<MatmulTestCase> normalizeMatmulCases(const Arguments& arguments)
    {
        const int32_t caseCount  = std::max(1, arguments.grouped_gemm);
        const int32_t batchCount = std::max(1, arguments.batch_count);
        if(caseCount > static_cast<int32_t>(MAX_SUPPORTED_NUM_PROBLEMS))
            throw std::invalid_argument("Grouped GEMM count exceeds the Arguments capacity.");

        hipblasLtBatchMode_t batchMode;
        switch(arguments.batch_mode)
        {
        case HIPBLASLT_BATCH_MODE_STRIDED:
            batchMode = HIPBLASLT_BATCH_MODE_STRIDED;
            break;
        case HIPBLASLT_BATCH_MODE_POINTER_ARRAY:
            batchMode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
            break;
        default:
            throw std::invalid_argument("Invalid hipBLASLt batch mode.");
        }

        const hipblasOperation_t operationA = normalizeOperation(arguments.transA, "transpose A");
        const hipblasOperation_t operationB = normalizeOperation(arguments.transB, "transpose B");

        if(arguments.c_equal_d && arguments.c_type != arguments.d_type)
            throw std::invalid_argument("C and D must have the same type when they share storage.");

        std::vector<MatmulTestCase> cases;
        cases.reserve(caseCount);
        for(int32_t index = 0; index < caseCount; ++index)
        {
            const int64_t m = arguments.M[index];
            const int64_t n = arguments.N[index];
            const int64_t k = arguments.K[index];

            const int64_t aRows    = operationA == HIPBLAS_OP_N ? m : k;
            const int64_t aColumns = operationA == HIPBLAS_OP_N ? k : m;
            const int64_t bRows    = operationB == HIPBLAS_OP_N ? k : n;
            const int64_t bColumns = operationB == HIPBLAS_OP_N ? n : k;

            const bool useArgumentStrides
                = batchMode == HIPBLASLT_BATCH_MODE_STRIDED && batchCount > 1;
            const int64_t strideA
                = useArgumentStrides
                      ? arguments.stride_a[index]
                      : checkedProduct(arguments.lda[index], aColumns, "canonical A batch stride");
            const int64_t strideB
                = useArgumentStrides
                      ? arguments.stride_b[index]
                      : checkedProduct(arguments.ldb[index], bColumns, "canonical B batch stride");
            const int64_t strideC
                = useArgumentStrides
                      ? arguments.stride_c[index]
                      : checkedProduct(arguments.ldc[index], n, "canonical C batch stride");
            const int64_t strideD
                = arguments.c_equal_d ? strideC
                  : useArgumentStrides
                      ? arguments.stride_d[index]
                      : checkedProduct(arguments.ldd[index], n, "canonical D batch stride");
            MatmulTestCase testCase{
                .m          = m,
                .n          = n,
                .k          = k,
                .operationA = operationA,
                .operationB = operationB,
                .batchMode  = batchMode,
                .batchCount = batchCount,
                .a          = normalizeMatrix(
                    arguments.a_type, aRows, aColumns, arguments.lda[index], strideA, batchCount),
                .b = normalizeMatrix(
                    arguments.b_type, bRows, bColumns, arguments.ldb[index], strideB, batchCount),
                .c = normalizeMatrix(
                    arguments.c_type, m, n, arguments.ldc[index], strideC, batchCount),
                .d
                = normalizeMatrix(arguments.d_type,
                                  m,
                                  n,
                                  arguments.c_equal_d ? arguments.ldc[index] : arguments.ldd[index],
                                  strideD,
                                  batchCount),
                .auxiliary = std::nullopt,
                .cEqualsD  = arguments.c_equal_d,
            };
            if(arguments.use_e)
            {
                const int64_t strideE
                    = useArgumentStrides
                          ? arguments.stride_e[index]
                          : checkedProduct(arguments.lde[index], n, "canonical E batch stride");
                testCase.auxiliary = normalizeMatrix(
                    arguments.aux_type, m, n, arguments.lde[index], strideE, batchCount);
            }
            cases.push_back(std::move(testCase));
        }
        return cases;
    }
} // namespace hipblaslt::client
