/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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
#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "testing_matmul.hpp"
#include "testing_matmul_batch_offset.hpp"
#include <cctype>
#include <cstring>
#include <hipblaslt/hostnumerics/Types.hpp>
#include <limits>
#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/reduction.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gtest/gtest-spi.h>
#include <vector>

TEST(HostNumericsTypeBridge, ConvergesOnScalarType)
{
    using namespace hipblaslt::hostnumerics;
    using roc::hostnumerics::ScalarType;

    EXPECT_EQ(scalarType<float>(), scalarType(HIP_R_32F));
    EXPECT_EQ(scalarType(static_cast<hipDataType>(HIP_R_8F_E5M3_EXT)), ScalarType::E5M3);
    EXPECT_FALSE(getScalarType(static_cast<hipDataType>(-1)));
}

TEST(HostNumericsTypeBridge, ValidatesComputeInputTypeBWhenAIsUnset)
{
    Arguments arguments{};
    arguments.init();
    arguments.compute_input_typeA = HIPBLASLT_DATATYPE_INVALID;
    arguments.compute_input_typeB = HIP_R_64F;

    EXPECT_THROW(hipblaslt::client::resolveMatmulDataTypes(arguments), std::invalid_argument);
}

TEST(MatmulOrchestration, MapsScaleModes)
{
    const std::array mappings{
        std::pair{hipblaslt_scaling_format::none, HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F},
        std::pair{hipblaslt_scaling_format::Scalar, HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F},
        std::pair{hipblaslt_scaling_format::Vector, HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F},
        std::pair{hipblaslt_scaling_format::Block_32_UE8M0,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0},
        std::pair{hipblaslt_scaling_format::Block_16_UE8M0,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE8M0_EXT},
        std::pair{hipblaslt_scaling_format::Block_32_UE4M3,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE4M3_EXT},
        std::pair{hipblaslt_scaling_format::Block_16_UE4M3,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        std::pair{hipblaslt_scaling_format::Block_32_UE5M3,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE5M3_EXT},
        std::pair{hipblaslt_scaling_format::Block_16_UE5M3,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE5M3_EXT},
        std::pair{hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT},
    };

    for(const auto& [format, expected] : mappings)
        EXPECT_EQ(hipblaslt::client::matmulScaleMode(format), expected);
    EXPECT_THROW(hipblaslt::client::matmulScaleMode(static_cast<hipblaslt_scaling_format>(-1)),
                 std::invalid_argument);
}

TEST(MatmulOrchestration, MapsSwizzledMatrixLayouts)
{
    EXPECT_TRUE(hipblaslt::client::supportsMatmulSwizzle(HIP_R_16F));
    EXPECT_TRUE(hipblaslt::client::supportsMatmulSwizzle(HIP_R_16BF));
    EXPECT_FALSE(hipblaslt::client::supportsMatmulSwizzle(HIP_R_32F));

    EXPECT_EQ(hipblaslt::client::matmulOrderForDataType(HIP_R_16F), HIPBLASLT_ORDER_COL16_4R8);
    EXPECT_EQ(hipblaslt::client::matmulOrderForDataType(HIP_R_8F_E4M3_FNUZ),
              HIPBLASLT_ORDER_COL16_4R16);
    EXPECT_EQ(hipblaslt::client::matmulOrderForDataType(HIP_R_4F_E2M1), HIPBLASLT_ORDER_COL16_4R32);
    EXPECT_THROW(hipblaslt::client::matmulOrderForDataType(HIP_R_32F), std::runtime_error);
}

TEST(MatmulOrchestration, MapsEpiloguePolicy)
{
    Arguments arguments;
    arguments.activation_type = hipblaslt_activation_type::none;
    arguments.bias_vector     = false;
    arguments.use_e           = false;
    arguments.gradient        = false;
    EXPECT_EQ(hipblaslt::client::matmulEpilogue(arguments), HIPBLASLT_EPILOGUE_DEFAULT);

    arguments.activation_type = hipblaslt_activation_type::relu;
    arguments.bias_vector     = true;
    EXPECT_EQ(hipblaslt::client::matmulEpilogue(arguments), HIPBLASLT_EPILOGUE_RELU_BIAS);

    arguments.use_e = true;
    EXPECT_EQ(hipblaslt::client::matmulEpilogue(arguments), HIPBLASLT_EPILOGUE_RELU_AUX_BIAS);

    arguments.activation_type = hipblaslt_activation_type::gelu;
    arguments.gradient        = true;
    EXPECT_EQ(hipblaslt::client::matmulEpilogue(arguments), HIPBLASLT_EPILOGUE_DGELU_BGRAD);

    arguments.activation_type = hipblaslt_activation_type::none;
    arguments.use_e           = false;
    arguments.bias_source     = hipblaslt_bias_source::a;
    EXPECT_EQ(hipblaslt::client::matmulEpilogue(arguments), HIPBLASLT_EPILOGUE_BGRADA);
    arguments.bias_source = hipblaslt_bias_source::b;
    EXPECT_EQ(hipblaslt::client::matmulEpilogue(arguments), HIPBLASLT_EPILOGUE_BGRADB);

    arguments                 = Arguments{};
    arguments.activation_type = hipblaslt_activation_type::none;
    arguments.bias_vector     = false;
    arguments.use_e           = true;
    arguments.gradient        = false;
    EXPECT_THROW(hipblaslt::client::matmulEpilogue(arguments), std::invalid_argument);

    arguments.use_e           = false;
    arguments.activation_type = static_cast<hipblaslt_activation_type>(-1);
    EXPECT_THROW(hipblaslt::client::matmulEpilogue(arguments), std::invalid_argument);
}

TEST(MatmulBatchOffsetPlan, ValidatesOffsetArithmetic)
{
    const auto negative = offsetMatrixPlan(4, -3);
    EXPECT_EQ(negative.padding, 3);
    EXPECT_EQ(negative.allocationElements, 7);
    EXPECT_EQ(negative.logicalStart(), 0);

    const auto positive = offsetMatrixPlan(4, 3);
    EXPECT_EQ(positive.padding, 0);
    EXPECT_EQ(positive.allocationElements, 7);
    EXPECT_EQ(positive.logicalStart(), 3);

    EXPECT_THROW(offsetMatrixPlan(1, std::numeric_limits<int64_t>::min()), std::overflow_error);
    EXPECT_THROW(offsetMatrixPlan(size_t(std::numeric_limits<ptrdiff_t>::max()), 1),
                 std::overflow_error);
}

TEST(MatmulBatchOffsetPlan, AcceptsZeroBatchAsEmptyWork)
{
    Arguments arguments{};
    arguments.init();
    arguments.a_type      = HIP_R_32F;
    arguments.b_type      = HIP_R_32F;
    arguments.c_type      = HIP_R_32F;
    arguments.d_type      = HIP_R_32F;
    arguments.transA      = 'N';
    arguments.transB      = 'N';
    arguments.batch_mode  = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
    arguments.batch_count = 0;

    EXPECT_NO_THROW(testing_matmul_batch_offset(arguments));
}

TEST(HostNumericsTensorManipulation, SwizzlePreservesPaddedMatrixEncoding)
{
    constexpr size_t rows             = 18;
    constexpr size_t columns          = 17;
    constexpr size_t leadingDimension = 20;
    constexpr size_t tileRows         = 16;
    constexpr size_t tileColumns      = 16;
    constexpr size_t paddedRows       = 32;
    constexpr size_t paddedColumns    = 32;
    constexpr size_t columnGroups     = 4;
    constexpr size_t valuesPerGroup   = 4;

    std::vector<float> source(rows * leadingDimension, -1.0f);
    for(size_t row = 0; row < rows; ++row)
        for(size_t column = 0; column < columns; ++column)
            source[row * leadingDimension + column] = static_cast<float>(1000 * row + column);

    std::vector<float> observed(paddedRows * paddedColumns, -1.0f);
    Arguments          arguments;
    arguments.compute_type = HIPBLAS_COMPUTE_32F;
    swizzle_tensor(observed.data(),
                   source.data(),
                   HIP_R_32F,
                   arguments,
                   1,
                   rows,
                   columns,
                   leadingDimension,
                   false);

    std::vector<float> expected(paddedRows * paddedColumns, 0.0f);
    const size_t       rowTileCount    = paddedRows / tileRows;
    const size_t       columnTileCount = paddedColumns / tileColumns;
    for(size_t row = 0; row < rows; ++row)
        for(size_t column = 0; column < columns; ++column)
        {
            const size_t rowTile      = row / tileRows;
            const size_t rowInTile    = row % tileRows;
            const size_t columnTile   = column / tileColumns;
            const size_t columnGroup  = (column % tileColumns) / valuesPerGroup;
            const size_t valueInGroup = column % valuesPerGroup;
            const size_t destination
                = (((rowTile * columnTileCount + columnTile) * columnGroups + columnGroup)
                       * tileRows
                   + rowInTile)
                      * valuesPerGroup
                  + valueInGroup;
            ASSERT_LT(rowTile, rowTileCount);
            expected[destination] = source[row * leadingDimension + column];
        }

    EXPECT_EQ(observed, expected);
}

TEST(HostNumericsEpilogue, SupportsTensorBackedProductComposition)
{
    using namespace roc::hostnumerics;

    std::array<float, 4> input{-2, 1, 3, -4};
    std::array<float, 2> bias{1, 2};
    const Layout         matrixLayout(Shape{2, 2}, {1, 2}, 0);
    const Tensor         inputTensor = Tensor::copyNativeStorage<float>(matrixLayout, input);
    Tensor               output(ScalarType::Float32, matrixLayout);
    Tensor               rawOutput(ScalarType::Float32, matrixLayout);
    Tensor               auxiliary(ScalarType::Float32, matrixLayout);
    Tensor               amax = Tensor::copyNativeValues<float>(Shape{1}, std::array<float, 1>{5});

    EpilogueOptions options(ScalarType::Float32);
    options.outputScale    = Tensor(2.0f);
    options.auxiliaryScale = Tensor(3.0f);
    options.bias           = Tensor::copyNativeValues<float>(Shape{2}, bias).expandDims(1);
    options.activation     = ReluActivation{};
    options.accumulateAmax = true;
    referenceEpilogueInto(
        inputTensor,
        {.output = output, .rawOutput = rawOutput, .auxiliaryOutput = auxiliary, .amax = amax},
        options);

    const std::array<float, 4> expectedOutput{0, 6, 8, 0};
    const std::array<float, 4> expectedAuxiliary{-3, 9, 12, -6};
    for(size_t index = 0; index < expectedOutput.size(); ++index)
    {
        const auto coordinates = matrixLayout.shape().coordinates(index, std::vector<size_t>{1, 0});
        EXPECT_EQ(output.loadAs<float>(coordinates), expectedOutput[index]);
        EXPECT_EQ(rawOutput.loadAs<float>(coordinates), expectedOutput[index]);
        EXPECT_EQ(auxiliary.loadAs<float>(coordinates), expectedAuxiliary[index]);
    }
    EXPECT_EQ(amax.loadAs<float>({0}), 5);
}

TEST(HostNumericsEpilogue, RoutesGradientAuxiliaryInput)
{
    using namespace roc::hostnumerics;

    std::array<float, 4> gradient{10, 20, 30, 40};
    std::array<float, 4> activationInput{-1, 1, 2, -2};
    const Layout         layout(Shape{2, 2}, {1, 2}, 0);
    const Tensor         gradientTensor = Tensor::copyNativeStorage<float>(layout, gradient);
    const Tensor activationTensor       = Tensor::copyNativeStorage<float>(layout, activationInput);
    Tensor       output(ScalarType::Float32, layout);

    EpilogueOptions options(ScalarType::Float32);
    options.auxiliaryInput        = activationTensor;
    options.activation            = ReluActivation{};
    options.activationApplication = ActivationApplication::Gradient;
    referenceEpilogueInto(gradientTensor, {.output = output}, options);

    const std::array<float, 4> expected{0, 20, 30, 0};
    for(size_t index = 0; index < expected.size(); ++index)
    {
        const auto coordinates = layout.shape().coordinates(index, std::vector<size_t>{1, 0});
        EXPECT_EQ(output.loadAs<float>(coordinates), expected[index]);
    }
    EXPECT_EQ(activationTensor.loadAs<float>({0, 0}), -1);
    EXPECT_EQ(activationTensor.loadAs<float>({1, 1}), -2);
}

TEST(HostNumericsEpilogue, SaturatesInt8Output)
{
    using namespace roc::hostnumerics;

    std::array<float, 4> input{-200.0f, -128.5f, 126.5f, 300.0f};
    const Layout         layout(Shape{2, 2}, {1, 2}, 0);
    const Tensor         inputTensor = Tensor::copyNativeStorage<float>(layout, input);
    Tensor               output(ScalarType::Int8, layout);
    EpilogueOptions      options(ScalarType::Float32);
    options.outputConversion = OutputConversion::SaturatingInt8;
    referenceEpilogueInto(inputTensor, {.output = output}, options);

    const std::array<int8_t, 4> expected{-128, -128, 126, 127};
    for(size_t index = 0; index < expected.size(); ++index)
    {
        const auto coordinates = layout.shape().coordinates(index, std::vector<size_t>{1, 0});
        EXPECT_EQ(output.loadAs<int8_t>(coordinates), expected[index]);
    }
}

TEST(HostNumericsEpilogue, UsesIdentityScaleDefaults)
{
    using namespace roc::hostnumerics;

    std::array<float, 4> input{-2, 1, 3, -4};
    const Layout         layout(Shape{2, 2}, {1, 2}, 0);
    const Tensor         inputTensor = Tensor::copyNativeStorage<float>(layout, input);
    Tensor               output(ScalarType::Float32, layout);
    Tensor               auxiliary(ScalarType::Float32, layout);
    referenceEpilogueInto(
        inputTensor,
        {.output = output, .auxiliaryOutput = auxiliary},
        roc::hostnumerics::EpilogueOptions{roc::hostnumerics::ScalarType::Float32});

    for(size_t index = 0; index < input.size(); ++index)
    {
        const auto coordinates = layout.shape().coordinates(index, std::vector<size_t>{1, 0});
        EXPECT_EQ(output.loadAs<float>(coordinates), input[index]);
        EXPECT_EQ(auxiliary.loadAs<float>(coordinates), input[index]);
    }
}

TEST(HostNumericsReduction, SumsStridedTensor)
{
    using namespace roc::hostnumerics;

    const std::array<float, 8> input{1, 2, -99, 3, 4, -99, 5, 6};
    const Tensor               inputTensor
        = Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 3}, 0), input);
    Tensor output(ScalarType::Float32, Shape{2});
    referenceSumInto(inputTensor, output, {1}, ScalarType::Float32);

    EXPECT_EQ(output.loadAs<float>({0}), 9);
    EXPECT_EQ(output.loadAs<float>({1}), 12);
}

namespace
{

    void expect_preswizzled_layout_error(const Arguments& arg, const char* expected)
    {
        try
        {
            testing_matmul(arg);
            FAIL() << "Expected " << expected;
        }
        catch(const std::runtime_error& error)
        {
            EXPECT_STREQ(error.what(), expected);
        }
    }

    // Runs the hipblaslt-bench -v checks of D (norm and allclose) through check() on host
    // buffers of T (f32 or f16), M x N per batch, and counts the checks that failed.
    template <typename T = float>
    struct BenchVerifyD
    {
        static constexpr hipDataType type = std::is_same_v<T, float> ? HIP_R_32F : HIP_R_16F;

        int64_t                    m, n, batches;
        hipblasLtBatchMode_t       mode;
        std::vector<HipHostBuffer> gold, result;
        double                     norm_error = 0, atol = 1, rtol = 1;

        BenchVerifyD(int64_t m, int64_t n, int64_t batches, hipblasLtBatchMode_t mode)
            : m(m)
            , n(n)
            , batches(batches)
            , mode(mode)
        {
            const bool   strided = mode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
            const size_t count   = strided ? 1 : batches;
            for(size_t i = 0; i < count; i++)
            {
                gold.emplace_back(type, m * n * (strided ? batches : 1));
                result.emplace_back(type, m * n * (strided ? batches : 1));
            }
            for(int64_t b = 0; b < batches; b++)
                for(int64_t i = 0; i < m * n; i++)
                    at(gold, b, i) = at(result, b, i) = T(1 + (i + b) % 3);
        }

        T& at(std::vector<HipHostBuffer>& buffers, int64_t batch, int64_t i)
        {
            if(mode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                return static_cast<T*>(buffers[batch].buf())[i];
            return static_cast<T*>(buffers[0].buf())[batch * m * n + i];
        }

        // output_type overrides the type check() is told D has, for a type it cannot compare.
        int64_t
            run(hipDataType input_type = type, hipDataType output_type = type, bool allclose = true)
        {
            Arguments arg{};
            arg.init();
            arg.unit_check        = 0;
            arg.norm_check        = 1;
            arg.allclose_check    = allclose;
            arg.norm_check_assert = false;
            arg.compute_type      = HIPBLAS_COMPUTE_32F;
            arg.a_type = arg.b_type = input_type;

            const uint32_t               gemm_count = 1;
            std::vector<HipDeviceBuffer> no_device;
            std::vector<HipHostBuffer>   no_host_gold, no_host;
            std::vector<double>          tol{0};
            double                       max_ulp = 0, avg_ulp = 0;
            const int64_t                before = hipblaslt_bench_options::verification_failures();
            check(nullptr,
                  arg,
                  gemm_count,
                  {m},
                  {n},
                  {m},
                  {m},
                  {m * n},
                  {m * n},
                  {int(batches)},
                  {0},
                  gold,
                  result,
                  no_device,
                  no_host_gold,
                  no_host,
                  no_device,
                  no_host_gold,
                  no_host,
                  no_device,
                  no_host_gold,
                  no_host,
                  no_device,
                  tol,
                  norm_error,
                  atol,
                  rtol,
                  max_ulp,
                  avg_ulp,
                  output_type,
                  HIP_R_32F,
                  HIP_R_32F,
                  HIP_R_32F,
                  mode);
            return hipblaslt_bench_options::verification_failures() - before;
        }
    };

    // ROCM-32137: one wrong element in the last batch fails both checks.
    TEST(BenchVerify, smoke_WrongLaterBatchCountsBothChecks)
    {
        BenchVerifyD d(8, 4, 3, HIPBLASLT_BATCH_MODE_STRIDED);
        d.at(d.result, 2, 5) = 7;
        EXPECT_EQ(d.run(), 2);
        EXPECT_EQ(d.atol, 1);
        EXPECT_GT(d.norm_error, 0.1);
    }

    TEST(BenchVerify, smoke_MatchingResultCountsNothing)
    {
        BenchVerifyD d(8, 4, 3, HIPBLASLT_BATCH_MODE_POINTER_ARRAY);
        EXPECT_EQ(d.run(), 0);
        EXPECT_EQ(d.norm_error, 0);
        EXPECT_EQ(d.atol, 1e-6);
    }

    // Every batch is off by about 4e-6 relative, under the f32 limit of 1e-5. The reported
    // norm_error sums the 8 batches past the limit, but each batch passes on its own.
    TEST(BenchVerify, smoke_AccurateBatchesDoNotAddUpToFailure)
    {
        BenchVerifyD d(8, 4, 8, HIPBLASLT_BATCH_MODE_STRIDED);
        for(int64_t b = 0; b < 8; b++)
            for(int64_t i = 0; i < 32; i++)
                d.at(d.result, b, i) *= 1.000004f;
        EXPECT_EQ(d.run(), 0);
        EXPECT_GT(d.norm_error, 1e-5);
    }

    // Pointer-array mode used to add the last batch's norm_error twice.
    TEST(BenchVerify, smoke_PointerArrayNormCountsEachBatchOnce)
    {
        BenchVerifyD d(8, 4, 3, HIPBLASLT_BATCH_MODE_POINTER_ARRAY);
        d.at(d.result, 2, 5) += 0.5f;
        const double last
            = norm_check_general('F', 8, 4, 8, 0, d.gold[2].buf(), d.result[2].buf(), 1, HIP_R_32F);
        d.run();
        EXPECT_GT(last, 0);
        EXPECT_DOUBLE_EQ(d.norm_error, last);
    }

    TEST(BenchVerify, smoke_MixedPrecisionNormUsesGemmTolerance)
    {
        BenchVerifyD d(8, 4, 1, HIPBLASLT_BATCH_MODE_STRIDED);
        for(int64_t i = 0; i < 32; i++)
            d.at(d.result, 0, i) *= 1.001f;
        EXPECT_EQ(d.run(HIP_R_16F), 0);
        EXPECT_GT(d.norm_error, norm_tolerance(HIP_R_32F));
        EXPECT_DOUBLE_EQ(norm_tolerance(HIP_R_32F, HIPBLAS_COMPUTE_32F, HIP_R_16F, HIP_R_16F),
                         0.01);
        EXPECT_DOUBLE_EQ(norm_tolerance(HIP_R_16F, HIPBLAS_COMPUTE_32F, HIP_R_16BF, HIP_R_16BF),
                         0.1);
        EXPECT_DOUBLE_EQ(
            norm_tolerance(HIP_R_32F, HIPBLAS_COMPUTE_32F_FAST_16BF, HIP_R_32F, HIP_R_32F),
            0.5);
    }

    // f16 output uses the f16 limit of 0.01: a 0.4% error passes, which the f32 limit would
    // fail, and a 2% error fails, which the bf16 limit would pass. Allclose passes both.
    TEST(BenchVerify, smoke_HalfOutputNormUsesHalfTolerance)
    {
        BenchVerifyD<hipblasLtHalf> within_limit(8, 4, 2, HIPBLASLT_BATCH_MODE_STRIDED);
        BenchVerifyD<hipblasLtHalf> beyond_limit(8, 4, 2, HIPBLASLT_BATCH_MODE_STRIDED);
        for(int64_t b = 0; b < 2; b++)
            for(int64_t i = 0; i < 32; i++)
            {
                within_limit.at(within_limit.result, b, i) *= hipblasLtHalf(1.004f);
                beyond_limit.at(beyond_limit.result, b, i) *= hipblasLtHalf(1.02f);
            }
        EXPECT_EQ(within_limit.run(), 0);
        EXPECT_GT(within_limit.norm_error, norm_tolerance(HIP_R_32F));
        EXPECT_EQ(beyond_limit.run(), 1);
    }

    // A D type that norm_check_general cannot compare is counted as a failure, so exit 0
    // under --verify_strict always means D was checked.
    TEST(BenchVerify, smoke_UncomparableOutputTypeCountsAsFailure)
    {
        BenchVerifyD d(8, 4, 3, HIPBLASLT_BATCH_MODE_STRIDED);
        EXPECT_GT(norm_tolerance(HIP_R_4F_E2M1), 0);
        EXPECT_FALSE(norm_check_general_supports(HIP_R_4F_E2M1));
        struct RestoreErrorStream
        {
            hipblaslt_internal_ostream saved = std::move(hipblaslt_cerr);
            ~RestoreErrorStream()
            {
                hipblaslt_cerr = std::move(saved);
            }
        } restore_error_stream;
        hipblaslt_cerr        = hipblaslt_internal_ostream{};
        const auto failures   = d.run(HIP_R_4F_E2M1, HIP_R_4F_E2M1, false);
        const auto diagnostic = hipblaslt_cerr.str();
        EXPECT_EQ(failures, 1);
        EXPECT_EQ(diagnostic.find("Error type in norm_check_general"), std::string::npos);
        EXPECT_NE(diagnostic.find("so D was not verified"), std::string::npos);
    }

    TEST(BenchVerify, smoke_UnknownOutputTypeCountsAsFailure)
    {
        BenchVerifyD d(8, 4, 3, HIPBLASLT_BATCH_MODE_STRIDED);
        EXPECT_EQ(d.run(HIP_R_32F, static_cast<hipDataType>(-1), false), 1);
    }

    // Runs a 64 x 64 x 64 f32 GEMM through the hipblaslt-bench timing path with
    // --post_bench_verify_iters 3 and C stored in D, and counts the checks that failed.
    int64_t bench_relaunch_failures(double beta)
    {
        Arguments arg{};
        arg.init();
        snprintf(arg.function, sizeof(arg.function), "%s", "matmul");
        arg.a_type = arg.b_type = arg.c_type = arg.d_type = arg.aux_type = HIP_R_32F;
        arg.compute_type                                                 = HIPBLAS_COMPUTE_32F;
        arg.transA = arg.transB = 'N';
        arg.M[0] = arg.N[0] = arg.K[0] = 64;
        arg.lda[0] = arg.ldb[0] = arg.ldc[0] = arg.ldd[0] = arg.lde[0] = 64;
        arg.stride_a[0] = arg.stride_b[0] = arg.stride_c[0] = arg.stride_d[0] = arg.stride_e[0]
            = 64 * 64;
        arg.alpha             = 1;
        arg.beta              = beta;
        arg.c_equal_d         = true;
        arg.timing            = 1;
        arg.iters             = 3;
        arg.cold_iters        = 1;
        arg.unit_check        = 0;
        arg.norm_check        = 1;
        arg.allclose_check    = 1;
        arg.norm_check_assert = false;
        arg.fast_check        = 0;

        const int32_t saved = hipblaslt_bench_options::post_bench_verify_iters();
        hipblaslt_bench_options::post_bench_verify_iters() = 3;
        const int64_t before = hipblaslt_bench_options::verification_failures();
        testing_matmul(arg);
        hipblaslt_bench_options::post_bench_verify_iters() = saved;
        return hipblaslt_bench_options::verification_failures() - before;
    }

    // With beta 1, every launch adds A * B to the D the previous launch left, so each relaunch
    // gives a different D from the first launch. The bench refuses this setup from the command
    // line; here it stands in for a solution that writes a wrong result on some launches.
    TEST(BenchVerify, smoke_RelaunchThatChangesDIsCaught)
    {
        EXPECT_EQ(bench_relaunch_failures(1), 1);
    }

    TEST(BenchVerify, smoke_RepeatableRelaunchesPass)
    {
        EXPECT_EQ(bench_relaunch_failures(0), 0);
    }

    TEST(PreswizzledScaleLayout, smoke_rejects_nontransposed_A)
    {
        Arguments arg{};
        arg.init();
        arg.scaleA = arg.scaleB = hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT;
        arg.transA              = 'N';
        arg.transB              = 'N';
        arg.M[0]                = 288;
        arg.N[0]                = 32;
        arg.K[0]                = 32;
        arg.lda[0]              = 288;
        arg.ldb[0]              = 32;
        arg.ldc[0] = arg.ldd[0] = 288;
        arg.freivalds           = 0;
        // This NN case used to allocate 512 scale bytes for A and write 2304.
        // Call the shared client entry so removing or moving its guard is observable.
        expect_preswizzled_layout_error(arg, "scaleA=1001 requires transA=T");
    }

    TEST(PreswizzledScaleLayout, smoke_rejects_transposed_B)
    {
        Arguments arg{};
        arg.init();
        arg.scaleA = arg.scaleB = hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT;
        arg.transA              = 'T';
        arg.transB              = 'T';
        arg.M[0]                = 32;
        arg.N[0]                = 288;
        arg.K[0]                = 32;
        arg.lda[0]              = 32;
        arg.ldb[0]              = 288;
        arg.ldc[0] = arg.ldd[0] = 32;
        arg.freivalds           = 0;
        // A has its supported orientation, so it cannot mask a missing B check.
        expect_preswizzled_layout_error(arg, "scaleB=1001 requires transB=N");
    }

    TEST(FreivaldsMx_pre_checkin, unsupported_transpose_layouts_are_refused)
    {
        Arguments arg{};
        arg.init();
        arg.initialization = hipblaslt_initialization::integer_exact;
        arg.freivalds      = 1;
        arg.batch_count    = 1;
        arg.M[0] = arg.N[0] = 64;
        arg.K[0]            = 256;
        arg.scaleA = arg.scaleB = hipblaslt_scaling_format::Block_32_UE8M0;
        for(char a : {'N', 'T', 'C'})
            for(char b : {'N', 'T', 'C'})
            {
                arg.transA = a;
                arg.transB = b;
                auto why   = freivalds_unsupported_reason(arg,
                                                        HIPBLASLT_BATCH_MODE_STRIDED,
                                                        false,
                                                        HIP_R_8F_E4M3,
                                                        HIP_R_8F_E5M2,
                                                        HIP_R_32F,
                                                        HIP_R_32F,
                                                        HIP_R_32F);
                if(a == 'T' && b == 'N')
                    EXPECT_TRUE(why.empty()) << why;
                else
                    EXPECT_NE(why.find("transA=T and transB=N"), std::string::npos)
                        << a << b << ": " << why;
            }
    }

    // ----------------------------------------------------------------------------
    // matmul
    // ----------------------------------------------------------------------------

    struct matmul_testing : hipblaslt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "matmul"))
                testing_matmul(arg);
            else if(!strcmp(arg.function, "matmul_bad_arg"))
                testing_matmul_bad_arg(arg);
            else if(!strcmp(arg.function, "matmul_batch_offset"))
                testing_matmul_batch_offset(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct matmul_test : RocBlasLt_Test<matmul_test, matmul_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return type_filter_functor{}(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "matmul") || !strcmp(arg.function, "matmul_bad_arg")
                   || !strcmp(arg.function, "matmul_batch_offset");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBlasLt_TestName<matmul_test> name(arg.name);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "bad_arg";
            }
            else
            {
                name << hip_datatype_to_string(arg.a_type) << hip_datatype_to_string(arg.b_type)
                     << hip_datatype_to_string(arg.c_type) << hip_datatype_to_string(arg.d_type)
                     << hipblas_computetype_to_string(arg.compute_type);

                if(arg.activation_type != hipblaslt_activation_type::none)
                {
                    name << '_' << hipblaslt_activation_type_to_string(arg.activation_type);
                }

                if(arg.bias_vector)
                {
                    name << "_BIAS" << hipblaslt_bias_source_to_string(arg.bias_source);
                    name << "_" << hip_datatype_to_string(arg.bias_type);
                }

                if(arg.gradient)
                {
                    if(arg.use_e)
                    {
                        name << "_GRAD";
                    }
                }
                else
                {
                    if(arg.use_e)
                    {
                        name << "_AUX";
                        if(arg.aux_type != HIPBLASLT_DATATYPE_INVALID)
                            name << "_" << hip_datatype_to_string(arg.aux_type);
                    }
                }

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M[0] << '_' << arg.N[0] << '_' << arg.K[0] << '_' << arg.alpha
                     << '_' << arg.lda[0] << '_' << arg.ldb[0] << '_' << arg.beta << '_'
                     << arg.ldc[0] << '_' << arg.ldd[0];

                if(arg.use_e)
                {
                    name << '_' << arg.lde[0];
                }

                name << '_' << arg.batch_count;

                if(arg.scaleA == hipblaslt_scaling_format::Scalar)
                    name << "_SA";
                else if(arg.scaleA == hipblaslt_scaling_format::Vector)
                    name << "_SAV";
                else if(arg.scaleA == hipblaslt_scaling_format::Block_32_UE8M0)
                    name << "_SAMX_32_UE8M0";
                else if(arg.scaleA == hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT)
                    name << "_SAMX_32_UE8M0_32_8";
                else if(isW4A16Scaling(arg.scaleA))
                    name << "_SAG" << w4a16GroupSize(arg.scaleA)
                         << (isW4A16ZeroPoint(arg.scaleA) ? "_ZP" : "")
                         << (arg.int4_encoding ? "_U4" : "_S4");

                if(arg.scaleB == hipblaslt_scaling_format::Scalar)
                    name << "_SB";
                else if(arg.scaleB == hipblaslt_scaling_format::Vector)
                    name << "_SBV";
                else if(arg.scaleB == hipblaslt_scaling_format::Block_32_UE8M0)
                    name << "_SBMX_32_UE8M0";
                else if(arg.scaleB == hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT)
                    name << "_SBMX_32_UE8M0_32_8";

                if(arg.scaleC)
                    name << "_SC";

                if(arg.scaleD)
                    name << "_SD";

                if(arg.scaleE)
                    name << "_SAux";

                if(arg.scaleAlpha_vector)
                    name << "_SAV";

                if(arg.amaxScaleA)
                    name << "_ASA";

                if(arg.amaxScaleB)
                    name << "_ASB";

                if(arg.amaxD)
                    name << "_AMaxD";

                if(arg.grouped_gemm > 0)
                    name << "_GG" << arg.grouped_gemm;

                if(arg.c_equal_d)
                    name << "_C_EQUAL_D";
                if(arg.bounds_check)
                    name << "_BOUNDS";
                if(arg.placement[0])
                    name << "_PLACE_" << arg.placement;
                if(arg.integer_exact_pattern[0])
                    name << "_IE_" << arg.integer_exact_pattern;
                if(arg.api_method == kApiMethodGroupedC)
                    name << "_APIGroupedC";
                else if(arg.use_ext || arg.grouped_gemm > 0)
                    name << "_APIExt";
                if(arg.use_ext_setproblem)
                    name << "_APIExtSet";
                if(arg.algo_method == 2)
                    name << "_APIAlgoIndex";
                else if(arg.algo_method == 1)
                    name << "_APIFindAllAlgo";
                if(arg.use_user_args)
                    name << "_UserArgs";
                if(arg.gsu_vector[0])
                    name << "_GSU" << (int)arg.gsu_vector[0];
                if(arg.wgm_vector[0])
                    name << "_WGM" << (int)arg.wgm_vector[0];
            }

            return std::move(name);
        }
    };

    TEST_P(matmul_test, matmul)
    {
        SKIP_UNLESS_SELECTED_BY_FILTER();
        RUN_TEST_ON_THREADS_STREAMS(matmul_testing{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(matmul_test);

#ifdef HIPBLASLT_USE_ROCROLLER
    // ----------------------------------------------------------------------------
    // rocRoller
    // ----------------------------------------------------------------------------

    struct rocroller_predicate_testing : hipblaslt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            testing_matmul(arg);
        }
    };

    struct rocroller_predicate_test
        : RocBlasLt_Test<rocroller_predicate_test, rocroller_predicate_testing>
    {
        static bool type_filter(const Arguments& arg)
        {
            return type_filter_functor{}(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "rocroller_predicate");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return matmul_test::name_suffix(arg);
        }
    };

    TEST_P(rocroller_predicate_test, unrollXYK)
    {
        // rocRoller has predicates that check the dimensions (M/N/K) must be
        // multiples of the work group sizes. This test set the K dimension
        // to not be a multiple, and thus we shall see failure.
        EXPECT_FATAL_FAILURE(rocroller_predicate_testing{}(GetParam()), "NO solution found!");
    }
    INSTANTIATE_TEST_CATEGORIES(rocroller_predicate_test);
#endif

} // namespace
