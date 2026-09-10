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
#include <hipblaslt/host_numerics/Types.hpp>
#include <limits>
#include <roc/host_numerics/epilogue.hpp>
#include <roc/host_numerics/reduction.hpp>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest-spi.h>

TEST(HostNumericsTypeBridge, ConvergesOnScalarType)
{
    using namespace hipblaslt::host_numerics;
    using roc::host_numerics::ScalarType;

    EXPECT_EQ(scalarType<float>(), scalarType(HIP_R_32F));
    EXPECT_EQ(scalarType(static_cast<hipDataType>(HIP_R_8F_E5M3_EXT)), ScalarType::E5M3);
    EXPECT_FALSE(tryScalarType(static_cast<hipDataType>(-1)));
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
        std::pair{hipblaslt_scaling_format::none,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F},
        std::pair{hipblaslt_scaling_format::Scalar,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F},
        std::pair{hipblaslt_scaling_format::Vector,
                  HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F},
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
    EXPECT_THROW(
        hipblaslt::client::matmulScaleMode(static_cast<hipblaslt_scaling_format>(-1)),
        std::invalid_argument);
}

TEST(MatmulOrchestration, MapsSwizzledMatrixLayouts)
{
    EXPECT_TRUE(hipblaslt::client::supportsMatmulSwizzle(HIP_R_16F));
    EXPECT_TRUE(hipblaslt::client::supportsMatmulSwizzle(HIP_R_16BF));
    EXPECT_FALSE(hipblaslt::client::supportsMatmulSwizzle(HIP_R_32F));

    EXPECT_EQ(hipblaslt::client::matmulOrderForDataType(HIP_R_16F),
              HIPBLASLT_ORDER_COL16_4R8);
    EXPECT_EQ(hipblaslt::client::matmulOrderForDataType(HIP_R_8F_E4M3_FNUZ),
              HIPBLASLT_ORDER_COL16_4R16);
    EXPECT_EQ(hipblaslt::client::matmulOrderForDataType(HIP_R_4F_E2M1),
              HIPBLASLT_ORDER_COL16_4R32);
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

    arguments                  = Arguments{};
    arguments.activation_type  = hipblaslt_activation_type::none;
    arguments.bias_vector      = false;
    arguments.use_e            = true;
    arguments.gradient         = false;
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
            source[row * leadingDimension + column]
                = static_cast<float>(1000 * row + column);

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
            const size_t rowTile     = row / tileRows;
            const size_t rowInTile   = row % tileRows;
            const size_t columnTile  = column / tileColumns;
            const size_t columnGroup = (column % tileColumns) / valuesPerGroup;
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
    using namespace roc::host_numerics;

    std::array<float, 4> input{-2, 1, 3, -4};
    std::array<float, 2> bias{1, 2};
    const Layout         matrixLayout(Shape{2, 2}, {1, 2});
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
        const auto coordinates
            = matrixLayout.shape().coordinates(index, IndexOrder::FirstDimensionFastest);
        EXPECT_EQ(output.loadAs<float>(coordinates), expectedOutput[index]);
        EXPECT_EQ(rawOutput.loadAs<float>(coordinates), expectedOutput[index]);
        EXPECT_EQ(auxiliary.loadAs<float>(coordinates), expectedAuxiliary[index]);
    }
    EXPECT_EQ(amax.loadAs<float>({0}), 5);
}

TEST(HostNumericsEpilogue, RoutesGradientAuxiliaryInput)
{
    using namespace roc::host_numerics;

    std::array<float, 4> gradient{10, 20, 30, 40};
    std::array<float, 4> activationInput{-1, 1, 2, -2};
    const Layout         layout(Shape{2, 2}, {1, 2});
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
        const auto coordinates
            = layout.shape().coordinates(index, IndexOrder::FirstDimensionFastest);
        EXPECT_EQ(output.loadAs<float>(coordinates), expected[index]);
    }
    EXPECT_EQ(activationTensor.loadAs<float>({0, 0}), -1);
    EXPECT_EQ(activationTensor.loadAs<float>({1, 1}), -2);
}

TEST(HostNumericsEpilogue, SaturatesInt8Output)
{
    using namespace roc::host_numerics;

    std::array<float, 4>  input{-200.0f, -128.5f, 126.5f, 300.0f};
    const Layout          layout(Shape{2, 2}, {1, 2});
    const Tensor          inputTensor = Tensor::copyNativeStorage<float>(layout, input);
    Tensor                output(ScalarType::Int8, layout);
    EpilogueOptions       options(ScalarType::Float32);
    options.outputConversion = OutputConversion::SaturatingInt8;
    referenceEpilogueInto(inputTensor, {.output = output}, options);

    const std::array<int8_t, 4> expected{-128, -128, 126, 127};
    for(size_t index = 0; index < expected.size(); ++index)
    {
        const auto coordinates
            = layout.shape().coordinates(index, IndexOrder::FirstDimensionFastest);
        EXPECT_EQ(output.loadAs<int8_t>(coordinates), expected[index]);
    }
}

TEST(HostNumericsEpilogue, UsesIdentityScaleDefaults)
{
    using namespace roc::host_numerics;

    std::array<float, 4> input{-2, 1, 3, -4};
    const Layout         layout(Shape{2, 2}, {1, 2});
    const Tensor         inputTensor = Tensor::copyNativeStorage<float>(layout, input);
    Tensor               output(ScalarType::Float32, layout);
    Tensor               auxiliary(ScalarType::Float32, layout);
    referenceEpilogueInto(inputTensor, {.output = output, .auxiliaryOutput = auxiliary});

    for(size_t index = 0; index < input.size(); ++index)
    {
        const auto coordinates
            = layout.shape().coordinates(index, IndexOrder::FirstDimensionFastest);
        EXPECT_EQ(output.loadAs<float>(coordinates), input[index]);
        EXPECT_EQ(auxiliary.loadAs<float>(coordinates), input[index]);
    }
}

TEST(HostNumericsReduction, SumsStridedTensor)
{
    using namespace roc::host_numerics;

    const std::array<float, 8> input{1, 2, -99, 3, 4, -99, 5, 6};
    const Tensor inputTensor = Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 3}), input);
    Tensor       output(ScalarType::Float32, Shape{2});
    referenceSumInto(inputTensor, output, {1}, ScalarType::Float32);

    EXPECT_EQ(output.loadAs<float>({0}), 9);
    EXPECT_EQ(output.loadAs<float>({1}), 12);
}

namespace
{

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
                // grouped gemm only supports ext
                if(arg.use_ext || arg.grouped_gemm > 0)
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
        SKIP_IF_KNOWN_BUG_FOR_PLATFORM();
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
        SKIP_IF_KNOWN_BUG_FOR_PLATFORM();
        // rocRoller has predicates that check the dimensions (M/N/K) must be
        // multiples of the work group sizes. This test set the K dimension
        // to not be a multiple, and thus we shall see failure.
        EXPECT_FATAL_FAILURE(rocroller_predicate_testing{}(GetParam()), "NO solution found!");
    }
    INSTANTIATE_TEST_CATEGORIES(rocroller_predicate_test);
#endif

} // namespace
