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
#include <hipblaslt/host_validation/Types.hpp>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest-spi.h>

TEST(HostValidationTypeBridge, ConvergesOnScalarType)
{
    using namespace hipblaslt::host_validation;
    using roc::host_validation::ScalarType;

    EXPECT_EQ(scalarType<float>(), scalarType(HIP_R_32F));
    EXPECT_EQ(scalarType(static_cast<hipDataType>(HIP_R_8F_E5M3_EXT)), ScalarType::E5M3);
    EXPECT_FALSE(tryScalarType(static_cast<hipDataType>(-1)));
}

TEST(HostValidationDataInitializationBridge, RuntimeRangeUsesTensorLayoutOffset)
{
    std::array<float, 6> values{1, 2, 3, 4, 5, 6};

    hipblaslt_init_zero(static_cast<void*>(values.data()), 2, 5, HIP_R_32F);

    EXPECT_EQ(values, (std::array<float, 6>{1, 2, 0, 0, 0, 6}));
}

TEST(HostValidationTensorManipulation, SwizzlePreservesPaddedMatrixEncoding)
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

TEST(HostValidationEpilogueBridge, DelegatesToProductIndependentComponent)
{
    std::array<float, 4> input{-2, 1, 3, -4};
    std::array<float, 4> output{};
    std::array<float, 4> rawOutput{};
    std::array<float, 4> auxiliary{};
    std::array<float, 2> bias{1, 2};
    float                amax   = 5;
    float                scaleD = 2;
    float                scaleE = 3;

    hipblaslt::host_validation::EpilogueArguments arguments;
    arguments.rows             = 2;
    arguments.columns          = 2;
    arguments.leadingDimension = 2;
    arguments.input            = input.data();
    arguments.output           = output.data();
    arguments.rawOutput        = rawOutput.data();
    arguments.amax             = &amax;
    arguments.auxiliary        = auxiliary.data();
    arguments.auxiliaryType    = HIP_R_32F;
    arguments.outputScale      = &scaleD;
    arguments.auxiliaryScale   = &scaleE;
    arguments.bias             = bias.data();
    arguments.biasType         = HIP_R_32F;
    arguments.activation       = roc::host_validation::Activation::Relu;
    arguments.outputType       = HIP_R_32F;
    arguments.computeType      = HIP_R_32F;
    hipblaslt::host_validation::referenceEpilogue(arguments);

    EXPECT_EQ(output, (std::array<float, 4>{0, 6, 8, 0}));
    EXPECT_EQ(rawOutput, output);
    EXPECT_EQ(auxiliary, (std::array<float, 4>{-3, 9, 12, -6}));
    EXPECT_EQ(amax, 5);
}

TEST(HostValidationEpilogueBridge, RoutesGradientAuxiliaryInput)
{
    std::array<float, 4> gradient{10, 20, 30, 40};
    std::array<float, 4> activationInput{-1, 1, 2, -2};
    std::array<float, 4> output{};
    float                one = 1;

    hipblaslt::host_validation::EpilogueArguments arguments;
    arguments.rows                  = 2;
    arguments.columns               = 2;
    arguments.leadingDimension      = 2;
    arguments.input                 = gradient.data();
    arguments.output                = output.data();
    arguments.auxiliary             = activationInput.data();
    arguments.auxiliaryType         = HIP_R_32F;
    arguments.outputScale           = &one;
    arguments.auxiliaryScale        = &one;
    arguments.activation            = roc::host_validation::Activation::Relu;
    arguments.activationApplication = roc::host_validation::ActivationApplication::Gradient;
    arguments.outputType            = HIP_R_32F;
    arguments.computeType           = HIP_R_32F;
    hipblaslt::host_validation::referenceEpilogue(arguments);

    EXPECT_EQ(output, (std::array<float, 4>{0, 20, 30, 0}));
    EXPECT_EQ(activationInput, (std::array<float, 4>{-1, 1, 2, -2}));
}

TEST(HostValidationEpilogueBridge, SaturatesInt8Output)
{
    std::array<float, 4>  input{-200.0f, -128.5f, 126.5f, 300.0f};
    std::array<int8_t, 4> output{};
    float                 one = 1;

    hipblaslt::host_validation::EpilogueArguments arguments;
    arguments.rows             = 2;
    arguments.columns          = 2;
    arguments.leadingDimension = 2;
    arguments.input            = input.data();
    arguments.output           = output.data();
    arguments.outputScale      = &one;
    arguments.auxiliaryScale   = &one;
    arguments.outputType       = HIP_R_8I;
    arguments.computeType      = HIP_R_32F;
    hipblaslt::host_validation::referenceEpilogue(arguments);

    EXPECT_EQ(output, (std::array<int8_t, 4>{-128, -128, 126, 127}));
}

TEST(HostValidationEpilogueBridge, UsesIdentityForNullScaleDefaults)
{
    std::array<float, 4> input{-2, 1, 3, -4};
    std::array<float, 4> output{};
    std::array<float, 4> auxiliary{};

    hipblaslt::host_validation::EpilogueArguments arguments;
    arguments.rows             = 2;
    arguments.columns          = 2;
    arguments.leadingDimension = 2;
    arguments.input            = input.data();
    arguments.output           = output.data();
    arguments.auxiliary        = auxiliary.data();
    arguments.outputType       = HIP_R_32F;
    arguments.computeType      = HIP_R_32F;
    hipblaslt::host_validation::referenceEpilogue(arguments);

    EXPECT_EQ(output, input);
    EXPECT_EQ(auxiliary, input);
}

TEST(HostValidationEpilogueBridge, RejectsOverflowingLeadingDimensionLayout)
{
    float input  = 0;
    float output = 0;

    hipblaslt::host_validation::EpilogueArguments arguments;
    arguments.rows             = 1;
    arguments.columns          = 3;
    arguments.leadingDimension = std::numeric_limits<decltype(arguments.leadingDimension)>::max();
    arguments.input            = &input;
    arguments.output           = &output;
    arguments.outputType       = HIP_R_32F;
    arguments.computeType      = HIP_R_32F;

    EXPECT_THROW(hipblaslt::host_validation::referenceEpilogue(arguments), std::overflow_error);
}

TEST(HostValidationReductionBridge, DelegatesStridedBiasSum)
{
    const std::array<float, 8> input{1, 2, -99, 3, 4, -99, 5, 6};
    std::array<float, 2>       output{};

    hipblaslt::host_validation::ReductionArguments arguments;
    arguments.rows            = 2;
    arguments.columns         = 3;
    arguments.rowStride       = 1;
    arguments.columnStride    = 3;
    arguments.input           = input.data();
    arguments.inputType       = HIP_R_32F;
    arguments.output          = output.data();
    arguments.outputType      = HIP_R_32F;
    arguments.accumulatorType = HIP_R_32F;
    hipblaslt::host_validation::referenceSum(arguments);

    EXPECT_EQ(output, (std::array<float, 2>{9, 12}));
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
