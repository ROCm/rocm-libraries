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
#include <type_traits>
#include <utility>

#include <gtest/gtest-spi.h>

namespace
{

    void expect_preswizzled_layout_error(const Arguments& arg, const char* expected)
    {
        try
        {
            testing_matmul_with_bias(arg,
                                     HIP_R_8F_E4M3,
                                     HIP_R_8F_E4M3,
                                     HIP_R_32F,
                                     HIP_R_32F,
                                     HIP_R_8F_E4M3,
                                     HIP_R_8F_E4M3,
                                     HIP_R_32F,
                                     HIP_R_32F);
            FAIL() << "Expected " << expected;
        }
        catch(const std::invalid_argument& error)
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
        arg.fast_check          = 0;
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
        arg.fast_check          = 0;
        // A has its supported orientation, so it cannot mask a missing B check.
        expect_preswizzled_layout_error(arg, "scaleB=1001 requires transB=N");
    }

    TEST(FastCheckMx_pre_checkin, unsupported_transpose_layouts_are_refused)
    {
        Arguments arg{};
        arg.init();
        arg.initialization = hipblaslt_initialization::integer_exact;
        arg.fast_check     = 1;
        arg.batch_count    = 1;
        arg.M[0] = arg.N[0] = 64;
        arg.K[0]            = 256;
        arg.scaleA = arg.scaleB = hipblaslt_scaling_format::Block_32_UE8M0;
        for(char a : {'N', 'T', 'C'})
            for(char b : {'N', 'T', 'C'})
            {
                arg.transA = a;
                arg.transB = b;
                auto why   = fast_check_unsupported_reason(arg,
                                                         HIPBLASLT_BATCH_MODE_STRIDED,
                                                         false,
                                                         HIP_R_8F_E4M3,
                                                         HIP_R_8F_E5M2,
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
                if(arg.placement[0])
                    name << "_PLACE_" << arg.placement;
                if(arg.integer_exact_pattern[0])
                    name << "_IE_" << arg.integer_exact_pattern;
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
