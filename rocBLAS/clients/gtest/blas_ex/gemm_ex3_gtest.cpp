/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#define ROCBLAS_BETA_FEATURES_API

#include "blas_ex/common_gemm_ex3.hpp"
#include "client_utility.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // Types of GEMM tests
    enum gemm_test_type
    {
        GEMM_EX3,
        GEMM_BATCHED_EX3,
        GEMM_STRIDED_BATCHED_EX3,
    };

    // ----------------------------------------------------------------------------
    // GEMM testing template
    // ----------------------------------------------------------------------------
    // The first template parameter is a class template which determines which
    // combination of types applies to this test, and for those that do, instantiates
    // the test code based on the function named in the test Arguments. The second
    // template parameter is an enum which allows the different flavors of GEMM to
    // be differentiated.
    //
    // The RocBLAS_Test base class takes this class (CRTP) and the first template
    // parameter as arguments, and provides common types such as type_filter_functor,
    // and derives from the Google Test parameterized base classes.
    //
    // This class defines functions for filtering the types and function names which
    // apply to this test, and for generating the suffix of the Google Test name
    // corresponding to each instance of this test.
    template <template <typename...> class FILTER, gemm_test_type GEMM_TYPE>
    struct gemm_test_template : RocBLAS_Test<gemm_test_template<FILTER, GEMM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_gemm_dispatch<gemm_test_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEMM_TYPE)
            {
#if(BUILD_WITH_TENSILE)

            case GEMM_EX3:
                return !strcmp(arg.function, "gemm_ex3")
                       || !strcmp(arg.function, "gemm_ex3_bad_arg");

            case GEMM_BATCHED_EX3:
                return !strcmp(arg.function, "gemm_batched_ex3")
                       || !strcmp(arg.function, "gemm_batched_ex3_bad_arg");

            case GEMM_STRIDED_BATCHED_EX3:
                return !strcmp(arg.function, "gemm_strided_batched_ex3")
                       || !strcmp(arg.function, "gemm_strided_batched_ex3_bad_arg");
#endif
            }

            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gemm_test_template> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                constexpr bool isBatched
                    = (GEMM_TYPE == GEMM_STRIDED_BATCHED_EX3 || GEMM_TYPE == GEMM_BATCHED_EX3);

                name << rocblas_datatype2string(arg.b_type) << rocblas_datatype2string(arg.c_type)
                     << rocblas_datatype2string(arg.d_type)
                     << rocblas_computetype2string(arg.composite_compute_type);

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                     << arg.lda;

                if(GEMM_TYPE == GEMM_STRIDED_BATCHED_EX3)
                    name << '_' << arg.stride_a;

                name << '_' << arg.ldb;

                if(GEMM_TYPE == GEMM_STRIDED_BATCHED_EX3)
                    name << '_' << arg.stride_b;

                name << '_' << arg.beta << '_' << arg.ldc;

                if(GEMM_TYPE == GEMM_STRIDED_BATCHED_EX3)
                    name << '_' << arg.stride_c;

                name << '_' << arg.ldd;

                if(GEMM_TYPE == GEMM_STRIDED_BATCHED_EX3)
                    name << '_' << arg.stride_d;

                if(isBatched)
                    name << '_' << arg.batch_count;
            }

            return std::move(name);
        }
    };

#if(BUILD_WITH_TENSILE)
    // ----------------------------------------------------------------------------
    // gemm_ex3
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename TiA,
              typename TiB = TiA,
              typename To  = TiA,
              typename Tc  = To,
              typename     = void>
    struct gemm_ex3_testing : rocblas_test_invalid
    {
    };

    // When Ti != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename TiA, typename TiB, typename To, typename Tc>
    struct gemm_ex3_testing<
        TiA,
        TiB,
        To,
        Tc,
        std::enable_if_t<(!std::is_same<TiA, void>{} && !std::is_same<TiB, void>{})
                         && ((std::is_same<TiA, rocblas_f8>{} || std::is_same<TiA, rocblas_bf8>{}
                              || std::is_same<TiA, rocblas_half>{} || std::is_same<TiA, float>{}))
                         && (std::is_same<TiB, rocblas_f8>{} || std::is_same<TiB, rocblas_bf8>{}
                             || std::is_same<TiB, rocblas_half>{} || std::is_same<TiB, float>{})>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemm_ex3"))
            {
                testing_gemm_ex3<TiA, TiB, To, Tc>(arg);
            }
            else if(!strcmp(arg.function, "gemm_ex3_bad_arg"))
            {
                testing_gemm_ex3_bad_arg<TiA, TiB, To, Tc>(arg);
            }
            else if(!strcmp(arg.function, "gemm_batched_ex3"))
            {
                testing_gemm_batched_ex3<TiA, TiB, To, Tc>(arg);
            }
            else if(!strcmp(arg.function, "gemm_batched_ex3_bad_arg"))
            {
                testing_gemm_batched_ex3_bad_arg<TiA, TiB, To, Tc>(arg);
            }
            else if(!strcmp(arg.function, "gemm_strided_batched_ex3"))
            {
                testing_gemm_strided_batched_ex3<TiA, TiB, To, Tc>(arg);
            }
            else if(!strcmp(arg.function, "gemm_strided_batched_ex3_bad_arg"))
            {
                testing_gemm_strided_batched_ex3_bad_arg<TiA, TiB, To, Tc>(arg);
            }
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemm_ex3 = gemm_test_template<gemm_ex3_testing, GEMM_EX3>;
    TEST_P(gemm_ex3, blas3_tensile)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocblas_gemm_dispatch<gemm_ex3_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_ex3);

    using gemm_batched_ex3 = gemm_test_template<gemm_ex3_testing, GEMM_BATCHED_EX3>;
    TEST_P(gemm_batched_ex3, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemm_dispatch<gemm_ex3_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_batched_ex3);

    using gemm_strided_batched_ex3 = gemm_test_template<gemm_ex3_testing, GEMM_STRIDED_BATCHED_EX3>;
    TEST_P(gemm_strided_batched_ex3, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemm_dispatch<gemm_ex3_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_strided_batched_ex3);

#endif //  BUILD_WITH_TENSILE

} // namespace
