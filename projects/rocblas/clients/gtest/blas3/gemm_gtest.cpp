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

#include "blas3/common_gemm.hpp"
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
        GEMM,
        GEMM_BATCHED,
        GEMM_STRIDED_BATCHED,
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
            case GEMM:
                return !strcmp(arg.function, "gemm") || !strcmp(arg.function, "gemm_bad_arg");

            case GEMM_BATCHED:
                return !strcmp(arg.function, "gemm_batched")
                       || !strcmp(arg.function, "gemm_batched_bad_arg");

            case GEMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "gemm_strided_batched")
                       || !strcmp(arg.function, "gemm_strided_batched_bad_arg");
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
                constexpr bool isBatched = (GEMM_TYPE != GEMM);

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                     << arg.lda << '_' << arg.ldb << '_' << arg.beta << '_' << arg.ldc;

                if(isBatched)
                    name << '_' << arg.batch_count;

                if(GEMM_TYPE == GEMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_a << '_' << arg.stride_b << '_' << arg.stride_c;

                if(arg.math_mode == rocblas_xf32_xdl_math_op)
                    name << "_xf32";
            }

            if(arg.api & c_API_64)
            {
                name << "_I64";
            }
            if(arg.api & c_API_FORTRAN)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    // ----------------------------------------------------------------------------
    // gemm
    // gemm_batched
    // gemm_strided_batched
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct gemm_testing : rocblas_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename T>
    struct gemm_testing<
        T,
        T,
        T,
        std::enable_if_t<!std::is_same_v<T, void> && !std::is_same_v<T, rocblas_bfloat16>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemm"))
                testing_gemm<T>(arg);
            else if(!strcmp(arg.function, "gemm_bad_arg"))
                testing_gemm_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemm_batched"))
                testing_gemm_batched<T>(arg);
            else if(!strcmp(arg.function, "gemm_batched_bad_arg"))
                testing_gemm_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemm_strided_batched"))
                testing_gemm_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "gemm_strided_batched_bad_arg"))
                testing_gemm_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemm = gemm_test_template<gemm_testing, GEMM>;
    TEST_P(gemm, blas3_tensile)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocblas_gemm_dispatch<gemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm);

    using gemm_batched = gemm_test_template<gemm_testing, GEMM_BATCHED>;
    TEST_P(gemm_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_batched);

    using gemm_strided_batched = gemm_test_template<gemm_testing, GEMM_STRIDED_BATCHED>;
    TEST_P(gemm_strided_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_strided_batched);

} // namespace
