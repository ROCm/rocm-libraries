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

#include "blas2/common_gemv.hpp"
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
    // possible gemv test cases
    enum gemv_test_type
    {
        GEMV,
        GEMV_BATCHED,
        GEMV_STRIDED_BATCHED,
    };

    //gemv test template
    template <template <typename...> class FILTER, gemv_test_type GEMV_TYPE>
    struct gemv_template : RocBLAS_Test<gemv_template<FILTER, GEMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<gemv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEMV_TYPE)
            {
            case GEMV:
                return !strcmp(arg.function, "gemv") || !strcmp(arg.function, "gemv_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gemv_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.transA)
                 << '_' << arg.M << '_' << arg.N << '_' << arg.alpha << '_' << arg.lda;

            name << '_' << arg.incx;

            name << '_' << arg.beta << '_' << arg.incy;

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

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct gemv_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct gemv_testing<T,
                        std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}
                                         || std::is_same<T, rocblas_float_complex>{}
                                         || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemv"))
                testing_gemv<T>(arg);
            else if(!strcmp(arg.function, "gemv_bad_arg"))
                testing_gemv_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    //gemv_batched and gemv_strided_batched test template
    template <template <typename...> class FILTER, gemv_test_type GEMV_TYPE>
    struct gemv_batched_and_strided_batched_template
        : RocBLAS_Test<gemv_batched_and_strided_batched_template<FILTER, GEMV_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_gemv_batched_and_strided_batched_dispatch<
                gemv_batched_and_strided_batched_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEMV_TYPE)
            {
            case GEMV_BATCHED:
                return !strcmp(arg.function, "gemv_batched")
                       || !strcmp(arg.function, "gemv_batched_bad_arg");
            case GEMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "gemv_strided_batched")
                       || !strcmp(arg.function, "gemv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gemv_batched_and_strided_batched_template> name(arg.name);
            if(arg.a_type == rocblas_datatype_bf16_r || arg.a_type == rocblas_datatype_f16_r)
                name << rocblas_datatype2string(arg.a_type) << '_'
                     << rocblas_datatype2string(arg.c_type) << '_'
                     << rocblas_datatype2string(arg.compute_type);
            else
                name << rocblas_datatype2string(arg.a_type);

            name << '_' << (char)std::toupper(arg.transA) << '_' << arg.M << '_' << arg.N << '_'
                 << arg.alpha << '_' << arg.lda;

            if(GEMV_TYPE == GEMV_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            name << '_' << arg.incx;

            if(GEMV_TYPE == GEMV_STRIDED_BATCHED)
                name << '_' << arg.stride_x;

            name << '_' << arg.beta << '_' << arg.incy;

            if(GEMV_TYPE == GEMV_STRIDED_BATCHED)
                name << '_' << arg.stride_y;

            if(GEMV_TYPE != GEMV)
                name << '_' << arg.batch_count;

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

    // rocblas-test coverage for gemv_batched and gemv_strided_batched
    template <typename Ti, typename Tex = Ti, typename To = Tex, typename = void>
    struct gemv_batched_and_strided_batched_testing : rocblas_test_invalid
    {
    };

    // When the condition in the fourth argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename Ti, typename Tex, typename To>
    struct gemv_batched_and_strided_batched_testing<
        Ti,
        Tex,
        To,
        std::enable_if_t<
            (std::is_same_v<
                 Ti,
                 Tex> && std::is_same_v<Tex, To> && (std::is_same_v<Ti, float> || std::is_same_v<Ti, double> || std::is_same_v<Ti, rocblas_float_complex> || std::is_same_v<Ti, rocblas_double_complex>))
            || (std::is_same_v<
                    Ti,
                    rocblas_half> && std::is_same_v<Tex, float> && (std::is_same_v<To, Ti> || std::is_same_v<To, float>))
            || (std::is_same_v<
                    Ti,
                    rocblas_bfloat16> && std::is_same_v<Tex, float> && (std::is_same_v<To, Ti> || std::is_same_v<To, float>))>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemv_batched"))
                testing_gemv_batched<Ti, Tex, To>(arg);
            else if(!strcmp(arg.function, "gemv_batched_bad_arg"))
                testing_gemv_batched_bad_arg<Ti, Tex, To>(arg);
            else if(!strcmp(arg.function, "gemv_strided_batched"))
                testing_gemv_strided_batched<Ti, Tex, To>(arg);
            else if(!strcmp(arg.function, "gemv_strided_batched_bad_arg"))
                testing_gemv_strided_batched_bad_arg<Ti, Tex, To>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemv = gemv_template<gemv_testing, GEMV>;
    TEST_P(gemv, blas2)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocblas_simple_dispatch<gemv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv);

    using gemv_batched
        = gemv_batched_and_strided_batched_template<gemv_batched_and_strided_batched_testing,
                                                    GEMV_BATCHED>;
    TEST_P(gemv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemv_batched_and_strided_batched_dispatch<
                gemv_batched_and_strided_batched_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv_batched);

    using gemv_strided_batched
        = gemv_batched_and_strided_batched_template<gemv_batched_and_strided_batched_testing,
                                                    GEMV_STRIDED_BATCHED>;
    TEST_P(gemv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemv_batched_and_strided_batched_dispatch<
                gemv_batched_and_strided_batched_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv_strided_batched);

} // namespace
