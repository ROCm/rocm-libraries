/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "blas_ex/common_gemmt.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum gemmt_test_type
    {
        GEMMT,
        GEMMT_BATCHED,
        GEMMT_STRIDED_BATCHED,
    };

    // test template
    template <template <typename...> class FILTER, gemmt_test_type GEMMT_TYPE>
    struct gemmt_template : RocBLAS_Test<gemmt_template<FILTER, GEMMT_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<gemmt_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEMMT_TYPE)
            {
            case GEMMT:
                return !strcmp(arg.function, "gemmt") || !strcmp(arg.function, "gemmt_bad_arg");
            case GEMMT_BATCHED:
                return !strcmp(arg.function, "gemmt_batched")
                       || !strcmp(arg.function, "gemmt_batched_bad_arg");
            case GEMMT_STRIDED_BATCHED:
                return !strcmp(arg.function, "gemmt_strided_batched")
                       || !strcmp(arg.function, "gemmt_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gemmt_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                constexpr bool isBatched
                    = (GEMMT_TYPE == GEMMT_STRIDED_BATCHED || GEMMT_TYPE == GEMMT_BATCHED);

                name << '_' << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA)
                     << (char)std::toupper(arg.transB);

                name << '_' << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_' << arg.lda
                     << '_' << arg.ldb << '_' << arg.beta << '_' << arg.ldc;

                if(isBatched)
                    name << '_' << arg.batch_count;

                if(GEMMT_TYPE != GEMMT)
                    name << '_' << arg.stride_a << '_' << arg.stride_b << '_' << arg.stride_c;
            }

            if(arg.api & c_API_64)
            {
                name << "_I64";
            }

            if(arg.api & c_API_FORTRAN)
                name << "_F";

            return std::move(name);
        }
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct gemmt_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct gemmt_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemmt"))
                testing_gemmt<T>(arg);
            else if(!strcmp(arg.function, "gemmt_bad_arg"))
                testing_gemmt_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemmt_batched"))
                testing_gemmt_batched<T>(arg);
            else if(!strcmp(arg.function, "gemmt_batched_bad_arg"))
                testing_gemmt_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemmt_strided_batched"))
                testing_gemmt_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "gemmt_strided_batched_bad_arg"))
                testing_gemmt_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemmt = gemmt_template<gemmt_testing, GEMMT>;
    TEST_P(gemmt, blas_ex)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<gemmt_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemmt);

    using gemmt_batched = gemmt_template<gemmt_testing, GEMMT_BATCHED>;
    TEST_P(gemmt_batched, blas_ex)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<gemmt_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemmt_batched);

    using gemmt_strided_batched = gemmt_template<gemmt_testing, GEMMT_STRIDED_BATCHED>;
    TEST_P(gemmt_strided_batched, blas_ex)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<gemmt_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemmt_strided_batched);

} // namespace
