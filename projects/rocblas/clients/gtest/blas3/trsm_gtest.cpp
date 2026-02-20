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

#include "blas3/common_trsm.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "testing_trsm_batched_ex.hpp"
#include "testing_trsm_ex.hpp"
#include "testing_trsm_strided_batched_ex.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible trsm test cases
    enum trsm_test_type
    {
        TRSM,
        TRSM_EX,
        TRSM_BATCHED,
        TRSM_BATCHED_EX,
        TRSM_STRIDED_BATCHED,
        TRSM_STRIDED_BATCHED_EX,
    };

    // trsm test template
    template <template <typename...> class FILTER, trsm_test_type TRSM_TYPE>
    struct trsm_template : RocBLAS_Test<trsm_template<FILTER, TRSM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<trsm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRSM_TYPE)
            {
            case TRSM:
                return !strcmp(arg.function, "trsm") || !strcmp(arg.function, "trsm_bad_arg")
                       || !strcmp(arg.function, "trsm_internal_interfaces");
            case TRSM_EX:
                return !strcmp(arg.function, "trsm_ex") || !strcmp(arg.function, "trsm_ex_bad_arg");
            case TRSM_BATCHED:
                return !strcmp(arg.function, "trsm_batched")
                       || !strcmp(arg.function, "trsm_batched_bad_arg")
                       || !strcmp(arg.function, "trsm_batched_internal_interfaces");
            case TRSM_BATCHED_EX:
                return !strcmp(arg.function, "trsm_batched_ex")
                       || !strcmp(arg.function, "trsm_batched_ex_bad_arg");
            case TRSM_STRIDED_BATCHED:
                return !strcmp(arg.function, "trsm_strided_batched")
                       || !strcmp(arg.function, "trsm_strided_batched_bad_arg");
            case TRSM_STRIDED_BATCHED_EX:
                return !strcmp(arg.function, "trsm_strided_batched_ex")
                       || !strcmp(arg.function, "trsm_strided_batched_ex_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<trsm_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else if(strstr(arg.function, "_internal_interfaces") != nullptr)
            {
                name << "_internal_interfaces";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.side) << (char)std::toupper(arg.uplo)
                     << (char)std::toupper(arg.transA) << (char)std::toupper(arg.diag) << '_'
                     << arg.M << '_' << arg.N << '_' << arg.alpha << '_' << arg.lda << '_';

                if(TRSM_TYPE == TRSM_STRIDED_BATCHED || TRSM_TYPE == TRSM_STRIDED_BATCHED_EX)
                    name << arg.stride_a << '_';

                name << arg.ldb;

                if(TRSM_TYPE == TRSM_STRIDED_BATCHED || TRSM_TYPE == TRSM_STRIDED_BATCHED_EX)
                    name << '_' << arg.stride_b;
                if(TRSM_TYPE == TRSM_STRIDED_BATCHED || TRSM_TYPE == TRSM_STRIDED_BATCHED_EX
                   || TRSM_TYPE == TRSM_BATCHED || TRSM_TYPE == TRSM_BATCHED_EX)
                    name << '_' << arg.batch_count;
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

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trsm_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trsm_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trsm"))
                testing_trsm<T>(arg);
            else if(!strcmp(arg.function, "trsm_bad_arg"))
                testing_trsm_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trsm_internal_interfaces"))
                testing_trsm_internal_interfaces<T>(arg);
            else if(!strcmp(arg.function, "trsm_ex"))
                testing_trsm_ex<T>(arg);
            else if(!strcmp(arg.function, "trsm_ex_bad_arg"))
                testing_trsm_ex_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched"))
                testing_trsm_batched<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched_bad_arg"))
                testing_trsm_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched_internal_interfaces"))
                testing_trsm_batched_internal_interfaces<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched_ex"))
                testing_trsm_batched_ex<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched_ex_bad_arg"))
                testing_trsm_batched_ex_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trsm_strided_batched"))
                testing_trsm_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "trsm_strided_batched_bad_arg"))
                testing_trsm_strided_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trsm_strided_batched_ex"))
                testing_trsm_strided_batched_ex<T>(arg);
            else if(!strcmp(arg.function, "trsm_strided_batched_ex_bad_arg"))
                testing_trsm_strided_batched_ex_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using trsm = trsm_template<trsm_testing, TRSM>;
    TEST_P(trsm, blas3_tensile)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocblas_simple_dispatch<trsm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm);

    using trsm_ex = trsm_template<trsm_testing, TRSM_EX>;
    TEST_P(trsm_ex, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_ex);

    using trsm_batched = trsm_template<trsm_testing, TRSM_BATCHED>;
    TEST_P(trsm_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_batched);

    using trsm_batched_ex = trsm_template<trsm_testing, TRSM_BATCHED_EX>;
    TEST_P(trsm_batched_ex, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_batched_ex);

    using trsm_strided_batched = trsm_template<trsm_testing, TRSM_STRIDED_BATCHED>;
    TEST_P(trsm_strided_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_strided_batched);

    using trsm_strided_batched_ex = trsm_template<trsm_testing, TRSM_STRIDED_BATCHED_EX>;
    TEST_P(trsm_strided_batched_ex, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<trsm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_strided_batched_ex);

} // namespace
