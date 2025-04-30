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
#include "blas3/common_dgmm.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible dgmm test cases
    enum dgmm_test_type
    {
        DGMM,
        DGMM_BATCHED,
        DGMM_STRIDED_BATCHED,
    };

    //dgmm test template
    template <template <typename...> class FILTER, dgmm_test_type DGMM_TYPE>
    struct dgmm_template : RocBLAS_Test<dgmm_template<FILTER, DGMM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<dgmm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(DGMM_TYPE)
            {
            case DGMM:
                return !strcmp(arg.function, "dgmm") || !strcmp(arg.function, "dgmm_bad_arg");
            case DGMM_BATCHED:
                return !strcmp(arg.function, "dgmm_batched")
                       || !strcmp(arg.function, "dgmm_batched_bad_arg");
            case DGMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "dgmm_strided_batched")
                       || !strcmp(arg.function, "dgmm_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<dgmm_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.side) << '_' << arg.M << '_' << arg.N;

                name << '_' << arg.lda;

                if(DGMM_TYPE == DGMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.incx;

                if(DGMM_TYPE == DGMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.ldc;

                if(DGMM_TYPE == DGMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(DGMM_TYPE != DGMM)
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

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct dgmm_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct dgmm_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "dgmm"))
                testing_dgmm<T>(arg);
            else if(!strcmp(arg.function, "dgmm_bad_arg"))
                testing_dgmm_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "dgmm_batched"))
                testing_dgmm_batched<T>(arg);
            else if(!strcmp(arg.function, "dgmm_batched_bad_arg"))
                testing_dgmm_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "dgmm_strided_batched"))
                testing_dgmm_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "dgmm_strided_batched_bad_arg"))
                testing_dgmm_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using dgmm = dgmm_template<dgmm_testing, DGMM>;
    TEST_P(dgmm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<dgmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(dgmm);

    using dgmm_batched = dgmm_template<dgmm_testing, DGMM_BATCHED>;
    TEST_P(dgmm_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<dgmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(dgmm_batched);

    using dgmm_strided_batched = dgmm_template<dgmm_testing, DGMM_STRIDED_BATCHED>;
    TEST_P(dgmm_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<dgmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(dgmm_strided_batched);

} // namespace
