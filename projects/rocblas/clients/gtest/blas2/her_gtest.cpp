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

#include "blas2/common_her.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum her_test_type
    {
        HER,
        HER_BATCHED,
        HER_STRIDED_BATCHED,
    };

    //her test template
    template <template <typename...> class FILTER, her_test_type HER_TYPE>
    struct her_template : RocBLAS_Test<her_template<FILTER, HER_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<her_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HER_TYPE)
            {
            case HER:
                return !strcmp(arg.function, "her") || !strcmp(arg.function, "her_bad_arg");
            case HER_BATCHED:
                return !strcmp(arg.function, "her_batched")
                       || !strcmp(arg.function, "her_batched_bad_arg");
            case HER_STRIDED_BATCHED:
                return !strcmp(arg.function, "her_strided_batched")
                       || !strcmp(arg.function, "her_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<her_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_' << arg.lda << '_'
                     << arg.alpha << '_' << arg.incx;

                if(HER_TYPE == HER_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                if(HER_TYPE == HER_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(HER_TYPE == HER_STRIDED_BATCHED || HER_TYPE == HER_BATCHED)
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
    struct her_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct her_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "her"))
                testing_her<T>(arg);
            else if(!strcmp(arg.function, "her_bad_arg"))
                testing_her_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "her_batched"))
                testing_her_batched<T>(arg);
            else if(!strcmp(arg.function, "her_batched_bad_arg"))
                testing_her_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "her_strided_batched"))
                testing_her_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "her_strided_batched_bad_arg"))
                testing_her_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using her = her_template<her_testing, HER>;
    TEST_P(her, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<her_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her);

    using her_batched = her_template<her_testing, HER_BATCHED>;
    TEST_P(her_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<her_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her_batched);

    using her_strided_batched = her_template<her_testing, HER_STRIDED_BATCHED>;
    TEST_P(her_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<her_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(her_strided_batched);

} // namespace
