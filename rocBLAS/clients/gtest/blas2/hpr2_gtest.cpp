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

#include "blas2/common_hpr2.hpp"
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
    enum hpr2_test_type
    {
        HPR2,
        HPR2_BATCHED,
        HPR2_STRIDED_BATCHED,
    };

    //hpr2 test template
    template <template <typename...> class FILTER, hpr2_test_type HPR2_TYPE>
    struct hpr2_template : RocBLAS_Test<hpr2_template<FILTER, HPR2_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<hpr2_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HPR2_TYPE)
            {
            case HPR2:
                return !strcmp(arg.function, "hpr2") || !strcmp(arg.function, "hpr2_bad_arg");
            case HPR2_BATCHED:
                return !strcmp(arg.function, "hpr2_batched")
                       || !strcmp(arg.function, "hpr2_batched_bad_arg");
            case HPR2_STRIDED_BATCHED:
                return !strcmp(arg.function, "hpr2_strided_batched")
                       || !strcmp(arg.function, "hpr2_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<hpr2_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                // T doesn't really matter here, just whether it's real or complex. In hpr2's case it's always complex
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_'
                     << arg.get_alpha<rocblas_double_complex>() << '_' << arg.incx << '_'
                     << arg.incy;

                if(HPR2_TYPE == HPR2_STRIDED_BATCHED)
                    name << '_' << arg.stride_x << '_' << arg.stride_y << '_' << arg.stride_a;

                if(HPR2_TYPE != HPR2)
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
    struct hpr2_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct hpr2_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "hpr2"))
                testing_hpr2<T>(arg);
            else if(!strcmp(arg.function, "hpr2_bad_arg"))
                testing_hpr2_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "hpr2_batched"))
                testing_hpr2_batched<T>(arg);
            else if(!strcmp(arg.function, "hpr2_batched_bad_arg"))
                testing_hpr2_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "hpr2_strided_batched"))
                testing_hpr2_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "hpr2_strided_batched_bad_arg"))
                testing_hpr2_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using hpr2 = hpr2_template<hpr2_testing, HPR2>;
    TEST_P(hpr2, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hpr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hpr2);

    using hpr2_batched = hpr2_template<hpr2_testing, HPR2_BATCHED>;
    TEST_P(hpr2_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hpr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hpr2_batched);

    using hpr2_strided_batched = hpr2_template<hpr2_testing, HPR2_STRIDED_BATCHED>;
    TEST_P(hpr2_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hpr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hpr2_strided_batched);

} // namespace
