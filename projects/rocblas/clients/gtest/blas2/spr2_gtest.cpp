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

#include "blas2/common_spr2.hpp"
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
    enum spr2_test_type
    {
        SPR2,
        SPR2_BATCHED,
        SPR2_STRIDED_BATCHED,
    };

    //spr2 test template
    template <template <typename...> class FILTER, spr2_test_type SPR2_TYPE>
    struct spr2_template : RocBLAS_Test<spr2_template<FILTER, SPR2_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<spr2_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SPR2_TYPE)
            {
            case SPR2:
                return !strcmp(arg.function, "spr2") || !strcmp(arg.function, "spr2_bad_arg");
            case SPR2_BATCHED:
                return !strcmp(arg.function, "spr2_batched")
                       || !strcmp(arg.function, "spr2_batched_bad_arg");
            case SPR2_STRIDED_BATCHED:
                return !strcmp(arg.function, "spr2_strided_batched")
                       || !strcmp(arg.function, "spr2_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<spr2_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.uplo) << '_' << arg.N << '_' << arg.alpha
                     << '_' << arg.incx;

                if(SPR2_TYPE == SPR2_STRIDED_BATCHED)
                    name << '_' << arg.stride_x;

                name << '_' << arg.incy;

                if(SPR2_TYPE == SPR2_STRIDED_BATCHED)
                    name << '_' << arg.stride_y;

                if(SPR2_TYPE == SPR2_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                if(SPR2_TYPE == !SPR2)
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
    struct spr2_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct spr2_testing<T, std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "spr2"))
                testing_spr2<T>(arg);
            else if(!strcmp(arg.function, "spr2_bad_arg"))
                testing_spr2_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "spr2_batched"))
                testing_spr2_batched<T>(arg);
            else if(!strcmp(arg.function, "spr2_batched_bad_arg"))
                testing_spr2_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "spr2_strided_batched"))
                testing_spr2_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "spr2_strided_batched_bad_arg"))
                testing_spr2_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using spr2 = spr2_template<spr2_testing, SPR2>;
    TEST_P(spr2, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<spr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(spr2);

    using spr2_batched = spr2_template<spr2_testing, SPR2_BATCHED>;
    TEST_P(spr2_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<spr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(spr2_batched);

    using spr2_strided_batched = spr2_template<spr2_testing, SPR2_STRIDED_BATCHED>;
    TEST_P(spr2_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<spr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(spr2_strided_batched);

} // namespace
