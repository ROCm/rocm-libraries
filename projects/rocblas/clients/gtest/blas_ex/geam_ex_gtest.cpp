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
#include "blas_ex/common_geam_ex.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    // possible geam_ex test cases
    enum geam_ex_test_type
    {
        GEAM_EX,
        GEAM_BATCHED_EX,
        GEAM_STRIDED_BATCHED_EX,
    };

    // geam_ex test template
    template <template <typename...> class FILTER, geam_ex_test_type GEAM_EX_TYPE>
    struct geam_ex_template : RocBLAS_Test<geam_ex_template<FILTER, GEAM_EX_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<geam_ex_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEAM_EX_TYPE)
            {
            case GEAM_EX:
                return !strcmp(arg.function, "geam_ex") || !strcmp(arg.function, "geam_ex_bad_arg");
                // case GEAM_BATCHED_EX:
                //     return !strcmp(arg.function, "geam_batched_ex")
                //            || !strcmp(arg.function, "geam_batched_bad_arg_ex");
                // case GEAM_STRIDED_BATCHED_EX:
                //     return !strcmp(arg.function, "geam_strided_batched_ex")
                //            || !strcmp(arg.function, "geam_strided_batched_bad_arg_ex");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<geam_ex_template> name(arg.name);

            if(rocblas_geam_ex_operation(arg.geam_ex_op) == rocblas_geam_ex_operation_min_plus)
                name << "min_plus";
            else if(rocblas_geam_ex_operation(arg.geam_ex_op) == rocblas_geam_ex_operation_plus_min)
                name << "plus_min";

            // No support for mixed precision
            name << '_' << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB)
                     << '_' << arg.M << '_' << arg.N << '_' << arg.K;

                name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(GEAM_EX_TYPE == GEAM_STRIDED_BATCHED_EX)
                    name << '_' << arg.stride_a;

                name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldb;

                if(GEAM_EX_TYPE == GEAM_STRIDED_BATCHED_EX)
                    name << '_' << arg.stride_b;

                name << '_' << arg.ldc;

                if(GEAM_EX_TYPE == GEAM_STRIDED_BATCHED_EX)
                    name << '_' << arg.stride_c;

                name << '_' << arg.ldd;

                if(GEAM_EX_TYPE == GEAM_STRIDED_BATCHED_EX)
                    name << '_' << arg.stride_d;

                if(GEAM_EX_TYPE == GEAM_STRIDED_BATCHED_EX || GEAM_EX_TYPE == GEAM_BATCHED_EX)
                    name << '_' << arg.batch_count;
            }

            if(arg.api == FORTRAN)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct geam_ex_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct geam_ex_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T,
                           float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_half>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "geam_ex"))
                testing_geam_ex<T>(arg);
            else if(!strcmp(arg.function, "geam_ex_bad_arg"))
                testing_geam_ex_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using geam_ex = geam_ex_template<geam_ex_testing, GEAM_EX>;
    TEST_P(geam_ex, blas_ex)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<geam_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geam_ex);

} // namespace
