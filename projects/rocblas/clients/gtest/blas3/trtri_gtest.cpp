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

#include "blas3/common_trtri.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    enum trtri_test_type
    {
        TRTRI,
        TRTRI_BATCHED,
        TRTRI_STRIDED_BATCHED,
    };

    // By default, this test does not apply to any types.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trtri_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trtri_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trtri"))
                testing_trtri<T>(arg);
            else if(!strcmp(arg.function, "trtri_bad_arg"))
                testing_trtri_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trtri_batched"))
                testing_trtri_batched<T>(arg);
            else if(!strcmp(arg.function, "trtri_batched_bad_arg"))
                testing_trtri_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trtri_strided_batched"))
                testing_trtri_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "trtri_strided_batched_bad_arg"))
                testing_trtri_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    template <trtri_test_type K>
    struct trtri_template : RocBLAS_Test<trtri_template<K>, trtri_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<trtri_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            if(K == TRTRI)
                return !strcmp(arg.function, "trtri") || !strcmp(arg.function, "trtri_bad_arg");
            else if(K == TRTRI_BATCHED)
                return !strcmp(arg.function, "trtri_batched")
                       || !strcmp(arg.function, "trtri_batched_bad_arg");
            else
                return !strcmp(arg.function, "trtri_strided_batched")
                       || !strcmp(arg.function, "trtri_strided_batched_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<trtri_template> name(arg.name);
            name << rocblas_datatype2string(arg.a_type) << '_' << (char)std::toupper(arg.uplo)
                 << (char)std::toupper(arg.diag) << '_' << arg.N << '_' << arg.lda;

            if(K == TRTRI_STRIDED_BATCHED)
                name << '_' << arg.stride_a;

            if(K != TRTRI)
                name << '_' << arg.batch_count;

            if(arg.api == FORTRAN)
            {
                name << "_F";
            }

            return std::move(name);
        }
    };

    using trtri = trtri_template<TRTRI>;
    TEST_P(trtri, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri);

    using trtri_batched = trtri_template<TRTRI_BATCHED>;
    TEST_P(trtri_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri_batched);

    using trtri_strided_batched = trtri_template<TRTRI_STRIDED_BATCHED>;
    TEST_P(trtri_strided_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri_strided_batched);

} // namespace
