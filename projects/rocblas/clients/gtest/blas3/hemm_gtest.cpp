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
 *
 * ************************************************************************ */

#include "blas3/common_symm_hemm.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum hemm_test_type
    {
        HEMM,
        HEMM_BATCHED,
        HEMM_STRIDED_BATCHED,
    };

    // test template
    template <template <typename...> class FILTER, hemm_test_type HEMM_TYPE>
    struct hemm_template : RocBLAS_Test<hemm_template<FILTER, HEMM_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<hemm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HEMM_TYPE)
            {
            case HEMM:
                return !strcmp(arg.function, "hemm") || !strcmp(arg.function, "hemm_bad_arg");
            case HEMM_BATCHED:
                return !strcmp(arg.function, "hemm_batched")
                       || !strcmp(arg.function, "hemm_batched_bad_arg");
            case HEMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "hemm_strided_batched")
                       || !strcmp(arg.function, "hemm_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<hemm_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (char)std::toupper(arg.side) << (char)std::toupper(arg.uplo) << '_'
                     << arg.M << '_' << arg.N;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(HEMM_TYPE == HEMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.ldb;

                if(HEMM_TYPE == HEMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_b;

                if(arg.b_type == rocblas_datatype_f32_c || arg.b_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_beta<rocblas_float_complex>();
                else
                    name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldc;

                if(HEMM_TYPE == HEMM_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(HEMM_TYPE == HEMM_STRIDED_BATCHED || HEMM_TYPE == HEMM_BATCHED)
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
    struct hemm_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct hemm_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            // testing_symm_hemm second argument is Hermitian to perform hemm function
            if(!strcmp(arg.function, "hemm"))
                testing_symm_hemm<T, true>(arg);
            else if(!strcmp(arg.function, "hemm_bad_arg"))
                testing_symm_hemm_bad_arg<T, true>(arg);
            else if(!strcmp(arg.function, "hemm_batched"))
                testing_symm_hemm_batched<T, true>(arg);
            else if(!strcmp(arg.function, "hemm_batched_bad_arg"))
                testing_symm_hemm_batched_bad_arg<T, true>(arg);
            else if(!strcmp(arg.function, "hemm_strided_batched"))
                testing_symm_hemm_strided_batched<T, true>(arg);
            else if(!strcmp(arg.function, "hemm_strided_batched_bad_arg"))
                testing_symm_hemm_strided_batched_bad_arg<T, true>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using hemm = hemm_template<hemm_testing, HEMM>;
    TEST_P(hemm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hemm);

    using hemm_batched = hemm_template<hemm_testing, HEMM_BATCHED>;
    TEST_P(hemm_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hemm_batched);

    using hemm_strided_batched = hemm_template<hemm_testing, HEMM_STRIDED_BATCHED>;
    TEST_P(hemm_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_simple_dispatch<hemm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hemm_strided_batched);

} // namespace
