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

#include "blas3/common_her2k.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    // possible test cases
    enum herkx_test_type
    {
        HERKX,
        HERKX_BATCHED,
        HERKX_STRIDED_BATCHED,
    };

    // test template
    template <template <typename...> class FILTER, herkx_test_type HERKX_TYPE>
    struct herkx_template : RocBLAS_Test<herkx_template<FILTER, HERKX_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<herkx_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HERKX_TYPE)
            {
            case HERKX:
                return !strcmp(arg.function, "herkx") || !strcmp(arg.function, "herkx_bad_arg");
            case HERKX_BATCHED:
                return !strcmp(arg.function, "herkx_batched")
                       || !strcmp(arg.function, "herkx_batched_bad_arg");
            case HERKX_STRIDED_BATCHED:
                return !strcmp(arg.function, "herkx_strided_batched")
                       || !strcmp(arg.function, "herkx_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<herkx_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                // beta only real

                name << '_' << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA) << '_'
                     << arg.N << '_' << arg.K;

                // use arg.get_alpha() to get real/complex alpha depending on datatype
                if(arg.a_type == rocblas_datatype_f32_c || arg.a_type == rocblas_datatype_f64_c)
                    name << '_' << arg.get_alpha<rocblas_float_complex>();
                else
                    name << '_' << arg.get_alpha<float>();

                name << '_' << arg.lda;

                if(HERKX_TYPE == HERKX_STRIDED_BATCHED)
                    name << '_' << arg.stride_a;

                name << '_' << arg.ldb;

                if(HERKX_TYPE == HERKX_STRIDED_BATCHED)
                    name << '_' << arg.stride_b;

                name << '_' << arg.get_beta<float>();

                name << '_' << arg.ldc;

                if(HERKX_TYPE == HERKX_STRIDED_BATCHED)
                    name << '_' << arg.stride_c;

                if(HERKX_TYPE != HERKX)
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
    struct herkx_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct herkx_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            // testing_her2k second template false for herkx
            if(!strcmp(arg.function, "herkx"))
                testing_her2k<T, false>(arg);
            else if(!strcmp(arg.function, "herkx_bad_arg"))
                testing_her2k_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "herkx_batched"))
                testing_her2k_batched<T, false>(arg);
            else if(!strcmp(arg.function, "herkx_batched_bad_arg"))
                testing_her2k_batched_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "herkx_strided_batched"))
                testing_her2k_strided_batched<T, false>(arg);
            else if(!strcmp(arg.function, "herkx_strided_batched_bad_arg"))
                testing_her2k_strided_batched_bad_arg<T, false>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using herkx = herkx_template<herkx_testing, HERKX>;
    TEST_P(herkx, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<herkx_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herkx);

    using herkx_batched = herkx_template<herkx_testing, HERKX_BATCHED>;
    TEST_P(herkx_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<herkx_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herkx_batched);

    using herkx_strided_batched = herkx_template<herkx_testing, HERKX_STRIDED_BATCHED>;
    TEST_P(herkx_strided_batched, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<herkx_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herkx_strided_batched);

} // namespace
