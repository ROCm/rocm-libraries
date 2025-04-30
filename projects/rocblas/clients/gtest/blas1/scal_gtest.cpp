/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "blas1_gtest.hpp"

#include "blas1/common_scal.hpp"

namespace
{
    // ----------------------------------------------------------------------------
    // BLAS1 testing template
    // ----------------------------------------------------------------------------
    template <template <typename...> class FILTER, blas1 BLAS1>
    struct scal_test_template : public RocBLAS_Test<scal_test_template<FILTER, BLAS1>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_blas1_dispatch<scal_test_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg);

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<scal_test_template> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);

            if(arg.a_type != arg.b_type)
                name << '_' << rocblas_datatype2string(arg.b_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                constexpr bool is_batched = (BLAS1 == blas1::scal_batched);
                constexpr bool is_strided = (BLAS1 == blas1::scal_strided_batched);

                name << '_' << arg.N << '_' << arg.alpha << "_" << arg.alphai;

                name << '_' << arg.incx;

                if(is_strided)
                {
                    name << '_' << arg.stride_x;
                }

                if(is_batched || is_strided)
                {
                    name << "_" << arg.batch_count;
                }
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

    // This tells whether the BLAS1 tests are enabled
    template <blas1 BLAS1, typename Ti, typename To>
    using scal_enabled = std::integral_constant<
        bool,
        ((BLAS1 == blas1::scal || BLAS1 == blas1::scal_batched
          || BLAS1 == blas1::scal_strided_batched)
         && (((sizeof(Ti) >= 4) && std::is_same_v<Ti, To>)
             || (std::is_same_v<Ti, rocblas_double_complex> && std::is_same_v<To, double>)
             || (std::is_same_v<Ti, rocblas_float_complex> && std::is_same_v<To, float>)))>;

// Creates tests for one of the BLAS 1 functions
// ARG passes 1-3 template arguments to the testing_* function
#define BLAS1_TESTING(NAME, ARG)                                                        \
    struct blas1_##NAME                                                                 \
    {                                                                                   \
        template <typename Ti, typename To = Ti, typename = void>                       \
        struct testing : rocblas_test_invalid                                           \
        {                                                                               \
        };                                                                              \
                                                                                        \
        template <typename Ti, typename To>                                             \
        struct testing<Ti, To, std::enable_if_t<scal_enabled<blas1::NAME, Ti, To>{}>>   \
            : rocblas_test_valid                                                        \
        {                                                                               \
            void operator()(const Arguments& arg)                                       \
            {                                                                           \
                if(!strcmp(arg.function, #NAME))                                        \
                    testing_##NAME<ARG(Ti, To)>(arg);                                   \
                else if(!strcmp(arg.function, #NAME "_bad_arg"))                        \
                    testing_##NAME##_bad_arg<ARG(Ti, To)>(arg);                         \
                else                                                                    \
                    FAIL() << "Internal error: Test called with unknown function: "     \
                           << arg.function;                                             \
            }                                                                           \
        };                                                                              \
    };                                                                                  \
                                                                                        \
    using NAME = scal_test_template<blas1_##NAME::template testing, blas1::NAME>;       \
                                                                                        \
    template <>                                                                         \
    inline bool NAME::function_filter(const Arguments& arg)                             \
    {                                                                                   \
        return !strcmp(arg.function, #NAME) || !strcmp(arg.function, #NAME "_bad_arg"); \
    }                                                                                   \
                                                                                        \
    TEST_P(NAME, blas1)                                                                 \
    {                                                                                   \
        RUN_TEST_ON_THREADS_STREAMS(                                                    \
            rocblas_blas1_dispatch<blas1_##NAME::template testing>(GetParam()));        \
    }                                                                                   \
                                                                                        \
    INSTANTIATE_TEST_CATEGORIES(NAME)

#define ARG2(Ti, To) Ti, To

    BLAS1_TESTING(scal, ARG2)
    BLAS1_TESTING(scal_batched, ARG2)
    BLAS1_TESTING(scal_strided_batched, ARG2)

} // namespace
