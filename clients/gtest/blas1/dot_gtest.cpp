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

#include "blas1/common_dot.hpp"

namespace
{
    // ----------------------------------------------------------------------------
    // BLAS1 testing template
    // ----------------------------------------------------------------------------
    template <template <typename...> class FILTER, blas1 BLAS1>
    struct dot_test_template : public RocBLAS_Test<dot_test_template<FILTER, BLAS1>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_blas1_dispatch<dot_test_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg);

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<dot_test_template> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                constexpr bool is_batched
                    = (BLAS1 == blas1::dot_batched || BLAS1 == blas1::dotc_batched);
                constexpr bool is_strided
                    = (BLAS1 == blas1::dot_strided_batched || BLAS1 == blas1::dotc_strided_batched);

                name << '_' << arg.N << '_' << arg.incx;

                if(is_strided)
                {
                    name << '_' << arg.stride_x;
                }

                name << '_' << arg.incy;

                if(is_strided)
                {
                    name << '_' << arg.stride_y;
                }

                if(is_batched || is_strided)
                {
                    name << "_" << arg.batch_count;
                }

                name << "_" << arg.algo;
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
    template <blas1 BLAS1, typename Ti, typename To, typename Tc>
    using dot_enabled = std::integral_constant<
        bool,
        ((BLAS1 == blas1::dot || BLAS1 == blas1::dot_batched || BLAS1 == blas1::dot_strided_batched)
             && (std::is_same_v<
                     Ti,
                     To> && std::is_same_v<To, Tc> && (std::is_same_v<Ti, rocblas_half> || std::is_same_v<Ti, rocblas_float_complex> || std::is_same_v<Ti, rocblas_double_complex> || std::is_same_v<Ti, float> || std::is_same_v<Ti, double>))
         || (std::is_same_v<
                 Ti,
                 To> && std::is_same_v<Ti, rocblas_bfloat16> && std::is_same_v<Tc, float>))

            || ((BLAS1 == blas1::dotc || BLAS1 == blas1::dotc_batched
                 || BLAS1 == blas1::dotc_strided_batched)
                && std::is_same_v<
                    To,
                    Ti> && std::is_same_v<To, Tc> && (std::is_same_v<Ti, rocblas_float_complex> || std::is_same_v<Ti, rocblas_double_complex>))>;

// Creates tests for one of the BLAS 1 functions
// ARG passes 1-3 template arguments to the testing_* function
#define BLAS1_TESTING(NAME, ARG)                                                             \
    struct blas1_##NAME                                                                      \
    {                                                                                        \
        template <typename Ti, typename To = Ti, typename Tc = To, typename = void>          \
        struct testing : rocblas_test_invalid                                                \
        {                                                                                    \
        };                                                                                   \
                                                                                             \
        template <typename Ti, typename To, typename Tc>                                     \
        struct testing<Ti, To, Tc, std::enable_if_t<dot_enabled<blas1::NAME, Ti, To, Tc>{}>> \
            : rocblas_test_valid                                                             \
        {                                                                                    \
            void operator()(const Arguments& arg)                                            \
            {                                                                                \
                if(!strcmp(arg.function, #NAME))                                             \
                    testing_##NAME<ARG(Ti, To, Tc)>(arg);                                    \
                else if(!strcmp(arg.function, #NAME "_bad_arg"))                             \
                    testing_##NAME##_bad_arg<ARG(Ti, To, Tc)>(arg);                          \
                else                                                                         \
                    FAIL() << "Internal error: Test called with unknown function: "          \
                           << arg.function;                                                  \
            }                                                                                \
        };                                                                                   \
    };                                                                                       \
                                                                                             \
    using NAME = dot_test_template<blas1_##NAME::template testing, blas1::NAME>;             \
                                                                                             \
    template <>                                                                              \
    inline bool NAME::function_filter(const Arguments& arg)                                  \
    {                                                                                        \
        return !strcmp(arg.function, #NAME) || !strcmp(arg.function, #NAME "_bad_arg");      \
    }                                                                                        \
                                                                                             \
    TEST_P(NAME, blas1)                                                                      \
    {                                                                                        \
        RUN_TEST_ON_THREADS_STREAMS(                                                         \
            rocblas_blas1_dispatch<blas1_##NAME::template testing>(GetParam()));             \
    }                                                                                        \
                                                                                             \
    INSTANTIATE_TEST_CATEGORIES(NAME)

#define ARG1(Ti, To, Tc) Ti

    BLAS1_TESTING(dot, ARG1)
    BLAS1_TESTING(dotc, ARG1)
    BLAS1_TESTING(dot_batched, ARG1)
    BLAS1_TESTING(dotc_batched, ARG1)
    BLAS1_TESTING(dot_strided_batched, ARG1)
    BLAS1_TESTING(dotc_strided_batched, ARG1)

} // namespace
