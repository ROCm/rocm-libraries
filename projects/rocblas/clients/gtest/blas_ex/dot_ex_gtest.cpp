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
#include "blas1_ex_gtest.hpp"

#include "blas_ex/common_dot_ex.hpp"

namespace
{

    // ----------------------------------------------------------------------------
    // BLAS1_ex testing template
    // ----------------------------------------------------------------------------
    template <template <typename...> class FILTER, blas1_ex BLAS1_EX>
    struct dot_ex_test_template
        : public RocBLAS_Test<dot_ex_test_template<FILTER, BLAS1_EX>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_blas1_ex_dispatch<dot_ex_test_template::template type_filter_functor>(
                arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg);

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<dot_ex_test_template> name(arg.name);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                constexpr bool is_batched = (BLAS1_EX == blas1_ex::dot_batched_ex
                                             || BLAS1_EX == blas1_ex::dotc_batched_ex);
                constexpr bool is_strided = (BLAS1_EX == blas1_ex::dot_strided_batched_ex
                                             || BLAS1_EX == blas1_ex::dotc_strided_batched_ex);

                name << rocblas_datatype2string(arg.a_type) << '_'
                     << rocblas_datatype2string(arg.b_type);

                name << '_' << rocblas_datatype2string(arg.c_type);

                name << '_' << rocblas_datatype2string(arg.compute_type);

                name << '_' << arg.N << '_' << arg.incx;

                if(is_strided)
                    name << '_' << arg.stride_x;

                name << '_' << arg.incy;

                if(is_strided)
                    name << '_' << arg.stride_y;

                if(is_batched || is_strided)
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

    // This tells whether the BLAS1_EX tests are enabled
    // Appears that we will need up to 4 template variables (see dot)
    template <blas1_ex BLAS1_EX, typename T1, typename T2, typename T3, typename T4>
    using blas1_ex_enabled = std::integral_constant<
        bool,

        // dot_ex
        // T1 is x_type, T2 is y_type, T3 is result_type, T4 is execution_type
        ((BLAS1_EX == blas1_ex::dot_ex || BLAS1_EX == blas1_ex::dot_batched_ex
          || BLAS1_EX == blas1_ex::dot_strided_batched_ex || BLAS1_EX == blas1_ex::dotc_ex
          || BLAS1_EX == blas1_ex::dotc_batched_ex || BLAS1_EX == blas1_ex::dotc_strided_batched_ex)
         && ((std::is_same_v<
                  T1,
                  T2> && std::is_same_v<T2, T3> && std::is_same_v<T3, T4> && (std::is_same_v<T1, float> || std::is_same_v<T1, double> || std::is_same_v<T1, rocblas_half> || std::is_same_v<T1, rocblas_float_complex> || std::is_same_v<T1, rocblas_double_complex>))
             || (std::is_same_v<
                     T1,
                     T2> && std::is_same_v<T2, T3> && std::is_same_v<T1, rocblas_half> && std::is_same_v<T4, float>)
             || (std::is_same_v<
                     T1,
                     T2> && std::is_same_v<T2, T3> && std::is_same_v<T1, rocblas_bfloat16> && std::is_same_v<T4, float>)
             || (std::is_same_v<
                     T1,
                     T2> && std::is_same_v<T3, T4> && std::is_same_v<T1, float> && std::is_same_v<T3, double>)))>;

// Creates tests for one of the BLAS 1 functions
// ARG passes 1-3 template arguments to the testing_* function
#define BLAS1_EX_TESTING(NAME, ARG)                                                           \
    struct blas1_ex_##NAME                                                                    \
    {                                                                                         \
        template <typename Ta,                                                                \
                  typename Tb  = Ta,                                                          \
                  typename Tc  = Tb,                                                          \
                  typename Tex = Tc,                                                          \
                  typename     = void>                                                        \
        struct testing : rocblas_test_invalid                                                 \
        {                                                                                     \
        };                                                                                    \
                                                                                              \
        template <typename Ta, typename Tb, typename Tc, typename Tex>                        \
        struct testing<Ta,                                                                    \
                       Tb,                                                                    \
                       Tc,                                                                    \
                       Tex,                                                                   \
                       std::enable_if_t<blas1_ex_enabled<blas1_ex::NAME, Ta, Tb, Tc, Tex>{}>> \
            : rocblas_test_valid                                                              \
        {                                                                                     \
            void operator()(const Arguments& arg)                                             \
            {                                                                                 \
                if(!strcmp(arg.function, #NAME))                                              \
                    testing_##NAME<ARG(Ta, Tb, Tc, Tex)>(arg);                                \
                else if(!strcmp(arg.function, #NAME "_bad_arg"))                              \
                    testing_##NAME##_bad_arg<ARG(Ta, Tb, Tc, Tex)>(arg);                      \
                else                                                                          \
                    FAIL() << "Internal error: Test called with unknown function: "           \
                           << arg.function;                                                   \
            }                                                                                 \
        };                                                                                    \
    };                                                                                        \
                                                                                              \
    using NAME = dot_ex_test_template<blas1_ex_##NAME::template testing, blas1_ex::NAME>;     \
                                                                                              \
    template <>                                                                               \
    inline bool NAME::function_filter(const Arguments& arg)                                   \
    {                                                                                         \
        return !strcmp(arg.function, #NAME) || !strcmp(arg.function, #NAME "_bad_arg");       \
    }                                                                                         \
                                                                                              \
    TEST_P(NAME, blas1_ex)                                                                    \
    {                                                                                         \
        RUN_TEST_ON_THREADS_STREAMS(                                                          \
            rocblas_blas1_ex_dispatch<blas1_ex_##NAME::template testing>(GetParam()));        \
    }                                                                                         \
                                                                                              \
    INSTANTIATE_TEST_CATEGORIES(NAME)

#define ARG4(Ta, Tb, Tc, Tex) Ta, Tb, Tc, Tex

    BLAS1_EX_TESTING(dot_ex, ARG4)
    BLAS1_EX_TESTING(dot_batched_ex, ARG4)
    BLAS1_EX_TESTING(dot_strided_batched_ex, ARG4)
    BLAS1_EX_TESTING(dotc_ex, ARG4)
    BLAS1_EX_TESTING(dotc_batched_ex, ARG4)
    BLAS1_EX_TESTING(dotc_strided_batched_ex, ARG4)

} // namespace
