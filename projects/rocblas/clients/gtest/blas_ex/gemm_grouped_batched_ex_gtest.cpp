/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "blas_ex/common_gemm_ex.hpp"
#include "blas_ex/testing_gemm_grouped_batched_ex.hpp"
#include "client_utility.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    template <template <typename...> class FILTER>
    struct gemm_grouped_batched_ex_test_template
        : RocBLAS_Test<gemm_grouped_batched_ex_test_template<FILTER>, FILTER>
    {
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_gemm_dispatch<
                gemm_grouped_batched_ex_test_template::template type_filter_functor>(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "gemm_grouped_batched_ex")
                   || !strcmp(arg.function, "gemm_grouped_batched_ex_bad_arg");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<gemm_grouped_batched_ex_test_template> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << rocblas_datatype2string(arg.b_type) << rocblas_datatype2string(arg.c_type)
                     << rocblas_datatype2string(arg.d_type)
                     << rocblas_datatype2string(arg.compute_type);

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);
                name << '_' << arg.M << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                     << arg.lda << '_' << arg.ldb << '_' << arg.beta << '_' << arg.ldc << '_'
                     << arg.ldd << '_' << arg.batch_count;
            }

            name << '_' << arg.stride_x;

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

    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct gemm_grouped_batched_ex_testing : rocblas_test_invalid
    {
    };

    template <typename Ti, typename To, typename Tc>
    struct gemm_grouped_batched_ex_testing<
        Ti,
        To,
        Tc,
        std::enable_if_t<
            !std::is_same_v<
                Ti,
                void> && !(std::is_same_v<Ti, Tc> && std::is_same_v<Ti, rocblas_bfloat16>)>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemm_grouped_batched_ex"))
                testing_gemm_grouped_batched_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_grouped_batched_ex_bad_arg"))
                testing_gemm_grouped_batched_ex_bad_arg<Ti, To, Tc>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemm_grouped_batched_ex
        = gemm_grouped_batched_ex_test_template<gemm_grouped_batched_ex_testing>;
    TEST_P(gemm_grouped_batched_ex, blas3_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_gemm_dispatch<gemm_grouped_batched_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_grouped_batched_ex);

} // namespace
