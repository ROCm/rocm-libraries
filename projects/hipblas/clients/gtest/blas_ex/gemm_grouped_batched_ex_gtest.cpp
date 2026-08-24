/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "blas_ex/testing_gemm_grouped_batched_ex.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    template <template <typename...> class FILTER>
    struct gemm_grouped_batched_ex_test_template
        : HipBLAS_Test<gemm_grouped_batched_ex_test_template<FILTER>, FILTER>
    {
        template <typename... T>
        struct type_filter_functor
        {
            bool operator()(const Arguments& args)
            {
                if(!hipblas_client_global_filters(args))
                    return false;
                return static_cast<bool>(FILTER<T...>{});
            }
        };

        static bool type_filter(const Arguments& arg)
        {
            return hipblas_gemm_dispatch<
                gemm_grouped_batched_ex_test_template::template type_filter_functor>(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "gemm_grouped_batched_ex")
                   || !strcmp(arg.function, "gemm_grouped_batched_ex_bad_arg");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            testname_gemm_grouped_batched_ex(arg, name);
            return std::move(name);
        }
    };

    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct gemm_grouped_batched_ex_testing : hipblas_test_invalid
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
                void> && !(std::is_same_v<Ti, Tc> && std::is_same_v<Ti, hipblasBfloat16>)>>
        : hipblas_test_valid
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
    TEST_P(gemm_grouped_batched_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_gemm_dispatch<gemm_grouped_batched_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_grouped_batched_ex);

} // namespace
