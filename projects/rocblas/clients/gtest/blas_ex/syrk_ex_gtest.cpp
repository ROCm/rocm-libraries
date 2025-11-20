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

#include "blas_ex/common_syrk_ex.hpp"
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
    // Types of SYRK tests
    enum syrk_ex_test_type
    {
        SYRK_EX,
        SYRK_BATCHED_EX,
        SYRK_STRIDED_BATCHED_EX,
    };

    // ----------------------------------------------------------------------------
    // SYRK testing template
    // ----------------------------------------------------------------------------
    // The first template parameter is a class template which determines which
    // combination of types applies to this test, and for those that do, instantiates
    // the test code based on the function named in the test Arguments. The second
    // template parameter is an enum which allows the different flavors of SYRK to
    // be differentiated.
    //
    // The RocBLAS_Test base class takes this class (CRTP) and the first template
    // parameter as arguments, and provides common types such as type_filter_functor,
    // and derives from the Google Test parameterized base classes.
    //
    // This class defines functions for filtering the types and function names which
    // apply to this test, and for generating the suffix of the Google Test name
    // corresponding to each instance of this test.
    template <template <typename...> class FILTER, syrk_ex_test_type SYRK_TYPE>
    struct syrk_ex_test_template : RocBLAS_Test<syrk_ex_test_template<FILTER, SYRK_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_syrk_ex_dispatch<syrk_ex_test_template::template type_filter_functor>(
                arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SYRK_TYPE)
            {
            case SYRK_EX:
                return !strcmp(arg.function, "syrk_ex") || !strcmp(arg.function, "syrk_ex_bad_arg");

            case SYRK_BATCHED_EX:
                return !strcmp(arg.function, "syrk_batched_ex")
                       || !strcmp(arg.function, "syrk_batched_ex_bad_arg");

            case SYRK_STRIDED_BATCHED_EX:
                return !strcmp(arg.function, "syrk_strided_batched_ex")
                       || !strcmp(arg.function, "syrk_strided_batched_ex_bad_arg");
            }

            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<syrk_ex_test_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type) << rocblas_datatype2string(arg.c_type)
                 << rocblas_datatype2string(arg.compute_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                constexpr bool isBatched = (SYRK_TYPE != SYRK_EX);

                name << '_' << (char)std::toupper(arg.uplo) << (char)std::toupper(arg.transA);

                name << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_' << arg.lda << '_'
                     << arg.beta << '_' << arg.ldc;

                if(isBatched)
                    name << '_' << arg.batch_count;

                if(SYRK_TYPE == SYRK_STRIDED_BATCHED_EX)
                    name << '_' << arg.stride_a << '_' << arg.stride_c;
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

    // ----------------------------------------------------------------------------
    // syrk_ex
    // syrk_batched_ex
    // syrk_strided_batched_ex
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct syrk_ex_testing : rocblas_test_invalid
    {
    };

    // When Ti != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename Ti, typename To, typename Tc>
    struct syrk_ex_testing<
        Ti,
        To,
        Tc,
        std::enable_if_t<
            !std::is_same_v<
                Ti,
                void> && !(std::is_same_v<Tc, rocblas_bfloat16> || std::is_same_v<Tc, rocblas_half>)>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "syrk_ex"))
                testing_syrk_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "syrk_ex_bad_arg"))
                testing_syrk_ex_bad_arg<Ti, To, Tc>(arg);
            // else if(!strcmp(arg.function, "syrk_batched_ex"))
            //     testing_syrk_batched_ex<Ti, To, Tc>(arg);
            // else if(!strcmp(arg.function, "syrk_batched_ex_bad_arg"))
            //     testing_syrk_batched_ex_bad_arg<Ti, To, Tc>(arg);
            // else if(!strcmp(arg.function, "syrk_strided_batched_ex"))
            //     testing_syrk_strided_batched_ex<Ti, To, Tc>(arg);
            // else if(!strcmp(arg.function, "syrk_strided_batched_ex_bad_arg"))
            //     testing_syrk_strided_batched_ex_bad_arg<Ti, To, Tc>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using syrk_ex = syrk_ex_test_template<syrk_ex_testing, SYRK_EX>;
    TEST_P(syrk_ex, blas3_tensile)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocblas_syrk_ex_dispatch<syrk_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syrk_ex);

    // using syrk_batched_ex = syrk_ex_test_template<syrk_ex_testing, SYRK_BATCHED_EX>;
    // TEST_P(syrk_batched_ex, blas3_tensile)
    // {
    //     CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
    //         rocblas_syrk_ex_dispatch<syrk_ex_testing>(GetParam()));
    // }
    // INSTANTIATE_TEST_CATEGORIES(syrk_batched_ex);

    // using syrk_strided_batched_ex = syrk_ex_test_template<syrk_ex_testing, SYRK_STRIDED_BATCHED_EX>;
    // TEST_P(syrk_strided_batched_ex, blas3_tensile)
    // {
    //     CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
    //         rocblas_syrk_ex_dispatch<syrk_ex_testing>(GetParam()));
    // }
    // INSTANTIATE_TEST_CATEGORIES(syrk_strided_batched_ex);

} // namespace
