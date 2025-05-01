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

#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "testing_set_get_matrix.hpp"
#include "testing_set_get_matrix_async.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    enum sync_type
    {
        SET_GET_MATRIX,
        SET_GET_MATRIX_ASYNC,
    };

    template <template <typename...> class FILTER, sync_type TRANSFER_TYPE>
    struct matrix_set_get_template
        : RocBLAS_Test<matrix_set_get_template<FILTER, TRANSFER_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<matrix_set_get_template::template type_filter_functor>(
                arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRANSFER_TYPE)
            {
            case SET_GET_MATRIX:
                return !strcmp(arg.function, "set_get_matrix");
            case SET_GET_MATRIX_ASYNC:
                return !strcmp(arg.function, "set_get_matrix_async");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<matrix_set_get_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << arg.M << '_' << arg.N << '_' << arg.lda << '_' << arg.ldb << '_' << arg.ldd;
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

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct set_get_matrix_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct set_get_matrix_testing<T, std::enable_if_t<!std::is_same_v<T, void>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "set_get_matrix"))
                testing_set_get_matrix<T>(arg);
            else if(!strcmp(arg.function, "set_get_matrix_async"))
                testing_set_get_matrix_async<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using set_get_matrix = matrix_set_get_template<set_get_matrix_testing, SET_GET_MATRIX>;
    TEST_P(set_get_matrix, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_matrix_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_matrix);

    using set_get_matrix_async
        = matrix_set_get_template<set_get_matrix_testing, SET_GET_MATRIX_ASYNC>;
    TEST_P(set_get_matrix_async, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_matrix_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_matrix_async);

} // namespace
