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
#include "testing_set_get_vector.hpp"
#include "testing_set_get_vector_async.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <type_traits>

namespace
{
    enum sync_type
    {
        SET_GET_VECTOR,
        SET_GET_VECTOR_ASYNC,
    };

    template <template <typename...> class FILTER, sync_type TRANSFER_TYPE>
    struct vec_set_get_template : RocBLAS_Test<vec_set_get_template<FILTER, TRANSFER_TYPE>, FILTER>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<vec_set_get_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRANSFER_TYPE)
            {
            case SET_GET_VECTOR:
                return !strcmp(arg.function, "set_get_vector");
            case SET_GET_VECTOR_ASYNC:
                return !strcmp(arg.function, "set_get_vector_async");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<vec_set_get_template> name(arg.name);

            name << rocblas_datatype2string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << arg.N << '_' << arg.incx << '_' << arg.incy << '_' << arg.ldd;
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
    struct set_get_vector_testing : rocblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct set_get_vector_testing<
        T,
        std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "set_get_vector"))
                testing_set_get_vector<T>(arg);
            else if(!strcmp(arg.function, "set_get_vector_async"))
                testing_set_get_vector_async<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using set_get_vector = vec_set_get_template<set_get_vector_testing, SET_GET_VECTOR>;
    TEST_P(set_get_vector, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_vector_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_vector);

    using set_get_vector_async = vec_set_get_template<set_get_vector_testing, SET_GET_VECTOR_ASYNC>;
    TEST_P(set_get_vector_async, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<set_get_vector_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_vector_async);

} // namespace
