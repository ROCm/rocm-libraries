/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include <string>

namespace
{
    template <typename...>
    struct testing_set_get_atomics_mode : rocblas_test_valid
    {
        void operator()(const Arguments&)
        {
            rocblas_handle handle;
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

            // Make sure the default atomics_mode is rocblas_atomics_allowed
            rocblas_atomics_mode mode = rocblas_atomics_allowed;
            CHECK_ROCBLAS_ERROR(rocblas_get_atomics_mode(handle, &mode));
            EXPECT_EQ(rocblas_atomics_not_allowed, mode);

            // Make sure set()/get() functions work
            CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_allowed));
            CHECK_ROCBLAS_ERROR(rocblas_get_atomics_mode(handle, &mode));
            EXPECT_EQ(rocblas_atomics_allowed, mode);

            CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_not_allowed));
            CHECK_ROCBLAS_ERROR(rocblas_get_atomics_mode(handle, &mode));
            EXPECT_EQ(rocblas_atomics_not_allowed, mode);

            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
        }
    };

    struct set_get_atomics_mode : RocBLAS_Test<set_get_atomics_mode, testing_set_get_atomics_mode>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_atomics_mode");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<set_get_atomics_mode>(arg.name);
        }
    };

    TEST_P(set_get_atomics_mode, auxiliary_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(testing_set_get_atomics_mode<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_atomics_mode)

} // namespace
