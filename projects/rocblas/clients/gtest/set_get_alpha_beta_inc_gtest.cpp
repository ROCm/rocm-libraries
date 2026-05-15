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
    struct testing_set_get_alpha_beta_inc : rocblas_test_valid
    {
        void operator()(const Arguments&)
        {
            rocblas_handle handle;
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

            rocblas_int alpha_inc = -1;
            rocblas_int beta_inc  = -1;
            CHECK_ROCBLAS_ERROR(rocblas_get_alpha_inc(handle, &alpha_inc));
            CHECK_ROCBLAS_ERROR(rocblas_get_beta_inc(handle, &beta_inc));
            EXPECT_EQ(0, alpha_inc);
            EXPECT_EQ(0, beta_inc);

            CHECK_ROCBLAS_ERROR(rocblas_set_alpha_inc(handle, 7));
            CHECK_ROCBLAS_ERROR(rocblas_get_alpha_inc(handle, &alpha_inc));
            EXPECT_EQ(7, alpha_inc);
            CHECK_ROCBLAS_ERROR(rocblas_get_beta_inc(handle, &beta_inc));
            EXPECT_EQ(0, beta_inc);

            CHECK_ROCBLAS_ERROR(rocblas_set_beta_inc(handle, 11));
            CHECK_ROCBLAS_ERROR(rocblas_get_beta_inc(handle, &beta_inc));
            EXPECT_EQ(11, beta_inc);
            CHECK_ROCBLAS_ERROR(rocblas_get_alpha_inc(handle, &alpha_inc));
            EXPECT_EQ(7, alpha_inc);

            CHECK_ROCBLAS_ERROR(rocblas_set_alpha_inc(handle, 0));
            CHECK_ROCBLAS_ERROR(rocblas_set_beta_inc(handle, 0));
            CHECK_ROCBLAS_ERROR(rocblas_get_alpha_inc(handle, &alpha_inc));
            CHECK_ROCBLAS_ERROR(rocblas_get_beta_inc(handle, &beta_inc));
            EXPECT_EQ(0, alpha_inc);
            EXPECT_EQ(0, beta_inc);

            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
        }
    };

    struct set_get_alpha_beta_inc
        : RocBLAS_Test<set_get_alpha_beta_inc, testing_set_get_alpha_beta_inc>
    {
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_alpha_beta_inc");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<set_get_alpha_beta_inc>(arg.name);
        }
    };

    TEST_P(set_get_alpha_beta_inc, auxiliary_tensile)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(testing_set_get_alpha_beta_inc<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_alpha_beta_inc)

} // namespace
