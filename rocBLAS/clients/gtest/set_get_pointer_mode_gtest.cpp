/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    template <typename...>
    struct testing_set_get_pointer_mode : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            rocblas_pointer_mode mode = rocblas_pointer_mode_device;
            rocblas_local_handle handle{arg};
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(rocblas_get_pointer_mode(handle, &mode));
            EXPECT_EQ(rocblas_pointer_mode_device, mode);
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR(rocblas_get_pointer_mode(handle, &mode));
            EXPECT_EQ(rocblas_pointer_mode_host, mode);
        }
    };

    struct set_get_pointer_mode : RocBLAS_Test<set_get_pointer_mode, testing_set_get_pointer_mode>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_pointer_mode");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<set_get_pointer_mode>(arg.name);
        }
    };

    TEST_P(set_get_pointer_mode, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(testing_set_get_pointer_mode<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_pointer_mode)

} // namespace
