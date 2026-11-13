/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../library/src/include/handle.hpp"
#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include <string>

namespace
{
    template <typename...>
    struct testing_set_get_emulation_strategy : rocblas_test_valid
    {
        void operator()(const Arguments&)
        {
            rocblas_handle handle;
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

            // A new handle uses rocblas_emulation_strategy_default
            rocblas_emulation_strategy strategy = rocblas_emulation_strategy_eager;
            CHECK_ROCBLAS_ERROR(rocblas_get_emulation_strategy(handle, &strategy));
            EXPECT_EQ(rocblas_emulation_strategy_default, strategy);

            // Every strategy can be set and read back
            for(rocblas_emulation_strategy s : {rocblas_emulation_strategy_performant,
                                                rocblas_emulation_strategy_eager,
                                                rocblas_emulation_strategy_default})
            {
                CHECK_ROCBLAS_ERROR(rocblas_set_emulation_strategy(handle, s));
                CHECK_ROCBLAS_ERROR(rocblas_get_emulation_strategy(handle, &strategy));
                EXPECT_EQ(s, strategy);
            }

            // An invalid value is rejected and leaves the handle unchanged
            CHECK_ROCBLAS_ERROR(
                rocblas_set_emulation_strategy(handle, rocblas_emulation_strategy_eager));
            EXPECT_ROCBLAS_STATUS(
                rocblas_set_emulation_strategy(handle, static_cast<rocblas_emulation_strategy>(3)),
                rocblas_status_invalid_value);
            CHECK_ROCBLAS_ERROR(rocblas_get_emulation_strategy(handle, &strategy));
            EXPECT_EQ(rocblas_emulation_strategy_eager, strategy);

            // Invalid handle and pointer
            EXPECT_ROCBLAS_STATUS(
                rocblas_set_emulation_strategy(nullptr, rocblas_emulation_strategy_eager),
                rocblas_status_invalid_handle);
            EXPECT_ROCBLAS_STATUS(rocblas_get_emulation_strategy(nullptr, &strategy),
                                  rocblas_status_invalid_handle);
            EXPECT_ROCBLAS_STATUS(rocblas_get_emulation_strategy(handle, nullptr),
                                  rocblas_status_invalid_pointer);

            // The strategy belongs to the handle
            rocblas_handle other;
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&other));
            CHECK_ROCBLAS_ERROR(rocblas_get_emulation_strategy(other, &strategy));
            EXPECT_EQ(rocblas_emulation_strategy_default, strategy);
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(other));

            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
        }
    };

    struct set_get_emulation_strategy
        : RocBLAS_Test<set_get_emulation_strategy, testing_set_get_emulation_strategy>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_emulation_strategy");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<set_get_emulation_strategy>(arg.name);
        }
    };

    TEST_P(set_get_emulation_strategy, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            testing_set_get_emulation_strategy<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_emulation_strategy)

} // namespace

namespace
{
    // ROCBLAS_EMULATE_DOUBLE_PRECISION / ROCBLAS_EMULATE_SINGLE_PRECISION are read when each
    // handle is created, so handles created with different values keep their own setting.
    template <typename...>
    struct testing_emulation_env_per_handle : rocblas_test_valid
    {
        void operator()(const Arguments&)
        {
            static const char* const env[]
                = {"ROCBLAS_EMULATE_DOUBLE_PRECISION", "ROCBLAS_EMULATE_SINGLE_PRECISION"};
            std::string saved[2];
            bool        existed[2] = {false, false};
            for(int i = 0; i < 2; i++)
                if(const char* v = getenv(env[i]))
                {
                    saved[i]   = v;
                    existed[i] = true;
                }

            rocblas_handle on, off, other;
            setenv(env[0], "1", true);
            setenv(env[1], "1", true);
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&on));
            setenv(env[0], "0", true);
            setenv(env[1], "0", true);
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&off));
            setenv(env[0], "true", true); // only "1" requests emulation
            setenv(env[1], "1", true);
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&other));

            for(int i = 0; i < 2; i++)
            {
                if(existed[i])
                    setenv(env[i], saved[i].c_str(), true);
                else
                    unsetenv(env[i]);
            }

#ifdef BUILD_WITH_HIPBLASLT
            const bool read = true;
#else
            const bool read = false; // the variables only matter for the hipBLASLt backend
#endif
            EXPECT_EQ(read, on->emulate_double_precision);
            EXPECT_EQ(read, on->emulate_single_precision);
            EXPECT_FALSE(off->emulate_double_precision);
            EXPECT_FALSE(off->emulate_single_precision);
            EXPECT_FALSE(other->emulate_double_precision);
            EXPECT_EQ(read, other->emulate_single_precision);

            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(other));
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(off));
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(on));
        }
    };

    struct emulation_env_per_handle
        : RocBLAS_Test<emulation_env_per_handle, testing_emulation_env_per_handle>
    {
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "emulation_env_per_handle");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<emulation_env_per_handle>(arg.name);
        }
    };

    TEST_P(emulation_env_per_handle, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(testing_emulation_env_per_handle<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(emulation_env_per_handle)

} // namespace
