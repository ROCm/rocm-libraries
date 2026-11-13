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
#include "rocblas_vector.hpp"
#include "testing_kernel_select.hpp"
#include <memory>
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

            // Declared before the handles, so on every exit path, including a failed check, the
            // handles are destroyed first and the variables are restored after.
            kernel_select::scoped_env env_double(env[0], "1");
            kernel_select::scoped_env env_single(env[1], "1");
            rocblas_local_handle      on;
            setenv(env[0], "0", true);
            setenv(env[1], "0", true);
            rocblas_local_handle off;
            setenv(env[0], "true", true); // only "1" requests emulation
            setenv(env[1], "1", true);
            rocblas_local_handle other;

            // the variables are only read by a rocBLAS built with emulation support
            const bool           read = rocblas_internal_fp_emulation_supported();
            const rocblas_handle h_on = on, h_off = off, h_other = other;
            EXPECT_EQ(read, h_on->emulate_double_precision);
            EXPECT_EQ(read, h_on->emulate_single_precision);
            EXPECT_FALSE(h_off->emulate_double_precision);
            EXPECT_FALSE(h_off->emulate_single_precision);
            EXPECT_FALSE(h_other->emulate_double_precision);
            EXPECT_EQ(read, h_other->emulate_single_precision);
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

namespace
{
    // Operands for an n x n x n f64_r gemm_ex: small integers, so every result is exact.
    struct f64_operands
    {
        device_vector<double> A, B, C;

        explicit f64_operands(rocblas_int n)
            : A(size_t(n) * n)
            , B(size_t(n) * n)
            , C(size_t(n) * n)
        {
        }

        hipError_t init(rocblas_int n)
        {
            const size_t        size = size_t(n) * n;
            host_vector<double> hA(size), hB(size);
            for(size_t i = 0; i < size; i++)
            {
                hA[i] = double(i % 7) - 3;
                hB[i] = double(i % 5) - 2;
            }
            for(hipError_t status : {A.memcheck(), B.memcheck(), C.memcheck()})
                if(status != hipSuccess)
                    return status;
            hipError_t status = A.transfer_from(hA);
            return status != hipSuccess ? status : B.transfer_from(hB);
        }
    };

    rocblas_status f64_gemm_ex(rocblas_handle handle, rocblas_int n, f64_operands& x)
    {
        const double alpha = 1, beta = 0;
        return rocblas_gemm_ex(handle,
                               rocblas_operation_none,
                               rocblas_operation_none,
                               n,
                               n,
                               n,
                               &alpha,
                               x.A,
                               rocblas_datatype_f64_r,
                               n,
                               x.B,
                               rocblas_datatype_f64_r,
                               n,
                               &beta,
                               x.C,
                               rocblas_datatype_f64_r,
                               n,
                               x.C,
                               rocblas_datatype_f64_r,
                               n,
                               rocblas_datatype_f64_r,
                               rocblas_gemm_algo_standard,
                               0,
                               0);
    }

    // The workspace hipBLASLt asks for to emulate an n^3 f64_r GEMM with the eager strategy, as the
    // rocBLAS library under test reports it; 0 where it does not emulate.
    rocblas_status eager_f64_emulation_workspace(rocblas_int n, size_t& workspace)
    {
        rocblas_local_handle handle;
        return rocblas_internal_fp_emulation_workspace_size(
            handle, rocblas_datatype_f64_r, n, n, n, rocblas_emulation_strategy_eager, &workspace);
    }

    // With emulation requested, a 4096^3 f64_r gemm_ex grows the handle's device memory pool to
    // the workspace emulation asks for, and destroying the handle frees it again. Without
    // emulation the pool keeps its size.
    template <typename...>
    struct testing_emulation_workspace_growth : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            const rocblas_int n = 4096;

            // Skip when the environment would change whether the pool grows
            auto env_nonzero = [](const char* name) {
                const char* value = getenv(name);
                return value && strtoull(value, nullptr, 0);
            };
            const char* use_hipblaslt   = getenv("ROCBLAS_USE_HIPBLASLT");
            const char* hipblas_emulate = getenv("HIPBLAS_EMULATE_DOUBLE_PRECISION");
            const char* rocblas_emulate = getenv("ROCBLAS_EMULATE_DOUBLE_PRECISION");
            if(env_nonzero("ROCBLAS_DEVICE_MEMORY_SIZE"))
                GTEST_SKIP() << "ROCBLAS_DEVICE_MEMORY_SIZE makes the pool user-managed";
            if(env_nonzero("ROCBLAS_STREAM_ORDER_ALLOC"))
                GTEST_SKIP() << "ROCBLAS_STREAM_ORDER_ALLOC allocates per call, not from the pool";
            if(arg.emulation > 0 && getenv("ROCBLAS_EMULATION_WORKSPACE_MAX_BYTES"))
                GTEST_SKIP() << "ROCBLAS_EMULATION_WORKSPACE_MAX_BYTES limits pool growth";
            if(arg.emulation > 0 && use_hipblaslt && strncmp(use_hipblaslt, "1", 1) != 0)
                GTEST_SKIP() << "ROCBLAS_USE_HIPBLASLT turns hipBLASLt off";
            if(arg.emulation <= 0 && hipblas_emulate && !strcmp(hipblas_emulate, "1"))
                GTEST_SKIP() << "HIPBLAS_EMULATE_DOUBLE_PRECISION emulates without rocBLAS";
            if(arg.emulation <= 0 && rocblas_emulate && !strcmp(rocblas_emulate, "1"))
                GTEST_SKIP() << "ROCBLAS_EMULATE_DOUBLE_PRECISION requests emulation";

            if(arg.emulation > 0)
            {
                // Skip devices hipBLASLt does not emulate on: the query returns 0 there
                size_t workspace = 0;
                ASSERT_EQ(rocblas_status_success, eager_f64_emulation_workspace(n, workspace));
                if(!workspace)
                    GTEST_SKIP() << "rocBLAS reports no f64_r emulation workspace for this handle "
                                    "and device";
            }

            f64_operands x(n);
            CHECK_DEVICE_ALLOCATION(x.init(n));

            size_t pool_before = 0, pool_after = 0, free_in_use = 0, free_freed = 0, total = 0;
            {
                rocblas_local_handle handle{arg};
                CHECK_ROCBLAS_ERROR(rocblas_get_device_memory_size(handle, &pool_before));
                CHECK_ROCBLAS_ERROR(f64_gemm_ex(handle, n, x));
                CHECK_HIP_ERROR(hipDeviceSynchronize());
                CHECK_ROCBLAS_ERROR(rocblas_get_device_memory_size(handle, &pool_after));
                CHECK_HIP_ERROR(hipMemGetInfo(&free_in_use, &total));
            }
            CHECK_HIP_ERROR(hipMemGetInfo(&free_freed, &total));

            if(arg.emulation > 0)
            {
                EXPECT_GT(pool_after, pool_before);
                // Destroying the handle frees the whole pool, growth included. hipMemGetInfo is
                // device-wide, so another process allocating meanwhile could fail this check.
                EXPECT_GE(free_freed, free_in_use + pool_after);
            }
            else
            {
                EXPECT_EQ(pool_after, pool_before);
            }
        }
    };

    struct emulation_workspace_growth
        : RocBLAS_Test<emulation_workspace_growth, testing_emulation_workspace_growth>
    {
        // Apply gpu_arch and the skip for builds without emulation support
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_client_global_filters(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "emulation_workspace_growth");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<emulation_workspace_growth>(arg.name) << arg.emulation;
        }
    };

    TEST_P(emulation_workspace_growth, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            testing_emulation_workspace_growth<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(emulation_workspace_growth)

    // A user workspace whose size is not a multiple of 64 bytes, below what emulation asks for.
    // rocBLAS must offer hipBLASLt a multiple of 64 bytes; otherwise emulation takes the whole
    // remainder, rocBLAS cannot allocate it (allocations round up to 64 bytes), and the GEMM
    // silently runs on another backend. The kernel-select trace checks that the GEMM ran on
    // hipBLASLt with no fallback; it does not show whether hipBLASLt emulated.
    template <typename...>
    struct testing_emulation_unaligned_user_workspace : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            const char* use_hipblaslt = getenv("ROCBLAS_USE_HIPBLASLT");
            const char* stream_order  = getenv("ROCBLAS_STREAM_ORDER_ALLOC");
            if(use_hipblaslt && strncmp(use_hipblaslt, "1", 1) != 0)
                GTEST_SKIP() << "ROCBLAS_USE_HIPBLASLT turns hipBLASLt off";
            if(stream_order && strtoull(stream_order, nullptr, 0))
                GTEST_SKIP() << "ROCBLAS_STREAM_ORDER_ALLOC allocates per call, not from the pool";

            const rocblas_int n         = 4096;
            size_t            workspace = 0;
            ASSERT_EQ(rocblas_status_success, eager_f64_emulation_workspace(n, workspace));
            if(workspace < 128)
                GTEST_SKIP()
                    << "rocBLAS reports no f64_r emulation workspace for this handle and device";

            // Below the request, and 63 bytes past a multiple of 64 whatever the request is
            const size_t pool = (workspace & ~size_t{63}) - 1;

            f64_operands x(n);
            CHECK_DEVICE_ALLOCATION(x.init(n));
            void* buffer = nullptr;
            CHECK_HIP_ERROR((hipMalloc)(&buffer, pool));
            const std::unique_ptr<void, hipError_t (*)(void*)> buffer_owner(buffer, hipFree);

            const kernel_select::trace t
                = kernel_select::capture(arg, "0x10", [&](rocblas_handle handle) {
                      CHECK_ROCBLAS_ERROR(rocblas_set_workspace(handle, buffer, pool));
                      CHECK_ROCBLAS_ERROR(f64_gemm_ex(handle, n, x));
                      CHECK_HIP_ERROR(hipDeviceSynchronize());
                      CHECK_ROCBLAS_ERROR(rocblas_set_workspace(handle, nullptr, 0));
                  });

            ASSERT_EQ(t.lines.size(), size_t{1}) << t.text;
            EXPECT_TRUE(kernel_select::contains(t.lines[0], "# source=hipblaslt ")) << t.lines[0];
            EXPECT_TRUE(kernel_select::contains(t.lines[0], " fallback_from=none ")) << t.lines[0];
        }
    };

    struct emulation_unaligned_user_workspace
        : RocBLAS_Test<emulation_unaligned_user_workspace,
                       testing_emulation_unaligned_user_workspace>
    {
        // Apply gpu_arch and the skip for builds without emulation support
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_client_global_filters(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "emulation_unaligned_user_workspace");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<emulation_unaligned_user_workspace>(arg.name);
        }
    };

    TEST_P(emulation_unaligned_user_workspace, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            testing_emulation_unaligned_user_workspace<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(emulation_unaligned_user_workspace)

} // namespace
