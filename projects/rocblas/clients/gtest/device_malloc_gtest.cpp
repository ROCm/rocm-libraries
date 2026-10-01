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

#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cstring>

// rocblas_device_malloc_alloc is the public entry point rocSOLVER and hipSOLVER reach
// device memory through, and the only caller of the count form of _device_malloc. On the
// stream-order path that constructor pushed one null pointer per requested count but left
// its success flag at the true it was initialised with, so the alloc returned
// rocblas_status_success holding nothing and a caller that checked the status went on to
// use a null pointer.
//
// Driven through the public API, since the flag only ever mattered through what
// rocblas_device_malloc_alloc did with it.
namespace
{
    // Too large for any device, but small enough that rounding up to the allocation
    // granularity cannot overflow size_t and wrap to something allocatable.
    constexpr size_t c_unsatisfiable_bytes = size_t(1) << 50;

    void testing_device_malloc_count_alloc_failure(const Arguments& arg)
    {
#if HIP_VERSION < 50300000
        GTEST_SKIP() << "hipMallocAsync on the default stream needs HIP 5.3, and the branch "
                        "this covers is compiled out below it";
#else
        // graph_test makes rocblas_local_handle set ROCBLAS_STREAM_ORDER_ALLOC around
        // rocblas_create_handle, and only a handle built that way has a destructor matching
        // its constructor. Calling set_stream_order_memory_allocation afterwards reaches the
        // same branch but pairs a hipMalloc'd workspace with hipFreeAsync; where that errors
        // the destructor calls rocblas_abort, which resets SIGABRT, so gtest could not turn
        // it into a failed test and the whole binary would go down. No graph capture happens
        // here -- that is pre_test's doing and this test never calls it.
        //
        // Asserted on the argument because the flag the branch reads,
        // _rocblas_handle::stream_order_alloc, is private with no getter. Without it, losing
        // the YAML line would silently drop the test onto the other allocator, where it
        // passes against an unfixed library.
        ASSERT_TRUE(arg.graph_test)
            << "this test needs a handle built with stream-order allocation; restore "
               "graph_test: true in device_malloc_gtest.yaml";

        // Probed before the handle is built: graph_test makes the constructor call
        // hipMallocAsync, which throws where memory pools are unsupported, and the harness
        // reports that as an uncaught exception -- an absent feature reading as a defect.
        int device          = 0;
        int pools_supported = 0;
        CHECK_HIP_ERROR(hipGetDevice(&device));
        CHECK_HIP_ERROR(hipDeviceGetAttribute(
            &pools_supported, hipDeviceAttributeMemoryPoolsSupported, device));
        if(!pools_supported)
            GTEST_SKIP() << "device " << device
                         << " does not support memory pools, so stream-order allocation "
                            "-- the branch this covers -- cannot be exercised here";

        rocblas_local_handle handle{arg};

        // The other half of the branch condition. A handle given a user workspace allocates
        // from it instead, on a path that sets the flag correctly and always has, so the
        // test would return rocblas_status_memory_error for an unrelated reason and stay
        // green while covering none of the fix. Skipped rather than failed because a
        // non-zero ROCBLAS_DEVICE_MEMORY_SIZE is a property of the run, not a regression.
        if(!rocblas_is_managing_device_memory(handle))
            GTEST_SKIP() << "handle is not managing its own device memory, so the count "
                            "allocation under test is not the one that runs; unset "
                            "ROCBLAS_DEVICE_MEMORY_SIZE to cover it";

        rocblas_device_malloc_base* mem = nullptr;

        rocblas_status status = rocblas_device_malloc_alloc(handle, &mem, 1, c_unsatisfiable_bytes);

        // The status is the whole point: before the fix this was rocblas_status_success.
        EXPECT_EQ(status, rocblas_status_memory_error)
            << "an unsatisfiable device allocation reported " << rocblas_status_to_string(status)
            << "; a caller that trusts the status would go on to use the null pointer it was "
               "handed";

        // Checked independently of the status: the defect produced a usable-looking object
        // alongside its success, and a caller holding one has nothing else to test.
        EXPECT_EQ(mem, nullptr) << "a failed allocation still produced an allocation object";

        // Unreachable once the fix is in place; present so a run against an unfixed library
        // does not also leak the object that failure handed back.
        if(mem)
            EXPECT_ROCBLAS_STATUS(rocblas_device_malloc_free(mem), rocblas_status_success);
#endif
    }

    template <typename...>
    struct device_malloc_testing : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "device_malloc_count_alloc_failure"))
                testing_device_malloc_count_alloc_failure(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct device_malloc : RocBLAS_Test<device_malloc, device_malloc_testing>
    {
        // Nothing here varies with the type, the allocation being counted in bytes.
        // Dispatched anyway rather than answering true, because type_filter_functor also
        // applies the global filters -- os_flags, gpu_arch, the no-Tensile exclusions.
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<type_filter_functor>(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "device_malloc_count_alloc_failure");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<device_malloc>(arg.name);
        }
    };

    TEST_P(device_malloc, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(device_malloc_testing<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(device_malloc)

} // namespace
