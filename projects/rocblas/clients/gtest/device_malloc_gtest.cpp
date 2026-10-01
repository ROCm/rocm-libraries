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
// device memory through, and it is the only caller of the count form of _device_malloc.
// On the stream-order path that constructor compared hipMallocAsync against hipSuccess,
// pushed one null pointer per requested count, and left its success flag at the true it
// was initialised with. rocblas_device_malloc_alloc read that flag, concluded the
// allocation had worked, took mem[0] as a base address to compute the rest from, and
// returned rocblas_status_success holding nothing. A caller that checked the status it
// was handed was told to go ahead and use a null pointer.
//
// Driven through the public API rather than the constructor, because the flag only ever
// mattered through what rocblas_device_malloc_alloc did with it.
namespace
{
    // Large enough that no device satisfies it, small enough that rounding it up to the
    // device memory granularity cannot overflow size_t. An overflowing size would wrap to
    // something allocatable and the test would pass having proved nothing.
    constexpr size_t c_unsatisfiable_bytes = size_t(1) << 50;

    void testing_device_malloc_count_alloc_failure(const Arguments& arg)
    {
#if HIP_VERSION < 50300000
        GTEST_SKIP() << "hipMallocAsync on the default stream needs HIP 5.3, and the branch "
                        "this covers is compiled out below it";
#else
        // graph_test is what makes rocblas_local_handle export ROCBLAS_STREAM_ORDER_ALLOC
        // before rocblas_create_handle and restore it afterwards, and a handle built that
        // way is the only one whose destructor matches its constructor. Calling
        // set_stream_order_memory_allocation on an already-built handle would reach the
        // same branch, but the handle allocated its workspace with hipMalloc and the
        // destructor would then release it with hipFreeAsync; on any runtime where that
        // returns an error the destructor calls rocblas_abort, which resets SIGABRT before
        // aborting, so gtest could not turn it into a failed test and the whole
        // rocblas-test binary would go down. rocblas_stream_begin_capture carries that same
        // setter call commented out in favour of the environment, for the same reason.
        //
        // No graph capture happens: that is pre_test's doing and this test never calls it.
        ASSERT_TRUE(arg.graph_test)
            << "this test needs a handle built with stream-order allocation; restore "
               "graph_test: true in device_malloc_gtest.yaml";

        rocblas_local_handle handle{arg};

        // The other half of the branch condition, and asserted rather than assumed: a
        // handle given a user workspace allocates from it instead, and that path sets the
        // same flag correctly and always has. This test would then return
        // rocblas_status_memory_error for a reason that has nothing to do with the fix and
        // stay green while covering none of it.
        //
        // Skipped rather than failed, because that is a property of the run and not of the
        // library: a non-zero ROCBLAS_DEVICE_MEMORY_SIZE in the environment makes every
        // handle in the process user managed, and failing there would report a configured
        // run as a regression. A skip still says so out loud, which a silent pass does not.
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

        // Checked independently of the status, because the defect produced a usable-looking
        // object alongside its success and a caller holding one has nothing else to test.
        EXPECT_EQ(mem, nullptr) << "a failed allocation still produced an allocation object";

        // Not reached once the fix is in place. Present so that running this test against an
        // unfixed library does not also leak the object that failure used to hand back.
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
        // Nothing here varies with the type, the allocation being counted in bytes, so
        // device_malloc_testing is valid for all of them. Dispatched anyway rather than
        // answering true, because type_filter_functor also applies the global filters --
        // os_flags, gpu_arch, the no-Tensile exclusions -- and those do apply.
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
