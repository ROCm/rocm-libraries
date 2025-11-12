// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rocfft/rocfft.h"

#include "../../shared/concurrency.h"
#include "../../shared/environment.h"
#include "../../shared/gpubuf.h"
#include "../../shared/params_gen.h"
#include "../../shared/precision_type.h"
#include "../../shared/rocfft_complex.h"
#include "hip/hip_runtime_api.h"
#include <boost/scope_exit.hpp>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

namespace fs = std::filesystem;

#ifndef WIN32
// get program_invocation_name
#include <errno.h>
#endif

TEST(rocfft_UnitTest, plan_description)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    rocfft_plan_description desc = nullptr;
    ASSERT_TRUE(rocfft_status_success == rocfft_plan_description_create(&desc));

    rocfft_array_type in_array_type  = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;

    size_t rank = 1;

    size_t i_strides[3] = {1, 1, 1};
    size_t o_strides[3] = {1, 1, 1};

    size_t idist = 0;
    size_t odist = 0;

    rocfft_plan plan   = NULL;
    size_t      length = 8;

    ASSERT_TRUE(rocfft_status_success
                == rocfft_plan_description_set_data_layout(desc,
                                                           in_array_type,
                                                           out_array_type,
                                                           0,
                                                           0,
                                                           rank,
                                                           i_strides,
                                                           idist,
                                                           rank,
                                                           o_strides,
                                                           odist));
    ASSERT_TRUE(rocfft_status_success
                == rocfft_plan_create(&plan,
                                      rocfft_placement_inplace,
                                      rocfft_transform_type_complex_forward,
                                      rocfft_precision_single,
                                      rank,
                                      &length,
                                      1,
                                      desc));

    ASSERT_TRUE(rocfft_status_success == rocfft_plan_description_destroy(desc));
    ASSERT_TRUE(rocfft_status_success == rocfft_plan_destroy(plan));
}

TEST(rocfft_UnitTest, plan_description_reuse)
{
    // check that a plan description can be reused between different
    // plans, with different layout parameters for each.

    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    // allocate plan description once
    rocfft_plan_description desc = nullptr;
    ASSERT_EQ(rocfft_plan_description_create(&desc), rocfft_status_success);

    std::vector<rocfft_complex<float>> output;

    // do length-8 FFTs with different strides.  first one is
    // stride-1 and we use that as our baseline to know what output
    // to expect for the rest
    const size_t length = 8;
    for(const size_t stride : {1, 2, 4})
    {
        // set layout for this stride
        ASSERT_EQ(rocfft_plan_description_set_data_layout(desc,
                                                          rocfft_array_type_complex_interleaved,
                                                          rocfft_array_type_complex_interleaved,
                                                          nullptr,
                                                          nullptr,
                                                          1,
                                                          &stride,
                                                          length * stride,
                                                          1,
                                                          &stride,
                                                          length * stride),
                  rocfft_status_success);

        static const rocfft_complex<float> input[8]{{-0.100, 0.380},
                                                    {0.0166, 0.439},
                                                    {-0.475, 0.212},
                                                    {0.440, -0.432},
                                                    {0.445, 0.0589},
                                                    {0.296, 0.164},
                                                    {-0.084, 0.077},
                                                    {0.320, 0.087}};

        // allocate host buffer.  initialize the whole thing to zero
        // but set a known input along the strides we want
        std::vector<rocfft_complex<float>> data_host(length * stride, {0.0, 0.0});
        for(size_t i = 0; i < length; ++i)
        {
            data_host[i * stride] = input[i];
        }

        // copy to device
        const size_t data_bytes = data_host.size() * sizeof(rocfft_complex<float>);
        gpubuf_t<rocfft_complex<float>> data_dev;
        ASSERT_EQ(data_dev.alloc(data_bytes), hipSuccess);
        void* data_dev_ptr = data_dev.data();
        ASSERT_EQ(hipMemcpy(data_dev_ptr, data_host.data(), data_bytes, hipMemcpyHostToDevice),
                  hipSuccess);

        // do the transform
        rocfft_plan plan = nullptr;
        ASSERT_EQ(rocfft_plan_create(&plan,
                                     rocfft_placement_inplace,
                                     rocfft_transform_type_complex_forward,
                                     rocfft_precision_single,
                                     1,
                                     &length,
                                     1,
                                     desc),
                  rocfft_status_success);
        ASSERT_EQ(rocfft_execute(plan, &data_dev_ptr, nullptr, nullptr), rocfft_status_success);
        ASSERT_EQ(hipMemcpy(data_host.data(), data_dev_ptr, data_bytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // save output for reference on first run
        if(output.empty())
        {
            output = data_host;
        }
        else
        {
            // check that the output matches output from the first
            // (stride-1) run.
            for(size_t i = 0; i < length; ++i)
                ASSERT_EQ(data_host[i * stride], output[i]);
        }
        ASSERT_EQ(rocfft_plan_destroy(plan), rocfft_status_success);
    }

    ASSERT_EQ(rocfft_plan_description_destroy(desc), rocfft_status_success);
}

// run a transform with all log levels enabled
TEST(rocfft_UnitTest, log_levels)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    // clean up environment and temporary file when we exit
    BOOST_SCOPE_EXIT_ALL(=)
    {
        rocfft_cleanup();
        // re-init logs with default logging
        rocfft_setup();
    };
    rocfft_cleanup();

    // enumerate all known log levels and direct all of the logs to nowhere
    EnvironmentSetTemp layer("ROCFFT_LAYER", std::to_string(0xffffffff).c_str());
#ifdef WIN32
    static const char* log_output = "NUL";
#else
    static const char* log_output   = "/dev/null";
#endif
    EnvironmentSetTemp log_trace_path("ROCFFT_LOG_TRACE_PATH", log_output);
    EnvironmentSetTemp log_bench_path("ROCFFT_LOG_BENCH_PATH", log_output);
    EnvironmentSetTemp log_profile_path("ROCFFT_LOG_PROFILE_PATH", log_output);
    EnvironmentSetTemp log_plan_path("ROCFFT_LOG_PLAN_PATH", log_output);
    EnvironmentSetTemp log_kernelio_path("ROCFFT_LOG_KERNELIO_PATH", log_output);
    EnvironmentSetTemp log_rtc_path("ROCFFT_LOG_RTC_PATH", log_output);
    EnvironmentSetTemp log_tuning_path("ROCFFT_LOG_TUNING_PATH", log_output);
    EnvironmentSetTemp log_graph_path("ROCFFT_LOG_GRAPH_PATH", log_output);

    rocfft_setup();

    // Test single-kernel Bluestein and a multi-kernel plan
    //
    // TODO: add fused L1D Bluestein case like 8191, as that does weird
    // things with buffers
    for(const size_t length : {
            37,
            64,
            32768,
        })
    {
        for(const auto type : {rocfft_transform_type_complex_forward,
                               rocfft_transform_type_real_forward,
                               rocfft_transform_type_real_inverse})
        {
            for(const auto precision :
                {rocfft_precision_single, rocfft_precision_double, rocfft_precision_half})
            {
                rocfft_plan plan = nullptr;
                ASSERT_EQ(
                    rocfft_plan_create(
                        &plan, rocfft_placement_inplace, type, precision, 1, &length, 1, nullptr),
                    rocfft_status_success);

                // assume transform uses complex, will overallocate for real
                // transforms but we only care about logging
                gpubuf data_dev;
                ASSERT_EQ(
                    data_dev.alloc(element_size(precision, rocfft_array_type_complex_interleaved)
                                   * length),
                    hipSuccess);

                void* data_dev_ptr = data_dev.data();
                ASSERT_EQ(rocfft_execute(plan, &data_dev_ptr, nullptr, nullptr),
                          rocfft_status_success);

                rocfft_plan_destroy(plan);
            }
        }
    }
}

// Check whether logs can be emitted from multiple threads properly
TEST(rocfft_UnitTest, log_multithreading)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    static const int   NUM_THREADS          = 10;
    static const int   NUM_ITERS_PER_THREAD = 50;
    static const char* TRACE_FILE           = "trace.log";

    // clean up environment and temporary file when we exit
    BOOST_SCOPE_EXIT_ALL(=)
    {
        rocfft_cleanup();
        remove(TRACE_FILE);
        // re-init logs with default logging
        rocfft_setup();
    };

    // ask for trace logging, since that's the easiest to trigger
    rocfft_cleanup();
    EnvironmentSetTemp layer("ROCFFT_LAYER", "1");
    EnvironmentSetTemp tracepath("ROCFFT_LOG_TRACE_PATH", TRACE_FILE);

    rocfft_setup();

    // run a whole bunch of threads in parallel, each one doing
    // something small that will write to the trace log
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([]() {
            for(int j = 0; j < NUM_ITERS_PER_THREAD; ++j)
            {
                rocfft_plan_description desc;
                rocfft_plan_description_create(&desc);
                rocfft_plan_description_destroy(desc);
            }
        });
    }

    for(auto& t : threads)
    {
        t.join();
    }

    rocfft_cleanup();

    // now verify that the trace log has one message per line, with nothing garbled
    std::ifstream trace_log(TRACE_FILE);
    std::string   line;
    std::regex    validator("^rocfft_(setup|cleanup|plan_description_(create|destroy),"
                         "description,[x0-9a-fA-F]+)$");
    while(std::getline(trace_log, line))
    {
        bool res = std::regex_match(line, validator);
        ASSERT_TRUE(res) << "line contains invalid content: " << line;
    }
}

// a function that accepts a plan's requested size on input, and
// returns the size to actually allocate for the test
typedef std::function<size_t(size_t)> workmem_sizer;

void workmem_test(workmem_sizer sizer,
                  rocfft_status exec_status_expected,
                  bool          give_null_work_buf = false)
{
    // Prime size requires Bluestein, which guarantees work memory.
    size_t      length = 8191;
    rocfft_plan plan   = NULL;

    ASSERT_EQ(rocfft_plan_create(&plan,
                                 rocfft_placement_inplace,
                                 rocfft_transform_type_complex_forward,
                                 rocfft_precision_single,
                                 1,
                                 &length,
                                 1,
                                 nullptr),
              rocfft_status_success);

    size_t requested_work_size = 0;
    ASSERT_EQ(rocfft_plan_get_work_buffer_size(plan, &requested_work_size), rocfft_status_success);
    ASSERT_GT(requested_work_size, 0U);

    rocfft_execution_info info;
    ASSERT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);

    size_t alloc_work_size = sizer(requested_work_size);
    gpubuf work_buffer;
    if(alloc_work_size)
    {
        ASSERT_EQ(work_buffer.alloc(alloc_work_size), hipSuccess);

        void*         work_buffer_ptr;
        rocfft_status set_work_expected_status;
        if(give_null_work_buf)
        {
            work_buffer_ptr          = nullptr;
            set_work_expected_status = rocfft_status_invalid_work_buffer;
        }
        else
        {
            work_buffer_ptr          = work_buffer.data();
            set_work_expected_status = rocfft_status_success;
        }
        ASSERT_EQ(rocfft_execution_info_set_work_buffer(info, work_buffer_ptr, alloc_work_size),
                  set_work_expected_status);
    }

    // allocate 2x length for complex
    std::vector<float> data_host(length * 2, 1.0f);
    gpubuf             data_device;
    auto               data_size_bytes = data_host.size() * sizeof(float);

    ASSERT_EQ(data_device.alloc(data_size_bytes), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(data_device.data(), data_host.data(), data_size_bytes, hipMemcpyHostToDevice),
        hipSuccess);
    std::vector<void*> ibuffers(1, static_cast<void*>(data_device.data()));

    ASSERT_EQ(rocfft_execute(plan, ibuffers.data(), nullptr, info), exec_status_expected);

    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
}

// check what happens if work memory is required but is not provided
// - library should allocate
TEST(rocfft_UnitTest, workmem_missing)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    workmem_test([](size_t) { return 0; }, rocfft_status_success);
}

// check what happens if work memory is required but not enough is provided
TEST(rocfft_UnitTest, workmem_small)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    workmem_test([](size_t requested) { return requested / 2; }, rocfft_status_invalid_work_buffer);
}

// hard to imagine this being a problem, but try giving too much as well
TEST(rocfft_UnitTest, workmem_big)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    workmem_test([](size_t requested) { return requested * 2; }, rocfft_status_success);
}

// check if a user explicitly gives a null pointer - set work buffer
// should fail, but transform should succeed because library
// allocates
TEST(rocfft_UnitTest, workmem_null)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    workmem_test([](size_t requested) { return requested; }, rocfft_status_success, true);
}

static const size_t RTC_PROBLEM_SIZE = 2304;
// runtime compilation cache tests main loop
void rtc_cache_main()
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    // PRECONDITIONS

    // - set cache location to custom path, requires uninitializing
    //   the lib and reinitializing with some env vars
    // - also enable RTC logging so we can tell when something was
    //   actually compiled
    const std::string rtc_cache_path = std::tmpnam(nullptr);
    const std::string rtc_log_path   = std::tmpnam(nullptr);

    void*  empty_cache           = nullptr;
    size_t empty_cache_bytes     = 0;
    void*  onekernel_cache       = nullptr;
    size_t onekernel_cache_bytes = 0;

    // cleanup
    BOOST_SCOPE_EXIT_ALL(=)
    {
        // close log file handles
        rocfft_cleanup();
        remove(rtc_cache_path.c_str());
        remove(rtc_log_path.c_str());
        // re-init lib now that the env vars are gone
        rocfft_setup();
        if(empty_cache)
            rocfft_cache_buffer_free(empty_cache);
        if(onekernel_cache)
            rocfft_cache_buffer_free(onekernel_cache);
    };

    rocfft_cleanup();
    EnvironmentSetTemp cache_env("ROCFFT_RTC_CACHE_PATH", rtc_cache_path.c_str());
    EnvironmentSetTemp layer_env("ROCFFT_LAYER", "32");
    EnvironmentSetTemp log_env("ROCFFT_LOG_RTC_PATH", rtc_log_path.c_str());
    rocfft_setup();

    // - serialize empty cache as baseline
    ASSERT_EQ(rocfft_cache_serialize(&empty_cache, &empty_cache_bytes), rocfft_status_success);

    // END PRECONDITIONS

    // pick a length that's runtime compiled
    auto build_plan = [&]() {
        rocfft_plan plan = nullptr;
        ASSERT_TRUE(rocfft_status_success
                    == rocfft_plan_create(&plan,
                                          rocfft_placement_inplace,
                                          rocfft_transform_type_complex_forward,
                                          rocfft_precision_single,
                                          1,
                                          &RTC_PROBLEM_SIZE,
                                          1,
                                          nullptr));
        // we don't need to actually execute the plan, so we can
        // destroy it right away.  this ensures that we don't hold on
        // to a plan after we cleanup the library
        rocfft_plan_destroy(plan);
        plan = nullptr;
    };
    // check the RTC log to see if an FFT kernel got compiled
    auto fft_kernel_was_compiled = [&]() {
        // HACK: logging is done in a worker thread, so sleep for a
        // bit to give it a chance to actually write.  It at least
        // should flush after writing.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // look for a ROCFFT_RTC_BEGIN line that indicates RTC happened
        std::ifstream logfile(rtc_log_path);
        std::string   line;
        while(std::getline(logfile, line))
        {
            if(line.find("ROCFFT_RTC_BEGIN") != std::string::npos
               && line.find("fft_") != std::string::npos)
                return true;
        }
        return false;
    };

    // build a plan that requires runtime compilation,
    // close logs and ensure a kernel was built
    build_plan();
    ASSERT_EQ(rocfft_cache_serialize(&onekernel_cache, &onekernel_cache_bytes),
              rocfft_status_success);
    rocfft_cleanup();
    ASSERT_TRUE(fft_kernel_was_compiled());

    // serialized cache should be bigger than empty cache
    ASSERT_GT(onekernel_cache_bytes, empty_cache_bytes);

    // blow away the cache, reinit the library,
    // retry building the plan again and ensure the kernel was rebuilt
    remove(rtc_cache_path.c_str());
    rocfft_setup();
    build_plan();
    rocfft_cache_buffer_free(onekernel_cache);
    onekernel_cache = nullptr;
    ASSERT_EQ(rocfft_cache_serialize(&onekernel_cache, &onekernel_cache_bytes),
              rocfft_status_success);
    rocfft_cleanup();
    ASSERT_TRUE(fft_kernel_was_compiled());
    ASSERT_GT(onekernel_cache_bytes, empty_cache_bytes);

    // re-init library without blowing away cache.  rebuild plan and
    // check that the kernel was not recompiled.
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_FALSE(fft_kernel_was_compiled());

    // blow away cache again, deserialize one-kernel cache.  re-init
    // library and rebuild plan - kernel should again not be
    // recompiled
    remove(rtc_cache_path.c_str());
    rocfft_setup();
    ASSERT_EQ(rocfft_cache_deserialize(onekernel_cache, onekernel_cache_bytes),
              rocfft_status_success);
    rocfft_cleanup();
    ASSERT_FALSE(fft_kernel_was_compiled());

    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_FALSE(fft_kernel_was_compiled());

    // use the cache as a system cache and make the user one an empty
    // in-memory cache.  kernel should still not be recompiled.
    EnvironmentSetTemp cache_sys_env("ROCFFT_RTC_SYS_CACHE_PATH", rtc_cache_path.c_str());
    EnvironmentSetTemp cache_empty_env("ROCFFT_RTC_CACHE_PATH", ":memory:");
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_FALSE(fft_kernel_was_compiled());

    // check that the system cache is not written to, even if it's
    // writable by the current user.  after removing the cache, the
    // kernel should always be recompiled since rocFFT has no durable
    // place to write it to.
    remove(rtc_cache_path.c_str());
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_TRUE(fft_kernel_was_compiled());
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_TRUE(fft_kernel_was_compiled());
}

// run the main body of rtc cache tests twice to uncover potential
// problems with thread reuse between iterations
TEST(rocfft_UnitTest, rtc_cache_iter_1)
{
    rtc_cache_main();
}

TEST(rocfft_UnitTest, rtc_cache_iter_2)
{
    rtc_cache_main();
}

// make sure cache API functions tolerate null pointers without crashing
TEST(rocfft_UnitTest, rtc_cache_null)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    void*  buf     = nullptr;
    size_t buf_len = 0;
    ASSERT_EQ(rocfft_cache_serialize(nullptr, &buf_len), rocfft_status_invalid_arg_value);
    ASSERT_EQ(rocfft_cache_serialize(&buf, nullptr), rocfft_status_invalid_arg_value);
    ASSERT_EQ(rocfft_cache_buffer_free(nullptr), rocfft_status_success);
    ASSERT_EQ(rocfft_cache_deserialize(nullptr, 12345), rocfft_status_invalid_arg_value);
    ASSERT_EQ(rocfft_cache_deserialize(&buf_len, 0), rocfft_status_invalid_arg_value);
}

// make sure RTC gracefully handles a helper process that crashes
TEST(rocfft_UnitTest, rtc_helper_crash)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

#ifdef WIN32
    char filename[MAX_PATH];
    GetModuleFileNameA(NULL, filename, MAX_PATH);
    fs::path test_exe    = filename;
    fs::path crasher_exe = test_exe.replace_filename("rtc_helper_crash.exe");
#else
    fs::path           test_exe     = program_invocation_name;
    fs::path           crasher_exe  = test_exe.replace_filename("rtc_helper_crash");
#endif

    // use the crashing helper
    EnvironmentSetTemp env_helper("ROCFFT_RTC_PROCESS_HELPER", crasher_exe.string().c_str());
    // don't touch the cache, to force compilation
    EnvironmentSetTemp env_read("ROCFFT_RTC_CACHE_READ_DISABLE", "1");
    EnvironmentSetTemp env_write("ROCFFT_RTC_CACHE_WRITE_DISABLE", "1");
    // force out-of-process compile
    EnvironmentSetTemp env_process("ROCFFT_RTC_PROCESS", "2");

    rocfft_plan plan = nullptr;
    ASSERT_TRUE(rocfft_status_success
                == rocfft_plan_create(&plan,
                                      rocfft_placement_inplace,
                                      rocfft_transform_type_complex_forward,
                                      rocfft_precision_single,
                                      1,
                                      &RTC_PROBLEM_SIZE,
                                      1,
                                      nullptr));

    // alloc a complex buffer
    gpubuf_t<rocfft_complex<float>> data;
    ASSERT_EQ(data.alloc(RTC_PROBLEM_SIZE * sizeof(rocfft_complex<float>)), hipSuccess);

    std::vector<void*> ibuffers(1, static_cast<void*>(data.data()));

    ASSERT_EQ(rocfft_execute(plan, ibuffers.data(), nullptr, nullptr), rocfft_status_success);

    rocfft_plan_destroy(plan);
    plan = nullptr;

    rocfft_cleanup();
    rocfft_setup();

    // also try with forcing use of the subprocess, which is a
    // different code path from the default "try in-process, then
    // fall back to out-of-process"
    EnvironmentSetTemp env_force("ROCFFT_RTC_PROCESS", "1");

    ASSERT_TRUE(rocfft_status_success
                == rocfft_plan_create(&plan,
                                      rocfft_placement_inplace,
                                      rocfft_transform_type_complex_forward,
                                      rocfft_precision_single,
                                      1,
                                      &RTC_PROBLEM_SIZE,
                                      1,
                                      nullptr));
    ASSERT_EQ(rocfft_execute(plan, ibuffers.data(), nullptr, nullptr), rocfft_status_success);

    rocfft_plan_destroy(plan);
    plan = nullptr;
}

TEST(rocfft_UnitTest, rtc_test_harness)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    // check that hipcc is available since this test requires it
    //
    // NOTE: using system() for launching subprocesses for simplicity
    // and portability
#ifdef WIN32
    static const char* test_command = "amdclang++ --version > NUL";
#else
    static const char* test_command = "amdclang++ --version > /dev/null";
#endif
    if(std::system(test_command) != 0)
        GTEST_SKIP();

    rocfft_cleanup();

    BOOST_SCOPE_EXIT_ALL()
    {
        // reinit rocFFT so caching goes back to normal
        rocfft_cleanup();
        rocfft_setup();
    };

    // extra scope to control lifetime of env vars
    {
        // create a temporary directory to hold all of the temp files
        // that get created
        const fs::path tmp_path = std::tmpnam(nullptr);
        try
        {
            fs::create_directory(tmp_path);
        }
        catch(fs::filesystem_error& e)
        {
            GTEST_SKIP() << "unable to create temp dir for test harnesses: " << e.what();
        }

        // activate writing of rtc test harnesses
        EnvironmentSetTemp env_harness("ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS", "1");

        // set path for writing rtc test harnesses source files
        EnvironmentSetTemp env_harness_path("ROCFFT_DEBUG_KERNEL_HARNESS_PATH",
                                            tmp_path.string().c_str());

        // ensure every kernel gets compiled once
        EnvironmentSetTemp env_cache("ROCFFT_RTC_CACHE_PATH", ":memory:");
        EnvironmentSetTemp env_sys_cache("ROCFFT_RTC_SYS_CACHE_PATH", ":memory:");

        rocfft_setup();

        // construct a few different types of plans to try to get all
        // different kernels compiled

        auto create_destroy_plan
            = [](rocfft_transform_type type, const size_t dim, const size_t* lengths) -> void {
            rocfft_plan plan = nullptr;
            ASSERT_EQ(rocfft_plan_create(&plan,
                                         rocfft_placement_inplace,
                                         type,
                                         rocfft_precision_single,
                                         dim,
                                         lengths,
                                         1,
                                         nullptr),
                      rocfft_status_success);
            ASSERT_EQ(rocfft_plan_destroy(plan), rocfft_status_success);
            plan = nullptr;
        };
        // large 1D R2C + C2R
        const size_t L1D_PROBLEM_SIZE[1] = {16384};
        create_destroy_plan(rocfft_transform_type_real_forward, 1, L1D_PROBLEM_SIZE);
        create_destroy_plan(rocfft_transform_type_real_inverse, 1, L1D_PROBLEM_SIZE);

        // small bluestein R2C + C2R (also covers odd length)
        const size_t SMALL_BLUESTEIN_PROBLEM_SIZE[1] = {37};
        create_destroy_plan(rocfft_transform_type_real_forward, 1, SMALL_BLUESTEIN_PROBLEM_SIZE);
        create_destroy_plan(rocfft_transform_type_real_inverse, 1, SMALL_BLUESTEIN_PROBLEM_SIZE);

        // large bluestein C2C
        const size_t LARGE_BLUESTEIN_PROBLEM_SIZE[1] = {8191};
        create_destroy_plan(rocfft_transform_type_complex_forward, 1, LARGE_BLUESTEIN_PROBLEM_SIZE);

        // L1D_TRTRT
        const size_t L1D_TRTRT_PROBLEM_SIZE[1] = {680};
        create_destroy_plan(rocfft_transform_type_complex_forward, 1, L1D_TRTRT_PROBLEM_SIZE);

        // small 3D (exercises 2D_SINGLE)
        const size_t SMALL_3D_PROBLEM_SIZE[3] = {25, 25, 25};
        create_destroy_plan(rocfft_transform_type_complex_forward, 3, SMALL_3D_PROBLEM_SIZE);

        // larger 3D
        const size_t LARGE_3D_PROBLEM_SIZE[3] = {200, 200, 200};
        create_destroy_plan(rocfft_transform_type_complex_forward, 3, LARGE_3D_PROBLEM_SIZE);

        // now try to compile each file - they'd need hand-editing to test
        // something useful, but we can at least ensure they build.

        // enumerate all the files
        std::vector<std::pair<std::string, int>> files;
        size_t                                   i = 0;
        for(;; ++i)
        {
            // construct name of main file
            fs::path main_file = tmp_path / ("rocfft_kernel_harness_" + std::to_string(i) + ".cpp");

            if(!fs::exists(main_file))
                break;

            files.emplace_back(main_file.string(), -1);
        }

        // we should have generated at least a few kernels
        ASSERT_FALSE(files.empty());

#ifdef _OPENMP
#pragma omp parallel for num_threads(rocfft_concurrency())
#endif
        for(i = 0; i < files.size(); ++i)
        {
#ifdef WIN32
            const std::string command = "amdclang++ -x hip -c -std=c++17 -o NUL " + files[i].first;
#else
            const std::string command
                = "amdclang++ -x hip -c -std=c++17 -o /dev/null " + files[i].first;
#endif
            files[i].second = std::system(command.c_str());
        }

        // check that all compiles succeeded
        for(const auto& file : files)
        {
            ASSERT_EQ(file.second, 0);
        }

        // clean up temporary files
        try
        {
            fs::remove_all(tmp_path);
        }
        catch(fs::filesystem_error&)
        {
            // this should work, but ignore errors as the build
            // status is what matters for this test
        }
    }
}
