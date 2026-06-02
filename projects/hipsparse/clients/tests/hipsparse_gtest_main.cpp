/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "utility.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <hip/hip_runtime_api.h>
#include <stdexcept>

#include "hipsparse_parse_data.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>

using namespace testing;

// Log device VRAM usage. A failed hipMemGetInfo is a strong signal that the
// device has been lost (rather than merely out of memory).
static bool hipsparse_log_device_memory(const char* context)
{
    size_t     free_mem  = 0;
    size_t     total_mem = 0;
    hipError_t status    = hipMemGetInfo(&free_mem, &total_mem);
    if(status == hipSuccess)
    {
        fprintf(stderr,
                "[ MEMORY   ] %s: device free/total = %zu/%zu MB\n",
                context,
                free_mem >> 20,
                total_mem >> 20);
        return true;
    }

    fprintf(stderr,
            "[ MEMORY   ] %s: hipMemGetInfo failed: hip error %d (%s) -- device may be lost\n",
            context,
            status,
            hipGetErrorString(status));
    return false;
}

// Print the currently running test on a fatal signal so a single crashed CI run
// is diagnosable without a rerun, then re-raise with the default handler.
extern "C" void hipsparse_fatal_signal_handler(int sig)
{
    const char* sig_name = (sig == SIGSEGV) ? "SIGSEGV" : (sig == SIGABRT) ? "SIGABRT" : "signal";

    const TestInfo* info = UnitTest::GetInstance()->current_test_info();
    if(info != nullptr)
    {
        fprintf(stderr,
                "\n[ FATAL    ] hipsparse-test TERMINATED BY %s during test: %s.%s\n",
                sig_name,
                info->test_case_name(),
                info->name());
    }
    else
    {
        fprintf(stderr,
                "\n[ FATAL    ] hipsparse-test TERMINATED BY %s (no test currently running)\n",
                sig_name);
    }
    fflush(stderr);

    // Restore the default disposition and re-raise so the process exit status
    // still reflects the original signal.
    signal(sig, SIG_DFL);
    raise(sig);
}

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* eventListener;

public:
    bool showTestCases; // Show the names of each test case.
    bool showTestNames; // Show the names of each test.
    bool showSuccesses; // Show each success.
    bool showInlineFailures; // Show each failure as it occurs.
    bool showEnvironment; // Show the setup of the global environment.

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
        , showTestCases(true)
        , showTestNames(true)
        , showSuccesses(true)
        , showInlineFailures(true)
        , showEnvironment(true)
    {
    }

    ~ConfigurableEventListener() override
    {
        delete eventListener;
    }

    void OnTestProgramStart(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramStart(unit_test);
        hipsparse_log_device_memory("test program start");
    }

    void OnTestIterationStart(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }

    void OnEnvironmentsSetUpStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpStart(unit_test);
        }
    }

    void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
        }
    }

    void OnTestCaseStart(const TestCase& test_case) override
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseStart(test_case);
        }
    }

    void OnTestStart(const TestInfo& test_info) override
    {
        if(showTestNames)
        {
            eventListener->OnTestStart(test_info);
        }
    }

    void OnTestPartResult(const TestPartResult& result) override
    {
        eventListener->OnTestPartResult(result);

        // On the first failure, capture device memory state. This both proves
        // the tests themselves are not the memory consumer and detects a device
        // that has collapsed mid-run. If the device is lost, abort the whole run
        // so the log shows one clear fault instead of a long cascade of
        // dependent failures (and a likely SIGSEGV) on the dead device.
        if(result.failed())
        {
            const bool device_alive = hipsparse_log_device_memory("on test failure");
            if(!device_alive)
            {
                fprintf(stderr,
                        "[ DEVICE   ] Device appears lost after a test failure; aborting the test "
                        "run to avoid a cascade of dependent failures.\n");
                fflush(stderr);
                // _Exit avoids running static destructors that may themselves
                // fault while the device is in a bad state.
                std::_Exit(EXIT_FAILURE);
            }
        }
    }

    void OnTestEnd(const TestInfo& test_info) override
    {
        if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
        {
            eventListener->OnTestEnd(test_info);
        }
    }

    void OnTestCaseEnd(const TestCase& test_case) override
    {
        if(showTestCases)
        {
            eventListener->OnTestCaseEnd(test_case);
        }
    }

    void OnEnvironmentsTearDownStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownStart(unit_test);
        }
    }

    void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
        {
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
        }
    }

    void OnTestIterationEnd(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramEnd(unit_test);
    }
};

hipsparseStatus_t hipsparse_record_output_legend(const std::string& s)
{
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparse_record_output(const std::string& s)
{
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparse_record_timing(double msec, double gflops, double gbs)
{
    return HIPSPARSE_STATUS_SUCCESS;
}

bool display_timing_info_is_stdout_disabled()
{
    return HIPSPARSE_STATUS_SUCCESS;
}

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Print version
    char version[512];
    query_version(version);

    // Get device id from command line
    int device_id = 0;

    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--device") == 0 && argc > i + 1)
        {
            device_id = atoi(argv[i + 1]);
        }

        if(strcmp(argv[i], "--matrices-dir") == 0)
        {
            if(argc > i + 1)
            {
                s_hipsparse_clients_matrices_dir = argv[i + 1];
            }
            else
            {
                fprintf(stderr, "missing argument from option --matrices-dir");
                return -1;
            }
        }

        if(strcmp(argv[i], "--version") == 0)
        {
            printf("hipSPARSE version: %s\n", version);
            return 0;
        }
        if(strcmp(argv[i], "--help") == 0)
        {
            fprintf(stderr,
                    "Usage: %s [--matrices-dir <matrix directory path>] [--device <device id>]\n",
                    argv[0]);
            fprintf(stderr,
                    "To specify the directory of matrix input files the user can export the "
                    "environment variable HIPSPARSE_CLIENTS_MATRICES_DIR or uses the command line "
                    "option '--matrices-dir'. If the command line option '--matrices-dir' is used "
                    "then the environment variable HIPSPARSE_CLIENTS_MATRICES_DIR is ignored.\n");
            return 0;
        }
    }

    // Device Query
    int device_count = query_device_property();

    if(device_count <= device_id)
    {
        fprintf(stderr, "Error: invalid device ID. There may not be such device ID. Will exit\n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    printf("hipSPARSE version: %s\n", version);

    std::string datapath = hipsparse_datapath();

    // Print test data path being used
    std::cout << "hipSPARSE data path: " << datapath << std::endl;

    // Set data file path
    hipsparse_parse_data(argc, argv, datapath + "hipsparse_test.data");

    // Initialize google test
    InitGoogleTest(&argc, argv);

    // Remove the default listener
    auto& listeners       = UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // Add our listener, by default everything is on (the same as using the default listener)
    // Here turning everything off so only the 3 lines for the result are visible
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto listener       = new ConfigurableEventListener(default_printer);
    auto gtest_listener = getenv("GTEST_LISTENER");

    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
    {
        listener->showTestNames = listener->showSuccesses = listener->showInlineFailures = false;
    }

    listeners.Append(listener);

    // Install fatal-signal handlers so a crash (e.g. a dereference of an
    // un-initialized device buffer after an allocation failure) still records
    // the test that was running before the process dies.
    signal(SIGSEGV, hipsparse_fatal_signal_handler);
    signal(SIGABRT, hipsparse_fatal_signal_handler);

    // Run all tests
    int ret = RUN_ALL_TESTS();

    // Reset HIP device
    (void)hipDeviceReset();

    return ret;
}
