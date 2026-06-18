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
#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "hipsparse_parse_data.hpp"
#include "program_options.hpp"

#include <csignal>

using testing::InitGoogleTest;
using testing::TestCase;
using testing::TestEventListener;
using testing::TestInfo;
using testing::TestPartResult;
using testing::UnitTest;

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
    return false;
}

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Parse hipsparse-test options. Unknown options (e.g. --gtest_*) are
    // ignored here so they can be forwarded to InitGoogleTest below. The
    // backing storage for matrices_dir is a function-scope static so that
    // s_hipsparse_clients_matrices_dir (a const char*) remains valid for the
    // entire program lifetime.
    std::string matrices_dir;
    int         device_id = 0;

    options_description desc("hipSPARSE test command line options");
    // clang-format off
    desc.add_options()
        ("help,h", 
            "Produces this help message and exits.")    
        ("version,v",
            "Prints the hipSPARSE version and exits.")
        ("device,d",
            value<int>(&device_id)->default_value(0),
            "Set the default device to use for the tests.")
        ("matrices-dir",
            value<std::string>(&matrices_dir),
            "Path to the directory containing the test matrix input files. "
            "Overrides the HIPSPARSE_CLIENTS_MATRICES_DIR environment variable "
            "when both are specified.");
    // clang-format on

    variables_map vm;
    try
    {
        store(parse_command_line(argc, argv, desc, /*ignoreUnknown=*/true), vm);
        notify(vm);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "Error parsing command line: %s\n", e.what());
        return -1;
    }

    if(vm.count("help"))
    {
        std::cout << "Usage: " << argv[0] << " [hipsparse-test options] [GoogleTest options]\n\n"
                  << desc
                  << "\nAny options not listed above (e.g. --gtest_filter, "
                     "--gtest_list_tests) are forwarded to GoogleTest.\n"
                     "To specify the directory of matrix input files, the user can either "
                     "export the environment variable HIPSPARSE_CLIENTS_MATRICES_DIR or "
                     "use the command line option '--matrices-dir'. If '--matrices-dir' "
                     "is used then the environment variable is ignored."
                  << std::endl;
        return 0;
    }

    // Set matrix directory
    if(!matrices_dir.empty())
    {
        s_hipsparse_clients_matrices_dir = matrices_dir.c_str();
    }

    // Validate and select the requested HIP device before any subsequent
    // hipSPARSE / HIP API call.
    int device_count = query_device_property();

    if(device_id < 0 || device_count <= device_id)
    {
        fprintf(stderr,
                "Error: invalid device ID %d (detected %d device(s)). Will exit.\n",
                device_id,
                device_count);
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    // Query the hipSPARSE version on the selected device.
    char version[512];
    query_version(version);

    if(vm.count("version"))
    {
        printf("hipSPARSE version: %s\n", version);
        return 0;
    }

    printf("hipSPARSE version: %s\n", version);

    std::string datapath = hipsparse_datapath();

    // Print test data path being used
    std::cout << "hipSPARSE data path: " << datapath << std::endl;

    // Set data file path
    hipsparse_parse_data(datapath + "hipsparse_test.data");

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
