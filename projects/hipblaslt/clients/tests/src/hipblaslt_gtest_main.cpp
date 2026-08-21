/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_parse_data.hpp"
#include "hipblaslt_test.hpp"
#include "test_cleanup.hpp"
#include "utility.hpp"
#ifndef _WIN32
#include "hipblaslt_parallel_test.hpp"
#endif
#include <algorithm>
#include <string>
#include <vector>

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* const eventListener;
    std::atomic_size_t       skipped_tests{0}; // Number of skipped tests.
    std::atomic_size_t       current_test_number{0}; // Current test number (incremental counter).

public:
    bool showTestCases      = true; // Show the names of each test case.
    bool showTestNames      = true; // Show the names of each test.
    bool showSuccesses      = true; // Show each success.
    bool showInlineFailures = true; // Show each failure as it occurs.
    bool showEnvironment    = true; // Show the setup of the global environment.
    bool showInlineSkips    = true; // Show when we skip a test.

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
    {
    }

    ~ConfigurableEventListener() override
    {
        delete eventListener;
    }

    void OnTestProgramStart(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramStart(unit_test);
    }

    void OnTestIterationStart(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }

    void OnEnvironmentsSetUpStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpStart(unit_test);
    }

    void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
    }

    void OnTestCaseStart(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseStart(test_case);
    }

    void OnTestStart(const TestInfo& test_info) override
    {
        ++current_test_number;
        if(showTestNames)
        {
            // Print test number and delegate to default listener
            int total_tests = UnitTest::GetInstance()->test_to_run_count();
            hipblaslt_cout << "[Test #" << current_test_number << "/" << total_tests << "] " << std::flush;
            eventListener->OnTestStart(test_info);
        }
    }

    void OnTestPartResult(const TestPartResult& result) override
    {
        if(!strcmp(result.message(), LIMITED_MEMORY_STRING_GTEST))
        {
            if(showInlineSkips)
                hipblaslt_cout << "Skipped test due to limited memory environment." << std::endl;
            ++skipped_tests;
        }
        else if(!strcmp(result.message(), TOO_MANY_DEVICES_STRING_GTEST))
        {
            if(showInlineSkips)
                hipblaslt_cout << "Skipped test due to too few GPUs." << std::endl;
            ++skipped_tests;
        }
        eventListener->OnTestPartResult(result);
    }

    void OnTestEnd(const TestInfo& test_info) override
    {
        if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
            eventListener->OnTestEnd(test_info);
    }

    void OnTestCaseEnd(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseEnd(test_case);
    }

    void OnEnvironmentsTearDownStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownStart(unit_test);
    }

    void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
    }

    void OnTestIterationEnd(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const UnitTest& unit_test) override
    {
        if(skipped_tests)
            hipblaslt_cout << "[ SKIPPED  ] " << skipped_tests << " tests." << std::endl;
        eventListener->OnTestProgramEnd(unit_test);
    }
};

// Set the listener for Google Tests
static void hipblaslt_set_listener()
{
    // remove the default listener
    auto& listeners       = testing::UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // add our listener, by default everything is on (the same as using the default listener)
    // here I am turning everything off so I only see the 3 lines for the result
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto* listener       = new ConfigurableEventListener(default_printer);
    auto* gtest_listener = getenv("GTEST_LISTENER");

    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
    {
        listener->showTestNames      = false;
        listener->showSuccesses      = false;
        listener->showInlineFailures = false;
        listener->showInlineSkips    = false;
    }

    listeners.Append(listener);
}

static int hipblaslt_version()
{
    int                    version;
    hipblaslt_local_handle handle;
    hipblasLtGetVersion(handle, &version);
    return version;
}

static void hipblaslt_print_usage_warning()
{
    std::string warning(
        "parsing of test data may take a couple minutes before any test output appears...");

    hipblaslt_cout << "info: " << warning << "\n" << std::endl;
}

static std::string hipblaslt_capture_args(int argc, char** argv)
{
    std::ostringstream cmdLine;
    cmdLine << "command line: ";
    for(int i = 0; i < argc; i++)
    {
        if(argv[i])
            cmdLine << std::string(argv[i]) << " ";
    }
    return cmdLine.str();
}

static void hipblaslt_print_args(const std::string& args)
{
    hipblaslt_cout << args << std::endl;
    hipblaslt_cout.flush();
}

// Device Query
static void hipblaslt_set_test_device()
{
    hipDeviceProp_t props;
    int device_id    = 0;
    int device_count = query_device_property(device_id, props);
    if(device_count <= device_id)
    {
        hipblaslt_cerr << "Error: invalid device ID. There may not be such device ID." << std::endl;
        exit(-1);
    }
    set_device(device_id);
}

/*****************
 * Main function *
 *****************/
int main(int argc, char** argv)
{
    std::string args = hipblaslt_capture_args(argc, argv);

    // Check for --help to add our custom options
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--help" || arg == "-h" || arg == "-?" || arg == "/?" || arg == "--help-all")
        {
            hipblaslt_cout << "\nhipBLASLt Test Options:\n";
            hipblaslt_cout << "  --num_gpus=N\n";
            hipblaslt_cout << "  --num_gpus N\n";
            hipblaslt_cout << "      Run tests in parallel across N GPUs (Unix/Linux only).\n";
            hipblaslt_cout << "      Tests are automatically split evenly across the specified\n";
            hipblaslt_cout << "      number of GPUs. Each GPU runs its assigned tests independently.\n";
            hipblaslt_cout << "      Example: ./hipblaslt-test --num_gpus 8 --gtest_filter=\"*smoke*\"\n";
            hipblaslt_cout << "      Note: If --gtest_output=json:file.json is specified, per-GPU\n";
            hipblaslt_cout << "            results are saved as file_gpu0.json, file_gpu1.json, etc.\n";
            hipblaslt_cout << "\n";
            break;
        }
    }

    // Parse and strip --num_gpus argument
    int num_gpus = 0;
    bool has_num_gpus_flag = false;
    std::vector<int> indices_to_remove;  // Track all indices to remove

    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg.find("--num_gpus=") == 0)
        {
            num_gpus = std::atoi(arg.substr(11).c_str());
            has_num_gpus_flag = true;
            indices_to_remove.push_back(i);
            break;
        }
        else if(arg == "--num_gpus" && i + 1 < argc)
        {
            num_gpus = std::atoi(argv[i + 1]);
            has_num_gpus_flag = true;
            indices_to_remove.push_back(i);
            indices_to_remove.push_back(i + 1);
            break;
        }
    }

    // Parse and strip --gtest_output to extract base filename
    std::string gtest_output_base;
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg.find("--gtest_output=") == 0)
        {
            size_t colon_pos = arg.find(":");
            if(colon_pos != std::string::npos)
            {
                // Format: --gtest_output=json:file.json
                std::string format = arg.substr(15, colon_pos - 15);
                std::string filename = arg.substr(colon_pos + 1);
                gtest_output_base = format + ":" + filename;
            }
            else
            {
                // Format: --gtest_output=json (uses default filename)
                std::string format = arg.substr(15);
                if(format == "json")
                {
                    gtest_output_base = "json:test_detail.json"; // GTest default
                }
                else
                {
                    gtest_output_base = format; // Pass through other formats
                }
            }
            indices_to_remove.push_back(i);
            break;
        }
    }

#ifdef _WIN32
    // On Windows, parallel GPU execution is not supported
    if(has_num_gpus_flag)
    {
        hipblaslt_cerr << "Error: --num_gpus is not supported on Windows." << std::endl;
        return 1;
    }
#else
    // Check for invalid --num_gpus values
    if(has_num_gpus_flag && num_gpus <= 1)
    {
        hipblaslt_cerr << "Error: --num_gpus requires a value greater than 1." << std::endl;
        return 1;
    }

    // If parallel GPUs requested, use parallel execution
    if(num_gpus > 1)
    {
        // Remove custom flags from argv before passing to parallel runner
        // Sort indices in descending order to remove from back to front
        std::sort(indices_to_remove.begin(), indices_to_remove.end(), std::greater<int>());
        for(int idx : indices_to_remove)
        {
            for(int i = idx; i + 1 < argc; i++)
            {
                argv[i] = argv[i + 1];
            }
            argc--;
        }

        return run_tests_parallel_gpus(argc, argv, num_gpus, gtest_output_base);
    }
#endif

    // Set signal handler
    hipblaslt_test_sigaction();

    hipblaslt_print_version();

    // Set test device
    hipblaslt_set_test_device();

    hipblaslt_print_usage_warning();

    // Set data file path
    hipblaslt_parse_data(argc, argv, hipblaslt_exepath() + "hipblaslt_gtest.data");

    // Initialize Google Tests
    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    // Set Google Test listener
    hipblaslt_set_listener();

    // Run the tests
    int status = RUN_ALL_TESTS();

    // Failures printed at end for reporting so repeat version info
    hipblaslt_print_version();

    // Print command line at the end
    hipblaslt_print_args(args);

    //hipblaslt_shutdown();

    return status;
}
