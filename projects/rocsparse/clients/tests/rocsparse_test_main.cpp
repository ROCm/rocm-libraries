/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_clients_envariables.hpp"
#include "rocsparse_parse_data.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_test_listeners.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

#include "test_check.hpp"

bool test_check::s_auto_testing_bad_arg;

bool display_timing_info_is_stdout_disabled()
{
    return false;
}

rocsparse_status rocsparse_record_output(const std::string& s)
{
    return rocsparse_status_success;
}

rocsparse_status rocsparse_record_output_legend(const std::string& s)
{
    return rocsparse_status_success;
}

rocsparse_status rocsparse_record_timing(double msec, double gflops, double gbs)
{
    return rocsparse_status_success;
}

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    //
    // Enable debug mode for testing.
    //
    rocsparse_enable_debug();

    //
    // Add additional debug check for kernel launches.
    //
    rocsparse_enable_debug_kernel_launch();

    //
    // Enable test debug arguments.
    //
    if(rocsparse_clients_envariables::is_defined(
           rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS)
       == false)
    {
        rocsparse_clients_envariables::set(rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS,
                                           true);
    }

    // Get version
    rocsparse_handle handle;
    rocsparse_status status = rocsparse_create_handle(&handle);
    if(rocsparse_status_success != status)
    {
        std::cerr << "The creation of the rocsparse_handle failed." << std::endl;
        if(0 == rocsparse_state_debug())
        {
            std::cerr << "To get more information, please export the ROCSPARSE_DEBUG environment "
                         "variable:"
                      << std::endl;
            std::cerr << "export ROCSPARSE_DEBUG=1" << std::endl;
        }
        return status;
    }

    int  ver;
    char rev[64];

    status = rocsparse_get_version(handle, &ver);
    if(rocsparse_status_success != status)
    {
        std::cerr << "rocsparse_get_version failed." << std::endl;
        return status;
    }

    status = rocsparse_get_git_rev(handle, rev);
    if(rocsparse_status_success != status)
    {
        std::cerr << "rocsparse_get_git_rev failed." << std::endl;
        return status;
    }

    status = rocsparse_destroy_handle(handle);
    if(rocsparse_status_success != status)
    {
        std::cerr << "rocsparse_destroy_handle failed." << std::endl;
        return status;
    }

    // Get user device id from command line
    int dev = 0;

    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--device") == 0 && argc > i + 1)
        {
            dev = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "--version") == 0)
        {
            // Print version and exit, if requested
            std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
                      << ver % 100 << "-" << rev << std::endl;

            return 0;
        }
    }

    // Device query
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "Error: cannot get device count" << std::endl;
        return -1;
    }

    std::cout << "Query device success: there are " << devs << " devices" << std::endl;

    for(int i = 0; i < devs; ++i)
    {
        hipDeviceProp_t prop;

        if(hipGetDeviceProperties(&prop, i) != hipSuccess)
        {
            std::cerr << "rocsparse-test error: cannot get device properties" << std::endl;
            return rocsparse_status_internal_error;
        }

        std::cout << "Device ID " << i << ": " << prop.name << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
        std::cout << "with " << (prop.totalGlobalMem >> 20) << "MB memory, clock rate "
                  << prop.clockRate / 1000 << "MHz @ computing capability " << prop.major << "."
                  << prop.minor << std::endl;
        std::cout << "maxGridDimX " << prop.maxGridSize[0] << ", sharedMemPerBlock "
                  << (prop.sharedMemPerBlock >> 10) << "KB, maxThreadsPerBlock "
                  << prop.maxThreadsPerBlock << std::endl;
        std::cout << "wavefrontSize " << prop.warpSize << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
    }

    // Set device
    if(hipSetDevice(dev) != hipSuccess || dev >= devs)
    {
        std::cerr << "Error: cannot set device ID " << dev << std::endl;
        return -1;
    }

    hipDeviceProp_t prop;
    if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
    {
        std::cerr << "rocsparse-test error: cannot get device properties" << std::endl;
        return rocsparse_status_internal_error;
    }

    std::cout << "Using device ID " << dev << " (" << prop.name << ") for rocSPARSE" << std::endl;
    std::cout << "-------------------------------------------------------------------------"
              << std::endl;

    // Print version
    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    std::string datapath = rocsparse_datapath();

    // Print test data path being used
    std::cout << "rocSPARSE data path: " << datapath << std::endl;

    // Set data file path
    rocsparse_parse_data(argc, argv, datapath + "rocsparse_test.data");

    // Initialize google test
    testing::InitGoogleTest(&argc, argv);

    // Free up all temporary data generated during test creation
    test_cleanup::cleanup();

    // Remove the default listener
    auto& listeners       = testing::UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // Add our listener, by default everything is on (the same as using the default listener)
    // Here turning everything off so only the 3 lines for the result are visible
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto gtest_listener = getenv("GTEST_LISTENER");

    if(!gtest_listener || strcmp(gtest_listener, "VERBOSE_PASS_IN_LOG") != 0)
    {
        // If the GTEST_LISTENER environment variable is not set to "VERBOSE_PASS_IN_LOG",
        // we use the configurable_event_listener to capture output.
        // This listener will redirect the output to a stringstream and print it only if a test fails.
        // listeners.Append(new rocsparse_clients::configurable_event_listener(default_printer));
        auto listener = new rocsparse_clients::configurable_event_listener(default_printer);
        if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
        {
            listener->showTestNames = listener->showSuccesses = listener->showInlineFailures
                = false;
        }

        listeners.Append(listener);
    }
    else
    {
        auto listener = new rocsparse_clients::configurable_event_listener(default_printer);
        listener->redirectOutput = false;

        listeners.Append(listener);
    }

    // Run all tests
    int ret = RUN_ALL_TESTS();

    auto& reproducibility = rocsparse_reproducibility_t::instance();
    if(reproducibility.config().is_enabled())
    {
        static hipDeviceProp_t prop;
        static int             dev;
        if(hipGetDevice(&dev) != hipSuccess)
        {
            return -1;
        }
        if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
        {
            return -1;
        }
        reproducibility.config().set_gpu_name(prop.gcnArchName);
        rocsparse_reproducibility_write_report(reproducibility);
    }

    // Reset HIP device
    if(hipDeviceReset() != hipSuccess)
    {
        std::cerr << "Error: cannot reset HIP device" << std::endl;
        return rocsparse_status_internal_error;
    }

    return ret;
}
