/*******************************************************************************
 *
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include "miopendriver_common.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/process.hpp>
#include <miopen/filesystem.hpp>

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

namespace fs = miopen::fs;

using ::testing::HasSubstr;
using ::testing::Not;

namespace hip_graph_exist {

struct HipGraphTestCase
{
    std::string driver_type;
    std::string driver_args;
    std::string test_name;
    bool expect_graph; // true if we expect HIP graph to be created

    friend std::ostream& operator<<(std::ostream& os, const HipGraphTestCase& tc)
    {
        return os << tc.driver_type << "_" << tc.test_name;
    }
};

std::vector<HipGraphTestCase> GenSmokeTestCases()
{
    // Use smaller shapes and --iter=1 to speed up tests
    return {
        {"conv",
         "-n 1 -c 3 -H 32 -W 32 -k 16 -y 3 -x 3 "
         "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 1",
         "conv_hip_graph",
         true},
        {"activ",
         "-n 8 -c 3 -H 16 -W 16 -m 3 -A 1 -B 1 -G 1 -F 0 -i 1 -V 1 -t 1",
         "activ_hip_graph",
         true},
        {"bnorm", "-F 2 -n 8 -c 64 -H 8 -W 8 -m 1 -r 1 -i 1 -V 1 -t 1", "bnorm_hip_graph", true},
        {"conv",
         "-n 1 -c 3 -H 32 -W 32 -k 16 -y 3 -x 3 "
         "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 1 --use_hip_graph 0",
         "no_graph",
         false}};
}

class GPU_HipGraphExistTest_FP32 : public testing::TestWithParam<HipGraphTestCase>
{
protected:
    std::string temp_dir;

    void SetUp() override
    {
        // Create temporary directory for test outputs
        temp_dir = (fs::temp_directory_path() / "miopen_hip_graph_test").string();
        fs::create_directories(temp_dir);
    }

    void TearDown() override
    {
        // Clean up temporary files
        try
        {
            if(fs::exists(temp_dir))
            {
                fs::remove_all(temp_dir);
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << "Warning: Failed to clean up temp directory: " << e.what() << std::endl;
        }
    }
};

// Count occurrences of a pattern in a string
int CountOccurrencesInString(const std::string& text, const std::string& search_string)
{
    int count  = 0;
    size_t pos = 0;
    while((pos = text.find(search_string, pos)) != std::string::npos)
    {
        count++;
        pos += search_string.length();
    }
    return count;
}

void RunHipGraphTest(const HipGraphTestCase& test_case, const std::string& temp_dir)
{
    // In CI/CD, run only bnorm_hip_graph (/2) to avoid LLVM OOM issues
    // Developers can run all tests locally
    const char* ci_env      = std::getenv("CI");
    const char* jenkins_env = std::getenv("JENKINS_URL");
    bool is_ci              = (ci_env != nullptr) || (jenkins_env != nullptr);

    if(is_ci && test_case.test_name != "bnorm_hip_graph")
    {
        GTEST_SKIP() << "Skipping " << test_case.test_name
                     << " in CI/CD (only bnorm_hip_graph runs in CI)";
    }

    // Get driver path using the common function
    const auto driver_path = MIOpenDriverExePath();
    if(driver_path.empty() || !fs::exists(driver_path))
    {
        GTEST_SKIP() << "MIOpenDriver not found at: " << driver_path.string();
    }

    // Capture stderr to reduce test noise and allow verification
    // NOTE: CaptureStderr must be called AFTER all GTEST_SKIP() checks,
    // otherwise GetCapturedStderr() won't be called and stderr will remain
    // redirected, causing issues for subsequent tests.
    testing::internal::CaptureStderr();

    // Build command arguments
    std::ostringstream args;
    args << test_case.driver_type << " " << test_case.driver_args;

    // Only add --use_hip_graph 1 if not already in driver_args and expect_graph is true
    if(test_case.expect_graph && test_case.driver_args.find("--use_hip_graph") == std::string::npos)
    {
        args << " --use_hip_graph 1";
    }

    // Set up environment variables for HIP API tracing
    // AMD_LOG_LEVEL=4 enables debug level logging which includes HIP API calls
    miopen::ProcessEnvironmentMap envVars;
    envVars["AMD_LOG_LEVEL"] = "4";

    std::cout << "Executing: " << driver_path.string() << " " << args.str() << std::endl;

    // Execute using miopen::Process with output capture
    std::stringstream output_stream;
    miopen::Process process{driver_path};
    int ret = process(args.str(), temp_dir, &output_stream, envVars);

    std::string output_content = output_stream.str();

    std::cout << "Command return code: " << ret << std::endl;

    // Check if command executed successfully
    // MIOpenDriver should return 0 on success
    if(ret != 0)
    {
        std::string captured_stderr = testing::internal::GetCapturedStderr();
        std::cout << "Driver output:\n" << output_content << std::endl;
        FAIL() << "MIOpenDriver failed with return code " << ret;
    }

    // Parse the output for HIP Graph API calls
    // AMD_LOG_LEVEL=4 will print HIP API calls like:
    // :HIP_API: hipStreamBeginCapture ...
    // :HIP_API: hipGraphInstantiate ...
    // :HIP_API: hipGraphLaunch ...

    // Count HIP Graph related API calls in the output
    int stream_begin_capture_count =
        CountOccurrencesInString(output_content, "hipStreamBeginCapture");
    int stream_end_capture_count = CountOccurrencesInString(output_content, "hipStreamEndCapture");
    int graph_instantiate_count  = CountOccurrencesInString(output_content, "hipGraphInstantiate");
    int graph_launch_count       = CountOccurrencesInString(output_content, "hipGraphLaunch");
    int graph_destroy_count      = CountOccurrencesInString(output_content, "hipGraphDestroy");

    // Print results
    std::cout << "\n=== HIP Graph API Call Summary (from AMD_LOG_LEVEL output) ===" << std::endl;
    std::cout << "hipStreamBeginCapture: " << stream_begin_capture_count << std::endl;
    std::cout << "hipStreamEndCapture:   " << stream_end_capture_count << std::endl;
    std::cout << "hipGraphInstantiate:   " << graph_instantiate_count << std::endl;
    std::cout << "hipGraphLaunch:        " << graph_launch_count << std::endl;
    std::cout << "hipGraphDestroy:       " << graph_destroy_count << std::endl;

    // Get captured stderr and verify no unexpected warnings
    std::string captured_stderr = testing::internal::GetCapturedStderr();

    // Verify no workspace warnings were emitted
    EXPECT_THAT(captured_stderr, Not(HasSubstr("Warning [IsEnoughWorkspace]")));

    if(test_case.expect_graph)
    {
        // Verify HIP Graph was created via Stream Capture API
        EXPECT_GT(stream_begin_capture_count, 0)
            << "hipStreamBeginCapture not called - HIP Graph capture was not started";
        EXPECT_GT(stream_end_capture_count, 0)
            << "hipStreamEndCapture not called - HIP Graph was not created from stream";
        EXPECT_GT(graph_instantiate_count, 0)
            << "hipGraphInstantiate not called - HIP Graph was not instantiated";
        EXPECT_GT(graph_launch_count, 0)
            << "hipGraphLaunch not called - HIP Graph was not executed";

        // Overall success check
        bool hip_graph_detected = (stream_begin_capture_count > 0 && stream_end_capture_count > 0 &&
                                   graph_instantiate_count > 0 && graph_launch_count > 0);

        ASSERT_TRUE(hip_graph_detected) << "HIP Graph was not properly created/executed for "
                                        << test_case.driver_type << ". Output content:\n"
                                        << output_content.substr(0, 2000);
    }
    else
    {
        // Verify HIP Graph was NOT created
        EXPECT_EQ(stream_begin_capture_count, 0)
            << "hipStreamBeginCapture was called even though HIP Graph should be disabled";
        EXPECT_EQ(stream_end_capture_count, 0)
            << "hipStreamEndCapture was called even though HIP Graph should be disabled";
        EXPECT_EQ(graph_launch_count, 0)
            << "hipGraphLaunch was called even though HIP Graph should be disabled";
    }
}

} // namespace hip_graph_exist

using namespace hip_graph_exist;

TEST_P(GPU_HipGraphExistTest_FP32, HipGraphExist)
{
    const auto& test_case = GetParam();
    RunHipGraphTest(test_case, temp_dir);
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_HipGraphExistTest_FP32, testing::ValuesIn(GenSmokeTestCases()));
