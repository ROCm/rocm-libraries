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
#include <cstdlib>
#include <sstream>
#include <iostream>

#ifdef _WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

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

std::string ExecuteCommand(const std::string& command)
{
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if(!pipe)
    {
        return "";
    }

    char buffer[128];
    while(fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        result += buffer;
    }

    pclose(pipe);
    return result;
}

int CountOccurrences(const std::string& filepath, const std::string& search_string)
{
    std::ifstream file(filepath);
    if(!file.is_open())
    {
        return 0;
    }

    int count = 0;
    std::string line;
    while(std::getline(file, line))
    {
        size_t pos = 0;
        while((pos = line.find(search_string, pos)) != std::string::npos)
        {
            count++;
            pos += search_string.length();
        }
    }
    return count;
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
    // Use the same GPU mask as other MIOpenDriver tests
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx900>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>())
    {
        GTEST_SKIP();
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

    // Build the command to run MIOpenDriver with AMD_LOG_LEVEL for HIP API tracing
    // AMD_LOG_LEVEL=4 enables debug level logging which includes HIP API calls
    std::string output_file = temp_dir + PATH_SEPARATOR + test_case.test_name + "_output.txt";

    std::ostringstream cmd;

    // Set up environment and library path
    auto lib_path = fs::absolute(driver_path).parent_path().parent_path() / "lib";
#ifdef _WIN32
    cmd << "cd /d \"" << temp_dir << "\" && ";
    cmd << "set AMD_LOG_LEVEL=4 && ";
#else
    cmd << "cd \"" << temp_dir << "\" && ";
    cmd << "AMD_LOG_LEVEL=4 ";
    cmd << "LD_LIBRARY_PATH=\"" << lib_path.string() << ":$LD_LIBRARY_PATH\" ";
#endif

    // Run MIOpenDriver directly (no need for rocprof)
    cmd << "\"" << fs::absolute(driver_path).string() << "\" " << test_case.driver_type << " ";
    cmd << test_case.driver_args;

    // Only add --use_hip_graph 1 if not already in driver_args and expect_graph is true
    if(test_case.expect_graph && test_case.driver_args.find("--use_hip_graph") == std::string::npos)
    {
        cmd << " --use_hip_graph 1";
    }

    // Capture both stdout and stderr to the output file
    cmd << " > \"" << output_file << "\" 2>&1";

    std::cout << "Executing command: " << cmd.str() << std::endl;

    // Execute the command
    int ret = std::system(cmd.str().c_str());

    std::cout << "Command return code: " << ret << std::endl;

    // Read the output file
    std::string output_content;
    {
        std::ifstream output_stream(output_file);
        if(output_stream.is_open())
        {
            std::ostringstream ss;
            ss << output_stream.rdbuf();
            output_content = ss.str();
        }
    }

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
