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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define popen _popen
#define pclose _pclose
#define PATH_SEPARATOR "\\"
#else
#include <unistd.h>
#define PATH_SEPARATOR "/"
#endif

namespace fs = std::filesystem;

namespace {

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

} // namespace

class CPU_HipGraphExist_NONE : public ::testing::TestWithParam<HipGraphTestCase>
{
protected:
    std::string temp_dir;
    std::string driver_path;
    std::string rocprof_cmd;

    void SetUp() override
    {
        // Create temporary directory for test outputs using std::filesystem
        temp_dir = (fs::temp_directory_path() / "miopen_hip_graph_test").string();

        // Create directory if it doesn't exist
        fs::create_directories(temp_dir);

        // Determine driver path (cross-platform)
        // Assumes the test is run from build directory
#ifdef _WIN32
        driver_path = "bin\\MIOpenDriver.exe";
        rocprof_cmd = "rocprof.exe";
#else
        driver_path = "bin/MIOpenDriver";
        rocprof_cmd = "rocprof";
#endif

        // Check if driver exists
        if(!fs::exists(driver_path))
        {
            // Try alternative path
            driver_path = std::string("..") + PATH_SEPARATOR + driver_path;
        }
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

    bool FileContains(const std::string& filepath, const std::string& search_string)
    {
        std::ifstream file(filepath);
        if(!file.is_open())
        {
            return false;
        }

        std::string line;
        while(std::getline(file, line))
        {
            if(line.find(search_string) != std::string::npos)
            {
                return true;
            }
        }
        return false;
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

    void RunHipGraphTest(const std::string& driver_type,
                         const std::string& driver_args,
                         const std::string& test_name,
                         bool expect_graph)
    {
        // Capture stderr to reduce test noise and allow verification
        testing::internal::CaptureStderr();

        // Skip test if rocprof is not available
        std::string rocprof_check = ExecuteCommand(rocprof_cmd + " --version 2>&1");
        if(rocprof_check.empty() || rocprof_check.find("rocprof") == std::string::npos)
        {
            GTEST_SKIP() << "rocprof not available, skipping test";
        }

        // Skip test if driver doesn't exist
        if(!fs::exists(driver_path))
        {
            GTEST_SKIP() << "MIOpenDriver not found at: " << driver_path;
        }

        // Construct rocprof command
        std::string driver_output = temp_dir + PATH_SEPARATOR + test_name + "_output.txt";

        std::ostringstream cmd;

        // Change to temp directory first, then run rocprof from there
#ifdef _WIN32
        cmd << "cd /d \"" << temp_dir << "\" && ";
#else
        cmd << "cd \"" << temp_dir << "\" && ";
#endif

        // Run rocprof without --output-file (it will use default names: results.json, etc.)
        cmd << rocprof_cmd << " --hip-trace ";
        cmd << "\"" << fs::absolute(driver_path).string() << "\" " << driver_type << " ";
        cmd << driver_args;
        // Only add --use_hip_graph 1 if not already in driver_args and expect_graph is true
        if(expect_graph && driver_args.find("--use_hip_graph") == std::string::npos)
        {
            cmd << " --use_hip_graph 1";
        }
        cmd << " ";
#ifdef _WIN32
        cmd << "> \"" << driver_output << "\" 2>&1";
#else
        cmd << "2>&1 | tee \"" << driver_output << "\"";
#endif

        std::cout << "Executing command: " << cmd.str() << std::endl;

        // Execute the command
        int ret = std::system(cmd.str().c_str());

        // Check if command executed successfully
        // Note: rocprof may return non-zero even on success, so we check for output files
        std::cout << "Command return code: " << ret << std::endl;

        // Look for rocprof output files (rocprof creates files with default names in working dir)
        // rocprof default output files: results.json, results.hip_stats.csv, etc.
        std::vector<std::string> possible_trace_files = {
            temp_dir + PATH_SEPARATOR + "results.json",
            temp_dir + PATH_SEPARATOR + "results.hip_stats.csv",
            temp_dir + PATH_SEPARATOR + "results.csv",
            temp_dir + PATH_SEPARATOR + "hip_api_trace.txt",
            temp_dir + PATH_SEPARATOR + "results.txt"};

        std::string trace_file;

        // List directory contents for debugging
        std::cout << "Checking directory: " << temp_dir << std::endl;
        std::cout << "Directory contents:" << std::endl;
        for(const auto& entry : fs::directory_iterator(temp_dir))
        {
            std::cout << "  " << entry.path().filename().string() << std::endl;
        }

        // Find the trace file
        for(const auto& file : possible_trace_files)
        {
            if(fs::exists(file) && fs::file_size(file) > 0)
            {
                trace_file = file;
                std::cout << "Found trace file: " << trace_file << " (size: " << fs::file_size(file)
                          << " bytes)" << std::endl;
                break;
            }
        }

        // If no trace file found, fail with helpful message
        if(trace_file.empty())
        {
            FAIL() << "rocprof did not generate trace file. "
                   << "Expected files like results.json or results.hip_stats.csv in " << temp_dir;
        }

        // Parse the trace file for HIP Graph API calls
        // Note: MIOpen uses Stream Capture API, not explicit Graph Construction API
        int stream_begin_capture_count = CountOccurrences(trace_file, "hipStreamBeginCapture");
        int stream_end_capture_count   = CountOccurrences(trace_file, "hipStreamEndCapture");
        int graph_instantiate_count    = CountOccurrences(trace_file, "hipGraphInstantiate");
        int graph_launch_count         = CountOccurrences(trace_file, "hipGraphLaunch");
        int graph_destroy_count        = CountOccurrences(trace_file, "hipGraphDestroy");

        // Print results
        std::cout << "\n=== HIP Graph API Call Summary ===" << std::endl;
        std::cout << "hipStreamBeginCapture: " << stream_begin_capture_count << std::endl;
        std::cout << "hipStreamEndCapture:   " << stream_end_capture_count << std::endl;
        std::cout << "hipGraphInstantiate:   " << graph_instantiate_count << std::endl;
        std::cout << "hipGraphLaunch:        " << graph_launch_count << std::endl;
        std::cout << "hipGraphDestroy:       " << graph_destroy_count << std::endl;

        // Get captured stderr and verify no unexpected warnings
        std::string captured_stderr = testing::internal::GetCapturedStderr();

        // Verify no workspace warnings were emitted
        EXPECT_THAT(captured_stderr,
                    ::testing::Not(::testing::HasSubstr("Warning [IsEnoughWorkspace]")));

        if(expect_graph)
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
            bool hip_graph_detected =
                (stream_begin_capture_count > 0 && stream_end_capture_count > 0 &&
                 graph_instantiate_count > 0 && graph_launch_count > 0);

            ASSERT_TRUE(hip_graph_detected)
                << "HIP Graph was not properly created/executed for " << driver_type;
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
};

TEST_P(CPU_HipGraphExist_NONE, Test)
{
    const auto& test_case = GetParam();
    RunHipGraphTest(
        test_case.driver_type, test_case.driver_args, test_case.test_name, test_case.expect_graph);
}

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_HipGraphExist_NONE, testing::ValuesIn(GenSmokeTestCases()));
