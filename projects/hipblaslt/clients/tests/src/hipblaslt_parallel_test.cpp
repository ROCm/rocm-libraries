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

#include "hipblaslt_parallel_test.hpp"
#include "utility.hpp"

#ifndef _WIN32
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>
#include <cstdio>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

// Run a single GPU shard as a child process
// This function sets up env vars, redirects output, and execs the test binary
[[noreturn]] void run_child_shard(int argc, char** argv, int gpu, int num_gpus,
                                  const std::string& log_file,
                                  const std::string& gtest_output_base)
{
    // Set which GPU to use
    std::string gpu_env = std::to_string(gpu);
    setenv("HIP_VISIBLE_DEVICES", gpu_env.c_str(), 1);

    // Set optimal OpenMP threads per GPU process
    const char* env_threads = getenv("OMP_NUM_THREADS");
    int current_threads = env_threads ? std::atoi(env_threads) : std::thread::hardware_concurrency();
    int threads_per_gpu = std::max(1, current_threads / num_gpus);
    setenv("OMP_NUM_THREADS", std::to_string(threads_per_gpu).c_str(), 1);

    // Use Google Test's built-in sharding
    setenv("GTEST_TOTAL_SHARDS", std::to_string(num_gpus).c_str(), 1);
    setenv("GTEST_SHARD_INDEX", std::to_string(gpu).c_str(), 1);

    // Redirect output to log file to avoid interleaved output
    int fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(fd >= 0)
    {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
    }

    // Build argv - argv is already clean (custom flags removed by main)
    std::vector<const char*> new_argv;
    std::vector<std::string> arg_storage; // Store modified arguments

    new_argv.push_back(argv[0]);

    // Pass through all arguments from the clean argv
    for(int i = 1; i < argc; i++)
    {
        new_argv.push_back(argv[i]);
    }

    // Add per-GPU gtest_output if it was specified
    if(!gtest_output_base.empty())
    {
        size_t colon_pos = gtest_output_base.find(":");
        if(colon_pos != std::string::npos)
        {
            // Format: json:file.json
            std::string format = gtest_output_base.substr(0, colon_pos);
            std::string filename = gtest_output_base.substr(colon_pos + 1);

            // Insert GPU number before file extension
            size_t dot_pos = filename.rfind(".");
            std::string new_filename;
            if(dot_pos != std::string::npos)
            {
                new_filename = filename.substr(0, dot_pos) + "_gpu" +
                             std::to_string(gpu) + filename.substr(dot_pos);
            }
            else
            {
                new_filename = filename + "_gpu" + std::to_string(gpu);
            }

            std::string new_output_arg = "--gtest_output=" + format + ":" + new_filename;
            arg_storage.push_back(new_output_arg);
            new_argv.push_back(arg_storage.back().c_str());
        }
        else
        {
            // Format without colon, just pass through with default filename
            std::string new_output_arg = "--gtest_output=" + gtest_output_base;
            arg_storage.push_back(new_output_arg);
            new_argv.push_back(arg_storage.back().c_str());
        }
    }

    new_argv.push_back(nullptr);

    // Execute the test binary
    execvp(argv[0], const_cast<char* const*>(new_argv.data()));

    // If exec fails
    hipblaslt_cerr << "Failed to exec for GPU " << gpu << std::endl;
    exit(1);
}

// Function to run tests in parallel across multiple GPUs
// argc/argv should already have --num_gpus and --gtest_output stripped
int run_tests_parallel_gpus(int argc, char** argv, int num_gpus,
                            const std::string& gtest_output_base)
{
    hipblaslt_cout << "\n========================================" << std::endl;
    hipblaslt_cout << "Parallel GPU Execution Mode" << std::endl;
    hipblaslt_cout << "Running tests across " << num_gpus << " GPUs" << std::endl;
    hipblaslt_cout << "========================================\n" << std::endl;

    // Check available GPUs
    int available_gpus = 0;
    if(hipGetDeviceCount(&available_gpus) != hipSuccess || available_gpus < 1)
    {
        hipblaslt_cerr << "Error: No GPUs detected" << std::endl;
        return 1;
    }

    if(num_gpus > available_gpus)
    {
        hipblaslt_cerr << "Warning: Requested " << num_gpus << " GPUs but only "
                       << available_gpus << " available. Using " << available_gpus << " GPUs."
                       << std::endl;
        num_gpus = available_gpus;
    }

    // Display sharding information
    hipblaslt_cout << "Tests will be sharded across " << num_gpus << " GPUs" << std::endl;

    // Calculate and display OpenMP thread distribution
    const char* env_threads = getenv("OMP_NUM_THREADS");
    int current_threads = env_threads ? std::atoi(env_threads) : std::thread::hardware_concurrency();
    int threads_per_gpu = std::max(1, current_threads / num_gpus);
    hipblaslt_cout << "OpenMP threads per GPU: " << threads_per_gpu
                   << " (total available: " << current_threads << ")" << std::endl;
    hipblaslt_cout << std::endl;

    // Split tests across GPUs using Google Test's built-in sharding
    std::vector<pid_t> child_pids;
    std::vector<std::string> output_files;
    bool fork_failed = false;

    for(int gpu = 0; gpu < num_gpus; gpu++)
    {
        std::string output_file = "/tmp/hipblaslt_gpu" + std::to_string(gpu) + "_" +
                                  std::to_string(getpid()) + ".log";
        output_files.push_back(output_file);

        hipblaslt_cout << "GPU " << gpu << ": Starting shard " << gpu << std::endl;

        pid_t pid = fork();
        if(pid == 0)
        {
            // Child process
            run_child_shard(argc, argv, gpu, num_gpus, output_file, gtest_output_base);
        }
        else if(pid > 0)
        {
            // Parent process
            child_pids.push_back(pid);
        }
        else
        {
            // Fork failed - need to clean up already-forked children
            hipblaslt_cerr << "Error: Failed to fork for GPU " << gpu << std::endl;
            fork_failed = true;
            break;
        }
    }

    // If fork failed, terminate and wait for any already-started children
    if(fork_failed)
    {
        hipblaslt_cerr << "Terminating " << child_pids.size() << " already-started child processes..." << std::endl;

        // Send SIGTERM to all children
        for(pid_t pid : child_pids)
        {
            kill(pid, SIGTERM);
        }

        // Wait for all children to terminate
        for(pid_t pid : child_pids)
        {
            int status;
            waitpid(pid, &status, 0);
        }

        return 1;
    }

    // Wait for all children and collect results (without printing yet)
    hipblaslt_cout << "\nWaiting for all GPUs to complete..." << std::endl;

    int total_exit_code = 0;
    int gpus_passed     = 0;
    int gpus_failed     = 0;
    int total_tests_ran    = 0;
    int total_tests_passed = 0;
    int total_tests_failed = 0;
    std::vector<int> exit_codes(child_pids.size());
    std::vector<bool> normal_exit(child_pids.size());
    std::vector<std::string> gpu_summaries(child_pids.size());
    std::vector<int> gpu_tests_ran(child_pids.size(), 0);
    std::vector<int> gpu_tests_passed(child_pids.size(), 0);
    std::vector<int> gpu_tests_failed(child_pids.size(), 0);
    std::vector<double> gpu_time_ms(child_pids.size(), 0.0);

    // Wait for ALL children first before printing any results
    for(size_t i = 0; i < child_pids.size(); i++)
    {
        int status;
        waitpid(child_pids[i], &status, 0);

        if(WIFEXITED(status))
        {
            normal_exit[i] = true;
            exit_codes[i]  = WEXITSTATUS(status);
            if(exit_codes[i] == 0)
            {
                gpus_passed++;
            }
            else
            {
                gpus_failed++;
                total_exit_code = 1;
            }
        }
        else
        {
            normal_exit[i] = false;
            exit_codes[i]  = -1;
            gpus_failed++;
            total_exit_code = 1;
        }

        // Extract summary from log file and parse test counts
        std::ifstream log_file(output_files[i]);
        if(log_file.is_open())
        {
            std::string line;
            std::vector<std::string> summary_lines;
            bool in_summary = false;

            while(std::getline(log_file, line))
            {
                // Parse total tests ran: "[==========] 87 tests from 1 test suite ran."
                if(line.find("[==========]") != std::string::npos && line.find("tests") != std::string::npos && line.find("ran.") != std::string::npos)
                {
                    in_summary = true;
                    // Extract number of tests
                    size_t pos = line.find("]");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_tests_ran[i];
                    }
                }

                // Parse passed tests: "[  PASSED  ] 87 tests."
                if(line.find("[  PASSED  ]") != std::string::npos && line.find("tests") != std::string::npos)
                {
                    size_t pos = line.find("]");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_tests_passed[i];
                    }
                }

                // Parse failed tests: "[  FAILED  ] 2 tests, listed below:"
                if(line.find("[  FAILED  ]") != std::string::npos && line.find("tests") != std::string::npos)
                {
                    size_t pos = line.find("]");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_tests_failed[i];
                    }
                }

                // Parse time: "[==========] 87 tests from 1 test suite ran. (1234 ms total)"
                if(line.find("[==========]") != std::string::npos && line.find("ms total") != std::string::npos)
                {
                    size_t pos = line.find("(");
                    if(pos != std::string::npos)
                    {
                        std::istringstream iss(line.substr(pos + 1));
                        iss >> gpu_time_ms[i];
                    }
                }

                if(in_summary)
                {
                    summary_lines.push_back(line);
                }
            }
            log_file.close();

            // Store the summary for this GPU
            if(!summary_lines.empty())
            {
                for(const auto& summary_line : summary_lines)
                {
                    gpu_summaries[i] += summary_line + "\n";
                }
            }

            // Accumulate totals
            total_tests_ran += gpu_tests_ran[i];
            total_tests_passed += gpu_tests_passed[i];
            total_tests_failed += gpu_tests_failed[i];
        }
    }

    // Now print all results together
    hipblaslt_cout << "\n========================================" << std::endl;
    hipblaslt_cout << "Parallel GPU Test Summary" << std::endl;
    hipblaslt_cout << "========================================\n" << std::endl;

    // Print individual GPU results with their GTest summaries
    for(size_t i = 0; i < child_pids.size(); i++)
    {
        hipblaslt_cout << "GPU " << i << ":" << std::endl;

        if(!gpu_summaries[i].empty())
        {
            hipblaslt_cout << gpu_summaries[i];
        }
        else if(normal_exit[i])
        {
            hipblaslt_cout << "  Exit code: " << exit_codes[i] << std::endl;
        }
        else
        {
            hipblaslt_cout << "  Terminated abnormally" << std::endl;
        }
        hipblaslt_cout << std::endl;
    }

    // Calculate average time
    double total_time_ms = 0.0;
    int valid_times = 0;
    for(size_t i = 0; i < gpu_time_ms.size(); i++)
    {
        if(gpu_time_ms[i] > 0.0)
        {
            total_time_ms += gpu_time_ms[i];
            valid_times++;
        }
    }
    double avg_time_ms = (valid_times > 0) ? (total_time_ms / valid_times) : 0.0;

    hipblaslt_cout << "----------------------------------------" << std::endl;
    hipblaslt_cout << "OVERALL SUMMARY (across all GPUs):" << std::endl;
    hipblaslt_cout << "Total tests run:  " << total_tests_ran << std::endl;
    hipblaslt_cout << "Total PASSED:     " << total_tests_passed << std::endl;
    hipblaslt_cout << "Total FAILED:     " << total_tests_failed << std::endl;
    if(avg_time_ms > 0.0)
    {
        hipblaslt_cout << "Average time:     " << avg_time_ms << " ms" << std::endl;
    }
    hipblaslt_cout << "\nGPU Summary:" << std::endl;
    hipblaslt_cout << "GPUs used:        " << num_gpus << std::endl;
    hipblaslt_cout << "GPUs passed:      " << gpus_passed << std::endl;
    hipblaslt_cout << "GPUs failed:      " << gpus_failed << std::endl;

    // Clean up log files on success to avoid accumulation in /tmp on CI systems
    if(total_exit_code == 0)
    {
        hipblaslt_cout << "\nAll tests passed - cleaning up log files" << std::endl;
        for(size_t i = 0; i < output_files.size(); i++)
        {
            remove(output_files[i].c_str());
        }
    }
    else
    {
        hipblaslt_cout << "\nLog files saved in:" << std::endl;
        for(size_t i = 0; i < output_files.size(); i++)
        {
            hipblaslt_cout << "  GPU " << i << ": " << output_files[i] << std::endl;
        }
    }
    hipblaslt_cout << "========================================\n" << std::endl;

    return total_exit_code;
}
#endif // _WIN32
