// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

// PlatformUtils.hpp includes <windows.h> on Windows with NOMINMAX and WIN32_LEAN_AND_MEAN.
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#ifndef _WIN32
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace hipdnn_test_sdk::utilities
{

/// Full path of the running executable. Throws std::runtime_error when it cannot be read.
inline std::filesystem::path currentExecutablePath()
{
#ifdef _WIN32
    std::string selfPath(MAX_PATH, '\0');
    const DWORD length
        = GetModuleFileNameA(nullptr, selfPath.data(), static_cast<DWORD>(selfPath.size()));
    if(length == 0 || length >= selfPath.size())
    {
        throw std::runtime_error("currentExecutablePath: GetModuleFileNameA failed");
    }
#else
    std::string selfPath(4096, '\0');
    const auto length = readlink("/proc/self/exe", selfPath.data(), selfPath.size() - 1);
    if(length <= 0 || static_cast<size_t>(length) >= selfPath.size() - 1)
    {
        throw std::runtime_error("currentExecutablePath: readlink(/proc/self/exe) failed");
    }
#endif
    selfPath.resize(static_cast<size_t>(length));
    return selfPath;
}

/// True when a shared library with the file name of @p libraryPath is loaded in this process.
inline bool isLibraryLoaded(const std::filesystem::path& libraryPath)
{
    const std::string fileName = libraryPath.filename().string();
#ifdef _WIN32
    return GetModuleHandleA(fileName.c_str()) != nullptr;
#else
    const std::string pathSuffix = "/" + fileName;
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while(std::getline(maps, line))
    {
        if(line.find(pathSuffix) != std::string::npos)
        {
            return true;
        }
    }
    return false;
#endif
}

/// Runs the current test again in a child process of this executable, filtered to that test,
/// and returns the child's exit code, so 0 when the child exits with status 0. On POSIX a child
/// killed by a signal returns minus the signal number (a shell between the two may instead
/// report 128 plus the signal number as its own exit code); on Windows a crash returns the
/// exception code. The child inherits this process's environment; set what it needs before
/// calling.
inline int runCurrentTestInChild()
{
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    if(info == nullptr)
    {
        throw std::runtime_error("runCurrentTestInChild: no test is running");
    }
    const std::string command = "\"" + currentExecutablePath().string() + "\" --gtest_filter="
                                + info->test_suite_name() + "." + info->name();
    const int status = std::system(command.c_str());
#ifdef _WIN32
    return status;
#else
    if(status == -1)
    {
        throw std::runtime_error("runCurrentTestInChild: could not start the child");
    }
    if(WIFSIGNALED(status))
    {
        return -WTERMSIG(status);
    }
    return WEXITSTATUS(status);
#endif
}

} // namespace hipdnn_test_sdk::utilities
