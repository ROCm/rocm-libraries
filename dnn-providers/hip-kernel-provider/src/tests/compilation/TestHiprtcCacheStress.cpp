// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Several processes, each with several threads, sharing one HIPRTC cache directory. Each case
// re-runs this binary as child processes (filtered to the same case, with a role in the
// environment), so every child starts with a fresh HIP runtime instead of a forked copy of the
// parent's.

#if !defined(_WIN32)

#include <gtest/gtest.h>

#include "compilation/HiprtcDiskCache.hpp"
#include "compilation/Program.hpp"
#include "engines/hip_mlops_engine/HipMlopsModuleCache.hpp"

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <spawn.h>
#include <sys/wait.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

extern char** environ; // NOLINT(readability-redundant-declaration)

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::compilation;

namespace
{

/// Set in a child's environment to the work it does; unset in the parent.
constexpr const char* ROLE_ENV = "HIP_KERNEL_PROVIDER_TEST_CACHE_STRESS_ROLE";

constexpr int PROCESSES = 4;
constexpr int THREADS = 2;

std::string role()
{
    return hipdnn_data_sdk::utilities::getEnv(ROLE_ENV);
}

/// The gtest filter that selects the running case alone.
std::string currentCaseFilter()
{
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    return std::string(info->test_suite_name()) + "." + info->name();
}

/// Runs this binary @p count times at once, each filtered to the running case with ROLE_ENV set
/// to @p childRole, and inheriting the rest of this environment (the cache directory among it).
/// @return How many children failed to start or exited with a failure.
int runChildren(const std::string& childRole, int count)
{
    const auto executable = std::filesystem::read_symlink("/proc/self/exe").string();
    const auto filter = "--gtest_filter=" + currentCaseFilter();
    const auto roleSetting = std::string(ROLE_ENV) + "=" + childRole;

    std::vector<char*> environment;
    for(char** entry = environ; *entry != nullptr; ++entry)
    {
        environment.push_back(*entry);
    }
    environment.push_back(const_cast<char*>(roleSetting.c_str()));
    environment.push_back(nullptr);

    std::vector<pid_t> children;
    int failures = 0;
    for(int i = 0; i < count; ++i)
    {
        std::array<char*, 3> arguments
            = {const_cast<char*>(executable.c_str()), const_cast<char*>(filter.c_str()), nullptr};
        pid_t child = 0;
        if(posix_spawn(
               &child, executable.c_str(), nullptr, nullptr, arguments.data(), environment.data())
           != 0)
        {
            ++failures;
            continue;
        }
        children.push_back(child);
    }
    for(const auto child : children)
    {
        int status = 0;
        if(waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        {
            ++failures;
        }
    }
    return failures;
}

std::vector<std::filesystem::path> filesWithExtension(const std::filesystem::path& directory,
                                                      const std::string& extension)
{
    std::vector<std::filesystem::path> found;
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(directory, error))
    {
        if(entry.path().extension() == extension)
        {
            found.push_back(entry.path());
        }
    }
    return found;
}

constexpr int KEYS = 6;
constexpr int ROUNDS = 40;

std::string stressKey(int index)
{
    return std::string(63, '0') + static_cast<char>('a' + index);
}

/// A payload of a size and content only @p index decides, so any reader can check one.
std::vector<char> stressPayload(int index)
{
    std::vector<char> bytes(static_cast<size_t>(40000 + index * 53000));
    for(size_t i = 0; i < bytes.size(); ++i)
    {
        bytes[i] = static_cast<char>((i * 31 + static_cast<size_t>(index) * 97) & 0xFF);
    }
    return bytes;
}

} // namespace

// Four processes of two threads read and rewrite the same six entries: every read is either
// absent or the complete, verified payload -- never a partial file.
TEST(TestHiprtcCacheStress, ConcurrentProcessesNeverReadATornEntry)
{
    if(role() == "entries")
    {
        std::atomic<int> torn{0};
        std::atomic<int> wrong{0};
        std::vector<std::thread> threads;
        threads.reserve(THREADS);
        for(int thread = 0; thread < THREADS; ++thread)
        {
            threads.emplace_back([&, thread] {
                for(int round = 0; round < ROUNDS; ++round)
                {
                    for(int index = 0; index < KEYS; ++index)
                    {
                        const auto path = hiprtc_cache::codeObjectPath(stressKey(index));
                        const auto read = hiprtc_cache::readCodeObject(path);
                        if(read.outcome == hiprtc_cache::ReadOutcome::CORRUPT)
                        {
                            ++torn;
                        }
                        else if(read.outcome == hiprtc_cache::ReadOutcome::VALID
                                && read.codeObject != stressPayload(index))
                        {
                            ++wrong;
                        }
                        // Rewrite present entries too, so renames land over files other
                        // processes are reading.
                        if(read.outcome == hiprtc_cache::ReadOutcome::ABSENT
                           || (round + thread + index) % 3 == 0)
                        {
                            hiprtc_cache::writeCodeObject(path, stressPayload(index));
                        }
                    }
                }
            });
        }
        for(auto& thread : threads)
        {
            thread.join();
        }
        EXPECT_EQ(torn.load(), 0);
        EXPECT_EQ(wrong.load(), 0);
        return;
    }

    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hiprtc-stress", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);
    const auto directory = hiprtc_cache::codeObjectPath(stressKey(0)).parent_path();
    ASSERT_FALSE(directory.empty());

    EXPECT_EQ(runChildren("entries", PROCESSES), 0);

    for(int index = 0; index < KEYS; ++index)
    {
        const auto read
            = hiprtc_cache::readCodeObject(hiprtc_cache::codeObjectPath(stressKey(index)));
        EXPECT_EQ(read.outcome, hiprtc_cache::ReadOutcome::VALID) << read.problem;
        EXPECT_EQ(read.codeObject, stressPayload(index));
    }
    EXPECT_EQ(filesWithExtension(directory, ".co").size(), static_cast<size_t>(KEYS));
    EXPECT_TRUE(filesWithExtension(directory, ".tmp").empty());
}

// Four processes of two threads compile the same small programs through one module cache each,
// sharing one cache directory: no compile fails, each process holds one module per program
// (both its threads get the same one), and the directory ends with one valid entry per program.
TEST(TestGpuHiprtcCacheStress, ConcurrentProcessesCompileOncePerProcessAndShareEntries)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<std::vector<std::string>> programs
        = {{"-O3", "-DFLOAT=float"}, {"-O2", "-DFLOAT=float"}, {"-O3", "-DFLOAT=double"}};

    if(role() == "programs")
    {
        HipMlopsModuleCache cache;
        std::vector<std::vector<std::shared_ptr<Program>>> obtained(
            THREADS, std::vector<std::shared_ptr<Program>>(programs.size()));
        std::vector<std::thread> threads;
        threads.reserve(THREADS);
        for(int thread = 0; thread < THREADS; ++thread)
        {
            threads.emplace_back([&, thread] {
                for(size_t i = 0; i < programs.size(); ++i)
                {
                    obtained[static_cast<size_t>(thread)][i]
                        = cache.getOrLoad("vector_add.cpp", programs[i]);
                }
            });
        }
        for(auto& thread : threads)
        {
            thread.join();
        }
        for(size_t i = 0; i < programs.size(); ++i)
        {
            ASSERT_NE(obtained[0][i], nullptr) << "program " << i << " failed to build";
            EXPECT_NE(obtained[0][i]->getKernel("vector_add"), nullptr);
            for(int thread = 1; thread < THREADS; ++thread)
            {
                EXPECT_EQ(obtained[static_cast<size_t>(thread)][i], obtained[0][i])
                    << "program " << i << " was built twice in one process";
            }
        }
        EXPECT_EQ(cache.size(), programs.size());
        return;
    }

    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hiprtc-stress-gpu", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);
    const auto directory = hiprtc_cache::codeObjectPath(std::string(64, '0')).parent_path();
    ASSERT_FALSE(directory.empty());

    EXPECT_EQ(runChildren("programs", PROCESSES), 0);

    const auto entries = filesWithExtension(directory, ".co");
    EXPECT_EQ(entries.size(), programs.size());
    for(const auto& entry : entries)
    {
        const auto read = hiprtc_cache::readCodeObject(entry);
        EXPECT_EQ(read.outcome, hiprtc_cache::ReadOutcome::VALID) << entry << ": " << read.problem;
    }
    EXPECT_TRUE(filesWithExtension(directory, ".tmp").empty());
}

#endif // !_WIN32
