// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "GpuProbeArgs.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

static std::optional<int32_t> parse(std::vector<const char*> args)
{
    args.insert(args.begin(), "gpu-probe");
    return parseGpuProbeArgs(static_cast<int>(args.size()), args.data());
}

TEST(GpuProbeArgs, NoArgsSelectsDeviceZero)
{
    EXPECT_EQ(parse({}), 0);
}

TEST(GpuProbeArgs, DeviceIndexIsParsed)
{
    EXPECT_EQ(parse({"--device-idx", "3"}), 3);
    EXPECT_EQ(parse({"--device-idx", "0"}), 0);
    EXPECT_EQ(parse({"--device-idx", "2147483647"}), INT32_MAX);
}

TEST(GpuProbeArgs, LastDeviceIndexWins)
{
    EXPECT_EQ(parse({"--device-idx", "1", "--device-idx", "2"}), 2);
}

TEST(GpuProbeArgs, MissingValueIsRejected)
{
    EXPECT_FALSE(parse({"--device-idx"}));
    EXPECT_FALSE(parse({"--device-idx", "1", "--device-idx"}));
}

TEST(GpuProbeArgs, InvalidValueIsRejected)
{
    EXPECT_FALSE(parse({"--device-idx", "-1"}));
    EXPECT_FALSE(parse({"--device-idx", "abc"}));
    EXPECT_FALSE(parse({"--device-idx", ""}));
    EXPECT_FALSE(parse({"--device-idx", "3x"}));
}

TEST(GpuProbeArgs, OutOfRangeValueIsRejected)
{
    EXPECT_FALSE(parse({"--device-idx", "2147483648"}));
    EXPECT_FALSE(parse({"--device-idx", "99999999999999999999"}));
}

TEST(GpuProbeArgs, UnknownArgumentIsRejected)
{
    EXPECT_FALSE(parse({"--bogus"}));
    EXPECT_FALSE(parse({"3"}));
    EXPECT_FALSE(parse({"--device-idx", "1", "--bogus"}));
}

#if defined(GPU_PROBE_PATH) && !defined(_WIN32)

namespace
{
    struct ProbeResult
    {
        int         exitCode = -1;
        int         signal   = 0;
        std::string out;
        std::string err;
    };
} // namespace

static std::string readAll(int fd)
{
    std::string text;
    char        chunk[4096];
    while(true)
    {
        ssize_t count = read(fd, chunk, sizeof(chunk));
        if(count < 0 && errno == EINTR)
            continue;
        if(count <= 0)
            return text;
        text.append(chunk, static_cast<size_t>(count));
    }
}

static void closePipe(int (&fds)[2])
{
    close(fds[0]);
    close(fds[1]);
}

// The probe prints a few lines at most, so reading the pipes one after the other cannot block it.
static ProbeResult runProbe(std::vector<const char*> args)
{
    ProbeResult result;
    int         outPipe[2];
    int         errPipe[2];
    if(pipe(outPipe) != 0)
    {
        ADD_FAILURE() << "pipe failed: " << strerror(errno);
        return result;
    }
    if(pipe(errPipe) != 0)
    {
        ADD_FAILURE() << "pipe failed: " << strerror(errno);
        closePipe(outPipe);
        return result;
    }

    args.insert(args.begin(), GPU_PROBE_PATH);
    args.push_back(nullptr);

    pid_t pid = fork();
    if(pid < 0)
    {
        ADD_FAILURE() << "fork failed: " << strerror(errno);
        closePipe(outPipe);
        closePipe(errPipe);
        return result;
    }
    if(pid == 0)
    {
        dup2(outPipe[1], STDOUT_FILENO);
        dup2(errPipe[1], STDERR_FILENO);
        closePipe(outPipe);
        closePipe(errPipe);
        // A faulted GPU can stall the probe; the alarm survives exec and ends it.
        alarm(60);
        execv(GPU_PROBE_PATH, const_cast<char* const*>(args.data()));
        _exit(127);
    }
    close(outPipe[1]);
    close(errPipe[1]);
    result.out = readAll(outPipe[0]);
    result.err = readAll(errPipe[0]);
    close(outPipe[0]);
    close(errPipe[0]);

    int   status = 0;
    pid_t waited = 0;
    do
        waited = waitpid(pid, &status, 0);
    while(waited < 0 && errno == EINTR);
    if(waited != pid)
        return result;
    if(WIFEXITED(status))
        result.exitCode = WEXITSTATUS(status);
    if(WIFSIGNALED(status))
        result.signal = WTERMSIG(status);
    return result;
}

static void expectProbeRan(ProbeResult const& result)
{
    ASSERT_NE(result.exitCode, 127) << "probe could not be executed";
    ASSERT_EQ(result.signal, 0) << "probe died by signal " << result.signal;
}

static void expectUsageError(std::vector<const char*> args)
{
    ProbeResult result = runProbe(args);
    expectProbeRan(result);
    EXPECT_EQ(result.exitCode, 2);
    EXPECT_NE(result.err.find("usage:"), std::string::npos) << result.err;
    EXPECT_TRUE(result.out.empty()) << result.out;
}

TEST(GpuProbeExitCode, UnknownFlagIsUsageError)
{
    expectUsageError({"--bogus"});
}

TEST(GpuProbeExitCode, MissingValueIsUsageError)
{
    expectUsageError({"--device-idx"});
}

TEST(GpuProbeExitCode, NegativeValueIsUsageError)
{
    expectUsageError({"--device-idx", "-1"});
}

TEST(GpuProbeExitCode, NonexistentDeviceIsFailure)
{
    ProbeResult result = runProbe({"--device-idx", "1000"});
    expectProbeRan(result);
    EXPECT_EQ(result.exitCode, 1);
    EXPECT_NE(result.err.find("gpu probe failed on device 1000"), std::string::npos) << result.err;
    EXPECT_NE(result.err.find("hip error"), std::string::npos) << result.err;
    EXPECT_TRUE(result.out.empty()) << result.out;
}

#else

TEST(GpuProbeExitCode, RequiresProbeExecutable)
{
    GTEST_SKIP() << "tensilelite-gpu-probe is not part of this build";
}

#endif
