// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// miopen::Process runs commands directly, without a shell, so shell metacharacters in
// arguments, environment values and the working directory must reach the child literally
// (ROCM-26841 / ROCM-27008). Each injection probe would create a marker file if a shell
// interpreted it.

#include <gtest/gtest.h>

#include <miopen/filesystem.hpp>
#include <miopen/process.hpp>
#include <miopen/tmp_dir.hpp>

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32

namespace {

struct CPU_Process_NONE : ::testing::Test
{
    miopen::TmpDir dir{"process_test"};
    miopen::fs::path marker = dir / "pwned";

    void ExpectNoMarker() const
    {
        EXPECT_FALSE(miopen::fs::exists(marker)) << "a shell interpreted the input";
    }
};

} // namespace

// TEST_F stays outside the anonymous namespace: cppcheck otherwise reports a false syntaxError
// (https://trac.cppcheck.net/ticket/9160).

TEST_F(CPU_Process_NONE, ArgumentMetacharactersAreLiteral)
{
    const auto m = marker.string();
    for(const auto& args : std::vector<std::string>{"a;touch " + m,
                                                    "a&&touch " + m,
                                                    "a||touch " + m,
                                                    "a|touch " + m,
                                                    "a&touch " + m,
                                                    "$(touch " + m + ")",
                                                    "`touch " + m + "`",
                                                    "a>" + m})
    {
        std::stringstream out;
        EXPECT_EQ(miopen::Process{"echo"}(args, "", &out), 0) << args;
        EXPECT_EQ(out.str(), args + "\n");
        ExpectNoMarker();
    }
}

TEST_F(CPU_Process_NONE, QuotesAreNotInterpreted)
{
    std::stringstream out;
    EXPECT_EQ(miopen::Process{"echo"}("\"a b\" 'c'", "", &out), 0);
    EXPECT_EQ(out.str(), "\"a b\" 'c'\n");
}

TEST_F(CPU_Process_NONE, EnvironmentValuesAreLiteral)
{
    const auto value = "v;touch " + marker.string() + ";$(touch " + marker.string() + ")";
    std::stringstream out;
    EXPECT_EQ(miopen::Process{"env"}(
                  "", "", &out, {{"MIOPEN_PROCESS_TEST_VAR", value}, {"HOME", "/overridden"}}),
              0);
    const auto env = "\n" + out.str();
    EXPECT_NE(env.find("\nMIOPEN_PROCESS_TEST_VAR=" + value + "\n"), std::string::npos);
    // An override replaces the inherited variable instead of duplicating it.
    EXPECT_NE(env.find("\nHOME=/overridden\n"), std::string::npos);
    EXPECT_EQ(env.find("\nHOME="), env.rfind("\nHOME="));
    // Everything else is inherited.
    EXPECT_NE(env.find("\nPATH="), std::string::npos);
    ExpectNoMarker();
}

TEST_F(CPU_Process_NONE, WorkingDirectoryIsLiteral)
{
    const auto cwd = dir / "a b;touch pwned&&$(touch pwned)";
    miopen::fs::create_directories(cwd);
    std::stringstream out;
    EXPECT_EQ(miopen::Process{"pwd"}("", cwd, &out), 0);
    EXPECT_EQ(out.str(), miopen::fs::canonical(cwd).string() + "\n");
    ExpectNoMarker();
    EXPECT_FALSE(miopen::fs::exists(cwd / "pwned"));
}

TEST_F(CPU_Process_NONE, CommandPathWithSpaces)
{
    // env, unlike echo, is never a shell builtin, so command -v yields its path.
    std::stringstream which;
    ASSERT_EQ(miopen::Process{"sh"}("-c command${IFS}-v${IFS}env", "", &which), 0);
    const auto env  = which.str().substr(0, which.str().find('\n'));
    const auto link = dir / "dir with space" / "my env";
    miopen::fs::create_directories(link.parent_path());
    miopen::fs::create_symlink(env, link);

    std::stringstream out;
    EXPECT_EQ(miopen::Process{link}("", "", &out, {{"MIOPEN_PROCESS_TEST_VAR", "hi"}}), 0);
    EXPECT_NE(out.str().find("MIOPEN_PROCESS_TEST_VAR=hi\n"), std::string::npos);
}

TEST_F(CPU_Process_NONE, ExitCodes)
{
    EXPECT_EQ(miopen::Process{"true"}(), 0);
    EXPECT_EQ(miopen::Process{"sh"}("-c exit${IFS}3"), 3);
    // Killed by a signal: 128 + signal number, as a shell reports it.
    EXPECT_EQ(miopen::Process{"sh"}("-c kill${IFS}-KILL${IFS}$$"), 128 + 9);
    // Command not found: 127, as a shell reports it.
    EXPECT_EQ(miopen::Process{dir / "does-not-exist"}(), 127);
    EXPECT_EQ(miopen::Process{dir / "does-not-exist"}("", dir.path), 127);
}

TEST_F(CPU_Process_NONE, CapturesStdoutAndStderr)
{
    std::stringstream out;
    EXPECT_EQ(miopen::Process{"sh"}("-c echo${IFS}out;echo${IFS}err>&2", "", &out), 0);
    EXPECT_EQ(out.str(), "out\nerr\n");
}

TEST_F(CPU_Process_NONE, FeedsStdin)
{
    const auto file = dir / "stdin.txt";
    std::string input;
    for(int i = 0; i < 1000; ++i)
        input += "line " + std::to_string(i) + "\n"; // Several pipe-buffer chunks.
    std::istringstream in{input};

    EXPECT_EQ(miopen::Process{"sh"}("-c cat>" + file.string(), "", nullptr, {}, &in), 0);

    std::ifstream written{file};
    const std::string contents{std::istreambuf_iterator<char>{written}, {}};
    EXPECT_EQ(contents, input);
}

TEST_F(CPU_Process_NONE, StdinIsEmptyWhenNotFed)
{
    std::stringstream out;
    EXPECT_EQ(miopen::Process{"cat"}("", "", &out), 0);
    EXPECT_EQ(out.str(), "");
}

TEST_F(CPU_Process_NONE, StdinWithStdoutThrows)
{
    std::istringstream in{"x"};
    std::stringstream out;
    EXPECT_ANY_THROW(miopen::Process{"cat"}("", "", &out, {}, &in));
}

TEST_F(CPU_Process_NONE, ConcurrentAsyncProcesses)
{
    constexpr int count = 8;
    std::vector<std::stringstream> outs(count);
    std::vector<miopen::ProcessAsync> children;
    children.reserve(count);
    for(int i = 0; i < count; ++i)
        children.emplace_back("echo", std::to_string(i), dir.path, &outs[i]);
    for(int i = 0; i < count; ++i)
    {
        EXPECT_EQ(children[i].Wait(), 0);
        EXPECT_EQ(outs[i].str(), std::to_string(i) + "\n");
    }
}

#endif // !_WIN32
