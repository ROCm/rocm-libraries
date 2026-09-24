// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-process.hpp"

#include <csignal>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs      = std::filesystem;
namespace process = hipblaslt_jit::process;

namespace
{
    void check(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    std::string hex(const std::string& value)
    {
        const char* digits = "0123456789abcdef";
        std::string result;
        for(unsigned char ch : value)
        {
            result += digits[ch >> 4];
            result += digits[ch & 15];
        }
        return result;
    }

    std::string read(const fs::path& path)
    {
        std::ifstream stream(path, std::ios::binary);
        std::string   result{std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>()};
        // The child writes textual records; the Windows CRT uses CRLF.
        for(std::size_t pos = 0; (pos = result.find("\r\n", pos)) != std::string::npos;)
            result.erase(pos, 1);
        return result;
    }

    void setEnvironment(const char* key, const char* value)
    {
#ifdef _WIN32
        _putenv_s(key, value ? value : "");
#else
        if(value)
            setenv(key, value, 1);
        else
            unsetenv(key);
#endif
    }

    struct SavedEnvironment
    {
        std::string name;
        bool        present;
        std::string value;
        explicit SavedEnvironment(const char* key)
            : name(key)
            , present(std::getenv(key) != nullptr)
        {
            if(present)
                value = std::getenv(key);
        }
        ~SavedEnvironment()
        {
            setEnvironment(name.c_str(), present ? value.c_str() : nullptr);
        }
    };

    int child(const std::vector<std::string>& args)
    {
        for(std::size_t i = 2; i < args.size(); ++i)
            std::cout << "arg=" << hex(args[i]) << '\n';
        std::cout << "cwd=" << hex(fs::current_path().u8string()) << '\n';
        for(const char* name :
            {"JIT_PROCESS_INHERITED", "JIT_PROCESS_OVERLAY", "JIT_PROCESS_EMPTY"})
        {
            const char* value = std::getenv(name);
            std::cout << name << '=' << (value ? hex(value) : "ABSENT") << '\n';
        }
        std::cout << "stdin_eof=" << (std::cin.get() == std::char_traits<char>::eof()) << '\n';
        std::cerr << "stderr_marker\n";
        return args[1] == "--child-fail" ? 17 : 0;
    }

    void quotingTests()
    {
        using process::detail::quoteWindowsArgument;
        check(quoteWindowsArgument(L"") == L"\"\"", "Windows empty argument");
        check(quoteWindowsArgument(L"plain") == L"\"plain\"", "Windows simple argument");
        check(quoteWindowsArgument(L"a b") == L"\"a b\"", "Windows spaced argument");
        check(quoteWindowsArgument(L"a\"b") == L"\"a\\\"b\"", "Windows embedded quote");
        check(quoteWindowsArgument(L"dir\\") == L"\"dir\\\\\"", "Windows trailing backslash");
        const std::wstring escapedExpected{
            L'"', L'o', L'n', L'e', L'\\', L'\\', L'\\', L'"', L't', L'w', L'o', L'"'};
        check(quoteWindowsArgument(L"one\\\"two") == escapedExpected,
              "Windows backslash immediately before quote");
    }

    int tests(const fs::path& executable, const fs::path& root)
    {
        check(!fs::exists(root), "Test root must be new");
        fs::create_directories(root);
        const auto cwd = root / fs::u8path(u8"space and \u03c0 cwd");
        fs::create_directory(cwd);
        const auto toolDirectory = root / "tool directory";
        fs::create_directory(toolDirectory);
#ifdef _WIN32
        const auto tool = toolDirectory / "provider executable.exe";
#else
        const auto tool = toolDirectory / "provider executable";
#endif
        fs::copy_file(executable, tool);
#ifndef _WIN32
        fs::permissions(tool, fs::perms::owner_exec, fs::perm_options::add);
#endif
        SavedEnvironment savedInherited("JIT_PROCESS_INHERITED");
        SavedEnvironment savedOverlay("JIT_PROCESS_OVERLAY");
        setEnvironment("JIT_PROCESS_INHERITED", "inherited value");
        setEnvironment("JIT_PROCESS_OVERLAY", "parent value");
        const auto parentCwd  = fs::current_path();
        auto       requestFor = [&](const std::string& name) {
            process::Request request;
            request.argv = {tool.u8string(), "--child"};
            request.environment
                = {{"JIT_PROCESS_OVERLAY", "child value"}, {"JIT_PROCESS_EMPTY", ""}};
            request.workingDirectory = cwd;
            request.logPath          = root / fs::u8path(name);
            return request;
        };

        quotingTests();
        std::cout << "PASS fixed Windows argv encoding cases\n";
        auto                           request   = requestFor(u8"space \u03c0 log.txt");
        const std::vector<std::string> arguments = {"",
                                                    "a b",
                                                    "a\"b",
                                                    "single'quote",
                                                    "dir\\",
                                                    "one\\\"two",
                                                    "$(touch SHOULD_NOT_EXIST)",
                                                    "a;echo injected",
                                                    "line1\nline2",
                                                    u8"unicode \u03c0"};
        request.argv.insert(request.argv.end(), arguments.begin(), arguments.end());
        auto result = process::run(request);
        check(result.succeeded(), "Successful launch: " + result.error);
        auto log = read(request.logPath);
        for(const auto& argument : arguments)
            check(log.find("arg=" + hex(argument) + "\n") != std::string::npos,
                  "Argument round trip");
        check(log.find("cwd=" + hex(cwd.u8string())) != std::string::npos, "Child cwd");
        check(log.find("JIT_PROCESS_INHERITED=" + hex("inherited value")) != std::string::npos,
              "Inherited environment");
        check(log.find("JIT_PROCESS_OVERLAY=" + hex("child value")) != std::string::npos,
              "Child environment overlay");
        check(log.find("JIT_PROCESS_EMPTY=\n") != std::string::npos, "Empty environment value");
        check(log.find("stderr_marker") != std::string::npos, "Stderr capture");
        check(log.find("stdin_eof=1") != std::string::npos, "Child stdin EOF");
        check(!fs::exists(cwd / "SHOULD_NOT_EXIST"), "Shell syntax must stay literal");
        check(fs::current_path() == parentCwd, "Parent cwd unchanged");
        check(std::string(std::getenv("JIT_PROCESS_OVERLAY")) == "parent value",
              "Parent env unchanged");
        std::cout << "PASS native argv/cwd/environment/stdout/stderr/stdin isolation\n";

        request         = requestFor("exit17.log");
        request.argv[1] = "--child-fail";
        result          = process::run(request);
        check(result.started && result.exited && result.exitCode == 17 && !result.error.empty(),
              "Nonzero child status");
        std::cout << "PASS nonzero exit diagnostics\n";

        request         = requestFor("missing-executable.log");
        request.argv[0] = (root / "does-not-exist").u8string();
        result          = process::run(request);
        check(!result.started && !result.exited && !result.error.empty(), "Missing executable");
        std::cout << "PASS launch failure diagnostics\n";

        request = requestFor("existing.log");
        {
            std::ofstream stream(request.logPath);
            stream << "preserved";
        }
        result = process::run(request);
        check(!result.started && read(request.logPath) == "preserved", "Existing log preserved");
        request = requestFor("bad-cwd.log");
        request.workingDirectory /= "missing";
        result = process::run(request);
        check(!result.started && !fs::exists(request.logPath),
              "Invalid cwd fails before log creation");
        request = requestFor("nul.log");
        request.argv.emplace_back("bad\0arg", 7);
        check(!process::run(request).started && !fs::exists(request.logPath),
              "NUL argument rejected");
        request = requestFor("invalid-env.log");
        request.environment.emplace_back("BAD=NAME", "value");
        check(!process::run(request).started, "Invalid environment name rejected");
        std::cout << "PASS exclusive log creation and invalid inputs\n";

#ifdef _WIN32
        request = requestFor("environment-case.log");
        request.environment.emplace_back("jit_process_overlay", "case-insensitive value");
        result = process::run(request);
        check(result.succeeded()
                  && read(request.logPath)
                             .find("JIT_PROCESS_OVERLAY=" + hex("case-insensitive value"))
                         != std::string::npos,
              "Windows case-insensitive environment overlay");
        request = requestFor("invalid-utf8.log");
        request.argv.emplace_back(1, static_cast<char>(0xff));
        check(!process::run(request).started, "Windows invalid UTF-8 rejected");
        std::cout << "PASS Windows environment case and UTF-8 validation\n";
#endif

        std::vector<std::future<bool>> tasks;
        for(int i = 0; i < 4; ++i)
            tasks.push_back(std::async(std::launch::async, [&, i] {
                auto concurrent = requestFor("parallel-" + std::to_string(i) + ".log");
                concurrent.environment.emplace_back("JIT_PROCESS_OVERLAY", std::to_string(i));
                auto childResult = process::run(concurrent);
                return childResult.succeeded()
                       && read(concurrent.logPath)
                                  .find("JIT_PROCESS_OVERLAY=" + hex(std::to_string(i)))
                              != std::string::npos;
            }));
        for(auto& task : tasks)
            check(task.get(), "Concurrent child environment isolation");
        check(fs::current_path() == parentCwd
                  && std::string(std::getenv("JIT_PROCESS_OVERLAY")) == "parent value",
              "Concurrent launch leaves parent state unchanged");
        std::cout << "PASS concurrent launch isolation\n";

#ifndef _WIN32
        request         = requestFor("signal.log");
        request.argv[1] = "--child-signal";
        result          = process::run(request);
        check(result.started && !result.exited && result.terminationSignal == SIGTERM
                  && !result.error.empty(),
              "Signal termination diagnostic");
        std::cout << "PASS POSIX signal status\n";
        if(fs::is_directory("/proc/self/fd"))
        {
            auto descriptorCount = [] {
                return std::distance(fs::directory_iterator("/proc/self/fd"),
                                     fs::directory_iterator());
            };
            auto before = descriptorCount();
            for(int i = 0; i < 16; ++i)
            {
                request         = requestFor("failed-" + std::to_string(i) + ".log");
                request.argv[0] = (root / "missing-executable").u8string();
                check(!process::run(request).started, "Repeated missing executable");
            }
            check(descriptorCount() == before, "Failure-path descriptors must close");
            std::cout << "PASS descriptor cleanup across repeated failed launches\n";
        }
#endif
        std::cout << "All portable provider process checks passed\n";
        return 0;
    }

    int entry(const std::vector<std::string>& args)
    {
        try
        {
            if(args.size() > 1 && (args[1] == "--child" || args[1] == "--child-fail"))
                return child(args);
#ifndef _WIN32
            if(args.size() > 1 && args[1] == "--child-signal")
            {
                std::raise(SIGTERM);
                return 1;
            }
#endif
            check(args.size() == 3 && args[1] == "--run-tests",
                  "Usage: process-test --run-tests NEW_ROOT");
            return tests(fs::absolute(fs::u8path(args[0])), fs::absolute(fs::u8path(args[2])));
        }
        catch(const std::exception& error)
        {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
{
    std::vector<std::string> args;
    for(int i = 0; i < argc; ++i)
    {
        int count = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string value(count, '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, value.data(), count, nullptr, nullptr);
        value.pop_back();
        args.push_back(std::move(value));
    }
    return entry(args);
}
#else
int main(int argc, char** argv)
{
    return entry(std::vector<std::string>(argv, argv + argc));
}
#endif
