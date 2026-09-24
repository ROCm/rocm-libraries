// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace hipblaslt_jit::process
{
    struct Request
    {
        // UTF-8 strings; argv[0] is the executable, never a shell command.
        std::vector<std::string> argv;
        // Overlay the inherited environment. Last value wins; empty is a value.
        std::vector<std::pair<std::string, std::string>> environment;
        // Existing directory and new log file. Neither may be empty.
        std::filesystem::path workingDirectory;
        std::filesystem::path logPath;
    };

    struct Result
    {
        bool          started           = false;
        bool          exited            = false;
        std::uint32_t exitCode          = 0;
        int           terminationSignal = 0;
        std::string   error;

        bool succeeded() const
        {
            return started && exited && exitCode == 0 && error.empty();
        }
    };

    // Synchronous, shell-free launch. Captures stdout/stderr in an exclusively
    // created log and supplies EOF on stdin. Never changes the caller's cwd,
    // environment or standard descriptors. Logs are retained on failure.
    // A path containing a directory is resolved relative to the caller's cwd;
    // bare executable names use the caller's PATH search. Callers must not
    // mutate process-wide environment/cwd concurrently with this call.
    Result run(const Request& request);

    namespace detail
    {
        // Microsoft CRT argv encoding, including empty strings and trailing '\'.
        // Exposed privately so its fixed regression examples run on either OS.
        std::wstring quoteWindowsArgument(const std::wstring& argument);
    }
}
