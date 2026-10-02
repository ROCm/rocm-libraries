// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace hipblaslt_jit_test
{
    // Runs argv through the shell in directory, with environment set on top of
    // the inherited one, no input, and stdout and stderr written to log. True
    // when it exits with 0. Arguments and values must not contain '"' on Windows,
    // where an empty value unsets the variable.
    inline bool runChild(const std::vector<std::string>&                         argv,
                         const std::vector<std::pair<std::string, std::string>>& environment,
                         const std::filesystem::path&                            directory,
                         const std::filesystem::path&                            log)
    {
#ifdef _WIN32
        const auto quote = [](const std::string& text) { return "\"" + text + "\""; };
        std::string command = "cd /d " + quote(directory.string());
        for(const auto& [name, value] : environment)
            command += " && set " + quote(name + "=" + value);
        command += " &&";
        const std::string empty = "NUL";
#else
        const auto quote = [](const std::string& text) {
            std::string quoted = "'";
            for(char c : text)
                quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
            return quoted + "'";
        };
        std::string command = "cd " + quote(directory.string()) + " &&";
        for(const auto& [name, value] : environment)
            command += " " + name + "=" + quote(value);
        const std::string empty = "/dev/null";
#endif
        for(const auto& argument : argv)
            command += " " + quote(argument);
        command += " <" + empty + " >" + quote(log.string()) + " 2>&1";
        return std::system(command.c_str()) == 0;
    }
}
