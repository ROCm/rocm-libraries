// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-heuristic.hpp"
#include "utility.hpp"
#include <iostream>
#include <mutex>
#include <set>

namespace hipblaslt_jit
{
    namespace
    {
        struct Reported
        {
            std::mutex            mutex;
            std::set<std::string> lines;
        };
    }

    void report(Severity severity, const std::string& message)
    {
        const auto line = std::string(severity == Severity::Error ? "hipblaslt error: JIT "
                                                                  : "hipblaslt warning: JIT ")
                          + message;
        if(severity == Severity::Error)
            log_error(__func__, message);
        else
            log_info(__func__, message);
        // Reports can come from threads still running at exit.
        static auto* reported = new Reported;
        {
            std::lock_guard<std::mutex> lock(reported->mutex);
            if(!reported->lines.insert(line).second)
                return;
        }
        std::cerr << line << std::endl;
    }

    std::string describe(const Status& status, const std::string& problem)
    {
        auto text = std::string(toString(status.stage)) + " failed for " + problem + ": "
                    + status.message;
        if(!status.logPath.empty() && status.message.find(status.logPath) == std::string::npos)
            text += " (log: " + status.logPath + ")";
        return text;
    }
}
