// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-debug.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

// HIPBLASLT_JIT_DEBUG for a generator child process: the categories it is
// asked for, the progress events it writes and its stage times.
namespace hipblaslt_jit::debug
{
    // The --debug value for a generator child: "timing", "progress" or
    // "timing,progress". Empty when no category is on.
    std::string childCategories();

    // Relays the progress events a generator child appends to events while it
    // runs, as child.* lines of the current context, and writes child.heartbeat
    // after 10 s without one. Accepts lines with "v":1 and increasing "seq", at
    // most 4 KiB each and 10,000 per child; drops the rest.
    class ChildObserver
    {
    public:
        explicit ChildObserver(std::filesystem::path events);
        ~ChildObserver();
        ChildObserver(const ChildObserver&)            = delete;
        ChildObserver& operator=(const ChildObserver&) = delete;
        // Reads what is left, discards a trailing partial line and joins. Idempotent.
        void   stop() noexcept;
        size_t events() const noexcept;
        size_t dropped() const noexcept;

    private:
        struct State;
        std::shared_ptr<State> m_state;
    };

    // The "child" object of a generation line from the generator's timing.json:
    // its total, CPU and span totals plus "unattributed", childNanoseconds minus
    // its total. Empty, with why set to "missing" or "invalid", otherwise.
    std::string childTiming(const std::filesystem::path& file,
                            uint64_t                     childNanoseconds,
                            std::string&                 why);
}
