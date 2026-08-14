// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/support/TimePassesInstrumentation.hpp"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <utility>

namespace stinkytofu {
namespace {
/// Per-thread so concurrent pipeline runs keep separate stacks and reports.
std::shared_ptr<TimePassesInstrumentation>& activeTimePasses() {
    static thread_local std::shared_ptr<TimePassesInstrumentation> active;
    return active;
}
}  // namespace

void TimePassesInstrumentation::enter(const std::string& passName) {
    running.push_back({passName, Clock::now(), 0.0});
}

void TimePassesInstrumentation::leave() {
    if (running.empty()) return;  // unbalanced callbacks: nothing to charge

    const Running done = std::move(running.back());
    running.pop_back();

    const double elapsed = std::chrono::duration<double>(Clock::now() - done.start).count();
    PassTime& time = passTimes[done.name];
    time.total += elapsed;
    time.self += elapsed - done.nested;
    time.runs++;

    if (!running.empty()) running.back().nested += elapsed;
}

void TimePassesInstrumentation::report(std::ostream& os, const std::string& label) const {
    std::vector<std::pair<std::string, PassTime>> rows(passTimes.begin(), passTimes.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second.self > b.second.self; });

    double wall = 0.0;
    for (const auto& [name, time] : rows) wall += time.self;

    // Composed in full and written once: kernels are generated in parallel
    // processes that share stderr, so a report emitted line by line would
    // interleave with another kernel's.
    std::ostringstream block;
    block << "===== StinkyTofu pass timing";
    if (!label.empty()) block << ": " << label;
    block << " =====\n";
    block << "   self(s)   total(s)   self%   runs  pass\n";

    auto printRow = [&block](double self, double total, double percent, unsigned runs,
                             const std::string& name) {
        block << std::fixed << std::setprecision(4) << std::setw(10) << self << std::setw(11)
              << total << std::setprecision(1) << std::setw(8) << percent << std::setw(7) << runs
              << "  " << name << '\n';
    };

    for (const auto& [name, time] : rows)
        printRow(time.self, time.total, wall > 0.0 ? 100.0 * time.self / wall : 0.0, time.runs,
                 name);

    block << std::fixed << std::setprecision(4) << std::setw(10) << wall << "  total\n";

    os << block.str();
    os.flush();
}

std::shared_ptr<TimePassesInstrumentation> getActiveTimePasses() {
    return activeTimePasses();
}

TimePassesSession::TimePassesSession(bool enable, std::string label, std::ostream& os)
    : label(std::move(label)), os(os) {
    if (!enable || activeTimePasses()) return;
    timer = std::make_shared<TimePassesInstrumentation>();
    activeTimePasses() = timer;
}

TimePassesSession::~TimePassesSession() {
    if (!timer) return;
    activeTimePasses().reset();
    timer->report(os, label);
}

}  // namespace stinkytofu
