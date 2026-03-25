// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Profiler implementation: Profiler::ScopedStopwatch RAII timing and report
// formatting. See Profiler.h for class documentation and usage examples.

#include "race-emulator/Profiler.h"
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <vector>

namespace raceemulator {

// --- Profiler::ScopedStopwatch ---

Profiler::ScopedStopwatch::ScopedStopwatch(Profiler *profiler, Entry *entry)
    : profiler(profiler), entry(entry), scopeId(profiler->nextScopeId++) {
  auto now = Clock::now();
  if (!profiler->scopeStack.empty()) {
    // Pause the currently running scope: flush its accumulated time.
    auto &top = profiler->scopeStack.back();
    top.entry->totalSeconds +=
        std::chrono::duration<double>(now - profiler->activeStart).count();
  }
  if (entry->depth < 0) {
    entry->depth = static_cast<int>(profiler->scopeStack.size());
  }
  profiler->scopeStack.push_back({entry, scopeId});
  profiler->activeStart = now;
  entry->count++;
}

Profiler::ScopedStopwatch::~ScopedStopwatch() { retire(); }

void Profiler::ScopedStopwatch::retire() {
  if (!entry) {
    return;
  }

  auto &stack = profiler->scopeStack;

  if (!stack.empty() && stack.back().id == scopeId) {
    // We are the active (most recent) scope: flush our time and resume the
    // previous scope.
    auto now = Clock::now();
    entry->totalSeconds +=
        std::chrono::duration<double>(now - profiler->activeStart).count();
    stack.pop_back();
    if (!stack.empty()) {
      profiler->activeStart = now;
    }
  } else {
    // We are a paused (middle) scope -- our time was already flushed when we
    // were paused. Just remove ourselves from the list.
    for (auto it = stack.begin(); it != stack.end(); ++it) {
      if (it->id == scopeId) {
        stack.erase(it);
        break;
      }
    }
  }

  entry = nullptr;
}

Profiler::ScopedStopwatch::ScopedStopwatch(ScopedStopwatch &&other) noexcept
    : profiler(other.profiler), entry(other.entry), scopeId(other.scopeId) {
  other.profiler = nullptr;
  other.entry = nullptr;
}

Profiler::ScopedStopwatch &
Profiler::ScopedStopwatch::operator=(ScopedStopwatch &&other) noexcept {
  if (this != &other) {
    retire();
    profiler = other.profiler;
    entry = other.entry;
    scopeId = other.scopeId;
    other.profiler = nullptr;
    other.entry = nullptr;
  }
  return *this;
}

// --- Profiler ---

Profiler::ScopedStopwatch Profiler::scopedStopwatch(std::string_view key) {
  if (!enabled) {
    return {};
  }
  return ScopedStopwatch(this, &entries[std::string(key)]);
}

double Profiler::getAccountedTime() const {
  double total = 0.0;
  for (const auto &[name, entry] : entries) {
    total += entry.totalSeconds;
  }
  return total;
}

double Profiler::getWallTime() const {
  auto elapsed = Clock::now() - wallStart;
  return std::chrono::duration<double>(elapsed).count();
}

void Profiler::reset() {
  assert(scopeStack.empty() &&
         "Profiler::reset() called while scopes are still active. "
         "Destroy all ScopedStopwatch instances before resetting.");
  entries.clear();
  wallStart = Clock::now();
}

void Profiler::report(std::ostream &os, double minPercentage) const {
  if (entries.empty()) {
    os << "(no profiling data)\n";
    return;
  }

  // Collect and sort by total time descending.
  struct Row {
    std::string name;
    double seconds;
    uint64_t count;
    int depth;
  };
  std::vector<Row> rows;
  rows.reserve(entries.size());
  for (const auto &[name, entry] : entries) {
    rows.push_back(
        {name, entry.totalSeconds, entry.count, std::max(0, entry.depth)});
  }
  std::sort(rows.begin(), rows.end(),
            [](const Row &a, const Row &b) { return a.seconds > b.seconds; });

  double wall = getWallTime();
  double accounted = getAccountedTime();
  double threshold = wall * minPercentage / 100.0;

  // Count entries below threshold to build the "other" label.
  int otherEntries = 0;
  for (const auto &row : rows) {
    if (row.seconds < threshold) {
      otherEntries++;
    }
  }
  // Find the longest scope name for column alignment.
  std::string otherLabel =
      otherEntries > 0 ? "other (" + std::to_string(otherEntries) + " entries)"
                       : "";
  size_t maxNameLen = std::max({size_t(5),  // "Scope"
                                size_t(11), // "Unaccounted"
                                otherLabel.size()});
  for (const auto &row : rows) {
    if (row.seconds >= threshold) {
      maxNameLen = std::max(maxNameLen, row.name.size());
    }
  }

  // Header and separator.
  int lineWidth = static_cast<int>(maxNameLen) + 2 + 12 + 12 + 12 + 8 + 6;
  os << std::left << std::setw(static_cast<int>(maxNameLen + 2)) << "Scope"
     << std::right << std::setw(12) << "Time [s]" << std::setw(12) << "Count"
     << std::setw(12) << "Avg [us]" << std::setw(8) << "%" << std::setw(6)
     << "Depth" << "\n";
  os << std::string(lineWidth, '-') << "\n";

  // Rows above threshold.
  double otherSeconds = 0.0;
  uint64_t otherCount = 0;

  for (const auto &row : rows) {
    if (row.seconds < threshold) {
      otherSeconds += row.seconds;
      otherCount += row.count;
      continue;
    }
    double avgUs = row.count > 0
                       ? (row.seconds / static_cast<double>(row.count)) * 1e6
                       : 0.0;
    double pct = wall > 0.0 ? (row.seconds / wall) * 100.0 : 0.0;

    os << std::left << std::setw(static_cast<int>(maxNameLen + 2)) << row.name
       << std::right << std::fixed << std::setprecision(6) << std::setw(12)
       << row.seconds << std::setw(12) << row.count << std::setw(12)
       << std::setprecision(2) << avgUs << std::setw(7) << std::setprecision(1)
       << pct << "%" << std::setw(5) << row.depth << "\n";
  }

  // "other" row if any entries were below threshold.
  if (otherEntries > 0) {
    double avgUs = otherCount > 0
                       ? (otherSeconds / static_cast<double>(otherCount)) * 1e6
                       : 0.0;
    double pct = wall > 0.0 ? (otherSeconds / wall) * 100.0 : 0.0;

    os << std::left << std::setw(static_cast<int>(maxNameLen + 2)) << otherLabel
       << std::right << std::fixed << std::setprecision(6) << std::setw(12)
       << otherSeconds << std::setw(12) << otherCount << std::setw(12)
       << std::setprecision(2) << avgUs << std::setw(7) << std::setprecision(1)
       << pct << "%" << "\n";
  }

  // Separator, accounted, unaccounted, and wall-clock total.
  os << std::string(lineWidth, '-') << "\n";

  double unaccounted = wall - accounted;
  double unaccountedPct = wall > 0.0 ? (unaccounted / wall) * 100.0 : 0.0;

  os << std::left << std::setw(static_cast<int>(maxNameLen + 2)) << "Accounted"
     << std::right << std::fixed << std::setprecision(6) << std::setw(12)
     << accounted << std::setw(12) << "" << std::setw(12) << "" << std::setw(7)
     << std::setprecision(1) << (wall > 0.0 ? (accounted / wall) * 100.0 : 0.0)
     << "%" << "\n";

  os << std::left << std::setw(static_cast<int>(maxNameLen + 2))
     << "Unaccounted" << std::right << std::fixed << std::setprecision(6)
     << std::setw(12) << unaccounted << std::setw(12) << "" << std::setw(12)
     << "" << std::setw(7) << std::setprecision(1) << unaccountedPct << "%"
     << "\n";

  os << std::left << std::setw(static_cast<int>(maxNameLen + 2)) << "Total"
     << std::right << std::fixed << std::setprecision(6) << std::setw(12)
     << wall << "\n";
}

std::string Profiler::reportStr(double minPercentage) const {
  std::ostringstream oss;
  report(oss, minPercentage);
  return oss.str();
}

} // namespace raceemulator
