// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_integration_tests::golden
{

// Why a bundle could not be verified. The two severities are printed in separate
// sections so a genuine reference bug is never lost among expected coverage gaps.
//
//   Unverifiable — expected coverage gap (no golden data, no reference can run
//                  the op, inputs could not be synthesized, ...). The engine was
//                  not accused; we simply had no oracle. Quiet but listed.
//   RefError     — a reference executor that CAN run the op threw at runtime
//                  (case C) and the harness fell through to keep verifying the
//                  engine. This is a reference bug and must be loud.
enum class UnverifiableSeverity
{
    UNVERIFIABLE,
    REF_ERROR,
};

// Process-wide collector of bundles that ended a run without a verdict. Mirrors
// SupportMatrixCollector: a thread-safe singleton populated during test execution
// and printed once after RUN_ALL_TESTS(). Records do not affect the GTest exit
// code — an unverifiable bundle SKIPs; this report is the visible trail.
class UnverifiableBundleReport
{
public:
    struct Record
    {
        std::string bundle; // bundle path / identifier
        std::string reason; // human-readable explanation
        UnverifiableSeverity severity;
    };

    static UnverifiableBundleReport& get()
    {
        static UnverifiableBundleReport s_instance;
        return s_instance;
    }

    UnverifiableBundleReport(const UnverifiableBundleReport&) = delete;
    UnverifiableBundleReport& operator=(const UnverifiableBundleReport&) = delete;
    UnverifiableBundleReport(UnverifiableBundleReport&&) = delete;
    UnverifiableBundleReport& operator=(UnverifiableBundleReport&&) = delete;

    void record(std::string bundle, std::string reason, UnverifiableSeverity severity)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.push_back({std::move(bundle), std::move(reason), severity});
    }

    std::vector<Record> getRecords() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _records;
    }

    void reset()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _records.clear();
    }

    // Print both severity sections to `os`. No-op when nothing was recorded.
    void print(std::ostream& os = std::cout) const
    {
        std::vector<Record> records;
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            records = _records;
        }
        if(records.empty())
        {
            return;
        }

        printSection(os, records, UnverifiableSeverity::REF_ERROR, "REFERENCE EXECUTOR ERRORS");
        printSection(os, records, UnverifiableSeverity::UNVERIFIABLE, "UNVERIFIABLE BUNDLES");
    }

private:
    UnverifiableBundleReport() = default;

    static void printSection(std::ostream& os,
                             const std::vector<Record>& records,
                             UnverifiableSeverity severity,
                             const char* heading)
    {
        size_t count = 0;
        for(const auto& r : records)
        {
            if(r.severity == severity)
            {
                ++count;
            }
        }
        if(count == 0)
        {
            return;
        }

        os << "\n==== " << heading << " (" << count << ") ====\n";
        for(const auto& r : records)
        {
            if(r.severity == severity)
            {
                os << "  - " << r.bundle << ": " << r.reason << "\n";
            }
        }
    }

    mutable std::mutex _mutex;
    std::vector<Record> _records;
};

} // namespace hipdnn_integration_tests::golden
