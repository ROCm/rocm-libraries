// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iostream>
#include <mutex>

#include <gtest/gtest.h>

#include "harness/Capture.hpp"
#include "harness/CurrentTestName.hpp"

namespace hipdnn_integration_tests
{

/// Process-wide count of the capture states reportCapture() published, printed in the
/// TEST COVERAGE SUMMARY. main() also reads it to fail a --graph-capture run that
/// verified tests but replayed none of them.
///
/// Like VerifierTally it covers one gtest iteration: CaptureTallyIterationReset clears
/// it as each iteration starts, so it matches the gtest counts printed beside it.
class CaptureTally
{
public:
    struct Counts
    {
        size_t replayed = 0;
        size_t mismatch = 0;
        size_t declined = 0;
        size_t failed = 0;
    };

    static CaptureTally& get()
    {
        static CaptureTally s_instance;
        return s_instance;
    }

    CaptureTally(const CaptureTally&) = delete;
    CaptureTally& operator=(const CaptureTally&) = delete;
    CaptureTally(CaptureTally&&) = delete;
    CaptureTally& operator=(CaptureTally&&) = delete;

    void record(const CaptureResult& result)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        switch(result.status)
        {
        case CaptureResult::Status::REPLAYED:
            if(result.outputsMismatched)
            {
                ++_counts.mismatch;
            }
            else
            {
                ++_counts.replayed;
            }
            break;
        case CaptureResult::Status::DECLINED:
            ++_counts.declined;
            break;
        case CaptureResult::Status::FAILED:
        default:
            ++_counts.failed;
            break;
        }
    }

    Counts counts() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _counts;
    }

    void reset()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _counts = {};
    }

private:
    CaptureTally() = default;

    mutable std::mutex _mutex;
    Counts _counts;
};

class CaptureTallyIterationReset : public ::testing::EmptyTestEventListener
{
public:
    void OnTestIterationStart(const ::testing::UnitTest& /*unitTest*/, int /*iteration*/) override
    {
        CaptureTally::get().reset();
    }
};

/// Publishes the capture result of the running test: a "[ CAPTURE ]" line on stdout,
/// a graph_capture property in --gtest_output files and a CaptureTally count. A failed
/// capture fails the test; a decline does not, the caller turns it into a skip.
inline void reportCapture(const CaptureResult& result)
{
    CaptureTally::get().record(result);

    const char* state = "replayed";
    switch(result.status)
    {
    case CaptureResult::Status::REPLAYED:
        if(result.outputsMismatched)
        {
            state = "mismatch";
        }
        break;
    case CaptureResult::Status::DECLINED:
        state = "declined";
        break;
    case CaptureResult::Status::FAILED:
        state = "failed";
        ADD_FAILURE() << "graph capture failed: " << result.message;
        break;
    default:
        state = "unknown";
        ADD_FAILURE() << "unknown graph capture status";
        break;
    }

    ::testing::Test::RecordProperty("graph_capture", state);

    auto testName = currentTestName();
    if(testName.empty())
    {
        testName = "<unknown test>";
    }
    std::cout << "[ CAPTURE ] " << testName << ": " << state;
    if(!result.message.empty())
    {
        std::cout << " (" << result.message << ")";
    }
    std::cout << "\n";
}

} // namespace hipdnn_integration_tests
