// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include <gtest/gtest.h>

#include "harness/Capture.hpp"
#include "harness/CurrentTestName.hpp"

namespace hipdnn_integration_tests
{

/// Publishes the capture result of the running test: a "[ CAPTURE ]" line on stdout
/// and a graph_capture property in --gtest_output files. A failed capture fails the
/// test; a decline does not, the caller turns it into a skip.
inline void reportCapture(const CaptureResult& result)
{
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
